"""Bounded CPU checks for the information/uncertainty modules.

Run with ``python -B -m research_v5.tests_information`` from the E-drive repo.
Only 64x64 software fixtures are used; no formal arrays or protocol are written.
"""
import os
from pathlib import Path

REPORT_DIRECTORY = Path(__file__).resolve().parents[1] / "reports/research_v5/implementation_checks/information"
# Before legacy runtime or torch: the task owns every cache/temp path below.
os.environ["THERMOMETRY_RUNTIME"] = str(REPORT_DIRECTORY / "runtime")
for _name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_name] = str(REPORT_DIRECTORY / "runtime/tmp")
os.environ["CUDA_CACHE_PATH"] = str(REPORT_DIRECTORY / "runtime/cuda")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import runtime
import copy
import json
import math
import time

import torch
from config import default_config
from physics import SpectralPhysics
from .identifiability import fisher_identifiability, optical_nullspace_tests
from .uncertainty import (fixed_mc_seed_plan, draw_raw_dn_sample,
                          monte_carlo_uncertainty, hessian_probe_diagnostics)

torch.set_num_threads(2)
_DETAILS = {}


def _fixture(dtype=torch.float64):
    p = SpectralPhysics(default_config(64), dtype=dtype).cpu()
    state = torch.zeros((1, 5, 64, 64), dtype=dtype)
    state[:, 0] = (3302.3-p.T0)/p.Tscale
    state[:, 1] = math.log(0.6)
    state[:, 4] = 90/p.Sscale
    beta = torch.tensor([[0.45, 0.06, -0.04]], dtype=dtype)
    return p, state, beta, p(state, beta)


def test_real_forward_optical_zero_transfer_decode_and_adjoint():
    results = {}
    for dtype in (torch.float32, torch.float64):
        p, _, _, _ = _fixture(dtype)
        report = optical_nullspace_tests(p)
        assert report["all_passed"], report
        assert report["high_frequency"]["object_OTF"] == 0
        assert report["high_frequency"]["encode_without_object_blur_output_to_input_norm"] > 0.01
        for probe in report["low_frequency_transfer_and_carrier_isolation"]:
            for row in probe["channels"]:
                tolerance = report["tolerance"]
                assert abs(row["measured_object_transmission"]-row["expected_object_OTF"]) < tolerance
        json.dumps(report, allow_nan=False)
        results[str(dtype)] = report
    _DETAILS["optical"] = results


def test_finite_modal_fisher_physical_units_nuisance_and_float64_fd():
    p, state, beta, raw = _fixture()
    report = fisher_identifiability(p, state, beta, raw)
    frame = report["frames"][0]
    assert frame["adjoint_all_passed"]
    assert frame["float64_finite_difference"]["all_checked_passed"]
    assert frame["float64_finite_difference"]["shifted_temperature_pixels"] == 0
    manual_derivative = p.op(p.M[:, 2][None, :, None, None].expand(1, 7, 64, 64))
    expected_information = ((manual_derivative.square())/(raw.clamp_min(0)+p.extra_variance)).sum()
    actual_information = frame["raw_S_information"]["gram"][0][0]
    assert math.isclose(actual_information, float(expected_information), rel_tol=1e-11)
    for projection in frame["nuisance_projected_S"].values():
        assert all(value <= 1+1e-10 for value in projection["retained_information_fraction_per_mode"])
        gram = torch.tensor(projection["gram"], dtype=torch.float64)
        assert torch.linalg.eigvalsh(gram).min() >= -1e-10
        s45 = projection["single_mode_detection_SNR"]["45.0"]
        s180 = projection["single_mode_detection_SNR"]["180.0"]
        assert all(math.isclose(a*4, b, abs_tol=1e-12) for a, b in zip(s45, s180))
    json.dumps(report, allow_nan=False)
    _DETAILS["fisher"] = report


def test_rank_deficient_beta_template_and_lut_knot_are_explicit():
    p, state, beta, _ = _fixture()
    p.M[:, 2] = p.q
    state[:, 0] = 0.3  # exact 3300 K LUT knot
    raw = p(state, beta)
    report = fisher_identifiability(p, state, beta, raw, {"fd_max_directions": 4})
    frame = report["frames"][0]
    projected = frame["nuisance_projected_S"]["beta_only"]
    assert projected["rank"] < projected["columns"]
    assert projected["condition_number_is_infinite"]
    assert projected["finite_modal_linearized_sigma_e"] is None
    assert projected["retained_information_fraction_per_mode"][0] < 1e-20
    fd = frame["float64_finite_difference"]
    assert fd["shifted_temperature_pixels"] == 64*64
    assert fd["Fisher_uses_unmodified_supplied_state"]
    assert fd["all_checked_passed"]
    json.dumps(report, allow_nan=False)
    _DETAILS["rank_and_knot"] = {
        "rank": projected["rank"], "columns": projected["columns"],
        "DC_retained_information": projected["retained_information_fraction_per_mode"][0],
        "float64_fd": fd,
    }


def _raw_only_test_callback(physics, continuum_calibration):
    """Small real-operator regression callback, never used for deployment.

    Continuum is explicit *external test calibration*. Color temperature is
    still recalculated on decoded raw, so MC checks propagated measurement
    variation. The formal callback must rerun its entire continuum solver.
    """
    calls = []
    one = torch.ones((1, *physics.cfg["shape"]), dtype=physics.M.dtype)
    template_bands = physics.M[:, 2][None, :, None, None] * one[:, None]
    template_raw = physics.op(template_bands)
    continuum_raw = physics.op(continuum_calibration)

    def callback(raw_dn, calibration, selection):
        assert selection == {"alpha_g": 0.0}
        assert calibration == {"fixture": "external-continuum-calibration"}
        # Exactly one raw-DN-to-electron conversion inside reconstruction.
        raw = physics.raw_to_electrons(raw_dn)
        weight = 1/(raw.clamp_min(0)+physics.extra_variance)
        coefficient = (weight*template_raw*(raw-continuum_raw)).sum()/(weight*template_raw.square()).sum()
        signed = coefficient * one
        decoded = physics.op.decode(raw)
        temperature, valid = physics.color_temperature(decoded)
        fitted = continuum_raw + coefficient * template_raw
        calls.append({"raw_sum_DN": float(raw_dn.sum()), "signed_e": float(coefficient)})
        return {"template_postblur_e": physics.op.object_blur(signed),
                "temperature_color_K": temperature, "valid": valid, "expected_e": fitted}
    return callback, calls


def test_mc_fixed_raw_dn_plan_reconstruction_and_signed_intervals():
    p, state, _, _ = _fixture()
    continuum, _, _, _ = p.components(state, torch.zeros((1, 3), dtype=state.dtype))
    expected = p.op(continuum + p.M[:, 2][None, :, None, None] * 90)
    original_dn = (expected/p.adc_step+p.cfg["black_level_DN"]).round()
    callback, calls = _raw_only_test_callback(p, continuum)
    options = {"repeats": 8, "keep_samples": True}
    result = monte_carlo_uncertainty(p, original_dn, callback,
                                     {"fixture": "external-continuum-calibration"},
                                     {"alpha_g": 0.0}, options)
    assert result["status"] == "completed", result
    assert len(calls) == 9
    assert result["summary"]["completed_repeats"] == 8
    assert result["summary"]["valid_coverage_fraction"] > 0.95
    assert torch.isfinite(result["sigma_S"][result["valid"]]).all()
    assert (result["sigma_S"][result["valid"]] > 0).all()
    assert (result["signed_ci_lower_e"][result["interval_valid"]] <=
            result["signed_ci_upper_e"][result["interval_valid"]]).all()
    assert float(result["temperature_mc_sigma_K"][result["valid"]].mean()) > 0
    replay = monte_carlo_uncertainty(p, original_dn, callback,
                                     {"fixture": "external-continuum-calibration"},
                                     {"alpha_g": 0.0}, options)
    assert torch.equal(result["template_samples_e"], replay["template_samples_e"])
    assert torch.equal(result["temperature_samples_K"], replay["temperature_samples_K"])
    assert len(set(row["raw_sum_DN"] for row in calls[1:9])) > 1
    json.dumps(result["summary"], allow_nan=False)
    _DETAILS["MC"] = dict(result["summary"], mean_sigma_S_e=float(result["sigma_S"][result["valid"]].mean()),
                          mean_temperature_sigma_K=float(result["temperature_mc_sigma_K"][result["valid"]].mean()))


def test_mc_invalid_poisson_mean_clipping_and_budget_abstain():
    p, _, _, expected = _fixture()
    raw_dn = (expected/p.adc_step+p.cfg["black_level_DN"]).round()
    baseline = {"template_postblur_e": torch.zeros_like(expected),
                "temperature_color_K": torch.full_like(expected, 3300),
                "valid": torch.ones_like(expected, dtype=torch.bool),
                "expected_e": -torch.ones_like(expected)}
    calls = []

    def unreachable(*arguments):
        calls.append(True)
        raise AssertionError("invalid original mean must abstain before reconstruction")

    result = monte_carlo_uncertainty(p, raw_dn, unreachable, baseline=baseline)
    assert result["status"] == "abstained" and "negative" in result["reason"]
    clipped = raw_dn.clone()
    clipped[0, 0, 0] = 4095
    result_clipped = monte_carlo_uncertainty(p, clipped, unreachable, baseline=baseline)
    assert result_clipped["status"] == "abstained" and "ADC-clipped" in result_clipped["reason"]
    assert not calls
    try:
        fixed_mc_seed_plan(repeats=128)
    except ValueError:
        pass
    else:
        raise AssertionError("128-repeat automatic plan was accepted")
    sample1, metadata1 = draw_raw_dn_sample(p, expected, fixed_mc_seed_plan()[0])
    sample2, metadata2 = draw_raw_dn_sample(p, expected, fixed_mc_seed_plan()[0])
    assert torch.equal(sample1, sample2) and metadata1 == metadata2
    assert torch.equal(sample1, sample1.round())
    _DETAILS["MC_invalid"] = {"negative_mean": result["summary"], "clipped_raw": result_clipped["summary"],
                              "128_repeat_plan_rejected": True}


def test_hessian_sandwich_prior_proxy_true_residual_and_failure():
    h = torch.tensor([2., 3., 5., 7., 11., 13.], dtype=torch.float64)
    f = torch.tensor([1., 2., 3., 5., 7., 11.], dtype=torch.float64)
    template = torch.zeros_like(h)
    options = {"probes": 8, "cg_tolerance": 1e-11, "cg_max_iterations": 20}
    sandwich = hessian_probe_diagnostics(lambda x: h*x, template, lambda x: f*x, options,
                                         residual_scale=h)
    inverse = hessian_probe_diagnostics(lambda x: h*x, template, options=options)
    assert sandwich["summary"]["all_converged"] and inverse["summary"]["all_converged"]
    assert torch.allclose(sandwich["diagonal_estimate"], f/h.square(), rtol=1e-10, atol=1e-12)
    assert torch.allclose(inverse["diagonal_estimate"], 1/h, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(sandwich["diagonal_estimate"], inverse["diagonal_estimate"])
    for row in sandwich["summary"]["solves"]:
        assert row["inverse_solve"]["true_relative_residual"] <= 1e-11
        assert row["sandwich_solve"]["true_relative_residual"] <= 1e-11
        assert row["inverse_solve"]["equilibrated_true_relative_residual"] <= 1e-11
        assert row["sandwich_solve"]["equilibrated_true_relative_residual"] <= 1e-11
    failure = hessian_probe_diagnostics(lambda x: -x, template, options={"probes": 2})
    assert not failure["summary"]["all_converged"]
    assert not failure["valid"].any() and torch.isnan(failure["sigma_proxy"]).all()
    nonfinite = hessian_probe_diagnostics(lambda x: x*float("nan"), template, options={"probes": 1})
    assert not nonfinite["summary"]["all_converged"]
    json.dumps(sandwich["summary"], allow_nan=False)
    json.dumps(failure["summary"], allow_nan=False)
    json.dumps(nonfinite["summary"], allow_nan=False)
    _DETAILS["hessian"] = {"sandwich": sandwich["summary"], "inverse_proxy": inverse["summary"],
                           "negative_curvature": failure["summary"]}


def main():
    tests = [test_real_forward_optical_zero_transfer_decode_and_adjoint,
             test_finite_modal_fisher_physical_units_nuisance_and_float64_fd,
             test_rank_deficient_beta_template_and_lut_knot_are_explicit,
             test_mc_fixed_raw_dn_plan_reconstruction_and_signed_intervals,
             test_mc_invalid_poisson_mean_clipping_and_budget_abstain,
             test_hessian_sandwich_prior_proxy_true_residual_and_failure]
    rows = []
    started = time.perf_counter()
    for test in tests:
        before = time.perf_counter()
        try:
            test()
        except Exception as error:
            rows.append({"test": test.__name__, "passed": False, "error_type": type(error).__name__,
                         "error": str(error), "duration_s": time.perf_counter()-before})
            print(test.__name__ + ": FAILED " + str(error), flush=True)
        else:
            rows.append({"test": test.__name__, "passed": True, "duration_s": time.perf_counter()-before})
            print(test.__name__ + ": passed", flush=True)
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), Path(__file__).with_name("identifiability.py"),
             Path(__file__).with_name("uncertainty.py"), root/"physics.py", root/"config.py"]
    receipt = {"schema": "v5-information-implementation-checks-001", "timestamp": runtime.now(),
               "execution": "bounded CPU-only 64x64 fixtures, no training/formal arrays",
               "torch_version": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
               "all_passed": all(row["passed"] for row in rows), "tests": rows,
               "duration_s": time.perf_counter()-started, "details": _DETAILS,
               "source_sha256": {str(path.relative_to(root)): runtime.sha256(path) for path in files},
               "runtime_root": os.environ["THERMOMETRY_RUNTIME"],
               "no_bytecode": True,
               "scope": "numerical implementation evidence; not scientific acceptance or empirical coverage"}
    runtime.write_json(REPORT_DIRECTORY/"cpu_checks.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("all_passed", "duration_s", "source_sha256")}), flush=True)
    if not receipt["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
