"""Bounded CPU fixtures, never development/confirmation/final inputs."""
from __future__ import annotations
from . import common
common.configure_runtime()

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
import torch

from . import decomposition as d
from physics import SpectralPhysics


OPTIONS = {"alpha_g": 1.0, "rtol": 1e-4, "max_iterations": 200,
           "diagonal_mode": "exact_cyclic"}


def fixture(dtype=torch.float32, size=16):
    cfg = copy.deepcopy(common.physics_config())
    base = round(size*.2)
    cfg["shape"] = [size, size]
    cfg["carrier_bins_xy"] = [[base, 0], [0, base], [base, base], [-base, base],
                              [2*base, 0], [0, 2*base], [2*base, base]]
    # Calibration is created in the original precision; double fixtures cast buffers.
    p = SpectralPhysics(cfg)
    if dtype == torch.float64:
        p._apply(lambda b: b.to(dtype=torch.complex128 if b.is_complex() else torch.float64)
                 if b.is_floating_point() or b.is_complex() else b)
    yy, xx = torch.meshgrid(torch.arange(size, dtype=dtype), torch.arange(size, dtype=dtype), indexing="ij")
    wave = torch.cos(2*torch.pi*xx/size)+.4*torch.sin(2*torch.pi*yy/size)
    ct = p.continuum(3300+15*wave)[0]
    st = 60*wave+12*((xx+yy).remainder(2)*2-1)
    bt = st.new_tensor([.3, -.07, .03])
    linear = common.frozen_v5("signed_linear_solver").SignedLinearOperator(p)
    ce = ct-1.5*p.M[:, 2, None, None]*wave
    leak = p.op((ct-ce)[None])[0]
    noise = .35*leak+.25*torch.sin(4*torch.pi*(xx+yy)/size)
    raw = p.op(ct[None])[0]+linear.apply_template(st)+linear.apply_beta(bt)+noise
    # A nonuniform positive W supplies the same actual objective in every source/mode.
    W = (raw.clamp_min(0)+p.extra_variance).reciprocal()
    options = {**OPTIONS, "weights": W}
    if dtype == torch.float64:
        options["diagonal_mode"] = "exact_small"
    return p, raw, ce, ct, st, bt, options


def dense_system(system):
    n = system["dimension"]
    if n > 1024:
        raise ValueError("Dense oracle is confined to <=1024 CPU fixture coordinates")
    eye = torch.eye(n, dtype=system["rhs"].dtype)
    return torch.stack([system["matvec"](e) for e in eye], 1)


def close(a, b, atol=1e-8, rtol=1e-8):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"max absolute difference {float((a-b).abs().max())}")


def numerical_checks(mode):
    f = fixture()
    result = d.decompose(*f[:6], f[6], mode)
    summary, a = result["summary"], result["arrays"]
    # High-count float32 fixtures intentionally expose two distinct numerical
    # limitations. Detecting them is successful implementation QA, not scientific
    # acceptance of their reconstructions, and does not change the default gates.
    expected_failure = "normal_equation_algebra_passed" if mode else "postblur_recomposition_passed"
    assert not summary["numerical_gate_passed"]
    assert expected_failure in summary["failure_reasons"], summary["failure_reasons"]
    assert summary["postblur_recomposition_passed"] == bool(mode)
    assert summary["full_leak_noise_bias_CG_converged"]
    assert summary["actual_v5_accepted"] and summary["actual_v5_CG_converged"]
    assert summary["raw_continuum_noise_correlation"]["pearson"] > .2
    close(a["component_sum_postblur_e"], a["leakage_postblur_e"]+a["noise_postblur_e"]+a["regularization_bias_postblur_e"], atol=1e-5, rtol=1e-6)
    close(torch.tensor(summary["postblur_mse_accounted_e2"]),
          torch.tensor(summary["postblur_mse_of_component_sum_e2"]), atol=1e-6, rtol=1e-6)
    # Every component has the same original dual true-residual budget.
    for cg in summary["component_CG"].values():
        assert cg["rtol"] == 1e-4 and cg["max_iterations"] == 200
        assert cg["relative_true_residual"] <= 1e-4
        assert cg["equilibrated_relative_true_residual"] <= 1e-4
    return {"mode": mode, "postblur_rms_gap_e": summary["postblur_recomposition_residual"]["rms"],
            "relative_l2": summary["postblur_recomposition_relative_l2"],
            "known_numerical_failure_correctly_detected": summary["failure_reasons"],
            "max_component_iterations": max(cg["iterations"] for cg in summary["component_CG"].values()),
            "wall_s": summary["wall_s"]}


def dense_reference(mode):
    p, raw, ce, ct, st, bt, options = fixture(torch.float64)
    cache = d.build_shared_cache(p, raw, ct, st, bt, options, mode)
    system = cache["system"]
    K = dense_system(system)
    close(K, K.T, atol=2e-10, rtol=2e-10)
    assert float(torch.linalg.eigvalsh(K).min()) > 0
    ce_raw = p.op(ce[None])[0]
    target = raw-ce_raw
    if mode:
        target -= system["linear_operator"].apply_beta(bt)
    rhs = system["measurement_adjoint"](system["weights"]*target)
    x = torch.linalg.solve(K, rhs)
    leak_rhs = system["measurement_adjoint"](system["weights"]*p.op((ct-ce)[None])[0])
    pieces = [torch.linalg.solve(K, r) for r in (leak_rhs, cache["noise"]["rhs"], cache["regularization_bias"]["rhs"])]
    discrepancy = x-cache["truth_vector"]-sum(pieces)
    s, _ = system["unpack"](discrepancy)
    rms = d._rms(p.op.object_blur(s))
    assert rms < 1e-7, rms
    # Verify numerical inverse separately from the exact algebraic identity.
    for name in ("noise", "regularization_bias"):
        residual = cache[name]["rhs"]-K@cache[name]["value"]
        norm = float(torch.linalg.vector_norm(cache[name]["rhs"]))
        assert float(torch.linalg.vector_norm(residual))/max(norm, 1e-30) <= 1.001e-4
    return {"mode": mode, "dense_dimension": system["dimension"], "postblur_dense_gap_e": rms}


def cache_weights_zero_and_modes():
    p, raw, ce, ct, st, bt, options = fixture()
    cache = d.build_shared_cache(p, raw, ct, st, bt, options, False)
    result = d.decompose(p, raw, ce, ct, st, bt, options, False, shared_cache=cache)
    oracle = d.decompose(p, raw, ct, ct, st, bt, options, False, shared_cache=cache)
    assert oracle["shared_cache"] is cache
    assert torch.equal(oracle["arrays"]["leakage_postblur_e"], torch.zeros_like(st))
    assert oracle["summary"]["component_CG"]["leakage"]["stop"] == "ZERO_RHS"
    assert result["summary"]["component_CG"]["noise"] == oracle["summary"]["component_CG"]["noise"]
    assert result["summary"]["component_CG"]["regularization_bias"] == oracle["summary"]["component_CG"]["regularization_bias"]
    assert result["summary"]["weights_sha256"] == oracle["summary"]["weights_sha256"]
    assert cache["noise"]["value"].data_ptr() == result["shared_cache"]["noise"]["value"].data_ptr()
    for changed in ({**options, "weights": options["weights"]*2}, {**options, "lambda0_factor": 2e-6}):
        try:
            d.decompose(p, raw, ce, ct, st, bt, changed, False, shared_cache=cache)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed fixed objective must invalidate cache")
    try:
        d.decompose(p, raw, ce, ct, st, bt, options, True, shared_cache=cache)
    except ValueError:
        pass
    else:
        raise AssertionError("Beta-mode changes must invalidate cache")
    # Uniform-W and nonuniform-W objectives really differ.
    other = d.build_shared_cache(p, raw, ct, st, bt, {**options, "weights": options["weights"].mean()}, False)
    probe = cache["truth_vector"]
    assert float((cache["system"]["measurement_matvec"](probe)-other["system"]["measurement_matvec"](probe)).abs().max()) > 1e-3
    for bad in (torch.zeros_like(raw), torch.full_like(raw, float("nan"))):
        try:
            d.build_shared_cache(p, raw, ct, st, bt, {**options, "weights": bad}, False)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid W must not be masked")
    return {"cache_noise_bias_reused": True, "C_true_leakage": "exact zero", "W_hash": cache["weight_sha256"]}


def all_zero_and_otf_null():
    p, _, _, ct, st, bt, options = fixture()
    zeros = torch.zeros_like(st)
    result = d.decompose(p, zeros, torch.zeros_like(ct), torch.zeros_like(ct), zeros,
                         torch.zeros_like(bt), options, False)
    assert result["summary"]["numerical_gate_passed"]
    assert all(cg["stop"] == "ZERO_RHS" for cg in result["summary"]["component_CG"].values())
    assert result["summary"]["raw_continuum_noise_correlation"]["pearson"] is None
    size = st.shape[0]
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    null = ((xx+yy).remainder(2)*2-1).to(st)*60
    assert float(p.op.object_blur(null).abs().max()) == 0
    ct = ct[:, :1, :1].expand_as(ct).clone()
    raw = p.op(ct[None])[0]
    hidden = d.decompose(p, raw, ct, ct, null, torch.zeros_like(bt), options, True)
    assert hidden["summary"]["numerical_gate_passed"]
    assert d._rms(hidden["arrays"]["template_latent_error_e"]) > 59
    assert d._rms(hidden["arrays"]["template_postblur_error_e"]) < .01
    assert bool((p.op.object_otf == 0).any())
    return {"zero_components_exact": True, "zero_otf_bins": int((p.op.object_otf == 0).sum()),
            "unobservable_latent_error_rms_e": d._rms(hidden["arrays"]["template_latent_error_e"]),
            "observable_error_rms_e": d._rms(hidden["arrays"]["template_postblur_error_e"])}


def single_blur_and_signed_symmetry():
    p, raw, ce, ct, st, bt, options = fixture()
    linear = common.frozen_v5("signed_linear_solver").SignedLinearOperator(p)
    full = p.op(p.M[:, 2, None, None]*st)[...]
    close(linear.apply_template(st), full, atol=2e-5, rtol=2e-5)
    # Applying H again measurably changes a supported nonconstant field.
    wrong = linear.apply_template(p.op.object_blur(st))
    assert float((wrong-full).square().mean().sqrt()) > 1
    outputs = []
    for sign in (1, -1):
        signed_raw = p.op(ct[None])[0]+linear.apply_template(sign*st)
        out = common.frozen_v5("signed_linear_solver").solve_signed_linear(
            p, signed_raw[None], ct[None], options, fixed_beta=torch.zeros_like(bt)[None])
        assert out["info"]["all_CG_converged"] and out["info"]["all_accepted"]
        outputs.append(out["template_postblur_e"][0])
    close(outputs[0], -outputs[1], atol=.01, rtol=1e-3)
    assert bool((outputs[0] < 0).any() and (outputs[0] > 0).any())
    return {"single_object_blur_verified": True, "signed_fields_preserved": True,
            "signed_symmetry_max_gap_e": float((outputs[0]+outputs[1]).abs().max())}


def equivalent_beta_units_and_weight_scale():
    p, raw, ce, ct, st, bt, options = fixture(torch.float64)
    p2 = copy.deepcopy(p)
    p2.Bscale *= 2
    p2.cfg = {**p2.cfg, "background_scale_e": p2.Bscale}
    caches = [d.build_shared_cache(p, raw, ct, st, bt, options, False),
              d.build_shared_cache(p2, raw, ct, st, bt/2, options, False),
              d.build_shared_cache(p, raw, ct, st, bt, {**options, "weights": options["weights"]*3}, False)]
    values = []
    for cache in caches:
        system = cache["system"]
        K = dense_system(system)
        target = raw-p.op(ce[None])[0]
        values.append(system["unpack"](torch.linalg.solve(K, system["measurement_adjoint"](system["weights"]*target))))
    close(values[0][0], values[1][0], atol=1e-7, rtol=1e-7)
    close(values[0][1], values[1][1]*2, atol=1e-7, rtol=1e-7)
    close(values[0][0], values[2][0], atol=1e-7, rtol=1e-7)
    close(values[0][1], values[2][1], atol=1e-7, rtol=1e-7)
    return {"equivalent_beta_units_dense": True, "uniform_W_scale_dense": True}


def visible_failure_and_optional_reference():
    p, raw, ce, ct, st, bt, options = fixture()
    passed = d.decompose(p, torch.zeros_like(raw), torch.zeros_like(ct), torch.zeros_like(ct),
                         torch.zeros_like(st), torch.zeros_like(bt), options, True, float64_reference=True)
    assert passed["summary"]["numerical_gate_passed"]
    assert not passed["summary"]["float64_reference"]["ran"]
    failed = d.decompose(p, raw, ce, ct, st, bt, options, False, float64_reference=True)
    assert not failed["summary"]["numerical_gate_passed"]
    assert failed["summary"]["float64_reference"]["ran"]
    assert "postblur_recomposition_passed" in failed["summary"]["failure_reasons"]
    rc = failed["shared_cache"]["float64_cache"]
    primary = failed["shared_cache"]
    assert all(torch.equal(b, dict(primary["physics"].named_buffers())[name].to(
                   dtype=torch.complex128 if b.is_complex() else torch.float64))
               for name, b in rc["physics"].named_buffers())
    for key in ("lambda_g", "lambda0", "lambda_beta"):
        assert float(rc["system"][key]) == float(primary["system"][key])
    for name in ("template_pattern", "beta_images", "stationary_normal_symbol", "laplacian_symbol"):
        assert torch.equal(getattr(rc["system"]["linear_operator"], name),
                           getattr(primary["system"]["linear_operator"], name).double())
    for name in ("noise", "regularization_bias"):
        assert torch.equal(rc[name]["rhs"], primary[name]["rhs"].double())
    for cg in failed["summary"]["float64_reference"]["diagnostic"]["component_CG"].values():
        assert cg["rtol"] == 1e-4 and cg["max_iterations"] == 200 and cg["dtype"] == "torch.float64"
    # Actual v5 rejection and nonconvergence remain independent visible failure gates.
    residual_diagnostic = failed["summary"]["finite_solve_residual_diagnostic"]
    assert residual_diagnostic["negative_actual_residual_CG"]["max_iterations"] == 200
    assert residual_diagnostic["combined_normal_gap_CG"]["max_iterations"] == 200
    assert residual_diagnostic["unexplained_after_combined_residual"]["rms"] < .005
    actual = common.frozen_v5("signed_linear_solver").solve_signed_linear(p, raw[None], ce[None], options)
    injected = copy.deepcopy(actual)
    injected["info"]["frames"][0]["accepted"] = False
    injected["info"]["frames"][0]["CG"]["converged"] = False
    flagged = d.decompose(p, raw, ce, ct, st, bt, options, False, v5_result=injected,
                          shared_cache=failed["shared_cache"])
    assert not flagged["summary"]["actual_v5_accepted"] and not flagged["summary"]["actual_v5_CG_converged"]
    assert not flagged["summary"]["numerical_gate_passed"]
    nonfinite = copy.deepcopy(actual)
    nonfinite["template_latent_e"][0, 0, 0] = float("nan")
    nonfinite["template_postblur_e"][0, 0, 0] = float("nan")
    invalid = d.decompose(p, raw, ce, ct, st, bt, options, False, v5_result=nonfinite,
                          shared_cache=failed["shared_cache"])
    assert not invalid["summary"]["all_outputs_finite"]
    assert invalid["summary"]["postblur_error"]["count"] == raw.numel()
    assert invalid["summary"]["postblur_error"]["rms"] is None
    assert not invalid["summary"]["numerical_gate_passed"]
    return {"failure_retained": True, "double_only_after_failure": True,
            "double_uses_primary_rhs": True, "double_cannot_rescue": True,
            "primary_gap_e": failed["summary"]["postblur_recomposition_residual"]["rms"],
            "finite_solve_residual_diagnostic": residual_diagnostic}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(common.ROOT/"reports/research_v6/implementation_checks/decomposition"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    common.ed_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checks = [
        ("fixed_beta_float32", lambda: numerical_checks(True)),
        ("estimated_beta_float32", lambda: numerical_checks(False)),
        ("fixed_beta_dense_reference", lambda: dense_reference(True)),
        ("estimated_beta_dense_reference", lambda: dense_reference(False)),
        ("shared_cache_fixed_W", cache_weights_zero_and_modes),
        ("zero_terms_zero_OTF", all_zero_and_otf_null),
        ("single_blur_signed_fields", single_blur_and_signed_symmetry),
        ("scale_equivalent_beta_units", equivalent_beta_units_and_weight_scale),
        ("visible_failure_optional64", visible_failure_and_optional_reference),
    ]
    rows, started = [], time.perf_counter()
    for name, check in checks:
        t = time.perf_counter()
        try:
            value = check()
            rows.append({"check": name, "passed": True, "details": value, "wall_s": time.perf_counter()-t})
        except Exception as exc:
            rows.append({"check": name, "passed": False, "error": str(exc), "traceback": traceback.format_exc(),
                         "wall_s": time.perf_counter()-t})
        print(name, "PASS" if rows[-1]["passed"] else "FAIL", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    p, raw, ce, ct, st, bt, fixture_options = fixture()
    fixture_path = common.ROOT/"outputs/research_v6/fixtures/decomposition"/("cpu16_"+stamp+".npz")
    fixture_receipt = common.save_npz(fixture_path, raw_e=raw.numpy(), continuum_est_latent_e=ce.numpy(),
                                     continuum_true_latent_e=ct.numpy(), template_latent_true_e=st.numpy(),
                                     beta_true=bt.numpy(), weights=fixture_options["weights"].numpy())
    report = {"schema": "v6-decomposition-cpu-fixtures-001", "timestamp": common.now(),
              "scope": "bounded CPU synthetic fixtures; no formal arrays; not Stage A scientific evidence",
              "all_passed": all(r["passed"] for r in rows), "checks": rows,
              "default_identity_criteria": d.DEFAULT_IDENTITY_CRITERIA,
              "numeric_budget": {"rtol": 1e-4, "max_iterations": 200, "threads": torch.get_num_threads()},
              "wall_s": time.perf_counter()-started,
              "fixture_artifact": fixture_receipt, "fixture_physics_config": p.cfg,
              "source_sha256": {common.relative(Path(d.__file__)): common.sha(d.__file__),
                                common.relative(Path(__file__)): common.sha(__file__)}}
    path = common.ed_path(args.output_dir)/("cpu_fixture_checks_"+stamp+".json")
    common.write_json(path, report)
    print(common.relative(path), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
