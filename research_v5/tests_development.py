"""Small CPU development/statistics/review contracts; no formal family arrays.

Run ordinary Python -B -m research_v5.tests_development in the E-drive repo.
The live numerical path uses only a declared 16x16 physics fixture. Receipts
are software controls, never evidence of scientific development acceptance.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import time
import uuid
from unittest.mock import patch

from . import common

common.configure_runtime()

import numpy as np
import torch

from . import development
from .candidate_review import choose_candidate, final_mc_case_ids
from .planning import build_family_plans
from .statistics import (aggregate_metrics, mother_bootstrap, mother_metric,
                         paired_statistics, statistics_plan)
from .signed_linear_solver import SignedLinearOperator, create_linear_system
from .signed_linear_solver import DEFAULT_OPTIONS as LINEAR_DEFAULTS
from .uncertainty import monte_carlo_uncertainty, hessian_probe_diagnostics


def _record(mother, case, method, mse, *, sign="positive", color=17., alpha=None, raw_hash=None):
    result = {"mother_id": mother, "case_id": case, "method": method, "sign": sign,
              "group": "normal", "input_kind": "noise", "partition_role": "development_selection",
              "raw_sha256": raw_hash or case, "numerical_execution_passed": True,
              "metrics": {"domain_pixels": 16, "accepted_pixels": 16,
                  "template_SSE": 16*mse, "template_truth_energy": 1600.,
                  "template_error_sum": 0., "temperature_SSE_K2": 16*color**2,
                  "temperature_error_sum_K": 0., "signed_projection_numerator": 1600.},
              "diagnostics": {"seconds": .01, "gradient_energy_S_e2": 10.,
                  "CG": {"converged": True, "relative_true_residual": 1e-6,
                         "equilibrated_relative_true_residual": 1e-6}}}
    if alpha is not None:
        result["alpha_g"] = alpha
    return result


def _fixture():
    cfg = copy.deepcopy(common.physics_config())
    old = cfg["shape"]
    cfg["shape"] = [16, 16]
    cfg["carrier_bins_xy"] = [[round(x*16/old[1]), round(y*16/old[0])] for x, y in cfg["carrier_bins_xy"]]
    p = common.make_physics("cpu", torch.float64, config=cfg)
    state = torch.zeros((1, 5, 16, 16), dtype=torch.float64)
    state[:, 0] = (3202.3-p.T0)/p.Tscale
    state[:, 1] = math.log(.6)
    state[:, 4] = 45./p.Sscale
    beta = torch.tensor([[.45, .02, -.01]], dtype=torch.float64)
    expected = p(state, beta)
    raw = expected/p.adc_step+p.cfg["black_level_DN"]
    core = {"device": "cpu", "dtype": "float64", "continuum_options": {"warm_iterations": 1},
            "linear_options": {"alpha_g": .1}}
    return p, raw, core


def run_tests(output=None):
    assert not sys.flags.optimize and sys.dont_write_bytecode
    started = time.perf_counter()
    torch.set_num_threads(2)
    checks, details = [], {}

    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})
        assert passed, checks[-1]

    # Whole-mother means are unchanged by extra repeats of just one mother.
    a = [_record("m1", "m1r0", "candidate", 1.),
         _record("m2", "m2r0", "candidate", 9.)]
    b = [*a, *[_record("m1", f"m1r{i}", "candidate", 1.) for i in range(1, 8)]]
    bs1, bs2 = mother_bootstrap(mother_metric(a)), mother_bootstrap(mother_metric(b))
    check("mother_cluster_not_pixel_or_repeat_independence", bs1 == bs2 and bs1["estimate"] == 5.)
    check("fixed_2000_seed_53130_bootstrap", bs1["draws"] == 2000 and bs1["seed"] == 53130)
    reference = [_record(r["mother_id"], r["case_id"], "reference", 2.) for r in a]
    paired = paired_statistics([*a, *reference], "candidate", "reference")
    check("same_raw_paired_mother_bootstrap", paired["complete"] and paired["paired_parent_RMSE_ratio"] == math.sqrt(5/2))
    check("missing_case_invalidates_paired_population", not paired_statistics([*a, reference[0]], "candidate", "reference")["complete"])
    try:
        wrong = copy.deepcopy(reference)
        wrong[0]["raw_sha256"] = "other_raw"
        paired_statistics([*a, *wrong], "candidate", "reference")
    except ValueError:
        check("different_raw_pair_hash_rejected", True)
    else:
        check("different_raw_pair_hash_rejected", False)
    rejected = copy.deepcopy(a)
    rejected[1]["metrics"]["accepted_pixels"] = 0
    ag = aggregate_metrics(rejected)
    check("rejection_never_shrinks_full_error_domain", ag["template_MSE_e2"] == 5. and ag["valid_fraction"] == .5)
    details["bootstrap"] = bs1

    rows = []
    for mid in ("m1", "m2"):
        for sign in ("positive", "negative"):
            cid = mid+"_"+sign
            for alpha, mse in ((0., 2.), (.01, 1.), (.1, .5), (1., .1)):
                rows.append(_record(mid, cid, "v5_physics", mse, sign=sign,
                                    color=21. if alpha == 1. else 17., alpha=alpha))
            rows.extend([_record(mid, cid, "B1_tikhonov", 1., sign=sign),
                         _record(mid, cid, "v3_physics", .9, sign=sign)])
    selection = development._selection(rows, [0., .01, .1, 1.])
    check("selection_only_normal_noisy_mother_mse_and_color_gate",
          selection["selected_alpha_g"] == .1 and selection["traditional_reference"] == "v3_physics")
    selected = [r for r in rows if r["method"] == "v3_physics" or r.get("alpha_g") == .1]
    null_eval = {"complete": True, "frame_FP": .02, "all_numerical_predictions_passed": True}
    mc = {"complete": True, "gate_coverage": .95}
    stage1 = {"stage1_evidence_complete": True, "physical_stage2_eligible": True, "ml_stop_required": False}
    gate = development.evaluate_physics_gate(selected, "v3_physics", True, null_eval, mc, stage1)
    check("all_declared_physics_gate_factors", gate["physics_eligible"] and gate["no_hardstop"])
    for field, bad in (("frame_FP", .051),):
        negative = dict(null_eval, **{field: bad})
        check("null_gate_fail_closed", not development.evaluate_physics_gate(selected, "v3_physics", True, negative, mc, stage1)["physics_eligible"])
    check("MC_coverage_gate_fail_closed", not development.evaluate_physics_gate(selected, "v3_physics", True, null_eval,
                               {"complete": True, "gate_coverage": .89}, stage1)["physics_eligible"])
    check("stage1_hard_stop_cannot_be_overridden", not development.evaluate_physics_gate(selected, "v3_physics", True, null_eval,
                               mc, dict(stage1, ml_stop_required=True))["physics_eligible"])

    # One exceptional pixel must affect max scoring despite a zero 99th percentile.
    template = torch.zeros((1, 16, 16), dtype=torch.float64)
    template[0, 0, 0] = 10.
    prediction = {"template_postblur_e": template, "valid": torch.zeros_like(template, dtype=torch.bool)}
    detection = development.detection_diagnostics(prediction, torch.ones_like(template), 5.)
    check("null_threshold_score_is_full_FOV_maximum", detection["frame_detected"] and
          detection["frame_detection_score"] == 10. and float(torch.quantile(template.flatten(), .99)) == 0.)
    template[0, 1, 1] = float("nan")
    check("nonfinite_pixel_cannot_disappear_from_FOV_control", not development.detection_diagnostics(prediction, torch.ones_like(template), 5.)["complete"])
    calibration = [{"case_id": f"c{i}", "mother_id": f"m{i//2}", "numerical_execution_passed": True,
                    "detection": {"complete": True, "frame_detection_score": float(i+1)}} for i in range(20)]
    threshold = development.calibrate_null_threshold(calibration)
    check("independent_null_higher_quantile_is_fixed", threshold["z_threshold"] == 20. and threshold["method"] == "higher"
          and threshold["retuned_on_evaluation"] is False)
    broken_cal = copy.deepcopy(calibration)
    broken_cal[0]["numerical_execution_passed"] = False
    check("failed_null_fit_invalidates_calibration", not development.calibrate_null_threshold(broken_cal)["complete"])
    heldout = [{"case_id": f"e{i}", "mother_id": f"e{i//8}", "numerical_execution_passed": True,
                "detection": {"frame_detected": i < 2, "pixels": 256, "detected_pixels": 1 if i < 2 else 0}}
               for i in range(96)]
    heldout_summary = development._null_summary(heldout, threshold["z_threshold"])
    check("null_FP_reports_frames_and_pixels_separately", heldout_summary["frame_FP"] == 2/96 and
          heldout_summary["pixel_FP"] == 2/(96*256) and heldout_summary["mothers"] == 12)

    p, raw, core = _fixture()
    proxy = development.raw_noise_proxy(p, raw, core)
    linear = SignedLinearOperator(p)
    raw_e = p.raw_to_electrons(raw)
    symbol = (raw_e.clamp_min(0)+p.extra_variance).reciprocal().mean()*linear.stationary_normal_symbol
    dense_diagonal, _ = linear.template_diagonal((raw_e[0].clamp_min(0)+p.extra_variance).reciprocal(), "exact_small")
    scale = dense_diagonal.median()
    expected_variance = (p.op.object_otf.square()*symbol/(symbol+.1*scale*linear.laplacian_symbol+1e-6*scale).square()).mean()
    check("raw_only_proxy_matches_declared_postblur_spectral_formula",
          torch.allclose(proxy["sigma_S"].square().mean(), expected_variance, rtol=1e-12, atol=1e-12))
    check("proxy_limitations_and_no_MC_claim", proxy["summary"]["truth_used"] is False and
          any("not an MC sigma" in text for text in proxy["summary"]["limitations"]))
    details["proxy"] = proxy["summary"]

    calls = []
    def actual_raw_only_callback(raw_dn, calibration, selection):
        assert set(calibration) == {"physics_config"}
        assert set(selection) == {"device", "dtype", "continuum_options", "linear_options"}
        calls.append(tuple(raw_dn.shape))
        return development.reconstruct(raw_dn, calibration, selection)
    baseline = actual_raw_only_callback(raw, {"physics_config": p.cfg}, core)
    calls.clear()
    uncertainty = monte_carlo_uncertainty(p, raw, actual_raw_only_callback, {"physics_config": p.cfg}, core,
                                         {"repeats": 8}, baseline=baseline)
    check("MC_real_raw_callback_reruns_core_eight_times", len(calls) == 8 and uncertainty["status"] == "completed"
          and uncertainty["summary"]["completed_repeats"] == 8)
    check("MC_nominal_coverage_is_external_not_inferred", uncertainty["summary"]["empirical_nominal_coverage"].startswith("not assessed"))
    details["MC"] = uncertainty["summary"]

    # Exercise the actual comparison orchestration with in-memory fixture I/O.
    # The four baseline components are explicit mock adapters, not old-weight
    # performance evidence; all four real signed solves and metrics execute.
    fixture_row = {"case_id": "cpu_contract_case", "mother_id": "cpu_contract_mother",
        "partition_role": "development_selection", "condition": "positive_normal", "group": "normal",
        "coefficient_e": 180., "input_kind": "noise", "geometry": "fixture", "template_geometry": "fixture",
        "noise_index": 0, "stress": None, "raw_file_sha256": "cpu_fixture_raw", "label_sha256": "cpu_fixture_label"}
    truth = {"evaluation_domain": np.ones((16, 16), dtype=bool),
             "template_postblur_e": np.full((16, 16), 45.),
             "temperature_color_K": np.full((16, 16), 3202.3)}
    saved, logs, continuum_calls, baseline_calls = {}, [], [], []
    estimator = development.estimate_continuum
    def continuum_once(*args, **kwargs):
        continuum_calls.append(1)
        return estimator(*args, **kwargs)
    def virtual_save(path, values):
        saved["values"] = {k: v.detach().clone() for k, v in values.items()}
        return {"path": "CPU_IN_MEMORY_FIXTURE", "sha256": "CPU_IN_MEMORY_FIXTURE"}
    class FourMockBaselines:
        def evaluate(self, observed_raw):
            baseline_calls.append(torch.equal(observed_raw, raw))
            result = {}
            for name in ("B1_tikhonov", "v3_physics", "old_B4", "v4.4_final_continuum_fixed"):
                result[name] = {**baseline, "info": {"method": name, "seconds": .01, "status": "SUCCESS"}}
            return result
    def virtual_load(rec, unused_physics):
        values, prefix = saved["values"], rec["prediction_cache"]["alpha_key"]
        out = {k: values[k].clone() for k in ("continuum_latent", "continuum_postblur", "temperature_color_K")}
        out.update({k: values[prefix+"__"+k].clone() for k in
                    ("template_latent_e", "template_postblur_e", "beta", "expected_e", "valid")})
        out["info"] = {"cache_of_same_raw": True}
        return out
    minimal_protocol = {"alpha_g_development_grid": [0., .01, .1, 1.],
                        "linear_defaults": LINEAR_DEFAULTS, "continuum_options": core["continuum_options"],
                        "baselines": ["B1_tikhonov", "v3_physics", "old_B4", "v4.4_final_continuum_fixed"]}
    with patch.object(development, "_inputs", return_value=(raw, truth)), \
         patch.object(development, "_save_arrays", side_effect=virtual_save), \
         patch.object(development, "estimate_continuum", side_effect=continuum_once), \
         patch.object(common, "append_jsonl", side_effect=lambda path, value: logs.append(value)), \
         patch.object(development, "load_cached_prediction", side_effect=virtual_load):
        grid_records = development._run_grid(p, common.DATA/"cpu_virtual", {"rows": [fixture_row]},
                                              minimal_protocol, FourMockBaselines(), common.Budget(15))
        picked = [r for r in grid_records if r.get("alpha_g") == .1]
        rescored = development._rescore_selected(p, common.DATA/"cpu_virtual", {"rows": [fixture_row]},
                                                  picked, core, 100., common.Budget(15))
    check("real_grid_once_continuum_four_alphas_and_same_raw_four_references",
          len(continuum_calls) == 1 and baseline_calls == [True] and len(grid_records) == 8)
    check("hash_bound_selected_cache_rescores_without_reconstruction_or_CG_loss",
          len(rescored) == 1 and rescored[0].get("prediction_cache") == picked[0]["prediction_cache"]
          and rescored[0]["diagnostics"]["CG"] == picked[0]["diagnostics"]["CG"]
          and rescored[0].get("detection", {}).get("complete") is True)
    check("all_real_fixture_comparison_records_JSON_safe", bool(json.dumps(logs, allow_nan=False)))

    # Exact diagonal control proves the sandwich target and coordinate units.
    diagonal = torch.arange(1., 8., dtype=torch.float64)
    probe = hessian_probe_diagnostics(lambda x: diagonal*x, torch.zeros(7, dtype=torch.float64),
                                     lambda x: 2*x, {"probes": 8}, preconditioner=lambda x: x/diagonal)
    check("eight_finite_Hessian_probes_are_packed_coordinate_sandwich", probe["summary"]["successful_probes"] == 8
          and torch.allclose(probe["diagonal_estimate"], 2/diagonal.square(), rtol=1e-12, atol=1e-12))
    system = create_linear_system(p, raw_e, baseline["continuum_latent"], core["linear_options"])
    check("real_selected_Hessian_coordinates_latent_S_plus_beta", system["dimension"] == 16*16+3
          and system["n_S"] == 256 and system["n_beta"] == 3)
    details["Hessian"] = probe["summary"]

    physics_choice = choose_candidate(gate)
    check("eligible_physics_does_not_force_neural_training", physics_choice["candidate_kind"] == "v5_physics" and physics_choice["eligible"])
    check("failed_physics_leaves_final_sealed", not choose_candidate(dict(gate, physics_eligible=False))["eligible"])
    train = {"accepted": True, "completed_registered_training": True, "actual_optimizer_updates": 1000,
             "selected_checkpoint": {"path": "outputs/research_v5/mock.pt", "sha256": "mock", "updates": 500,
                "accepted": True, "weights_changed_from_initial": True, "gates": {"positive": True, "negative": True}}}
    check("one_accepted_neural_checkpoint_preserved", choose_candidate(gate, train)["candidate_kind"] == "v5_neural")
    train["selected_checkpoint"]["gates"]["negative"] = False
    check("single_sign_neural_claim_rejected", not choose_candidate(gate, train)["eligible"])
    plans = build_family_plans()  # Descriptor metadata only; never generate arrays.
    ids = final_mc_case_ids(plans)
    check("final_MC_subset_frozen_from_metadata_only", len(ids) == len(set(ids)) == 12
          and len(plans["plans"]["signed_confirmation"]) == 288 and len(plans["plans"]["final_once"]) == 512)
    check("statistical_plan_has_no_pixel_independence", statistics_plan()["pixel_independence_assumed"] is False)
    from .final import frozen_evaluation_fields, validate_candidate_and_protocol
    frozen_fields = frozen_evaluation_fields(plans)
    candidate = {**physics_choice, "core_selection": core, "statistics": statistics_plan(),
                 "thresholds": {"path": "cpu_mock", "sha256": "cpu_mock", "z_threshold": 3.}}
    frozen = {"status": "FROZEN_CANDIDATE_THRESHOLDS_STATISTICS", "candidate_kind": "v5_physics",
              "core_selection": core, "thresholds": candidate["thresholds"], "statistics": statistics_plan(),
              "plan_hashes": plans["plan_hashes"], **frozen_fields}
    check("review_and_final_consumer_share_exact_frozen_evaluation_schema",
          validate_candidate_and_protocol(candidate, frozen, {"family_plan_hashes": plans["plan_hashes"]}, plans))
    sources = {common.relative(common.CODE/name): common.sha(common.CODE/name)
               for name in ("development.py", "statistics.py", "candidate_review.py", "tests_development.py")}
    receipt = {"schema": "v5-development-implementation-controls-001", "all_passed": all(c["passed"] for c in checks),
               "checks": checks, "source_sha256": sources, "details": details,
               "device": "cpu", "numerical_fixture_shape": [16, 16],
               "assertions_enabled": True, "no_bytecode": True,
               "formal_data_generated": False, "GPU_used": False, "training_executed": False,
               "sealed_final_or_confirmation_arrays_opened": False,
               "duration_seconds": time.perf_counter()-started,
               "scope": "CPU software contracts and real tiny inference; no scientific acceptance"}
    if output is None:
        output = common.REPORTS.parent/"implementation_checks"/"development"/("cpu_"+uuid.uuid4().hex)/"test_receipt.json"
    common.write_json(output, receipt)
    return receipt, Path(output)


if __name__ == "__main__":
    value, path = run_tests()
    print(json.dumps({"all_passed": value["all_passed"], "checks": len(value["checks"]),
                      "duration_seconds": value["duration_seconds"], "receipt": str(path)}, ensure_ascii=False))
