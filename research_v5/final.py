"""Freeze-checked, O_EXCL, one-time signed confirmation and original final test.

This consumer has no threshold selection, checkpoint selection, recalibration,
or retry path. It opens the reserved mothers only after the exact candidate,
statistics, thresholds, plans and source hashes have been verified and the
single durable start guard exists. Failure is retained without retuning.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from research_v5 import common

common.configure_runtime()

import numpy as np
import torch

from research_v5.train import exclusive_json, _checked_case, _checked_manifest, _metadata

FROZEN_STATUS = "FROZEN_CANDIDATE_THRESHOLDS_STATISTICS"
CANONICAL_CANDIDATE = common.REPORTS / "candidate_review" / "candidate_receipt.json"
CASE_COUNTS = {"signed_confirmation": 288, "final_once": 512, "total": 800}
EVALUATION_RULES = {
    "raw_likelihood_count": 1, "truth_in_reconstruction": False,
    "targets": ["template_postblur_e", "temperature_color_K"],
    "domain": "full registered truth evaluation_domain including invalid/rejected estimates",
    "statistical_unit": "mother_scene", "partitions_reported_separately": ["signed_confirmation", "final_once"],
    "sign_and_group_reported_separately": True, "all_four_baselines_on_the_same_raw": True,
    "nominal_postblur_color_RMSE_K_max": 20.,
    "normal_signed_RMSE_factor_over_frozen_traditional": 1.05,
    "valid_fraction_min": .99,
    "final_null_frame_FP_max": .05,
    "MC_nominal_parent_mean_interval_coverage_min": .90,
    "normal_amplitude_projection_rule": "absolute deviation from unit projection no larger than 1.05 times frozen traditional",
    "frame_detection_score": "max_abs_template_over_raw_proxy_full_FOV",
    "weak_population_rule": "raw-only frozen-threshold detection; sign is external scoring only",
    "undetectable_weak_required_L2_below_one": False,
    "below_limit_output": "template_abstain and low_confidence with approximate signed interval",
    "proxy_is_not_calibrated_confidence_interval": True,
    "calibration_stress_reporting": "separate existing development simulation only; no nominal pooling or real-temperature claim",
    "final_threshold_recalibration": False,
    "final_model_or_selection_adaptation": False,
}


def final_mc_case_ids(plans):
    """Metadata-only fixed 12 cases; choose before opening final payloads."""
    signed = plans["plans"]["signed_confirmation"]
    final = plans["plans"]["final_once"]
    signed_mothers = [m["mother_id"] for m in plans["mothers"]["signed_confirmation"][:3]]
    final_mothers = [m["mother_id"] for m in plans["mothers"]["final_once"][:6]]
    selected = [r["case_id"] for r in signed if r["mother_id"] in signed_mothers
                and r["group"] == "weak" and r["noise_index"] == 0]
    selected += [r["case_id"] for r in final if r["mother_id"] in final_mothers
                 and r["group"] == "weak" and r["noise_index"] == 0]
    if len(selected) != 12 or len(set(selected)) != 12:
        raise ValueError("Final MC metadata subset must be the fixed 6 signed + 6 original-final weak inputs")
    return selected


def frozen_evaluation_fields(plans):
    """Fields used by candidate review when freezing the final protocol."""
    return {"case_counts": copy.deepcopy(CASE_COUNTS), "evaluation_rules": copy.deepcopy(EVALUATION_RULES),
            "final_MC_case_ids": final_mc_case_ids(plans),
            "final_MC_rule": "first three registered signed mothers, both weak signs/noise0; first six registered original-final mothers weak/noise0",
            "final_MC": {"repeats": 8, "seed": 53125, "confidence_level": .95,
                "rerun_continuum_each_repeat": True, "truth_used_only_for_external_coverage": True,
                "MC_coverage_claim_from_proxy": False}}


def _artifact(path_value, digest, expected_path=None):
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError("Frozen artifact needs its exact path and SHA256")
    path = common.ed_path(common.ROOT / path_value)
    if not path.is_relative_to(common.ROOT.resolve()) or (expected_path is not None and path != expected_path.resolve()):
        raise ValueError("Frozen artifact is outside its canonical source path")
    if not path.is_file() or common.sha(path) != digest:
        raise ValueError("Frozen artifact hash mismatch: " + path_value)
    return path


def validate_candidate_and_protocol(candidate, frozen, protocol, plans):
    """Pure JSON checks; no payload generation or inference is allowed here."""
    if candidate.get("accepted") is not True or candidate.get("eligible") is not True:
        raise ValueError("Candidate is not eligible and accepted")
    if candidate.get("candidate_kind") not in ("v5_physics", "v5_neural"):
        raise ValueError("Unknown exact frozen candidate version")
    if not isinstance(candidate.get("core_selection"), dict):
        raise ValueError("Candidate has no frozen physical selection")
    core = candidate["core_selection"]
    if set(core) != {"continuum_options", "linear_options", "device", "dtype"}:
        raise ValueError("Candidate physical selection has unregistered keys")
    if candidate.get("candidate_kind") == "v5_neural":
        checkpoint = candidate.get("checkpoint")
        if (not isinstance(checkpoint, dict) or checkpoint.get("updates") not in (200, 500, 1000)
                or not checkpoint.get("sha256") or not checkpoint.get("path")):
            raise ValueError("Neural candidate has no accepted registered checkpoint")
    elif candidate.get("checkpoint") is not None:
        raise ValueError("Physical-only candidate must not have a neural checkpoint")
    thresholds = candidate.get("thresholds")
    if (not isinstance(thresholds, dict) or not isinstance(thresholds.get("z_threshold"), (int, float))
            or not math.isfinite(thresholds["z_threshold"]) or thresholds["z_threshold"] <= 0):
        raise ValueError("Frozen candidate has no finite positive null threshold")
    from research_v5.statistics import statistics_plan
    if candidate.get("statistics") != statistics_plan():
        raise ValueError("Candidate statistics differ from the registered mother-scene plan")
    if (frozen.get("status") != FROZEN_STATUS
            or frozen.get("candidate_kind") != candidate["candidate_kind"]
            or frozen.get("core_selection") != core
            or frozen.get("thresholds") != thresholds
            or frozen.get("statistics") != candidate["statistics"]
            or frozen.get("plan_hashes") != protocol["family_plan_hashes"]
            or frozen.get("plan_hashes") != plans["plan_hashes"]):
        raise ValueError("Final protocol does not bind the exact candidate, thresholds, statistics and plans")
    expected = frozen_evaluation_fields(plans)
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("Final evaluation rules/counts/MC subset were not frozen exactly")
    if {key: len(plans["plans"][key]) for key in ("signed_confirmation", "final_once")} != {"signed_confirmation": 288, "final_once": 512}:
        raise ValueError("Reserved evaluation population differs from the original 800 noise plan")
    return True


def freeze_final_protocol(candidate_path=CANONICAL_CANDIDATE):
    """Validate/reuse candidate-review freeze; write only when still absent."""
    candidate_path = common.ed_path(candidate_path)
    if candidate_path != CANONICAL_CANDIDATE.resolve():
        raise ValueError("Only the canonical candidate-review receipt is accepted")
    protocol = common.require_registered()
    candidate = common.read_json(candidate_path)
    plans = common.read_json(common.CODE / "family_plans.json")
    if common.sha(common.CODE / "family_plans.json") != protocol["family_plans_sha256"]:
        raise ValueError("Frozen family-plan file hash differs")
    candidate_digest = common.sha(candidate_path)
    frozen_path = common.REPORTS / "final_protocol.json"
    if frozen_path.exists():
        frozen = common.read_json(frozen_path)
    else:
        frozen = {"schema": "v5-final-once-frozen-protocol-001", "status": FROZEN_STATUS,
            "frozen_at": common.now(), "protocol_sha256": common.sha(common.CODE / "protocol.json"),
            "candidate_receipt_path": common.relative(candidate_path),
            "candidate_receipt_sha256": candidate_digest,
            "candidate_kind": candidate.get("candidate_kind"),
            "core_selection": copy.deepcopy(candidate.get("core_selection")),
            "thresholds": copy.deepcopy(candidate.get("thresholds")),
            "statistics": copy.deepcopy(candidate.get("statistics")),
            "plan_hashes": copy.deepcopy(plans["plan_hashes"]),
            "checkpoint": copy.deepcopy(candidate.get("checkpoint")),
            "traditional_reference": candidate.get("traditional_reference"),
            "family_plans_sha256": protocol["family_plans_sha256"],
            **frozen_evaluation_fields(plans)}
        validate_candidate_and_protocol(candidate, frozen, protocol, plans)
        common.write_json(frozen_path, frozen)
    if (frozen.get("candidate_receipt_sha256") != candidate_digest
            or frozen.get("candidate_receipt_path") != common.relative(candidate_path)
            or frozen.get("protocol_sha256") != common.sha(common.CODE / "protocol.json")
            or frozen.get("family_plans_sha256") != protocol["family_plans_sha256"]
            or frozen.get("checkpoint") != candidate.get("checkpoint")
            or frozen.get("traditional_reference") != candidate.get("traditional_reference")):
        raise ValueError("Final protocol hash/source fields differ from candidate receipt")
    validate_candidate_and_protocol(candidate, frozen, protocol, plans)
    selection_path = _artifact(candidate["selection_path"], candidate["selection_sha256"],
                               common.REPORTS / "development" / "selection.json")
    selected = common.read_json(selection_path)
    if selected["core_selection"] != candidate["core_selection"] or selected["traditional_reference"] != candidate["traditional_reference"]:
        raise ValueError("Frozen candidate does not reproduce the development selection")
    thresholds = candidate["thresholds"]
    threshold_path = _artifact(thresholds["path"], thresholds["sha256"],
                               common.REPORTS / "development" / "null_threshold.json")
    threshold = common.read_json(threshold_path)
    if threshold["z_threshold"] != thresholds["z_threshold"] or threshold["selection_hash"] != candidate["selection_sha256"]:
        raise ValueError("Threshold receipt does not bind the selected physical estimator")
    gate_receipt = candidate["development_gate"]
    gate_path = _artifact(gate_receipt["path"], gate_receipt["sha256"],
                          common.REPORTS / "development" / "development_gate.json")
    gate = common.read_json(gate_path)
    if (gate.get("physics_eligible") is not True or gate.get("no_hardstop") is not True
            or gate.get("selection_hash") != candidate["selection_sha256"]
            or gate.get("threshold_hash") != thresholds["sha256"]):
        raise ValueError("Frozen candidate development gate is not eligible or bound")
    decision = common.read_json(common.REPORTS / "DECISION_1.json")
    if (decision.get("stage1_evidence_complete") is not True
            or decision.get("physical_stage2_eligible") is not True
            or decision.get("ml_stop_required") is not False):
        raise ValueError("Independent stage1 evidence forbids final unsealing")
    if candidate["candidate_kind"] == "v5_neural":
        checkpoint = candidate["checkpoint"]
        _artifact(checkpoint["path"], checkpoint["sha256"])
        training = common.read_json(common.REPORTS / "TRAIN_SUMMARY.json")
        selected_checkpoint = training.get("selected_checkpoint") or {}
        if (training.get("accepted") is not True or training.get("eligible") is not True
                or {key: selected_checkpoint.get(key) for key in ("path", "sha256", "updates")} != checkpoint):
            raise ValueError("Neural receipt does not reproduce the one accepted training checkpoint")
    return candidate, frozen, protocol, plans


@torch.no_grad()
def attach_proxy_uncertainty(output, sigma, z_threshold):
    """Frozen raw-only detection/abstention and explicitly approximate intervals."""
    result = dict(output)
    valid_sigma = torch.isfinite(sigma) & (sigma > 0)
    if sigma.shape != output["template_postblur_e"].shape or not bool(valid_sigma.all()):
        raise ValueError("Raw-derived uncertainty proxy is invalid")
    s = output["template_postblur_e"].detach()
    valid = output["valid"].bool()
    snr = s.abs() / sigma
    low = (~valid) | (snr <= z_threshold)
    result.update(sigma_S=sigma.detach(), detection_sigma_proxy_e=sigma.detach(),
        sigma_S_source="raw-derived uniform-W postblur proxy; not MC sigma", SNR=snr,
        undetectable=valid & (snr <= z_threshold),
        low_confidence=low, template_abstain=low, detection_limit_e=z_threshold*sigma,
        signed_ci_lower_e=s - 1.959963984540054*sigma,
        signed_ci_upper_e=s + 1.959963984540054*sigma,
        signed_interval_source="normal approximate raw-derived postblur uniform-W proxy; continuum/beta model error omitted",
        interval_empirical_coverage_claimed=False)
    result["info"] = copy.deepcopy(output.get("info", {}))
    result["info"]["frozen_detection"] = {"z_threshold": z_threshold,
        "calibrated_on": "independent development null calibration mothers only",
        "raw_only_rule": True, "threshold_recalibrated": False,
        "below_limit_template_abstained": True,
        "color_temperature_validity_unchanged": True,
        "proxy_is_MC_sigma": False}
    return result


def _detection_metrics(output, sigma, threshold):
    value = output["template_postblur_e"].detach()
    standardized = value.abs() / sigma
    valid = output["valid"].bool()
    finite = bool(torch.isfinite(standardized).all())
    maximum = float(standardized.max()) if finite else None
    return {"frame_detection_score": maximum,
        "frame_detection_score_kind": "max_abs_template_over_raw_proxy_full_FOV",
        "frame_threshold": threshold, "frame_detected": bool(maximum > threshold) if maximum is not None else None,
        "z_threshold": threshold,
        "proxy_detection_valid_fraction": float(valid.double().mean()),
        "low_confidence_fraction": float((~valid | (standardized <= threshold)).double().mean()),
        "undetectable_fraction": float((valid & (standardized <= threshold)).double().mean()),
        "raw_only_detectability": bool(maximum > threshold) if maximum is not None else False,
        "detection_threshold_recalibrated": False}


def _diagnostics(output, seconds):
    info = output.get("info", {})
    linear = info.get("linear", {})
    frames = linear.get("frames", [])
    frame = frames[0] if frames else {}
    return {"seconds": seconds, "CG": copy.deepcopy(frame.get("CG")),
        "gradient_energy_S_e2": frame.get("objective_terms_after", {}).get("gradient_energy_S_e2"),
        "all_CG_converged": linear.get("all_CG_converged"),
        "continuum_frozen": info.get("continuum_frozen", False),
        "expected_e_source": output.get("expected_e_source", "physical measurement prediction"),
        "expected_e_is_modified_template_raw_refit": False if "denoiser" in info else None,
        "truth_or_identity_access": False,
        "info": info}


def _record(row, method, output, truth, physics, sigma, threshold, seconds, raw_digest, mc=None):
    from research_v5.metrics import score_prediction
    scored = score_prediction(output, truth, physics, sigma_S=sigma, z_threshold=threshold)
    metrics = scored.get("metrics")
    if metrics is not None:
        metrics.update(_detection_metrics(output, sigma, threshold))
        if float(row["coefficient_e"]) == 0:
            metrics["null_frame_FP"] = metrics["frame_detected"]
        metrics["raw_fit_source"] = output.get("expected_e_source", "physical measurement prediction")
        metrics["modified_template_raw_refit_assessed"] = False if method == "v5_neural" else None
    diagnostics = _diagnostics(output, seconds)
    numerical = scored["numerical_execution_passed"]
    if method in ("v5_physics", "v5_neural"):
        numerical = numerical and diagnostics["all_CG_converged"] is True
    return {**_metadata(row), "method": method, "raw_sha256": raw_digest,
        **scored, "metrics": metrics, "numerical_execution_passed": bool(numerical),
        "diagnostics": diagnostics, "MC": None if mc is None else mc["summary"],
        "calibration_scope": "nominal externally fixed synthetic calibration",
        "real_temperature_accuracy_claimed": False}


def _failure_record(row, method, error, raw_digest=None):
    return {**_metadata(row), "method": method, "raw_sha256": raw_digest,
        "status": "EXECUTION_FAILED", "numerical_execution_passed": False, "metrics": None,
        "error_type": type(error).__name__, "error": str(error),
        "diagnostics": {"truth_or_identity_access": False}}


def _save_prediction(path, arrays):
    """Preserve abstention NaNs and verify them on reopen, rather than drop them."""
    path = common.ed_path(path)
    if not path.is_relative_to(common.DATA.parent.resolve()) or path.exists():
        raise ValueError("Final prediction must be a new v5 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as reopened:
        if set(reopened.files) != set(arrays) or not all(np.array_equal(reopened[k], v, equal_nan=True)
                                                       for k, v in arrays.items()):
            raise RuntimeError("Final prediction reopen differs, including preserved NaN abstentions")
    return common.sha(path)


def summarize_final_records(records, candidate_kind, traditional_reference, mc_case_ids):
    """Frozen comparisons by partition/sign and whole-mother bootstrap."""
    from research_v5.statistics import (aggregate_metrics, filter_records, mother_metric,
                                         mother_bootstrap, paired_statistics)
    selected = [r for r in records if r["method"] == candidate_kind]
    methods = sorted({r["method"] for r in records})
    method_summaries = {method: {"pooled": aggregate_metrics(filter_records(records, method=method)),
        "by_partition": {partition: aggregate_metrics(filter_records(records, method=method, partition_role=partition))
                         for partition in ("signed_confirmation", "final_once")}}
                        for method in methods}
    aggregates, comparisons, mother_intervals = {}, {}, {}
    gates = {}
    for partition in ("signed_confirmation", "final_once"):
        population = filter_records(selected, partition_role=partition)
        aggregates[partition] = aggregate_metrics(population)
        for group in sorted({r["group"] for r in population}):
            for sign in sorted({r["sign"] for r in population if r["group"] == group}):
                subset = filter_records(population, group=group, sign=sign)
                key = f"{partition}/{group}/{sign}"
                aggregates[key] = aggregate_metrics(subset)
                mother_intervals[key] = {"template_MSE_e2": mother_bootstrap(mother_metric(subset)),
                    "temperature_MSE_K2": mother_bootstrap(mother_metric(subset, field="temperature"))}
                if group == "normal":
                    pair_records = filter_records(records, partition_role=partition)
                    paired = paired_statistics(pair_records, candidate_kind, traditional_reference,
                        group=group, input_kind="noise", sign=sign)
                    comparisons[key] = paired
                    ratio = paired["paired_parent_RMSE_ratio"]
                    candidate_projection = paired["candidate_pooled"].get("signed_projection")
                    reference_projection = paired["reference_pooled"].get("signed_projection")
                    gates[key + "/RMSE_noninferiority"] = bool(paired["complete"] and ratio is not None and ratio <= 1.05)
                    gates[key + "/amplitude_projection"] = bool(candidate_projection is not None and
                        reference_projection is not None and abs(candidate_projection-1) <= 1.05*abs(reference_projection-1))
    total = aggregate_metrics(selected)
    gates["all_800_candidate_predictions_numerically_passed"] = (len(selected) == 800 and
        all(r["numerical_execution_passed"] for r in selected))
    gates["nominal_color_RMSE"] = bool(total.get("temperature_RMSE_K") is not None and total["temperature_RMSE_K"] <= 20)
    gates["nominal_valid_coverage"] = bool(total.get("valid_fraction") is not None and total["valid_fraction"] >= .99)
    nulls = filter_records(selected, group="blank", partition_role="final_once")
    known_nulls = all(r.get("metrics") is not None and r["metrics"].get("null_frame_FP") is not None for r in nulls)
    null_fp = float(np.mean([r["metrics"]["null_frame_FP"] for r in nulls])) if nulls and known_nulls else None
    gates["original_final_null_frame_FP"] = bool(len(nulls) == 128 and null_fp is not None and null_fp <= .05)
    weak = filter_records(selected, group="weak")
    detectable = [r for r in weak if r.get("metrics", {}) and r["metrics"].get("raw_only_detectability") is True]
    undetectable = [r for r in weak if r.get("metrics", {}) and r["metrics"].get("raw_only_detectability") is False]
    weak_report = {}
    for label, population in (("above_frozen_limit", detectable), ("below_frozen_limit", undetectable)):
        weak_report[label] = {"cases": len(population), "by_sign": {},
            "mandatory_L2_below_one": False if label == "below_frozen_limit" else None,
            "rule": "frozen raw-only frame detection score; external sign used only for summary"}
        for sign in ("positive", "negative"):
            subset = filter_records(population, sign=sign)
            weak_report[label]["by_sign"][sign] = aggregate_metrics(subset)
    mc = [r for r in selected if r["case_id"] in set(mc_case_ids)]
    mc_mothers = {}
    for r in mc:
        interval = (r.get("metrics") or {}).get("signed_interval", {})
        coverage = interval.get("coverage_full_truth_domain") if (r.get("MC") or {}).get("status") == "completed" else None
        mc_mothers.setdefault(r["mother_id"], []).append(coverage)
    coverage_by_mother = {mid: float(np.mean(values)) if all(v is not None for v in values) else None
                           for mid, values in mc_mothers.items()}
    mc_coverage = mother_bootstrap(coverage_by_mother)
    gates["fixed_final_MC_interval_coverage"] = bool(len(mc) == 12 and mc_coverage["complete"]
        and mc_coverage["estimate"] >= .90)
    failures = [{"case_id": r["case_id"], "mother_id": r["mother_id"], "method": r["method"],
                  "status": r["status"], "error": r.get("error"), "metrics": r.get("metrics"),
                  "diagnostics": r.get("diagnostics")} for r in records if not r["numerical_execution_passed"]]
    return {"candidate_pooled": total, "aggregates_by_partition_group_sign": aggregates,
        "all_frozen_method_summaries": method_summaries,
        "numerical_execution_scope": "all 800 frozen candidate frames; every baseline failure is separately retained",
        "mother_bootstrap_CI95": mother_intervals, "paired_normal_comparisons": comparisons,
        "weak_detectability": weak_report,
        "final_null": {"frames": len(nulls), "frame_FP": null_fp, "same_frozen_threshold": True},
        "fixed_final_MC": {"cases": len(mc), "case_ids": mc_case_ids,
            "signed_interval_full_domain_mother_coverage_CI95": mc_coverage},
        "gates": gates, "numerical_execution_passed": gates["all_800_candidate_predictions_numerically_passed"],
        "scientific_acceptance_passed": all(gates.values()), "failures": failures}


def run_final_once(device="cuda"):
    summary_path = common.REPORTS / "FINAL_ONCE_SUMMARY.json"
    started_path = common.REPORTS / "FINAL_ONCE_STARTED.json"
    receipt_path = common.REPORTS / "FINAL_ONCE_RECEIPT.json"
    if started_path.exists() or receipt_path.exists() or summary_path.exists():
        raise FileExistsError("Final confirmation/test phase was already attempted; never rerun or retune")
    summary = {"schema": "v5-one-time-independent-evaluation-001", "status": "NOT_ELIGIBLE",
        "numerical_execution_passed": False, "scientific_acceptance_passed": False,
        "case_counts": copy.deepcopy(CASE_COUNTS), "evaluated_candidate_cases": 0,
        "failures": [], "final_arrays_generated": False, "reserved_payloads_opened": 0,
        "threshold_recalibrated": False, "post_final_adaptation": False,
        "real_gas_temperature_or_10kHz_claimed": False}
    # Ineligible candidates never generate or load any reserved family array.
    if not CANONICAL_CANDIDATE.exists():
        summary["reason"] = "canonical_candidate_receipt_missing"
        common.write_json(summary_path, summary)
        return summary
    candidate = common.read_json(CANONICAL_CANDIDATE)
    if candidate.get("accepted") is not True or candidate.get("eligible") is not True:
        summary["reason"] = "candidate_not_eligible_and_accepted"
        common.write_json(summary_path, summary)
        return summary
    candidate, frozen, protocol, plans = freeze_final_protocol()
    if str(device) != str(candidate["core_selection"]["device"]):
        raise ValueError("Final execution device differs from the frozen physical candidate")
    frozen_path = common.REPORTS / "final_protocol.json"
    frozen_digest = common.sha(frozen_path)
    start_hash = exclusive_json(started_path, {"schema": "v5-final-once-start-001", "started_at": common.now(),
        "final_protocol_sha256": frozen_digest, "candidate_receipt_sha256": common.sha(CANONICAL_CANDIDATE),
        "case_counts": copy.deepcopy(CASE_COUNTS), "retuning_permitted": False,
        "independent_evaluation_attempt": 1})
    records, mc_records = [], []
    summary.update(status="FINAL_ONCE_STARTED", final_protocol_sha256=frozen_digest,
        started_receipt_sha256=start_hash, candidate_kind=candidate["candidate_kind"],
        candidate_receipt_sha256=common.sha(CANONICAL_CANDIDATE))
    budget = common.Budget(protocol["phase_budgets_s"]["final-once"])
    try:
        from research_v5.family_data import generate_family
        from research_v5.baselines import BaselineSet, BASELINE_NAMES
        from research_v5.reconstruct import reconstruct
        from research_v5.development import raw_noise_proxy
        from research_v5.signed_gain_denoiser import load_denoiser, apply_denoiser
        from research_v5.uncertainty import monte_carlo_uncertainty
        core = copy.deepcopy(candidate["core_selection"])
        core["device"] = str(device)
        physics = common.make_physics(device, getattr(torch, core["dtype"]))
        if candidate["traditional_reference"] not in ("B1_tikhonov", "v3_physics"):
            raise ValueError("Final traditional comparator must be the single development-selected physical reference")
        model = None
        if candidate["candidate_kind"] == "v5_neural":
            checkpoint = candidate["checkpoint"]
            model, payload = load_denoiser(common.ROOT / checkpoint["path"], device, expected_sha256=checkpoint["sha256"])
            if payload["actual_optimizer_updates"] != checkpoint["updates"]:
                raise ValueError("Neural checkpoint actual optimizer count differs from frozen receipt")

        @torch.no_grad()
        def callback(raw_dn, calibration, selection):
            physical = reconstruct(raw_dn, calibration, selection)
            if model is None:
                return physical
            sigma = raw_noise_proxy(physics, raw_dn, selection)["sigma_S"]
            return apply_denoiser(model, raw_dn, physics, physical, sigma)

        baselines = BaselineSet(device, cfg=protocol["physics_config"])
        record_path = common.REPORTS / "final_once" / "records.jsonl"
        failed_path = common.REPORTS / "final_once" / "case_failures.jsonl"
        mc_ids = set(frozen["final_MC_case_ids"])
        threshold = float(candidate["thresholds"]["z_threshold"])
        for family in ("signed_confirmation", "final_once"):
            budget.check()
            plan = plans["plans"][family]
            folder = common.DATA / family
            generate_family(physics, plan, folder, options={"fixture_only": False, "authorization": {
                "phase": "final-once", "partition_role": family,
                "plan_sha256": common.canonical_hash(plan), "protocol_sha256": common.sha(common.CODE / "protocol.json"),
                "final_protocol_path": str(frozen_path.resolve()), "final_protocol_sha256": frozen_digest,
                "candidate_receipt_sha256": common.sha(CANONICAL_CANDIDATE)}})
            summary["final_arrays_generated"] = True
            manifest = _checked_manifest(folder, plan, protocol)
            for row in manifest["rows"]:
                budget.check()
                methods = [candidate["candidate_kind"], *BASELINE_NAMES]
                raw_digest = row.get("raw_file_sha256")
                try:
                    raw, truth = _checked_case(folder, row)
                    summary["reserved_payloads_opened"] += 2
                    raw = torch.as_tensor(raw, device=device, dtype=physics.M.dtype)
                    if raw.ndim == 2:
                        raw = raw[None]
                    sigma = raw_noise_proxy(physics, raw, core)["sigma_S"]
                except Exception as error:
                    for method in methods:
                        record = _failure_record(row, method, error, raw_digest)
                        records.append(record)
                        common.append_jsonl(record_path, record)
                        common.append_jsonl(failed_path, record)
                    continue
                outputs, timings = {}, {}
                mc = None
                try:
                    common.synchronize(device)
                    tic = time.perf_counter()
                    selected = callback(raw, physics.cfg, core)
                    selected["raw_e"] = common.raw_electrons(physics, raw)
                    selected = attach_proxy_uncertainty(selected, sigma, threshold)
                    common.synchronize(device)
                    timings[candidate["candidate_kind"]] = time.perf_counter()-tic
                    if row["case_id"] in mc_ids:
                        budget.check()
                        mc = monte_carlo_uncertainty(physics, raw, callback, physics.cfg, core,
                            options={"repeats": 8, "seed": 53125, "confidence_level": .95}, baseline=selected)
                        if mc["status"] == "completed":
                            for key in ("signed_ci_lower_e", "signed_ci_upper_e", "temperature_ci_lower_K",
                                        "temperature_ci_upper_K", "temperature_mc_sigma_K", "interval_valid"):
                                selected[key] = mc[key]
                            selected["template_mc_sigma_e"] = mc["sigma_S"]
                            selected["signed_interval_source"] = "fixed 8-repeat original-raw linear-fit plug-in MC; no threshold recalibration"
                        mc_records.append({**_metadata(row), "method": candidate["candidate_kind"], "summary": mc["summary"]})
                    outputs[candidate["candidate_kind"]] = selected
                except Exception as error:
                    record = _failure_record(row, candidate["candidate_kind"], error, raw_digest)
                    records.append(record)
                    common.append_jsonl(record_path, record)
                    common.append_jsonl(failed_path, record)
                try:
                    for method, value in baselines.evaluate(raw).items():
                        outputs[method] = value
                        info = value.get("info", {})
                        timings[method] = float(info.get("seconds", 0)) + float(info.get("shared_inverse_seconds", 0))
                except Exception as error:
                    for method in BASELINE_NAMES:
                        record = _failure_record(row, method, error, raw_digest)
                        records.append(record)
                        common.append_jsonl(record_path, record)
                        common.append_jsonl(failed_path, record)
                for method, output in outputs.items():
                    budget.check()
                    try:
                        record = _record(row, method, output, truth, physics, sigma, threshold,
                            timings[method], raw_digest, mc if method == candidate["candidate_kind"] else None)
                        arrays = {key: value.detach().cpu().numpy() for key, value in output.items() if isinstance(value, torch.Tensor)}
                        output_path = common.DATA / "final_predictions" / method / (row["case_id"] + ".npz")
                        record["prediction_output"] = {"path": common.relative(output_path),
                                                       "sha256": _save_prediction(output_path, arrays)}
                    except Exception as error:
                        record = _failure_record(row, method, error, raw_digest)
                    records.append(record)
                    common.append_jsonl(record_path, record)
                    if not record["numerical_execution_passed"]:
                        common.append_jsonl(failed_path, record)
                summary["evaluated_candidate_cases"] = sum(r["method"] == candidate["candidate_kind"] for r in records)
        for record in mc_records:
            common.append_jsonl(common.REPORTS / "final_once" / "fixed_MC_records.jsonl", record)
        statistics = summarize_final_records(records, candidate["candidate_kind"], candidate["traditional_reference"], frozen["final_MC_case_ids"])
        summary.update(statistics)
        stress_path = common.REPORTS / "development" / "calibration_stress_records.jsonl"
        summary.update(status="FINAL_SCIENTIFIC_ACCEPTANCE_PASSED" if summary["scientific_acceptance_passed"] else "FINAL_FAILED_NO_RETUNING",
            finished_at=common.now(), budget_elapsed_s=budget.elapsed(), records_sha256=common.sha(record_path),
            record_count=len(records), evaluated_candidate_cases=sum(r["method"] == candidate["candidate_kind"] for r in records),
            failure_records_path=common.relative(failed_path) if failed_path.exists() else None,
            calibration_stress_evidence={"path": common.relative(stress_path), "sha256": common.sha(stress_path)} if stress_path.exists() else None,
            calibration_stress_scope="separate synthetic mismatch screen; no real-temperature claim")
        common.write_json(summary_path, summary)
        exclusive_json(receipt_path, {"status": "FINAL_ONCE_COMPLETED_NO_RETUNING", "completed_at": common.now(),
            "final_protocol_sha256": frozen_digest, "candidate_receipt_sha256": common.sha(CANONICAL_CANDIDATE),
            "summary_path": common.relative(summary_path), "summary_sha256": common.sha(summary_path),
            "records_sha256": common.sha(record_path), "case_counts": copy.deepcopy(CASE_COUNTS),
            "evaluated_candidate_cases": summary["evaluated_candidate_cases"],
            "numerical_execution_passed": summary["numerical_execution_passed"],
            "scientific_acceptance_passed": summary["scientific_acceptance_passed"],
            "threshold_recalibrated": False, "post_final_adaptation": False})
        return summary
    except Exception as error:
        summary["evaluated_candidate_cases"] = sum(r["method"] == candidate["candidate_kind"] for r in records)
        summary.update(status="FINAL_EXECUTION_FAILED_NO_RETRY_OR_RETUNING", finished_at=common.now(),
                       budget_elapsed_s=budget.elapsed(), numerical_execution_passed=False,
                       scientific_acceptance_passed=False)
        summary["failures"].append({"error_type": type(error).__name__, "error": str(error),
                                    "evaluated_candidate_cases": summary["evaluated_candidate_cases"]})
        if not summary_path.exists():
            common.write_json(summary_path, summary)
        # STARTED remains forever even if no completion receipt was possible.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    summary = run_final_once(args.device)
    print(json.dumps({key: summary.get(key) for key in ("status", "numerical_execution_passed",
        "scientific_acceptance_passed", "evaluated_candidate_cases")}))


if __name__ == "__main__":
    main()
