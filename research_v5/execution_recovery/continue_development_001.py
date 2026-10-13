"""Complete only unexecuted fixed diagnostics after the original wall timeout.

This separately registered execution harness does not modify or reselect the
frozen estimator, protocol, cohorts, seeds, threshold, or scientific criteria.
The original FAILED receipt remains authoritative for its original invocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
from statistics import NormalDist
import sys
import traceback

from research_v5 import common

common.configure_runtime()

import torch

from research_v5 import development as dev
from research_v5 import reports
from research_v5.statistics import aggregate_metrics, filter_records, mother_bootstrap, paired_statistics, statistics_plan
from research_v5.uncertainty import fixed_mc_seed_plan, MC_DEFAULTS


REC = common.REPORTS / "execution_recovery_001"
DIRECTORY = common.REPORTS / "development"
REGISTRATION = REC / "registration.json"
ORIGINAL_RECEIPT = common.REPORTS / "phase_receipts/development-compare_2026-10-13T085923367640_0000.json"
SUPPLEMENT_SECONDS = 5400.0
SOURCE = Path(__file__).resolve()


def bound(path):
    return {"path": common.relative(path), "sha256": common.sha(path)}


def verify_binding(receipt):
    path = (common.ROOT / receipt["path"]).resolve()
    assert path.is_relative_to(common.ROOT.resolve())
    assert path.is_file() and common.sha(path) == receipt["sha256"], receipt["path"]
    return path


def no_sealed_open():
    assert not (common.REPORTS / "FINAL_ONCE_STARTED.json").exists()
    assert not (common.REPORTS / "final_protocol.json").exists()
    assert all(not (common.DATA / name).exists() for name in ("signed_confirmation", "final_once", "training"))


def frozen_inputs():
    protocol = common.require_registered()
    assert common.source_hashes() == protocol["source_sha256"]
    records = common.jsonl(DIRECTORY / "records.jsonl")
    scored = common.jsonl(DIRECTORY / "selected_records.jsonl")
    selection = common.read_json(DIRECTORY / "selection.json")
    null = common.read_json(DIRECTORY / "null_threshold.json")
    plans = common.read_json(common.CODE / "family_plans.json")
    folder = common.DATA / "development"
    manifest = common.read_json(folder / "data_manifest.json")
    assert common.sha(common.CODE / "family_plans.json") == protocol["family_plans_sha256"]
    assert common.sha(folder / "data_manifest.json") == selection["development_manifest"]["sha256"]
    assert null["selection_hash"] == common.sha(DIRECTORY / "selection.json")
    assert selection["protocol_sha256"] == null["protocol_sha256"] == common.sha(common.CODE / "protocol.json")
    assert selection["status"] == "FROZEN_BEFORE_NULL_THRESHOLD_CALIBRATION"
    assert null["threshold_frozen_before_evaluation"] is True and null["retuned_on_evaluation"] is False
    calibration = common.read_json(verify_binding(null["calibration_threshold_freeze"]))
    assert calibration["z_threshold"] == null["z_threshold"]
    assert calibration["selection_hash"] == common.sha(DIRECTORY / "selection.json")
    assert selection["statistics"] == statistics_plan()
    assert len(manifest["rows"]) == len(plans["plans"]["development"]) == 540
    case_ids = {r["case_id"] for r in manifest["rows"]}
    assert len(case_ids) == 540
    expected = {(cid, "v5_physics", float(alpha)) for cid in case_ids for alpha in protocol["alpha_g_development_grid"]}
    expected.update((cid, method, None) for cid in case_ids for method in protocol["baselines"])
    actual = [(r["case_id"], r["method"], r.get("alpha_g")) for r in records]
    assert len(actual) == 4320 and len(set(actual)) == 4320 and set(actual) == expected
    assert len(scored) == 540 and {r["case_id"] for r in scored} == case_ids
    selected = filter_records(records, method="v5_physics", alpha_g=selection["selected_alpha_g"])
    assert len(selected) == 540
    for name in ("null_calibration", "null_evaluation"):
        rows = common.jsonl(DIRECTORY / (name + "_records.jsonl"))
        assert len(rows) == len({r["case_id"] for r in rows}) == len(plans["plans"][name]) == 96
        assert {r["case_id"] for r in rows} == {r["case_id"] for r in plans["plans"][name]}
    assert selection["MC_case_ids"] == dev._mc_cases(plans["plans"]["development"])
    assert len(selection["MC_case_ids"]) == 12
    no_sealed_open()
    return protocol, records, scored, selection, null, plans, folder, manifest, selected


def recovered_mc_case(physics, folder, row, record, protocol, selection, array_receipt):
    """Score existing intervals; never call reconstruction or draw any sample."""
    path = verify_binding(array_receipt)
    values = common.arrays(path)
    required = {"sigma_S", "temperature_mc_sigma_K", "signed_ci_lower_e", "signed_ci_upper_e",
        "temperature_ci_lower_K", "temperature_ci_upper_K", "mc_template_mean_e", "mc_temperature_mean_K",
        "mc_valid_fraction", "valid", "interval_valid", "SNR", "undetectable", "low_confidence"}
    assert set(values) == required
    fields = {k: torch.as_tensor(v, device=physics.M.device) for k, v in values.items()}
    assert all(tuple(v.shape) == (1, *physics.cfg["shape"]) for v in fields.values())
    assert bool((fields["mc_valid_fraction"] == 1).all())
    assert bool(fields["interval_valid"].all())
    assert bool(torch.isfinite(fields["sigma_S"]).all() and torch.isfinite(fields["temperature_mc_sigma_K"]).all())
    raw_dn, truth = dev._inputs(folder, row, physics)
    baseline = dev.load_cached_prediction(record, physics)
    z = NormalDist().inv_cdf((1 + protocol["MC"]["confidence_level"]) / 2)
    for target, sigma, lower, upper in (
        ("template_postblur_e", "sigma_S", "signed_ci_lower_e", "signed_ci_upper_e"),
        ("temperature_color_K", "temperature_mc_sigma_K", "temperature_ci_lower_K", "temperature_ci_upper_K")):
        center = baseline[target].double()
        assert torch.allclose(fields[lower], center-z*fields[sigma], rtol=1e-12, atol=1e-8, equal_nan=True)
        assert torch.allclose(fields[upper], center+z*fields[sigma], rtol=1e-12, atol=1e-8, equal_nan=True)
    merged = {**baseline, **fields, "raw_e": common.raw_electrons(physics, raw_dn)}
    metrics = dev.score_prediction(merged, truth, physics)["metrics"]
    domain = torch.as_tensor(truth["evaluation_domain"], device=physics.M.device).bool()
    if domain.ndim == 2:
        domain = domain[None]
    options = {**MC_DEFAULTS, "repeats": protocol["MC"]["repeats"], "seed": protocol["MC"]["seed"],
               "confidence_level": protocol["MC"]["confidence_level"]}
    return dev._plain({"case_id": row["case_id"], "mother_id": row["mother_id"], "sign": record["sign"],
        "geometry": row["geometry"], "template_geometry": row["template_geometry"], "status": "completed",
        "raw_sha256": row["raw_file_sha256"], "original_same_raw_cache": record["prediction_cache"],
        "signed_interval": metrics["signed_interval"], "temperature_interval": metrics["temperature_interval"],
        "uncertainty_arrays": array_receipt,
        "MC": {"schema": "v5-mc-persisted-array-recovery-v1", "status": "completed",
            "options": options, "seed_plan_from_frozen_code": fixed_mc_seed_plan(options["seed"], options["repeats"]),
            "per_repeat_records": None, "per_repeat_records_status": "not persisted before original timeout",
            "completed_repeats_inferred_from_valid_fraction": 8,
            "inference_basis": "Every saved mc_valid_fraction equals1; frozen loop has8 samples, and save occurs only after status=completed.",
            "per_repeat_CG_timing_clipping_details": "unavailable; not reconstructed or invented",
            "reconstructions_executed_during_recovery": 0,
            "interval_centers_checked_against_hash_bound_original_prediction": True,
            "scoring_device": str(physics.M.device), "original_execution_device": selection["core_selection"]["device"]},
        "bias_variance": {"baseline_template_bias_e": metrics.get("template_bias_e"),
            "baseline_temperature_bias_K": metrics.get("temperature_bias_K"),
            "MC_empirical_postblur_template_variance_mean_e2": float(fields["sigma_S"][domain].square().mean()),
            "MC_empirical_color_temperature_variance_mean_K2": float(fields["temperature_mc_sigma_K"][domain].square().mean()),
            "variance_scope": "finite raw-plugin noise repeats; calibration/model discrepancy omitted"}})


def scientific_stop(records, selected, selection):
    noisy = aggregate_metrics(filter_records(selected, input_kind="noise"))
    comparisons = {}
    for sign in ("positive", "negative"):
        candidate = aggregate_metrics(filter_records(selected, group="normal", input_kind="noise", sign=sign))
        reference = aggregate_metrics(filter_records(records, method=selection["traditional_reference"], group="normal", input_kind="noise", sign=sign))
        ratio = candidate["template_RMSE_e"] / reference["template_RMSE_e"]
        projection_pass = abs(candidate["signed_projection"]-1) <= 1.05*abs(reference["signed_projection"]-1)
        comparisons[sign] = {"physics": candidate, "traditional": reference,
            "RMSE_ratio": ratio, "RMSE_pass": ratio <= 1.05, "amplitude_projection_pass": projection_pass}
    stop = noisy["temperature_RMSE_K"] > 20 or not selection["selection_feasible"] or any(
        not row["RMSE_pass"] or not row["amplitude_projection_pass"] for row in comparisons.values())
    assert stop, "This diagnostic recovery is registered for an already failed physical branch."
    return {"schema": "v5-root-development-scientific-stop-v1", "created_at": common.now(),
        "scientific_stop_established_from_complete_development_grid": True,
        "execution_timeout_is_not_scientific_evidence": True,
        "selection": bound(DIRECTORY / "selection.json"), "records": bound(DIRECTORY / "records.jsonl"),
        "task_spec": bound(common.CODE / "TASK_SPEC.md"), "protocol": bound(common.CODE / "protocol.json"),
        "diagnostic_alpha_g": selection["selected_alpha_g"], "selection_feasible": selection["selection_feasible"],
        "nominal_noisy_pooled": noisy, "signed_normal": comparisons,
        "projection_rule": "abs(projection-1) <= 1.05*abs(frozen traditional projection-1), separately for both signs",
        "projection_scope": "Task4B compliance review with the already frozen final rule; no selection/threshold change",
        "ML_authorized": False, "final_evaluation_authorized": False, "sealed_arrays_opened": 0,
        "CNN_tested": False, "final_accuracy": "unknown; sealed studies remain unopened",
        "remaining_required_diagnostics": ["complete12case_MC_coverage", "Hessian8probes", "calibration_stress36cases", "final_evidence_review_and_delivery"]}


def prepare():
    assert not REGISTRATION.exists(), "Supplement registration is immutable."
    assert not (common.REPORTS / "PHASE_ACTIVE.json").exists()
    protocol, records, scored, selection, null, plans, folder, manifest, selected = frozen_inputs()
    original = common.read_json(ORIGINAL_RECEIPT)
    assert original["status"] == "FAILED" and original["error_type"] == "TimeoutError"
    assert original["error"] == "Registered stage wall-clock budget exceeded"
    assert protocol["phase_budgets_s"]["development-compare"] == 21600.0
    for name in ("mc_coverage.json", "hessian_diagnostics.json", "calibration_stress_records.jsonl", "development_gate.json", "PHYSICS_ONLY_V5_REPORT.md"):
        assert not (DIRECTORY / name).exists(), name
    assert not (common.DATA / "calibration_stress").exists()
    existing_files = sorted((common.DATA / "mc_uncertainty_predictions").glob("*.npz"))
    existing = {path.stem: bound(path) for path in existing_files}
    mc_ids = selection["MC_case_ids"]
    assert set(existing) == set(mc_ids[:3]), "Unexpected MC persistence boundary; manual evidence audit required."
    inputs = [common.CODE / "protocol.json", common.CODE / "family_plans.json", ORIGINAL_RECEIPT,
        common.REPORTS / "DECISION_1.json", common.REPORTS / "LINEAR_SOLVER_CHECKS.json",
        DIRECTORY / "DEVELOPMENT_STARTED.json", DIRECTORY / "records.jsonl", DIRECTORY / "selected_records.jsonl",
        DIRECTORY / "selection.json", DIRECTORY / "null_calibration_records.jsonl",
        DIRECTORY / "null_calibration_threshold.json", DIRECTORY / "null_evaluation_records.jsonl", DIRECTORY / "null_threshold.json",
        folder / "data_manifest.json", common.DATA / "null_calibration/data_manifest.json", common.DATA / "null_evaluation/data_manifest.json"]
    physics = common.make_physics("cpu", torch.float32, config=protocol["physics_config"])
    by_case = {r["case_id"]: r for r in selected}
    rows = {r["case_id"]: r for r in manifest["rows"]}
    recovered = []
    for cid in mc_ids[:3]:
        case = recovered_mc_case(physics, folder, rows[cid], by_case[cid], protocol, selection, existing[cid])
        output = REC / "mc_cases" / (cid + ".json")
        common.write_json(output, case)
        recovered.append(bound(output))
    comparisons = [paired_statistics(records, "v5_physics", selection["traditional_reference"], group=group,
        sign=sign, alpha_g=selection["selected_alpha_g"]) for group in ("normal", "weak") for sign in (None, "positive", "negative")]
    stats = {"schema": "v5-development-paired-statistics-001", "plan": statistics_plan(),
        "selection_hash": common.sha(DIRECTORY / "selection.json"), "comparisons": comparisons, "frozen_before_final": True}
    common.write_json(DIRECTORY / "paired_statistics.json", dev._plain(stats))
    inputs.append(DIRECTORY / "paired_statistics.json")
    stop_path = REC / "scientific_stop.json"
    common.write_json(stop_path, scientific_stop(records, selected, selection))
    registration = {"schema": "v5-post-timeout-diagnostic-completion-registration-v1", "registered_at": common.now(),
        "harness": bound(SOURCE), "source_sha256": common.source_hashes(), "original_failure_receipt": bound(ORIGINAL_RECEIPT),
        "original_registered_budget_s": 21600.0, "original_actual_wall_s": original["wall_s"],
        "supplement_wall_budget_s": SUPPLEMENT_SECONDS,
        "execution_budget_deviation": "Additional bounded diagnostic completion after original21600s budget exhaustion; not within original budget and not preregistered at stage0.",
        "scientific_parameters_or_protocol_modified": False, "original_failed_receipt_preserved": True,
        "bound_inputs": [bound(path) for path in inputs], "existing_MC_arrays": existing,
        "recovered_MC_case_reports": recovered, "predeclared_MC_case_ids": mc_ids,
        "remaining_MC_case_ids": mc_ids[3:], "new_MC_reconstructions_planned": 72,
        "existing_MC_reconstructions_repeated": 0, "new_Hessian_probes": 8,
        "calibration_stress_inputs": 36, "calibration_stress_methods_per_input": 5,
        "no_grid_or_null_or_oracle_reconstruction": True,
        "grid_counts": dict(Counter(r["method"] for r in records)), "grid_statuses": dict(Counter(r["status"] for r in records)),
        "scientific_stop": bound(stop_path), "training_authorized": False, "final_authorized": False,
        "root_review_still_required_after_completion": True}
    common.write_json(REGISTRATION, registration)
    print(json.dumps({"status": "RECOVERY_REGISTERED", "registration": str(REGISTRATION),
        "existing_MC_reopened_without_reconstruction": 3, "remaining_MC_cases": 9,
        "scientific_ML_stop": True, "formal_GPU_started": False}), flush=True)


def aggregate_mc(cases, selection):
    assert [r["case_id"] for r in cases] == selection["MC_case_ids"]
    parents = {}
    for row in cases:
        parents.setdefault(row["mother_id"], []).append(row)
    assert len(parents) == 6 and all(len(rows) == 2 for rows in parents.values())
    mother_s = {mid: sum(r["signed_interval"].get("coverage_full_truth_domain", 0.) or 0. for r in rows)/len(rows) for mid, rows in parents.items()}
    mother_t = {mid: sum(r["temperature_interval"].get("coverage_full_truth_domain", 0.) or 0. for r in rows)/len(rows) for mid, rows in parents.items()}
    s, t = sum(mother_s.values())/len(mother_s), sum(mother_t.values())/len(mother_t)
    return {"schema": "v5-external-nominal-mc-coverage-001", "complete": len(cases) == 12 and all(r["status"] == "completed" for r in cases),
        "cases": cases, "predeclared_case_ids": selection["MC_case_ids"], "selection_sha256": common.sha(DIRECTORY / "selection.json"),
        "nominal_only": True, "truth_entered_reconstruction": False,
        "coverage_rule": "mean of full truth-domain pixel interval coverage within each mother, then equal mother mean; failed MC counts as zero coverage",
        "independent_parent_count": len(parents), "pixels_assumed_independent": False,
        "signed_parent_mean_coverage": s, "temperature_parent_mean_coverage": t,
        "signed_parent_coverage_bootstrap": mother_bootstrap(mother_s), "temperature_parent_coverage_bootstrap": mother_bootstrap(mother_t),
        "gate_coverage": min(s, t), "small_MC_normal_intervals_have_no_coverage_guarantee": True,
        "execution_recovery": {"registration": bound(REGISTRATION), "existing_cases_rescored_without_reconstruction": 3,
            "remaining_cases_executed": 9, "original_per_repeat_logs_unavailable_for_first_three_cases": True}}


def run():
    registration = common.read_json(REGISTRATION)
    verify_binding(registration["harness"])
    assert registration["supplement_wall_budget_s"] == SUPPLEMENT_SECONDS
    for receipt in registration["bound_inputs"] + registration["recovered_MC_case_reports"]:
        verify_binding(receipt)
    for receipt in registration["existing_MC_arrays"].values():
        verify_binding(receipt)
    protocol, records, scored, selection, null, plans, folder, manifest, selected = frozen_inputs()
    assert not (REC / "STARTED.json").exists(), "Never repeat an already started supplemental invocation."
    assert common.read_json(verify_binding(registration["scientific_stop"]))["ML_authorized"] is False
    lock = common.ed_path(common.REPORTS / "PHASE_ACTIVE.json")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"phase": "development-compare", "execution_kind": "post_timeout_diagnostic_completion",
            "pid": os.getpid(), "started_at": common.now(), "one_root_GPU_job_at_a_time": True}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    budget = common.Budget(SUPPLEMENT_SECONDS)
    event = {"schema": "v5-supplemental-execution-event-v1", "phase": "development-compare",
        "execution_kind": "post_timeout_diagnostic_completion", "time": common.now(), "status": "STARTED",
        "registration": bound(REGISTRATION), "original_failure_receipt": registration["original_failure_receipt"],
        "scientific_gate_passed": False, "numerical_execution_passed": False, "pid": os.getpid()}
    common.write_json(REC / "STARTED.json", event)
    common.append_jsonl(common.REPORTS / "PHASE_EVENTS.jsonl", event)
    gate = None
    hessian_probe_count = 0
    hessian_probes_accounted_for = False
    try:
        physics = common.make_physics("cuda", torch.float32, config=protocol["physics_config"])
        core = selection["core_selection"]
        assert core["device"] == "cuda" and core["dtype"] == "float32"
        cases = {common.read_json(verify_binding(ref))["case_id"]: common.read_json(verify_binding(ref)) for ref in registration["recovered_MC_case_reports"]}
        for cid in registration["remaining_MC_case_ids"]:
            budget.check()
            assert not (common.DATA / "mc_uncertainty_predictions" / (cid + ".npz")).exists()
            singleton = dev._mc_audit(physics, folder, manifest, selected, core, protocol, [cid], budget)
            assert len(singleton["cases"]) == 1 and singleton["cases"][0]["case_id"] == cid
            case = singleton["cases"][0]
            case["execution_provenance"] = "original frozen MC function; previously unexecuted case; same protocol/seeds/options"
            common.write_json(REC / "mc_cases" / (cid + ".json"), dev._plain(case))
            cases[cid] = case
            common.append_jsonl(REC / "events.jsonl", {"stage": "MC", "case_id": cid, "status": case["status"], "time": common.now()})
            print(json.dumps({"stage": "MC", "case_id": cid, "status": case["status"], "accounted_cases": len(cases)}), flush=True)
        mc = aggregate_mc([cases[cid] for cid in selection["MC_case_ids"]], selection)
        common.write_json(DIRECTORY / "mc_coverage.json", dev._plain(mc))
        budget.check()
        hessian = dev._hessian_audit(physics, folder, manifest, selected, core, protocol, selection["MC_case_ids"][0])
        common.write_json(DIRECTORY / "hessian_diagnostics.json", dev._plain(hessian))
        hessian_summary = hessian.get("summary", {})
        hessian_solves = hessian_summary.get("solves", [])
        hessian_probe_count = len(hessian_solves)
        hessian_probes_accounted_for = (hessian_summary.get("options", {}).get("probes") == 8
            and hessian_probe_count == 8 and {r.get("probe") for r in hessian_solves} == set(range(8)))
        if hessian_probes_accounted_for:
            verify_binding(hessian["arrays"])
        common.append_jsonl(REC / "events.jsonl", {"stage": "Hessian", "time": common.now(),
            "all_converged": hessian_summary.get("all_converged", False),
            "actual_probe_records": hessian_probe_count, "all_registered_probes_accounted_for": hessian_probes_accounted_for})
        budget.check()
        from research_v5.baselines import BaselineSet
        baselines = BaselineSet("cuda", physics.cfg)
        stress_folder, stress_manifest, stress_receipt = dev._verified_family(physics, plans, "calibration_stress", protocol)
        stress = dev._run_selected_family(physics, stress_folder, stress_manifest, core, "calibration_stress", budget, null["z_threshold"], baselines)
        assert len(stress) == 180 and len({r["case_id"] for r in stress}) == 36
        assert len({(r["case_id"], r["method"]) for r in stress}) == 180
        budget.check()
        assert hessian_probes_accounted_for, "Hessian helper failed without eight actual probe records; diagnostics remain incomplete."
        stage1_path = common.REPORTS / "DECISION_1.json"
        stage1 = common.read_json(stage1_path)
        gate = dev.evaluate_physics_gate([*selected, *filter_records(records, method=selection["traditional_reference"])],
            selection["traditional_reference"], selection["selection_feasible"], null["evaluation"], mc, stage1)
        assert gate["physics_eligible"] is False
        gate.update(selection_hash=common.sha(DIRECTORY / "selection.json"), selection_path=common.relative(DIRECTORY / "selection.json"),
            threshold_hash=common.sha(DIRECTORY / "null_threshold.json"), threshold_path=common.relative(DIRECTORY / "null_threshold.json"),
            weakdetectable_by_sign=dev._weak_detectable(scored),
            stage1={"path": common.relative(stage1_path), "sha256": common.sha(stage1_path), **stage1},
            manifest_receipts={"development": selection["development_manifest"], "null_calibration": null["calibration_manifest"],
                "null_evaluation": null["evaluation_manifest"], "calibration_stress": stress_receipt},
            statistics={**bound(DIRECTORY / "paired_statistics.json"), "plan": statistics_plan()},
            source_sha256=dev._source_receipt(), protocol_sha256=common.sha(common.CODE / "protocol.json"),
            elapsed_seconds=registration["original_actual_wall_s"]+budget.elapsed(), pure_physics_candidate_kind="v5_physics",
            final_payloads_opened=0, confirmation_payloads_opened=0,
            root_task_spec_projection_review=registration["scientific_stop"],
            execution_recovery={"registration": bound(REGISTRATION), "original_phase_status": "FAILED_TimeoutError",
                "original_wall_s": registration["original_actual_wall_s"], "supplement_wall_s": budget.elapsed(),
                "original_registered_budget_passed": False, "original_failure_not_relabelled": True,
                "original_MC_repeat_details_missing_cases": selection["MC_case_ids"][:3]})
        gate["report"] = dev._write_report(gate, selection, null, mc, hessian, stress)
        report_path = DIRECTORY / "PHYSICS_ONLY_V5_REPORT.md"
        with report_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n## Execution recovery and task compliance\n\nThe original21600-second invocation ended FAILED/TimeoutError after "
                + str(registration["original_actual_wall_s"]) + " seconds. A separately registered5400-second diagnostic completion reused the "
                "complete4320-row grid, fixed selection,96+96 null rows and540 selected scores, without rerunning them. "
                "Three saved MC interval arrays were externally scored; their per-repeat logs were not persisted. "
                "Only the remaining9 MC cases, original8 Hessian probes and36 stress inputs were executed. "
                "All preexisting evidence and the original failure are preserved. This is an explicit execution-budget deviation, "
                "not a successful run within the original budget.\n\nThe additional root Task4B projection review is in "
                "../execution_recovery_001/scientific_stop.json. Both signed normal projection checks fail, as do the registered "
                "color and normal-template RMSE requirements. Alpha1 is an infeasible diagnostic fallback. CNN remains untested; "
                "sealed final accuracy is unknown.\n")
        gate["report"] = bound(report_path)
        common.write_json(DIRECTORY / "development_gate.json", dev._plain(gate))
        for receipt in registration["bound_inputs"]:
            verify_binding(receipt)
        for receipt in registration["existing_MC_arrays"].values():
            verify_binding(receipt)
        common.require_registered()
        no_sealed_open()
        evidence_names = ["records.jsonl", "selected_records.jsonl", "selection.json", "null_calibration_records.jsonl",
            "null_calibration_threshold.json", "null_evaluation_records.jsonl", "null_threshold.json", "mc_coverage.json",
            "hessian_diagnostics.json", "calibration_stress_records.jsonl", "paired_statistics.json", "development_gate.json", "PHYSICS_ONLY_V5_REPORT.md"]
        numerical = mc["complete"] and hessian.get("summary", {}).get("all_converged") is True and all(r.get("numerical_execution_passed") is True for r in stress)
        event.update(status="COMPLETED_WITH_SUPPLEMENTAL_DIAGNOSTICS", numerical_execution_passed=numerical,
            scientific_gate_passed=False, all_planned_diagnostics_accounted_for=True,
            MC_cases=len(cases), newly_executed_MC_cases=9, existing_MC_cases_not_repeated=3,
            actual_Hessian_probe_records=hessian_probe_count, Hessian_probes_accounted_for=hessian_probes_accounted_for,
            stress_cases=36, stress_rows=len(stress), original_grid_solver_failure_rows=2,
            evidence=[bound(DIRECTORY / name) for name in evidence_names])
    except Exception as error:
        event.update(status="FAILED", error_type=type(error).__name__, error=str(error),
            all_planned_diagnostics_accounted_for=False, numerical_execution_passed=False)
        traceback.print_exc()
    finally:
        event.update(finished_at=common.now(), supplemental_actual_wall_s=budget.elapsed(),
            supplemental_registered_budget_s=SUPPLEMENT_SECONDS, supplemental_within_wall_budget=budget.elapsed() <= SUPPLEMENT_SECONDS,
            actual_Hessian_probe_records=hessian_probe_count, Hessian_probes_accounted_for=hessian_probes_accounted_for,
            original_failure_preserved=True, training_executed=False, sealed_evaluation_opened=False)
        common.write_json(REC / "END.json", dev._plain(event))
        common.append_jsonl(common.REPORTS / "PHASE_EVENTS.jsonl", dev._plain(event))
        try:
            reports.write_scientific_decision(protocol)
        finally:
            assert lock == common.ed_path(common.REPORTS / "PHASE_ACTIVE.json")
            lock.unlink()
    print(json.dumps({"status": event["status"], "receipt": str(REC / "END.json"),
        "scientific_gate_passed": False, "supplemental_wall_s": event["supplemental_actual_wall_s"]}), flush=True)
    return 0 if event["status"] == "COMPLETED_WITH_SUPPLEMENTAL_DIAGNOSTICS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
