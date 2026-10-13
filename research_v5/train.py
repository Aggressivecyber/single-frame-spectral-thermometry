"""Exactly one gated bounded-gain training experiment, with outer supervision.

No formal work occurs at import. The root dispatcher owns invoking this phase;
all weights, caches and receipts are new v5 files on E/D. Failed gates and all
registered checkpoints are retained, including a failed neural branch.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from research_v5 import common

common.configure_runtime()

import numpy as np
import torch

from research_v5.signed_gain_denoiser import (
    SignedBoundedGainDenoiser, build_gain_features, FEATURE_NAMES, WIDTH, GAIN_BOUND,
)

SEED = 53131
LEARNING_RATE = 3e-4
MAX_UPDATES = 1000
CHECKPOINTS = (0, 200, 500, 1000)


def exclusive_json(path, value):
    """O_EXCL guard; a crashed/failed started experiment cannot be restarted."""
    path = common.ed_path(path)
    if not path.is_relative_to(common.REPORTS.parent.resolve()):
        raise ValueError("Exclusive stage guards must remain under v5 reports")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if common.read_json(path) != value:
        raise RuntimeError("Exclusive guard failed reopen verification")
    return common.sha(path)


def validate_training_prerequisites(protocol, gate, decision, information):
    """Pure fail-closed check before model creation or training generation."""
    if protocol.get("training", {}).get("model") != "SignedBoundedGainDenoiser":
        return False, "unregistered_training_model"
    spec = protocol["training"]
    exact = {"width": WIDTH, "gain_bound": GAIN_BOUND, "last_head_zero": True,
             "seed": SEED, "optimizer": "Adam", "lr": LEARNING_RATE,
             "max_updates": MAX_UPDATES, "checkpoints": list(CHECKPOINTS),
             "train_mothers": 36, "geometry_factorial": [6, 6], "reuse_old_bias_heads": False}
    if any(spec.get(key) != value for key, value in exact.items()):
        return False, "training_registration_differs_from_single_fixed_experiment"
    if (decision.get("stage1_evidence_complete") is not True
            or decision.get("physical_stage2_eligible") is not True
            or decision.get("ml_stop_required") is not False):
        return False, "stage1_hardstop_or_incomplete_evidence"
    if (information.get("complete") is not True
            or information.get("numerical_checks_passed") is not True
            or information.get("optical_all_passed") is not True
            or information.get("returned_linearizations") != 24):
        return False, "information_numerical_or_optical_gate_failed"
    if gate.get("physics_eligible") is not True or gate.get("no_hardstop") is not True:
        return False, "physical_development_gate_failed"
    weak = gate.get("weakdetectable_by_sign", {})
    for sign in ("positive", "negative"):
        ids = weak.get(sign, {}).get("detectable_case_ids")
        if not isinstance(ids, list) or not ids:
            return False, "no_detectable_weak_both_signs_preregistered_skip"
        if len(ids) != len(set(ids)):
            return False, "duplicate_detectable_weak_case_ids"
    return True, None


def validate_training_partition(plans):
    train = plans["plans"]["training"]
    mothers = plans["mothers"]["training"]
    development = plans["mothers"]["development"]
    if len(mothers) != 36 or len(train) != 360:
        raise ValueError("Training must be 36 fresh mothers x 5 signed/zero strengths x 2 noise repeats")
    pairs = {(m["temperature_geometry"], m["template_geometry"]) for m in mothers}
    if len(pairs) != 36 or len({p[0] for p in pairs}) != 6 or len({p[1] for p in pairs}) != 6:
        raise ValueError("Training must retain the complete 6 x 6 geometry factorial")
    if ({m["mother_id"] for m in mothers} & {m["mother_id"] for m in development}
            or {m["seed"] for m in mothers} & {m["seed"] for m in development}):
        raise ValueError("Train/validation mother identities and scene seeds must be disjoint")
    cells = {}
    for row in train:
        if row.get("partition_role") != "training" or row.get("input_kind") != "noise":
            raise ValueError("Unregistered training role/input")
        cells.setdefault(row["mother_id"], set()).add((row["coefficient_e"], row["noise_index"]))
    expected = {(s, repeat) for s in (0., 45., -45., 180., -180.) for repeat in (0, 1)}
    if len(cells) != 36 or any(cell != expected for cell in cells.values()):
        raise ValueError("Every training mother must have matched +S/-S/0 and both noises")
    return train


def _metadata(row):
    coefficient = float(row["coefficient_e"])
    return {key: copy.deepcopy(row.get(key)) for key in
            ("case_id", "mother_id", "partition_role", "condition", "group", "input_kind", "noise_index")} | {
                "coefficient_e": coefficient,
                "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"}


def _checked_case(folder, row):
    if row.get("raw_available") is not True:
        raise ValueError("Raw generation failed: " + str(row.get("raw_generation_status")))
    folder = common.ed_path(folder)
    raw_path = common.ed_path(folder / row["raw_path"])
    truth_path = common.ed_path(folder / row["label_path"])
    if not raw_path.is_relative_to(folder) or not truth_path.is_relative_to(folder):
        raise ValueError("Manifest payload path escaped its family")
    if common.sha(raw_path) != row["raw_file_sha256"] or common.sha(truth_path) != row["label_sha256"]:
        raise ValueError("Frozen family payload hash differs")
    return common.arrays(raw_path, raw_only=True)["raw_dn"], common.arrays(truth_path)


def _checked_manifest(folder, plan, protocol):
    manifest = common.read_json(folder / "data_manifest.json")
    if (manifest.get("fixture_only") is not False
            or manifest.get("plan_sha256") != common.canonical_hash(plan)
            or manifest.get("physics_config_sha256") != protocol["physics_config_sha256"]
            or manifest.get("protocol_sha256") != common.sha(common.CODE / "protocol.json")
            or [row["case_id"] for row in manifest["rows"]] != [row["case_id"] for row in plan]):
        raise ValueError("Family manifest is not the complete frozen formal plan")
    return manifest


def _prepare_case(physics, raw, truth, core, physical=None):
    from research_v5.reconstruct import reconstruct
    from research_v5.development import raw_noise_proxy
    raw_tensor = torch.as_tensor(raw, device=physics.M.device, dtype=physics.M.dtype)
    if raw_tensor.ndim == 2:
        raw_tensor = raw_tensor[None]
    if physical is None:
        physical = reconstruct(raw_tensor, physics.cfg, core)
    if not bool(torch.isfinite(physical["template_postblur_e"]).all()):
        raise ValueError("Nonfinite frozen physical prediction")
    if float(physical["valid"].double().mean()) < .99:
        raise ValueError("Frozen physical prediction validity below .99")
    sigma = raw_noise_proxy(physics, raw_tensor, core)["sigma_S"].detach()
    features = build_gain_features(raw_tensor, physics,
        template_postblur_e=physical["template_postblur_e"], expected_e=physical["expected_e"],
        continuum_postblur=physical["continuum_postblur"], sigma_S=sigma)
    # Labels exist only in this external preparation/supervision layer.
    target = np.asarray(truth["template_postblur_e"])
    domain = np.asarray(truth["evaluation_domain"], dtype=bool)
    if target.ndim == 2:
        target, domain = target[None], domain[None]
    if target.shape != tuple(physical["template_postblur_e"].shape) or not domain.any() or not np.isfinite(target[domain]).all():
        raise ValueError("Training/scoring truth domain is invalid")
    return {
        "features": features.detach().cpu().numpy(),
        "linear_S": physical["template_postblur_e"].detach().cpu().numpy(),
        "sigma_S": sigma.detach().cpu().numpy(), "target_S": target,
        "domain": domain, "valid": physical["valid"].detach().cpu().numpy(),
    }


def _case_record(row, s, prepared, threshold, *, method):
    domain = prepared["domain"].astype(bool)
    target = prepared["target_S"].astype(np.float64)
    prediction = np.asarray(s, dtype=np.float64)
    finite = bool(np.isfinite(prediction).all() and np.isfinite(prepared["sigma_S"]).all())
    count = int(domain.sum())
    sse = float(np.square(prediction[domain] - target[domain]).sum()) if finite else None
    standardized = np.abs(prediction) / prepared["sigma_S"]
    frame_score = float(standardized.max()) if finite else None
    blank = float(row["coefficient_e"]) == 0
    energy = float(np.square(target[domain]).sum())
    metadata = _metadata(row)
    return {**metadata, "method": method, "status": "SUCCESS" if finite else "NONFINITE_PREDICTION",
            "numerical_execution_passed": finite, "metrics": {
                "domain_pixels": count, "template_SSE": sse,
                "template_truth_energy": energy,
                "template_RMSE_e": math.sqrt(sse/count) if sse is not None else None,
                "template_relative_L2": math.sqrt(sse/energy) if sse is not None and energy else None,
                "valid_fraction": float(prepared["valid"][domain].mean()),
                "z_threshold": float(threshold), "frame_threshold": float(threshold),
                "frame_detection_score": frame_score,
                "frame_detection_score_kind": "max_abs_template_over_raw_proxy_full_FOV",
                "frame_detected": bool(frame_score > threshold) if frame_score is not None else None,
                "null_frame_FP": bool(frame_score > threshold) if blank and frame_score is not None else None,
                "null_pixel_FP": float((standardized[domain] > threshold).mean()) if blank and finite else None,
                "raw_fit_source": "unaltered linear measurement prediction",
                "modified_template_raw_refit_assessed": False}}


def _mother_mse(rows):
    grouped = {}
    for row in rows:
        metrics = row["metrics"]
        if metrics.get("template_SSE") is None or not metrics.get("domain_pixels"):
            return None
        grouped.setdefault(row["mother_id"], []).append(metrics["template_SSE"]/metrics["domain_pixels"])
    return float(np.mean([np.mean(values) for values in grouped.values()])) if grouped else None


def evaluate_checkpoint_gate(neural_records, physical_records, detectable_by_sign, z_threshold):
    """One weight must pass both signs and independent nulls at the same z."""
    neural = {r["case_id"]: r for r in neural_records}
    physical = {r["case_id"]: r for r in physical_records}
    complete = (len(neural) == len(neural_records) and len(physical) == len(physical_records)
                and set(neural) == set(physical) and bool(neural))
    numerical = complete and all(r.get("numerical_execution_passed") is True
        and r["metrics"].get("valid_fraction", 0) >= .99
        and r["metrics"].get("z_threshold") == z_threshold for r in neural_records)
    gates = {"complete_paired_validation": complete, "numerical_and_validity_passed": numerical}
    comparisons = {}
    for sign in ("positive", "negative"):
        for group, factor in (("normal", 1.05), ("detectable_weak", .95)):
            if group == "normal":
                ids = [r["case_id"] for r in physical_records if r.get("group") == "normal" and r.get("sign") == sign]
            else:
                ids = list(detectable_by_sign.get(sign, {}).get("detectable_case_ids", []))
            present = bool(ids) and len(ids) == len(set(ids)) and set(ids).issubset(neural)
            selected_n = [neural[i] for i in ids] if present else []
            selected_p = [physical[i] for i in ids] if present else []
            if group == "detectable_weak":
                present = present and all(r.get("group") == "weak" and r.get("sign") == sign for r in selected_p)
            n_mse, p_mse = _mother_mse(selected_n), _mother_mse(selected_p)
            passed = bool(present and n_mse is not None and p_mse is not None
                          and p_mse > 0 and n_mse <= factor*p_mse)
            key = sign + "_" + group
            gates[key] = passed
            comparisons[key] = {"cases": len(ids), "neural_mother_mean_MSE_e2": n_mse,
                "physics_mother_mean_MSE_e2": p_mse, "maximum_MSE_factor": factor,
                "MSE_factor": n_mse/p_mse if n_mse is not None and p_mse else None,
                "passed": passed, "unit": "equal-weight mother mean of frame full-truth-domain MSE"}
    nulls = [r for r in neural_records if r.get("group") == "blank"]
    null_complete = bool(nulls) and all(r["metrics"].get("null_frame_FP") is not None for r in nulls)
    frame_fp = float(np.mean([r["metrics"]["null_frame_FP"] for r in nulls])) if null_complete else None
    pixel_fp = float(np.mean([r["metrics"]["null_pixel_FP"] for r in nulls])) if null_complete else None
    gates["same_frozen_threshold_null_FP"] = bool(null_complete and frame_fp <= .05)
    weak_losses = [c["neural_mother_mean_MSE_e2"] for name, c in comparisons.items() if "weak" in name]
    objective = float(np.mean(weak_losses)) if all(v is not None for v in weak_losses) and weak_losses else None
    return {"accepted": all(gates.values()), "gates": gates, "comparisons": comparisons,
            "null_evaluation": {"frames": len(nulls), "frame_FP": frame_fp, "pixel_FP": pixel_fp,
                "z_threshold": z_threshold, "threshold_recalibrated": False},
            "selection_objective_detectable_weak_mother_MSE_e2": objective}


def _save_checkpoint(path, model, optimizer, updates, initial_state_hash, selection_hash, protocol_hash):
    path = common.ed_path(path)
    if not path.is_relative_to(common.DATA.parent.resolve()) or path.exists():
        raise ValueError("Checkpoint must be a new v5 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state_hash = common.canonical_hash({key: common.canonical_hash(value.tolist()) for key, value in state.items()})
    payload = {"schema": "v5-single-bounded-gain-checkpoint-001", "model_state": state,
        "optimizer_state": optimizer.state_dict(), "actual_optimizer_updates": int(updates),
        "model_spec": {"model": "SignedBoundedGainDenoiser", "width": WIDTH, "gain_bound": GAIN_BOUND,
                       "feature_names": list(FEATURE_NAMES), "last_head_zero_at_initialization": True},
        "seed": SEED, "lr": LEARNING_RATE, "maximum_updates": MAX_UPDATES,
        "initial_state_hash": initial_state_hash, "model_state_hash": state_hash,
        "selection_sha256": selection_hash, "protocol_sha256": protocol_hash,
        "continuum_trainable_state": False, "supervision_only_outer_loss": True}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded["actual_optimizer_updates"] != updates or not all(torch.equal(v, loaded["model_state"][k]) for k, v in state.items()):
        raise RuntimeError("Checkpoint failed reopen verification")
    steps = [int(value["step"].item()) for value in optimizer.state.values() if "step" in value]
    if updates and (not steps or any(step != updates for step in steps)):
        raise RuntimeError("Actual Adam state does not certify the claimed optimizer updates")
    return {"updates": updates, "path": common.relative(path), "sha256": common.sha(path),
            "model_state_hash": state_hash, "weights_changed_from_initial": state_hash != initial_state_hash,
            "optimizer_state_steps": sorted(set(steps))}


def _write_training_summary(path, summary):
    common.write_json(path, summary)
    from research_v5.reports import write_train_report
    write_train_report(summary)


def train_denoiser(device="cuda"):
    protocol = common.require_registered()
    summary_path = common.REPORTS / "TRAIN_SUMMARY.json"
    if summary_path.exists() or (common.REPORTS / "TRAIN_STARTED.json").exists():
        raise FileExistsError("The single registered neural experiment was already attempted")
    summary = {"schema": "v5-single-neural-experiment-001", "started_at": common.now(),
        "status": "NOT_ELIGIBLE", "eligible": False, "accepted": False, "skip_reason": None,
        "actual_optimizer_updates": 0, "checkpoints": [], "failures": [],
        "formal_training_executed": False, "continuum_trainable_state": False,
        "feature_uncertainty_kind": "frozen raw-derived uniform-W postblur proxy; not MC sigma",
        "old_weights_loaded": False, "threshold_recalibrated": False,
        "physics_candidate_allowed_after_neural_failure": True}
    try:
        gate = common.read_json(common.REPORTS / "development" / "development_gate.json")
        decision = common.read_json(common.REPORTS / "DECISION_1.json")
        information = common.read_json(common.REPORTS / "INFORMATION_SUMMARY.json")
        eligible, reason = validate_training_prerequisites(protocol, gate, decision, information)
        if not eligible:
            summary.update(skip_reason=reason, status="SKIPPED_NO_DETECTABLE_WEAK_BOTH_SIGNS" if
                           reason == "no_detectable_weak_both_signs_preregistered_skip" else "NOT_ELIGIBLE")
            _write_training_summary(summary_path, summary)
            return summary
        selection_path = common.REPORTS / "development" / "selection.json"
        selection = common.read_json(selection_path)
        threshold_path = common.ROOT / selection["thresholds_file"]
        threshold = common.read_json(threshold_path)
        selection_hash, threshold_hash = common.sha(selection_path), common.sha(threshold_path)
        if (gate["selection_hash"] != selection_hash or gate["threshold_hash"] != threshold_hash
                or threshold["selection_hash"] != selection_hash):
            raise ValueError("Development selection/threshold gate hashes differ")
        z_threshold = float(threshold["z_threshold"])
        if not math.isfinite(z_threshold) or z_threshold <= 0:
            raise ValueError("A finite positive frozen null threshold is required")
        core = copy.deepcopy(selection["core_selection"])
        core["device"] = str(device)
        plans = common.read_json(common.CODE / "family_plans.json")
        training_plan = validate_training_partition(plans)
        summary.update(eligible=True, selection_sha256=selection_hash, thresholds_sha256=threshold_hash,
            training_plan_sha256=common.canonical_hash(training_plan), z_threshold=z_threshold,
            training_plan_cases=360, training_mothers=36, maximum_actual_optimizer_updates=MAX_UPDATES,
            frozen_checkpoint_selection_rule="passing 200/500/1000 with minimum equal-sign detectable-weak mother MSE; earliest update breaks ties")
        protocol_hash = common.sha(common.CODE / "protocol.json")
        exclusive_json(common.REPORTS / "TRAIN_STARTED.json", {"started_at": common.now(),
            "protocol_sha256": protocol_hash, "selection_sha256": selection_hash,
            "threshold_sha256": threshold_hash, "seed": SEED, "one_experiment": True})
        budget = common.Budget(protocol["phase_budgets_s"]["train-denoiser"])
        from research_v5.family_data import generate_family
        from research_v5.development import load_cached_prediction
        physics = common.make_physics(device, getattr(torch, core.get("dtype", "float32")))
        train_folder = common.DATA / "training"
        manifest = generate_family(physics, training_plan, train_folder, options={"fixture_only": False,
            "authorization": {"phase": "train-denoiser", "partition_role": "training",
                "plan_sha256": common.canonical_hash(training_plan), "protocol_sha256": protocol_hash}})
        _checked_manifest(train_folder, training_plan, protocol)
        if not manifest.get("matched_T_loga_beta_shape_verified") or manifest["generated_raw_rows"] != 360:
            raise ValueError("The complete matched fresh training family was not generated")
        prepared_train, prepared_validation, physical_records = {}, [], []
        cache_root = common.DATA / "denoiser_prepared"
        for row in manifest["rows"]:
            budget.check()
            raw, truth = _checked_case(train_folder, row)
            arrays = _prepare_case(physics, raw, truth, core)
            path = cache_root / "training" / (row["case_id"] + ".npz")
            digest = common.save_npz(path, **arrays)
            prepared_train.setdefault(row["mother_id"], []).append({"path": common.relative(path), "sha256": digest, "row": row})
        records = common.jsonl(common.REPORTS / "development" / "records.jsonl")
        development_records = {r["case_id"]: r for r in records if r.get("method") == "v5_physics"
            and r.get("alpha_g") == selection["selected_alpha_g"] and r.get("input_kind") == "noise"}
        null_records = {r["case_id"]: r for r in common.jsonl(common.REPORTS / "development" / "null_evaluation_records.jsonl")}
        for family, baseline_records in (("development", development_records), ("null_evaluation", null_records)):
            plan = plans["plans"][family]
            family_folder = common.DATA / family
            validation_manifest = _checked_manifest(family_folder, plan, protocol)
            for row in validation_manifest["rows"]:
                if row["input_kind"] != "noise":
                    continue
                budget.check()
                raw, truth = _checked_case(family_folder, row)
                baseline_record = baseline_records.get(row["case_id"])
                if baseline_record is not None and (baseline_record.get("raw_sha256") != row["raw_file_sha256"]
                        or baseline_record.get("mother_id") != row["mother_id"]):
                    raise ValueError("Physical cache is bound to a different validation raw or mother")
                physical = load_cached_prediction(baseline_record, physics) if baseline_record is not None else None
                arrays = _prepare_case(physics, raw, truth, core, physical)
                path = cache_root / family / (row["case_id"] + ".npz")
                digest = common.save_npz(path, **arrays)
                prepared_validation.append({"row": row, "path": common.relative(path), "sha256": digest})
                physical_records.append(_case_record(row, arrays["linear_S"], arrays, z_threshold, method="v5_physics"))
        if len(prepared_train) != 36 or any(len(v) != 10 for v in prepared_train.values()) or len(prepared_validation) != 276:
            raise ValueError("Incomplete train/validation preparation")
        # All input features are frozen on disk. Labels are used only below.
        torch.manual_seed(SEED)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        model = SignedBoundedGainDenoiser().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        initial_hash = common.canonical_hash({k: common.canonical_hash(v.detach().cpu().tolist()) for k, v in model.state_dict().items()})
        rng = torch.Generator(device="cpu").manual_seed(SEED)
        mothers = sorted(prepared_train)
        order, cursor = [], 0
        loss_records = common.REPORTS / "train" / "optimizer_updates.jsonl"
        summary["status"] = "TRAINING_STARTED"
        for update in range(MAX_UPDATES + 1):
            budget.check()
            if update in CHECKPOINTS:
                model.eval()
                checkpoint = _save_checkpoint(common.DATA / "checkpoints" / f"update_{update:04d}.pt",
                    model, optimizer, update, initial_hash, selection_hash, protocol_hash)
                neural_records = []
                with torch.no_grad():
                    for example in prepared_validation:
                        budget.check()
                        path = common.ROOT / example["path"]
                        if common.sha(path) != example["sha256"]:
                            raise ValueError("Frozen validation feature cache changed")
                        arrays = common.arrays(path)
                        s = torch.as_tensor(arrays["linear_S"], device=device)
                        z = torch.as_tensor(arrays["features"], device=device)
                        prediction = model(s, z).cpu().numpy()
                        neural_records.append(_case_record(example["row"], prediction, arrays, z_threshold, method="v5_neural"))
                gate_result = evaluate_checkpoint_gate(neural_records, physical_records, gate["weakdetectable_by_sign"], z_threshold)
                checkpoint.update(gate_result)
                if update == 0:
                    # An identity at update0 cannot be evidence of denoising gain.
                    checkpoint["initial_identity"] = all(r["metrics"]["template_SSE"] == p["metrics"]["template_SSE"]
                                                          for r, p in zip(neural_records, physical_records))
                    if not checkpoint["initial_identity"]:
                        raise RuntimeError("Zero-head initialization did not reproduce the linear estimate")
                    checkpoint["accepted"] = False
                common.write_json(common.REPORTS / "train" / f"checkpoint_{update:04d}.json", checkpoint)
                for row in neural_records:
                    common.append_jsonl(common.REPORTS / "train" / f"checkpoint_{update:04d}_records.jsonl", row)
                summary["checkpoints"].append(checkpoint)
            if update == MAX_UPDATES:
                break
            if cursor >= len(order):
                order = torch.randperm(len(mothers), generator=rng).tolist()
                cursor = 0
            examples = prepared_train[mothers[order[cursor]]]
            cursor += 1
            loaded = []
            for example in examples:
                path = common.ROOT / example["path"]
                if common.sha(path) != example["sha256"]:
                    raise ValueError("Frozen training feature cache changed")
                loaded.append(common.arrays(path))
            tensors = {key: torch.as_tensor(np.concatenate([a[key] for a in loaded], axis=0), device=device)
                       for key in ("features", "linear_S", "target_S", "domain")}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tensors["linear_S"], tensors["features"])
            domain = tensors["domain"].bool()
            # Single preregistered supervised outer loss; no truth in features.
            loss = ((prediction - tensors["target_S"]).square()[domain].mean() / 180**2)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite outer supervised loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Missing or nonfinite neural gradients")
            optimizer.step()
            summary.update(actual_optimizer_updates=update+1, formal_training_executed=True)
            common.append_jsonl(loss_records, {"actual_optimizer_update": update+1, "loss": float(loss.detach()),
                "mother_id": mothers[order[cursor-1]], "matched_frames": len(examples), "seed": SEED})
        passing = [c for c in summary["checkpoints"] if c["updates"] > 0 and c["accepted"] and c["weights_changed_from_initial"]]
        selected = min(passing, key=lambda c: (c["selection_objective_detectable_weak_mother_MSE_e2"], c["updates"])) if passing else None
        summary.update(status="NEURAL_CANDIDATE_ACCEPTED" if selected else "NEURAL_BRANCH_FAILED_KEEP_PHYSICS",
            accepted=bool(selected), selected_checkpoint=selected,
            checkpoint_count=len(summary["checkpoints"]), completed_registered_training=True,
            validation_cases=len(prepared_validation), independent_null_evaluation_frames=96,
            budget_elapsed_s=budget.elapsed(), finished_at=common.now())
        _write_training_summary(summary_path, summary)
        return summary
    except Exception as error:
        summary.update(status="NEURAL_EXECUTION_FAILED_KEEP_PHYSICS", accepted=False, finished_at=common.now())
        summary["failures"].append({"type": type(error).__name__, "error": str(error),
                                     "actual_optimizer_updates": summary["actual_optimizer_updates"]})
        if not summary_path.exists():
            _write_training_summary(summary_path, summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    summary = train_denoiser(args.device)
    print(json.dumps({k: summary.get(k) for k in ("status", "eligible", "accepted", "skip_reason", "actual_optimizer_updates")}))


if __name__ == "__main__":
    main()
