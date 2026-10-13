"""Hash-bound candidate review and one frozen final study authorization.

A failed physical development experiment cannot enter either sealed study.
The optional neural control needs one accepted checkpoint; otherwise the
eligible fixed physics plus its uncertainty is retained without fabricating
neural training or forcing an additional experiment.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from . import common

common.configure_runtime()

from .statistics import statistics_plan


DIRECTORY = common.REPORTS/"candidate_review"


def _bound_file(receipt, *, expected=None):
    path = (common.ROOT/receipt["path"]).resolve()
    if expected is not None and path != Path(expected).resolve():
        raise ValueError("Evidence refers to a noncanonical path")
    if not path.is_relative_to(common.ROOT.resolve()) or not path.is_file():
        raise ValueError("Referenced evidence does not exist in this checkout")
    if common.sha(path) != receipt["sha256"]:
        raise ValueError("Evidence hash changed: "+receipt["path"])
    return path


def choose_candidate(gate, training=None):
    """Pure decision seam: no failed physics, no forced optional learning."""
    if gate.get("physics_eligible") is not True or gate.get("no_hardstop") is not True:
        return {"accepted": False, "eligible": False, "candidate_kind": None,
                "reason": "physical development or stage-1 hard-stop gate failed",
                "checkpoint": None}
    training = training or {}
    checkpoint = training.get("selected_checkpoint")
    if training.get("accepted") is True and checkpoint:
        if (training.get("completed_registered_training") is not True
                or training.get("actual_optimizer_updates") != 1000
                or checkpoint.get("accepted") is not True
                or checkpoint.get("updates") not in (200, 500, 1000)
                or checkpoint.get("weights_changed_from_initial") is not True
                or not checkpoint.get("gates") or not all(checkpoint["gates"].values())):
            return {"accepted": False, "eligible": False, "candidate_kind": None,
                    "reason": "accepted neural claim lacks one completed bounded passing checkpoint",
                    "checkpoint": None}
        return {"accepted": True, "eligible": True, "candidate_kind": "v5_neural",
                "reason": "one same checkpoint passes both signs, detectable weak improvement and the same null threshold",
                "checkpoint": {k: checkpoint[k] for k in ("path", "sha256", "updates")}}
    return {"accepted": True, "eligible": True, "candidate_kind": "v5_physics",
            "reason": ("optional neural control has no accepted eligible checkpoint; retain fixed physics plus uncertainty"),
            "checkpoint": None,
            "neural_status": training.get("status", "NOT_RUN_OPTIONAL_CONTROL"),
            "neural_training_claimed": training.get("formal_training_executed") is True}


def final_mc_case_ids(plans):
    """Metadata-only deterministic bounded MC subset, before final raw access."""
    from .final import final_mc_case_ids as registered_subset
    return registered_subset(plans)


def review_candidate(device="cuda"):
    """Freeze one real eligible candidate or record failure and keep sealed."""
    protocol = common.require_registered()
    path = DIRECTORY/"candidate_receipt.json"
    if path.exists() or (common.REPORTS/"final_protocol.json").exists():
        raise FileExistsError("Candidate review/final protocol is never retuned or replaced")
    gate_path = common.REPORTS/"development"/"development_gate.json"
    gate = common.read_json(gate_path)
    selection_path = common.REPORTS/"development"/"selection.json"
    selection = common.read_json(selection_path)
    threshold_path = common.ROOT/selection["thresholds_file"]
    threshold = common.read_json(threshold_path)
    selection_hash, threshold_hash = common.sha(selection_path), common.sha(threshold_path)
    if (gate.get("selection_hash") != selection_hash or gate.get("threshold_hash") != threshold_hash
            or threshold.get("selection_hash") != selection_hash
            or selection.get("protocol_sha256") != common.sha(common.CODE/"protocol.json")
            or gate.get("protocol_sha256") != common.sha(common.CODE/"protocol.json")):
        raise ValueError("Candidate development evidence hashes do not bind the same frozen selection")
    if selection.get("status") != "FROZEN_BEFORE_NULL_THRESHOLD_CALIBRATION":
        raise ValueError("One configuration was not frozen before null calibration")
    if selection.get("statistics") != statistics_plan() or gate.get("statistics", {}).get("plan") != statistics_plan():
        raise ValueError("Frozen mother statistical plan differs")
    if protocol["bootstrap"] != {k: statistics_plan()[k] for k in protocol["bootstrap"]}:
        raise ValueError("Protocol and candidate bootstrap rules differ")
    _bound_file(gate["statistics"], expected=common.REPORTS/"development"/"paired_statistics.json")
    _bound_file(gate["stage1"], expected=common.REPORTS/"DECISION_1.json")
    calibration_path = _bound_file(threshold["calibration_threshold_freeze"],
                                  expected=common.REPORTS/"development"/"null_calibration_threshold.json")
    calibration = common.read_json(calibration_path)
    if calibration["selection_hash"] != selection_hash or calibration["z_threshold"] != threshold["z_threshold"]:
        raise ValueError("Null evaluation threshold differs from the immutable calibration freeze")
    for name, receipt in gate["manifest_receipts"].items():
        _bound_file(receipt, expected=common.DATA/name/"data_manifest.json")
    training_path = common.REPORTS/"TRAIN_SUMMARY.json"
    training = common.read_json(training_path) if training_path.exists() else None
    decision = choose_candidate(gate, training)
    z = threshold.get("z_threshold")
    if decision["eligible"] and (threshold.get("complete") is not True or not isinstance(z, (int, float))
            or isinstance(z, bool) or not math.isfinite(z) or z <= 0
            or threshold.get("retuned_on_evaluation") is not False
            or threshold.get("threshold_frozen_before_evaluation") is not True):
        decision = {"accepted": False, "eligible": False, "candidate_kind": None,
                    "reason": "a positive finite independent threshold was not frozen and evaluated without retuning",
                    "checkpoint": None}
    if decision["candidate_kind"] == "v5_neural":
        checkpoint_path = _bound_file(decision["checkpoint"])
        if not checkpoint_path.is_relative_to(common.DATA.parent.resolve()):
            raise ValueError("Neural checkpoint is not a new v5 artifact")
        if (training["selection_sha256"] != selection_hash or training["thresholds_sha256"] != threshold_hash
                or training["threshold_recalibrated"] is not False
                or training.get("continuum_trainable_state") is not False):
            raise ValueError("Neural checkpoint selection/threshold/continuum contract differs")
    plans = common.read_json(common.CODE/"family_plans.json")
    assert common.sha(common.CODE/"family_plans.json") == protocol["family_plans_sha256"]
    from .final import frozen_evaluation_fields
    evaluation_fields = frozen_evaluation_fields(plans)
    mc_cases = evaluation_fields["final_MC_case_ids"]
    if "final_MC_case_ids" in protocol and protocol["final_MC_case_ids"] != mc_cases:
        raise ValueError("Final MC subset differs from the predeclared metadata")
    counts = {name: len(plans["plans"][name]) for name in ("signed_confirmation", "final_once")}
    assert counts == {"signed_confirmation": 288, "final_once": 512}
    core = copy.deepcopy(selection["core_selection"])
    if str(device) != str(core["device"]):
        raise ValueError("Review execution device must match frozen core selection")
    receipt = {"schema": "v5-hash-bound-candidate-review-001", "reviewed_at": common.now(), **decision,
               "status": "ELIGIBLE_CANDIDATE_FROZEN" if decision["eligible"] else "NOT_ELIGIBLE_FINAL_REMAINS_SEALED",
               "config": core, "config_sha256": common.canonical_hash(core), "core_selection": core,
               "selection_path": common.relative(selection_path), "selection_sha256": selection_hash,
               "thresholds": {"path": common.relative(threshold_path), "sha256": threshold_hash, "z_threshold": z},
               "statistics": statistics_plan(), "uncertainty_proxy": selection["uncertainty_proxy"],
               "traditional_reference": selection["traditional_reference"],
               "development_gate": {"path": common.relative(gate_path), "sha256": common.sha(gate_path)},
               "training_summary": {"path": common.relative(training_path), "sha256": common.sha(training_path)} if training else None,
               "protocol_sha256": common.sha(common.CODE/"protocol.json"),
               "plan_hashes": copy.deepcopy(protocol["family_plan_hashes"]),
               "source_sha256": common.source_hashes(),
               "final_and_confirmation_sealed": True, "final_payloads_opened": 0,
               "all_failed_development_gates": gate["failed_gates"],
               "scientific_claims": "synthetic single-frame postblur signed deviation and continuum color under fixed calibration"}
    digest = common.write_json(path, receipt)
    if decision["eligible"]:
        common.write_json(common.REPORTS/"final_protocol.json", {
            "schema": "v5-one-time-final-protocol-001", "status": "FROZEN_CANDIDATE_THRESHOLDS_STATISTICS",
            "frozen_at": common.now(), "candidate_receipt_path": common.relative(path),
            "candidate_receipt_sha256": digest, "thresholds": receipt["thresholds"],
            "statistics": receipt["statistics"], "selection_sha256": selection_hash,
            "core_selection": core, "candidate_kind": receipt["candidate_kind"],
            "checkpoint": receipt["checkpoint"], "traditional_reference": receipt["traditional_reference"],
            "protocol_sha256": receipt["protocol_sha256"], "plan_hashes": receipt["plan_hashes"],
            "family_plans_sha256": common.sha(common.CODE/"family_plans.json"),
            **evaluation_fields,
            "source_sha256": receipt["source_sha256"], "confirmation_and_final_arrays_opened": False,
            "retuning_permitted": False, "joint_final_evaluations_allowed": 1})
    return receipt
