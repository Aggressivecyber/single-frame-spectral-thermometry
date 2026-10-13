"""Freeze a proposed data-repair registry, never generate synthetic arrays."""
from hashlib import sha256
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
assert ROOT.drive.upper() in ("E:", "D:")


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def freeze(name, value):
    path = OUT / name
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == content, "A frozen design must not be replaced."
    else:
        path.write_text(content, encoding="utf-8")


m4 = read("reports/training_v4/run_001/data_manifest.json")
m41 = read("reports/training_v4_1/run_001/data_manifest.json")
conditions = ["negative_weak", "negative_normal", "positive_weak", "positive_normal"]
temps = ["front", "multipeak", "ring", "stripe", "flat", "smooth_random"]
shapes = ["blob", "multipeak", "ring", "stripe", "filament", "smooth_random"]
old_signed = [m for m in m41["signed_mothers"] if m["split"] == "train"]
old_pairs = {(m["temperature_geometry"], m["template_geometry"]) for m in old_signed}
assert len(old_pairs) == len(old_signed) == 12
new_parents = []
for split, seed_base, role in (("train", 600000, "training"),
                               ("validation", 700000, "development_confirmation_locked")):
    for ti, t in enumerate(temps):
        for si, s in enumerate(shapes):
            if split == "train" and (t, s) in old_pairs:
                continue
            index = 6 * ti + si
            seed = seed_base + 1009 * index
            new_parents.append({"mother_id": f"signed_repair_v1_{role}_t{ti:02d}_s{si:02d}",
                "split": split, "partition_role": role, "pair_index": index,
                "temperature_geometry": t, "template_geometry": s,
                "seed": seed, "temperature_seed": seed + 1, "amplitude_seed": seed + 29,
                "template_seed": seed + 53, "background_seed": seed + 71,
                "noise_seeds": {c: [20000000 + seed + 17 * ci + ni for ni in (0, 1)]
                                for ci, c in enumerate(conditions)}})

extra_noise = []
for m in old_signed:
    for ci, condition in enumerate(conditions):
        extra_noise.append({"mother_id": m["mother_id"], "split": "train",
            "condition": condition, "noise_index": 1,
            "noise_seed": 22000000 + 1009 * m["index"] + 17 * ci + 1,
            "truth_source": "exact existing label expected_e; original label file stays immutable",
            "not_an_independent_mother": True})

old_mothers = m4["mothers"] + m41["signed_mothers"] + m4["final_mothers_registered_but_not_generated"]
seed_keys = ("seed", "temperature_seed", "amplitude_seed", "template_seed", "background_seed")
old_seeds = {m[k] for m in old_mothers for k in seed_keys}
old_seeds |= {r["noise_seed"] for r in m4["rows"] if r.get("noise_seed") is not None}
old_seeds |= {m["seed"] + 10000000 + (71 if c.startswith("positive") else 0) + (17 if c.endswith("normal") else 0)
              for m in m41["signed_mothers"] for c in conditions}
# The original final protocol provides descriptor and raw-noise seed metadata.
old_seeds |= {m["raw_noise_seed_base"] + 17 * ci + ni
              for m in m4["final_mothers_registered_but_not_generated"] for ci in range(4) for ni in (0, 1)}
new_seeds = [m[k] for m in new_parents for k in seed_keys]
new_seeds += [s for m in new_parents for ss in m["noise_seeds"].values() for s in ss]
new_seeds += [r["noise_seed"] for r in extra_noise]
assert len(new_seeds) == len(set(new_seeds))
assert not set(new_seeds) & old_seeds
assert len([m for m in new_parents if m["split"] == "train"]) == 24
assert len([m for m in new_parents if m["split"] == "validation"]) == 36
assert len(extra_noise) == 48
registry = {"status": "DESIGN_ONLY_NO_ARRAYS_GENERATED", "new_parents": new_parents,
            "existing_train_mother_additional_noise": extra_noise}
freeze("repair_parent_registry.json", registry)

plan = {
    "status": "FROZEN_DATA_DESIGN_ONLY_NOT_LAUNCHED",
    "anchor_commit": "623568a4f866859a9185f8ddb2ab16bc14c0d2bf",
    "anchor_manifest": "reports/training_v4_1/run_001/data_manifest.json",
    "anchor_manifest_sha256": sha256((ROOT / "reports/training_v4_1/run_001/data_manifest.json").read_bytes()).hexdigest(),
    "registry_sha256": sha256((OUT / "repair_parent_registry.json").read_bytes()).hexdigest(),
    "activation_condition": "The frozen-continuum contrast must finish its registered budget and verification. Activate only if signed distortion remains, after a separate bounded generation/training registration. This file authorizes no execution.",
    "question": "Does complete signed geometry pairing and a second independent noise realization help beyond repeating the current 12 training mothers?",
    "minimality": "Retain the 12 current TRAIN signed pairs and add exactly the 24 missing pairs. The repeatedly inspected 6 current DEV signed mothers cannot become untouched data, so create one fresh locked DEV confirmation mother for each of all 36 pairs. This is a coverage floor, not a power calculation or generalization guarantee.",
    "counts": {"new_train_mothers": 24, "retained_signed_train_mothers": 12,
        "total_signed_train_mothers": 36, "new_locked_development_mothers": 36,
        "new_train_mother_rows": 384, "second_noise_rows_for_old_train_mothers": 48,
        "total_new_train_rows": 432, "new_locked_development_rows": 576,
        "total_new_raw_rows": 1008, "total_new_condition_labels": 240,
        "existing_condition_labels_reused_for_second_noise": 48,
        "train_rows_after_merge": 3008, "train_mothers_after_merge": 292,
        "old_development_rows_retained_separately": 776},
    "crossed_factors": {
        "temperature_geometry": temps, "template_geometry": shapes,
        "conditions": conditions, "preblur_signed_peaks_e": [-45, -180, 45, 180],
        "input_kinds_per_condition": ["continuous", "quantization", "noise0", "noise1"],
        "mothers_per_geometry_pair_in_repaired_train": 1,
        "mothers_per_geometry_pair_in_locked_development": 1,
        "noise_realizations_per_mother_condition": 2,
        "noise_adds_mothers": False},
    "generator_freeze": {
        "base_mother_function": "training_v4/data.py:mother_state",
        "signed_transform_and_legality": "training_v4_1/data.py:generate",
        "temperature_shift_K": 250.0, "photon_noise": "Poisson(expected_e)",
        "read_noise_sigma_e": 3, "adc_bits": 12, "native_shape": [256, 256],
        "sensor_physics_calibration": "bytewise/canonical identity with anchor; no new sensor randomization",
        "one_shared_scale_for_matched_signs_and_strengths": True,
        "legality_rule": "Keep existing deterministic continuum amplitude and signed template amplitude reductions and 24-attempt cap. Record actual scales; retain illegal cases and stop instead of resampling seeds.",
        "no_expansion_of_amplitude_temperature_or_geometry_family_ranges": True,
        "original_files_immutable": True},
    "independence_guards": [
        "All conditions, raw input types and noise realizations for a mother keep the same split; split is assigned before field construction.",
        "Explicitly persist every raw noise_index and noise_seed, actual post-transform field hashes, generator and physics hashes; do not rely on derived seeds alone.",
        "Compare new TRAIN, current DEV, locked DEV and final registration IDs, seed namespaces, scene-field hashes, raw hashes and label hashes. A duplicate or collision stops generation; it never triggers an unregistered redraw.",
        "Keep exact cached physical inverse identities based on raw/config/calibration/solver. Labels can score and supervise TRAIN but cannot enter inversion or inference.",
        "Preserve the current 776 DEV cases as known development evidence. New locked DEV raw, labels, scores and residual distributions do not select a model, stopping time, sampler, tau, power, thresholds or implementation.",
        "Freeze the model and endpoint before one model evaluation on locked DEV. Any later adaptation converts that set to development evidence; do not rename it independent confirmation.",
        "Report parent-cluster paired statistics and per-geometry/per-sign/per-strength tables. The new confirmation unit count is 36, not 576 input rows or pixels.",
        "Final-test frames, labels and evaluation results remain closed. This proposal creates development confirmation data and does not alter the existing final-test registration."
    ],
    "training_comparison_requirements_not_an_execution_plan": [
        "Register a finite paired contrast separately. Freeze initialization, continuum handling, optimizer, loss, model, fixed budget and all existing gates before generation or performance inspection.",
        "Only TRAIN data determine sampling and any recalibration. Retain the original same-input control and report unique mothers actually drawn, not just available manifest counts.",
        "For a data-only claim, keep condition/input-kind exposure counts and all unsigned sample IDs paired; change only signed mother/noise selection. Additional sampling-policy changes require an explicit separate factor.",
        "A pool with full factorial availability may still be poorly exposed by the old 5-percent signed sampler. Report actual drawn coverage and stop before claiming coverage repair if planned cells were not exposed.",
        "One locked-DEV mother per cell cannot estimate a cell-level variance. Failure stops this bounded question; success warrants an independently registered confirmation, not a real-camera claim."
    ],
    "checks_performed_now": {"metadata_only": True, "new_seed_values": len(new_seeds),
        "new_seed_duplicates": 0, "seed_overlap_with_old_train_dev_or_registered_final": 0,
        "array_generation_calls": 0, "gpu_calls": 0, "optimizer_updates": 0},
    "explicit_noncoverage": ["signed strong", "signed near-zero continuum", "bipolar within one template map",
        "new geometry families", "sensor/PSF/filter/QE drift", "experimental calibration", "independent acquisition"]
}
freeze("frozen_data_repair_plan.json", plan)
print(json.dumps({"status": plan["status"], "counts": plan["counts"], "checks": plan["checks_performed_now"]}, ensure_ascii=False))
