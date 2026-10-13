"""CPU-only metadata audit. Never imports project code or opens frame/label arrays.

Run with Python -B. Reads an explicit allowlist, writes only beside this script.
Final-test access is limited to registered mother/seed metadata already embedded
in the v4 data manifest; final frames, labels and evaluation results are excluded.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
assert ROOT.drive.upper() in ("E:", "D:")
READS = []


def load(rel):
    path = ROOT / rel
    data = path.read_bytes()
    READS.append({"path": rel, "sha256": sha256(data).hexdigest(), "bytes": len(data)})
    return json.loads(data.decode("utf-8-sig"))


def write(name, obj):
    path = OUT / name
    assert path.parent == OUT
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def frequency(records, keys):
    counts = Counter(tuple(r.get(k) for k in keys) for r in records)
    return [{**dict(zip(keys, key)), "count": count}
            for key, count in sorted(counts.items(), key=lambda kv: repr(kv[0]))]


def split_overlap(records, keys):
    result = {}
    for key in keys:
        parts = {}
        for split in ("train", "validation"):
            ids = defaultdict(list)
            for r in records:
                if r["split"] == split and r.get(key) is not None:
                    ids[r[key]].append(r.get("case_id", r.get("mother_id")))
            parts[split] = ids
        common = sorted(set(parts["train"]) & set(parts["validation"]))
        result[key] = {"train_unique": len(parts["train"]),
                       "validation_unique": len(parts["validation"]),
                       "intersection_count": len(common),
                       "intersections": [{"value": value, "train": parts["train"][value],
                                           "validation": parts["validation"][value]} for value in common]}
    return result


def overview(rs):
    return {"rows": len(rs), "mothers": len({r["mother_id"] for r in rs}),
            "condition_labels": len({(r["mother_id"], r["condition"]) for r in rs}),
            "unique_label_sha256": len({r["label_sha256"] for r in rs}),
            "unique_raw_sha256": len({r["raw_hash"] for r in rs}),
            "by_condition_kind": frequency(rs, ("condition", "input_kind")),
            "by_origin": frequency(rs, ("origin",))}


m4 = load("reports/training_v4/run_001/data_manifest.json")
m41 = load("reports/training_v4_1/run_001/data_manifest.json")
sequence = load("reports/training_v4_1/run_001/sampling_sequence.json")
balanced = load("reports/training_v4_3/run_001/negative_balanced/sampling_sequence.json")
legal = load("reports/training_v4_1/run_001/signed_expectation_legality.json")
receipt = load("reports/local_session_20261013/final_receipt.json")
calibration = load("reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json")
rows = m41["rows"]
by_id = {r["case_id"]: r for r in rows}
assert len(by_id) == len(rows)
assert {r["split"] for r in rows} == {"train", "validation"}
mothers = m4["mothers"] + m41["signed_mothers"]
mother_by_id = {m["mother_id"]: m for m in mothers}
assert len(mother_by_id) == len(mothers)
assert all(r["mother_id"] in mother_by_id for r in rows)

summary = {"scope": "metadata_only_train_and_development_validation",
           "audit_utc": datetime.now(timezone.utc).isoformat(),
           "git_head_at_audit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "frame_label_arrays_opened": 0, "project_modules_imported": 0,
           "gpu_calls": 0, "optimizer_updates": 0,
           "final_metadata_only": True, "final_frame_label_or_result_opened": False,
           "all_rows": len(rows), "all_mothers": len(mothers),
           "split_summaries": {}, "geometry_coverage": {},
           "reported_prior_delivery_receipt": receipt}

temp_types = ["front", "multipeak", "ring", "stripe", "flat", "smooth_random"]
template_types = ["blob", "multipeak", "ring", "stripe", "filament", "smooth_random"]
for split in ("train", "validation"):
    rs = [r for r in rows if r["split"] == split]
    summary["split_summaries"][split] = {
        "total": overview(rs),
        "main_noisy": overview([r for r in rs if r["input_kind"] == "noise" and not r["signed_aux"]]),
        "main_protection": overview([r for r in rs if r["input_kind"] != "noise" and not r["signed_aux"]]),
        "signed": overview([r for r in rs if r["signed_aux"]]),
        "signed_negative_noisy": overview([r for r in rs if r["signed_aux"] and r["condition"].startswith("negative") and r["input_kind"] == "noise"])}
    for name, ms in (("main", m4["mothers"]), ("protection", m41["selected_mothers"]), ("signed", m41["signed_mothers"])):
        subset = [m for m in ms if m["split"] == split]
        cnt = Counter((m["temperature_geometry"], m["template_geometry"]) for m in subset)
        matrix = [{"temperature_geometry": t, "template_geometry": s,
                   "mother_count": cnt[(t, s)]} for t in temp_types for s in template_types]
        summary["geometry_coverage"][split + "_" + name] = {
            "mothers": len(subset), "nonempty_pairs_of_36": len(cnt),
            "per_temperature": frequency(subset, ("temperature_geometry",)),
            "per_template": frequency(subset, ("template_geometry",)), "matrix": matrix}

summary["cross_split_row_metadata_overlap"] = split_overlap(rows,
    ("case_id", "mother_id", "raw_path", "raw_hash", "raw_file_sha256", "label_path", "label_sha256"))
summary["cross_split_mother_metadata_overlap"] = split_overlap(mothers,
    ("mother_id", "seed", "temperature_seed", "amplitude_seed", "template_seed", "background_seed",
     "temperature_field_sha256", "amplitude_field_sha256", "template_shape_sha256"))

seed_records = []
signed_missing_noise_seed = 0
for r in rows:
    if r["input_kind"] != "noise":
        continue
    seed = r.get("noise_seed")
    provenance = "recorded"
    if r["signed_aux"]:
        signed_missing_noise_seed += seed is None
        m = mother_by_id[r["mother_id"]]
        seed = m["seed"] + 10_000_000 + (71 if r["condition"].startswith("positive") else 0) + (17 if r["condition"].endswith("normal") else 0)
        provenance = "derived_from_frozen_generator_formula_not_stored_in_row"
    seed_records.append({"case_id": r["case_id"], "mother_id": r["mother_id"], "split": r["split"],
                         "noise_seed": seed, "provenance": provenance})
summary["noise_seed_audit"] = {"rows": len(seed_records),
    "signed_noisy_rows_with_no_explicit_noise_seed": signed_missing_noise_seed,
    "cross_split_overlap": split_overlap(seed_records, ("noise_seed",)),
    "within_split_unique": {s: len({r["noise_seed"] for r in seed_records if r["split"] == s}) for s in ("train", "validation")},
    "warning": "Different RNG seeds identify noise realizations; they do not add independent parent scenes."}

original = {r["case_id"]: r for r in m4["rows"] if not r["clean_identity"]}
reused = [r for r in rows if r["origin"] == "v4_exact_reuse"]
reused_fields = ("mother_id", "split", "condition", "noise_seed", "noise_index", "raw_hash", "raw_file_sha256", "label_sha256")
summary["reuse_checks"] = {"old_noisy_rows": len(original), "reused_rows": len(reused),
    "all_old_noisy_ids_reused_exactly": {r["case_id"] for r in reused} == set(original),
    "all_selected_metadata_equal": all(all(r[k] == original[r["case_id"]][k] for k in reused_fields) for r in reused),
    "all_protection_mothers_are_same_split_main_mothers": all(m["mother_id"] in {x["mother_id"] for x in m4["mothers"] if x["split"] == m["split"]} for m in m41["selected_mothers"]),
    "protection_added_independent_mothers": 0}


def sampling_summary(draws):
    flat = [by_id[c] for batch in draws for c in batch]
    assert all(r["split"] == "train" for r in flat)
    negative = [r for r in flat if r["input_kind"] == "noise" and r["condition"].startswith("negative")]
    counts = Counter(r["case_id"] for r in negative)
    per_case = [{"case_id": case, "draws": n} for case, n in sorted(counts.items())]
    return {"updates": len(draws), "microbatches": len(flat),
            "unique_cases": len({r["case_id"] for r in flat}),
            "unique_mothers": len({r["mother_id"] for r in flat}),
            "unique_signed_mothers": len({r["mother_id"] for r in flat if r["signed_aux"]}),
            "signed_microbatches": sum(r["signed_aux"] for r in flat),
            "negative_noisy_microbatches": len(negative),
            "unique_negative_noisy_cases": len(counts),
            "unique_negative_noisy_mothers": len({r["mother_id"] for r in negative}),
            "negative_noisy_per_case": per_case,
            "condition_kind_counts": frequency(flat, ("condition", "input_kind")),
            "in_batch_duplicate_count": sum(len(set(b)) != len(b) for b in draws),
            "non_train_draws": 0}


summary["sampler"] = {"v4_1_all_1500": sampling_summary(sequence["draws"]),
    "v4_1_prefix_500": sampling_summary(sequence["draws"][:500]),
    "negative_balanced_500": sampling_summary(balanced["draws"])}
slots = Counter(batch[1] for batch in balanced["draws"])
summary["sampler"]["negative_balanced_slot1"] = {
    "updates": len(balanced["draws"]), "unique_cases": len(slots),
    "unique_mothers": len({by_id[c]["mother_id"] for c in slots}),
    "case_repeat_min": min(slots.values()), "case_repeat_max": max(slots.values()),
    "per_case_repeat_distribution": dict(sorted(Counter(slots.values()).items())),
    "all_other_slots_match_original": all(a[i] == b[i] for a, b in zip(sequence["draws"][:500], balanced["draws"]) for i in (0, 2, 3)),
    "matching_unchanged_slots": sum(a[i] == b[i] for a, b in zip(sequence["draws"][:500], balanced["draws"]) for i in (0, 2, 3)),
    "new_data_cases": 0}

summary["signed_generation"] = {
    "temperature_shift_K": sorted({m["temperature_shift_K"] for m in m41["signed_mothers"]}),
    "continuum_amplitude_scales": frequency(m41["signed_mothers"], ("split", "continuum_amplitude_scale")),
    "legal_template_scales": frequency(m41["signed_mothers"], ("split", "legal_amplitude_scale")),
    "actual_peak_e_by_condition": {c: sorted({r["actual_preblur_peak_e"] for r in legal if r["condition"] == c}) for c in sorted({r["condition"] for r in legal})},
    "minimum_expected_e": min(r["minimum_expected_e"] for r in legal),
    "maximum_expected_e": max(r["maximum_expected_e"] for r in legal),
    "minimum_band_e": min(r["minimum_band_e"] for r in legal),
    "mother_temperature_hash_scope": "pre +250 K latent temperature field as returned by mother_state; actual labels are post forward and post blur",
    "one_noise_realization_per_signed_mother_condition": True,
    "paired_signs_strengths_share_continuum_amplitude_background_template_shape": True}

final_parents = m4["final_mothers_registered_but_not_generated"]
training_parent_ids = {m["mother_id"] for m in mothers}
training_seed_values = {m[k] for m in mothers for k in ("seed", "temperature_seed", "amplitude_seed", "template_seed", "background_seed")}
summary["final_registration_metadata"] = {"registered_mothers": len(final_parents),
    "final_parent_ids_overlap_used_splits": len(training_parent_ids & {m["mother_id"] for m in final_parents}),
    "final_seed_values_overlap_used_splits": len(training_seed_values & {m[k] for m in final_parents for k in ("seed", "temperature_seed", "amplitude_seed", "template_seed", "background_seed")}),
    "raw_or_labels_opened": False, "evaluation_results_opened": False,
    "current_absence_of_unregistered_final_outputs_independently_checked": False}
summary["calibration_boundary"] = {k: calibration[k] for k in
    ("tau", "power", "training_cases", "protected_count", "noise_count", "calibration_inputs", "validation_motivated_diagnostic", "final_test_used")}

summary["source_provenance_checks"] = {}
for rel, expected in (("training_v4/data.py", m4["generator_source_sha256"]),
                      ("training_v4_1/data.py", m41["generator_sha256"])):
    raw = (ROOT / rel).read_bytes()
    actual = sha256(raw).hexdigest()
    READS.append({"path": rel, "sha256": actual, "bytes": len(raw)})
    normalized_lf = sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    summary["source_provenance_checks"][rel] = {"manifest_sha256": expected,
        "current_file_sha256": actual, "bytewise_matches": actual == expected,
        "LF_normalized_sha256": normalized_lf, "LF_normalized_matches": normalized_lf == expected}
summary["source_provenance_checks"]["v4_manifest_hash"] = {"expected_by_v4_1": m41["v4_manifest_sha256"],
    "current": READS[0]["sha256"], "bytewise_matches": m41["v4_manifest_sha256"] == READS[0]["sha256"]}

for rel in ("training_v4_1/sampling.py", "training_v4_3/balanced_sampling.py", "training_v4_1/evaluate.py",
            "training_v4_1/cache.py", "training_v4/select.py", "training_v4_1/tikhonov.py",
            "training_v4_3/residual_scaling.py", "reports/local_session_20261013/LOCAL_STUDY_REPORT.md"):
    raw = (ROOT / rel).read_bytes()
    READS.append({"path": rel, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
summary["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
summary["read_inventory"] = READS
write("audit_metadata.json", summary)
write("signed_noise_seed_derivation.json", [r for r in seed_records if r["provenance"].startswith("derived")])
print(json.dumps({"output": str(OUT / "audit_metadata.json"),
    "rows": summary["all_rows"], "mothers": summary["all_mothers"],
    "splits": {s: summary["split_summaries"][s]["total"] for s in ("train", "validation")},
    "row_overlap_counts": {k: v["intersection_count"] for k, v in summary["cross_split_row_metadata_overlap"].items()},
    "field_overlap_counts": {k: v["intersection_count"] for k, v in summary["cross_split_mother_metadata_overlap"].items()},
    "geometry_nonempty": {k: v["nonempty_pairs_of_36"] for k, v in summary["geometry_coverage"].items()},
    "slot1": summary["sampler"]["negative_balanced_slot1"],
    "signed_generation": summary["signed_generation"],
    "provenance": summary["source_provenance_checks"]}, ensure_ascii=False))
