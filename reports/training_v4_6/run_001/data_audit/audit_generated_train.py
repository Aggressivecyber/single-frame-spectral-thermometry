"""Independent CPU read-only audit of completed v4.6 TRAIN generation.

No project modules, torch, GPU, inverse, sampler mutation or data generation.
Only this script's report directory can receive writes. Formal array reads
require an explicit --execute and the completed generation receipt byte hash.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPORT = HERE / "attempt_002"
ROOT = HERE.parents[3]
RUN = HERE.parent
DATA = ROOT / "outputs/training_v4_6/run_001"
OLD_PATH = ROOT / "reports/training_v4_1/run_001/data_manifest.json"
V4_PATH = ROOT / "reports/training_v4/run_001/data_manifest.json"
REGISTRY_PATH = ROOT / "reports/training_v4_4/run_001/data_audit/repair_parent_registry.json"
CONDITIONS = ("negative_weak", "negative_normal", "positive_weak", "positive_normal")
TEMP = ("front", "multipeak", "ring", "stripe", "flat", "smooth_random")
SPECIES = ("blob", "multipeak", "ring", "stripe", "filament", "smooth_random")
LABEL_KEYS = {"template_e", "temperature_color_K", "band_images_e", "expected_e", "evaluation_domain"}
checks, files, source_checks, noise_records, legal_records = [], [], [], [], []


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def rooted(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Non-relative or escaping manifest path")
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError("Manifest path escaped checkout")
    return result


def check(predicate, name, **details):
    okay = bool(predicate)
    checks.append({"check": name, "pass": okay, **details})
    return okay


def need(predicate, name, **details):
    if not check(predicate, name, **details):
        raise ValueError("Audit precondition failed: " + name)


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE.resolve()):
        raise ValueError("Audit write escaped its own report directory")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def audit_sources(protocol, phase):
    for name, expected in protocol["frozen_inputs_sha256"].items():
        path = rooted(name)
        need(path.suffix.lower() != ".npz", "source_gate_does_not_open_arrays", path=name)
        actual = file_hash(path)
        item = {"phase": phase, "path": name, "expected_sha256": expected, "actual_sha256": actual,
                "pass": actual == expected}
        source_checks.append(item)
        need(item["pass"], "frozen_source_identity", phase=phase, path=name)


def planned_rows(registry, old):
    """Derive expected paths and seed records independently of generator code."""
    planned = {}
    for mother in registry["new_parents"]:
        if mother["split"] != "train":
            continue
        mid = mother["mother_id"]
        for condition in CONDITIONS:
            for suffix, kind, index in (("continuous", "continuous", None), ("quantization", "quantization", None),
                                        ("noise0", "noise", 0), ("noise1", "noise", 1)):
                case = mid + "_" + condition + "_" + suffix
                planned[case] = {"case_id": case, "mother_id": mid, "split": "train", "condition": condition,
                    "input_kind": kind, "noise_index": index,
                    "noise_seed": None if index is None else mother["noise_seeds"][condition][index],
                    "raw_path": relative(DATA / "raw" / mid / (condition + "_" + suffix + ".npz")),
                    "label_path": relative(DATA / "labels" / mid / (condition + ".npz")),
                    "origin": "v4_6_new_signed_train_mother", "existing_label_reused": False}
    old_noise = {(r["mother_id"], r["condition"]): r for r in old["rows"]
                 if r["split"] == "train" and r["signed_aux"] and r["input_kind"] == "noise"}
    for extra in registry["existing_train_mother_additional_noise"]:
        mid, condition = extra["mother_id"], extra["condition"]
        original = old_noise[(mid, condition)]
        case = mid + "_" + condition + "_noise1"
        planned[case] = {"case_id": case, "mother_id": mid, "split": "train", "condition": condition,
            "input_kind": "noise", "noise_index": 1, "noise_seed": extra["noise_seed"],
            "raw_path": relative(DATA / "raw" / mid / (condition + "_noise1.npz")),
            "label_path": original["label_path"], "label_sha256": original["label_sha256"],
            "origin": "v4_6_old_train_second_noise", "existing_label_reused": True,
            "source_existing_case_id": original["case_id"]}
    return planned


def load_arrays(path, expected_path_set, role):
    import numpy as np
    path = Path(path).resolve()
    need(path in expected_path_set, "array_path_is_explicitly_allowlisted", path=relative(path), role=role)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    summary = {}
    for key, value in arrays.items():
        finite = bool(np.isfinite(value).all()) if value.dtype.kind in "bifu" else False
        check(finite, "array_is_finite_numeric", path=relative(path), array=key, dtype=str(value.dtype))
        summary[key] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": array_hash(value),
                        "finite": finite,
                        "minimum": float(value.min()) if finite and value.size else None,
                        "maximum": float(value.max()) if finite and value.size else None}
    item = {"path": relative(path), "role": role, "file_sha256": file_hash(path),
            "bytes": path.stat().st_size, "arrays": summary}
    files.append(item)
    return arrays, item


def run(receipt_sha256):
    import numpy as np
    need(ROOT.drive.upper() in ("E:", "D:"), "audit_checkout_is_on_E_or_D")
    receipt_path = RUN / "generation_receipt.json"
    need(file_hash(receipt_path) == receipt_sha256, "caller_pinned_generation_receipt")
    receipt = read(receipt_path)
    need(receipt["stage"] == "COMPLETE" and read(RUN / "status.json")["stage"] == "COMPLETE",
         "formal_generation_is_complete")
    protocol_path = RUN / "execution_protocol.json"
    need(file_hash(protocol_path) == receipt["protocol_sha256"], "generation_protocol_identity")
    protocol = read(protocol_path)
    need(protocol["frozen"] is True and protocol["authorized_by"] == "root" and protocol["split"] == "train",
         "protocol_frozen_for_train")
    need(read(RUN / "protocol.json") == protocol, "protocol_copies_match")
    audit_sources(protocol, "before_array_reads")
    for name, key in (("data_manifest.json", "merged_manifest_sha256"), ("new_rows.json", "new_rows_sha256")):
        need(file_hash(RUN / name) == receipt[key], "generated_metadata_identity", path=name)
    need(file_hash(RUN / "sampling_sequence.json") == receipt["sampling_sequence_sha256"]
         == protocol["sampling_sequence_sha256"], "sampling_file_identity")
    old, old_v4, registry = read(OLD_PATH), read(V4_PATH), read(REGISTRY_PATH)
    manifest, new_document = read(RUN / "data_manifest.json"), read(RUN / "new_rows.json")
    new_rows, rows = new_document["rows"], manifest["rows"]
    expected = planned_rows(registry, old)
    need(len(old["rows"]) == 3352 and rows[:3352] == old["rows"], "all_3352_inherited_rows_are_unchanged")
    need(len(new_rows) == 432 and rows[3352:] == new_rows, "new_rows_equal_432_manifest_suffix")
    need(Counter(r["split"] for r in rows) == {"train": 3008, "validation": 776}, "merged_split_counts")
    need(len({r["case_id"] for r in rows}) == 3784, "merged_case_ids_are_unique")
    need([r["case_id"] for r in new_rows] == list(expected), "new_row_order_matches_frozen_registry")
    need(Counter(r["input_kind"] for r in new_rows) == {"noise": 240, "continuous": 96, "quantization": 96},
         "new_input_kind_counts")
    for row in new_rows:
        for key, value in expected[row["case_id"]].items():
            need(row.get(key) == value, "row_matches_independent_registry_derivation", case_id=row["case_id"], field=key)
    need(new_document["mode"] == manifest["mode"] == "FORMAL_CUDA", "formal_artifacts_are_not_mock_fixtures")
    cfg = manifest["physics_config"]
    need(cfg == old["physics_config"] and cfg["shape"] == [256, 256], "native_frozen_physics_configuration")
    cfg_hash = canonical(cfg)
    need(all(r["physics_config_sha256"] == r["calibration_sha256"] == cfg_hash for r in new_rows),
         "new_rows_share_frozen_physics_and_calibration")
    for key, value in (("new_train_rows", 432), ("merged_rows", 3784), ("merged_train_rows", 3008),
                       ("old_validation_rows", 776), ("new_train_mothers", 24), ("new_label_files", 96),
                       ("old_train_label_files_reverified", 48), ("locked_development_arrays_generated_or_opened", 0),
                       ("final_test_arrays_or_results_opened", 0), ("cache_cases_materialized", 0), ("optimizer_updates", 0)):
        check(receipt.get(key) == value, "generation_receipt_counts_and_boundaries", field=key, expected=value)
    check(receipt.get("old_3352_rows_unchanged") is True and receipt.get("all_new_npz_reopened_exactly") is True
          and receipt.get("physical_buffers_unchanged") is True and receipt.get("coefficient_recalibration_performed") is False,
          "generation_receipt_integrity_attestations")

    mothers = [m for m in manifest["signed_mothers"] if m["split"] == "train"]
    mother_by = {m["mother_id"]: m for m in mothers}
    pair_counts = Counter((m["temperature_geometry"], m["template_geometry"]) for m in mothers)
    check(len(mothers) == len(mother_by) == 36 and pair_counts == Counter((a, b) for a in TEMP for b in SPECIES),
          "one_signed_training_mother_per_6_by_6_geometry_pair")
    check(len({r["mother_id"] for r in rows if r["split"] == "train"}) == 292, "total_train_parent_count")
    old_signed = {m["mother_id"]: m for m in old["signed_mothers"] if m["split"] == "train"}
    grouped_signed = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["signed_aux"]:
            grouped_signed[(row["mother_id"], row["condition"])].append(row)
    check(len(grouped_signed) == 144, "signed_parent_condition_cells")
    for (mid, condition), group in grouped_signed.items():
        check(Counter(r["input_kind"] for r in group) == {"continuous": 1, "quantization": 1, "noise": 2},
              "each_signed_cell_has_two_noise_inputs", mother_id=mid, condition=condition)
        noisy = [r for r in group if r["input_kind"] == "noise"]
        normalized = []
        for row in noisy:
            if row["case_id"] in expected:
                seed, index, scope = row["noise_seed"], row["noise_index"], "explicit_new_row"
            else:
                descriptor = old_signed[mid]
                seed = descriptor["seed"] + 10000000 + (71 if condition.startswith("positive") else 0) + (17 if condition.endswith("normal") else 0)
                index, scope = 0, "derived_from_frozen_old_generator_and_mother_metadata"
            normalized.append({"case_id": row["case_id"], "noise_index": index, "noise_seed": seed,
                               "seed_evidence": scope, "raw_sha256": row["raw_hash"]})
        check(len(normalized) == 2 and {r["noise_index"] for r in normalized} == {0, 1}
              and len({r["noise_seed"] for r in normalized}) == 2
              and len({r["raw_sha256"] for r in normalized}) == 2,
              "two_noise_seeds_and_raw_hashes_are_distinct", mother_id=mid, condition=condition)
        noise_records.append({"mother_id": mid, "condition": condition, "realizations": normalized})
    noise_seeds = [r["noise_seed"] for cell in noise_records for r in cell["realizations"]]
    check(len(noise_seeds) == len(set(noise_seeds)) == 288, "all_288_signed_train_noise_seed_records_are_unique")

    allowed_raw = {rooted(r["raw_path"]) for r in new_rows}
    allowed_labels = {rooted(r["label_path"]) for r in new_rows}
    new_labels = {p for p in allowed_labels if p.is_relative_to((DATA / "labels").resolve())}
    old_labels = allowed_labels - new_labels
    need(len(allowed_raw) == 432 and len(new_labels) == 96 and len(old_labels) == 48, "explicit_array_allowlist_counts")
    need(all(p.is_relative_to((DATA / "raw").resolve()) for p in allowed_raw), "all_new_raw_paths_under_v46_raw")
    for row in new_rows:
        if row["existing_label_reused"]:
            path = ROOT / "outputs/training_v4_1/run_001/labels" / row["mother_id"] / (row["condition"] + ".npz")
            need(rooted(row["label_path"]) == path.resolve() and row["mother_id"] in old_signed,
                 "reused_label_is_registered_old_train", case_id=row["case_id"])
    check({p.resolve() for p in (DATA / "raw").rglob("*.npz")} == allowed_raw, "new_raw_disk_inventory_exact")
    check({p.resolve() for p in (DATA / "labels").rglob("*.npz")} == new_labels, "new_label_disk_inventory_exact")

    by_label = defaultdict(list)
    for row in new_rows:
        by_label[rooted(row["label_path"])].append(row)
    old_raw_hashes = {r["raw_hash"] for r in old["rows"]}
    old_raw_file_hashes = {r["raw_file_sha256"] for r in old["rows"]}
    old_label_hashes = {r["label_sha256"] for r in old["rows"]}
    raw_hashes, raw_file_hashes, new_label_files, new_label_arrays, old_label_arrays = set(), set(), set(), set(), set()
    quant_comparisons, continuous_comparisons = 0, 0
    adc_step = cfg["full_well_e"] / (2 ** cfg["adc_bits"] - 1 - cfg["black_level_DN"])
    shape = tuple(cfg["shape"])
    for label_path, group in by_label.items():
        is_new = label_path in new_labels
        truth, info = load_arrays(label_path, allowed_labels, "new_train_label" if is_new else "reused_old_train_label")
        need(set(truth) == LABEL_KEYS, "label_array_keys", path=relative(label_path))
        check(all(info["file_sha256"] == r["label_sha256"] for r in group), "all_label_file_hash_references_match", path=relative(label_path))
        actual_label_hashes = {key: item["sha256"] for key, item in info["arrays"].items()}
        check(all(r["label_array_sha256"] == actual_label_hashes for r in group), "all_label_array_hash_references_match", path=relative(label_path))
        for key, value in truth.items():
            expected_shape = (len(cfg["centers_nm"]), *shape) if key == "band_images_e" else shape
            expected_dtype = "bool" if key == "evaluation_domain" else "float32"
            check(value.shape == expected_shape and str(value.dtype) == expected_dtype, "label_shape_and_dtype", path=relative(label_path), array=key)
        check(float(truth["expected_e"].min()) >= 0 and float(truth["expected_e"].max()) < 58000
              and float(truth["band_images_e"].min()) >= 0, "signed_expectation_and_band_legality", path=relative(label_path))
        content_hash = canonical(actual_label_hashes)
        if is_new:
            check(info["file_sha256"] not in old_label_hashes | new_label_files, "new_label_file_has_no_exact_duplicate", path=relative(label_path))
            check(content_hash not in new_label_arrays, "new_label_array_collection_has_no_exact_duplicate", path=relative(label_path))
            new_label_files.add(info["file_sha256"]); new_label_arrays.add(content_hash)
            legal_records.append({"mother_id": group[0]["mother_id"], "condition": group[0]["condition"],
                                  "minimum_expected_e": float(truth["expected_e"].min()),
                                  "maximum_expected_e": float(truth["expected_e"].max()),
                                  "minimum_band_e": float(truth["band_images_e"].min())})
        else:
            check(info["file_sha256"] in old_label_hashes, "reused_label_matches_inherited_metadata", path=relative(label_path))
            old_label_arrays.add(content_hash)
        continuous = None
        for row in sorted(group, key=lambda r: {"continuous": 0, "quantization": 1, "noise": 2}[r["input_kind"]]):
            raw_path = rooted(row["raw_path"])
            arrays, raw_info = load_arrays(raw_path, allowed_raw, "new_train_raw")
            need(set(arrays) == {"raw_dn"}, "raw_array_keys", case_id=row["case_id"])
            raw = arrays["raw_dn"]
            digest = raw_info["arrays"]["raw_dn"]["sha256"]
            check(digest == row["raw_hash"] and raw_info["file_sha256"] == row["raw_file_sha256"],
                  "raw_file_and_array_hash_match", case_id=row["case_id"])
            expected_dtype = "float32" if row["input_kind"] == "continuous" else "uint16"
            check(raw.shape == shape and str(raw.dtype) == expected_dtype, "raw_shape_and_dtype", case_id=row["case_id"])
            check(float(raw.min()) >= 0 and float(raw.max()) < 4095, "raw_adc_bounds_without_saturation", case_id=row["case_id"])
            check(digest not in raw_hashes | old_raw_hashes and raw_info["file_sha256"] not in raw_file_hashes | old_raw_file_hashes,
                  "raw_has_no_exact_duplicate_new_or_inherited", case_id=row["case_id"])
            raw_hashes.add(digest); raw_file_hashes.add(raw_info["file_sha256"])
            if row["input_kind"] == "continuous":
                continuous = raw
                expected_dn = truth["expected_e"] / adc_step + cfg["black_level_DN"]
                tolerance = 8 * np.finfo(np.float32).eps * max(1., float(np.abs(expected_dn).max()))
                error = float(np.abs(raw - expected_dn).max())
                check(error <= tolerance, "continuous_raw_matches_recorded_expectation", case_id=row["case_id"],
                      maximum_absolute_error_DN=error, tolerance_DN=tolerance)
                continuous_comparisons += 1
            elif row["input_kind"] == "quantization":
                check(continuous is not None and np.array_equal(raw, np.rint(continuous).astype(np.uint16)),
                      "quantized_raw_exactly_rounds_continuous", case_id=row["case_id"])
                quant_comparisons += 1
    check(not new_label_arrays & old_label_arrays, "new_labels_not_identical_to_48_reused_old_train_labels")
    check(continuous_comparisons == quant_comparisons == 96, "continuous_quantized_comparison_counts")
    check(len(files) == 576 and Counter(x["role"] for x in files) ==
          {"new_train_raw": 432, "new_train_label": 96, "reused_old_train_label": 48}, "all_576_required_array_files_audited")
    saved_legality = {(r["mother_id"], r["condition"]): r for r in read(RUN / "signed_expectation_legality.json")}
    check(len(saved_legality) == 96 and all(all(saved_legality[(r["mother_id"], r["condition"])][key] == r[key]
          for key in ("minimum_expected_e", "maximum_expected_e", "minimum_band_e")) for r in legal_records),
          "legality_report_matches_actual_label_arrays")
    for file in files:
        check(file_hash(rooted(file["path"])) == file["file_sha256"], "array_file_unchanged_during_audit", path=file["path"])
    audit_sources(protocol, "after_array_reads")
    check(file_hash(receipt_path) == receipt_sha256, "generation_receipt_unchanged_during_audit")
    check(file_hash(RUN / "data_manifest.json") == receipt["merged_manifest_sha256"]
          and file_hash(RUN / "new_rows.json") == receipt["new_rows_sha256"], "generated_metadata_unchanged_during_audit")
    return {"generation_receipt_sha256": receipt_sha256, "protocol_sha256": receipt["protocol_sha256"],
            "new_raw_files": 432, "new_label_files": 96, "reused_old_train_label_files": 48,
            "merged_rows": 3784, "train_rows": 3008, "inherited_development_rows": 776,
            "signed_train_geometry_pairs": 36, "signed_train_noise_seed_records": len(noise_seeds),
            "unique_new_raw_array_hashes": len(raw_hashes), "unique_new_label_array_collections": len(new_label_arrays),
            "array_bytes_read_once": sum(x["bytes"] for x in files),
            "physical_buffer_unchanged_is_generation_receipt_attestation": True,
            "physical_buffers_independently_reconstructed": False,
            "old_noise_zero_seed_scope": "metadata derivation only; old raw arrays were not opened"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--generation-receipt-sha256", required=True)
    args = parser.parse_args()
    if not args.execute or len(args.generation_receipt_sha256) != 64:
        parser.error("Explicit --execute and completed generation receipt SHA256 are required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for name in ("array_audit_receipt.json", "array_audit_files.json", "array_audit_checks.json",
                 "array_audit_sources.json", "two_noise_seed_records.json", "ARRAY_AUDIT.md"):
        if (REPORT / name).exists():
            raise FileExistsError("Preserve prior audit results: " + name)
    REPORT.mkdir(exist_ok=False)
    started, outcome, error = time.monotonic(), {}, None
    try:
        outcome = run(args.generation_receipt_sha256)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    failures = [item for item in checks if not item["pass"]]
    status = "PASS" if error is None and not failures else "FAIL"
    receipt = {"status": status, "time": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started,
        "auditor_sha256": file_hash(__file__), "mode": "INDEPENDENT_CPU_READ_ONLY_ARRAY_AUDIT", **outcome,
        "checks": len(checks), "failed_checks": len(failures), "exception": error,
        "array_files_opened": len(files), "audited_array_file_sha256_map": {r["path"]: r["file_sha256"] for r in files},
        "new_locked_development_arrays_opened": 0, "inherited_development_arrays_opened": 0,
        "final_test_arrays_or_results_opened": 0, "old_raw_arrays_opened": 0,
        "source_or_data_files_modified": False, "seeds_redrawn": 0, "gpu_calls": 0,
        "torch_imported": "torch" in sys.modules, "numpy_threads": 1,
        "limitations": ["No physical forward or inverse is rerun.",
                        "No noise is regenerated; noise independence is supported by seed identity and distinct raw hashes only.",
                        "No statistical near-duplicate or experimental realism assessment is performed.",
                        "Inherited development/final arrays are not inspected; comparisons to them use frozen manifest hashes only."]}
    write_new(REPORT / "array_audit_files.json", files)
    write_new(REPORT / "array_audit_checks.json", checks)
    write_new(REPORT / "array_audit_sources.json", source_checks)
    write_new(REPORT / "two_noise_seed_records.json", noise_records)
    receipt["detail_files_sha256"] = {name: file_hash(REPORT / name) for name in
        ("array_audit_files.json", "array_audit_checks.json", "array_audit_sources.json", "two_noise_seed_records.json")}
    write_new(REPORT / "array_audit_receipt.json", receipt)
    lines = ["# v4.6 TRAIN generated-array audit", "", "Status: **" + status + "**", "",
        f"Independent CPU audit: {len(files)} array files; {len(checks)} checks; {len(failures)} failed checks.", "",
        "Scope: 432 new TRAIN raw files, 96 new TRAIN labels and 48 reused old TRAIN labels.",
        "Inherited manifest rows, split counts, 36 signed TRAIN geometry pairs and two-noise records are checked.",
        "All frozen source/input hashes are checked before and after the array reads.", "",
        "No new locked development, inherited development, final-test or old raw arrays are opened.",
        "No source/data file is modified, no seed is regenerated, and no GPU is used.", "",
        "Physical forward/inverse numerics and buffer invariance are not independently rerun here.",
        "Seed uniqueness and distinct raw hashes are not a statistical proof of independence.", ""]
    if failures or error:
        lines += ["Failures:", "", *["- " + json.dumps(item, ensure_ascii=False) for item in failures[:30]]]
        if error:
            lines += ["- " + error["type"] + ": " + error["message"]]
    with (REPORT / "ARRAY_AUDIT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "audited_array_file_sha256_map"}, ensure_ascii=False))
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
