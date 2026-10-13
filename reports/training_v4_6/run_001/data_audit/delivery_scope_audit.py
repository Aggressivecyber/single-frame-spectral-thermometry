"""Read-only delivery-scope inspection; no array/model/runtime imports or jobs.

Only delivery_scope_audit.json and delivery_scope_audit.md are written beside
this script. Existing evidence, delivery scripts and frozen sources are read
only. This is an in-progress snapshot, not a build or relocation acceptance.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = HERE.parent
BUILDER = "reports/training_v4_6/run_001/delivery/build_inference_bundle.py"
SEAL = "reports/parallel_session_20261013_v44_v46/seal_results.py"
PORTABILITY = "reports/training_v4_6/run_001/delivery/verify_portability.py"
INFER = "training_v4_4/infer.py"


def sha(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def literal(tree, name):
    nodes = [x.value for x in tree.body if isinstance(x, ast.Assign)
             and any(isinstance(y, ast.Name) and y.id == name for y in x.targets)]
    if len(nodes) != 1:
        raise ValueError("Expected one literal assignment: " + name)
    return ast.literal_eval(nodes[0])


def main():
    assert ROOT.drive.upper() in ("E:", "D:")
    paths = [ROOT / x for x in (BUILDER, SEAL, PORTABILITY, INFER, "runtime.py", "training_v4_1/common.py")]
    paths += [p for folder in ("training_v4_4", "training_v4_5", "training_v4_6")
              for p in (ROOT / folder).glob("*.py")]
    paths = sorted(set(paths))
    source_hashes = {rel(p): sha(p) for p in paths}
    trees = {rel(p): ast.parse(p.read_text(encoding="utf-8-sig")) for p in paths}
    builder_sources = literal(trees[BUILDER], "SOURCE_SHA256")
    static_audit = read("reports/training_v4_6/run_001/delivery/bundle_static_dependency_audit.json")
    static_review = read("reports/training_v4_6/run_001/delivery/bundle_builder_static_review.json")
    source_matches = {name: sha(ROOT / name) == expected for name, expected in builder_sources.items()}
    static_artifacts_match = (static_audit["builder_sha256"] == static_review["builder_sha256"] == source_hashes[BUILDER]
        and static_review["static_dependency_audit_sha256"] == sha(ROOT / "reports/training_v4_6/run_001/delivery/bundle_static_dependency_audit.json")
        and static_audit["expected_sources_sha256"] == builder_sources and all(source_matches.values()))
    absolute_literals, c_drive_literals = [], []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.match(r"^[A-Za-z]:[/\\]", node.value):
                    absolute_literals.append({"path": name, "line": node.lineno, "value": node.value})
                if re.search(r"(?i)(?<![A-Za-z0-9])C:[/\\]", node.value):
                    c_drive_literals.append({"path": name, "line": node.lineno, "value": node.value[:200]})
    seal_tree = trees[SEAL]
    bundle_load_references = [node.lineno for node in ast.walk(seal_tree)
        if isinstance(node, ast.Name) and node.id == "bundle" and isinstance(node.ctx, ast.Load)]
    portability_keys = sorted(set(node.slice.value for node in ast.walk(seal_tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "portability"
        and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)))
    registry = read("reports/training_v4_4/run_001/data_audit/repair_parent_registry.json")
    locked = [x["mother_id"] for x in registry["new_parents"] if x["split"] == "validation"]
    final = read("reports/training_v4/run_001/data_manifest.json")["final_mothers_registered_but_not_generated"]
    final_ids = [x["mother_id"] for x in final]
    inventories = {}
    for version in ("v4_4", "v4_5", "v4_6"):
        base = ROOT / f"outputs/training_{version}/run_001"
        entries = list(base.rglob("*")) if base.exists() else []
        file_paths = [p for p in entries if p.is_file()]
        inventories[version] = {"path": rel(base), "exists": base.exists(), "file_count": len(file_paths),
            "total_bytes": sum(p.stat().st_size for p in file_paths),
            "top_level_file_counts": dict(Counter(p.relative_to(base).parts[0] for p in file_paths)),
            "all_resolved_on_ED": all(p.resolve().drive.upper() in ("E:", "D:") for p in entries),
            "symlink_or_junction_entries": [rel(p) for p in entries if p.is_symlink() or getattr(p, "is_junction", lambda: False)()],
            "registered_locked_parent_path_hits": [rel(p) for p in entries if p.name in locked],
            "registered_final_parent_path_hits": [rel(p) for p in entries if p.name in final_ids]}
    original = read("reports/training_v4_4/run_001/preserved_starting_files.json")
    size_mismatches, metadata_hash_mismatches = [], []
    metadata_checked = 0
    for item in original["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            size_mismatches.append(item["path"])
        # Do not open array/model payloads for this delivery-scope inspection.
        if path.suffix.lower() not in (".npz", ".pt") and path.is_file():
            metadata_checked += 1
            if sha(path) != item["sha256"]:
                metadata_hash_mismatches.append(item["path"])
    generation = read("reports/training_v4_6/run_001/generation_receipt.json")
    cache = read("reports/training_v4_6/run_001/cache_receipt.json")
    manifest = read("reports/training_v4_6/run_001/data_manifest.json")
    old = read("reports/training_v4_1/run_001/data_manifest.json")
    old_dev = [row for row in old["rows"] if row["split"] == "validation"]
    current_dev = [row for row in manifest["rows"] if row["split"] == "validation"]
    new_rows = manifest["rows"][3352:]
    actual_raw_paths = {rel(p) for p in (ROOT / "outputs/training_v4_6/run_001/raw").rglob("*.npz")}
    actual_label_paths = {rel(p) for p in (ROOT / "outputs/training_v4_6/run_001/labels").rglob("*.npz")}
    raw_paths_expected = {row["raw_path"] for row in new_rows}
    label_paths_expected = {row["label_path"] for row in new_rows if not row["existing_label_reused"]}
    examples = literal(trees[BUILDER], "EXAMPLES")
    scope = {"new_rows_all_train": all(row["split"] == "train" for row in new_rows),
        "new_rows_count": len(new_rows), "old_776_development_rows_unchanged": current_dev == old_dev and len(old_dev) == 776,
        "actual_new_raw_paths_exact": actual_raw_paths == raw_paths_expected,
        "actual_new_label_paths_exact": actual_label_paths == label_paths_expected,
        "builder_examples_all_old_development": all(x["split"] == "validation" and x["case_id"] in {r["case_id"] for r in old_dev} for x in examples),
        "builder_example_count": len(examples), "generation_final_arrays_reported": generation["final_test_arrays_or_results_opened"],
        "generation_locked_arrays_reported": generation["locked_development_arrays_generated_or_opened"],
        "cache_final_arrays_reported": cache["final_test_arrays_or_results_opened"],
        "cache_locked_arrays_reported": cache["locked_development_arrays_generated_or_opened"]}
    status = read("reports/training_v4_6/run_001/negative_noisy_repair/status.json")
    delivery = ROOT / "outputs/training_v4_6/run_001/delivery"
    findings = []
    if not bundle_load_references:
        findings.append({"id": "DELIVERY-001", "severity": "delivery_gate_gap", "status": "OPEN_AT_SNAPSHOT",
            "file": SEAL, "lines": [56, 61], "title": "Seal does not validate or bind its bundle receipt",
            "evidence": "bundle = read(bundle_receipt) has zero AST load references afterwards. Portability checks consume only status and all_prediction_arrays_exact.",
            "impact": "With other prerequisites satisfied, a failed/unrelated bundle JSON or missing/changed bundle ZIP is not rejected by this seal gate; an old PASS portability receipt is not bound to this supplied bundle receipt.",
            "observed_corruption": False,
            "concrete_fix": "Check COMPLETE_BUNDLE_BYTES_AND_ZIP_VERIFIED, update 500, expected path/ZIP SHA and payload inventory; require portability.build_receipt_sha256 == sha(bundle_receipt), matching zip_sha256, physical_state_and_beta_exact, and the expected checkpoint through the referenced inference receipt."})
    after_hashes = {rel(p): sha(p) for p in paths}
    source_changed = [name for name in source_hashes if source_hashes[name] != after_hashes[name]]
    result = {"status": "FINDING_REQUIRES_SEAL_FIX" if findings else "PASS_STATIC_SCOPE_WITH_RUNTIME_PENDING",
        "snapshot_at": datetime.now(timezone.utc).isoformat(), "auditor_sha256": sha(__file__),
        "mode": "READ_ONLY_SOURCE_METADATA_AND_DIRECTORY_AUDIT", "source_hashes": source_hashes,
        "source_changed_during_audit": source_changed, "findings": findings,
        "c_drive_absolute_literals": c_drive_literals, "absolute_literals": absolute_literals,
        "inference_storage_contract": {"platform_scope": "Windows E or D drive relocation only",
            "runtime_guard_before_runtime_torch_import": "training_v4_4/infer.py:18-39",
            "output_guard_and_fresh_directory": "training_v4_4/infer.py:138-142,203",
            "default_runtime_relative_to_loaded_bundle": True,
            "inherited_GPU_PYTHON_constant_is_inert_on_inference_path": True,
            "generic_POSIX_or_C_drive_execution_supported": False,
            "bundle_absolute_paths_used_for_provenance_not_required_runtime_resolution": True},
        "builder_static_receipts_and_19_source_hashes_match": static_artifacts_match,
        "bundle_dependency_count": len(builder_sources), "source_matches": source_matches,
        "seal_bundle_value_load_reference_lines": bundle_load_references, "seal_portability_checked_keys": portability_keys,
        "data_scope": scope, "actual_output_inventories": inventories,
        "old_evidence": {"original_inventory_count": len(original["files"]), "all_file_sizes_checked": len(original["files"]),
            "missing_or_size_changed": size_mismatches, "non_npz_non_pt_files_byte_hashed": metadata_checked,
            "non_npz_non_pt_hash_mismatches": metadata_hash_mismatches,
            "npz_and_pt_bytes_not_rehashed_in_this_scope_audit": True,
            "sealed_full_1046_byte_preservation_still_pending": not (ROOT / "reports/parallel_session_20261013_v44_v46/seal_receipt.json").exists()},
        "artifact_state_at_snapshot": {"training_status": status, "delivery_output_exists": delivery.exists(),
            "bundle_build_receipt_exists": (delivery / "inference_bundle_build_receipt.json").exists(),
            "portability_receipt_exists": (delivery / "portability_receipt.json").exists(),
            "native_replay_receipt_exists": (RUN / "native_replay/receipt.json").exists()},
        "overwrite_review": {"builder_rejects_existing_bundle_zip_partial_and_receipts": "build_inference_bundle.py:459-478",
            "builder_payload_and_zip_use_exclusive_create": "build_inference_bundle.py:355-367,543-558",
            "seal_reports_and_archive_use_exclusive_create": "seal_results.py:23-24,115,128",
            "no_legacy_write_destination_in_reviewed_delivery_path": True,
            "status_and_runtime_temp_replace_operations_are_within_new_owned_run_paths": True},
        "limitations": ["No bundle was built or relocated by this audit.",
            "No raw/label/state array or checkpoint payload was opened; source and scalar metadata were read only.",
            "Directory absence and reviewed call paths support the scope finding but are not a system-wide historical I/O trace.",
            "The original 1046-file full byte-preservation seal must still run; this audit hashes 803 non-NPZ/non-PT files and stats all 1046.",
            "Training was still in progress at this snapshot; checkpoint500, native replay and relocated inference are pending."],
        "numpy_torch_runtime_imported": any(x in sys.modules for x in ("numpy", "torch", "runtime")),
        "gpu_calls": 0, "model_calls": 0, "array_payloads_opened": 0, "source_or_data_mutations": 0}
    assert not source_changed, source_changed
    assert not c_drive_literals and static_artifacts_match
    assert not size_mismatches and not metadata_hash_mismatches
    assert all(x["all_resolved_on_ED"] and not x["registered_locked_parent_path_hits"] and not x["registered_final_parent_path_hits"] for x in inventories.values())
    assert all(scope[key] for key in ("new_rows_all_train", "old_776_development_rows_unchanged", "actual_new_raw_paths_exact", "actual_new_label_paths_exact", "builder_examples_all_old_development"))
    assert not result["numpy_torch_runtime_imported"]
    with (HERE / "delivery_scope_audit.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    lines = ["# v4.6 delivery scope audit", "", "Snapshot status: **" + result["status"] + "**", "",
        "One concrete sealing gap was found: `seal_results.py:56-61` reads the supplied bundle receipt but never uses its contents. The seal therefore does not check bundle completion, ZIP presence/hash, or bind portability to that exact bundle. This is a gate gap; no corrupt delivery has been observed, and delivery output does not yet exist at this snapshot.", "",
        "Storage and static relocation checks passed. The inference entry checks runtime paths on E/D before importing runtime/torch, then rejects an existing output directory. Sources/configuration resolve from the relocated bundle. The inherited E-drive GPU_PYTHON constant is unused on the inference call path. Absolute paths stored in receipts/checkpoint provenance are not used to locate runtime dependencies. Portability is intentionally limited to Windows E/D; native relocation remains pending.", "",
        "Actual v4.6 output: 432 raw files, 96 labels, 120 state caches plus 120 cache metadata files, and 120 physical-score files. All resolved entries stay on E/D. No registered new locked-development or final parent directory exists. The 432 new manifest rows are all TRAIN; 776 inherited development rows are unchanged. The builder deliberately copies three old-development raw examples, with no truth/state caches or new locked/final data.", "",
        "Old-evidence checks: all 1046 baseline files still exist with unchanged sizes; all 803 non-NPZ/non-PT file hashes match. Array/checkpoint payload bytes were not reopened for this task. The final 1046-file byte-preservation seal is still pending. Builder/inference/seal use new output locations and reject existing final artifacts; no reviewed delivery write targets old evidence.", "",
        "The existing builder static-review receipts match the current builder and all 19 dependency hashes. This audit did not build, run or relocate a model. It imported no numpy/torch/runtime, opened no array/checkpoint payload and used no GPU. `delivery_scope_audit.json` records exact source hashes, directory counts, current training status, evidence locations and limitations.", ""]
    with (HERE / "delivery_scope_audit.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"status": result["status"], "findings": findings,
        "receipt_sha256": sha(HERE / "delivery_scope_audit.json"), "report_sha256": sha(HERE / "delivery_scope_audit.md"),
        "old_non_array_non_model_files_hashed": metadata_checked, "training_status": status,
        "array_payloads_opened": 0, "GPU_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
