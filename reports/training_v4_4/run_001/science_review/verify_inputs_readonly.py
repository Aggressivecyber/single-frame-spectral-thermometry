"""Standard-library-only independent v4.4 input audit; never imports the model.

Reads the registered 500 draws, all 776 development cases, their frozen raw,
labels, physical scores and cache files. Rehashes current disk bytes against
both the manifest/score and the prior immutable local-finalization receipt.
The only output is a new receipt under this script's E/D science_review folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class AuditFailure(RuntimeError):
    pass


def run(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    permitted_output = Path(__file__).resolve().parent
    if root.drive.upper() not in ("E:", "D:"):
        raise RuntimeError("Repository must be on E/D")
    if output.parent != permitted_output or output.drive.upper() not in ("E:", "D:"):
        raise RuntimeError("Receipt must stay in this E/D science_review directory")
    if output.exists():
        raise FileExistsError("Existing receipt is immutable: " + str(output))

    started = time.monotonic()
    receipt = {
        "schema": "thermometry-v4.4-independent-input-integrity-v1",
        "started_utc": now(), "root": str(root), "status": "RUNNING",
        "audit_source_path": str(Path(__file__).resolve()),
        "execution_scope": "CPU standard-library file reads and SHA256; no project imports, model inference, training or final-test access",
        "timing_interpretation": "Independent current-disk audit concurrent with this run; not evidence that the training runner performed these checks before starting.",
        "optimizer_updates": 0, "model_inferences": 0, "final_test_accesses": 0,
        "failures": [], "cases": [], "unique_files": {}, "preflight": {}
    }

    def require(condition, message):
        if not condition:
            print(json.dumps({"INPUT_AUDIT_FAILURE": message}), flush=True)
            raise AuditFailure(message)

    def resolve_relative(relative):
        relative_path = Path(relative)
        require(not relative_path.is_absolute(), "Unexpected absolute input path: " + str(relative))
        path = (root / relative_path).resolve()
        require(path.is_relative_to(root), "Input outside repository: " + str(relative))
        require("final_unopened" not in str(path), "Final-test input is forbidden: " + str(relative))
        return path

    def digest(path):
        before = path.stat()
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "File changed during hashing: " + str(path))
        return value.hexdigest(), after.st_size

    def verify_file(relative, expected):
        path = resolve_relative(relative)
        key = str(path.relative_to(root)).replace("\\", "/")
        previous = receipt["unique_files"].get(key)
        if previous is not None:
            require(previous["sha256"] == expected, "Conflicting frozen hashes: " + key)
            return previous["sha256"]
        actual, size = digest(path)
        require(actual == expected, "SHA256 mismatch: " + key)
        receipt["unique_files"][key] = {"sha256": actual, "bytes": size}
        return actual

    try:
        protocol_path = root / "reports/training_v4_4/run_001/protocol.json"
        manifest_relative = "reports/training_v4_1/run_001/data_manifest.json"
        draws_path = root / "reports/training_v4_1/run_001/sampling_sequence.json"
        previous_path = root / "reports/local_session_20261013/input_hash_receipt.json"
        selection_path = root / "reports/training_v4/run_001/backbone_selection.json"
        protocol = read_json(protocol_path)
        receipt["protocol_sha256"] = digest(protocol_path)[0]
        manifest_sha = verify_file(manifest_relative, protocol["manifest_sha256"])
        manifest = read_json(root / manifest_relative)
        rows = manifest["rows"]
        by_id = {r["case_id"]: r for r in rows}
        require(len(by_id) == len(rows), "Duplicate IDs in frozen manifest")
        validation = [r["case_id"] for r in rows if r["split"] == "validation"]
        require(len(validation) == 776, "Wrong full validation count")
        require(canonical_hash(validation) == protocol["validation_ids_sha256"],
                "Validation ID hash differs from registered protocol")
        draws = read_json(draws_path)["draws"][:500]
        require(len(draws) == 500 and all(len(batch) == 4 for batch in draws),
                "Draw budget/accumulation differs from 500 x 4")
        require(canonical_hash(draws) == protocol["draws_sha256"],
                "Draw prefix hash differs from registered protocol")
        train_ids = [case_id for batch in draws for case_id in batch]
        require(all(by_id[i]["split"] == "train" for i in train_ids),
                "Non-training input in optimizer draw stream")
        used = list(dict.fromkeys(train_ids + validation))
        require(not set(train_ids).intersection(validation), "Train/validation case overlap")
        previous_receipt = read_json(previous_path)
        previous = {r["case_id"]: r for r in previous_receipt["cases"]}
        require(all(i in previous for i in used), "Used case absent from prior frozen input receipt")
        selection = read_json(selection_path)
        config_hash = canonical_hash(selection["physics_config"])
        solver_hash = canonical_hash(selection["solver_config"])
        receipt["preflight"] = {
            "manifest_sha256": manifest_sha, "manifest_matches_protocol": True,
            "validation_ids_sha256": canonical_hash(validation), "validation_ids_match_protocol": True,
            "draw_prefix_sha256": canonical_hash(draws), "draw_prefix_matches_protocol": True,
            "optimizer_draws": len(draws), "microbatch_positions": len(train_ids),
            "unique_train_cases": len(set(train_ids)), "validation_cases": len(validation),
            "used_unique_cases": len(used), "train_validation_case_sets_disjoint": True,
            "prior_input_receipt_sha256": digest(previous_path)[0],
            "backbone_selection_sha256": digest(selection_path)[0],
            "physics_config_sha256": config_hash, "solver_config_sha256": solver_hash,
            "all_cases_have_prior_frozen_receipt": True
        }
        print(json.dumps({"input_audit_preflight": "PASS", **receipt["preflight"]}), flush=True)

        for index, case_id in enumerate(used):
            row = by_id[case_id]
            old = previous[case_id]
            score_relative = "outputs/training_v4_1/run_001/physical_scores/" + case_id + ".json"
            score_sha = verify_file(score_relative, old["score_sha256"])
            score = read_json(resolve_relative(score_relative))
            require(score["case_id"] == case_id, "Physical score case ID mismatch: " + case_id)
            require(score["raw_hash"] == row["raw_hash"] == old["raw_hash"],
                    "Raw identity mismatch: " + case_id)
            require(score["label_sha256"] == row["label_sha256"],
                    "Physical-score label identity mismatch: " + case_id)
            require(score["cache_key"] == old["cache_key"], "Cache key mismatch: " + case_id)
            cache_relative = score["cache_path"]
            metadata_relative = str(Path(cache_relative).with_suffix(".json")).replace("\\", "/")
            files = {row["raw_path"]: row["raw_file_sha256"],
                     row["label_path"]: row["label_sha256"],
                     cache_relative: score["cache_sha256"],
                     metadata_relative: score["cache_metadata_sha256"]}
            require(files == old["files"], "Input hash declaration differs from frozen receipt: " + case_id)
            for relative, expected in files.items():
                verify_file(relative, expected)
            metadata = read_json(resolve_relative(metadata_relative))
            require(metadata["cache_key"] == score["cache_key"] and
                    metadata["raw_hash"] == row["raw_hash"] and
                    metadata["cache_sha256"] == score["cache_sha256"],
                    "Cache metadata identity mismatch: " + case_id)
            require(metadata["physics_config_sha256"] == config_hash and
                    metadata["calibration_sha256"] == row["calibration_sha256"] and
                    metadata["solver_config_sha256"] == solver_hash and
                    metadata["method"] == selection["method"] and
                    metadata["labels_used"] is False,
                    "Cache physics/calibration/solver metadata mismatch: " + case_id)
            key = score["cache_key"]
            candidate_paths = [root / prefix / "cache" / key[:2] / (key + ".npz")
                for prefix in ("outputs/training_v4/run_001", "outputs/training_v4_1/run_001")]
            actual_load_path = next((p for p in candidate_paths if p.with_suffix(".json").exists()), None)
            require(actual_load_path is not None and actual_load_path.resolve() == resolve_relative(cache_relative),
                    "Runtime cache_location resolves a different file: " + case_id)
            receipt["cases"].append({"case_id": case_id, "split": row["split"],
                "raw_hash": row["raw_hash"], "cache_key": key,
                "physical_score_sha256": score_sha, "files": files,
                "declarations_match_prior_frozen_receipt": True, "all_current_bytes_match": True,
                "runtime_cache_location_matches_score": True})
            if (index + 1) % 250 == 0 or index + 1 == len(used):
                print(json.dumps({"input_cases_verified": index + 1, "total": len(used),
                    "unique_files_hashed": len(receipt["unique_files"])}), flush=True)

        # Cheap registration rechecks prevent silently using a changed manifest or
        # protocol during this audit; input files are immutable by project policy.
        require(digest(protocol_path)[0] == receipt["protocol_sha256"], "Protocol changed during audit")
        require(digest(root / manifest_relative)[0] == manifest_sha, "Manifest changed during audit")
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        receipt["finished_utc"] = now()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["checked_cases"] = len(receipt["cases"])
        receipt["unique_file_count"] = len(receipt["unique_files"])
        receipt["unique_bytes_hashed"] = sum(r["bytes"] for r in receipt["unique_files"].values())
        receipt["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": receipt["status"], "checked_cases": receipt["checked_cases"],
            "unique_file_count": receipt["unique_file_count"],
            "unique_bytes_hashed": receipt["unique_bytes_hashed"],
            "elapsed_seconds": receipt["elapsed_seconds"], "receipt": str(output),
            "failures": receipt["failures"]}), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
        default=Path("E:/CodexWork/thermometry/public_repository_001"))
    parser.add_argument("--output", type=Path,
        default=Path(__file__).resolve().with_name("input_integrity_receipt.json"))
    args = parser.parse_args()
    sys.exit(run(args.root, args.output))
