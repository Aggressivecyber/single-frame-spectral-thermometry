"""Recompute v4.5 screen aggregates from saved metrics without model execution."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "reports/training_v4_5/run_001"
RUN = BASE / "balanced_fixed_continuum"
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    protocol_hash = file_hash(BASE / "protocol.json")
    old_rows = read(ROOT / "reports/training_v4_1/run_001/data_manifest.json")["rows"]
    validation_ids = [r["case_id"] for r in old_rows if r["split"] == "validation"]
    assert len(validation_ids) == 776
    curves = lines(RUN / "training_curve.jsonl")
    control = lines(ROOT / "reports/training_v4_3/run_001/negative_balanced/training_curve.jsonl")
    assert len(curves) == len(control) == 500
    assert [r["optimizer_update"] for r in curves] == list(range(1, 501))
    assert all(a["samples"] == b["samples"] for a, b in zip(curves, control))
    status = read(RUN / "status.json")
    training = read(RUN / "training_receipt.json")
    assert status["stage"] == "COMPLETE" and training["updates"] == 500
    checkpoints = []
    for item in training["checkpoints"]:
        assert file_hash(ROOT / item["path"]) == item["sha256"]
        assert item["Adam_steps"] == ([] if item["update"] == 0 else [item["update"]])
        checkpoints.append({"update": item["update"], "file_sha256_matches_receipt": True,
                            "declared_Adam_steps": item["Adam_steps"]})
    records = []
    for update in range(0, 501, 100):
        case_path = RUN / f"validation_update_{update:04d}.jsonl"
        rows = lines(case_path)
        summary = read(RUN / f"validation_summary_{update:04d}.json")
        science = read(RUN / f"science_summary_{update:04d}.json")
        assert [r["case_id"] for r in rows] == validation_ids
        assert len(rows) == 776 and all(r["optimizer_update"] == update and r["status"] == "SUCCESS" for r in rows)
        assert all(r["invalid_output_fraction"] == 0 and r["temperature_correction_RMS_K"] == 0 for r in rows)
        assert all(math.isfinite(v) for r in rows for v in r["metrics"].values() if v is not None)
        assert science["validation_sha256"] == file_hash(case_path) and science["protocol_sha256"] == protocol_hash
        assert science["old_eligible"] == summary["eligible"]
        signed = []
        for group in science["signed_noisy_conditions"]:
            subset = [r for r in rows if r["signed_aux"] and r["condition"] == group["condition"] and r["input_kind"] == "noise"]
            assert len(subset) == 6
            sse = sum(r["metrics"]["template_SSE"] for r in subset)
            pixels = sum(r["metrics"]["pixels"] for r in subset)
            energy = sum(r["metrics"]["truth_energy"] for r in subset)
            mse, l2 = sse / pixels, math.sqrt(sse / energy)
            physical_mse = sum(r["physical"]["template_MSE_e2"] for r in subset) / len(subset)
            limit = 1.05 * physical_mse + .25
            for a, b in ((mse, group["template_MSE_e2"]), (l2, group["template_L2"]), (limit, group["MSE_limit_e2"])):
                assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            mse_pass = mse <= limit
            weak_pass = not group["condition"].endswith("weak") or l2 < 1
            assert mse_pass == group["MSE_pass"] and weak_pass == group["weak_L2_pass"]
            assert group["passed"] == (mse_pass and weak_pass)
            signed.append({"condition": group["condition"], "RMSE_e": math.sqrt(mse),
                           "RMSE_limit_e": math.sqrt(limit), "pooled_L2": l2, "passed": group["passed"]})
        assert science["new_development_screen_pass"] == all(x["passed"] for x in signed)
        assert science["science_candidate"] == (summary["eligible"] and science["new_development_screen_pass"])
        records.append({"update": update, "old_eligible": summary["eligible"], "science_candidate": science["science_candidate"],
                        "main_weak_L2": summary["weak_L2"], "validation_records": 776,
                        "all_saved_metrics_finite": True, "invalid_output_fraction_max": 0.,
                        "temperature_correction_RMS_K_max": 0., "signed_noisy": signed})
    selection = read(BASE / "candidate_selection.json")
    assert sum(x["science_candidate"] for x in records) == selection["candidate_count"] == 0
    assert selection["selected"] is None
    result = {"status": "PASS_SAVED_METRICS_RECOMPUTED", "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": file_hash(Path(__file__)), "protocol_sha256": protocol_hash,
        "training_receipt_sha256": file_hash(RUN / "training_receipt.json"),
        "selection_sha256": file_hash(BASE / "candidate_selection.json"),
        "training_updates": 500, "microbatches": 2000, "draw_sample_records_match_old_balanced": True,
        "checkpoint_files": checkpoints, "checkpoints": records, "candidate_count": 0,
        "new_optimizer_or_model_execution": False, "array_reads": 0, "gpu_calls": 0,
        "checkpoint_payload_loaded": False, "native_replay_independently_rerun": False,
        "claim_limit": "Recomputes aggregate screens from saved scalar scores; does not independently validate the score_arrays model-output calculation.",
        "final_test_used": False}
    target = OUT / "saved_screen_audit.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read(target) == result
    print(json.dumps({"status": result["status"], "candidate_count": 0, "receipt_sha256": file_hash(target), "endpoint": records[-1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
