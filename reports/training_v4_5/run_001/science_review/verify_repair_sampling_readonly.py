"""Independent metadata-only audit of the frozen v4.6 sampling design.

No array, model, registration, generation, cache, or training operation is used.
The only write is the audit receipt alongside this source file.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def main():
    assert ROOT.drive.upper() in ("E:", "D:")
    sources = [ROOT / "training_v4_6/protocol.py", ROOT / "training_v4_6/data.py"]
    sources_before = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources}
    from training_v4_6.protocol import planned_sampling
    from training_v4_6.data import new_case_specs
    sampling = planned_sampling()
    specs = new_case_specs()
    old = read(ROOT / "reports/training_v4_1/run_001/data_manifest.json")
    original = read(ROOT / "reports/training_v4_3/run_001/negative_balanced/sampling_sequence.json")["draws"]
    original_flat = [c for b in original for c in b]
    draws = sampling["draws"]
    current_flat = [c for b in draws for c in b]
    by = {r["case_id"]: r for r in old["rows"] + specs}
    old_ids = {r["case_id"] for r in old["rows"]}
    new_ids = {r["case_id"] for r in specs}
    registry = read(ROOT / "reports/training_v4_4/run_001/data_audit/repair_parent_registry.json")
    mothers = {m["mother_id"]: m for m in old["signed_mothers"] if m["split"] == "train"}
    mothers.update({m["mother_id"]: m for m in registry["new_parents"] if m["split"] == "train"})
    changed_positions = []
    eligible_positions = []
    unmodified_positions = []
    assert len(original) == len(draws) == 500
    assert all(len(b) == 4 for b in draws)
    assert len(original_flat) == len(current_flat) == 2000
    for i, (a, b) in enumerate(zip(original_flat, current_flat)):
        ra, rb = by[a], by[b]
        eligible = ra["signed_aux"] and ra["condition"] in ("negative_weak", "negative_normal") and ra["input_kind"] == "noise"
        assert rb["split"] == "train"
        assert (ra["condition"], ra["input_kind"], ra["signed_aux"]) == (rb["condition"], rb["input_kind"], rb["signed_aux"])
        if eligible:
            eligible_positions.append(i)
        else:
            assert a == b
            unmodified_positions.append(i)
        if a != b:
            assert eligible
            changed_positions.append(i)
    assert len(eligible_positions) == 513 and len(unmodified_positions) == 1487
    assert sum(by[c]["signed_aux"] for c in original_flat) == 575
    assert sum(not by[c]["signed_aux"] for c in original_flat) == 1425
    assert len(specs) == 432 and len(new_ids) == 432 and not old_ids & new_ids
    assert Counter(r["input_kind"] for r in specs) == {"continuous": 96, "quantization": 96, "noise": 240}
    assert {r["split"] for r in specs} == {"train"}
    new_used = set(current_flat) & new_ids
    assert len(new_used) == 120 and new_used == set(sampling["new_unique_cases_to_cache"])
    assert all(by[c]["condition"].startswith("negative_") and by[c]["input_kind"] == "noise" for c in new_used)
    exposure = []
    for condition in ("negative_weak", "negative_normal"):
        selected = [by[c] for c in current_flat if by[c]["signed_aux"] and by[c]["condition"] == condition and by[c]["input_kind"] == "noise"]
        counts = Counter(r["case_id"] for r in selected)
        mids = {r["mother_id"] for r in selected}
        pairs = {(mothers[m]["temperature_geometry"], mothers[m]["template_geometry"]) for m in mids}
        assert len(counts) == 72 and len(mids) == 36 and len(pairs) == 36
        assert min(counts.values()) == 3 and max(counts.values()) == 4
        variants_by_mother = {m: {0 if r.get("noise_index") is None else r["noise_index"] for r in selected if r["mother_id"] == m} for m in mids}
        assert all(indices == {0, 1} for indices in variants_by_mother.values())
        exposure.append({"condition": condition, "draws": len(selected), "unique_cases": len(counts),
                         "unique_mothers": len(mids), "geometry_pairs": len(pairs),
                         "noise_realisations_per_mother": 2, "minimum_case_draws": min(counts.values()),
                         "maximum_case_draws": max(counts.values()), "frequency_histogram": {str(k): v for k, v in Counter(counts.values()).items()}})
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    sources_after = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources}
    assert sources_before == sources_after
    assert canonical(draws) == sampling["draw_sha256"]
    receipt = {"status": "PASS_METADATA_ONLY", "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent descriptor and positional audit; no arrays, model or execution actions.",
        "source_sha256": sources_before, "source_files_stable_during_audit": True,
        "verifier_source_sha256": file_hash(Path(__file__)), "draw_sha256": canonical(draws),
        "source_draw_sha256": canonical(original), "updates": len(draws), "microbatches": len(current_flat),
        "unsigned_microbatches": 1425, "signed_microbatches": 575,
        "negative_noisy_positions_resampled": len(eligible_positions),
        "actual_case_ids_changed": len(changed_positions),
        "other_positions_same_case_id": len(unmodified_positions),
        "new_train_case_specs": len(specs), "new_used_unique_cases": len(new_used),
        "new_unused_case_specs": len(new_ids - new_used), "negative_noisy_exposure": exposure,
        "all_other_signed_positions_retained": 62,
        "new_positive_or_clean_cases_sampled": 0,
        "training_run": False, "model_inference_run": False, "torch_imported": False,
        "numpy_imported": False, "array_reads": 0, "gpu_calls": 0,
        "registration_called": False, "generator_called": False, "cache_called": False,
        "new_locked_dev_generated": False, "final_test_used": False}
    target = OUT / "data_repair_sampling_audit.json"
    target.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read(target) == receipt
    print(json.dumps({k: receipt[k] for k in ("status", "draw_sha256", "negative_noisy_positions_resampled", "actual_case_ids_changed", "other_positions_same_case_id", "new_used_unique_cases", "negative_noisy_exposure")}, ensure_ascii=False))
    print(json.dumps({"receipt_path": target.relative_to(ROOT).as_posix(), "receipt_sha256": file_hash(target)}))


if __name__ == "__main__":
    main()
