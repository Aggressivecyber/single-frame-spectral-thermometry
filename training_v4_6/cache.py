"""Single-worker physical caches for only the new TRAIN inputs actually drawn.

The original scalar inverse, camera configuration and physical cache identity
are retained. Labels are opened only after inversion, for scoring sidecars.
"""
from pathlib import Path
import argparse
import json
import time
import traceback
from training_v4_6 import data

ROOT, OUT, DATA = data.ROOT, data.OUT, data.DATA


def selected_new_case_ids(draws, new_rows):
    """Stable first-use order, deduplicated; no arrays or project code imports."""
    new_ids = {r["case_id"] for r in new_rows}
    seen, selected = set(), []
    for batch in draws:
        for case in batch:
            if case in new_ids and case not in seen:
                selected.append(case)
                seen.add(case)
    return selected


def key_for(row, selection):
    from training_v4.cache import cache_identity
    return cache_identity(row["raw_hash"], selection["physics_config"], row["calibration_sha256"],
                          selection["method"], selection["solver_config"])


def _cache_status(stage, **values):
    data.write_json(OUT / "cache_status.json", {"stage": stage, "time": data.now(), **values}, replace=True)


def _verify_cache_identity(row, selection, key, path, meta, score):
    """Validate identities before cache arrays or training labels are opened."""
    config_hash = data.canonical_hash(selection["physics_config"])
    data.require(score["cache_key"] == key and meta["cache_key"] == key
                 and score["raw_hash"] == meta["raw_hash"] == row["raw_hash"]
                 and score["label_sha256"] == row["label_sha256"], "New TRAIN cache/score identity mismatch")
    data.require(data.file_hash(path.with_suffix(".json")) == score["cache_metadata_sha256"],
                 "New TRAIN cache metadata drift")
    data.require(meta["physics_config_sha256"] == row["physics_config_sha256"] == config_hash
                 and meta["calibration_sha256"] == row["calibration_sha256"]
                 and meta["solver_config_sha256"] == data.canonical_hash(selection["solver_config"])
                 and meta["method"] == selection["method"]
                 and meta["engine_sha256"] == data.file_hash(ROOT / "training_v4/scalar_engine.py"),
                 "New TRAIN physical cache configuration drift")
    data.require(meta["labels_used"] is False and score["labels_enter_inversion"] is False
                 and score["truth_used_for_scoring_only"] is True, "New TRAIN cache label-use contract changed")
    data.require(score["identity"] == data.canonical_hash(
        {"key": key, "label": row["label_sha256"], "scores_schema": "v4_1_output_MSE"}),
        "New TRAIN physical score identity drift")
    data.require(data.file_hash(path) == meta["cache_sha256"] == score["cache_sha256"],
                 "New TRAIN cache file drift")


def _cache_one(physics, row, selection, *, root=ROOT, data_dir=DATA, inverse=None, output_fn=None, key_fn=None):
    """One scalar inverse, then scoring; CPU mock tests inject tiny pure stubs."""
    import numpy as np
    import torch
    p = physics
    if inverse is None:
        from training_v4.scalar_engine import invert_single
        inverse = invert_single
    if output_fn is None:
        from training_v4.model import physical_output
        output_fn = physical_output
    key_fn = key_for if key_fn is None else key_fn
    root, data_dir = Path(root), Path(data_dir)
    data.require(row["split"] == "train" and row["origin"].startswith("v4_6_"), "Only newly generated TRAIN rows may be cached")
    data.require(data.canonical_hash(p.cfg) == data.canonical_hash(selection["physics_config"]), "Physics configuration mismatch")
    data.require(row["physics_config_sha256"] == data.canonical_hash(p.cfg), "Row physics identity mismatch")
    raw_path = data.rooted(row["raw_path"], root)
    data.require(raw_path.is_relative_to(data_dir.resolve()), "New raw must be in the v4.6 output directory")
    data.require(data.file_hash(raw_path) == row["raw_file_sha256"], "New raw file hash mismatch")
    raw_arrays = data.read_arrays(raw_path)
    data.require(set(raw_arrays) == {"raw_dn"}, "Expected only a raw_dn measurement array")
    raw_array = raw_arrays["raw_dn"]
    data.require(data.array_hash(raw_array) == row["raw_hash"], "Raw array identity mismatch")
    data.require(tuple(raw_array.shape) == tuple(p.cfg["shape"]) and np.isfinite(raw_array).all(), "Raw shape/nonfinite mismatch")
    data.require(not (raw_array >= 4095).any(), "Saturated frame is retained as failure")
    cache_key = key_fn(row, selection)
    cache_path = data_dir / "cache" / cache_key[:2] / (cache_key + ".npz")
    score_path = data_dir / "physical_scores" / (row["case_id"] + ".json")
    data.require(not cache_path.exists() and not cache_path.with_suffix(".json").exists() and not score_path.exists(),
                 "A cache/score already exists; no automatic overwrite or retry")
    raw = torch.as_tensor(np.array(raw_array, dtype=np.float32, copy=True), device=p.M.device)[None]
    raw_e = p.raw_to_electrons(raw)
    # No label value/path enters the inverse function's arguments.
    state, beta, history, seconds = inverse(p, raw_e, selection["method"], selection["solver_config"])
    data.require(len(state) == 1 and bool(torch.isfinite(state).all()) and bool(torch.isfinite(beta).all()),
                 "Nonfinite scalar inverse output")
    saved = data.save_arrays(cache_path, {"state": state[0].detach().cpu().numpy(), "beta": beta[0].detach().cpu().numpy()})
    meta = {"cache_key": cache_key, "raw_hash": row["raw_hash"], "cache_sha256": saved["file_sha256"],
        "physics_config_sha256": row["physics_config_sha256"], "calibration_sha256": row["calibration_sha256"],
        "solver_config_sha256": data.canonical_hash(selection["solver_config"]),
        "method": selection["method"], "seconds": seconds, "history": history[0],
        "engine_sha256": data.file_hash(ROOT / "training_v4/scalar_engine.py"),
        "labels_used": False, "array_reopen_exact": saved["array_reopen_exact"],
        "cache_path": data.rel(cache_path, root)}
    data.write_json(cache_path.with_suffix(".json"), meta)
    # Open a TRAIN label only after the raw-only inverse and cache are durable.
    label_path = data.rooted(row["label_path"], root)
    is_new_label = label_path.is_relative_to((data_dir / "labels").resolve())
    old_label = root / "outputs/training_v4_1/run_001/labels" / row["mother_id"] / (row["condition"] + ".npz")
    data.require(is_new_label or (row["mother_id"].startswith("v4_1_signed_train_mother_") and label_path == old_label.resolve()),
                 "Only registered TRAIN labels may be scored")
    data.require(data.file_hash(label_path) == row["label_sha256"], "TRAIN scoring label hash mismatch")
    truth = data.read_arrays(label_path)
    data.require(all(data.array_hash(truth[key]) == digest for key, digest in row["label_array_sha256"].items()),
                 "Scoring label arrays changed")
    with torch.no_grad():
        out = output_fn(p, state, beta, raw_e)
        template = torch.as_tensor(truth["template_e"], device=p.M.device)[None]
        temperature = torch.as_tensor(truth["temperature_color_K"], device=p.M.device)[None]
        metrics = {"template_MSE_e2": float((out["template_e"] - template).square().mean()),
                   "temperature_MSE_K2": float((out["temperature_color_K"] - temperature).square().mean()),
                   "weighted_raw_RMS": float((out["raw_residual_e"].square() / (raw_e.clamp_min(0) + p.extra_variance)).mean().sqrt())}
    data.require(all(np.isfinite(value) for value in metrics.values()), "Nonfinite physical scoring sidecar")
    score = {"identity": data.canonical_hash({"key": cache_key, "label": row["label_sha256"], "scores_schema": "v4_1_output_MSE"}),
        "case_id": row["case_id"], "raw_hash": row["raw_hash"], "label_sha256": row["label_sha256"],
        "cache_key": cache_key, "cache_path": data.rel(cache_path, root),
        "cache_sha256": saved["file_sha256"], "cache_metadata_sha256": data.file_hash(cache_path.with_suffix(".json")),
        **metrics, "truth_used_for_scoring_only": True, "labels_enter_inversion": False}
    data.write_json(score_path, score)
    return {**score, "score_path": data.rel(score_path, root), "score_sha256": data.file_hash(score_path),
            "seconds": seconds, "status": "SUCCESS", "split": "train", "reused_old_cache": False}


def _formal_cache(protocol, bundle, gate):
    generation = data.read(OUT / "generation_receipt.json")
    data.require(generation.get("stage") == "COMPLETE" and generation["protocol_sha256"] == gate["protocol_sha256"],
                 "The matching TRAIN generation must be COMPLETE first")
    for key, name in (("new_rows_sha256", "new_rows.json"), ("merged_manifest_sha256", "data_manifest.json")):
        data.require(data.file_hash(OUT / name) == generation[key], "Generated manifest identity changed")
    new_rows = data.read(OUT / "new_rows.json")["rows"]
    data.require(len(new_rows) == 432 and all(r["split"] == "train" for r in new_rows), "Expected exactly 432 new TRAIN rows")
    draws = data.read(data.SAMPLING)["draws"]
    selected = selected_new_case_ids(draws, new_rows)
    data.require(selected, "No new TRAIN input is present in the frozen draws")
    if "expected_new_cache_cases" in protocol:
        data.require(len(selected) == protocol["expected_new_cache_cases"], "Actual cache selection differs from protocol")
    data.require(not (OUT / "cache_status.json").exists() and not (OUT / "cache_per_case.jsonl").exists(),
                 "Caching was already started; preserve results and do not extend its budget automatically")
    by_id = {r["case_id"]: r for r in new_rows}
    source_hashes = dict(protocol["frozen_inputs_sha256"])
    started, records, current_case = time.monotonic(), [], None
    _cache_status("INITIALIZING", selected_case_count=len(selected), completed=0)
    try:
        data.configure_runtime(DATA)
        import torch
        from infer import load_model
        torch.set_num_threads(protocol["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = load_model(device="cuda"); p = model.physics
        selection = data.read(data.SELECTION)
        data.require(data.canonical_hash(p.cfg) == data.canonical_hash(selection["physics_config"]), "Fixed sensor physics changed")
        before = {name: data.array_hash(value.detach().cpu().numpy()) for name, value in p.named_buffers()}
        for index, case in enumerate(selected):
            current_case = case
            data.deadline_check(started + protocol["maximum_cache_seconds"])
            _cache_status("MATERIALIZING", selected_case_count=len(selected), completed=index,
                          current_case=case, elapsed_s=time.monotonic() - started)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                record = _cache_one(p, by_id[case], selection)
            record["peak_GPU_allocated_MiB"] = float(torch.cuda.max_memory_allocated() / 2**20)
            data.append_jsonl(OUT / "cache_per_case.jsonl", record)
            records.append(record)
            data.deadline_check(started + protocol["maximum_cache_seconds"])
        data.require(before == {name: data.array_hash(value.detach().cpu().numpy()) for name, value in p.named_buffers()},
                     "Physical buffers changed during caching")
        data.verify_execution("cache_drawn_new_train", gate["protocol_sha256"])
        receipt = {"stage": "COMPLETE", "time": data.now(), **gate,
            "selected_case_count": len(selected), "selected_case_ids": selected,
            "selected_case_ids_sha256": data.canonical_hash(selected), "cache_record_count": len(records),
            "all_selected_new_cases_materialized": len(records) == len(selected),
            "new_rows_sha256": generation["new_rows_sha256"], "merged_manifest_sha256": generation["merged_manifest_sha256"],
            "cache_per_case_sha256": data.file_hash(OUT / "cache_per_case.jsonl"),
            "elapsed_s": time.monotonic() - started, "source_hashes_at_start": source_hashes,
            "physical_buffers_unchanged": True, "labels_enter_inversion": False,
            "old_cache_files_written": 0, "unselected_new_cases_materialized": 0,
            "old_validation_arrays_opened": 0, "locked_development_arrays_generated_or_opened": 0,
            "final_test_arrays_or_results_opened": 0, "coefficient_recalibration_performed": False,
            "workers": 1, "cpu_threads": protocol["cpu_threads"], "torch_version": torch.__version__}
        data.write_json(OUT / "cache_receipt.json", receipt)
        _cache_status("COMPLETE", selected_case_count=len(selected), completed=len(records), elapsed_s=receipt["elapsed_s"])
        return receipt
    except BaseException as exc:
        data.write_json(OUT / "cache_failure.json", {"time": data.now(), "case_id": current_case,
            "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
            "completed": len(records), "selected_case_count": len(selected), "elapsed_s": time.monotonic() - started,
            "failed_files_deleted": False, "replacement_seeds": 0, "budget_automatically_extended": False})
        _cache_status("FAILED", completed=len(records), selected_case_count=len(selected), message=str(exc))
        raise


def load(row, selection, labels=True):
    """Route old rows read-only to v4.1; new TRAIN rows use their v4.6 caches."""
    if not row.get("origin", "").startswith("v4_6_"):
        from training_v4_1.cache import load as old_load
        return old_load(row, selection, labels=labels)
    import numpy as np
    import torch
    data.require(row["split"] == "train", "v4.6 introduced TRAIN inputs only")
    key = key_for(row, selection)
    path = DATA / "cache" / key[:2] / (key + ".npz")
    meta = data.read(path.with_suffix(".json"))
    score = data.read(DATA / "physical_scores" / (row["case_id"] + ".json"))
    _verify_cache_identity(row, selection, key, path, meta, score)
    raw_path = data.rooted(row["raw_path"])
    data.require(data.file_hash(raw_path) == row["raw_file_sha256"], "New TRAIN raw file drift")
    raw = data.read_arrays(raw_path)["raw_dn"]
    data.require(data.array_hash(raw) == row["raw_hash"], "New TRAIN raw array drift")
    cached = data.read_arrays(path)
    inputs = (torch.as_tensor(raw.astype(np.float32), device="cuda")[None],
              torch.as_tensor(cached["state"], device="cuda")[None], torch.as_tensor(cached["beta"], device="cuda")[None])
    if not labels:
        return inputs, score
    label_path = data.rooted(row["label_path"])
    data.require(data.file_hash(label_path) == row["label_sha256"], "New TRAIN label file drift")
    target = data.read_arrays(label_path)
    truth = {key: torch.as_tensor(target[key], device="cuda")[None]
             for key in ("template_e", "temperature_color_K", "band_images_e")}
    return inputs, truth, score


def materialize_selected(*, protocol_sha256):
    protocol, bundle, gate = data.verify_execution("cache_drawn_new_train", protocol_sha256)
    return _formal_cache(protocol, bundle, gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    data.require(args.execute, "Caching requires explicit --execute and a frozen protocol hash")
    print(json.dumps(materialize_selected(protocol_sha256=args.protocol_sha256), ensure_ascii=False))


if __name__ == "__main__":
    main()
