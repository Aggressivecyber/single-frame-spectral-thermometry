"""Replay frozen outputs and add descriptive diagnostics; no training or selection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import runtime
import collections
import json
import time
import numpy as np
import torch
from research_joint_imaging.core import Imager, Radiometry
from research_joint_imaging_v2.evaluate import model_for, inference, rows_for
from research_joint_imaging_v2.losses import readouts

OUT = Path(__file__).resolve().parent
RUN = ROOT / "reports/joint_imaging_v2_recovery_002"
protocol = json.loads((ROOT / "research_joint_imaging_v2/protocol.json").read_text())
receipt = json.loads((RUN / "DATA_RECEIPT.json").read_text())
registration = json.loads((ROOT / "research_joint_imaging_v2/REGISTRATION.json").read_text())
summary = json.loads((RUN / "SUMMARY.json").read_text())
frozen = json.loads((RUN / "THRESHOLDS_FROZEN.json").read_text())
selection = json.loads((RUN / "SELECTION_FROZEN.json").read_text())
start = time.monotonic()
assert json.loads((RUN / "EXIT.json").read_text())["returncode"] == 0
assert selection["time"] < frozen["time"] < summary["time"]
for name, digest in registration["source_sha256"].items():
    assert runtime.sha256(ROOT / name) == digest, name
assert runtime.sha256(ROOT.parent / "joint_imaging_v2_recovery_002.tar.gz") == "08ada8a19b911d16822c5a25420c9a543d10da554cc8a644cfde328c559fc013"
source_manifest = json.loads((RUN / "OUTPUT_MANIFEST.json").read_text())
for name, item in source_manifest.items():
    assert runtime.sha256(RUN / name) == item["sha256"], name
for item in receipt["mothers"]:
    assert runtime.sha256(RUN / item["file"]) == item["sha256"], item["file"]
split_ids = {split: {p["mother"] for p in receipt["mothers"] if p["split"] == split}
             for split in ("train", "validation", "calibration", "heldout")}
assert [len(v) for v in split_ids.values()] == [64, 16, 8, 16]
assert len(set.union(*split_ids.values())) == 104
assert sum(item["observations"] for item in receipt["mothers"]) == 2496
training = {}
for method in protocol["learning"]["methods"]:
    rows = [json.loads(line) for line in (RUN / (method + "_training.jsonl")).read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 6001)), method
    counts = dict(collections.Counter(r["phase"] for r in rows))
    assert counts == {"thermal": 1500, "visible": 1500, "joint": 3000}, (method, counts)
    validation = json.loads((RUN / (method + "_validation.json")).read_text())
    assert [v["step"] for v in validation["rows"]] == list(range(200, 6001, 200))
    best = min(validation["rows"], key=lambda v: v["selection_score"])
    checkpoint = torch.load(RUN / (method + "_best.pt"), map_location="cpu", weights_only=False)
    assert checkpoint["method"] == method and checkpoint["seed"] == 811
    assert checkpoint["step"] == best["step"] == validation["best_step"]
    assert runtime.sha256(RUN / (method + "_best.pt")) == frozen["checkpoint_sha256"][method]
    assert max(r["time"] for r in rows) < selection["time"]
    training[method] = {"updates": len(rows), "phase_counts": counts, "best_step": best["step"],
                        "best_validation_score": best["selection_score"],
                        "seed": checkpoint["seed"], "checkpoint_sha256": frozen["checkpoint_sha256"][method]}
runtime.write_json(OUT / "PROVENANCE_AUDIT.json", {"time": runtime.now(), "source_files_verified": len(registration["source_sha256"]),
                   "cache_files_verified": len(receipt["mothers"]), "all_original_output_hashes_verified": True,
                   "cache_bytes": sum(p["bytes"] for p in receipt["mothers"]), "raw_observations": 2496,
                   "split_counts": {k: len(v) for k, v in split_ids.items()}, "training": training,
                   "selection_then_threshold_then_heldout_summary": True,
                   "selection_time": selection["time"], "threshold_time": frozen["time"],
                   "original_completion_time": json.loads((RUN / "COMPLETION.json").read_text())["time"]})

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.cuda.set_per_process_memory_fraction(9500 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
device = "cuda"
op, rad = Imager(protocol["model"]).to(device), Radiometry(protocol, device)
methods = ["tv", "tv_400", *protocol["learning"]["methods"]]
heldout = [p for p in receipt["mothers"] if p["split"] == "heldout"]
calibration = [p for p in receipt["mothers"] if p["split"] == "calibration"]
all_rows, native = {}, {}
with torch.no_grad():
    for method in methods:
        model = model_for(method, protocol, RUN, device)
        scores = []
        for item in calibration:
            data = torch.load(RUN / item["file"], map_location="cpu", weights_only=False)
            ids = [i for i, r in enumerate(data["metadata"]) if r["condition"] == "blank"]
            tensors = {k: v[ids] for k, v in data["tensors"].items()}
            images, _ = inference(model, op, tensors, device, "reference_400" if method == "tv_400" else "anchor")
            _, excess = readouts(rad, images.to(device))
            sd = (tensors["noise"][:, :3] * tensors["scale"])[:, :, 0, 0].to(device)
            scores.append((excess / sd[:, :, None, None]).cpu())
        scores = torch.cat(scores)
        repeated_thresholds = [float(torch.quantile(scores[:, c].flatten(), .99)) for c in range(3)]
        assert np.allclose(repeated_thresholds, frozen["thresholds"][method], rtol=.001, atol=1e-5), method
        with np.load(RUN / (method + "_heldout_predictions.npz")) as opened:
            predictions = opened["images"]
        assert predictions.shape == (384, 6, 128, 128), predictions.shape
        saved_rows = json.loads((RUN / (method + "_METRICS.json")).read_text())["per_frame"]
        replay_error, metric_differences, method_rows = [], {}, []
        offset = 0
        for item in heldout:
            data = torch.load(RUN / item["file"], map_location="cpu", weights_only=False)
            tensors, metadata = data["tensors"], data["metadata"]
            images = torch.from_numpy(predictions[offset:offset + 24])
            replayed, _ = inference(model, op, tensors, device, "reference_400" if method == "tv_400" else "anchor")
            relative = float((replayed - images).norm() / images.norm())
            assert relative <= .001, (method, item["mother"], relative)
            replay_error.append(relative)
            rows, _, _ = rows_for(rad, images, tensors, metadata, device, frozen["thresholds"][method])
            flattened = images.to(device).transpose(0, 1).reshape(6, 24 * 128, 128)
            _, continuum, _, _ = rad.fit(flattened)
            continua = continuum[:3].reshape(3, 24, 128, 128).transpose(0, 1).cpu().double()
            visible = images[:, :3].double()
            ratio_error = (images[:, [3, 5]].double().log() - images[:, 4:5].double().log()
                           - tensors["target_e"][:, [3, 5]].double().log() + tensors["target_e"][:, 4:5].double().log())
            pairing = {(r["level_e"], r["condition"], r["repeat"]): i for i, r in enumerate(metadata)}
            for index, row in enumerate(rows):
                old = saved_rows[offset + index]
                assert all(row[k] == old[k] for k in ("mother", "level_e", "condition", "repeat"))
                for key in ("temperature_MSE_K2", "band_relative_L2", "band_centered_relative_L2", "band_mean_bias_fraction",
                            "emission_actual_MSE_e2", "emission_clean_readout_MSE_e2", "noiseless_mixing_MSE_e2", "detected_fraction"):
                    a, b = np.asarray(row[key]), np.asarray(old[key])
                    assert np.allclose(a, b, rtol=1e-5, atol=1e-5), (method, index, key)
                    metric_differences[key] = max(metric_differences.get(key, 0.), float(np.max(np.abs(a - b))))
                row["log_ratio_bias"] = ratio_error[index].mean((-2, -1)).tolist()
                row["log_ratio_MSE"] = ratio_error[index].square().mean((-2, -1)).tolist()
                row["algorithm_mixing_cross_e2"] = ((np.asarray(row["emission_actual_MSE_e2"]) - np.asarray(row["emission_clean_readout_MSE_e2"])
                                                     - np.asarray(row["noiseless_mixing_MSE_e2"])) / 2).tolist()
                if row["condition"] in ("CH_weak", "AlO_weak", "C2_weak"):
                    j = ("CH_weak", "AlO_weak", "C2_weak").index(row["condition"])
                    blank = pairing[(row["level_e"], "blank", row["repeat"])]
                    signal = tensors["emission_e"][index, j].double()
                    denominator = signal.square().sum()
                    hv = ((visible[index] - visible[blank]) * signal).sum((-2, -1)) / denominator
                    hc = ((continua[index] - continua[blank]) * signal).sum((-2, -1)) / denominator
                    he = np.asarray(row["injection_response_columns"][str(j)])
                    assert np.allclose(hv.numpy() - hc.numpy(), he, atol=1e-5, rtol=1e-5)
                    row["response_visible"] = hv.tolist()
                    row["response_continuum"] = hc.tolist()
                method_rows.append(row)
            offset += 24
        native[method] = {"predictions_reopened": 384, "replay_frames": 384, "replay_mothers": 16,
                          "maximum_native_replay_relative_L2": max(replay_error), "replay_tolerance": .001,
                          "metric_maximum_absolute_differences": metric_differences,
                          "repeated_calibration_thresholds": repeated_thresholds,
                          "frozen_thresholds_used_for_all_reported_metrics": frozen["thresholds"][method]}
        all_rows[method] = method_rows
        runtime.write_json(OUT / "NATIVE_REPLAY_PROGRESS.json", {"time": runtime.now(), "completed": native})
        print(json.dumps({"method": method, "replayed": 384, "max_relative": max(replay_error)}), flush=True)
        del model, predictions, scores
runtime.write_json(OUT / "REOPENED_METRICS.json", all_rows)

# Descriptive paired-mother intervals. Thresholds, checkpoints and original gates stay frozen.
draws = 2000
indices = np.random.default_rng(73191).integers(0, 16, (draws, 16))
supplement = {"scope": "post-selection descriptive diagnostics, original development gates unchanged", "draws": draws,
              "resampling_unit": "independent mother carrying all repeats and paired methods/conditions",
              "multiple_comparison_adjustment": False, "methods": {}}
rmse_fields = ("temperature_MSE_K2", "emission_actual_MSE_e2", "emission_clean_readout_MSE_e2", "log_ratio_MSE")
mean_fields = ("band_relative_L2", "band_centered_relative_L2", "band_mean_bias_fraction", "detected_fraction", "log_ratio_bias", "algorithm_mixing_cross_e2")
samples = {}
for method in methods:
    output = {}
    for level in (1000, 4000):
        for condition in protocol["design"]["conditions"]:
            group = [r for r in all_rows[method] if r["level_e"] == level and r["condition"] == condition]
            item = {}
            for field in (*rmse_fields, *mean_fields):
                values = np.array([np.mean([r[field] for r in group if r["mother"] == mother], axis=0) for mother in sorted(split_ids["heldout"])])
                sampled = values[indices].mean(axis=1)
                point = values.mean(axis=0)
                if field in rmse_fields:
                    sampled, point = np.sqrt(sampled), np.sqrt(point)
                item[field] = {"point": np.asarray(point).tolist(), "CI95": np.quantile(sampled, [.025, .975], axis=0).tolist()}
                if field in rmse_fields or field in ("band_relative_L2", "band_centered_relative_L2"):
                    samples[(method, level, condition, field)] = sampled
                    item[field]["paired_ratio_to_TV_CI95"] = np.quantile(sampled / samples[("tv", level, condition, field)], [.025, .975], axis=0).tolist()
            output[f"{level}/{condition}"] = item
        response = {}
        for field in ("injection_response_columns", "response_visible", "response_continuum"):
            matrices = []
            for mother in sorted(split_ids["heldout"]):
                columns = []
                for j, condition in enumerate(("CH_weak", "AlO_weak", "C2_weak")):
                    group = [r for r in all_rows[method] if r["mother"] == mother and r["level_e"] == level and r["condition"] == condition]
                    columns.append(np.mean([r[field][str(j)] if field == "injection_response_columns" else r[field] for r in group], axis=0))
                matrices.append(np.stack(columns, axis=1))
            values = np.asarray(matrices)
            response[field] = {"point": values.mean(axis=0).tolist(), "CI95": np.quantile(values[indices].mean(axis=1), [.025, .975], axis=0).tolist()}
        output[f"{level}/response_matrices"] = response
        contrasts = {}
        for j, condition in enumerate(("CH_weak", "AlO_weak", "C2_weak")):
            differences = []
            for mother in sorted(split_ids["heldout"]):
                g = [r for r in all_rows[method] if r["mother"] == mother and r["level_e"] == level]
                active = np.mean([r["detected_fraction"] for r in g if r["condition"] == condition], axis=0)
                blank = np.mean([r["detected_fraction"] for r in g if r["condition"] == "blank"], axis=0)
                differences.append(active - blank)
            values = np.asarray(differences)
            contrasts[condition] = {"non_target_channels": [k for k in range(3) if k != j],
                                    "FPR_minus_blank_point": values.mean(axis=0).tolist(),
                                    "FPR_minus_blank_CI95": np.quantile(values[indices].mean(axis=1), [.025, .975], axis=0).tolist()}
        output[f"{level}/single_injection_FPR_contrasts"] = contrasts
    supplement["methods"][method] = output
runtime.write_json(OUT / "SUPPLEMENTAL_STATISTICS.json", supplement)
runtime.write_json(OUT / "AUDIT_COMPLETION.json", {"time": runtime.now(), "execution_verified": True,
                   "all_native_replays_pass": True, "native": native, "wall_seconds": time.monotonic() - start,
                   "GPU_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2,
                   "training_started": False, "thresholds_or_truth_changed": False,
                   "full_model_development_pass": summary["development_gates"]["full"]["passed"],
                   "grouped_post_development_pass": summary["development_gates"]["grouped_post"]["passed"],
                   "formal_three_seed_acceptance": False})
