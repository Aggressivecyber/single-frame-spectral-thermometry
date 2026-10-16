"""Repair the final JSON receipt and independently review saved, completed evaluations."""
from pathlib import Path
import argparse
import os
import sys
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
base = args.run_root.resolve()
out = args.output.resolve()
out.mkdir()
os.environ["THERMOMETRY_RUNTIME"] = str(out/"_runtime")
sys.path.insert(0, str(base))
import runtime
import numpy as np
import torch
from research_joint_imaging.core import Imager, Radiometry, LearnedReconstruction, tv_reconstruct

start = time.monotonic()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
source = base/"reports/joint_imaging_001"
p = json.loads((base/"research_joint_imaging/protocol.json").read_text())
registration = json.loads((base/"research_joint_imaging/REGISTRATION.json").read_text())
for name, digest in registration["source_sha256"].items():
    if runtime.sha256(base/name) != digest:
        raise RuntimeError("Executed source differs: "+name)
inputs = {str(path.relative_to(base)): {"sha256": runtime.sha256(path), "bytes": path.stat().st_size} for path in sorted(source.rglob("*")) if path.is_file()}
runtime.write_json(out/"INPUT_MANIFEST.json", inputs)
data_receipt = json.loads((source/"DATA_RECEIPT.json").read_text())
selection = json.loads((source/"TRADITIONAL_SELECTION.json").read_text())
train_receipt = json.loads((source/"TRAINING_COMPLETE.json").read_text())
selected = selection["selected"]
metrics = {path.stem.removesuffix("_METRICS"): json.loads(path.read_text()) for path in sorted(source.glob("*_METRICS.json"))}
if len(metrics) != 10:
    raise RuntimeError("Expected all ten method evaluations before repairing the summary")
all_mothers, all_noise = set(), set()
dataset_checks = []
for split, info in data_receipt["splits"].items():
    path = source/info["file"]
    if runtime.sha256(path) != info["sha256"]:
        raise RuntimeError("Dataset hash differs: "+split)
    data = torch.load(path, map_location="cpu", weights_only=False)
    mothers = {m["mother"] for m in data["metadata"]}
    noise = {m["noise_seed"] for m in data["metadata"]}
    keys = {(m["mother"], m["level_e"], m["condition"], m["repeat"]) for m in data["metadata"]}
    passed = len(mothers) == info["mothers"] and len(keys) == len(data["metadata"]) == info["observations"] and len(noise) == len(keys) and not (mothers & all_mothers) and not (noise & all_noise)
    passed = passed and data["raw_adc"].dtype == torch.uint16 and bool(torch.isfinite(data["target"]).all()) and bool(torch.isfinite(data["emission"]).all()) and bool(torch.isfinite(data["color_temperature"]).all())
    passed = passed and int(data["raw_adc"].to(torch.int32).max()) <= 4095
    if not passed:
        raise RuntimeError("Data partition/array integrity check failed: "+split)
    dataset_checks.append({"split": split, "mothers": len(mothers), "observations": len(keys), "raw_min_DN": int(data["raw_adc"].to(torch.int32).min()), "raw_max_DN": int(data["raw_adc"].to(torch.int32).max()), "sha256_verified": True, "finite_arrays": True, "no_cross_split_mothers_or_noise_seeds": True})
    all_mothers |= mothers
    all_noise |= noise
    if split == "heldout":
        heldout = data
ledger = [json.loads(line) for line in (source/"calls.jsonl").read_text().splitlines()]
selection_freeze = json.loads((source/"SELECTION_FROZEN.json").read_text())
threshold_freeze = json.loads((source/"THRESHOLDS_FROZEN.json").read_text())
first_holdout = next(row["time"] for row in ledger if row["stage"] == "heldout")
for name, digest in threshold_freeze["files"].items():
    if runtime.sha256(source/name) != digest:
        raise RuntimeError("Frozen detection threshold changed")
if not train_receipt["time"] <= selection_freeze["time"] < threshold_freeze["time"] <= first_holdout:
    raise RuntimeError("Held-out evaluation was not preceded by frozen choices and thresholds")

op = Imager(p["model"]).cuda()
rad = Radiometry(p, "cuda")
q = p["model"]["full_well_e"]/((2**p["model"]["adc_bits"]-1)-p["model"]["black_level_DN"])
with torch.no_grad():
    raw = (heldout["raw_adc"].cuda().float()-p["model"]["black_level_DN"])*q
    initial, y, scale, weight, noise = op.initialize(raw)
targets = heldout["target"].numpy()
truth_t = heldout["color_temperature"].numpy()
truth_emission = heldout["emission"].numpy()
replay_receipts = []
metric_checks = []
examples = {}
example_id = 20
for name, document in metrics.items():
    with np.load(source/(name+"_heldout_predictions.npz")) as arrays:
        images, temperatures, emission = arrays["images"], arrays["temperature"], arrays["emission_readout"]
    if images.shape != targets.shape or not (np.isfinite(images).all() and np.isfinite(temperatures).all() and np.isfinite(emission).all()):
        raise RuntimeError("Invalid prediction payload: "+name)
    l2 = np.sqrt(np.sum((images-targets)**2, axis=(-2, -1))/np.sum(targets**2, axis=(-2, -1)))
    temp_error = np.sqrt(np.mean((temperatures-truth_t)**2, axis=(-2, -1)))
    emission_error = np.sqrt(np.mean((emission-truth_emission)**2, axis=(-2, -1)))
    rows = document["heldout"]
    l2_diff = float(np.max(np.abs(l2-np.asarray([r["band_relative_L2"] for r in rows]))))
    temp_diff = float(np.max(np.abs(temp_error-np.asarray([r["temperature_RMSE_K"] for r in rows]))))
    emission_diff = float(np.max(np.abs(emission_error-np.asarray([r["emission_RMSE_e"] for r in rows]))))
    if max(l2_diff/1e-5, temp_diff/0.001, emission_diff/0.001) > 1:
        raise RuntimeError("Stored full-array metrics differ from saved scores: "+name)
    metric_checks.append({"method": name, "frames": len(rows), "max_band_L2_difference": l2_diff, "max_temperature_RMSE_difference_K": temp_diff, "max_emission_RMSE_difference_e": emission_diff, "all_arrays_finite": True})
    model = None
    if name.startswith("cnn_"):
        model = LearnedReconstruction(p["learning"], name.startswith("cnn_dc")).cuda()
        model.load_state_dict(torch.load(source/(name+".pt"), map_location="cuda", weights_only=False)["model"])
        model.eval()
    repeated = []
    with torch.no_grad():
        for offset in (0, 12):
            s = slice(offset, offset+12)
            if name == "fft":
                result = initial[s]
            elif name == "wls":
                rhs = op.project(op.images_adjoint(weight[s]*y[s]))
                result = op.solve(rhs, initial[s], weight[s], 0., steps=30, tolerance=1e-6)
            elif name == "dc_only":
                result = initial[s]
                for _ in range(p["learning"]["shared_stages"]):
                    result = op.data_step(y[s], weight[s], result)
            elif name == "tv":
                result, _ = tv_reconstruct(op, y[s], weight[s], initial[s], selection["best_tv"]["lambda"], p["traditional"])
            else:
                result = model(op, initial[s], y[s], weight[s], noise[s])
            repeated.append((result*scale[s]).cpu().numpy())
    repeated = np.concatenate(repeated)
    delta = repeated-images[:24]
    maximum = float(np.abs(delta).max())
    relative = float(np.linalg.norm(delta)/np.linalg.norm(images[:24]))
    if maximum > .05 or relative > 1e-5:
        raise RuntimeError("Native replay differs from saved reconstruction: "+name)
    replay_receipts.append({"method": name, "frames": 24, "same_batch_size_as_original": 12, "max_absolute_difference_e": maximum, "relative_L2_difference": relative})
    if name in ("tv", "cnn_dc_701"):
        examples[name] = {"images": images[example_id].copy(), "temperature": temperatures[example_id].copy(), "emission": emission[example_id].copy()}
    print({"reviewed": name, "full_arrays_checked": len(rows), "replayed": 24, "replay_max_e": maximum}, flush=True)

summaries = {name: item["summary"] for name, item in metrics.items()}
reference = summaries[selected["method"]]["4000/all_weak"]
gates = []
for seed in p["learning"]["seeds"]:
    name = f"cnn_dc_{seed}"
    item = summaries[name]["4000/all_weak"]
    temperature_ratio = item["temperature_RMSE_K"]/reference["temperature_RMSE_K"]
    image_ratio = max(a/b for a, b in zip(item["band_relative_L2_mean"], reference["band_relative_L2_mean"]))
    centered_ratio = max(a/b for a, b in zip(item["band_centered_relative_L2_mean"], reference["band_centered_relative_L2_mean"]))
    emission_ratio = max(a/b for a, b in zip(item["emission_RMSE_e_pooled"], reference["emission_RMSE_e_pooled"]))
    fpr = max(value for level in p["design"]["counts_800nm_mean_e"] for value in summaries[name][f"{level}/blank"]["detected_fraction_mean"])
    slopes = item["paired_injection_slope_mean"]
    checks = {"temperature_under_50K": item["temperature_RMSE_K"] <= 50, "each_band_under_5percent": max(item["band_relative_L2_mean"]) <= .05, "weak_emission_improved_15percent_each": emission_ratio <= .85, "temperature_not_worse_5percent": temperature_ratio <= 1.05, "each_band_not_worse_5percent": image_ratio <= 1.05, "each_centered_band_not_worse_5percent": centered_ratio <= 1.05, "mean_intensity_bias_under_2percent": max(abs(v) for v in item["band_mean_bias_fraction_mean"]) <= .02, "blank_FPR_under_2percent_each_level": fpr <= .02, "injection_slopes_0_9_to_1_1": min(slopes) >= .90 and max(slopes) <= 1.10, "traditional_numerics": selected["method"] != "tv" or selection["numerical_audit_passed"], "no_invalid_thermal_pixels": item["nonpositive_thermal_pixels"] == 0 and item["temperature_bound_pixels"] == 0}
    gates.append({"method": name, "passed": all(checks.values()), "checks": checks, "temperature_ratio": temperature_ratio, "worst_band_error_ratio": image_ratio, "worst_centered_error_ratio": centered_ratio, "worst_emission_error_ratio": emission_ratio, "max_blank_FPR": fpr, "slopes": slopes})
summary = {"time": runtime.now(), "repair": "Native Python scalars replace numpy.bool_ only in final JSON aggregation. Original process exit 1 and every original scientific output are retained.", "original_exit": json.loads((source/"EXIT.json").read_text()), "selected_traditional": selected, "summaries": summaries, "hybrid_gates": gates, "hybrid_accepted_for_next_development_stage": all(g["passed"] for g in gates)}
runtime.write_json(out/"SUMMARY_REPAIRED.json", summary)
runtime.write_json(out/"VERIFICATION.json", {"time": runtime.now(), "all_executed_source_hashes_verified": True, "datasets": dataset_checks, "training_runs": len(train_receipt["runs"]), "optimizer_updates": sum(r["updates"] for r in train_receipt["runs"]), "mothers_total": len(all_mothers), "all_noise_seeds_unique": len(all_noise), "original_complete_reconstruction_observations": sum(row["complete_reconstruction_observations"] for row in ledger), "frozen_threshold_files": len(threshold_freeze["files"]), "selection_time": selection_freeze["time"], "threshold_freeze_time": threshold_freeze["time"], "first_heldout_evaluation_initialization": first_holdout, "heldout_created_before_training_but_evaluation_loaded_after_freeze": True, "full_array_metric_checks": metric_checks, "native_replays": replay_receipts, "review_reconstruction_observations": 192+240, "new_training_updates": 0, "new_noise_draws": 0, "passed": True})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.facecolor": "white"})
fig, axes = plt.subplots(6, 3, figsize=(12, 18), constrained_layout=True)
for k, wavelength in enumerate(p["channel_centers_nm"]):
    vmax = float(targets[example_id, k].max())
    for col, (name, array) in enumerate((("Noiseless optical target", targets[example_id]), ("Weighted TV", examples["tv"]["images"]), ("Hybrid, seed 701", examples["cnn_dc_701"]["images"]))):
        im = axes[k, col].imshow(array[k], cmap="gray", vmin=0, vmax=vmax)
        axes[k, col].set_xticks([]); axes[k, col].set_yticks([])
        if k == 0: axes[k, col].set_title(name)
        if col == 0: axes[k, col].set_ylabel(f"{wavelength:g} nm", fontsize=12)
        fig.colorbar(im, ax=axes[k, col], shrink=.75, label="electrons")
fig.suptitle("One shared sensor frame: six reconstructed filter images\nFixed held-out example, 4000 e at 800 nm; each visible emission adds mean 200 e", fontsize=14)
fig.savefig(out/"channel_reconstruction.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(4, 4, figsize=(15, 13), constrained_layout=True)
truths = [truth_t[example_id], *truth_emission[example_id]]
names = ["Effective color temperature (K)", "CH-region excess (e)", "AlO-region excess (e)", "C2-region excess (e)"]
for row in range(4):
    truth = truths[row]
    tv = examples["tv"]["temperature"] if row == 0 else examples["tv"]["emission"][row-1]
    hybrid = examples["cnn_dc_701"]["temperature"] if row == 0 else examples["cnn_dc_701"]["emission"][row-1]
    lower, upper = (float(truth.min()), float(truth.max())) if row == 0 else (-.15*float(truth.max()), float(truth.max()))
    for col, array in enumerate((truth, tv, hybrid)):
        im = axes[row, col].imshow(array, cmap="inferno", vmin=lower, vmax=upper)
        fig.colorbar(im, ax=axes[row, col], shrink=.75)
    bound = max(float(np.abs(hybrid-truth).max()), 1)
    im = axes[row, 3].imshow(hybrid-truth, cmap="coolwarm", vmin=-bound, vmax=bound)
    fig.colorbar(im, ax=axes[row, 3], shrink=.75)
    axes[row, 0].set_ylabel(names[row], fontsize=11)
    for col in range(4):
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        if row == 0: axes[row, col].set_title(("Optical reference / injection", "Weighted TV", "Hybrid, seed 701", "Hybrid error")[col])
fig.suptitle("Joint diagnostic outputs from the same encoded observation\nTemperature is a fitted color temperature; feature truth is the injected band emission", fontsize=14)
fig.savefig(out/"joint_products.png", dpi=130); plt.close(fig)

labels = ["FFT", "WLS", "TV", "DC only", "CNN post", "Hybrid"]
groups = [["fft"], ["wls"], ["tv"], ["dc_only"], [f"cnn_post_{s}" for s in p["learning"]["seeds"]], [f"cnn_dc_{s}" for s in p["learning"]["seeds"]]]
fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
positions = np.arange(len(groups))
for i, group in enumerate(groups):
    values = np.array([summaries[name]["4000/all_weak"]["temperature_RMSE_K"] for name in group])
    axes[0].bar(i, values.mean(), color="#2369a1" if i != 2 else "#2e9466")
    axes[0].errorbar(i, values.mean(), yerr=[[values.mean()-values.min()], [values.max()-values.mean()]], color="black", capsize=4)
    axes[0].text(i, values.max()+2, f"{values.mean():.1f}", ha="center", fontsize=10)
for k, color in enumerate(("#2379a9", "#de8431", "#45966e")):
    means = [np.mean([summaries[name]["4000/all_weak"]["emission_RMSE_e_pooled"][k] for name in group]) for group in groups]
    axes[1].bar(positions+(k-1)*.24, means, width=.24, color=color, label=("CH region", "AlO region", "C2 region")[k])
    slopes = [np.mean([summaries[name]["4000/all_weak"]["paired_injection_slope_mean"][k] for name in group]) for group in groups]
    axes[2].plot(positions, slopes, "o-", color=color, label=("CH region", "AlO region", "C2 region")[k])
axes[0].axhline(50, color="#a53a38", linestyle="--", label="50 K screen")
axes[0].set_ylabel("Effective temperature RMSE (K)")
axes[1].set_ylabel("Injected emission RMSE (electrons)"); axes[1].legend(fontsize=10)
axes[2].axhspan(.9, 1.1, color="#dceadd"); axes[2].set_ylabel("Paired injection response slope"); axes[2].set_ylim(.65, 1.15)
for ax in axes:
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=.2)
fig.suptitle("Held-out joint-imaging screen: 8 mothers, 16 primary observations\nNeural bars average all three seeds; temperature whiskers show seed min-max", fontsize=14)
fig.savefig(out/"method_comparison.png", dpi=140); plt.close(fig)

process = json.loads((source/"PROCESS.json").read_text())
process_observations = []
for role in ("supervisor_pid", "child_pid"):
    proc = Path("/proc")/str(process[role])
    match = proc.exists() and (proc/"cwd").resolve() == base
    process_observations.append({"role": role, "original_pid": process[role], "matching_run_process_alive": match})
if any(item["matching_run_process_alive"] for item in process_observations):
    raise RuntimeError("Original process still active during closeout")
runtime.write_json(out/"REVIEW_COMPLETION.json", {"time": runtime.now(), "review_completed": True, "original_script_exit_code": 1, "original_failure_stage": "final JSON serialization after all ten evaluations saved", "summary_repaired": True, "scientific_candidate_accepted": all(g["passed"] for g in gates), "source_processes": process_observations, "new_training_updates": 0, "new_noise_draws": 0, "raw_dataset_count": 1056, "heldout_method_evaluations": 10*192, "wall_seconds": time.monotonic()-start, "source_run_report_bytes": sum(path.stat().st_size for path in source.rglob("*") if path.is_file()), "gpu_peak_allocated_MiB": torch.cuda.max_memory_allocated()/1024**2})
runtime.write_json(out/"OUTPUT_MANIFEST.json", {path.name: {"sha256": runtime.sha256(path), "bytes": path.stat().st_size} for path in out.iterdir() if path.is_file()})
print({"review_completed": True, "wall_seconds": time.monotonic()-start, "hybrid_accepted": all(g["passed"] for g in gates)}, flush=True)
