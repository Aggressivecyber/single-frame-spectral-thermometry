"""Frozen six-band pilot: conventional controls, independent training, calibrated readout."""
import runtime
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from research_joint_imaging.core import Imager, Radiometry, LearnedReconstruction, training_loss, tv_reconstruct
from research_joint_imaging.data import prepare


parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cuda"], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/"research_joint_imaging/protocol.json").read_text())
registration = json.loads((root/"research_joint_imaging/REGISTRATION.json").read_text())
for name, expected in registration["source_sha256"].items():
    if runtime.sha256(root/name) != expected:
        raise RuntimeError("Transferred source hash mismatch: "+name)
torch.set_num_threads(protocol["resources"]["cpu_threads"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
start = time.monotonic()
out = root/"reports"/protocol["run_id"]
out.mkdir(parents=True, exist_ok=True)
ledger = (out/"calls.jsonl").open("x", encoding="utf-8")
op = Imager(protocol["model"]).to(args.device)
rad = Radiometry(protocol, args.device)
q = protocol["model"]["full_well_e"]/((2**protocol["model"]["adc_bits"]-1)-protocol["model"]["black_level_DN"])


def record(stage, method, count, **extra):
    ledger.write(json.dumps({"time": runtime.now(), "stage": stage, "method": method, "complete_reconstruction_observations": count, **extra})+"\n")
    ledger.flush()


def load_split(name):
    expected = data_receipt["splits"][name]["sha256"]
    path = out/"data"/(name+".pt")
    if runtime.sha256(path) != expected:
        raise RuntimeError("Dataset changed before use: "+name)
    data = torch.load(path, map_location=args.device, weights_only=False)
    raw = (data["raw_adc"].float()-protocol["model"]["black_level_DN"])*q
    initial, y, scale, weight, noise = op.initialize(raw)
    data.update(initial=initial, y=y, scale=scale, weight=weight, noise=noise)
    record(name, "FFT_initialization", len(data["metadata"]))
    return data


@torch.no_grad()
def reconstruct(data, method, model=None, strength=None, indices=None, iterations=None):
    selected = list(range(len(data["metadata"]))) if indices is None else indices
    chunks, diagnostics = [], []
    tick = time.monotonic()
    for offset in range(0, len(selected), 12):
        ids = selected[offset:offset+12]
        initial, y, weight = data["initial"][ids], data["y"][ids], data["weight"][ids]
        if method == "fft":
            result = initial
        elif method == "wls":
            rhs = op.project(op.images_adjoint(weight*y))
            result = op.solve(rhs, initial, weight, 0.0, steps=30, tolerance=1e-6)
            residual = op.normal(result, weight, 0.0, 0.0)-rhs
            diagnostics.append({"indices": ids, "normal_equation_relative_residual": float(residual.norm()/rhs.norm())})
        elif method == "dc_only":
            result = initial
            for _ in range(protocol["learning"]["shared_stages"]):
                result = op.data_step(y, weight, result)
        elif method == "tv":
            result, history = tv_reconstruct(op, y, weight, initial, strength, protocol["traditional"], iterations)
            diagnostics.append({"indices": ids, "history": history})
        else:
            result = model(op, initial, y, weight, data["noise"][ids])
        chunks.append((result*data["scale"][ids]).cpu())
    torch.cuda.synchronize()
    record("evaluation", method, len(selected), tv_lambda=strength, wall_s=time.monotonic()-tick)
    return torch.cat(chunks), diagnostics


def band_score(pred, target):
    return float(((pred-target).square().sum((-2, -1))/target.square().sum((-2, -1))).sqrt().mean())


@torch.no_grad()
def readouts(prediction, data):
    temperatures, excesses, rows = [], [], []
    for i, prediction_cpu in enumerate(prediction):
        pred = prediction_cpu.to(args.device)
        true = data["target"][i]
        t, _, excess, fit_residual = rad.fit(pred)
        _, _, clean_excess, _ = rad.fit(true)
        error = pred-true
        pcenter = pred-pred.mean((-2, -1), keepdim=True)
        tcenter = true-true.mean((-2, -1), keepdim=True)
        rows.append({**data["metadata"][i],
                     "band_relative_L2": (error.square().sum((-2, -1))/true.square().sum((-2, -1))).sqrt().cpu().tolist(),
                     "band_mean_bias_fraction": (error.mean((-2, -1))/true.mean((-2, -1))).cpu().tolist(),
                     "band_centered_relative_L2": ((pcenter-tcenter).square().sum((-2, -1))/tcenter.square().sum((-2, -1))).sqrt().cpu().tolist(),
                     "temperature_RMSE_K": float((t-data["color_temperature"][i]).square().mean().sqrt()),
                     "emission_RMSE_e": (excess-data["emission"][i]).square().mean((-2, -1)).sqrt().cpu().tolist(),
                     "noiseless_readout_RMSE_e": (excess-clean_excess).square().mean((-2, -1)).sqrt().cpu().tolist(),
                     "thermal_fit_residual_mean": float(fit_residual.mean()),
                     "nonpositive_thermal_pixels": int((pred[3:] <= 0).any(0).sum()),
                     "temperature_bound_pixels": int(((t <= 1800) | (t >= 4500)).sum())})
        temperatures.append(t.cpu())
        excesses.append(excess.cpu())
    return torch.stack(temperatures), torch.stack(excesses), rows


@torch.no_grad()
def evaluate_heldout(name, heldout_prediction, heldout):
    temp, heldout_excess, rows = readouts(heldout_prediction, heldout)
    hold_sd = (heldout["noise"][:, :3, 0, 0]*heldout["scale"].flatten(1)[:, :1]).cpu()
    threshold = json.loads((out/(name+"_THRESHOLDS.json")).read_text())["thresholds"]
    threshold_tensor = torch.tensor(threshold)[None, :, None, None]
    detected = heldout_excess/hold_sd[:, :, None, None] > threshold_tensor
    pairing = {(r["mother"], r["level_e"], r["repeat"], r["condition"]): i for i, r in enumerate(rows)}
    for i, row in enumerate(rows):
        truth = heldout["emission"][i].cpu()
        active = [k for k, amount in enumerate(protocol["design"]["conditions"][row["condition"]]) if amount > 0]
        row["detected_fraction"] = detected[i].float().mean((-2, -1)).tolist()
        row["sensitivity"] = [float(detected[i, k][truth[k] > 0.2*truth[k].mean()].float().mean()) if k in active else None for k in range(3)]
        if active:
            blank_id = pairing[(row["mother"], row["level_e"], row["repeat"], "blank")]
            response = heldout_excess[i]-heldout_excess[blank_id]
            row["paired_injection_slope"] = [float((response[k]*truth[k]).sum()/truth[k].square().sum()) if k in active else None for k in range(3)]
            row["inactive_increment_RMSE_e"] = [float(response[k].square().mean().sqrt()) if k not in active else None for k in range(3)]
    summary = {}
    for level in protocol["design"]["counts_800nm_mean_e"]:
        for condition in protocol["design"]["conditions"]:
            group = [r for r in rows if r["level_e"] == level and r["condition"] == condition]
            item = {"frames": len(group), "mothers": len({r["mother"] for r in group})}
            for key in ("band_relative_L2", "band_mean_bias_fraction", "band_centered_relative_L2", "emission_RMSE_e", "noiseless_readout_RMSE_e", "detected_fraction"):
                item[key+"_mean"] = np.mean([r[key] for r in group], axis=0).tolist()
            item["temperature_RMSE_K"] = float(np.sqrt(np.mean([r["temperature_RMSE_K"]**2 for r in group])))
            item["emission_RMSE_e_pooled"] = np.sqrt(np.mean(np.asarray([r["emission_RMSE_e"] for r in group])**2, axis=0)).tolist()
            item["nonpositive_thermal_pixels"] = sum(r["nonpositive_thermal_pixels"] for r in group)
            item["temperature_bound_pixels"] = sum(r["temperature_bound_pixels"] for r in group)
            if condition != "blank":
                item["paired_injection_slope_mean"] = [None if group[0]["paired_injection_slope"][k] is None else float(np.mean([r["paired_injection_slope"][k] for r in group])) for k in range(3)]
                item["sensitivity_mean"] = [None if group[0]["sensitivity"][k] is None else float(np.mean([r["sensitivity"][k] for r in group])) for k in range(3)]
            summary[f"{level}/{condition}"] = item
    runtime.write_json(out/(name+"_METRICS.json"), {"summary": summary, "heldout": rows})
    np.savez_compressed(out/(name+"_heldout_predictions.npz"), images=heldout_prediction.numpy(), temperature=temp.numpy(), emission_readout=heldout_excess.numpy())
    return summary


data_receipt = prepare(protocol, op, rad, out)
train = load_split("train")
validation = load_split("validation")
validation_ids = [i for i, meta in enumerate(validation["metadata"]) if meta["level_e"] == 4000 and meta["repeat"] == 0]
validation_target = validation["target"][validation_ids].cpu()
validation_scores = []
fft_prediction, _ = reconstruct(validation, "fft", indices=validation_ids)
validation_scores.append({"method": "fft", "lambda": None, "score": band_score(fft_prediction, validation_target)})
wls_prediction, wls_history = reconstruct(validation, "wls", indices=validation_ids)
validation_scores.append({"method": "wls", "lambda": None, "score": band_score(wls_prediction, validation_target)})
runtime.write_json(out/"WLS_VALIDATION.json", {"score": validation_scores[-1], "solver": wls_history})
for strength in protocol["traditional"]["tv_lambda_candidates"]:
    pred, history = reconstruct(validation, "tv", strength=strength, indices=validation_ids)
    validation_scores.append({"method": "tv", "lambda": strength, "score": band_score(pred, validation_target)})
    runtime.write_json(out/("TV_VALIDATION_"+str(strength)+".json"), {"score": validation_scores[-1], "solver": history})
    print({"stage": "traditional_validation", **validation_scores[-1]}, flush=True)
selected = min(validation_scores, key=lambda item: item["score"])
best_tv = min([item for item in validation_scores if item["method"] == "tv"], key=lambda item: item["score"])
audit_ids = validation_ids[:2]
normal_pred, normal_history = reconstruct(validation, "tv", strength=best_tv["lambda"], indices=audit_ids)
long_pred, long_history = reconstruct(validation, "tv", strength=best_tv["lambda"], indices=audit_ids, iterations=2*protocol["traditional"]["maximum_iterations"])
numerical_change = float((normal_pred-long_pred).norm()/long_pred.norm())
runtime.write_json(out/"TRADITIONAL_SELECTION.json", {"time": runtime.now(), "validation_scores": validation_scores, "selected": selected, "best_tv": best_tv, "doubling_iteration_output_relative_change": numerical_change, "numerical_audit_passed": numerical_change <= 0.001, "normal_solver": normal_history, "doubled_solver": long_history, "selected_before_heldout_open": True})

training_receipts = []
for method in protocol["learning"]["methods"]:
    for seed in protocol["learning"]["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = LearnedReconstruction(protocol["learning"], method == "cnn_dc").to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning"]["learning_rate"])
        order = np.random.default_rng(seed)
        history = []
        tick = time.monotonic()
        for step in range(protocol["learning"]["updates_per_run"]):
            ids = order.choice(len(train["metadata"]), protocol["learning"]["batch_size"], replace=False).tolist()
            optimizer.zero_grad(set_to_none=True)
            pred = model(op, train["initial"][ids], train["y"][ids], train["weight"][ids], train["noise"][ids])
            loss = training_loss(pred, train["target"][ids]/train["scale"][ids])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            optimizer.step()
            record("training", method, len(ids), seed=seed, update=step+1, loss=float(loss.detach()))
            if step == 0 or (step+1) % 25 == 0:
                progress = {"time": runtime.now(), "stage": "optimizer", "method": method, "seed": seed, "update": step+1, "loss": float(loss.detach()), "elapsed_s": time.monotonic()-start}
                history.append(progress)
                runtime.write_json(out/"PROGRESS.json", progress)
                print(progress, flush=True)
        checkpoint = out/(f"{method}_{seed}.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "updates": step+1, "method": method}, checkpoint)
        training_receipts.append({"method": method, "seed": seed, "updates": step+1, "parameters": sum(v.numel() for v in model.parameters()), "wall_s": time.monotonic()-tick, "checkpoint_sha256": runtime.sha256(checkpoint), "history": history})
runtime.write_json(out/"TRAINING_COMPLETE.json", {"time": runtime.now(), "runs": training_receipts, "heldout_opened": False})
runtime.write_json(out/"SELECTION_FROZEN.json", {"time": runtime.now(), "traditional": selected, "neural_methods": protocol["learning"]["methods"], "all_seeds": protocol["learning"]["seeds"], "checkpoint_rule": "final registered update, no selection using heldout", "heldout_opened": False})
calibration = load_split("calibration")
method_specs = [("fft", None), ("wls", None), ("tv", None), ("dc_only", None)]+[(method, seed) for method in protocol["learning"]["methods"] for seed in protocol["learning"]["seeds"]]
cal_sd = (calibration["noise"][:, :3, 0, 0]*calibration["scale"].flatten(1)[:, :1]).cpu()
null_ids = [i for i, meta in enumerate(calibration["metadata"]) if meta["condition"] == "blank"]
for method, seed in method_specs:
    name = method if seed is None else f"{method}_{seed}"
    model = None
    if seed is not None:
        model = LearnedReconstruction(protocol["learning"], method == "cnn_dc").to(args.device)
        checkpoint = torch.load(out/(name+".pt"), map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    cal_pred, cal_history = reconstruct(calibration, method, model=model, strength=best_tv["lambda"])
    _, excess, rows = readouts(cal_pred, calibration)
    score = excess/cal_sd[:, :, None, None]
    threshold = [float(torch.quantile(score[null_ids, k].flatten(), 0.99)) for k in range(3)]
    runtime.write_json(out/(name+"_THRESHOLDS.json"), {"time": runtime.now(), "source": "blank calibration mothers only", "score": "excess / upstream raw-derived noise RMS", "quantile": 0.99, "thresholds": threshold, "blank_frames": len(null_ids), "calibration_mothers": 6, "heldout_opened": False})
    runtime.write_json(out/(name+"_CALIBRATION.json"), {"rows": rows, "solver": cal_history})
    print({"stage": "threshold_calibrated", "method": name}, flush=True)
runtime.write_json(out/"THRESHOLDS_FROZEN.json", {"time": runtime.now(), "files": {name.name: runtime.sha256(name) for name in out.glob("*_THRESHOLDS.json")}, "heldout_opened": False})
heldout = load_split("heldout")
summaries = {}
for method, seed in method_specs:
    name = method if seed is None else f"{method}_{seed}"
    model = None
    if seed is not None:
        model = LearnedReconstruction(protocol["learning"], method == "cnn_dc").to(args.device)
        checkpoint = torch.load(out/(name+".pt"), map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    hold_pred, hold_history = reconstruct(heldout, method, model=model, strength=best_tv["lambda"])
    summaries[name] = evaluate_heldout(name, hold_pred, heldout)
    runtime.write_json(out/(name+"_HELDOUT_SOLVER.json"), {"solver": hold_history})
    print({"stage": "heldout_evaluated", "method": name}, flush=True)

base = summaries[selected["method"]]["4000/all_weak"]
gates = []
for seed in protocol["learning"]["seeds"]:
    name = f"cnn_dc_{seed}"
    result = summaries[name]["4000/all_weak"]
    t_ratio = result["temperature_RMSE_K"]/base["temperature_RMSE_K"]
    image_ratio = float(max(np.asarray(result["band_relative_L2_mean"])/np.asarray(base["band_relative_L2_mean"])))
    centered_ratio = float(max(np.asarray(result["band_centered_relative_L2_mean"])/np.asarray(base["band_centered_relative_L2_mean"])))
    emission_ratio = float(max(np.asarray(result["emission_RMSE_e_pooled"])/np.asarray(base["emission_RMSE_e_pooled"])))
    fpr = max(value for level in protocol["design"]["counts_800nm_mean_e"] for value in summaries[name][f"{level}/blank"]["detected_fraction_mean"])
    slopes = result["paired_injection_slope_mean"]
    checks = {"temperature_under_50K": result["temperature_RMSE_K"] <= 50, "each_band_under_5percent": max(result["band_relative_L2_mean"]) <= .05, "weak_emission_improved_15percent_each": emission_ratio <= .85, "temperature_not_worse_5percent": t_ratio <= 1.05, "each_band_not_worse_5percent": image_ratio <= 1.05, "each_centered_band_not_worse_5percent": centered_ratio <= 1.05, "mean_intensity_bias_under_2percent": max(abs(v) for v in result["band_mean_bias_fraction_mean"]) <= .02, "blank_FPR_under_2percent_each_level": fpr <= .02, "injection_slopes_0_9_to_1_1": min(slopes) >= .90 and max(slopes) <= 1.10, "traditional_numerics": selected["method"] != "tv" or numerical_change <= .001, "no_invalid_thermal_pixels": result["nonpositive_thermal_pixels"] == 0 and result["temperature_bound_pixels"] == 0}
    gates.append({"method": name, "checks": checks, "passed": all(checks.values()), "temperature_ratio": t_ratio, "worst_band_error_ratio": image_ratio, "worst_emission_error_ratio": emission_ratio, "max_blank_FPR": fpr, "slopes": slopes})
runtime.write_json(out/"SUMMARY.json", {"time": runtime.now(), "evidence_level": protocol["evidence_level"], "selected_traditional": selected, "summaries": summaries, "hybrid_gates": gates, "hybrid_accepted_for_next_development_stage": all(g["passed"] for g in gates)})
ledger.close()
artifact_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
if artifact_bytes > protocol["resources"]["maximum_new_artifact_bytes"]:
    raise RuntimeError("Registered new-artifact allocation exceeded")
peak_memory = torch.cuda.max_memory_allocated()/1024**2
runtime.write_json(out/"COMPLETION.json", {"time": runtime.now(), "completed": True, "wall_s": time.monotonic()-start, "raw_observations": data_receipt["total_raw_observations"], "training_runs": len(training_receipts), "total_optimizer_updates": sum(item["updates"] for item in training_receipts), "peak_allocated_gpu_MiB": peak_memory, "artifact_bytes": artifact_bytes, "hybrid_accepted": all(g["passed"] for g in gates)})
print({"stage": "complete", "wall_s": time.monotonic()-start, "hybrid_accepted": all(g["passed"] for g in gates)}, flush=True)
