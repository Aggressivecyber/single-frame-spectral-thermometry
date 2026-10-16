"""Bounded independent-channel image simulation; no training or spectral fitting."""
import runtime
from pathlib import Path
import argparse
import copy
import json
import math
import time
import numpy as np
import torch
from research_filter_decode.operator import CodedOperator, filt


def images(seed, shape):
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(-1, 1, shape[0], endpoint=False),
                       np.linspace(-1, 1, shape[1], endpoint=False), indexing="ij")
    result = []
    for channel in range(3):
        a = np.full(shape, 0.20)
        for _ in range(4):
            cx, cy = rng.uniform(-0.6, 0.6, 2)
            angle = rng.uniform(0, np.pi)
            u = (x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)
            v = -(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)
            a += rng.uniform(0.4, 0.9)*np.exp(-0.5*((u/rng.uniform(.12, .35))**2
                                                        +(v/rng.uniform(.025, .12))**2))
        a += .08*(np.sin(2*np.pi*(x*(5+channel)+y*(3-channel))) > .5)
        a += .12*((np.abs(x-rng.uniform(-.3, .3)) < .2)
                   & (np.abs(y-rng.uniform(-.3, .3)) < .2))
        result.append(a/a.mean())
    return np.stack(result)


def metrics(pred, target):
    error = pred-target
    pc = pred-pred.mean((-2, -1), keepdim=True)
    tc = target-target.mean((-2, -1), keepdim=True)
    out = []
    for k in range(3):
        out.append({"channel": k,
                    "relative_L2": float(error[k].norm()/target[k].norm()),
                    "mean_bias_fraction": float(error[k].mean()/target[k].mean()),
                    "centered_relative_L2": float((pc[k]-tc[k]).norm()/tc[k].norm()),
                    "contrast_gain": float((pc[k]*tc[k]).sum()/tc[k].square().sum()),
                    "rmse_e": float(error[k].square().mean().sqrt())})
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
torch.set_num_threads(2)
started = time.monotonic()
root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/"research_filter_decode/protocol.json").read_text())
cfg = protocol["model"]
op = CodedOperator(cfg, torch.float64).to(args.device)
out = root/"reports"/protocol["run_id"]
out.mkdir(parents=True, exist_ok=True)
unit = torch.tensor(images(protocol["design"]["image_seed"], cfg["shape"]),
                    dtype=torch.float64, device=args.device)
target = op.object_blur(unit)

if args.smoke:
    recovery = op.decode(op.encode_images(target))
    error = float((recovery-target).norm()/target.norm())
    inactive = []
    for k in range(3):
        active = torch.zeros_like(target)
        active[k] = target[k]
        decoded = op.decode(op.encode_images(active))
        inactive.extend(float(decoded[j].norm()/target[k].norm()) for j in range(3) if j != k)
    rng = np.random.default_rng(24601)
    probe = torch.tensor(rng.standard_normal(cfg["shape"]), dtype=torch.float64, device=args.device)
    left = (op.encode_images(target)*probe).sum()
    right = (target*op.images_adjoint(probe)).sum()
    adjoint = float((left-right).abs()/torch.maximum(left.abs(), right.abs()))
    checks = {"time": runtime.now(), "device": args.device, "dtype": "float64",
              "relative_L2": error, "max_inactive_channel_leakage": max(inactive),
              "adjoint_relative_error": adjoint, "geometry": op.geometry,
              "number_of_complete_decodes": 4,
              "all_passed": error < 1e-10 and max(inactive) < 1e-10 and adjoint < 1e-10,
              "wall_s": time.monotonic()-started}
    runtime.write_json(out/"CPU_CONTRACTS.json", checks)
    print(json.dumps(checks), flush=True)
    if not checks["all_passed"]:
        raise RuntimeError("Image-only operator contracts failed")
    raise SystemExit(0)

# This is an external artifact boundary: the transferred source must match registration.
registration = json.loads((root/"research_filter_decode/REGISTRATION.json").read_text())
for name, expected in registration["source_sha256"].items():
    if runtime.sha256(root/name) != expected:
        raise RuntimeError(f"Frozen source differs: {name}")
ledger_path = out/"decode_calls.jsonl"
ledger = ledger_path.open("x", encoding="utf-8")
calls = 0


def decode(operator, raw, role, identity):
    global calls
    calls += 1
    ledger.write(json.dumps({"call": calls, "role": role, "identity": identity})+"\n")
    ledger.flush()
    return operator.decode(raw)


design = protocol["design"]
levels = design["bright_channel_mean_e"]
profiles = design["brightness_profiles"]
raw_dir = out/"raw_adc"
raw_dir.mkdir(exist_ok=False)
originals, targets, rows, noiseless, leakage, stress = [], [], [], [], [], []
max_code = 2**cfg["adc_bits"]-1
adc_step = cfg["full_well_e"]/(max_code-cfg["black_level_DN"])
phase_cfg, gain_cfg = copy.deepcopy(cfg), copy.deepcopy(cfg)
phase_cfg["phase_rad"] = [v+math.radians(2) for v in cfg["phase_rad"]]
gain_cfg["channel_gain"] = [v*1.05 for v in cfg["channel_gain"]]
phase_op = CodedOperator(phase_cfg, torch.float64).to(args.device)
gain_op = CodedOperator(gain_cfg, torch.float64).to(args.device)

for mother in range(design["mothers"]):
    unit = torch.tensor(images(design["image_seed"]+mother*1009, cfg["shape"]),
                        dtype=torch.float64, device=args.device)
    blurred = op.object_blur(unit)
    originals.append(unit.cpu().numpy())
    targets.append(blurred.cpu().numpy())
    for active_k in range(3):
        active = torch.zeros_like(blurred)
        active[active_k] = blurred[active_k]
        got = decode(op, op.encode_images(active), "single_channel", [mother, active_k])
        leakage.append({"mother": mother, "active": active_k,
                        "ratios": [float(got[j].norm()/active[active_k].norm())
                                   if j != active_k else float((got[j]-active[j]).norm()/active[j].norm())
                                   for j in range(3)]})
    for level_index, level in enumerate(levels):
        for profile_index, (profile, factors) in enumerate(profiles.items()):
            target = blurred*torch.tensor(factors, dtype=torch.float64, device=args.device)[:, None, None]*level
            expected = op.encode_images(target)
            if float(target.min()) < 0 or float(expected.min()) < 0:
                raise ValueError("Negative physical photon expectation")
            ident = [mother, level, profile]
            clean = decode(op, expected, "noiseless", ident)
            noiseless.append({"identity": ident, "channels": metrics(clean, target),
                              "raw_min_e": float(expected.min()), "raw_max_e": float(expected.max())})
            raw_frames = []
            mu = expected.cpu().numpy()
            for repeat in range(design["noise_repeats"]):
                case = (((mother*len(levels)+level_index)*len(profiles)+profile_index)
                        *design["noise_repeats"]+repeat)
                seed = design["noise_seed"]+case*1009
                rng = np.random.default_rng(seed)
                analog = rng.poisson(mu)+rng.normal(0, cfg["read_noise_e"], mu.shape)
                unbounded_dn = np.rint(analog/adc_step+cfg["black_level_DN"])
                saturated = int(np.count_nonzero(unbounded_dn >= max_code))
                below_zero = int(np.count_nonzero(unbounded_dn < 0))
                dn = unbounded_dn.clip(0, max_code).astype(np.uint16)
                raw_frames.append(dn)
                measured = torch.tensor((dn.astype(np.float64)-cfg["black_level_DN"])*adc_step,
                                        device=args.device)
                got = decode(op, measured, "noisy", ident+[repeat, seed])
                rows.append({"mother": mother, "bright_mean_e": level, "profile": profile,
                             "repeat": repeat, "noise_seed": seed, "saturated_samples": saturated,
                             "below_zero_samples": below_zero, "channels": metrics(got, target)})
                if mother == 0 and level == 4000 and repeat == 0:
                    np.savez_compressed(out/f"example_{profile}.npz", pre_optics=unit.cpu().numpy()
                                        *np.array(factors)[:, None, None]*level,
                                        target=target.cpu().numpy(), raw_e=measured.cpu().numpy(),
                                        decoded=got.cpu().numpy())
            np.savez_compressed(raw_dir/f"m{mother:02d}_e{level}_{profile}.npz",
                                raw_DN=np.stack(raw_frames))
    for name, operator, target in [("no_object_limit", op, unit*4000),
                                   ("phase_plus_2deg", phase_op, blurred*4000),
                                   ("gain_plus_5percent", gain_op, blurred*4000)]:
        got = decode(operator, op.encode_images(target), "stress", [mother, name])
        stress.append({"mother": mother, "case": name, "channels": metrics(got, target)})
    runtime.write_json(out/"PROGRESS.json", {"time": runtime.now(), "mothers_completed": mother+1,
                                            "complete_decodes": calls, "wall_s": time.monotonic()-started})
    print(f"mother {mother+1}/{design['mothers']} completed; decodes={calls}", flush=True)

ledger.close()
np.savez_compressed(out/"independent_images.npz", pre_optics=np.stack(originals),
                    post_optics=np.stack(targets))
summary = []
for level in levels:
    for profile in profiles:
        subset = [r for r in rows if r["bright_mean_e"] == level and r["profile"] == profile]
        for k in range(3):
            item = {"bright_mean_e": level, "profile": profile, "channel": k,
                    "independent_mothers": design["mothers"], "observations": len(subset),
                    "saturated_samples": sum(r["saturated_samples"] for r in subset)}
            for key in ["relative_L2", "mean_bias_fraction", "centered_relative_L2", "contrast_gain", "rmse_e"]:
                values = np.array([r["channels"][k][key] for r in subset])
                item[key+"_mean"] = float(values.mean())
                item[key+"_p95"] = float(np.quantile(np.abs(values) if key == "mean_bias_fraction" else values, .95))
            summary.append(item)
primary = [r for r in summary if r["bright_mean_e"] == 4000 and r["profile"] == "balanced"]
max_clean = max(c["relative_L2"] for r in noiseless for c in r["channels"])
max_leak = max(max(r["ratios"]) for r in leakage)
primary_pass = all(r["relative_L2_p95"] <= .05 and r["mean_bias_fraction_p95"] <= .02
                   and r["centered_relative_L2_p95"] <= .10 and r["saturated_samples"] == 0 for r in primary)
detail = [float(np.linalg.norm(t-o)/np.linalg.norm(o)) for o, t in zip(np.stack(originals).reshape(-1, *cfg["shape"]),
                                                                          np.stack(targets).reshape(-1, *cfg["shape"]))]
receipt = {"time": runtime.now(), "evidence_level": protocol["evidence_level"], "device": args.device,
           "torch_version": torch.__version__, "numpy_version": np.__version__,
           "complete_decodes": calls, "noisy_observations": len(rows), "all_finite": True,
           "geometry": op.geometry, "noiseless_max_relative_L2": max_clean,
           "single_channel_max_error_or_leakage": max_leak,
           "primary_operating_point_passed": primary_pass,
           "engineering_screen_passed": primary_pass and max_clean < 1e-10 and max_leak < 1e-10,
           "optical_detail_loss_relative_L2": {"all_channels_and_mothers": detail, "mean": float(np.mean(detail)), "max": max(detail)},
           "peak_gpu_memory_MiB": torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0,
           "wall_s": time.monotonic()-started, "training_updates": 0,
           "hardware_validation": False, "summary": summary}
runtime.write_json(out/"METRICS.json", {"noisy": rows, "noiseless": noiseless, "single_channel": leakage, "stress": stress})
runtime.write_json(out/"RESULTS.json", receipt)
if calls != protocol["resources"]["expected_total_complete_decodes"]:
    raise RuntimeError("Complete-decode count differs from registration")
if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > protocol["resources"]["maximum_new_artifact_bytes"]:
    raise RuntimeError("Registered artifact limit exceeded")
runtime.write_json(out/"COMPLETION.json", {"time": runtime.now(), "process_completed": True,
                                         "engineering_screen_passed": receipt["engineering_screen_passed"],
                                         "complete_decodes": calls, "wall_s": time.monotonic()-started})
print(json.dumps({k:v for k,v in receipt.items() if k != "summary"}), flush=True)
