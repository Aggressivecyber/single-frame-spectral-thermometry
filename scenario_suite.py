"""Fixed-weight synthetic scenario evaluation; no training or model selection."""
import runtime
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from physics import SpectralPhysics
from infer import load_model, DEFAULT_CHECKPOINT, CHECKPOINT_SHA256

CONDITIONS = ("signal", "null_clean", "null_background")
FAMILIES = (
    {"name": "original_blob", "axis": "matched synthetic control", "shape": "blob"},
    {"name": "multiple_peaks", "axis": "geometry transfer", "shape": "multiple"},
    {"name": "thin_filament", "axis": "geometry transfer", "shape": "filament"},
    {"name": "annulus", "axis": "geometry transfer", "shape": "annulus"},
    {"name": "fragmented", "axis": "geometry transfer", "shape": "fragmented"},
    {"name": "near_periodic_edge", "axis": "periodic boundary", "shape": "edge"},
    {"name": "weak_emission", "axis": "signal intensity", "peak_e": [30., 60.]},
    {"name": "strong_emission", "axis": "signal intensity", "peak_e": [600., 900.]},
    {"name": "cold_continuum", "axis": "temperature transfer", "base_temperature_K": 2400.},
    {"name": "hot_continuum", "axis": "temperature transfer", "base_temperature_K": 3300.},
    {"name": "temperature_front", "axis": "temperature geometry", "temperature_front": True},
    {"name": "bright_saturated", "axis": "saturation stress", "base_temperature_K": 3600., "amplitude": 1.15},
    {"name": "background_minus_5pct", "axis": "background mismatch", "background_fraction": -.05},
    {"name": "background_plus_10pct", "axis": "background mismatch", "background_fraction": .10},
    {"name": "channel_gain_1pct", "axis": "calibration mismatch", "gain_error": .01},
    {"name": "channel_gain_3pct", "axis": "calibration mismatch", "gain_error": .03},
    {"name": "carrier_phase_0p1rad", "axis": "calibration mismatch", "phase_error_rad": .10},
    {"name": "optical_bandwidth_0p06", "axis": "calibration mismatch", "object_cutoff_cpp": .06},
    {"name": "read_noise_30e", "axis": "noise mismatch", "read_noise_e": 30.},
    {"name": "non_gray_continuum", "axis": "continuum model mismatch", "continuum_multiplier": [1.12, 1.10, 1.06, 1.03, 1., .97, 1.]},
)
THRESHOLDS = {"temperature_RMSE_K_max": 75., "relative_L2_max": .25,
              "absolute_integral_bias_max": .10, "null_FP_max": .01,
              "background_null_FP_max": .01, "coverage_min": .98,
              "detection_threshold_e": 60.}
REPRESENTATIVES = {0: "signal", 1: "signal", 2: "signal", 4: "signal", 6: "signal",
                   8: "signal", 9: "signal", 10: "signal", 11: "signal",
                   13: "null_background", 15: "null_clean", 19: "null_clean"}


def specification(replicates=8, seed=621001, max_wall_s=900):
    return {"schema": "expanded-synthetic-scenarios-v1", "checkpoint_sha256": CHECKPOINT_SHA256,
            "families": list(FAMILIES), "conditions": list(CONDITIONS),
            "replicates_per_condition_per_family": replicates, "seed": seed,
            "scene_id_base": 910000, "frames": len(FAMILIES)*len(CONDITIONS)*replicates,
            "optimizer_updates": 0, "checkpoint_selection": "none; published weights fixed",
            "models": ["published_model"], "thresholds": THRESHOLDS,
            "max_wall_s": max_wall_s, "max_GPU_allocated_MiB": 3500., "max_host_RSS_MiB": 4096.,
            "representatives": {FAMILIES[index]["name"]: {"replicate": 0, "condition": condition}
                                for index, condition in REPRESENTATIVES.items()},
            "truth": "post-acquisition-object-blur effective continuum color temperature and signed template coefficient",
            "calibration_policy": "perturb acquisition only; reconstruct with unchanged nominal checkpoint",
            "noise_pairing": "same generator seed for three conditions within a replicate; Poisson draws depend on expectation",
            "coverage_policy": "template error uses full image with invalid predictions filled by zero; temperature error is conditional on jointly valid pixels",
            "scope": "synthetic single-template engineering scenarios; no independent measured spectra or hardware"}


def acquisition_physics(nominal, family):
    cfg = copy.deepcopy(nominal.cfg)
    alternating = [1., -1., 1., -1., 1., -1., 1.]
    if "gain_error" in family:
        cfg["channel_gain"] = [g*(1.+family["gain_error"]*s) for g, s in zip(cfg["channel_gain"], alternating)]
    if "phase_error_rad" in family:
        cfg["phase_rad"] = [p+family["phase_error_rad"]*s for p, s in zip(cfg["phase_rad"], alternating)]
    for key in ("object_cutoff_cpp", "read_noise_e"):
        if key in family:
            cfg[key] = family[key]
    return SpectralPhysics(cfg, nominal.M.dtype).to(nominal.M.device)


def make_case(nominal, acquisition, family_index, replicate, condition, seed=621001,
              emission_pattern=None, emission_peak_e=None, background_pattern=None,
              structured_background_peak_e=0., background_fraction_override=None,
              scene_id_override=None):
    family = FAMILIES[family_index]
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    device, dtype = nominal.M.device, nominal.M.dtype
    scene_id = 910000 + family_index*1000 + replicate if scene_id_override is None else scene_id_override
    noise_seed = seed*10000000 + scene_id
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    def draw(low, high):
        return low+(high-low)*float(torch.rand((), generator=generator, device=device, dtype=dtype))
    yy, xx = acquisition.background_basis[2], acquisition.background_basis[1]
    cx, cy = draw(-.4, .4), draw(-.4, .4)
    width = draw(.27, .35)
    def gaussian(x0, y0, sigma):
        return torch.exp(-.5*((xx-x0).square()+(yy-y0).square())/sigma**2)
    shape = family.get("shape", "blob")
    pattern = gaussian(cx, cy, width)
    if shape == "multiple":
        for _ in range(2):
            pattern += draw(.4, .9)*gaussian(draw(-.6, .6), draw(-.6, .6), draw(.14, .25))
    elif shape == "filament":
        width = draw(.04, .07)
        angle = draw(-math.pi, math.pi)
        u = math.cos(angle)*(xx-cx)+math.sin(angle)*(yy-cy)
        v = -math.sin(angle)*(xx-cx)+math.cos(angle)*(yy-cy)
        pattern = torch.exp(-.5*((u/.6).square()+((v-.15*torch.sin(6*u))/width).square()))
    elif shape == "annulus":
        width = draw(.06, .10)
        radius = draw(.3, .5)
        pattern = torch.exp(-.5*((torch.sqrt((xx-cx).square()+(yy-cy).square())-radius)/width).square())
    elif shape == "fragmented":
        width = draw(.04, .08)
        pattern = torch.zeros_like(xx)
        for _ in range(6):
            pattern += draw(.3, 1.)*gaussian(draw(-.65, .65), draw(-.65, .65), width)
    elif shape == "edge":
        cx = draw(.82, .94)
        cy = draw(-.5, .5)
        pattern = gaussian(cx, cy, width)
    low, high = family.get("peak_e", [180., 240.])
    peak = draw(low, high)
    if emission_pattern is not None:
        pattern = torch.as_tensor(emission_pattern, device=device, dtype=dtype)
        if pattern.shape != xx.shape or not torch.isfinite(pattern).all() or pattern.min() < 0:
            raise ValueError("Invalid supplied spatial pattern")
    if emission_peak_e is not None:
        peak = emission_peak_e
    pattern = pattern/pattern.max().clamp_min(1e-12)*peak
    temperature = family.get("base_temperature_K", 2950.35) + 120*torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy) + draw(-7.5, 7.5)
    if family.get("temperature_front"):
        temperature = 2950.35+300*torch.tanh(14*(xx-.12*torch.sin(3*yy)))
    amplitude = family.get("amplitude", .72)+.08*torch.sin(torch.pi*xx)
    state = torch.zeros(1, 5, *xx.shape, device=device, dtype=dtype)
    state[:, 0] = (temperature-acquisition.T0)/acquisition.Tscale
    state[:, 1] = amplitude.log()
    if condition == "signal":
        state[:, 4] = pattern/acquisition.Sscale
    assert int(acquisition.active.sum()) == 1 and bool(acquisition.active[2])
    beta = torch.tensor([[.45, .06, -.04]], device=device, dtype=dtype)
    continuum, species, background, _ = acquisition.components(state, beta)
    if "continuum_multiplier" in family:
        continuum = continuum*torch.tensor(family["continuum_multiplier"], device=device, dtype=dtype)[None, :, None, None]
    mismatch = .05 if condition == "null_background" else 0.
    if "background_fraction" in family and condition != "null_clean":
        mismatch = family["background_fraction"]
    if background_fraction_override is not None:
        mismatch = background_fraction_override
    extra = torch.zeros_like(continuum)
    extra[:, 2] = mismatch*continuum[:, 2]
    if background_pattern is not None and structured_background_peak_e:
        bgshape = torch.as_tensor(background_pattern, device=device, dtype=dtype)
        if bgshape.shape != xx.shape or not torch.isfinite(bgshape).all() or bgshape.min() < 0:
            raise ValueError("Invalid supplied background pattern")
        extra = extra + acquisition.q[None, :, None, None]*bgshape[None, None]*structured_background_peak_e
    expected = acquisition.op(continuum+species+background+extra)
    if not torch.isfinite(expected).all() or expected.min() < 0:
        raise ValueError("Invalid photon expectation")
    raw_e = torch.poisson(expected, generator=generator)+torch.randn(expected.shape, device=device, dtype=dtype,
                         generator=generator)*acquisition.cfg["read_noise_e"]
    raw_dn = (raw_e/acquisition.adc_step+acquisition.cfg["black_level_DN"]).round().clamp(0, 2**acquisition.cfg["adc_bits"]-1)
    continuum_images = acquisition.op.object_blur(continuum)
    color, valid = nominal.color_temperature(continuum_images)
    target = acquisition.op.object_blur(state[:, 4:5]*acquisition.Sscale)[:, 0]
    metadata = {"family": family["name"], "axis": family["axis"], "family_index": family_index,
                "scene_id": scene_id, "replicate": replicate, "condition": condition, "seed": seed,
                "noise_seed": noise_seed, "shape": shape, "width_norm": width,
                "preblur_peak_e": peak if condition == "signal" else 0.,
                "postblur_peak_e": float(target.max()), "background_error_fraction": mismatch,
                "structured_background_peak_e": structured_background_peak_e,
                "acquisition_channel_gain": acquisition.cfg["channel_gain"],
                "acquisition_phase_rad": acquisition.cfg["phase_rad"],
                "acquisition_object_cutoff_cpp": acquisition.cfg["object_cutoff_cpp"],
                "acquisition_read_noise_e": acquisition.cfg["read_noise_e"],
                "continuum_multiplier": family.get("continuum_multiplier", [1.]*7),
                "true_temperature_min_K": float(temperature.min()), "true_temperature_max_K": float(temperature.max()),
                "effective_color_temperature_min_K": float(color.min()), "effective_color_temperature_max_K": float(color.max()),
                "mean_raw_expected_e": float(expected.mean()), "maximum_raw_expected_e": float(expected.max()),
                "scope": "synthetic engineering fixture"}
    return {"raw_dn": raw_dn[:, None], "truth": {"temperature_color_K": color, "template_e": target, "valid": valid},
            "metadata": metadata}


def evaluate_case(model, frozen, use_prior):
    device = next(model.parameters()).device
    raw = frozen["raw_dn"].to(device)
    truth = {key: value.to(device) for key, value in frozen["truth"].items()}
    with torch.no_grad():
        predicted = model(raw, use_prior=use_prior)
    if any(not torch.isfinite(value).all() for value in predicted.values()):
        raise RuntimeError("Nonfinite prediction")
    valid = predicted["valid"] & truth["valid"]
    estimate, target = predicted["species_signed_e"][:, 2], truth["template_e"]
    filled = torch.where(valid, estimate, torch.zeros_like(estimate))
    error = filled-target
    energy = target.square().sum()
    projection = (filled*target).sum()/energy.clamp_min(1e-12)
    residual = predicted["reprojection_residual_e"]
    variance = predicted["expected_e"].clamp_min(0)+model.physics.extra_variance
    signal_region = target >= THRESHOLDS["detection_threshold_e"]
    positive = (estimate > THRESHOLDS["detection_threshold_e"]) & valid
    row = {**frozen["metadata"], "temperature_sse": float((predicted["temperature_color_K"]-truth["temperature_color_K"])[valid].square().sum()),
           "valid_count": int(valid.sum()), "pixels": valid.numel(), "truth_valid_count": int(truth["valid"].sum()),
           "species_sse": float(error.square().sum()), "species_truth_sq": float(energy),
           "species_error_sum": float(error.sum()), "species_truth_sum": float(target.sum()),
           "positive_count": int(positive.sum()), "saturated_pixels": int(predicted["saturated"].sum()),
           "amplitude_projection": float(projection), "orthogonal_sse": float((filled-projection*target).square().sum()),
           "signal_region_pixels": int(signal_region.sum()), "true_positive_count": int((positive & signal_region).sum()),
           "raw_residual_sse": float(residual.square().sum()), "raw_whitened_sse": float((residual.square()/variance).sum()),
           "template_relative_L2": float((error.square().sum()/energy).sqrt()) if float(energy) > 0 else None,
           "coverage": float(valid.double().mean())}
    arrays = {"raw_DN": raw[0, 0].cpu().numpy(), "truth_template_e": target[0].cpu().numpy(),
              "pred_template_e": estimate[0].cpu().numpy(), "truth_color_temperature_K": truth["temperature_color_K"][0].cpu().numpy(),
              "pred_color_temperature_K": predicted["temperature_color_K"][0].cpu().numpy(),
              "valid": valid[0].cpu().numpy(), "saturated": predicted["saturated"][0].cpu().numpy()}
    return row, arrays


def aggregate(rows):
    def total(key):
        return sum(row[key] for row in rows)
    energy, truth_sum = total("species_truth_sq"), total("species_truth_sum")
    valid_count = total("valid_count")
    individual_l2 = [row["template_relative_L2"] for row in rows if row["template_relative_L2"] is not None]
    return {"cases": len(rows), "temperature_RMSE_K": (total("temperature_sse")/valid_count)**.5 if valid_count else None,
            "template_relative_L2": (total("species_sse")/energy)**.5 if energy else None,
            "template_integral_bias_fraction": total("species_error_sum")/truth_sum if truth_sum else None,
            "null_positive_fraction_valid": total("positive_count")/valid_count if valid_count else None,
            "coverage": valid_count/total("pixels"), "minimum_case_coverage": min(row["coverage"] for row in rows),
            "saturated_pixels": total("saturated_pixels"),
            "raw_reprojection_RMSE_e": (total("raw_residual_sse")/total("pixels"))**.5,
            "raw_noise_normalized_RMSE": (total("raw_whitened_sse")/total("pixels"))**.5,
            "orthogonal_relative_L2": (total("orthogonal_sse")/energy)**.5 if energy else None,
            "signal_region_recall_valid": total("true_positive_count")/total("signal_region_pixels") if total("signal_region_pixels") else None,
            "median_case_L2": float(np.median(individual_l2)) if individual_l2 else None,
            "maximum_case_L2": max(individual_l2) if individual_l2 else None}


def screens(grouped):
    p, n, b = (grouped[condition] for condition in CONDITIONS)
    return {"temperature": p["temperature_RMSE_K"] is not None and p["temperature_RMSE_K"] <= THRESHOLDS["temperature_RMSE_K_max"],
            "feature_L2": p["template_relative_L2"] <= THRESHOLDS["relative_L2_max"],
            "feature_integral_bias": abs(p["template_integral_bias_fraction"]) <= THRESHOLDS["absolute_integral_bias_max"],
            "clean_null_FP": n["null_positive_fraction_valid"] is not None and n["null_positive_fraction_valid"] <= THRESHOLDS["null_FP_max"],
            "background_FP": b["null_positive_fraction_valid"] is not None and b["null_positive_fraction_valid"] <= THRESHOLDS["background_null_FP_max"],
            "coverage": min(group["minimum_case_coverage"] for group in grouped.values()) >= THRESHOLDS["coverage_min"],
            "no_saturation": sum(group["saturated_pixels"] for group in grouped.values()) == 0}


def grouped_report(rows):
    report = {}
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family["name"]]
        grouped = {condition: aggregate([row for row in subset if row["condition"] == condition]) for condition in CONDITIONS}
        result = screens(grouped)
        report[family["name"]] = {"axis": family["axis"], "grouped": grouped, "screens": result,
                                  "passed_screens": sum(result.values()), "all_screens_passed": all(result.values())}
    return report


def state_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=runtime.ROOT / "outputs/expanded_001")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=621001)
    parser.add_argument("--max-wall-s", type=float, default=900.)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.replicates < 1 or args.replicates >= 1000:
        raise ValueError("Replicates must be in [1, 999]")
    registration = specification(args.replicates, args.seed, args.max_wall_s)
    if args.describe:
        print(json.dumps(registration, ensure_ascii=False))
        return
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default bounded evaluation")
    output = runtime.check_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fixtures, representatives = output / "frozen", output / "representatives"
    fixtures.mkdir()
    representatives.mkdir()
    registration["registered_utc"] = runtime.now()
    registration["source_sha256"] = {path.name: runtime.sha256(path) for path in sorted(runtime.ROOT.glob("*.py"))}
    runtime.write_json(output / "registration.json", registration)
    start = time.perf_counter()
    model = load_model(DEFAULT_CHECKPOINT, args.device)
    before = state_digest(model)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def check_budget():
        if time.perf_counter()-start > args.max_wall_s:
            raise RuntimeError("Registered wall-time budget exceeded")
        if args.device == "cuda" and torch.cuda.max_memory_allocated()/2**20 > registration["max_GPU_allocated_MiB"]:
            raise RuntimeError("Registered GPU budget exceeded")
    manifest = []
    # Complete and hash the entire new fixture set before the first inference.
    for index, family in enumerate(FAMILIES):
        acquisition = acquisition_physics(model.physics, family)
        for replicate in range(args.replicates):
            for condition in CONDITIONS:
                check_budget()
                case = make_case(model.physics, acquisition, index, replicate, condition, args.seed)
                frozen = {"raw_dn": case["raw_dn"].cpu(), "truth": {key: value.cpu() for key, value in case["truth"].items()},
                          "metadata": case["metadata"]}
                path = fixtures / f"{index:02d}_{replicate:02d}_{condition}.pt"
                torch.save(frozen, path)
                manifest.append({"path": str(path.relative_to(output)), "sha256": runtime.sha256(path),
                                 "bytes": path.stat().st_size, **case["metadata"]})
        del acquisition
    assert len(manifest) == registration["frames"]
    runtime.write_json(output / "fixture_manifest.json", {"frozen_before_inference": True, "cases": manifest})
    rows = []
    for count, fixture in enumerate(manifest, 1):
        check_budget()
        path = output / fixture["path"]
        assert runtime.sha256(path) == fixture["sha256"]
        frozen = torch.load(path, map_location="cpu", weights_only=True)
        row, arrays = evaluate_case(model, frozen, use_prior=True)
        row["input_fixture_sha256"] = fixture["sha256"]
        rows.append(row)
        if fixture["replicate"] == 0 and REPRESENTATIVES.get(fixture["family_index"]) == fixture["condition"]:
            merged = arrays
            dest = representatives / f"{fixture['family']}_{fixture['condition']}.npz"
            np.savez_compressed(dest, **merged)
            with np.load(dest, allow_pickle=False) as reopened:
                assert all(np.array_equal(reopened[key], value) for key, value in merged.items())
        if count % 24 == 0 or count == len(manifest):
            runtime.write_json(output / "progress.json", {"utc": runtime.now(), "evaluated_frames": count,
                               "registered_frames": len(manifest), "seconds": time.perf_counter()-start})
            print(json.dumps({"frames": count, "total": len(manifest), "seconds": time.perf_counter()-start}), flush=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
    assert state_digest(model) == before
    assert runtime.sha256(DEFAULT_CHECKPOINT) == registration["checkpoint_sha256"]
    assert all(runtime.sha256(runtime.ROOT / key) == value for key, value in registration["source_sha256"].items())
    results = {"checkpoint_sha256": CHECKPOINT_SHA256, "registration": registration,
               "results": grouped_report(rows), "rows": rows,
               "notes": ["Temperature RMSE is conditional on jointly valid pixels; inspect coverage and saturation.",
                         "Null false-positive rates use signed coefficient >60 e; relative L2 keeps negative predictions.",
                         "Internal screens applied independently per family; stress failures do not change the checkpoint."]}
    runtime.write_json(output / "results.json", results)
    try:
        import resource
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except ImportError:
        rss_mib = None
    receipt = {"success": True, "utc": runtime.now(), "frames": len(manifest), "inference_calls": len(manifest),
               "optimizer_updates": 0, "weights_and_sources_unchanged": True, "state_tensor_sha256": before,
               "checkpoint_sha256": runtime.sha256(DEFAULT_CHECKPOINT), "device": args.device,
               "torch_version": torch.__version__, "CUDA_version": torch.version.cuda,
               "GPU_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
               "seconds": time.perf_counter()-start,
               "peak_GPU_allocated_MiB": torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0.,
               "peak_host_RSS_MiB": rss_mib, "frozen_before_inference": True,
               "all_families_passing_screens": sum(item["all_screens_passed"] for item in results["results"].values()),
               "representatives_reopened_exactly": True, "formal_scientific_acceptance": "not_accepted"}
    check_budget()
    if rss_mib is not None and rss_mib > registration["max_host_RSS_MiB"]:
        raise RuntimeError("Registered host memory budget exceeded")
    runtime.write_json(output / "evaluation_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
