"""Evaluate the published model on experimental shapes with constructed measurements."""
import runtime
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from infer import load_model, DEFAULT_CHECKPOINT, CHECKPOINT_SHA256
from scenario_suite import make_case, evaluate_case, aggregate, state_digest, THRESHOLDS

CONDITIONS = ("signal_nominal", "signal_weak", "null_clean", "null_structured_background")


def report(rows):
    results = {}
    for partition in sorted({row["partition"] for row in rows}):
        subset = [row for row in rows if row["partition"] == partition]
        grouped = {condition: aggregate([row for row in subset if row["condition"] == condition]) for condition in CONDITIONS}
        p, w, n, b = (grouped[condition] for condition in CONDITIONS)
        screens = {"temperature_assumed_fixture": p["temperature_RMSE_K"] is not None and p["temperature_RMSE_K"] <= THRESHOLDS["temperature_RMSE_K_max"],
                   "nominal_feature_L2": p["template_relative_L2"] <= THRESHOLDS["relative_L2_max"],
                   "nominal_integral_bias": abs(p["template_integral_bias_fraction"]) <= THRESHOLDS["absolute_integral_bias_max"],
                   "clean_null_FP": n["null_positive_fraction_valid"] is not None and n["null_positive_fraction_valid"] <= THRESHOLDS["null_FP_max"],
                   "structured_background_FP": b["null_positive_fraction_valid"] is not None and b["null_positive_fraction_valid"] <= THRESHOLDS["background_null_FP_max"],
                   "coverage": min(g["minimum_case_coverage"] for g in grouped.values()) >= THRESHOLDS["coverage_min"],
                   "no_saturation": sum(g["saturated_pixels"] for g in grouped.values()) == 0}
        results[partition] = {"dataset": subset[0]["dataset"], "source": subset[0]["source"], "grouped": grouped,
                              "screens": screens, "passed_screens": sum(screens.values()), "all_screens_passed": all(screens.values()),
                              "weak_feature_L2_screen": w["template_relative_L2"] <= THRESHOLDS["relative_L2_max"]}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-wall-s", type=float, default=900.)
    args = parser.parse_args()
    torch.set_num_threads(2)
    shape_manifest = json.loads((args.shapes/"manifest.json").read_text(encoding="utf-8"))
    output = runtime.check_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fixtures, representatives = output/"frozen", output/"representatives"
    fixtures.mkdir()
    representatives.mkdir()
    registration = {"schema": "real-image-shape-transfer-v1", "registered_utc": runtime.now(),
                    "checkpoint_sha256": CHECKPOINT_SHA256, "models": ["published_model"], "optimizer_updates": 0,
                    "measured_shape_frames": shape_manifest["measured_shape_frames"], "source_groups": 6, "acquisition_datasets": 3,
                    "frames": len(shape_manifest["records"])*len(CONDITIONS), "conditions": list(CONDITIONS),
                    "nominal_template_peak_e": 210., "weak_template_peak_e": 45., "structured_background_peak_e": 100.,
                    "seed": 622001, "scene_id_base": 1200000, "thresholds": THRESHOLDS,
                    "shape_manifest_sha256": runtime.sha256(args.shapes/"manifest.json"),
                    "source_sha256": {path.name: runtime.sha256(path) for path in sorted(runtime.ROOT.glob("*.py"))},
                    "max_wall_s": args.max_wall_s, "max_GPU_allocated_MiB": 3500., "max_host_RSS_MiB": 4096.,
                    "representatives": "replicate 0 per source partition, nominal, weak and structured-background conditions",
                    "source_pairing": "constructed independent shape/background combinations; synchronization not established",
                    "measured": "image spatial shape/texture only; native source and processed hashes recorded",
                    "synthetic": "continuum temperature, photon scale, seven-band spectrum, AlO-named template response, spatial code and noise",
                    "scope": "experimental-image-derived simulation, not direct real seven-band measurement or real temperature/AlO accuracy"}
    runtime.write_json(output/"registration.json", registration)
    start = time.perf_counter()
    model = load_model(DEFAULT_CHECKPOINT, args.device)
    before = state_digest(model)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def budget():
        if time.perf_counter()-start > args.max_wall_s:
            raise RuntimeError("Wall budget exceeded")
        if args.device == "cuda" and torch.cuda.max_memory_allocated()/2**20 > registration["max_GPU_allocated_MiB"]:
            raise RuntimeError("GPU budget exceeded")
    manifest = []
    for index, source in enumerate(shape_manifest["records"]):
        budget()
        shape_file = args.shapes/source["processed_file"]
        bg = source["background_source"]
        bg_file = args.shapes/bg["processed_file"]
        assert runtime.sha256(shape_file) == source["processed_sha256"]
        assert runtime.sha256(bg_file) == bg["processed_sha256"]
        with np.load(shape_file, allow_pickle=False) as opened:
            spatial_shape = opened["shape"].copy()
        with np.load(bg_file, allow_pickle=False) as opened:
            background_shape = opened["shape"].copy()
        for kind_index, condition in enumerate(CONDITIONS):
            peak = 45. if condition == "signal_weak" else 210.
            mapped = "signal" if condition.startswith("signal") else ("null_background" if condition == "null_structured_background" else "null_clean")
            case = make_case(model.physics, model.physics, 0, index, mapped, seed=622001,
                             emission_pattern=spatial_shape, emission_peak_e=peak,
                             background_pattern=background_shape if condition == "null_structured_background" else None,
                             structured_background_peak_e=100. if condition == "null_structured_background" else 0.,
                             background_fraction_override=0., scene_id_override=1200000+index*10+kind_index)
            case["metadata"].update({"partition": source["partition"], "dataset": source["dataset"], "channel": source["channel"],
                                     "family": source["partition"], "condition": condition, "replicate": source["replicate"],
                                     "source_frame_index": source["frame_index"], "source": source["source"],
                                     "source_shape_sha256": source["processed_sha256"], "source_archive_sha256": source["source_sha256"],
                                     "source_preprocessing": source["preprocessing"], "background_source": bg,
                                     "scope": registration["scope"]})
            frozen = {"raw_dn": case["raw_dn"].cpu(), "truth": {key: value.cpu() for key, value in case["truth"].items()},
                      "metadata": case["metadata"]}
            path = fixtures/f"{index:03d}_{condition}.pt"
            torch.save(frozen, path)
            manifest.append({"path": str(path.relative_to(output)), "sha256": runtime.sha256(path), **case["metadata"]})
    assert len(manifest) == registration["frames"]
    runtime.write_json(output/"fixture_manifest.json", {"frozen_before_inference": True, "cases": manifest})
    rows = []
    for count, fixture in enumerate(manifest, 1):
        budget()
        path = output/fixture["path"]
        assert runtime.sha256(path) == fixture["sha256"]
        frozen = torch.load(path, map_location="cpu", weights_only=True)
        row, arrays = evaluate_case(model, frozen, use_prior=True)
        row["input_fixture_sha256"] = fixture["sha256"]
        rows.append(row)
        if fixture["replicate"] == 0 and fixture["condition"] != "null_clean":
            source = next(item for item in shape_manifest["records"] if item["processed_sha256"] == fixture["source_shape_sha256"])
            with np.load(args.shapes/source["processed_file"], allow_pickle=False) as opened:
                arrays["experimental_shape_normalized"] = opened["shape"].copy()
                arrays["source_valid_fraction"] = opened["source_valid_fraction"].copy()
            if fixture["condition"] == "null_structured_background":
                with np.load(args.shapes/source["background_source"]["processed_file"], allow_pickle=False) as opened:
                    arrays["experimental_background_normalized"] = opened["shape"].copy()
            dest = representatives/f"{fixture['partition']}_{fixture['condition']}.npz"
            np.savez_compressed(dest, **arrays)
            with np.load(dest, allow_pickle=False) as reopened:
                assert all(np.array_equal(reopened[key], value) for key, value in arrays.items())
        if count % 24 == 0 or count == len(manifest):
            runtime.write_json(output/"progress.json", {"utc": runtime.now(), "evaluated_frames": count,
                               "registered_frames": len(manifest), "seconds": time.perf_counter()-start})
            print(json.dumps({"real_shape_frames": count, "total": len(manifest), "seconds": time.perf_counter()-start}), flush=True)
    assert state_digest(model) == before
    assert runtime.sha256(DEFAULT_CHECKPOINT) == CHECKPOINT_SHA256
    assert all(runtime.sha256(runtime.ROOT/key) == value for key, value in registration["source_sha256"].items())
    results = {"checkpoint_sha256": CHECKPOINT_SHA256, "registration": registration, "results": report(rows), "rows": rows,
               "notes": ["CH and OH source images supply morphology only; they are not relabelled as measured AlO or its concentration.",
                         "Temperature RMSE uses the assumed synthetic continuum, not a measured thermal reference.",
                         "No weight updates, checkpoint selection or physics/CNN contribution comparison.",
                         "Frames within each source sequence are correlated; source groups are not independent temperature validations."]}
    runtime.write_json(output/"results.json", results)
    try:
        import resource
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except ImportError:
        rss_mib = None
    receipt = {"success": True, "utc": runtime.now(), "frames": len(manifest), "inference_calls": len(manifest),
               "measured_shape_frames": len(shape_manifest["records"]), "optimizer_updates": 0,
               "weights_and_sources_unchanged": True, "state_tensor_sha256": before, "checkpoint_sha256": CHECKPOINT_SHA256,
               "device": args.device, "torch_version": torch.__version__, "seconds": time.perf_counter()-start,
               "peak_GPU_allocated_MiB": torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0.,
               "peak_host_RSS_MiB": rss_mib, "frozen_before_inference": True, "representatives_reopened_exactly": True,
               "formal_scientific_acceptance": "not_accepted", "real_temperature_AlO_accuracy_validated": False}
    budget()
    if rss_mib is not None and rss_mib > registration["max_host_RSS_MiB"]:
        raise RuntimeError("Host memory budget exceeded")
    runtime.write_json(output/"evaluation_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
