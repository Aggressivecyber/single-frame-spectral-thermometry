"""Bounded first-round orchestration and receipts; aggregation never repeats inference."""
import copy
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import runtime
from infer import load_model, DEFAULT_CHECKPOINT
from network import ParameterUnfoldedNet
from physics import SpectralPhysics
from scenes import make_scene
from scenario_suite import REPRESENTATIVES
from . import METRIC_SCHEMA
from .common import (canonical_hash, array_hash, model_fingerprints, MeasurementInput,
                     Journal, execute_record, outcome_counts, case_identity)
from .fixtures import geometry, make_frame, freeze_file
from .physics_v2 import adc_quantize
from .methods import METHODS, RECONSTRUCTING, Reconstructor
from .metrics import score_prediction, legacy_scores, calibrate_detector
from .protocol import PROTOCOL


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    paths = list(runtime.ROOT.glob("*.py")) + list((runtime.ROOT/"evaluation_v2").glob("*.py"))
    return {str(p.relative_to(runtime.ROOT)).replace("\\", "/"): runtime.sha256(p) for p in sorted(paths)}


def configure_device(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Registered local CUDA device unavailable")
        torch.cuda.set_per_process_memory_fraction(.4)


def check_tests(output):
    gate = output/"tests_gate.xml"
    if not gate.exists():
        raise RuntimeError("Run the production v2 metric/physics tests before freezing/evaluation")
    suites = ET.parse(gate).getroot().iter("testsuite")
    values = list(suites)
    if not values or any(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in values):
        raise RuntimeError("Metric/physics gate did not pass")
    return {"passed": True, "tests": sum(int(s.get("tests", 0)) for s in values),
            "xml_sha256": runtime.sha256(gate),
            "tested_source_sha256": {str(p.relative_to(runtime.ROOT)).replace("\\", "/"): runtime.sha256(p)
                 for p in [runtime.ROOT/"tests/test_metrics_v2.py", runtime.ROOT/"tests/test_physics_v2.py", runtime.ROOT/"tests/test_methods_v2.py",
                           runtime.ROOT/"evaluation_v2/metrics.py", runtime.ROOT/"evaluation_v2/physics_v2.py"]}}


def inventory(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    baseline = read(output/"baseline_tracked_files.json")
    if any(runtime.sha256(runtime.ROOT/k) != v for k, v in baseline["initial_tracked_sha256"].items()):
        raise RuntimeError("An existing tracked file changed; inspect before continuing")
    model = load_model(DEFAULT_CHECKPOINT, "cpu")
    fingerprint = model_fingerprints(model, DEFAULT_CHECKPOINT)
    inv = {"utc": runtime.now(), "original_HEAD": baseline["HEAD"],
           "initial_dirty_diff": baseline["dirty_diff"], "fingerprints": fingerprint,
           "python": sys.version, "python_executable": sys.executable,
           "torch": str(torch.__version__), "CUDA_build": torch.version.cuda,
           "CUDA_available": torch.cuda.is_available(), "platform": platform.platform(),
           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
           "GPU_total_memory_MiB": torch.cuda.get_device_properties(0).total_memory/2**20 if torch.cuda.is_available() else None,
           "sources": source_hashes(), "protocol_sha256": canonical_hash(PROTOCOL),
           "unprovided_optional_attachments": ["metric_logic_checks.py", "physics_sensitivity_checks.py"],
           "all_original_tracked_files_unchanged": True, "tests": check_tests(output)}
    review = {}
    for name in ("synthetic", "experimental_shape_test"):
        root = runtime.ROOT/"reports/expanded_evaluation_001"/name
        records = read(root/"results.json")["rows"]
        fixture_manifest = read(root/"fixture_manifest.json")["cases"]
        receipt = read(root/"evaluation_receipt.json")
        progress = read(root/"progress.json")
        count = read(root/"registration.json")["frames"]
        success = receipt.get("success") is True and len(records) == count and progress["evaluated_frames"] == count
        assert {r["input_fixture_sha256"] for r in records} == {r["sha256"] for r in fixture_manifest}
        review[name] = {"registered": count, "attempted": len(records), "completed": len(records) if success else 0,
                        "failed": 0 if success else None, "remaining": count-len(records),
                        "status": "NORMAL_COMPLETE" if success else "INCOMPLETE",
                        "published_result_sha256": runtime.sha256(root/"results.json"),
                        "receipt": receipt, "progress": progress, "fixture_files_in_public_repo": 0,
                        "frozen_archive": "Complete frames retained in original research archive; selected downloads hash-verified separately"}
    runtime.write_json(output/"baseline_inventory.json", inv)
    runtime.write_json(output/"legacy_result_review.json", review)
    runtime.write_json(output/"protocol.json", PROTOCOL)
    runtime.write_json(output/"tests_gate_receipt.json", inv["tests"])
    print(json.dumps({"phase": "inventory", "status": "COMPLETE", "legacy_frames": sum(v["completed"] for v in review.values()),
                      "fingerprints": {k: v for k, v in fingerprint.items() if k != "config"}, "device": inv["device"]}), flush=True)


def selected_legacy_cases():
    result = []
    for name in ("synthetic", "experimental_shape_test"):
        root = runtime.ROOT/"reports/expanded_evaluation_001"/name
        for f in read(root/"fixture_manifest.json")["cases"]:
            if name == "synthetic":
                chosen = (f["replicate"] == 0 and REPRESENTATIVES.get(f["family_index"]) == f["condition"])
                chosen |= f["family"] == "temperature_front" and f["condition"] == "null_clean"
                chosen |= f["family"] == "hot_continuum" and f["condition"] == "null_background" and f["replicate"] == 0
            else:
                chosen = f["replicate"] == 0 and f["condition"] != "null_clean"
                chosen |= f["partition"] == "flat_flame_OH" and f["condition"] == "signal_weak"
            if chosen:
                result.append((name, f))
    return result


def legacy_temperature_fields(model, metadata, truth):
    p = model.physics
    T = truth["temperature_color_K"].to(p.M.device)
    xx, yy = p.background_basis[1], p.background_basis[2]
    if metadata["family"] == "temperature_front":
        latent = 2950.35+300*torch.tanh(14*(xx-.12*torch.sin(3*yy)))
    else:
        wave = 120*torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy)
        latent = metadata["true_temperature_min_K"]-wave.min()+wave
    # Latent is supplemental reconstruction from the archived generator recipe,
    # not an originally saved thermal measurement. Keep the original color truth.
    amplitude = (1.15 if metadata["family"] == "bright_saturated" else .72)+.08*torch.sin(torch.pi*xx)
    cfg = copy.deepcopy(p.cfg)
    cfg["object_cutoff_cpp"] = metadata["acquisition_object_cutoff_cpp"]
    acquisition = SpectralPhysics(cfg, p.M.dtype).to(p.M.device)
    continuum, _ = acquisition.continuum(latent[None])
    continuum *= amplitude[None, None]
    ref, _ = p.color_temperature(acquisition.op.object_blur(continuum))
    return latent.cpu().numpy(), ref[0].cpu().numpy(), T[0].cpu().numpy()


def freeze_legacy(args, model):
    cases, checks = [], []
    shape_root = runtime.ROOT/"reports/expanded_evaluation_001/experimental_shapes"
    sources = read(shape_root/"manifest.json")["records"]
    for name, f in selected_legacy_cases():
        pt = args.data_root/"archive"/name/Path(f["path"]).name
        if not pt.exists():
            raise FileNotFoundError("Selected original frozen frame not found: "+str(pt))
        assert runtime.sha256(pt) == f["sha256"]
        old = torch.load(pt, map_location="cpu", weights_only=True)
        raw = old["raw_dn"][0, 0].numpy()
        target = old["truth"]["template_e"][0].numpy()
        latent, reference, observed = legacy_temperature_fields(model, old["metadata"], old["truth"])
        arrays = {"raw_dn": raw, "template_e": target, "temperature_latent_K": latent,
                  "temperature_reference_color_K": reference, "temperature_observed_color_K": observed,
                  "evaluation_domain": old["truth"]["valid"][0].numpy(),
                  "uniform_temperature_domain": np.zeros(raw.shape, bool), "saturated": raw >= 4095}
        metadata = {**old["metadata"], "parent_scene_id": f"legacy_{name}_{f['family']}_{f['replicate']}",
                    "original_fixture_sha256": f["sha256"], "origin": "original research archive, no noise resampling",
                    "supplemental_latent_reference": "reconstructed from archived generator recipe; original observed color truth retained"}
        rep = runtime.ROOT/"reports/expanded_evaluation_001"/name/"representatives"/f"{f['family']}_{f['condition']}.npz"
        is_rep = f["replicate"] == 0 and rep.exists()
        if is_rep:
            with np.load(rep, allow_pickle=False) as r:
                assert np.array_equal(raw, r["raw_DN"])
                assert np.array_equal(target, r["truth_template_e"])
                assert np.array_equal(observed, r["truth_color_temperature_K"])
                metadata["published_representative_sha256"] = runtime.sha256(rep)
        if name == "experimental_shape_test":
            src = next(s for s in sources if s["processed_sha256"] == f["source_shape_sha256"])
            source_path = shape_root/src["processed_file"]
            assert runtime.sha256(source_path) == src["processed_sha256"]
            with np.load(source_path, allow_pickle=False) as s:
                arrays["source_valid_fraction"] = s["source_valid_fraction"].copy()
                arrays["source_image_domain"] = arrays["source_valid_fraction"] > .5
                arrays["source_image_domain_any"] = arrays["source_valid_fraction"] > 0
                arrays["experimental_shape_normalized"] = s["shape"].copy()
        case_id = "legacy_"+name+"_"+Path(f["path"]).stem
        cases.append(freeze_file(args.data_root, case_id, arrays, metadata, "legacy"))
        checks.append({"case_id": case_id, "original_frozen_sha256": f["sha256"],
                       "raw_array_sha256": array_hash(raw), "published_NPZ_exact_match": bool(is_rep)})
    runtime.write_json(args.output/"legacy_frame_verification.json", {"cases": checks,
                "original_frozen_frames": len(checks), "published_representatives_exactly_matched": sum(c["published_NPZ_exact_match"] for c in checks)})
    return cases


def training_statistics(args, model):
    # Reconstruct the actual selected-candidate training stream, rather than fit
    # any temperature to the audit frames. Compare every original raw hash.
    archive = runtime.ROOT.parent/"parameter_unfold_v004/runs/server_finetune_001"
    expected = read(archive/"training_stream_hashes.json")["low_lr_control"]
    fields, replay, hashes = [], [], []
    for update in range(160):
        mismatch = (((update*37)%101)/100-.5)*.1 if update % 2 else 0.
        case = make_scene(model.physics, "train", 30000+update, null=update % 3 == 0,
                          band_background_error_fraction=mismatch)
        fields.append(case["truth"]["temperature_color_K"][0].cpu().numpy())
        import hashlib
        h = hashlib.sha256(case["raw_dn"].cpu().numpy().tobytes()).hexdigest()
        replay.append(h == expected[update])
        hashes.append(h)
    mean_field = np.stack(fields).mean(0, dtype=np.float64)
    path = args.data_root/"training_statistics.npz"
    np.savez_compressed(path, mean_field_K=mean_field)
    stats = {"constant_K": float(mean_field.mean()), "scene_ids": list(range(30000, 30160)),
             "split": "train", "selected_candidate": "low_lr_control", "used_test_truth": False,
             "training_raw_replay_hash_matches": sum(replay), "training_frames": 160,
             "temperature_truth_kind": "registered training-stream postblur nominal color; replayed on current CUDA device",
             "replay_raw_hashes": hashes, "expected_archive_sha256": runtime.sha256(archive/"training_stream_hashes.json"),
             "mean_field_file_sha256": runtime.sha256(path)}
    runtime.write_json(args.output/"training_statistics.json", stats)
    return stats


def freeze_diagnostics(args, model, legacy):
    p = model.physics
    cases = []
    original = next(f for f in legacy if f["metadata"]["family"] == "temperature_front" and
                    f["metadata"]["condition"] == "null_clean" and f["metadata"]["replicate"] == 0)
    with np.load(args.data_root/original["path"], allow_pickle=False) as v:
        arrays = {k: v[k].copy() for k in v.files}
    z = torch.zeros((1, 5, *p.cfg["shape"]), dtype=p.M.dtype, device=p.M.device)
    z[:, 0] = (torch.as_tensor(arrays["temperature_latent_K"], device=p.M.device)-p.T0)/p.Tscale
    z[:, 1] = (.72+.08*torch.sin(torch.pi*p.background_basis[1])).log()
    beta = torch.tensor([[.45, .06, -.04]], device=p.M.device, dtype=p.M.dtype)
    expected = p(z, beta)[0].detach().cpu().numpy()
    arrays["expected_e"] = expected
    base = {"family": "temperature_front", "condition": "null_clean", "parent_scene_id": "old_front_fixed_null",
            "origin_original_frozen_sha256": original["metadata"]["original_fixture_sha256"],
            "scope": "same old-front fields and S=0; continuous electrons are an oracle numerical diagnostic"}
    for kind in ("continuous_e", "quantization_only", "old_noisy_quantized"):
        this = {k: v.copy() for k, v in arrays.items()}
        if kind != "old_noisy_quantized":
            this["raw_dn"] = adc_quantize(torch.as_tensor(expected), p).numpy()
            this["saturated"] = this["raw_dn"] >= 4095
        meta = {**base, "input_kind": "oracle_continuous_e" if kind == "continuous_e" else "raw_DN", "variant": kind}
        cases.append(freeze_file(args.data_root, "front_"+kind, this, meta, "diagnostics"))
    # Paired relative-vs-absolute continuum mismatch: same amplitude/background,
    # independently change T and mismatch mechanism. None of these is training.
    template_fields = geometry(p, "nominal_blob", 900)
    for T in PROTOCOL["small_scans"]["hot_uniform_T_K"]:
        fields = copy.deepcopy(template_fields)
        fields["temperature"][:] = T
        fields["amplitude"][:] = .6
        fields["parent_scene_id"] = "hot_uniform_parent0"
        for mechanism in ("control", "relative_5pct", "absolute_100e"):
            a, m = make_frame(p, fields, "null_clean", family="hot_background_scan", noise_seed=73115001,
                              relative_mismatch=.05 if mechanism == "relative_5pct" else 0.,
                              absolute_mismatch_e=100. if mechanism == "absolute_100e" else 0.)
            m.update({"variant": mechanism, "uniform_temperature_K": T})
            cases.append(freeze_file(args.data_root, f"hot_{round(T)}_{mechanism}", a, m, "diagnostics"))
    # Generation-time normalization to postblur integral and L2 energy; never
    # rescale a prediction. Keep common T/a and acquisition across morphologies.
    fields = geometry(p, "nominal_blob", 901)
    fields["temperature"][:] = 3000.
    fields["amplitude"][:] = .6
    ref, _ = make_frame(p, fields, "signal_nominal", family="morphology_budget", noise_seed=73116001)
    budgets = {"fixed_peak": None, "fixed_integral": float(ref["template_e"].sum()),
               "fixed_L2_energy": float(np.linalg.norm(ref["template_e"]))}
    for family, morphology in (("nominal_blob", "blob"), ("multiple_peaks", "multiple"), ("thin_filament", "filament")):
        pattern_fields = geometry(p, family, 901)
        fields["pattern"] = pattern_fields["pattern"]
        fields["parent_scene_id"] = "morphology_budget_parent0"
        for control, budget in budgets.items():
            a, m = make_frame(p, fields, "signal_nominal", family="morphology_budget", noise_seed=73116001,
                              normalization=control, target_budget=budget)
            m.update({"morphology": morphology, "variant": control})
            cases.append(freeze_file(args.data_root, f"morph_{morphology}_{control}", a, m, "diagnostics"))
    return cases


def freeze_smoke(args):
    check_tests(args.output)
    configure_device(args.device)
    assert read(args.output/"protocol.json") == PROTOCOL
    manifest_path = args.output/"frozen_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Frozen manifest already exists; reuse it, never regenerate")
    args.data_root.mkdir(parents=True, exist_ok=True)
    model = load_model(DEFAULT_CHECKPOINT, args.device)
    before = model_fingerprints(model, DEFAULT_CHECKPOINT)
    cases = []
    for family_index, family in enumerate(PROTOCOL["families"]):
        for split, parents in (("smoke", [0, 1]), ("validation", [100, 101])):
            for parent in parents:
                fields = geometry(model.physics, family, parent)
                conditions = PROTOCOL["conditions"]
                for condition in conditions:
                    a, m = make_frame(model.physics, fields, condition, family=family,
                               noise_seed=PROTOCOL["noise_seed_base"]+family_index*1000+parent,
                               gain_error=.03 if family == "channel_gain_3pct" else 0.,
                               continuum_multiplier=[1.12, 1.10, 1.06, 1.03, 1., .97, 1.] if family == "non_gray_continuum" else None)
                    m["split"] = split
                    cases.append(freeze_file(args.data_root, f"{split}_{family}_{parent}_{condition}", a, m, split))
    legacy = freeze_legacy(args, model)
    cases += legacy
    cases += freeze_diagnostics(args, model, legacy)
    training_statistics(args, model)
    assert model_fingerprints(model, DEFAULT_CHECKPOINT) == before
    manifest = {"schema": "audit-v2-frozen-inputs", "utc": runtime.now(), "data_root": str(args.data_root),
                "frozen_before_inference": True, "protocol_sha256": canonical_hash(PROTOCOL),
                "sources": source_hashes(), "fingerprints": before,
                "cases": cases, "counts": {s: sum(c["suite"] == s for c in cases) for s in ("smoke", "validation", "legacy", "diagnostics")}}
    runtime.write_json(manifest_path, manifest)
    print(json.dumps({"phase": "freeze-smoke", "status": "COMPLETE", "counts": manifest["counts"]}), flush=True)


def load_fixture(data_root, fixture):
    path = data_root/fixture["path"]
    if runtime.sha256(path) != fixture["file_sha256"]:
        raise RuntimeError("Frozen container changed: "+fixture["case_id"])
    with np.load(path, allow_pickle=False) as npz:
        a = {k: npz[k].copy() for k in npz.files}
    if any(array_hash(a[k]) != v for k, v in fixture["array_sha256"].items()):
        raise RuntimeError("Frozen array hash mismatch")
    return a


def stats_for_inference(args):
    d = read(args.output/"training_statistics.json")
    path = args.data_root/"training_statistics.npz"
    if runtime.sha256(path) != d["mean_field_file_sha256"]:
        raise RuntimeError("Frozen B0 training-mean field changed")
    with np.load(path, allow_pickle=False) as f:
        d["mean_field_K"] = f["mean_field_K"].copy()
    return d


def register_methods(args):
    """Generation source and later corrected evaluation source stay separate."""
    path = args.output/"evaluation_registration_v002.json"
    if path.exists():
        raise FileExistsError("Evaluation registration is immutable")
    gate = check_tests(args.output)
    model = load_model(DEFAULT_CHECKPOINT, "cpu")
    manifest = read(args.output/"frozen_manifest.json")
    assert model_fingerprints(model, DEFAULT_CHECKPOINT) == manifest["fingerprints"]
    stats_for_inference(args)
    reg = {"utc": runtime.now(), "revision": "v2.2", "metric_schema": METRIC_SCHEMA,
           "frozen_manifest_sha256": runtime.sha256(args.output/"frozen_manifest.json"),
           "generation_sources": manifest["sources"], "evaluation_sources": source_hashes(),
           "model_fingerprints": manifest["fingerprints"], "tests": gate,
           "training_statistics_sha256": runtime.sha256(args.output/"training_statistics.json"),
           "training_mean_file_sha256": runtime.sha256(args.data_root/"training_statistics.npz"),
           "training_temperature_cross_device_check_sha256": runtime.sha256(args.output/"training_temperature_draw_cross_device_check.json"),
           "reason": "Repair resume alias: include case ID, actual input kind/hash and training-statistic fingerprints. Prior partial records retained; frozen arrays never regenerated"}
    runtime.write_json(path, reg)
    print(json.dumps({"phase": "register-methods", "status": "COMPLETE", "revision": reg["revision"], "tests": gate["tests"]}), flush=True)


def validate_methods(args, model, manifest, stats):
    path = args.output/"detector_calibration.json"
    if path.exists():
        existing = read(path)
        if existing["frozen_manifest_sha256"] != runtime.sha256(args.output/"frozen_manifest.json"):
            raise RuntimeError("Validation calibration is for different frozen inputs")
        return existing
    candidates = PROTOCOL["traditional_spatial_regularizer"]["candidate_strengths"]
    validation = [f for f in manifest["cases"] if f["suite"] == "validation"]
    val = [f for f in validation if f["metadata"]["condition"].startswith("null_")]
    results, detector_arrays = [], {}
    for strength in candidates:
        recon = Reconstructor(model, "B1_tikhonov", stats, regularization=strength)
        signal_errors, null_errors = [], []
        for f in validation:
            a = load_fixture(args.data_root, f)
            out = recon.reconstruct(MeasurementInput(a["raw_dn"]))
            pred = out["prediction"]["template_e"]
            sse = float(np.square(pred-a["template_e"], dtype=np.float64).sum())
            energy = float(np.square(a["template_e"], dtype=np.float64).sum())
            if energy:
                signal_errors.append(sse/energy)
            else:
                null_errors.append(sse/pred.size/60**2)
        results.append({"strength": strength, "validation_signal_mean_L2_squared": float(np.mean(signal_errors)),
                        "validation_null_mean_RMSE60_squared": float(np.mean(null_errors)),
                        "validation_loss": float(np.mean(signal_errors)+np.mean(null_errors))})
    strength = min(results, key=lambda r: (r["validation_loss"], r["strength"]))["strength"]
    j = Journal(args.output/"validation_per_case.jsonl")
    for name in METHODS:
        arrays = []
        recon = Reconstructor(model, name, stats, regularization=strength if name == "B1_tikhonov" else 0.)
        for f in val:
            a = load_fixture(args.data_root, f)
            out = recon.reconstruct(MeasurementInput(a["raw_dn"]))
            if out["status"] != "INFERRED":
                raise RuntimeError("Validation numerical failure: "+f["case_id"])
            arrays.append(out["prediction"]["template_e"])
            j.append({"case_id": f["case_id"], "parent_scene_id": f["metadata"]["parent_scene_id"], "method": name,
                      "status": "SUCCESS", "seconds": out["seconds"], "raw_array_sha256": f["raw_array_sha256"],
                      "template_RMSE_e": float(np.sqrt(np.mean(np.square(arrays[-1], dtype=np.float64))))})
        detector_arrays[name] = calibrate_detector(arrays)
    result = {"frozen_manifest_sha256": runtime.sha256(args.output/"frozen_manifest.json"),
              "validation_parent_ids": sorted({f["metadata"]["parent_scene_id"] for f in val}),
              "validation_null_frames": len(val), "detectors": detector_arrays,
              "regularizer_validation_results": results, "selected_regularization": strength,
              "hyperparameters_selected_on_validation_only": True, "test_truth_used": False}
    runtime.write_json(path, result)
    return result


def compare(args):
    configure_device(args.device)
    check_tests(args.output)
    manifest = read(args.output/"frozen_manifest.json")
    registration = read(args.output/"evaluation_registration_v002.json")
    compatibility_path = args.output/"orchestration_compatibility_v003.json"
    compatibility = read(compatibility_path) if compatibility_path.exists() else None
    expected_sources = compatibility["current_sources"] if compatibility else registration["evaluation_sources"]
    if compatibility and compatibility["original_registration_sha256"] != runtime.sha256(args.output/"evaluation_registration_v002.json"):
        raise RuntimeError("Orchestration compatibility is for a different numerical configuration")
    if expected_sources != source_hashes():
        raise RuntimeError("Evaluation sources changed after registration; never silently resume a different method")
    if registration["frozen_manifest_sha256"] != runtime.sha256(args.output/"frozen_manifest.json"):
        raise RuntimeError("Frozen data manifest changed")
    if registration["training_statistics_sha256"] != runtime.sha256(args.output/"training_statistics.json") or registration["training_mean_file_sha256"] != runtime.sha256(args.data_root/"training_statistics.npz"):
        raise RuntimeError("Frozen training-statistic configuration changed")
    model = load_model(DEFAULT_CHECKPOINT, args.device)
    before = model_fingerprints(model, DEFAULT_CHECKPOINT)
    if before != manifest["fingerprints"]:
        raise RuntimeError("Checkpoint/config/buffers changed")
    stats = stats_for_inference(args)
    calibration = validate_methods(args, model, manifest, stats)
    suites = ("legacy", "smoke", "diagnostics") if args.suite == "all" else (args.suite,)
    fixtures = [f for f in manifest["cases"] if f["suite"] in suites]
    tasks = [(f, name) for f in fixtures for name in METHODS]
    if "legacy" in suites:
        # These are independently supplied known-calibration diagnostics, separate
        # method identities and physics fingerprints, not autonomous calibration.
        gain = known_calibration_fixture(fixtures)
        tasks += [(gain, "B4_known_true_calibration"), (gain, "B1_known_700_905_ratio")]
    journal = Journal(args.output/"per_case.jsonl")
    config_hash = canonical_hash({"protocol": PROTOCOL, "evaluation_registration": registration,
                                  "calibration": calibration, "fingerprints": before})
    runtime.write_json(args.output/"active_configuration.json", {"config_sha256": config_hash,
                       "evaluation_registration_sha256": runtime.sha256(args.output/"evaluation_registration_v002.json"),
                       "metric_schema": METRIC_SCHEMA})
    started = time.perf_counter()
    registered_identities = []
    for fixture, name in tasks:
        input_kind = fixture["metadata"].get("input_kind", "raw_DN")
        input_hash = fixture["array_sha256"]["expected_e"] if input_kind == "oracle_continuous_e" else fixture["raw_array_sha256"]
        identity = case_identity(config_hash=config_hash, case_id=fixture["case_id"], fixture_hash=fixture["file_sha256"],
                                 method=name, input_kind=input_kind, input_hash=input_hash, run_id=PROTOCOL["run_id"])
        registered_identities.append(identity)
        if args.resume and journal.succeeded(identity):
            continue
        base = {"identity_sha256": identity, "config_sha256": config_hash, "run_id": PROTOCOL["run_id"],
                "metric_schema": METRIC_SCHEMA, "suite": fixture["suite"], "case_id": fixture["case_id"],
                "scene": fixture["metadata"]["family"], "parent_scene_id": fixture["metadata"]["parent_scene_id"],
                "condition": fixture["metadata"]["condition"], "method": name,
                "input_hash": input_hash, "fixture_sha256": fixture["file_sha256"], "input_kind": input_kind,
                "metadata": fixture["metadata"]}
        if time.perf_counter()-started > PROTOCOL["per_phase_wall_s"]:
            execute_record(journal, base, lambda: (_ for _ in ()).throw(TimeoutError("Phase wall budget")))
            break
        def work():
            a = load_fixture(args.data_root, fixture)
            if base["input_kind"] == "raw_DN" and np.any(a["raw_dn"] >= 4095):
                return {"status": "SATURATION_REJECTED", "metrics": None, "numerical_execution_passed": False,
                        "saturated_measurement_pixels": int(np.sum(a["raw_dn"] >= 4095)),
                        "legacy_scores": "Historical saturated output retained in original published representative as failure diagnostic"}
            this_model = model
            selected = name
            if name == "B4_known_true_calibration":
                cfg = copy.deepcopy(model.cfg)
                cfg["channel_gain"] = fixture["metadata"]["acquisition_channel_gain"]
                this_model = ParameterUnfoldedNet(cfg).to(args.device).eval()
                # Only the learned CNN is copied; known physical buffers regenerated.
                cnn = dict(model.named_parameters())
                with torch.no_grad():
                    for k, value in this_model.named_parameters():
                        value.copy_(cnn[k])
            recon = Reconstructor(this_model, selected, stats,
                         regularization=calibration["selected_regularization"] if name == "B1_tikhonov" else 0.,
                         known_ratio=.97/1.03 if name == "B1_known_700_905_ratio" else None)
            traced = (fixture["suite"] == "diagnostics" and fixture["case_id"].startswith("front_")) or (
                fixture["suite"] == "legacy" and (fixture["case_id"].endswith("10_00_null_clean") or
                fixture["metadata"].get("partition") == "flat_flame_OH" and fixture["metadata"].get("replicate") == 0 and fixture["metadata"]["condition"] == "signal_weak"))
            if base["input_kind"] == "oracle_continuous_e":
                base["input_hash"] = array_hash(a["expected_e"])
                out = recon.diagnose_continuous_electrons(a["expected_e"], save_trace=traced)
            else:
                out = recon.reconstruct(MeasurementInput(a["raw_dn"]), save_trace=traced)
            pred = out.pop("prediction")
            trace = out.pop("trace")
            detector_name = "B4_published" if name == "B4_known_true_calibration" else ("B1_demod_fit" if name == "B1_known_700_905_ratio" else name)
            scored = score_prediction(a, pred, calibration["detectors"][detector_name])
            numerical = {k: v for k, v in pred.items() if isinstance(v, np.ndarray)}
            attempt_number = 1+sum(r["identity_sha256"] == identity for r in journal.rows())
            prediction_path = args.data_root/"predictions"/config_hash[:16]/fixture["case_id"]/(name+f"_attempt{attempt_number:03d}.npz")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_path, **numerical)
            row = {**out, **scored, "legacy_scores": legacy_scores(a, pred),
                   "prediction_path": str(prediction_path.relative_to(args.data_root)).replace("\\", "/"),
                   "prediction_sha256": runtime.sha256(prediction_path),
                   "method_fingerprints": model_fingerprints(this_model, DEFAULT_CHECKPOINT),
                   "known_calibration_oracle": name in ("B4_known_true_calibration", "B1_known_700_905_ratio")}
            if trace:
                trace_arrays, trace_summary = {}, []
                for index, step in enumerate(trace):
                    step_pred = {k: step[k] for k in ("temperature_K", "template_e", "valid")}
                    step_metrics = score_prediction(a, step_pred, calibration["detectors"][detector_name])
                    trace_summary.append({"label": step["label"], "objective": step["standardized_objective"],
                                          "status": step_metrics["status"], "metrics": step_metrics["metrics"]})
                    for key, value in step_pred.items():
                        trace_arrays[f"{index:02d}_{key}"] = value
                trace_path = prediction_path.with_name(name+"_trace.npz")
                np.savez_compressed(trace_path, **trace_arrays)
                row["trace"] = trace_summary
                row["trace_path"] = str(trace_path.relative_to(args.data_root)).replace("\\", "/")
                row["trace_sha256"] = runtime.sha256(trace_path)
            return row
        row = execute_record(journal, base, work)
        print(json.dumps({"case": fixture["case_id"], "method": name, "status": row["status"],
                          "seconds": row.get("seconds")}), flush=True)
        if torch.cuda.is_available() and torch.cuda.max_memory_allocated()/2**20 > PROTOCOL["max_GPU_allocated_MiB"]:
            break
    unchanged = model_fingerprints(model, DEFAULT_CHECKPOINT) == before
    relevant = [r for r in journal.rows() if r["identity_sha256"] in registered_identities]
    # Count all planned tasks even if budget prevented attempts.
    counts = outcome_counts(relevant, len(tasks))
    receipt = {"utc": runtime.now(), "suite": args.suite, **counts, "seconds": time.perf_counter()-started,
               "optimizer_updates": 0, "fingerprints_unchanged": unchanged,
               "config_sha256": config_hash, "device": args.device, "GPU": torch.cuda.get_device_name() if args.device == "cuda" else None,
               "all_inputs_hash_verified": True, "formal_scientific_acceptance": "not_accepted"}
    runtime.write_json(args.output/("compare_"+args.suite+"_receipt.json"), receipt)
    assert unchanged
    print(json.dumps(receipt), flush=True)


def known_calibration_fixture(fixtures):
    """Stable across --suite legacy/all ordering; never substitute a smoke parent."""
    return next(f for f in fixtures if f["suite"] == "legacy" and
                f["metadata"]["family"] == "channel_gain_3pct" and
                f["metadata"]["condition"] == "null_clean" and f["metadata"]["replicate"] == 0)


def summarize(args):
    from .report import summarize_run
    summarize_run(args)
