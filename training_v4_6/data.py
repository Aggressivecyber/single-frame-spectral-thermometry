"""Frozen TRAIN-only signed coverage repair; formal generation is explicitly gated.

Importing this module and --check-registry use the standard library only. They do
not create directories, import torch/runtime, inspect arrays, or grant execution.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import copy
import json
import os
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/training_v4_6/run_001"
DATA = ROOT / "outputs/training_v4_6/run_001"
AUDIT = ROOT / "reports/training_v4_4/run_001/data_audit"
PLAN = AUDIT / "frozen_data_repair_plan.json"
REGISTRY = AUDIT / "repair_parent_registry.json"
OLD_MANIFEST = ROOT / "reports/training_v4_1/run_001/data_manifest.json"
V4_MANIFEST = ROOT / "reports/training_v4/run_001/data_manifest.json"
V45_STATUS = ROOT / "reports/training_v4_5/run_001/balanced_fixed_continuum/status.json"
V45_CANDIDATES = ROOT / "reports/training_v4_5/run_001/candidate_selection.json"
EXECUTION_PROTOCOL = OUT / "execution_protocol.json"
SAMPLING = OUT / "sampling_sequence.json"
SELECTION = ROOT / "reports/training_v4/run_001/backbone_selection.json"
CALIBRATION = ROOT / "reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json"
CONDITIONS = ("negative_weak", "negative_normal", "positive_weak", "positive_normal")
PEAKS = {"negative_weak": -45., "negative_normal": -180., "positive_weak": 45., "positive_normal": 180.}
TEMPERATURE_TYPES = ("front", "multipeak", "ring", "stripe", "flat", "smooth_random")
TEMPLATE_TYPES = ("blob", "multipeak", "ring", "stripe", "filament", "smooth_random")
EXPECTED_COUNTS = {"new_train_rows": 432, "merged_rows": 3784, "merged_train_rows": 3008,
                   "old_validation_rows": 776, "new_train_mothers": 24}
REQUIRED_FROZEN_INPUTS = (
    "reports/training_v4/run_001/data_manifest.json",
    "reports/training_v4_1/run_001/data_manifest.json",
    "reports/training_v4_4/run_001/data_audit/frozen_data_repair_plan.json",
    "reports/training_v4_4/run_001/data_audit/repair_parent_registry.json",
    "training_v4/data.py", "training_v4_1/data.py", "physics.py", "infer.py", "network.py",
    "checkpoints/selected_checkpoint.pt", "reports/training_v4/run_001/backbone_selection.json",
    "reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json",
    "training_v4/scalar_engine.py", "training_v4/cache.py", "repair_v3/solver.py", "training_v4/model.py",
    "training_v4_6/data.py", "training_v4_6/cache.py", "training_v4_6/train.py",
    "reports/training_v4_5/run_001/balanced_fixed_continuum/status.json",
    "reports/training_v4_5/run_001/candidate_selection.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rel(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def rooted(relative, root=ROOT):
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "Only repository-relative paths are allowed")
    path = (Path(root) / value).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Path escaped the repository")
    return path


def now():
    return datetime.now(timezone.utc).isoformat()


def load_bundle():
    """Read and validate descriptors only, including sealed DEV metadata only."""
    require(ROOT.drive.upper() in ("E:", "D:"), "This project requires an E/D checkout")
    old, old_v4, plan, registry = map(read, (OLD_MANIFEST, V4_MANIFEST, PLAN, REGISTRY))
    require(plan["status"] == "FROZEN_DATA_DESIGN_ONLY_NOT_LAUNCHED", "Unexpected data design status")
    require(file_hash(OLD_MANIFEST) == plan["anchor_manifest_sha256"], "Old manifest drift")
    require(file_hash(REGISTRY) == plan["registry_sha256"], "Frozen registry drift")
    require(file_hash(V4_MANIFEST) == old["v4_manifest_sha256"], "v4 manifest drift")
    require(file_hash(ROOT / "training_v4/data.py") == old_v4["generator_source_sha256"], "v4 generator drift")
    require(file_hash(ROOT / "training_v4_1/data.py") == old["generator_sha256"], "v4.1 generator drift")
    require(len(old["rows"]) == 3352, "The inherited manifest must have 3352 rows")
    require(Counter(r["split"] for r in old["rows"]) == {"train": 2576, "validation": 776}, "Inherited split counts drift")
    require(len({r["case_id"] for r in old["rows"]}) == 3352, "Inherited case IDs are not unique")
    parents = [m for m in registry["new_parents"] if m["split"] == "train"]
    locked = [m for m in registry["new_parents"] if m["split"] == "validation"]
    extra = registry["existing_train_mother_additional_noise"]
    old_signed = [m for m in old["signed_mothers"] if m["split"] == "train"]
    require(len(parents) == 24 and len(locked) == 36 and len(extra) == 48 and len(old_signed) == 12, "Repair registry counts drift")
    all_pairs = {(t, s) for t in TEMPERATURE_TYPES for s in TEMPLATE_TYPES}
    old_pairs = {(m["temperature_geometry"], m["template_geometry"]) for m in old_signed}
    new_pairs = {(m["temperature_geometry"], m["template_geometry"]) for m in parents}
    require(len(old_pairs) == 12 and len(new_pairs) == 24 and not old_pairs & new_pairs
            and old_pairs | new_pairs == all_pairs, "TRAIN does not complete the frozen 6 by 6 grid")
    require({(m["temperature_geometry"], m["template_geometry"]) for m in locked} == all_pairs,
            "Locked DEV descriptor grid drift; no DEV arrays are opened")
    require(Counter((r["mother_id"], r["condition"]) for r in extra) ==
            Counter((m["mother_id"], c) for m in old_signed for c in CONDITIONS), "Second-noise entries drift")
    require(all(r["split"] == "train" and r["noise_index"] == 1 for r in extra), "Extra noise must stay in TRAIN")
    old_ids = {m["mother_id"] for m in old_v4["mothers"] + old["signed_mothers"]}
    planned_ids = [m["mother_id"] for m in registry["new_parents"]]
    require(len(set(planned_ids)) == 60 and not old_ids & set(planned_ids), "Parent ID collision")
    seed_keys = ("seed", "temperature_seed", "amplitude_seed", "template_seed", "background_seed")
    known = old_v4["mothers"] + old["signed_mothers"] + old_v4["final_mothers_registered_but_not_generated"]
    old_seeds = {m[k] for m in known for k in seed_keys}
    old_seeds |= {r["noise_seed"] for r in old_v4["rows"] if r.get("noise_seed") is not None}
    old_seeds |= {m["seed"] + 10000000 + (71 if c.startswith("positive") else 0) + (17 if c.endswith("normal") else 0)
                  for m in old["signed_mothers"] for c in CONDITIONS}
    old_seeds |= {m["raw_noise_seed_base"] + 17 * ci + ni
                  for m in old_v4["final_mothers_registered_but_not_generated"] for ci in range(4) for ni in (0, 1)}
    planned_seeds = [m[k] for m in registry["new_parents"] for k in seed_keys]
    planned_seeds += [s for m in registry["new_parents"] for values in m["noise_seeds"].values() for s in values]
    planned_seeds += [r["noise_seed"] for r in extra]
    require(len(planned_seeds) == 828 and len(set(planned_seeds)) == 828 and not set(planned_seeds) & old_seeds,
            "Frozen seed namespace collision")
    return {"old": old, "old_v4": old_v4, "plan": plan, "registry": registry,
            "train_parents": parents, "additional_noise": extra, "old_signed_train": old_signed}


def new_case_specs(bundle=None):
    """Return exactly 432 planned TRAIN row descriptors, without array hashes.

    Naming and ordering are frozen for external sampler registration. Old rows
    are never rewritten to add seed fields; only these new rows carry new seeds.
    """
    bundle = load_bundle() if bundle is None else bundle
    rows = []
    for mother in bundle["train_parents"]:
        mid = mother["mother_id"]
        for condition in CONDITIONS:
            for kind, noise_index in (("continuous", None), ("quantization", None), ("noise", 0), ("noise", 1)):
                suffix = kind if noise_index is None else f"noise{noise_index}"
                case = mid + "_" + condition + "_" + suffix
                rows.append({"case_id": case, "mother_id": mid, "split": "train", "condition": condition,
                    "group": "negative" if condition.startswith("negative") else condition.split("_")[1],
                    "geometry": mother["temperature_geometry"], "template_geometry": mother["template_geometry"],
                    "input_kind": kind, "noise_index": noise_index,
                    "noise_seed": None if noise_index is None else mother["noise_seeds"][condition][noise_index],
                    "signed_aux": True, "origin": "v4_6_new_signed_train_mother",
                    "raw_path": rel(DATA / "raw" / mid / (condition + "_" + suffix + ".npz")),
                    "label_path": rel(DATA / "labels" / mid / (condition + ".npz")),
                    "existing_label_reused": False})
    old_noisy = {(r["mother_id"], r["condition"]): r for r in bundle["old"]["rows"]
                 if r["split"] == "train" and r["signed_aux"] and r["input_kind"] == "noise"}
    require(len(old_noisy) == 48, "Old signed TRAIN noise references must be unambiguous")
    for extra in bundle["additional_noise"]:
        old = old_noisy[(extra["mother_id"], extra["condition"])]
        mid, condition = extra["mother_id"], extra["condition"]
        rows.append({"case_id": mid + "_" + condition + "_noise1", "mother_id": mid, "split": "train",
            "condition": condition, "group": old["group"], "geometry": old["geometry"],
            "template_geometry": old["template_geometry"], "input_kind": "noise", "noise_index": 1,
            "noise_seed": extra["noise_seed"], "signed_aux": True, "origin": "v4_6_old_train_second_noise",
            "raw_path": rel(DATA / "raw" / mid / (condition + "_noise1.npz")),
            "label_path": old["label_path"], "label_sha256": old["label_sha256"],
            "existing_label_reused": True, "source_existing_case_id": old["case_id"]})
    require(len(rows) == 432 and len({r["case_id"] for r in rows}) == 432, "New case counts/case IDs drift")
    require(not {r["case_id"] for r in rows} & {r["case_id"] for r in bundle["old"]["rows"]}, "Old/new case ID collision")
    require(Counter(r["input_kind"] for r in rows) == {"noise": 240, "continuous": 96, "quantization": 96}, "Input-type counts drift")
    return rows


def validate_protocol_fields(protocol, action):
    require(protocol.get("schema") == "training_v4_6_execution_v1", "An explicit v4.6 execution protocol is required")
    require(protocol.get("frozen") is True and protocol.get("authorized_by") == "root", "Execution protocol is not frozen by root")
    require(protocol.get("split") == "train" and action in protocol.get("allowed_actions", []), "Action/split is not authorized")
    require(protocol.get("counts") == EXPECTED_COUNTS, "Execution counts differ from the frozen TRAIN design")
    require(protocol.get("locked_development_generated") is False and protocol.get("final_test_used") is False,
            "Locked DEV and final test must remain closed")
    require(protocol.get("cpu_threads") == 2 and protocol.get("cache_workers") == 1,
            "Use one GPU worker with two CPU threads")
    require(0 < protocol.get("maximum_generation_seconds", 0) <= 600, "Generation budget must be at most 600 seconds")
    require(0 < protocol.get("maximum_cache_seconds", 0) <= 3600, "Cache budget must be at most 3600 seconds")


def verify_execution(action, protocol_sha256):
    """No project imports, directory creation or array access before this gate."""
    require(isinstance(protocol_sha256, str) and len(protocol_sha256) == 64, "Supply the frozen execution-protocol file SHA256")
    require(EXECUTION_PROTOCOL.is_file(), "Root must register execution_protocol.json first")
    require(file_hash(EXECUTION_PROTOCOL) == protocol_sha256, "Execution protocol byte hash mismatch")
    protocol = read(EXECUTION_PROTOCOL)
    validate_protocol_fields(protocol, action)
    required = protocol.get("frozen_inputs_sha256", {})
    require(all(name in required for name in REQUIRED_FROZEN_INPUTS), "Required frozen source/input is missing")
    for name, expected in required.items():
        path = rooted(name)
        require(path.suffix.lower() != ".npz", "Protocol gates must not open data arrays")
        require(file_hash(path) == expected, "Frozen input missing or changed: " + name)
    require(read(V45_STATUS).get("stage") == "COMPLETE", "v4.5 must be COMPLETE first")
    candidates = read(V45_CANDIDATES)
    require(candidates.get("candidate_count") == 0 and candidates.get("selected") is None,
            "Data repair is allowed only after v4.5 has no candidate")
    sampling_path = rooted(protocol["sampling_sequence_path"])
    require(sampling_path == SAMPLING, "Use the registered v4.6 sampling_sequence.json")
    require(file_hash(sampling_path) == protocol["sampling_sequence_sha256"], "Frozen sampling file changed")
    bundle = load_bundle()
    specs = new_case_specs(bundle)
    by_id = {r["case_id"]: r for r in bundle["old"]["rows"] + specs}
    draws = read(sampling_path)["draws"]
    require(len(draws) == 500 and all(len(batch) == 4 for batch in draws), "Only frozen 500 by 4 draws are permitted")
    require(all(case in by_id and by_id[case]["split"] == "train" for batch in draws for case in batch),
            "Sampling contains a non-TRAIN or unknown case")
    return protocol, bundle, {"protocol_sha256": protocol_sha256,
        "sampling_sequence_sha256": file_hash(sampling_path), "verified_frozen_input_paths": list(required)}


def check_registry():
    bundle = load_bundle()
    specs = new_case_specs(bundle)
    return {"status": "METADATA_ONLY_CHECK_PASSED", "new_case_specs": len(specs),
        "new_mothers": len(bundle["train_parents"]), "new_noise": sum(r["input_kind"] == "noise" for r in specs),
        "merged_train_rows": 2576 + len(specs), "old_development_rows": 776, "merged_rows": 3352 + len(specs),
        "array_access": 0, "writes": 0, "gpu_calls": 0, "execution_authorized": False}


def write_json(path, value, *, replace=False):
    """Atomic write; immutable artifacts refuse an existing target or temp file."""
    path = Path(path)
    require(replace or not path.exists(), "Refusing to replace existing artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    require(replace or not path.exists(), "Artifact appeared during write: " + str(path))
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_arrays(path, arrays, *, allow_nonfinite=False):
    """Save new files exclusively, reopen them, and fingerprint actual arrays."""
    import numpy as np
    path = Path(path)
    require(not path.exists(), "Refusing to overwrite array file: " + str(path))
    values = {key: np.asarray(value) for key, value in arrays.items()}
    require(all(value.dtype.kind in "bifu" for value in values.values()), "Object/unsupported arrays are forbidden")
    require(allow_nonfinite or all(np.isfinite(value).all() for value in values.values()), "Nonfinite array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush(); os.fsync(handle.fileno())
    require(not path.exists(), "Array target appeared during write")
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as reopened:
        require(set(reopened.files) == set(values), "Array keys changed on reopen")
        for key, value in values.items():
            require(np.array_equal(reopened[key], value, equal_nan=allow_nonfinite), "Array changed on reopen: " + key)
    return {"file_sha256": file_hash(path), "array_sha256": {key: array_hash(value) for key, value in values.items()},
            "array_reopen_exact": True}


def read_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def deadline_check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Frozen wall-clock budget exceeded; retain all completed and failed artifacts")


def set_status(output, stage, **values):
    write_json(Path(output) / "status.json", {"stage": stage, "time": now(), **values}, replace=True)


def configure_runtime(data_dir):
    """Call only after formal authorization; all dependency caches stay on E/D."""
    import sys
    base = Path(data_dir) / "runtime"
    for name in ("tmp", "torch", "cache", "mpl", "cuda"):
        (base / name).mkdir(parents=True, exist_ok=True)
    os.environ["THERMOMETRY_RUNTIME"] = str(base)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(base / "tmp")
    for key, name in (("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "cache"),
                      ("MPLCONFIGDIR", "mpl"), ("CUDA_CACHE_PATH", "cuda")):
        os.environ[key] = str(base / name)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def _generate_train_rows(*, root, output, data_dir, bundle, physics, mother_function,
                         deadline=None, mode="FORMAL_CUDA", progress=True):
    """Shared generation core; small CPU mocks use a separate temporary root.

    This is deliberately not a CLI entry point. The public formal entry checks
    authorization and all frozen identities before constructing CUDA physics.
    """
    import numpy as np
    import torch
    p = physics
    root, output, data_dir = map(Path, (root, output, data_dir))
    require(root.drive.upper() in ("E:", "D:"), "All generation/test writes must stay on E/D")
    require(data_dir.resolve().is_relative_to(root.resolve()) and output.resolve().is_relative_to(root.resolve()),
            "Generation outputs must stay under the selected root")
    require(canonical_hash(p.cfg) == canonical_hash(bundle["old"]["physics_config"]), "Physics config changed")
    specs = new_case_specs(bundle)
    spec_by_id = {r["case_id"]: r for r in specs}
    new_rows, mother_records, legality, reused_labels = [], [], [], {}
    config_hash = canonical_hash(p.cfg)
    raw_hashes = {r["raw_hash"] for r in bundle["old"]["rows"]}
    raw_file_hashes = {r["raw_file_sha256"] for r in bundle["old"]["rows"]}
    label_file_hashes = {r["label_sha256"] for r in bundle["old"]["rows"]}
    new_label_content_hashes = set()
    actual_field_hashes = {}
    old_fields = bundle["old_v4"]["mothers"] + bundle["old"]["signed_mothers"]
    field_keys = ("temperature_field_sha256", "amplitude_field_sha256", "template_shape_sha256")
    field_hashes = {k: {m[k] for m in old_fields} for k in field_keys}
    current = {"phase": "starting", "descriptor": None, "arrays": {}}

    def capture(**values):
        current["arrays"] = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                             for key, value in values.items()}

    def finite(*values):
        require(all(bool(torch.isfinite(value).all()) for value in values), "Nonfinite generated field")

    def noise_array(expected, seed):
        generator = torch.Generator(device=expected.device).manual_seed(int(seed))
        noisy = torch.poisson(expected, generator=generator) + torch.randn(
            expected.shape, generator=generator, device=expected.device, dtype=expected.dtype) * p.cfg["read_noise_e"]
        raw_unclamped = (noisy / p.adc_step + p.cfg["black_level_DN"]).round()
        raw = raw_unclamped.clamp(0, 4095)
        capture(expected_e=expected, noisy_e=noisy, raw_dn_unclamped=raw_unclamped, raw_dn=raw)
        finite(noisy, raw)
        require(not bool((raw >= 4095).any()), "Saturated generated noise frame; no replacement seed is allowed")
        return raw.to(torch.uint16).cpu().numpy(), {"clipped_below_zero_pixels": int((raw_unclamped < 0).sum()),
                                                  "saturated_pixels": int((raw >= 4095).sum())}

    def store_raw(spec, raw, label_info, provenance, diagnostic=None):
        deadline_check(deadline)
        require(spec["split"] == "train", "Non-TRAIN array generation is forbidden")
        digest = array_hash(raw)
        require(digest not in raw_hashes, "Duplicate raw array hash; retain and diagnose, never redraw")
        path = rooted(spec["raw_path"], root)
        require(path.is_relative_to(data_dir.resolve()), "New raw path escaped the v4.6 output directory")
        info = save_arrays(path, {"raw_dn": raw})
        require(info["file_sha256"] not in raw_file_hashes, "Duplicate raw file hash; retain and diagnose")
        raw_file_hashes.add(info["file_sha256"])
        row = {**copy.deepcopy(spec), "raw_hash": digest, "raw_file_sha256": info["file_sha256"],
            "label_sha256": label_info["file_sha256"], "label_array_sha256": label_info["array_sha256"],
            "physics_config_sha256": config_hash, "calibration_sha256": config_hash,
            "shape": list(p.cfg["shape"]), "field_provenance": provenance,
            "array_reopen_exact": info["array_reopen_exact"], "sensor_diagnostic": diagnostic or {}}
        raw_hashes.add(digest); new_rows.append(row)
        append_jsonl(output / "new_rows.partial.jsonl", row)

    try:
        with torch.no_grad():
            for mother in bundle["train_parents"]:
                deadline_check(deadline)
                current = {"phase": "mother_state", "descriptor": copy.deepcopy(mother), "arrays": {}}
                append_jsonl(output / "generation_journal.jsonl", {"event": "MOTHER_STARTED", "time": now(),
                    "mother_id": mother["mother_id"], "seed": mother["seed"], "split": "train"})
                z, beta, shape, original = mother_function(p, mother)
                capture(state=z, beta=beta, template_shape=shape)
                finite(z, beta, shape)
                z[:, 0] += 250 / p.Tscale
                amplitude_scale = 1.
                while bool(p(z, beta).max() >= 52000):
                    deadline_check(deadline)
                    z[:, 1] += np.log(.8); amplitude_scale *= .8
                capture(state=z, beta=beta, template_shape=shape)
                finite(z, beta, shape)
                current["phase"] = "signed_legality"
                scale, valid = 1., False
                for attempt in range(24):
                    deadline_check(deadline)
                    valid = True
                    for sign in (-1, 1):
                        zz = z.clone(); zz[:, 4] = shape * 180 * scale * sign / p.Sscale
                        c, s, b, _ = p.components(zz, beta)
                        bands = p.op.object_blur(c + s + b); expected = p.op.encode_images(bands)
                        capture(state=zz, beta=beta, template_shape=shape, band_images_e=bands, expected_e=expected)
                        finite(bands, expected)
                        valid &= bool((bands.min() >= 0) & (expected.min() >= 0) & (expected.max() < 58000))
                    if valid:
                        break
                    scale *= .8
                require(valid, "Physically illegal signed mother after the original 24 attempts; retain original seed")
                for key in field_keys:
                    require(original[key] not in field_hashes[key], "Duplicate mother field: " + key)
                    field_hashes[key].add(original[key])
                actual = {
                    "temperature_field_after_shift_sha256": array_hash((p.T0 + p.Tscale * z[0, 0]).cpu().numpy()),
                    "amplitude_field_after_scale_sha256": array_hash(z[0, 1].exp().cpu().numpy()),
                    "template_shape_sha256": array_hash(shape.cpu().numpy()),
                    "background_beta_sha256": array_hash(beta[0].cpu().numpy()),
                    "continuum_state_sha256": array_hash(z[:, :2].cpu().numpy())}
                # Compare each new parent once. Conditions from this same parent
                # deliberately share continuum, temperature, amplitude and beta.
                for key, digest in actual.items():
                    known_actual = actual_field_hashes.setdefault(key, set())
                    require(digest not in known_actual, "Duplicate actual field across new mothers: " + key)
                    known_actual.add(digest)
                record = {**copy.deepcopy(mother), **original, "temperature_hash_scope": "pre +250 K from original mother_state",
                    **actual, "temperature_shift_K": 250., "continuum_amplitude_scale": amplitude_scale,
                    "legal_amplitude_scale": scale, "legality_attempts": attempt + 1}
                mother_records.append(record)
                append_jsonl(output / "mother_journal.jsonl", record)
                for condition in CONDITIONS:
                    deadline_check(deadline)
                    current["phase"] = "label_forward"; current["condition"] = condition
                    zz = z.clone(); zz[:, 4] = shape * PEAKS[condition] * scale / p.Sscale
                    c, s, b, _ = p.components(zz, beta)
                    bands = p.op.object_blur(c + s + b); expected = p.op.encode_images(bands)[0]
                    temperature, domain = p.color_temperature(p.op.object_blur(c))
                    template = p.op.object_blur(zz[:, 4] * p.Sscale)[0]
                    capture(state=zz, beta=beta, template_shape=shape, band_images_e=bands, expected_e=expected,
                            temperature_color_K=temperature, template_e=template)
                    finite(bands, expected, temperature, template)
                    require(bool((bands.min() >= 0) & (expected.min() >= 0) & (expected.max() < 58000)),
                            "Condition violates frozen expected/band legality")
                    mid = mother["mother_id"]
                    label_path = data_dir / "labels" / mid / (condition + ".npz")
                    label_info = save_arrays(label_path, {"template_e": template.cpu().numpy(),
                        "temperature_color_K": temperature[0].cpu().numpy(), "band_images_e": bands[0].cpu().numpy(),
                        "expected_e": expected.cpu().numpy(), "evaluation_domain": domain[0].cpu().numpy()})
                    require(label_info["file_sha256"] not in label_file_hashes,
                            "Duplicate new label file against old TRAIN/DEV or another new condition")
                    label_file_hashes.add(label_info["file_sha256"])
                    content_hash = canonical_hash(label_info["array_sha256"])
                    require(content_hash not in new_label_content_hashes, "Identical complete label arrays across new conditions")
                    new_label_content_hashes.add(content_hash)
                    provenance = {**actual, "template_field_preblur_sha256": array_hash((zz[0, 4] * p.Sscale).cpu().numpy()),
                        "actual_preblur_peak_e": PEAKS[condition] * scale,
                        "continuum_amplitude_scale": amplitude_scale, "legal_amplitude_scale": scale,
                        "scope": "actual latent fields after signed transforms plus separately hashed postblur label arrays"}
                    legality.append({"mother_id": mid, "condition": condition,
                        "actual_preblur_peak_e": PEAKS[condition] * scale,
                        "minimum_expected_e": float(expected.min()), "maximum_expected_e": float(expected.max()),
                        "minimum_band_e": float(bands.min()), "legal_amplitude_scale": scale})
                    dn = expected / p.adc_step + p.cfg["black_level_DN"]
                    finite(dn)
                    for suffix in ("continuous", "quantization", "noise0", "noise1"):
                        spec = spec_by_id[mid + "_" + condition + "_" + suffix]
                        current["phase"] = "raw_forward"; current["case_id"] = spec["case_id"]
                        current["noise_seed"] = spec["noise_seed"]
                        if suffix == "continuous":
                            capture(expected_e=expected, raw_dn=dn)
                            raw, diagnostic = dn.cpu().numpy(), {}
                        elif suffix == "quantization":
                            quant = dn.round()
                            capture(expected_e=expected, raw_dn=quant)
                            require(bool(((quant >= 0) & (quant < 4095)).all()), "Quantized protection frame is outside frozen ADC bounds")
                            raw, diagnostic = quant.to(torch.uint16).cpu().numpy(), {"saturated_pixels": 0}
                        else:
                            raw, diagnostic = noise_array(expected, spec["noise_seed"])
                        store_raw(spec, raw, label_info, provenance, diagnostic)
                if progress:
                    set_status(output, "GENERATING", new_mothers=len(mother_records), new_rows=len(new_rows), mode=mode)
                append_jsonl(output / "generation_journal.jsonl", {"event": "MOTHER_COMPLETE", "time": now(),
                    "mother_id": mother["mother_id"], "new_rows_completed": len(new_rows)})
            # These 48 labels are existing TRAIN labels only; no DEV labels open.
            for spec in (r for r in specs if r["existing_label_reused"]):
                deadline_check(deadline)
                current = {"phase": "old_train_second_noise", "descriptor": copy.deepcopy(spec),
                           "case_id": spec["case_id"], "noise_seed": spec["noise_seed"], "arrays": {}}
                label_path = rooted(spec["label_path"], root)
                expected_label_path = root / "outputs/training_v4_1/run_001/labels" / spec["mother_id"] / (spec["condition"] + ".npz")
                require(label_path == expected_label_path.resolve() and spec["mother_id"].startswith("v4_1_signed_train_mother_"),
                        "Only registered old TRAIN signed labels may be opened")
                require(file_hash(label_path) == spec["label_sha256"], "Existing TRAIN label file hash changed")
                arrays = read_arrays(label_path)
                require(set(arrays) == {"template_e", "temperature_color_K", "band_images_e", "expected_e", "evaluation_domain"},
                        "Unexpected old label array fields")
                require(all(np.isfinite(value).all() for value in arrays.values()), "Nonfinite old TRAIN label")
                require(tuple(arrays["expected_e"].shape) == tuple(p.cfg["shape"]), "Old TRAIN expected array shape mismatch")
                label_info = {"file_sha256": spec["label_sha256"], "array_sha256": {k: array_hash(v) for k, v in arrays.items()}}
                reused_labels[spec["label_path"]] = spec["label_sha256"]
                expected = torch.as_tensor(arrays["expected_e"], device=p.M.device)
                raw, diagnostic = noise_array(expected, spec["noise_seed"])
                store_raw(spec, raw, label_info, {"scope": "existing postblur truth arrays, without reconstructing latent fields",
                    "source_existing_case_id": spec["source_existing_case_id"], "latent_fields_regenerated": False}, diagnostic)
            require(len(new_rows) == 432 and len(mother_records) == 24 and len(legality) == 96,
                    "Generated counts did not match the frozen design")
            require([r["case_id"] for r in new_rows] == [r["case_id"] for r in specs], "Generated row order differs from frozen specifications")
            require(len(reused_labels) == 48, "Second noise must reference exactly 48 existing TRAIN labels")
            for name, digest in reused_labels.items():
                require(file_hash(rooted(name, root)) == digest, "Existing label changed during generation")
            old = bundle["old"]
            merged_rows = copy.deepcopy(old["rows"]) + new_rows
            require(Counter(r["split"] for r in merged_rows) == {"train": 3008, "validation": 776}, "Merged split counts are incorrect")
            require(merged_rows[:3352] == old["rows"], "Inherited manifest rows were modified")
            require(len({r["mother_id"] for r in merged_rows if r["split"] == "train"}) == 292, "TRAIN parent count is incorrect")
            merged = {**copy.deepcopy(old), "rows": merged_rows,
                "signed_mothers": copy.deepcopy(old["signed_mothers"]) + mother_records,
                "inherited_counts": copy.deepcopy(old["counts"]), "counts": dict(EXPECTED_COUNTS),
                "schema": "training_v4_6_train_only_signed_coverage_repair",
                "generator_sha256": file_hash(Path(__file__)), "source_v4_1_manifest_sha256": file_hash(OLD_MANIFEST),
                "new_locked_development_mothers_generated": 0, "final_test_opened": False,
                "new_rows_manifest": rel(output / "new_rows.json", root), "mode": mode}
            write_json(output / "new_rows.json", {"rows": new_rows, "count": 432, "all_train": True,
                "new_mothers": mother_records, "old_labels_reused_read_only": reused_labels, "mode": mode})
            write_json(output / "signed_expectation_legality.json", legality)
            write_json(output / "data_manifest.json", merged)
            return {"new_rows": new_rows, "manifest": merged, "new_mothers": mother_records,
                    "existing_labels_reverified": len(reused_labels), "new_label_files": 96}
    except BaseException as exc:
        # Preserve the exact offending values and the originally registered seed.
        diagnostic_path = None
        try:
            if current.get("arrays"):
                diagnostic_path = data_dir / "failures/failed_values.npz"
                save_arrays(diagnostic_path, current["arrays"], allow_nonfinite=True)
        except BaseException as save_exc:
            current["failure_array_save_error"] = repr(save_exc)
        failure = {k: v for k, v in current.items() if k != "arrays"}
        failure.update(status="FAILED", time=now(), exception_type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(), completed_new_rows=len(new_rows), completed_new_mothers=len(mother_records),
            failed_values_path=None if diagnostic_path is None else rel(diagnostic_path, root),
            failed_seed_replaced=False, failed_files_deleted=False, mode=mode)
        write_json(output / "generation_failure.json", failure)
        set_status(output, "FAILED", new_rows=len(new_rows), mode=mode, message=str(exc))
        raise


def _formal_generate(protocol, bundle, gate):
    started = time.monotonic()
    require(not DATA.exists(), "The generation directory already exists; preserve it and register explicit recovery")
    require(not (OUT / "status.json").exists() and not (OUT / "generation_receipt.json").exists(),
            "Generation was already started; no automatic overwrite or budget extension")
    DATA.mkdir(parents=True, exist_ok=False)
    write_json(DATA / "generation_identity.json", {**gate, "started": now(), "counts": EXPECTED_COUNTS})
    set_status(OUT, "INITIALIZING", new_rows=0, limit=432)
    try:
        configure_runtime(DATA)
        import torch
        from infer import load_model
        from training_v4.data import mother_state
        torch.set_num_threads(protocol["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = load_model(device="cuda"); p = model.physics
        require(list(p.cfg["shape"]) == [256, 256], "Formal generation requires the frozen native 256 by 256 grid")
        before = {name: array_hash(value.detach().cpu().numpy()) for name, value in p.named_buffers()}
        result = _generate_train_rows(root=ROOT, output=OUT, data_dir=DATA, bundle=bundle, physics=p,
            mother_function=mother_state, deadline=started + protocol["maximum_generation_seconds"])
        require(before == {name: array_hash(value.detach().cpu().numpy()) for name, value in p.named_buffers()},
                "Physical buffers changed during generation")
        verify_execution("generate_train", gate["protocol_sha256"])
        receipt = {"stage": "COMPLETE", "time": now(), **gate, **EXPECTED_COUNTS,
            "generation_seconds": time.monotonic() - started,
            "new_rows_sha256": file_hash(OUT / "new_rows.json"),
            "merged_manifest_sha256": file_hash(OUT / "data_manifest.json"),
            "new_label_files": result["new_label_files"], "old_train_label_files_reverified": result["existing_labels_reverified"],
            "old_3352_rows_unchanged": True, "native_shape": [256, 256],
            "physics_config_sha256": canonical_hash(p.cfg), "physical_buffers_unchanged": True,
            "physics_buffers_sha256": canonical_hash(before), "noise_realizations_explicit": True,
            "coefficient_recalibration_performed": False, "locked_development_arrays_generated_or_opened": 0,
            "final_test_arrays_or_results_opened": 0, "cache_cases_materialized": 0, "optimizer_updates": 0,
            "torch_version": torch.__version__, "device": str(p.M.device), "all_new_npz_reopened_exactly": True}
        receipt["collision_checks"] = {
            "registered_parent_and_seed_namespaces_vs_old_train_dev_and_final_metadata": True,
            "new_pretransform_parent_field_hashes_vs_old_manifest_and_other_new_parents": True,
            "actual_posttransform_field_hashes_between_new_parents": True,
            "same_parent_continuum_reuse_across_conditions_allowed": True,
            "new_raw_array_and_file_hashes_vs_old_manifest_and_other_new_raws": True,
            "new_label_file_hashes_vs_old_train_dev_manifest_and_other_new_labels": True,
            "complete_new_label_array_hashes_between_new_conditions": True,
            "old_train_second_noise_intentionally_reuses_48_label_files": True,
            "old_dev_or_final_arrays_opened_for_collision_checks": False,
            "new_vs_old_posttransform_fields_and_label_array_hashes_directly_compared": False,
            "near_duplicate_or_statistical_independence_estimated": False}
        write_json(OUT / "generation_receipt.json", receipt)
        set_status(OUT, "COMPLETE", new_rows=432, merged_train_rows=3008, old_development_rows=776)
        return receipt
    except BaseException as exc:
        if not (OUT / "generation_failure.json").exists():
            write_json(OUT / "generation_failure.json", {"phase": "formal_initialization_or_final_verification",
                "time": now(), "exception_type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(), "failed_files_deleted": False, "replacement_seeds": 0})
            set_status(OUT, "FAILED", message=str(exc))
        raise


def generate(*, split="train", protocol_sha256):
    require(split == "train", "Only --split train is implemented")
    protocol, bundle, gate = verify_execution("generate_train", protocol_sha256)
    return _formal_generate(protocol, bundle, gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--check-registry", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.check_registry:
        require(not args.execute, "Metadata checking and execution are separate actions")
        result = check_registry()
    else:
        require(args.execute, "Use --check-registry, or explicit --execute with a frozen protocol hash")
        result = generate(split=args.split, protocol_sha256=args.protocol_sha256)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
