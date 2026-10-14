"""Immutable Stage A references and separate measurement/diagnostic loaders.

No generator or inverse is imported here. A reference carries external audit
metadata; only ``load_raw``'s returned tensor crosses the deployment boundary.
Only the two already opened v5 families and the old deterministic16 are allowed.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from . import common

CONDITIONS = ("blank", "positive_weak", "negative_weak", "positive_normal", "negative_normal")
INPUT_KINDS = ("continuous", "quantization", "noise")
FAMILIES = {
    "development": ("outputs/research_v5/run_001/development", 36, 540,
                    "5a7df07eb00bdfc0de0a17315c59761cc81562edac02c28da291c6d3650d4514"),
    "oracle": ("outputs/research_v5/run_001/oracle_data", 12, 180,
               "8e31229a869a783384ae58b6af2f0d5569a0587bcd039a19ab29427a21f771b4"),
}
DEVELOPMENT_RECORDS = "reports/research_v5/run_001/development/records.jsonl"
DEVELOPMENT_RECORDS_SHA256 = "dd04da43edecce5320cd98e5359f6e034c64115c3f77c89248fabbbc19d3b46c"
REGRESSION_INDEX = "training_v4/development_inputs/index.json"
REGRESSION_INDEX_SHA256 = "7a3124efd80b3c283acb975f5859bbcfbc3c169d156d188dce9b78ed68647fbb"
V3_CACHE_ROOT = "outputs/training_v4/run_001"


class InputIntegrityError(ValueError):
    pass


def array_hash(value):
    """The unchanged v4/v5 dtype+shape+bytes hash convention."""
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _path_in(path, parent):
    p = Path(path)
    if not p.is_absolute():
        p = common.ROOT/p
    p, parent = p.resolve(), Path(parent).resolve()
    if p != parent and parent not in p.parents:
        raise InputIntegrityError("Read path escapes its approved family: " + str(p))
    return p


def _family_path(ref, kind):
    """Reject sealed/unknown cohorts before stat, hashing or NPZ opening."""
    cohort = ref.get("cohort")
    if cohort in FAMILIES and not ref.get("fixture_only", False):
        parent = common.ROOT/FAMILIES[cohort][0]
        directory = "raw" if kind == "raw" else "diagnostic_labels"
        return _path_in(ref[kind+"_reference"]["path"], parent/directory)
    if cohort == "regression16" and not ref.get("fixture_only", False):
        return _path_in(ref[kind+"_reference"]["path"], common.ROOT/"training_v4/development_inputs")
    if cohort == "implementation_fixture" and ref.get("fixture_only") is True:
        return _path_in(ref[kind+"_reference"]["path"], common.ROOT/"outputs/research_v6")
    raise InputIntegrityError("Sealed or unregistered cohort is not readable: " + str(cohort))


def _verify_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise InputIntegrityError("A registered SHA256 is required")
    if not path.is_file():
        raise InputIntegrityError("Immutable input/cache is missing: " + str(path))
    actual = common.sha(path)
    if actual != expected:
        raise InputIntegrityError("Immutable file SHA256 mismatch: " + str(path))
    return actual


def _raw_array(ref):
    import numpy as np
    path = _family_path(ref, "raw")
    _verify_file(path, ref["raw_reference"]["sha256"])
    with np.load(path, allow_pickle=False) as z:
        if "raw_dn" not in z.files:
            raise InputIntegrityError("Immutable input has no raw_dn")
        # The regression files contain diagnostics too; they are not materialized.
        raw = z["raw_dn"].copy()
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise InputIntegrityError("Raw must be a finite H,W array")
    if array_hash(raw) != ref["raw_reference"]["array_sha256"]:
        raise InputIntegrityError("Immutable raw dtype/shape/bytes hash mismatch")
    if "dtype" in ref["raw_reference"] and str(raw.dtype) != ref["raw_reference"]["dtype"]:
        raise InputIntegrityError("Immutable raw dtype differs from registration")
    if "shape" in ref["raw_reference"] and list(raw.shape) != ref["raw_reference"]["shape"]:
        raise InputIntegrityError("Immutable raw shape differs from registration")
    return raw


def load_raw(ref, physics):
    """Return only one B=1 raw-DN tensor after byte and array verification."""
    import torch
    raw = _raw_array(ref)
    if tuple(raw.shape) != tuple(physics.cfg["shape"]):
        raise InputIntegrityError("Raw shape differs from the frozen calibration")
    return torch.as_tensor(raw, dtype=physics.M.dtype, device=physics.M.device)[None]


def load_diagnostic_truth(ref, physics):
    """External oracle/scoring labels; never pass this mapping to inference."""
    import numpy as np
    import torch
    path = _family_path(ref, "label")
    _verify_file(path, ref["label_reference"]["sha256"])
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files if key != "raw_dn"}
    tensors = {}
    for key, value in values.items():
        if value.dtype.kind not in "buifc":
            raise InputIntegrityError("Diagnostic labels must be numeric arrays: " + key)
        dtype = torch.bool if value.dtype.kind == "b" else physics.M.dtype
        tensors[key] = torch.as_tensor(value, dtype=dtype, device=physics.M.device)[None]
    if "temperature_reference_color_K" not in tensors and "temperature_color_K_reference" in tensors:
        tensors["temperature_reference_color_K"] = tensors["temperature_color_K_reference"]
    return tensors


def _reference(cohort, row, manifest_receipt):
    family = common.ROOT/FAMILIES[cohort][0]
    if row.get("raw_available", row.get("raw_generation_status") == "GENERATED") is not True:
        raise InputIntegrityError("Registered source raw unavailable: " + row["case_id"])
    raw_path = _path_in(family/row["raw_path"], family/"raw")
    label_path = _path_in(family/row["label_path"], family/"diagnostic_labels")
    keys = ("case_id", "mother_id", "condition", "group", "coefficient_e", "input_kind",
            "geometry", "temperature_geometry", "template_geometry", "noise_index", "noise_seed")
    result = {k: copy.deepcopy(row.get(k)) for k in keys}
    result.update(cohort=cohort, fixture_only=False,
                  partition_role="development_selection" if cohort == "development" else "oracle_diagnostic",
                  raw_reference={"path": common.relative(raw_path), "sha256": row["raw_file_sha256"],
                                 "array_sha256": row["raw_hash"], "dtype": row["dtype"],
                                 "shape": row["shape"]},
                  label_reference={"path": common.relative(label_path), "sha256": row["label_sha256"]},
                  manifest_reference=copy.deepcopy(manifest_receipt),
                  C_v3_cache=None, C_v5_cache=None)
    return result


def inventory_stage_a():
    """Metadata-only inventory of the unchanged 540+180 raw observations.

    This reads permitted manifests, development records and legacy cache JSON
    metadata. It never opens any raw, label or cache NPZ and runs no inverse.
    """
    rows, manifests = [], {}
    physics_sha = common.canonical_hash(common.physics_config())
    for cohort, (folder, mothers, count, frozen_hash) in FAMILIES.items():
        path = common.ROOT/folder/"data_manifest.json"
        _verify_file(path, frozen_hash)
        manifest = common.read_json(path)
        if manifest.get("fixture_only") is not False or manifest["physics_config_sha256"] != physics_sha:
            raise InputIntegrityError("Unexpected formal manifest/calibration")
        receipt = {"path": common.relative(path), "sha256": frozen_hash,
                   "plan_sha256": manifest["plan_sha256"], "rows": count, "mothers": mothers}
        manifests[cohort] = receipt
        family_rows = [_reference(cohort, row, receipt) for row in manifest["rows"]]
        pairs = Counter((r["condition"], r["input_kind"]) for r in family_rows)
        if len(family_rows) != count or len({r["mother_id"] for r in family_rows}) != mothers:
            raise InputIntegrityError("Frozen family count changed")
        if pairs != Counter({(c, k): mothers for c in CONDITIONS for k in INPUT_KINDS}):
            raise InputIntegrityError("Frozen signed/input-form coverage changed")
        by_mother = {}
        for r in family_rows:
            by_mother.setdefault(r["mother_id"], set()).add((r["condition"], r["input_kind"]))
        if any(len(v) != 15 for v in by_mother.values()):
            raise InputIntegrityError("Each immutable mother must have exactly15 variants")
        rows.extend(family_rows)
    by_case = {r["case_id"]: r for r in rows}
    if len(by_case) != 720:
        raise InputIntegrityError("Case identifiers must be unique across Stage A")
    records_path = common.ROOT/DEVELOPMENT_RECORDS
    _verify_file(records_path, DEVELOPMENT_RECORDS_SHA256)
    for record in common.jsonl(records_path):
        if record.get("method") != "v5_physics" or record.get("alpha_g") != 1.0:
            continue
        ref = by_case[record["case_id"]]
        cache = record.get("prediction_cache")
        if cache is None:
            # Absence is explicit in the inventory; no fabricated continuum.
            ref["C_v5_cache_failure"] = {"status": record.get("status"), "error": record.get("error")}
            continue
        if record["raw_sha256"] != ref["raw_reference"]["sha256"]:
            raise InputIntegrityError("Development cached source belongs to another raw")
        cache_path = _path_in(cache["path"], common.ROOT/"outputs/research_v5/run_001/development_grid_predictions")
        ref["C_v5_cache"] = {"path": common.relative(cache_path), "sha256": cache["sha256"],
            "raw_file_sha256": ref["raw_reference"]["sha256"],
            "raw_array_sha256": ref["raw_reference"]["array_sha256"],
            "raw_dtype": ref["raw_reference"]["dtype"], "raw_shape": ref["raw_reference"]["shape"],
            "source": "C_v5", "cache_kind": "frozen_v5_development_latent",
            "record_reference": {"path": DEVELOPMENT_RECORDS, "sha256": DEVELOPMENT_RECORDS_SHA256},
            "original_route_seconds": record.get("diagnostics", {}).get("seconds"),
            "original_source_seconds": None,
            "time_scope": "v5 stored route includes continuum+signed solve; continuum-only time unavailable"}
    by_raw = {}
    for ref in rows:
        by_raw.setdefault(ref["raw_reference"]["array_sha256"], []).append(ref)
    # Cache metadata may describe legacy partitions; no payload is opened until
    # the raw hash has matched one of the above approved references.
    cache_root = common.ROOT/V3_CACHE_ROOT
    metadata_count = 0
    for path in sorted((cache_root/"cache").rglob("*.json")):
        metadata_count += 1
        meta = common.read_json(path)
        matched = by_raw.get(meta.get("raw_hash"), [])
        if not matched:
            continue
        cache_path = _path_in(cache_root/meta["cache_path"], cache_root/"cache")
        for ref in matched:
            if ref["C_v3_cache"] is not None:
                raise InputIntegrityError("Ambiguous legacy raw-matched v3 cache")
            ref["C_v3_cache"] = {"path": common.relative(cache_path), "sha256": meta["cache_sha256"],
                "metadata_path": common.relative(path), "metadata_sha256": common.sha(path),
                "raw_file_sha256": ref["raw_reference"]["sha256"],
                "raw_array_sha256": ref["raw_reference"]["array_sha256"],
                "raw_dtype": ref["raw_reference"]["dtype"], "raw_shape": ref["raw_reference"]["shape"],
                "source": "C_v3", "cache_kind": "frozen_v4_state_beta", "legacy_metadata": meta,
                "original_source_seconds": meta.get("seconds"),
                "original_peak_GPU_allocated_MiB": meta.get("peak_GPU_allocated_MiB")}
    counts = {cohort: sum(r["cohort"] == cohort for r in rows) for cohort in FAMILIES}
    cached = {source: sum(r[source+"_cache"] is not None for r in rows) for source in ("C_v3", "C_v5")}
    return {"schema": "v6-immutable-stage-a-inputs-001", "rows": rows, "manifests": manifests,
            "counts": counts, "mothers": {c: FAMILIES[c][1] for c in FAMILIES},
            "cached_sources": cached, "new_inference_required": {k: 720-v for k, v in cached.items()},
            "legacy_cache_metadata_inspected": metadata_count,
            "raw_payloads_opened": False, "sealed_payloads_opened": False,
            "inputs_generated_or_reconstructed": False,
            "reference_sha256": common.canonical_hash(rows)}


def integrity_receipt(rows, payload_audit=False):
    """Validate exact references; optional raw shape/hash audit without inversion."""
    refs = list(rows)
    if len({r["case_id"] for r in refs}) != len(refs):
        raise InputIntegrityError("Duplicate registered input")
    checked_files, shapes = {}, Counter()
    for ref in refs:
        for kind in ("raw", "label"):
            path = _family_path(ref, kind)
            receipt = ref[kind+"_reference"]
            relative_path = common.relative(path)
            if relative_path not in checked_files:
                _verify_file(path, receipt["sha256"])
                checked_files[relative_path] = receipt["sha256"]
        if payload_audit:
            raw = _raw_array(ref)
            shapes[str(tuple(raw.shape))] += 1
            if ref["cohort"] in FAMILIES and tuple(raw.shape) != (256, 256):
                raise InputIntegrityError("Native formal geometry must remain256x256")
    return {"schema": "v6-input-integrity-001", "passed": True, "rows": len(refs),
            "mothers": len({r["mother_id"] for r in refs}),
            "cohort_counts": dict(Counter(r["cohort"] for r in refs)),
            "condition_input_counts": dict(Counter(r["cohort"]+":"+r["condition"]+":"+r["input_kind"] for r in refs)),
            "reference_sha256": common.canonical_hash(refs), "unique_files_hashed": len(checked_files),
            "file_hashes_sha256": common.canonical_hash(checked_files),
            "payload_audit": bool(payload_audit), "raw_shapes": dict(shapes),
            "sealed_payloads_opened": False, "scientific_inversions": 0}


def regression16_references():
    """Locate original deterministic inputs without reading/running their arrays."""
    index = common.ROOT/REGRESSION_INDEX
    _verify_file(index, REGRESSION_INDEX_SHA256)
    rows = common.read_json(index)
    if len(rows) != 16:
        raise InputIntegrityError("Original deterministic regression count changed")
    refs = []
    for row in rows:
        path = common.ROOT/"training_v4/development_inputs"/(row["case_id"]+".npz")
        meta = row.get("metadata", {})
        refs.append({"case_id": row["case_id"], "mother_id": meta.get("parent_scene_id", row["case_id"]),
            "cohort": "regression16", "fixture_only": False, "condition": meta.get("condition"),
            "input_kind": meta.get("input_kind"), "original_path": row["path"],
            "raw_reference": {"path": common.relative(path), "sha256": row["file_sha256"],
                              "array_sha256": row["raw_array_sha256"]},
            "label_reference": {"path": common.relative(path), "sha256": row["file_sha256"]},
            "index_reference": {"path": REGRESSION_INDEX, "sha256": REGRESSION_INDEX_SHA256}})
    return refs

