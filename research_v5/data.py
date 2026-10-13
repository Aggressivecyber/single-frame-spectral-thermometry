"""Registered, matched signed diagnostics from already-opened v4 mothers.

Planning is stdlib-only and has no generation side effects.  Generation replays
the original mother function, changes only the active template coefficient, and
keeps truth in separate ``diagnostic_labels`` files.  It never repairs a negative
Poisson expectation or rescales a mother to make a signed experiment legal.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


TEMPERATURE_GEOMETRIES = (
    "front", "multipeak", "ring", "stripe", "flat", "smooth_random"
)
TEMPLATE_GEOMETRIES = (
    "blob", "multipeak", "ring", "stripe", "filament", "smooth_random"
)
COEFFICIENTS_E = (0.0, 45.0, -45.0, 180.0, -180.0)
CONDITIONS = ("blank", "positive_weak", "negative_weak", "positive_normal", "negative_normal")
INPUT_KINDS = ("continuous", "quantization", "noise")
SCHEMA = "v5-stage1-matched-signed-diagnostic-v1"


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(old_manifest: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], str | None]:
    if isinstance(old_manifest, Mapping):
        return copy.deepcopy(dict(old_manifest)), None
    path = Path(old_manifest)
    return json.loads(path.read_text(encoding="utf-8")), _file_hash(path)


def _noise_seed(mother_id: str, condition: str) -> int:
    """A preregistered, distinct stream per *different* Poisson mean."""
    key = f"v5-stage1-noise-v1/{mother_id}/{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)


def build_diagnostic_plan(old_manifest: Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    """Select twelve opened validation mothers and register all 180 inputs.

    The metadata selection reproduces ``training_v4_1.data.select_mothers`` for
    ``split='validation', n=12`` without importing any legacy generator.  Each
    original temperature geometry has two mothers; template geometries are
    balanced using the old local/global/index tie break.  Original geometry
    names are preserved, including ``stripe`` (a two-stripe old field).
    """
    old, source_file_hash = _read_manifest(old_manifest)
    mothers = old.get("mothers")
    if not isinstance(mothers, list):
        raise ValueError("The opened v4 manifest must contain its mother descriptors")
    ids = [m.get("mother_id") for m in mothers]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Mother identities must be unique strings")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", value) is None for value in ids):
        raise ValueError("Mother identities must be safe single path components")
    global_counts = {kind: 0 for kind in TEMPLATE_GEOMETRIES}
    selected: list[dict[str, Any]] = []
    for geometry in TEMPERATURE_GEOMETRIES:
        candidates = [m for m in mothers if m.get("split") == "validation"
                      and m.get("temperature_geometry") == geometry]
        if len(candidates) < 2:
            raise ValueError(f"Need two opened validation mothers for {geometry}")
        counts = {kind: 0 for kind in TEMPLATE_GEOMETRIES}
        for _ in range(2):
            available = [m for m in candidates if m not in selected]
            if any(m.get("template_geometry") not in TEMPLATE_GEOMETRIES for m in available):
                raise ValueError("Unknown template geometry in the source mother metadata")
            chosen = min(available, key=lambda m: (counts[m["template_geometry"]],
                         global_counts[m["template_geometry"]], m["index"]))
            if "final" in chosen["mother_id"] or "confirmation" in chosen["mother_id"]:
                raise ValueError("Sealed final/confirmation mothers are forbidden")
            selected.append(copy.deepcopy(chosen))
            counts[chosen["template_geometry"]] += 1
            global_counts[chosen["template_geometry"]] += 1
    source_config = old.get("physics_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("The opened source physics configuration is required")
    source_config = copy.deepcopy(dict(source_config))
    source_manifest_hash = source_file_hash or _canonical_hash(old)
    config_hash = _canonical_hash(source_config)
    plan: list[dict[str, Any]] = []
    noise_seeds: set[int] = set()
    for mother in selected:
        for condition, coefficient in zip(CONDITIONS, COEFFICIENTS_E):
            seed = _noise_seed(mother["mother_id"], condition)
            if seed in noise_seeds:
                raise ValueError("Preregistered noise seed collision")
            noise_seeds.add(seed)
            for kind in INPUT_KINDS:
                plan.append({
                    "schema": SCHEMA,
                    "case_id": f'{mother["mother_id"]}_v5_{condition}_{kind}',
                    "mother_id": mother["mother_id"],
                    "split": "development_validation",
                    "source_split": "validation",
                    "mother": copy.deepcopy(mother),
                    "geometry": mother["temperature_geometry"],
                    "temperature_geometry": mother["temperature_geometry"],
                    "template_geometry": mother["template_geometry"],
                    "condition": condition,
                    "group": "blank" if coefficient == 0 else "weak" if abs(coefficient) == 45 else "normal",
                    "coefficient_e": coefficient,
                    "input_kind": kind,
                    "noise_seed": seed if kind == "noise" else None,
                    "registered_noise_seed": seed,
                    "source_manifest_sha256": source_manifest_hash,
                    "source_physics_config_sha256": config_hash,
                    "source_physics_shape": list(source_config["shape"]),
                    "matched_fields": ["T", "loga", "beta", "template_shape", "geometry"],
                    "same_noise_realization_claimed": False,
                    "temperature_shift_K": 0.0,
                    "continuum_amplitude_scale": 1.0,
                    "template_shape_amplitude_scale": 1.0,
                    "truth_use": "external diagnostic oracle and scoring only",
                    "signed_semantics": "synthetic template deviation; negative values are not concentrations",
                })
    if len(plan) != 180:
        raise RuntimeError("The diagnostic registration must contain exactly 180 inputs")
    return plan


def _ed_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.drive.upper() not in ("E:", "D:"):
        raise ValueError("All new diagnostic data, reports and temporary files must remain on E/D")
    if "research_v5" not in [part.lower() for part in resolved.parts]:
        raise ValueError("Use an explicitly new research_v5 output directory")
    return resolved


def _array_hash(value: Any) -> str:
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _save_npz(path: Path, values: Mapping[str, Any]) -> None:
    import numpy as np
    if path.exists():
        raise FileExistsError(f"Diagnostic artifacts are immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Diagnostic artifacts are immutable: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite_extrema(tensor: Any) -> tuple[float | None, float | None]:
    import torch
    finite = tensor[torch.isfinite(tensor)]
    return (float(finite.min()), float(finite.max())) if finite.numel() else (None, None)


def _expectation_diagnostics(physics: Any, expected: Any, latent_bands: Any, blurred_bands: Any) -> dict[str, Any]:
    import torch
    finite = torch.isfinite(expected)
    lo, hi = _finite_extrema(expected)
    continuous_dn = expected / physics.adc_step + physics.cfg["black_level_DN"]
    maximum_dn = 2 ** int(physics.cfg["adc_bits"]) - 1
    return {
        "expected_all_finite": bool(finite.all()),
        "expected_nonnegative": bool(finite.all() and (expected >= 0).all()),
        "minimum_expected_e": lo,
        "maximum_expected_e": hi,
        "negative_expected_pixels": int((finite & (expected < 0)).sum()),
        "expected_ADC_high_clip_pixels": int((continuous_dn > maximum_dn).sum()),
        "expected_ADC_low_clip_pixels": int((continuous_dn < 0).sum()),
        "expected_ADC_at_or_above_saturation_pixels": int((continuous_dn >= maximum_dn).sum()),
        "minimum_latent_band_e": _finite_extrema(latent_bands)[0],
        "minimum_postblur_band_e": _finite_extrema(blurred_bands)[0],
        "latent_bands_all_finite": bool(torch.isfinite(latent_bands).all()),
        "postblur_bands_all_finite": bool(torch.isfinite(blurred_bands).all()),
        "latent_bands_nonnegative": bool(torch.isfinite(latent_bands).all() and (latent_bands >= 0).all()),
        "postblur_bands_nonnegative": bool(torch.isfinite(blurred_bands).all() and (blurred_bands >= 0).all()),
        "latent_band_minima_e": [_finite_extrema(band)[0] for band in latent_bands[0]],
        "postblur_band_minima_e": [_finite_extrema(band)[0] for band in blurred_bands[0]],
        "negative_latent_band_pixels": int((latent_bands < 0).sum()),
        "negative_postblur_band_pixels": int((blurred_bands < 0).sum()),
        "negative_band_policy": "retain numerical signed-stress diagnostics; do not claim negative concentration",
        "mother_fields_modified_for_legality": False,
    }


def generate_diagnostics(physics: Any, plan: list[dict[str, Any]], output_root: str | Path,
                         options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write only new raw-DN artifacts, separate truth labels, and a manifest.

    Formal execution is the root orchestrator's responsibility after protocol
    freeze. ``fixture_only=True`` permits a smaller physics grid and a plan
    subset for bounded CPU contract checks, and marks every result as a fixture.
    A negative/nonfinite Poisson mean is retained in labels; its noise descriptor
    is unavailable. It is never silently clamped into an alternative experiment.
    The caller must prepare an E/D runtime before importing legacy physics.
    """
    opts = dict(options or {})
    unknown = set(opts) - {"fixture_only"}
    if unknown:
        raise ValueError(f"Unknown diagnostic generation options: {sorted(unknown)}")
    fixture = bool(opts.get("fixture_only", False))
    root = _ed_path(output_root)
    if root.exists() and any(child.name != "_runtime" or not child.is_dir() for child in root.iterdir()):
        raise FileExistsError("Use a new empty diagnostic output directory; previous results are immutable")
    if not plan:
        raise ValueError("An explicitly registered diagnostic plan is required")
    if not fixture and len(plan) != 180:
        raise ValueError("Formal stage1 requires the complete 180-descriptor plan")
    cases = [d.get("case_id") for d in plan]
    if any(not isinstance(case, str) for case in cases) or len(set(cases)) != len(cases):
        raise ValueError("Diagnostic case identities must be unique")
    config_hash = _canonical_hash(physics.cfg)
    mothers_by_id: dict[str, str] = {}
    factorial_by_id: dict[str, set[tuple[float, str]]] = {}
    for desc in plan:
        mother = desc.get("mother", {})
        if mother.get("split") != "validation" or desc.get("source_split") != "validation":
            raise ValueError("Only already-opened validation mothers are allowed")
        if desc.get("mother_id") != mother.get("mother_id") or "final" in desc["mother_id"] or "confirmation" in desc["mother_id"]:
            raise ValueError("Mother identity mismatch or sealed data access")
        if re.fullmatch(r"[A-Za-z0-9_-]+", desc["mother_id"]) is None:
            raise ValueError("Mother identity must be a safe single path component")
        identity = _canonical_hash(mother)
        if desc["mother_id"] in mothers_by_id and mothers_by_id[desc["mother_id"]] != identity:
            raise ValueError("One mother cannot have conflicting seed or geometry descriptors")
        mothers_by_id[desc["mother_id"]] = identity
        if desc.get("input_kind") not in INPUT_KINDS or desc.get("coefficient_e") not in COEFFICIENTS_E:
            raise ValueError("Unregistered input kind or coefficient")
        factorial = factorial_by_id.setdefault(desc["mother_id"], set())
        cell = (float(desc["coefficient_e"]), desc["input_kind"])
        if cell in factorial:
            raise ValueError("Repeated mother/coefficient/input-kind cells are forbidden")
        factorial.add(cell)
        if not fixture and desc.get("source_physics_config_sha256") != config_hash:
            raise ValueError("Formal generation must replay the unchanged source physics configuration")
        for scale in ("continuum_amplitude_scale", "template_shape_amplitude_scale"):
            if desc.get(scale) != 1.0:
                raise ValueError("Mother amplitude or shape must not be rescaled")
        if desc.get("temperature_shift_K") != 0.0:
            raise ValueError("Mother temperature must not be shifted")
        expected_condition = CONDITIONS[COEFFICIENTS_E.index(desc["coefficient_e"])]
        if desc.get("condition") != expected_condition:
            raise ValueError("Signed coefficient/condition mismatch")
        expected_seed = _noise_seed(desc["mother_id"], expected_condition)
        if desc.get("registered_noise_seed") != expected_seed or (
                desc["input_kind"] == "noise" and desc.get("noise_seed") != expected_seed):
            raise ValueError("Noise streams must be preregistered per distinct mean")
    if not fixture:
        complete = {(coefficient, kind) for coefficient in COEFFICIENTS_E for kind in INPUT_KINDS}
        if len(mothers_by_id) != 12 or any(cells != complete for cells in factorial_by_id.values()):
            raise ValueError("Formal stage1 must contain twelve complete 5 x 3 matched mothers")
    # Guard before lazy numerical imports; never create bytecode on C.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import numpy as np
    import torch
    from training_v4.data import mother_state
    if fixture and physics.M.device.type != "cpu":
        raise ValueError("Implementation fixtures are CPU-only")
    if int(torch.count_nonzero(physics.active)) != 1 or not bool(physics.active[2]):
        raise ValueError("Stage1 fixes the original active synthetic template at index 2")
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    mother_cache: dict[str, Any] = {}
    label_cache: dict[tuple[str, float], Any] = {}
    maximum_dn = 2 ** int(physics.cfg["adc_bits"]) - 1
    with torch.no_grad():
        for desc in plan:
            mother_id = desc["mother_id"]
            mother = desc["mother"]
            if mother_id not in mother_cache:
                base, beta, shape, provenance = mother_state(physics, mother)
                mother_cache[mother_id] = (base.clone(), beta.clone(), shape.clone(), provenance)
            base, beta, shape, provenance = mother_cache[mother_id]
            key = (mother_id, float(desc["coefficient_e"]))
            if key not in label_cache:
                state = base.clone()
                state[:, 4] = shape * desc["coefficient_e"] / physics.Sscale
                if not torch.equal(state[:, :4], base[:, :4]):
                    raise RuntimeError("The signed injection altered a protected mother field")
                continuum, species, background, _ = physics.components(state, beta)
                latent_bands = continuum + species + background
                blurred_bands = physics.op.object_blur(latent_bands)
                expected = physics.op.encode_images(blurred_bands)[0]
                continuum_postblur = physics.op.object_blur(continuum)
                template_latent = state[:, 4] * physics.Sscale
                template_postblur = physics.op.object_blur(template_latent)
                color, valid = physics.color_temperature(continuum_postblur)
                diagnostic = _expectation_diagnostics(physics, expected, latent_bands, blurred_bands)
                arrays = {
                    "latent_state": state[0].cpu().numpy(),
                    "latent_temperature_K": (physics.T0 + physics.Tscale * state[0, 0]).cpu().numpy(),
                    "latent_log_amplitude": state[0, 1].cpu().numpy(),
                    "beta": beta[0].cpu().numpy(),
                    "template_shape": shape.cpu().numpy(),
                    "coefficient_e": np.asarray(desc["coefficient_e"], dtype=np.float64),
                    "continuum_latent": continuum[0].cpu().numpy(),
                    "continuum_postblur": continuum_postblur[0].cpu().numpy(),
                    "template_latent_e": template_latent[0].cpu().numpy(),
                    "template_postblur_e": template_postblur[0].cpu().numpy(),
                    "template_e": template_postblur[0].cpu().numpy(),
                    "temperature_color_K": color[0].cpu().numpy(),
                    "evaluation_domain": valid[0].cpu().numpy(),
                    "latent_band_images_e": latent_bands[0].cpu().numpy(),
                    "band_images_e": blurred_bands[0].cpu().numpy(),
                    "expected_e": expected.cpu().numpy(),
                }
                label_path = root / "diagnostic_labels" / mother_id / (desc["condition"] + ".npz")
                _save_npz(label_path, arrays)
                label = {
                    "mother_id": mother_id, "condition": desc["condition"],
                    "coefficient_e": desc["coefficient_e"],
                    "path": label_path.relative_to(root).as_posix(),
                    "sha256": _file_hash(label_path),
                    "truth_use": "external diagnostic oracle and scoring only",
                    "protected_field_hashes": {name: _array_hash(value) for name, value in {
                        "latent_temperature_K": arrays["latent_temperature_K"],
                        "latent_log_amplitude": arrays["latent_log_amplitude"],
                        "beta": arrays["beta"], "template_shape": arrays["template_shape"],
                        "continuum_latent": arrays["continuum_latent"],
                    }.items()},
                    "replayed_mother_provenance": provenance,
                    "original_provenance_exact_hash_match": {name: mother.get(name) == provenance.get(name)
                        for name in ("temperature_field_sha256", "amplitude_field_sha256", "template_shape_sha256")},
                    "actual_template_preblur_min_e": float(template_latent.min()),
                    "actual_template_preblur_max_e": float(template_latent.max()),
                    "actual_template_postblur_min_e": float(template_postblur.min()),
                    "actual_template_postblur_max_e": float(template_postblur.max()),
                    "diagnostics": diagnostic,
                }
                labels.append(label)
                label_cache[key] = (expected, label)
            expected, label = label_cache[key]
            row = {name: copy.deepcopy(value) for name, value in desc.items() if name != "mother"}
            row.update({"fixture_only": fixture, "physics_config_sha256": config_hash,
                        "label_path": label["path"], "label_sha256": label["sha256"],
                        "expectation_diagnostics": copy.deepcopy(label["diagnostics"]),
                        "raw_available": False, "raw_path": None, "raw_hash": None,
                        "raw_file_sha256": None, "raw_generation_status": "PENDING"})
            if not label["diagnostics"]["expected_all_finite"]:
                row["raw_generation_status"] = "UNAVAILABLE_NONFINITE_EXPECTATION"
                rows.append(row)
                continue
            kind = desc["input_kind"]
            if kind == "noise" and not label["diagnostics"]["expected_nonnegative"]:
                row["raw_generation_status"] = "UNAVAILABLE_NEGATIVE_POISSON_MEAN"
                rows.append(row)
                continue
            if kind == "noise":
                generator = torch.Generator(device=physics.M.device).manual_seed(desc["noise_seed"])
                measured = torch.poisson(expected, generator=generator)
                measured = measured + torch.randn(expected.shape, device=expected.device,
                                dtype=expected.dtype, generator=generator) * physics.cfg["read_noise_e"]
            else:
                measured = expected
            unquantized_dn = measured / physics.adc_step + physics.cfg["black_level_DN"]
            rounded = unquantized_dn.round()
            adc_diagnostic = {
                "unquantized_below_zero_pixels": int((unquantized_dn < 0).sum()),
                "unquantized_above_maximum_pixels": int((unquantized_dn > maximum_dn).sum()),
                "rounded_low_clip_pixels": int((rounded < 0).sum()),
                "rounded_high_clip_pixels": int((rounded > maximum_dn).sum()),
            }
            raw = unquantized_dn if kind == "continuous" else rounded.clamp(0, maximum_dn).to(torch.uint16)
            # Saturation is diagnosed, never grounds for changing a mother.
            adc_diagnostic["raw_saturated_pixels"] = int((raw.to(expected.dtype) >= maximum_dn).sum())
            adc_diagnostic["raw_at_zero_pixels"] = int((raw.to(expected.dtype) <= 0).sum())
            raw_array = raw.cpu().numpy()
            raw_path = root / "raw" / mother_id / (desc["condition"] + "_" + kind + ".npz")
            _save_npz(raw_path, {"raw_dn": raw_array})
            row.update({"raw_available": True, "raw_path": raw_path.relative_to(root).as_posix(),
                        "raw_hash": _array_hash(raw_array), "raw_file_sha256": _file_hash(raw_path),
                        "raw_generation_status": "GENERATED", "ADC_diagnostics": adc_diagnostic,
                        "shape": list(raw_array.shape), "dtype": str(raw_array.dtype)})
            rows.append(row)
    # Verify all +/- and blank labels keep the same protected continuum fields.
    grouped: dict[str, set[str]] = {}
    for label in labels:
        grouped.setdefault(label["mother_id"], set()).add(_canonical_hash(label["protected_field_hashes"]))
    if any(len(values) != 1 for values in grouped.values()):
        raise RuntimeError("Matched signed conditions changed protected mother fields")
    counts = {kind: sum(r["input_kind"] == kind and r["raw_available"] for r in rows) for kind in INPUT_KINDS}
    manifest = {
        "schema": SCHEMA, "fixture_only": fixture,
        "physics_config": copy.deepcopy(physics.cfg), "physics_config_sha256": config_hash,
        "generator_source_sha256": _file_hash(Path(__file__)),
        "plan_sha256": _canonical_hash(plan),
        "source_manifest_sha256": sorted({d["source_manifest_sha256"] for d in plan}),
        "planned_rows": len(plan), "generated_raw_rows": sum(counts.values()),
        "unavailable_rows": len(rows) - sum(counts.values()), "counts": counts,
        "selected_mothers": [copy.deepcopy(next(d["mother"] for d in plan if d["mother_id"] == mid))
                             for mid in mother_cache],
        "labels": labels, "rows": rows,
        "matched_continuum_background_shape_verified": True,
        "same_noise_realization_claimed": False,
        "noise_policy": "distinct preregistered streams for every mother/coefficient Poisson mean",
        "protected_geometry_note": "Original front/stripe geometry names and seeds are unchanged; old stripe is a two-stripe field",
        "negative_concentration_claimed": False,
        "labels_accepted_by_deployment": False,
        "final_test_opened": False, "confirmation_test_opened": False,
    }
    _json_write(root / "data_manifest.json", manifest)
    return manifest
