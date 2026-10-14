"""Raw-only frozen continuum sources and explicit external cache adapters.

The v3 source always uses the selected old-init joint inverse. Its continuum
is rebuilt from state,beta, never from an already blurred legacy image. Cache
selection belongs to the orchestrator; a bad cache is a source failure and
never triggers a silent new inverse. C_true is supplied only by diagnostics.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
import time

from . import common
from .inputs import InputIntegrityError, _path_in, _verify_file, array_hash

BACKBONE_PATH = "reports/training_v4/run_001/backbone_selection.json"
BACKBONE_SHA256 = "d4f6cb17450308717623821995c7d2ab10d487d560c87c3f8979418708b64f71"
CONSUMED_SELECTION_SHA256 = "e8a00514ef2b761d1dbd531adbda957d09d4f2883382574dc6a4f75ff35bf667"
V3_SOURCE_SHA256 = {
    "training_v4/scalar_engine.py": "e6f9436dbd2bf890e1b27e6e0d4a8e4feb4f3e470cc338510da1772eb772ca8c",
    "training_v4/model.py": "627ef9ffb216ad28389a60fd0040a706a8ab36aad677bcf20807a838b1a33100",
    "repair_v3/solver.py": "f10f26e5eab130363ddb0e32ffce7511f6bc8f449bf2dbeb484ede49b5fb659e",
    "solver.py": "2c6eb39c24ea4aa4ef07c3ad6423edcdccb869b24592fad49624b86cd54e8a2c",
    "physics.py": "536654ce52dff3de99490182320859aa25a96e29737cf169c9ec1b7a0f7d109a",
}


def frozen_selection():
    path = common.ROOT/BACKBONE_PATH
    _verify_file(path, BACKBONE_SHA256)
    selection = common.read_json(path)
    consumed = {k: selection[k] for k in ("method", "solver_config", "physics_config")}
    if common.canonical_hash(consumed) != CONSUMED_SELECTION_SHA256:
        raise InputIntegrityError("Frozen v3 backbone selection changed")
    return selection


def _check_selection(physics, selection):
    if not isinstance(selection, Mapping):
        raise TypeError("Pass the external frozen backbone selection mapping")
    consumed = {k: selection[k] for k in ("method", "solver_config", "physics_config")}
    if common.canonical_hash(consumed) != CONSUMED_SELECTION_SHA256:
        raise InputIntegrityError("Only the frozen old_init__joint_update_v3 selection is admitted")
    if common.canonical_hash(physics.cfg) != common.canonical_hash(selection["physics_config"]):
        raise InputIntegrityError("Physics differs from the frozen v3 calibration")
    return {"consumed_selection_sha256": CONSUMED_SELECTION_SHA256,
            "physics_config_sha256": common.canonical_hash(physics.cfg),
            "solver_config_sha256": common.canonical_hash(selection["solver_config"]),
            "method": selection["method"], "backbone_selection_sha256": BACKBONE_SHA256}


def _v3_functions():
    common.configure_runtime()
    for path, expected in V3_SOURCE_SHA256.items():
        _verify_file(common.ROOT/path, expected)
    from training_v4.scalar_engine import invert_single
    from training_v4.model import physical_output
    return invert_single, physical_output


def _raw(raw_dn, physics):
    import torch
    if isinstance(raw_dn, Mapping):
        raise TypeError("Inference accepts the raw DN array/tensor only")
    raw = torch.as_tensor(raw_dn, dtype=physics.M.dtype, device=physics.M.device).detach()
    if raw.ndim == 2:
        raw = raw[None]
    if raw.ndim != 3 or tuple(raw.shape) != (1, *physics.cfg["shape"]):
        raise ValueError("Exactly one finite B=1,H,W raw DN frame is required")
    maximum = 2**physics.cfg["adc_bits"]-1
    if not bool(torch.isfinite(raw).all()) or bool(((raw < 0) | (raw >= maximum)).any()):
        raise ValueError("Nonfinite/out-of-range/saturated raw is a retained source failure")
    return raw


def _sync(physics):
    import torch
    if physics.M.device.type == "cuda":
        torch.cuda.synchronize(physics.M.device)


def _memory(physics):
    import torch
    result = {"peak_CPU_RSS_MiB": None, "CPU_peak_scope": "process lifetime high-water mark",
              "peak_GPU_allocated_MiB": None, "GPU_peak_scope": "process counter since most recent reset"}
    try:
        import psutil
        info = psutil.Process().memory_info()
        result["CPU_RSS_MiB"] = info.rss/2**20
        if hasattr(info, "peak_wset"):
            result["peak_CPU_RSS_MiB"] = info.peak_wset/2**20
        else:
            import resource
            result["peak_CPU_RSS_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except (ImportError, AttributeError, OSError):
        pass
    if physics.M.device.type == "cuda":
        result["peak_GPU_allocated_MiB"] = torch.cuda.max_memory_allocated(physics.M.device)/2**20
    return result


def _metadata(source, mode, raw, physics, started, **extra):
    _sync(physics)
    return {"source": source, "mode": mode, "wall_s": time.perf_counter()-started,
            "raw_array_sha256": array_hash(raw[0].detach().cpu().numpy()),
            "physics_config_sha256": common.canonical_hash(physics.cfg),
            "dtype": str(raw.dtype), "device": str(raw.device),
            "truth_or_identity_used": False, "optimizer_updates": 0,
            "continuum_domain": "latent; postblur formed once", **_memory(physics), **extra}


def _failure(source, error, physics, started, mode):
    _sync(physics)
    return {"status": "SOURCE_FAILED", "continuum_latent": None, "continuum_postblur": None,
            "temperature_color_K": None, "valid": None, "baseline_output": None,
            "source_metadata": {"source": source, "mode": mode, "wall_s": time.perf_counter()-started,
                                "truth_or_identity_used": False, **_memory(physics)},
            "error": str(error), "error_type": type(error).__name__}


def _v3_result(raw, physics, state, beta, history, physical_output, metadata):
    import torch
    state, beta = state.detach(), beta.detach()
    if tuple(state.shape) != (1, 5, *physics.cfg["shape"]) or tuple(beta.shape) != (1, 3):
        raise ValueError("Frozen v3 state/beta shape mismatch")
    if not bool(torch.isfinite(state).all() and torch.isfinite(beta).all()):
        raise ValueError("Frozen v3 returned nonfinite state/beta")
    raw_e = physics.raw_to_electrons(raw)
    original = physical_output(physics, state, beta, raw_e)
    latent = physics.components(state, beta)[0].detach()
    # Original physical_output blurred latent continuum exactly once. It is
    # reused as postblur, never fed back into object_blur as a latent source.
    postblur = original["continuum_images_e"].detach()
    for name in ("template_e", "temperature_color_K", "expected_e", "continuum_images_e"):
        if not bool(torch.isfinite(original[name]).all()):
            raise ValueError("Frozen v3 physical output nonfinite: " + name)
    baseline = {**original, "template_postblur_e": original["template_e"].detach(),
                "template_latent_e": state[:, 4]*physics.Sscale,
                "continuum_latent": latent, "continuum_postblur": postblur,
                "raw_e": raw_e, "raw_residual_e": original["expected_e"]-raw_e,
                "info": {"method": "v3_physics", "status": "SUCCESS",
                         "inverse_history": history, "source_metadata": metadata,
                         "solver_budget_metadata_enforced_by_scalar_engine": False}}
    return {"status": "SUCCESS", "state": state, "beta": beta,
            "continuum_latent": latent, "continuum_postblur": postblur,
            "temperature_color_K": original["temperature_color_K"].detach(),
            "valid": original["valid"].detach().bool(), "baseline_output": baseline,
            "source_metadata": metadata}


def infer_v3(raw_dn, physics, selection):
    """Exact frozen v3 inverse, with no labels, case identity or truth argument."""
    import torch
    started = time.perf_counter()
    try:
        provenance = _check_selection(physics, selection)
        raw = _raw(raw_dn, physics)
        invert_single, physical_output = _v3_functions()
        _sync(physics)
        with torch.no_grad():
            state, beta, history, engine_seconds = invert_single(
                physics, physics.raw_to_electrons(raw), selection["method"], copy.deepcopy(selection["solver_config"]))
            metadata = _metadata("C_v3", "new_frozen_inference", raw, physics, started,
                                 engine_seconds=engine_seconds, source_sha256=V3_SOURCE_SHA256,
                                 **provenance)
            result = _v3_result(raw, physics, state, beta, history, physical_output, metadata)
        _sync(physics)
        result["source_metadata"].update(_memory(physics))
        result["source_metadata"]["wall_s"] = time.perf_counter()-started
        return result
    except Exception as error:
        return _failure("C_v3", error, physics, started, "new_frozen_inference")


def _cache_raw_check(raw, receipt):
    import numpy as np
    expected = receipt.get("raw_array_sha256")
    # Deployment converts uint16 camera DN to calibrated floating point. The
    # stored raw hash includes its original dtype, so reconstruct that dtype
    # only when every value survives an exact round trip. Fractional/truncated
    # inputs never acquire the hash of a valid integer observation.
    observed = raw[0].detach().cpu().numpy()
    if receipt.get("raw_dtype") not in ("uint16", "float32"):
        raise InputIntegrityError("Registered original raw dtype is required for cached sources")
    if list(observed.shape) != receipt.get("raw_shape"):
        raise InputIntegrityError("Source cache original raw geometry mismatch")
    restored = observed.astype(np.dtype(receipt["raw_dtype"]))
    if not np.array_equal(restored.astype(observed.dtype), observed):
        raise InputIntegrityError("Source cache original dtype reconstruction is not lossless")
    if expected is None or array_hash(restored) != expected:
        raise InputIntegrityError("Source cache raw dtype/shape/bytes mismatch")
    return {"raw_array_sha256": expected, "original_raw_dtype": receipt["raw_dtype"],
            "inference_raw_array_sha256": array_hash(observed),
            "original_dtype_roundtrip_verified": True}


def cached_v3(raw_dn, physics, selection, cache_receipt):
    """Reuse hash-bound original state/beta; invalid cache remains a failure."""
    import numpy as np
    import torch
    started = time.perf_counter()
    try:
        provenance = _check_selection(physics, selection)
        raw = _raw(raw_dn, physics)
        if not cache_receipt or cache_receipt.get("cache_kind") != "frozen_v4_state_beta":
            raise InputIntegrityError("A frozen original v3 state/beta cache receipt is required")
        raw_provenance = _cache_raw_check(raw, cache_receipt)
        parent = common.ROOT/"outputs/training_v4/run_001/cache"
        meta_path = _path_in(cache_receipt["metadata_path"], parent)
        _verify_file(meta_path, cache_receipt["metadata_sha256"])
        meta = common.read_json(meta_path)
        if meta != cache_receipt["legacy_metadata"]:
            raise InputIntegrityError("v3 cache metadata changed since registration")
        expected = {"raw_hash": cache_receipt["raw_array_sha256"], "method": selection["method"],
                    "physics_config_sha256": provenance["physics_config_sha256"],
                    "calibration_sha256": provenance["physics_config_sha256"],
                    "solver_config_sha256": provenance["solver_config_sha256"],
                    "engine": "single_frame_cached_Jacobian_v3_math", "labels_used": False}
        if any(meta.get(k) != v for k, v in expected.items()):
            raise InputIntegrityError("v3 legacy cache raw/method/config/label provenance mismatch")
        path = _path_in(cache_receipt["path"], parent)
        if meta["cache_sha256"] != cache_receipt["sha256"] or path.name != meta["cache_key"]+".npz":
            raise InputIntegrityError("v3 cache metadata/file identity mismatch")
        _verify_file(path, cache_receipt["sha256"])
        with np.load(path, allow_pickle=False) as z:
            state = torch.as_tensor(z["state"].copy(), dtype=physics.M.dtype, device=physics.M.device)[None]
            beta = torch.as_tensor(z["beta"].copy(), dtype=physics.M.dtype, device=physics.M.device)[None]
        _, physical_output = _v3_functions()
        with torch.no_grad():
            metadata = _metadata("C_v3", "original_state_beta_cache", raw, physics, started,
                cache_path=cache_receipt["path"], cache_sha256=cache_receipt["sha256"],
                original_source_seconds=meta.get("seconds"),
                original_peak_GPU_allocated_MiB=meta.get("peak_GPU_allocated_MiB"),
                legacy_engine_torch_version="not recorded separately in legacy metadata",
                source_sha256=V3_SOURCE_SHA256, **provenance, **raw_provenance)
            result = _v3_result(raw, physics, state, beta, [meta.get("history", {})], physical_output, metadata)
        _sync(physics)
        result["source_metadata"].update(_memory(physics))
        result["source_metadata"]["wall_s"] = time.perf_counter()-started
        return result
    except Exception as error:
        return _failure("C_v3", error, physics, started, "original_state_beta_cache")


def _v5_options(physics, options):
    protocol = common.read_json(common.ROOT/"research_v5/protocol.json")
    frozen = protocol["continuum_options"]
    if common.canonical_hash(physics.cfg) != protocol["physics_config_sha256"]:
        raise InputIntegrityError("C_v5 requires the frozen calibration")
    if options is not None and common.canonical_hash(options) != common.canonical_hash(frozen):
        raise InputIntegrityError("C_v5 warm options differ from the frozen estimator")
    return copy.deepcopy(frozen)


def infer_v5(raw_dn, physics, options=None):
    """The exact standalone frozen 700/905 estimator, no v5 package import."""
    import torch
    started = time.perf_counter()
    try:
        raw = _raw(raw_dn, physics)
        frozen = _v5_options(physics, options)
        estimator = common.frozen_v5("continuum_solver").estimate_continuum
        with torch.no_grad():
            output = estimator(physics, physics.raw_to_electrons(raw), frozen)
        metadata = _metadata("C_v5", "new_frozen_inference", raw, physics, started,
                             continuum_options_sha256=common.canonical_hash(frozen))
        return {**output, "status": "SUCCESS", "source_metadata": metadata}
    except Exception as error:
        return _failure("C_v5", error, physics, started, "new_frozen_inference")


def cached_v5(raw_dn, physics, cache_receipt):
    """Read frozen latent C directly; retain its original once-blurred output."""
    import numpy as np
    import torch
    started = time.perf_counter()
    try:
        raw = _raw(raw_dn, physics)
        frozen = _v5_options(physics, None)
        if not cache_receipt or cache_receipt.get("cache_kind") != "frozen_v5_development_latent":
            raise InputIntegrityError("A frozen latent C_v5 development cache receipt is required")
        raw_provenance = _cache_raw_check(raw, cache_receipt)
        path = _path_in(cache_receipt["path"], common.ROOT/"outputs/research_v5/run_001/development_grid_predictions")
        _verify_file(path, cache_receipt["sha256"])
        output = {}
        with np.load(path, allow_pickle=False) as z:
            for key in ("continuum_latent", "continuum_postblur", "temperature_color_K", "continuum_valid"):
                value = torch.as_tensor(z[key].copy(), device=physics.M.device,
                                       dtype=torch.bool if key == "continuum_valid" else physics.M.dtype)
                output["valid" if key == "continuum_valid" else key] = value
        for key in ("continuum_latent", "continuum_postblur"):
            if tuple(output[key].shape) != (1, 7, *physics.cfg["shape"]) or not bool(torch.isfinite(output[key]).all()):
                raise InputIntegrityError("Cached C_v5 shape/finiteness mismatch: " + key)
        if tuple(output["temperature_color_K"].shape) != tuple(raw.shape) or tuple(output["valid"].shape) != tuple(raw.shape):
            raise InputIntegrityError("Cached C_v5 color/valid geometry mismatch")
        if not bool(torch.isfinite(output["temperature_color_K"]).all()):
            raise InputIntegrityError("Cached C_v5 color nonfinite")
        # Check domain semantics without changing the cached arrays. This is a
        # forward-only consistency check, never a second blur of postblur C.
        with torch.no_grad():
            expected_postblur = physics.op.object_blur(output["continuum_latent"])
        if not torch.allclose(expected_postblur, output["continuum_postblur"], rtol=3e-6, atol=3e-3):
            raise InputIntegrityError("Cached C_v5 latent/postblur one-blur relationship failed")
        metadata = _metadata("C_v5", "frozen_latent_cache", raw, physics, started,
            cache_path=cache_receipt["path"], cache_sha256=cache_receipt["sha256"],
            original_source_seconds=cache_receipt.get("original_source_seconds"),
            original_route_seconds=cache_receipt.get("original_route_seconds"),
            original_time_scope=cache_receipt.get("time_scope"),
            continuum_options_sha256=common.canonical_hash(frozen), **raw_provenance)
        return {**output, "status": "SUCCESS", "source_metadata": metadata}
    except Exception as error:
        return _failure("C_v5", error, physics, started, "frozen_latent_cache")
