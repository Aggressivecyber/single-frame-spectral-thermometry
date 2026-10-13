"""Four labelled diagnostic oracles sharing one raw frame and one solver.

O3 is evaluated before reading truth.  Its continuum estimate and signed linear
solve receive only the observed raw electrons, passed calibration, and the fixed
numerical options.  Truth is used later by O0/O1/O2 and external scoring.
"""
from __future__ import annotations

import copy
import hashlib
import json
from time import perf_counter
from typing import Any, Mapping


LADDER_DEFAULTS = {
    "alpha_g": 0.0,
    "lambda0_factor": 1e-6,
    "lambda_beta_factor": 1e-6,
    "rtol": 1e-4,
    "max_iterations": 200,
}


def _array_hash(value: Any) -> str:
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """Make solver diagnostics strict-JSON-safe; nonfinite diagnostics are null."""
    import math
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _numpy_image(value: Any) -> Any:
    import numpy as np
    import torch
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Scoring requires a single H x W image")
    return np.asarray(array, dtype=np.float64)


def _tensor(value: Any, physics: Any, unbatched_ndim: int) -> Any:
    import torch
    tensor = torch.as_tensor(value, dtype=physics.M.dtype, device=physics.M.device)
    if tensor.ndim == unbatched_ndim:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != unbatched_ndim + 1 or tensor.shape[0] != 1:
        raise ValueError("The oracle ladder evaluates exactly one raw frame at a time")
    return tensor


def _options(options: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outer = dict(options or {})
    unknown = set(outer) - {"continuum", "linear", "score"}
    if unknown:
        raise ValueError(f"Oracle options cannot contain case/truth metadata: {sorted(unknown)}")
    continuum = copy.deepcopy(dict(outer.get("continuum", {})))
    linear = copy.deepcopy(LADDER_DEFAULTS)
    requested = dict(outer.get("linear", {}))
    for key, value in requested.items():
        if key not in LADDER_DEFAULTS or value != LADDER_DEFAULTS[key]:
            raise ValueError(f"The registered oracle ladder fixes {key}; use development outside this ladder")
    linear.update(requested)
    score = {"support_fraction": 0.2, "threshold_e": 60.0}
    requested_score = dict(outer.get("score", {}))
    if set(requested_score) - set(score):
        raise ValueError("Unknown oracle scoring option")
    score.update(requested_score)
    if not 0 < float(score["support_fraction"]) <= 1 or float(score["threshold_e"]) != 60.0:
        raise ValueError("Invalid registered support fraction or detection threshold")
    return continuum, linear, score


def _score(physics: Any, raw_e: Any, output: Mapping[str, Any], continuum: Mapping[str, Any],
           truth: Mapping[str, Any], score_options: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    prediction = _numpy_image(output["template_postblur_e"])
    target = _numpy_image(truth.get("template_postblur_e", truth.get("template_e")))
    color = _numpy_image(continuum["temperature_color_K"])
    true_color = _numpy_image(truth["temperature_color_K"])
    valid = _numpy_image(continuum["valid"]).astype(bool)
    domain = _numpy_image(truth.get("evaluation_domain", np.ones_like(target, dtype=bool))).astype(bool)
    if any(image.shape != target.shape for image in (prediction, color, true_color, valid, domain)):
        raise ValueError("Truth, predictions and registered domains must share the same shape")
    finite = np.isfinite(prediction)
    truth_finite = np.isfinite(target)
    complete = bool(finite.all() and truth_finite.all())
    error = prediction - target
    energy = float(np.square(target).sum()) if truth_finite.all() else None
    peak = float(np.abs(target).max()) if truth_finite.all() else None
    support = truth_finite & (np.abs(target) >= float(score_options["support_fraction"]) * (peak or 0.0)) & (target != 0)
    positive, negative = support & (target > 0), support & (target < 0)
    threshold = float(score_options["threshold_e"])
    sign_correct = finite & (((target > 0) & (prediction > 0)) | ((target < 0) & (prediction < 0)))
    threshold_correct = finite & (((target > 0) & (prediction > threshold)) | ((target < 0) & (prediction < -threshold)))
    color_domain = domain & np.isfinite(true_color)
    color_complete = bool(np.isfinite(color[color_domain]).all())
    color_error = color - true_color
    expected = output["expected_e"]
    weights = output.get("weights")
    if weights is None:
        weights = (raw_e.clamp_min(0) + physics.extra_variance).reciprocal()
    residual = expected - raw_e
    raw_finite = bool(torch.isfinite(residual).all() and torch.isfinite(weights).all())
    metrics = {
        "pixels": int(target.size),
        "template_SSE": float(np.square(error).sum()) if complete else None,
        "truth_energy": energy,
        "template_MSE_e2": float(np.square(error).mean()) if complete else None,
        "template_RMSE_e": float(np.sqrt(np.square(error).mean())) if complete else None,
        "template_L2": float(np.sqrt(np.square(error).sum() / energy)) if complete and energy else None,
        "signed_projection": float((prediction * target).sum() / energy) if complete and energy else None,
        "sign_recall": float(sign_correct[support].mean()) if support.any() else None,
        "positive_sign_recall": float(sign_correct[positive].mean()) if positive.any() else None,
        "negative_sign_recall": float(sign_correct[negative].mean()) if negative.any() else None,
        "recall_at_fixed60": float(threshold_correct[support].mean()) if support.any() else None,
        "positive_recall60": float((finite[positive] & (prediction[positive] > threshold)).mean()) if positive.any() else None,
        "negative_recall60": float((finite[negative] & (prediction[negative] < -threshold)).mean()) if negative.any() else None,
        "null_pixel_FP60": float((finite & (np.abs(prediction) > threshold)).mean()) if energy == 0 else None,
        "null_frame_FP60": bool((finite & (np.abs(prediction) > threshold)).any()) if energy == 0 else None,
        "null_signed_bias_e": float(prediction.mean()) if energy == 0 and complete else None,
        "truth_support_pixels": int(support.sum()),
        "support_fraction_of_own_postblur_peak": float(score_options["support_fraction"]),
        "detection_threshold_e": threshold,
        "template_finite_coverage": float(finite.mean()),
        "color_finite_coverage": float(np.isfinite(color).mean()),
        "valid_coverage": float(valid.mean()),
        "valid_and_finite_coverage": float((valid & np.isfinite(color) & finite).mean()),
        "registered_color_domain_pixels": int(color_domain.sum()),
        "color_RMSE_K": float(np.sqrt(np.square(color_error[color_domain]).mean())) if color_complete and color_domain.any() else None,
        "color_bias_K": float(color_error[color_domain].mean()) if color_complete and color_domain.any() else None,
        "raw_reprojection_RMSE_e": float(residual.square().mean().sqrt()) if raw_finite else None,
        "raw_weighted_MSE": float((weights * residual.square()).mean()) if raw_finite else None,
        "raw_weighted_RMSE": float((weights * residual.square()).mean().sqrt()) if raw_finite else None,
        "raw_residual_all_finite": raw_finite,
        "raw_weight_sha256": _array_hash(weights.detach().cpu().numpy()),
        "raw_weight_source": "same observed frame; fixed during each linear solve",
        "all_template_pixels_finite": complete,
    }
    if "latent_state" in truth and "state" in continuum:
        true_state = _tensor(truth["latent_state"], physics, 3)
        predicted_state = _tensor(continuum["state"], physics, 3)
        delta = physics.Tscale * (predicted_state[:, 0] - true_state[:, 0])
        metrics["latent_temperature_RMSE_K_diagnostic"] = float(delta.square().mean().sqrt()) if bool(torch.isfinite(delta).all()) else None
        metrics["latent_temperature_truth_is_deployable"] = False
    return _plain(metrics)


def _cg_diagnostic(info: Mapping[str, Any]) -> dict[str, Any]:
    """Record the final true residual and hindsight trace without selecting output."""
    frames = info.get("frames", [])
    frame = frames[0] if frames else info
    cg = frame.get("CG", frame.get("cg", {}))
    history = cg.get("true_residual_history", [])
    best: dict[str, Any] | None = None
    objective_history = frame.get("objective_history", cg.get("objective_history", []))
    best_objective: dict[str, Any] | None = None
    import math
    for index, entry in enumerate(history):
        if isinstance(entry, Mapping):
            iteration = entry.get("iteration", entry.get("iter", index))
            value = entry.get("relative_true_residual", entry.get("relative_residual", entry.get("residual")))
        else:
            iteration, value = index, entry
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and (best is None or value < best["value"]):
            best = {"iteration": int(iteration), "value": value}
    for index, entry in enumerate(objective_history):
        if isinstance(entry, Mapping):
            iteration = entry.get("iteration", entry.get("iter", index))
            value = entry.get("objective", entry.get("value"))
        else:
            iteration, value = index, entry
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and (best_objective is None or value < best_objective["value"]):
            best_objective = {"iteration": int(iteration), "value": value}
    return _plain({
        "iterations": cg.get("iterations"),
        "relative_true_residual": cg.get("relative_true_residual"),
        "converged": cg.get("converged"),
        "stop": cg.get("stop"),
        "accepted": frame.get("accepted"),
        "true_residual_history": history,
        "objective_history": objective_history,
        "best_residual_iterate_diagnostic": best,
        "best_iterate_diagnostic": {
            "criterion": "minimum recorded fixed raw quadratic objective",
            "best": best_objective,
            "role": "retrospective diagnostic only",
            "output_selected_by_this_diagnostic": False,
            "truth_best_iterate_available": False,
            "returned_output": "unaltered final solver output",
        },
    })


def _physical_result(estimated: Mapping[str, Any], linear: Mapping[str, Any]) -> dict[str, Any]:
    """Compose the same validity boundary as the independent deployment API."""
    import torch
    frames = linear.get("info", {}).get("frames", [])
    expected = linear["expected_e"]
    solved = expected.new_tensor([
        bool(frame.get("accepted", False) and frame.get("CG", {}).get("converged", False)
             and frame.get("expected_e_nonnegative", bool((expected[index] >= 0).all())))
        for index, frame in enumerate(frames)
    ], dtype=torch.bool)
    if len(solved) != len(expected):
        raise ValueError("Deployment validity requires one signed-solver report per frame")
    valid = estimated["valid"] & solved[:, None, None]
    return {**estimated, **linear, "valid": valid, "abstain": ~valid,
            "info": {"continuum": estimated.get("info", {}), "linear": linear.get("info", {}),
                     "continuum_frozen": True, "truth_or_identity_access": False,
                     "diagnostic_fixed_beta_used": False,
                     "validity_rule": "postblur color validity and finite accepted converged signed solve with nonnegative raw expectation"}}


def run_oracle_case(physics: Any, raw_dn: Any, truth: Mapping[str, Any],
                    continuum_result: Mapping[str, Any] | None = None,
                    options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return four JSON rows, solver outputs, and the deployable O3 result.

    ``continuum_result`` is an optional externally computed estimate used only
    for a consistency check. O1/O3 always use a fresh ``estimate_continuum``
    evaluation of this same raw. This prevents an oracle-labelled cache or a
    accidentally true continuum from becoming the deployed O3 input.
    Scoring and consistency diagnostics cannot choose a solver iterate.
    """
    import numpy as np
    import torch
    from research_v5.continuum_solver import estimate_continuum
    from research_v5.signed_linear_solver import solve_signed_linear
    if isinstance(raw_dn, Mapping):
        raise TypeError("The oracle receives the raw-DN array directly, never a label-bearing record")
    continuum_options, linear_options, score_options = _options(options)
    raw = _tensor(raw_dn, physics, 2)
    if tuple(raw.shape[-2:]) != tuple(physics.cfg["shape"]):
        raise ValueError("Raw geometry differs from the passed calibration")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Nonfinite raw input must be retained as a failed input by the caller")
    raw_e = physics.raw_to_electrons(raw)
    raw_hash = _array_hash(np.asarray(raw_dn) if not isinstance(raw_dn, torch.Tensor) else raw_dn.detach().cpu().numpy())
    electron_hash = _array_hash(raw_e.detach().cpu().numpy())
    start = perf_counter()
    estimated = estimate_continuum(physics, raw_e, copy.deepcopy(continuum_options))
    continuum_latency = perf_counter() - start
    estimated_C = estimated["continuum_latent"]
    outputs: dict[str, Any] = {}
    timings: dict[str, float] = {}
    failures: dict[str, dict[str, str]] = {}

    def solve(level: str, C: Any, fixed_beta: Any = None) -> None:
        begin = perf_counter()
        try:
            outputs[level] = solve_signed_linear(physics, raw_e, C,
                                copy.deepcopy(linear_options), fixed_beta=fixed_beta)
        except Exception as error:
            outputs[level] = None
            failures[level] = {"type": type(error).__name__, "message": str(error)}
        timings[level] = perf_counter() - begin

    # This call precedes every access to truth or the supplied reference cache.
    solve("O3", estimated_C)
    true_C = _tensor(truth["continuum_latent"], physics, 3)
    true_beta = _tensor(truth["beta"], physics, 1)
    true_postblur = physics.op.object_blur(true_C)
    true_color, true_valid = physics.color_temperature(true_postblur)
    true_continuum = {"continuum_latent": true_C, "continuum_postblur": true_postblur,
                      "temperature_color_K": true_color, "valid": true_valid}
    if "latent_state" in truth:
        true_continuum["state"] = _tensor(truth["latent_state"], physics, 3)
    solve("O0", true_C, true_beta)
    solve("O1", estimated_C, true_beta)
    solve("O2", true_C)
    # No truth or supplied-cache field participates in deployment validity.
    o3 = None if outputs["O3"] is None else _physical_result(estimated, outputs["O3"])

    supplied_consistency = None
    if continuum_result is not None:
        reference = _tensor(continuum_result["continuum_latent"], physics, 3)
        difference = reference - estimated_C
        supplied_consistency = {
            "supplied_estimate_used_for_solver": False,
            "fresh_estimator_used_for_O1_O3": True,
            "continuum_max_abs_difference_e": float(difference.abs().max()) if bool(torch.isfinite(difference).all()) else None,
        }
    rows: list[dict[str, Any]] = []
    levels = (("O0", "true", "fixed_true", true_continuum),
              ("O1", "estimated", "fixed_true", estimated),
              ("O2", "true", "jointly_estimated", true_continuum),
              ("O3", "estimated", "jointly_estimated", o3 or estimated))
    for level, C_source, beta_mode, C_result in levels:
        output = outputs[level]
        row = {"level": level, "oracle_level": level,
               "diagnostic_oracle": level != "O3", "deployable_path": level == "O3",
               "continuum_source": C_source, "beta_mode": beta_mode,
               "raw_dn_sha256": raw_hash, "raw_e_sha256": electron_hash,
               "linear_options": copy.deepcopy(linear_options),
               "solver_latency_s": timings[level],
               "continuum_latency_s": continuum_latency if C_source == "estimated" else 0.0,
               "latency_s": timings[level] + (continuum_latency if C_source == "estimated" else 0.0),
               "latency_note": "estimated-continuum time included per deployable level; shared estimate actually run once",
               "status": "FAILED" if output is None else "RETURNED",
               "truth_used_for_scoring_only_after_O3": True,
               "output_selected_by_truth": False}
        if output is None:
            row.update({"error": failures[level], "metrics": None, "CG": None, "solver_info": None})
        else:
            info = _plain(output.get("info", {}))
            row.update({"metrics": _score(physics, raw_e, output, C_result, truth, score_options),
                        "CG": _cg_diagnostic(info), "solver_info": info})
        rows.append(_plain(row))
    weight_hashes = {r["metrics"]["raw_weight_sha256"] for r in rows if r["metrics"] is not None}
    if len(weight_hashes) > 1:
        raise RuntimeError("The oracle levels did not use the same observed-frame weights")
    return {"rows": rows, "outputs": outputs, "continuum_result": estimated,
            "o3": o3, "provided_continuum_consistency": _plain(supplied_consistency),
            "same_raw_and_weights_for_all_levels": len(weight_hashes) == 1,
            "oracle_role": "O0/O1/O2 diagnostic; O3 raw-only physical pathway"}
