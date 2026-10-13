"""External scoring only: fixed truth domains, signed targets, honest abstention.

Truth enters this function after raw-only inference. Unconditional field error
uses every pixel of the truth domain even if the estimator marks pixels invalid.
Coverage and conditional errors are separate; rejecting difficult pixels cannot
silently improve the primary full-domain RMSE.
"""
import math

import runtime
import numpy as np
import torch


METRIC_SCHEMA = "v5-signed-postblur-fixed-domain-001"


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field(value, shape, name, dtype=np.float64):
    result = _array(value)
    if result.shape == (1, *shape):
        result = result[0]
    if result.shape != shape:
        raise ValueError(name + " must have one H,W frame (or B=1,H,W)")
    return result.astype(dtype, copy=False)


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def _errors(prediction, target, mask):
    count = int(mask.sum())
    if count == 0 or not np.isfinite(prediction[mask]).all():
        return {"pixels": count, "SSE": None, "error_sum": None, "RMSE": None, "bias": None}
    error = prediction[mask]-target[mask]
    sse = float(np.square(error).sum())
    error_sum = float(error.sum())
    return {"pixels": count, "SSE": sse, "error_sum": error_sum,
            "RMSE": math.sqrt(sse/count), "bias": error_sum/count}


def _interval_metrics(truth, lower, upper, domain):
    finite = domain & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    count = int(finite.sum())
    covered = finite & (truth >= lower) & (truth <= upper)
    return {"available_pixels": count, "covered_pixels": int(covered.sum()),
            "available_fraction": _ratio(count, int(domain.sum())),
            "coverage_full_truth_domain": _ratio(int(covered.sum()), int(domain.sum())),
            "coverage_conditional_on_interval_available": _ratio(int(covered.sum()), count),
            "mean_width": float((upper[finite]-lower[finite]).mean()) if count else None}


def score_prediction(result, truth, physics, *, threshold_e=60, sigma_S=None, z_threshold=None):
    """Score one canonical prediction; return JSON-safe status and metrics.

    ``truth`` must include ``evaluation_domain``, ``template_postblur_e`` (or
    legacy ``template_e``), and ``temperature_color_K``. Latent temperature and
    observed/reference color comparisons are additional explicitly named fields.
    Raw residual metrics read observed ``raw_e``/``raw_residual_e`` from result,
    never substitute truth's clean ``expected_e`` for the observed measurement.
    Detection is two-sided for signed templates. The fixed frame score is the
    full-FOV 99th percentile of absolute amplitude (or SNR if z is selected).
    """
    base = {"metric_schema": METRIC_SCHEMA, "numerical_execution_passed": False}
    if not isinstance(result, dict) or not isinstance(truth, dict):
        return {**base, "status": "INVALID_INPUT_MAPPING", "metrics": None}
    if "evaluation_domain" not in truth:
        return {**base, "status": "MISSING_TRUTH_DOMAIN", "metrics": None}
    domain = _array(truth["evaluation_domain"])
    if domain.ndim == 3 and domain.shape[0] == 1:
        domain = domain[0]
    if domain.ndim != 2:
        return {**base, "status": "INVALID_TRUTH_DOMAIN_SHAPE", "metrics": None}
    domain = domain.astype(bool)
    shape = domain.shape
    if not domain.any():
        return {**base, "status": "EMPTY_TRUTH_DOMAIN", "metrics": None}
    if not math.isfinite(float(threshold_e)) or threshold_e < 0:
        raise ValueError("threshold_e must be a fixed finite nonnegative threshold")
    if z_threshold is not None and (not math.isfinite(float(z_threshold)) or z_threshold <= 0):
        raise ValueError("z_threshold must be a fixed finite positive threshold")
    required = ("template_postblur_e", "temperature_color_K", "valid", "expected_e")
    if any(key not in result for key in required):
        return {**base, "status": "MISSING_PREDICTION", "metrics": None}
    try:
        target = _field(truth.get("template_postblur_e", truth.get("template_e")), shape, "truth template")
        target_temperature = _field(truth["temperature_color_K"], shape, "truth postblur color temperature")
        template = _field(result["template_postblur_e"], shape, "predicted template")
        temperature = _field(result["temperature_color_K"], shape, "predicted color temperature")
        valid = _field(result["valid"], shape, "valid", bool)
        expected = _field(result["expected_e"], shape, "expected_e")
    except (ValueError, KeyError, TypeError) as error:
        return {**base, "status": "INVALID_PREDICTION_OR_TRUTH_SHAPE", "error": str(error), "metrics": None}
    if not (np.isfinite(target[domain]).all() and np.isfinite(target_temperature[domain]).all()):
        return {**base, "status": "NONFINITE_TRUTH", "metrics": None}
    count = int(domain.sum())
    accepted = domain & valid
    s = _errors(template, target, domain)
    t = _errors(temperature, target_temperature, domain)
    sc = _errors(template, target, accepted)
    tc = _errors(temperature, target_temperature, accepted)
    energy = float(np.square(target[domain]).sum())
    finite_template = np.isfinite(template[domain]).all()
    finite_temperature = np.isfinite(temperature[domain]).all()
    nonfinite = (~np.isfinite(template) | ~np.isfinite(temperature) | ~np.isfinite(expected))
    lower_T, upper_T = physics.cfg["temperature_range_K"]
    out_of_bounds = domain & np.isfinite(temperature) & ((temperature < lower_T) | (temperature > upper_T))
    metrics = {
        "domain_pixels": count, "accepted_pixels": int(accepted.sum()),
        "valid_fraction": _ratio(int(accepted.sum()), count),
        "full_FOV_valid_fraction": float(valid.mean()),
        "nonfinite_domain_pixels": int((domain & nonfinite).sum()),
        "nonfinite_output_pixels": int(nonfinite.sum()),
        "temperature_out_of_bounds_pixels": int(out_of_bounds.sum()),
        "template_SSE": s["SSE"], "template_truth_energy": energy,
        "template_error_sum": s["error_sum"], "template_RMSE_e": s["RMSE"],
        "template_bias_e": s["bias"],
        "template_relative_L2": math.sqrt(s["SSE"]/energy) if s["SSE"] is not None and energy > 0 else None,
        "temperature_SSE_K2": t["SSE"], "temperature_error_sum_K": t["error_sum"],
        "temperature_RMSE_K": t["RMSE"], "temperature_bias_K": t["bias"],
        "temperature_truth_role": "postblur continuum color temperature",
        "conditional_template_RMSE_e": sc["RMSE"], "conditional_temperature_RMSE_K": tc["RMSE"],
        "threshold_e": float(threshold_e), "z_threshold": z_threshold,
        "support_fraction_of_absolute_peak": 0.2,
    }
    if energy > 0 and finite_template:
        projection_numerator = float((template[domain]*target[domain]).sum())
        projection = projection_numerator/energy
        orthogonal = template[domain]-projection*target[domain]
        metrics.update(signed_projection=projection, signed_projection_numerator=projection_numerator,
                       orthogonal_template_RMSE_e=math.sqrt(float(np.square(orthogonal).mean())))
    else:
        metrics.update(signed_projection=None, signed_projection_numerator=None,
                       orthogonal_template_RMSE_e=None)
    peak = float(np.abs(target[domain]).max())
    support = domain & (np.abs(target) >= 0.2*peak) if peak > 0 else np.zeros(shape, dtype=bool)
    positive, negative = support & (target > 0), support & (target < 0)
    sigma = None
    if sigma_S is None:
        sigma_S = result.get("sigma_S")
    if sigma_S is not None:
        sigma = _field(sigma_S, shape, "sigma_S")
    if z_threshold is not None:
        if sigma is None:
            raise ValueError("A frozen z_threshold requires measured/estimated sigma_S")
        sigma_valid = np.isfinite(sigma) & (sigma > 0)
        snr = np.full(shape, np.nan, dtype=np.float64)
        np.divide(np.abs(template), sigma, out=snr, where=sigma_valid)
        detected = sigma_valid & (snr > z_threshold)
        frame_values, frame_threshold = snr, float(z_threshold)
        frame_score_kind = "full_FOV_99th_percentile_absolute_SNR"
        metrics["sigma_valid_fraction"] = float((domain & sigma_valid).sum()/count)
    else:
        detected = np.abs(template) > threshold_e
        frame_values, frame_threshold = np.abs(template), float(threshold_e)
        frame_score_kind = "full_FOV_99th_percentile_absolute_template_e"
    correct_sign = (template*target) > 0
    support_count = int(support.sum())
    sign_true = int((support & correct_sign).sum()) if finite_template else None
    detected_signed = int((support & correct_sign & detected & valid).sum()) if finite_template else None
    metrics.update(support_pixels=support_count, sign_correct_pixels=sign_true,
                   signed_detected_pixels=detected_signed,
                   sign_recall=_ratio(sign_true, support_count) if sign_true is not None else None,
                   signed_recall=_ratio(detected_signed, support_count) if detected_signed is not None else None,
                   signed_recall_unconditional=_ratio(int((support & correct_sign & detected).sum()), support_count) if finite_template else None,
                   signed_recall_rule="correct sign and threshold crossed and valid; denominator is fixed truth support")
    for name, mask in (("positive", positive), ("negative", negative)):
        denominator = int(mask.sum())
        numerator = int((mask & correct_sign & detected & valid).sum()) if finite_template else None
        metrics[name+"_support_pixels"] = denominator
        metrics[name+"_detected_pixels"] = numerator
        metrics[name+"_recall"] = _ratio(numerator, denominator) if numerator is not None else None
    frame_finite = np.isfinite(frame_values).all()
    frame_score = float(np.quantile(frame_values, 0.99)) if frame_finite else None
    frame_detected = bool(frame_score > frame_threshold) if frame_score is not None else None
    null_pixels = int((domain & detected).sum()) if energy == 0 and finite_template else None
    metrics.update(frame_detection_score=frame_score, frame_detection_score_kind=frame_score_kind,
                   frame_threshold=frame_threshold, frame_detected=frame_detected,
                   null_pixel_FP_count=null_pixels,
                   null_pixel_FP=_ratio(null_pixels, count) if null_pixels is not None else None,
                   null_frame_FP=frame_detected if energy == 0 else None,
                   null_FOV_FPR_control_claimed=False)
    raw = None
    if "raw_e" in result:
        raw = _field(result["raw_e"], shape, "observed raw_e")
    elif "raw_dn" in result:
        raw = (_field(result["raw_dn"], shape, "observed raw_dn")-physics.cfg["black_level_DN"])*physics.adc_step
    if raw is not None:
        residual = expected-raw
        raw_finite = np.isfinite(raw).all() and np.isfinite(residual).all()
        weighted_sse = float((np.square(residual)/(np.maximum(raw, 0)+physics.extra_variance)).sum()) if raw_finite else None
        metrics.update(raw_pixels=raw.size, raw_weighted_SSE=weighted_sse,
                       raw_weighted_RMSE=math.sqrt(weighted_sse/raw.size) if weighted_sse is not None else None,
                       raw_RMSE_e=float(np.sqrt(np.square(residual).mean())) if raw_finite else None,
                       raw_residual_target="same observed raw, fixed variance from observed raw")
    else:
        metrics.update(raw_pixels=0, raw_weighted_SSE=None, raw_weighted_RMSE=None, raw_RMSE_e=None,
                       raw_residual_target="unavailable: observed raw is not present; clean truth expected_e not substituted")
    for key, label in (("temperature_observed_color_K", "observed_color"),
                       ("temperature_reference_color_K", "reference_color"),
                       ("latent_temperature_K", "latent_truth"),
                       ("temperature_latent_K", "latent_truth")):
        if key in truth and ("temperature_"+label+"_RMSE_K") not in metrics:
            reference = _field(truth[key], shape, key)
            mask = domain.copy()
            if label == "latent_truth" and "uniform_temperature_domain" in truth:
                mask &= _field(truth["uniform_temperature_domain"], shape, "uniform temperature domain", bool)
            error = _errors(temperature, reference, mask)
            metrics["temperature_"+label+"_RMSE_K"] = error["RMSE"]
            metrics["temperature_"+label+"_bias_K"] = error["bias"]
            metrics["temperature_"+label+"_comparison_pixels"] = error["pixels"]
    for prefix, true_value, low_key, high_key in (
        ("signed_interval", target, "signed_ci_lower_e", "signed_ci_upper_e"),
        ("temperature_interval", target_temperature, "temperature_ci_lower_K", "temperature_ci_upper_K")):
        if low_key in result and high_key in result:
            lower, upper = _field(result[low_key], shape, low_key), _field(result[high_key], shape, high_key)
            metrics[prefix] = _interval_metrics(true_value, lower, upper, domain)
    finite = metrics["nonfinite_output_pixels"] == 0
    status = "NONFINITE_PREDICTION" if not finite else ("TEMPERATURE_OUT_OF_BOUNDS" if out_of_bounds.any() else "SUCCESS")
    if result.get("info", {}).get("status") == "ABSTAINED" or result.get("status") == "abstained":
        status = "ABSTAINED"
    return {**base, "status": status, "metrics": metrics,
            "numerical_execution_passed": status == "SUCCESS"}
