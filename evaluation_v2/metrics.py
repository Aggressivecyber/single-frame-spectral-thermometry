"""Fixed-domain metrics, truth support independent of the detector, explicit failures."""
import numpy as np
from . import METRIC_SCHEMA

REQUIRED_SCREENS = ("numerical_execution_passed", "nominal_screen_passed",
                    "weak_signal_screen_passed", "null_screen_passed", "robustness_screen_passed")


def all_required_screens(screens, required=REQUIRED_SCREENS):
    if any(k not in screens or screens[k] is None for k in required):
        return {"status": "incomplete", "all_required_screens_passed": False,
                "required": list(required)}
    passed = all(screens[k] is True for k in required)
    return {"status": "passed" if passed else "failed", "required": list(required),
            "all_required_screens_passed": passed}


def support_mask(target, domain, fraction=.2):
    """Equality belongs to truth support; zero truth has no signal support."""
    t, d = np.asarray(target), np.asarray(domain, bool)
    peak = float(np.max(t[d])) if d.any() else 0.
    return d & (t >= fraction*peak) if peak > 0 else np.zeros_like(d)


def detection_score(template):
    """Fixed full-FOV percentile, available without test truth or test ROI."""
    a = np.asarray(template)
    return float(np.quantile(a, .99)) if np.isfinite(a).all() and a.size else None


def calibrate_detector(validation_null_predictions):
    if not validation_null_predictions:
        raise ValueError("An independent validation-null set is required")
    values = [np.asarray(a).ravel() for a in validation_null_predictions]
    if any(not np.isfinite(a).all() for a in values):
        raise ValueError("Validation detector predictions must be finite")
    scores = [detection_score(a) for a in validation_null_predictions]
    return {"pixel_threshold_e": float(np.quantile(np.concatenate(values), .99)),
            "frame_threshold_e": float(max(scores)), "pixel_quantile": .99,
            "frame_score": "full_FOV_signed_template_99th_percentile",
            "comparison": "prediction strictly greater than threshold",
            "validation_null_frames": len(values),
            "frame_threshold_rule": "maximum validation-null score, strict >",
            "statistical_limit": "Small validation set; no certified 1% frame-FPR control"}


def _ratio(a, b):
    return float(a/b) if b else None


def _rmse(error):
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))) if error.size else None


def _domain_metrics(target, pred, temp_truth, temp_pred, domain, accepted, pixel_threshold):
    d = np.asarray(domain, bool)
    t, p = np.asarray(target, np.float64), np.asarray(pred, np.float64)
    tt, tp = np.asarray(temp_truth, np.float64), np.asarray(temp_pred, np.float64)
    n = int(d.sum())
    finite_template = np.isfinite(p[d]).all()
    finite_temperature = np.isfinite(tp[d]).all()
    error = p[d]-t[d]
    energy = float(np.square(t[d]).sum())
    truth_sum = float(t[d].sum())
    ok = accepted & d
    coverage = _ratio(int(ok.sum()), n)
    out = {"domain_pixels": n, "coverage": coverage, "accepted_pixels": int(ok.sum()),
           "template_truth_energy": energy, "template_truth_sum": truth_sum,
           "template_SSE": float(np.square(error).sum()) if n and finite_template else None,
           "template_RMSE_e": _rmse(error) if finite_template else None,
           "template_signed_mean_e": float(p[d].mean()) if n and finite_template else None,
           "template_error_sum": float(error.sum()) if n and finite_template else None,
           "template_integral_bias_fraction": _ratio(float(error.sum()), truth_sum) if finite_template else None,
           "template_relative_L2": float(np.sqrt(np.square(error).sum()/energy)) if n and energy > 0 and finite_template else None,
           "temperature_SSE_K2": float(np.square(tp[d]-tt[d]).sum()) if n and finite_temperature else None,
           "temperature_observed_RMSE_K": _rmse(tp[d]-tt[d]) if finite_temperature else None,
           "conditional_temperature_RMSE_K": _rmse(tp[ok]-tt[ok]) if np.isfinite(tp[ok]).all() else None,
           "conditional_template_RMSE_e": _rmse(p[ok]-t[ok]) if np.isfinite(p[ok]).all() else None,
           "null_FP_fixed_domain": _ratio(int(np.sum(p[d] > pixel_threshold)), n) if energy == 0 and finite_template else None,
           "null_FP_accepted": _ratio(int(np.sum(p[ok] > pixel_threshold)), int(ok.sum())) if energy == 0 and np.isfinite(p[ok]).all() else None,
           "historical60_null_FP_fixed_domain": _ratio(int(np.sum(p[d] > 60.)), n) if energy == 0 and finite_template else None,
           "historical60_null_FP_accepted": _ratio(int(np.sum(p[ok] > 60.)), int(ok.sum())) if energy == 0 and np.isfinite(p[ok]).all() else None}
    if energy > 0 and finite_template:
        projection = float(np.sum(p[d]*t[d])/energy)
        orthogonal = p[d]-projection*t[d]
        out.update({"amplitude_projection": projection,
                    "orthogonal_RMSE_e": _rmse(orthogonal),
                    "orthogonal_relative_L2": float(np.sqrt(np.square(orthogonal).sum()/energy))})
    else:
        out.update({"amplitude_projection": None, "orthogonal_RMSE_e": None,
                    "orthogonal_relative_L2": None})
    for fraction in (.1, .2, .3):
        support = support_mask(t, d, fraction)
        positive = d & (p > pixel_threshold)
        prefix = "support"+str(round(100*fraction))
        out[prefix+"_pixels"] = int(support.sum())
        out[prefix+"_recall"] = _ratio(int(np.sum(positive & support)), int(support.sum())) if finite_template else None
        out[prefix+"_precision"] = _ratio(int(np.sum(positive & support)), int(positive.sum())) if finite_template else None
    support = support_mask(t, d)
    out["historical60_recall_on_truth_support20"] = _ratio(int(np.sum(d & support & (p > 60.))), int(support.sum())) if finite_template else None
    out["historical60_target_region_pixels"] = int(np.sum(d & (t >= 60.)))
    # The published valid indicator is Boolean, not a confidence ranking. Report
    # both the accepted operating point and full coverage; do not invent confidence.
    out["risk_coverage"] = [{"rule": "model_valid", "coverage": coverage,
                             "temperature_RMSE_K": out["conditional_temperature_RMSE_K"],
                             "template_RMSE_e": out["conditional_template_RMSE_e"]},
                            {"rule": "accept_all_finite_outputs", "coverage": 1. if n else None,
                             "temperature_RMSE_K": out["temperature_observed_RMSE_K"],
                             "template_RMSE_e": out["template_RMSE_e"]}]
    return out


def score_prediction(truth, prediction, detector, *, saturated=False):
    """Truth belongs exclusively to this scorer. No prediction-dependent domain."""
    base = {"metric_schema": METRIC_SCHEMA, "numerical_execution_passed": False}
    if saturated:
        return {**base, "status": "SATURATION_REJECTED", "metrics": None}
    if "evaluation_domain" not in truth:
        return {**base, "status": "MISSING_DOMAIN", "metrics": None}
    d = np.asarray(truth["evaluation_domain"], bool)
    if not d.any():
        return {**base, "status": "EMPTY_DOMAIN", "metrics": None}
    required = ("template_e", "temperature_K", "valid")
    if any(k not in prediction or np.asarray(prediction[k]).shape != d.shape for k in required):
        return {**base, "status": "MISSING_PREDICTION", "metrics": None}
    if not np.isfinite(np.asarray(truth["template_e"])[d]).all() or not np.isfinite(np.asarray(truth["temperature_observed_color_K"])[d]).all():
        return {**base, "status": "INVALID_TRUTH", "metrics": None}
    p, tp = np.asarray(prediction["template_e"]), np.asarray(prediction["temperature_K"])
    temperature_applicable = prediction.get("temperature_applicable", True)
    nonfinite = int(np.sum(d & (~np.isfinite(p) | (~np.isfinite(tp) if temperature_applicable else False))))
    bounds = int(np.sum(d & np.isfinite(tp) & ((tp < 1200) | (tp > 4500)))) if temperature_applicable else 0
    m = _domain_metrics(truth["template_e"], p, truth["temperature_observed_color_K"], tp,
                        d, np.asarray(prediction["valid"], bool), detector["pixel_threshold_e"])
    m.update({"nonfinite_pixels": nonfinite, "temperature_out_of_bounds_pixels": bounds,
              "prediction_missing_pixels": 0})
    finite = nonfinite == 0
    m["frame_detection_score_e"] = detection_score(p)
    m["frame_detected"] = bool(m["frame_detection_score_e"] > detector["frame_threshold_e"]) if m["frame_detection_score_e"] is not None else None
    m["blank_frame_false_positive"] = m["frame_detected"] if m["template_truth_energy"] == 0 else None
    ref = truth.get("temperature_reference_color_K")
    latent = truth.get("temperature_latent_K")
    obs = np.asarray(truth["temperature_observed_color_K"])
    m["temperature_reference_RMSE_K"] = _rmse(tp[d]-np.asarray(ref)[d]) if ref is not None and finite and temperature_applicable else None
    m["observed_reference_spectral_shift_RMSE_K"] = _rmse(obs[d]-np.asarray(ref)[d]) if ref is not None else None
    uniform = d & np.asarray(truth.get("uniform_temperature_domain", np.zeros_like(d)), bool)
    m["latent_uniform_pixels"] = int(uniform.sum())
    m["temperature_latent_uniform_RMSE_K"] = _rmse(tp[uniform]-np.asarray(latent)[uniform]) if latent is not None and finite and temperature_applicable else None
    m["temperature_latent_uniform_signed_bias_K"] = float((tp[uniform]-np.asarray(latent)[uniform]).mean()) if latent is not None and finite and temperature_applicable and uniform.any() else None
    m["temperature_applicable"] = temperature_applicable
    source = truth.get("source_image_domain")
    if source is not None:
        source = d & np.asarray(source, bool)
        outside = d & ~source
        m["source_image_domain"] = _domain_metrics(truth["template_e"], p, obs, tp, source,
                                                  np.asarray(prediction["valid"], bool), detector["pixel_threshold_e"])
        m["outside_source_template_RMSE_e"] = _rmse(p[outside]) if finite else None
        m["outside_source_positive_fraction"] = _ratio(int(np.sum(p[outside] > detector["pixel_threshold_e"])), int(outside.sum())) if finite else None
        if "source_image_domain_any" in truth:
            source_any = d & np.asarray(truth["source_image_domain_any"], bool)
            m["source_image_domain_any_sensitivity"] = _domain_metrics(truth["template_e"], p, obs, tp, source_any,
                                            np.asarray(prediction["valid"], bool), detector["pixel_threshold_e"])
    status = "NONFINITE_PREDICTION" if nonfinite else ("OUT_OF_BOUNDS" if bounds else "SUCCESS")
    return {**base, "status": status, "metrics": m,
            "numerical_execution_passed": status == "SUCCESS"}


def legacy_scores(truth, prediction):
    """V1 arithmetic in float64 for side-by-side audit; original files untouched."""
    t, p = np.asarray(truth["template_e"], np.float64), np.asarray(prediction["template_e"], np.float64)
    v = np.asarray(prediction["valid"], bool) & np.asarray(truth["evaluation_domain"], bool)
    temp = np.asarray(prediction["temperature_K"])-np.asarray(truth["temperature_observed_color_K"])
    filled = np.where(v, p, 0.)
    energy = np.square(t).sum()
    support = t >= 60.
    return {"metric_schema": "v1_recomputed_float64", "template_relative_L2": float(np.sqrt(np.square(filled-t).sum()/energy)) if energy else None,
            "temperature_RMSE_K_conditional": _rmse(temp[v]) if np.isfinite(temp[v]).all() else None,
            "recall_target_ge60_prediction_gt60": _ratio(int(np.sum((p > 60.) & v & support)), int(support.sum())),
            "null_FP_accepted60": _ratio(int(np.sum((p > 60.) & v)), int(v.sum())) if energy == 0 else None}


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "P90": float(np.quantile(a, .9)) if a.size else None,
            "max": float(a.max()) if a.size else None}


def aggregate_metrics(rows):
    ms = [r["metrics"] for r in rows if r.get("metrics") is not None]
    complete = len(ms) == len(rows) and all(r.get("status") == "SUCCESS" for r in rows)
    energy = sum(m["template_truth_energy"] for m in ms)
    template_sse = [m["template_SSE"] for m in ms]
    errsum = [m["template_error_sum"] for m in ms]
    tempsse = [m["temperature_SSE_K2"] for m in ms]
    total = sum(m["domain_pixels"] for m in ms)
    truthsum = sum(m["template_truth_sum"] for m in ms)
    def pooled(vals, den, root=False):
        if not vals or any(v is None for v in vals) or not den or not complete:
            return None
        value = sum(vals)/den
        return float(np.sqrt(value) if root else value)
    return {"metric_schema": METRIC_SCHEMA, "cases": len(rows), "complete": complete,
            "successful_cases": sum(r.get("status") == "SUCCESS" for r in rows),
            "failure_fraction": 1.-sum(r.get("status") == "SUCCESS" for r in rows)/len(rows) if rows else None,
            "pooled_template_L2": pooled(template_sse, energy, True),
            "pooled_signed_integral_bias": pooled(errsum, truthsum),
            "pooled_temperature_RMSE_K": pooled(tempsse, total, True),
            "per_case_L2": distribution(m["template_relative_L2"] for m in ms),
            "per_case_template_RMSE_e": distribution(m["template_RMSE_e"] for m in ms),
            "per_case_temperature_RMSE_K": distribution(m["temperature_observed_RMSE_K"] for m in ms),
            "per_case_absolute_integral_bias": distribution(abs(m["template_integral_bias_fraction"]) if m["template_integral_bias_fraction"] is not None else None for m in ms),
            "per_case_null_FP": distribution(m["null_FP_fixed_domain"] for m in ms),
            "per_case_historical60_null_FP": distribution(m["historical60_null_FP_fixed_domain"] for m in ms),
            "frame_detection_rate": float(np.mean([m["frame_detected"] for m in ms if m["frame_detected"] is not None])) if ms else None,
            "blank_frame_false_positive_rate": float(np.mean([m["blank_frame_false_positive"] for m in ms if m["blank_frame_false_positive"] is not None])) if any(m["blank_frame_false_positive"] is not None for m in ms) else None}
