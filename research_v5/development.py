"""One finite physics selection, independent null threshold and uncertainty audit.

Root calls run_development after registration. This module never opens either
sealed evaluation partition. Inference receives raw DN plus fixed external
calibration/options; scene descriptors and truth are external scoring inputs.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

from . import common

common.configure_runtime()

import numpy as np
import torch

from .statistics import (aggregate_metrics, filter_records, mother_bootstrap,
                         mother_metric, paired_statistics, statistics_plan)
from .continuum_solver import estimate_continuum
from .signed_linear_solver import (SignedLinearOperator, create_linear_system,
                                   solve_signed_linear, DEFAULT_OPTIONS as LINEAR_DEFAULTS)
from .reconstruct import reconstruct
from .metrics import score_prediction
from .uncertainty import monte_carlo_uncertainty, hessian_probe_diagnostics


DIRECTORY = common.REPORTS / "development"
ALPHA_KEYS = {0.: "a0", .01: "a001", .1: "a01", 1.: "a1"}
PROXY_SPEC = {
    "kind": "raw_derived_uniform_W_postblur_noise_proxy",
    "raw_weight_rule": "W=1/(clamp(observed raw_e,0)+read_noise_e^2+adc_step^2/12)",
    "formula": "sigma_proxy^2=mean_f[H(f)^2*F(f)/(F(f)+lambda_g*L(f)+lambda0)^2]; F=mean(W)*stationary_A_S_normal_symbol",
    "regularization_scale": "median of the selected actual diag(A_S^T W A_S); distinct from the proxy stationary F(f)",
    "spatial_form": "one positive scalar broadcast over the full FOV per observed raw",
    "target": "postblur signed template, electrons",
    "limitations": ["uniform-W diagonal propagation approximation; not full spatial covariance",
                    "continuum uncertainty and jointly estimated beta are omitted",
                    "not an MC sigma, measurement CRLB, or guaranteed confidence interval",
                    "independent nominal null calibration supplies the operational threshold",
                    "calibration/model mismatch can invalidate nominal null control"],
    "raw_only": True, "truth_used": False, "floor_rule": "none; nonpositive or nonfinite proxy invalidates detection",
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _plain(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _plain(value.item())
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _source_receipt():
    return {common.relative(common.CODE/name): common.sha(common.CODE/name)
            for name in ("development.py", "statistics.py", "candidate_review.py")}


def _save_arrays(path, values):
    """Durable new arrays; retain NaNs marking unavailable uncertainty."""
    path = common.ed_path(path)
    if not path.is_relative_to(common.DATA.parent.resolve()):
        raise ValueError("Prediction arrays must stay in new v5 outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    arrays = {name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
              else np.asarray(value) for name, value in values.items()}
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as reopened:
        assert set(reopened.files) == set(arrays)
        assert all(np.array_equal(reopened[k], v, equal_nan=True) for k, v in arrays.items())
    return {"path": common.relative(path), "sha256": common.sha(path)}


@torch.no_grad()
def raw_noise_proxy(physics, raw_dn, core_selection):
    """Raw-derived stationary postblur noise scale for empirical null ranking.

    This deliberately does not accept a prediction/truth/identity. It uses the
    selected linear regularizers and observed raw noise calibration only.
    """
    if not isinstance(core_selection, dict) or set(core_selection)-{
            "continuum_options", "linear_options", "device", "dtype"}:
        raise ValueError("Expected raw-only core selection")
    options = core_selection.get("linear_options", {})
    if options.get("weights") is not None:
        raise ValueError("Deployment proxy cannot consume diagnostic weights")
    raw_e = common.raw_electrons(physics, raw_dn)
    linear = SignedLinearOperator(physics)
    fields, details = [], []
    for frame in raw_e:
        weights = (frame.clamp_min(0)+physics.extra_variance).reciprocal()
        symbol = weights.mean()*linear.stationary_normal_symbol
        diagonal, diagonal_info = linear.template_diagonal(
            weights, options.get("diagonal_mode", LINEAR_DEFAULTS["diagonal_mode"]))
        scale = diagonal.median()
        alpha = float(options.get("alpha_g", 0.))
        if alpha not in ALPHA_KEYS:
            raise ValueError("Unregistered alpha for uncertainty proxy")
        ridge = float(options.get("lambda0_factor", 1e-6))*scale
        gradient = alpha*scale
        denominator = symbol+gradient*linear.laplacian_symbol+ridge
        variance = (physics.op.object_otf.square()*symbol/denominator.square()).mean()
        if not bool(torch.isfinite(variance)) or float(variance) <= 0:
            raise ValueError("Nonpositive/nonfinite raw noise proxy; detection unavailable")
        sigma = variance.sqrt()
        fields.append(torch.ones_like(frame)*sigma)
        details.append({"sigma_proxy_e": float(sigma), "raw_weight_mean": float(weights.mean()),
                        "raw_variance_min_e2": float((frame.clamp_min(0)+physics.extra_variance).min()),
                        "raw_variance_max_e2": float((frame.clamp_min(0)+physics.extra_variance).max()),
                        "lambda_g": float(gradient), "lambda0": float(ridge),
                        "diagonal_scale": float(scale), "regularization_diagonal_method": diagonal_info})
    return {"sigma_S": torch.stack(fields),
            "summary": {"schema": "v5-raw-noise-proxy-001", **copy.deepcopy(PROXY_SPEC),
                        "frames": details, "alpha_g": float(options.get("alpha_g", 0.))}}


def detection_diagnostics(output, sigma_S, z_threshold=None):
    """Two-sided full-FOV maximum score; invalid pixels never disappear."""
    template = torch.as_tensor(output["template_postblur_e"])
    sigma = torch.as_tensor(sigma_S, dtype=template.dtype, device=template.device)
    if sigma.shape != template.shape or template.ndim != 3 or len(template) != 1:
        raise ValueError("Detection requires matching one-frame B,H,W template/proxy")
    finite = torch.isfinite(template) & torch.isfinite(sigma) & (sigma > 0)
    valid = torch.as_tensor(output["valid"], device=template.device).bool()
    complete = bool(finite.all())
    score = template.abs()/sigma
    if z_threshold is not None and (not _finite(z_threshold) or z_threshold <= 0):
        raise ValueError("A frozen positive finite null threshold is required")
    crossed = finite & (score > z_threshold) if z_threshold is not None else None
    return {"complete": complete, "frame_detection_score": float(score.max()) if complete else None,
            "frame_detection_score_kind": "max_abs_template_over_raw_proxy_full_FOV",
            "frame_detected": bool(crossed.any()) if complete and crossed is not None else None,
            "frame_threshold": float(z_threshold) if z_threshold is not None else None,
            "pixels": template.numel(), "proxy_valid_fraction": float(finite.double().mean()),
            "valid_fraction": float(valid.double().mean()),
            "detected_pixels": int(crossed.sum()) if complete and crossed is not None else None,
            "detected_pixel_fraction": float(crossed.double().mean()) if complete and crossed is not None else None,
            "abstained_pixel_fraction": float((~valid | ~finite).double().mean()),
            "undetectable_pixel_fraction": float((finite & ~crossed).double().mean()) if crossed is not None and complete else None,
            "median_detection_limit_e": float((z_threshold*sigma).median()) if z_threshold is not None and complete else None,
            "threshold_controls": "maximum raw-only standardized postblur amplitude; independent simulated null calibration"}


def calibrate_null_threshold(records, *, quantile=.95):
    if quantile != .95:
        raise ValueError("Only the preregistered .95 higher-quantile threshold is allowed")
    rows = list(records)
    scores = [r.get("detection", {}).get("frame_detection_score") for r in rows]
    complete = bool(rows) and all(_finite(v) for v in scores) and all(
        r.get("numerical_execution_passed") is True and r.get("detection", {}).get("complete") is True
        for r in rows)
    threshold = float(np.quantile(scores, quantile, method="higher")) if complete else None
    # Zero is a legitimate all-zero calibration maximum but cannot define a
    # standardized nonzero threshold. Preserve this degeneracy as failure.
    complete = complete and threshold > 0
    return {"complete": complete, "z_threshold": threshold, "quantile": quantile,
            "method": "higher", "calibration_frame_score": "max_abs_template_over_raw_proxy_full_FOV",
            "calibration_frames": len(rows), "calibration_mothers": len({r["mother_id"] for r in rows}),
            "calibration_mother_ids": sorted({r["mother_id"] for r in rows}),
            "calibration_scores": [{"case_id": r["case_id"], "mother_id": r["mother_id"], "score": v}
                                   for r, v in zip(rows, scores)],
            "retuned_on_evaluation": False, "operational_FOV_rule": "strict score > threshold",
            "finite_sample_or_instrument_guarantee": False}


def _null_summary(rows, threshold):
    values, mother = [], {}
    for row in rows:
        detection = row.get("detection", {})
        value = detection.get("frame_detected")
        if value is None and _finite(detection.get("frame_detection_score")) and _finite(threshold):
            value = detection["frame_detection_score"] > threshold
        values.append(value)
        mother.setdefault(row["mother_id"], []).append(value)
    complete = bool(values) and all(v is not None for v in values)
    pixels = sum(r.get("detection", {}).get("pixels", 0) for r in rows)
    counts = [r.get("detection", {}).get("detected_pixels") for r in rows]
    mother_values = {mid: sum(vals)/len(vals) if all(v is not None for v in vals) else None
                     for mid, vals in mother.items()}
    return {"complete": complete, "frames": len(rows), "mothers": len(mother),
            "frame_FP": sum(values)/len(values) if complete else None,
            "frame_FP_count": sum(values) if complete else None,
            "pixel_FP": sum(counts)/pixels if pixels and all(v is not None for v in counts) else None,
            "frame_FP_mother_bootstrap": mother_bootstrap(mother_values),
            "invalid_detection_frames": sum(v is None for v in values),
            "all_numerical_predictions_passed": all(r.get("numerical_execution_passed") for r in rows),
            "interpretation": "disjoint nominal synthetic null evaluation; not a guaranteed future FOV error rate"}


def _verified_family(physics, plans, name, protocol):
    from .family_data import generate_family
    role = {"development": "development_selection"}.get(name, name)
    folder = common.DATA/name
    manifest_path = folder/"data_manifest.json"
    plan = plans["plans"][name]
    if not manifest_path.exists():
        generate_family(physics, plan, folder, options={"fixture_only": False,
            "authorization": {"phase": "development-compare", "partition_role": role,
                              "plan_sha256": common.canonical_hash(plan),
                              "protocol_sha256": common.sha(common.CODE/"protocol.json")}})
    manifest = common.read_json(manifest_path)
    assert not manifest["fixture_only"] and manifest["partition_role"] == role
    assert manifest["plan_sha256"] == protocol["family_plan_hashes"][name] == common.canonical_hash(plan)
    assert manifest["protocol_sha256"] == common.sha(common.CODE/"protocol.json")
    assert manifest["physics_config_sha256"] == protocol["physics_config_sha256"]
    assert manifest["generator_source_sha256"] == common.sha(common.CODE/"family_data.py")
    assert len(manifest["rows"]) == len(plan)
    planned = {r["case_id"]: r for r in plan}
    assert len(planned) == len(plan) and set(planned) == {r["case_id"] for r in manifest["rows"]}
    for row in manifest["rows"]:
        descriptor = planned[row["case_id"]]
        assert all(row[k] == v for k, v in descriptor.items() if k != "mother")
    return folder, manifest, {"path": common.relative(manifest_path), "sha256": common.sha(manifest_path),
                              "plan_sha256": manifest["plan_sha256"], "partition_role": role}


def _inputs(folder, row, physics):
    if not row.get("raw_available"):
        raise ValueError("Registered raw unavailable: "+row.get("raw_generation_status", "unknown"))
    raw_path = (folder/row["raw_path"]).resolve()
    label_path = (folder/row["label_path"]).resolve()
    if not raw_path.is_relative_to(folder.resolve()) or not label_path.is_relative_to(folder.resolve()):
        raise ValueError("Manifest path escapes its family")
    assert common.sha(raw_path) == row["raw_file_sha256"]
    assert common.sha(label_path) == row["label_sha256"]
    raw = common.arrays(raw_path, raw_only=True)["raw_dn"]
    raw = torch.as_tensor(raw, dtype=physics.M.dtype, device=physics.M.device)
    if raw.ndim == 2:
        raw = raw[None]
    return raw, common.arrays(label_path)


def _metadata(row, method, *, alpha=None):
    coefficient = float(row["coefficient_e"])
    out = {k: copy.deepcopy(row.get(k)) for k in ("case_id", "mother_id", "partition_role", "condition",
               "group", "coefficient_e", "input_kind", "geometry", "template_geometry", "noise_index", "stress")}
    out.update(method=method, sign="positive" if coefficient > 0 else "negative" if coefficient < 0 else "blank",
               raw_sha256=row.get("raw_file_sha256"), label_sha256=row.get("label_sha256"))
    if alpha is not None:
        out["alpha_g"] = float(alpha)
    return out


def _failure_record(row, method, error, *, alpha=None):
    return {**_metadata(row, method, alpha=alpha), "status": "FAILED_OR_ABSTAINED",
            "numerical_execution_passed": False, "metrics": None,
            "error": str(error), "error_type": type(error).__name__, "diagnostics": {}}


def _physical_prediction(continuum, linear, raw_e):
    frame = linear["info"]["frames"][0]
    solved = bool(frame["accepted"] and frame["CG"]["converged"] and frame["expected_e_nonnegative"])
    result = {**{k: v for k, v in continuum.items() if isinstance(v, torch.Tensor)},
              **{k: v for k, v in linear.items() if isinstance(v, torch.Tensor)},
              "valid": continuum["valid"] & solved, "raw_e": raw_e,
              "info": {"continuum": continuum["info"], "linear": linear["info"],
                       "continuum_frozen": True, "truth_or_identity_access": False}}
    return result


def _diagnostics(output, seconds):
    info = output.get("info", {})
    linear = info.get("linear", {})
    if linear.get("frames"):
        frame = linear["frames"][0]
        return _plain({"seconds": seconds, "CG": frame["CG"],
                       "gradient_energy_S_e2": frame["objective_terms_after"]["gradient_energy_S_e2"],
                       "lambda_g": frame["lambda_g"], "lambda0": frame["lambda0"],
                       "lambda_beta": frame["lambda_beta"], "accepted": frame["accepted"],
                       "expected_e_nonnegative": frame["expected_e_nonnegative"],
                       "diagonal_S": frame["diagonal_S"],
                       "continuum_frozen": True, "continuum_once_before_four_alphas": True})
    effective_seconds = seconds + info.get("shared_inverse_seconds", 0.)
    return _plain({"seconds": effective_seconds, "CG": None,
                   "CG_unavailable_reason": "baseline adapter has no true normal-residual certificate",
                   "baseline_info": info,
                   "seconds_scope": "entire route, including v3 inverse for v4.4"})


def _record(row, method, output, truth, physics, seconds, *, alpha=None, sigma=None, threshold=None):
    score_truth = dict(truth)
    if "temperature_color_K_reference" in truth:
        score_truth["temperature_reference_color_K"] = truth["temperature_color_K_reference"]
    scored = score_prediction(output, score_truth, physics, sigma_S=sigma, z_threshold=threshold)
    rec = {**_metadata(row, method, alpha=alpha), **_plain(scored),
           "diagnostics": _diagnostics(output, seconds)}
    frames = output.get("info", {}).get("linear", {}).get("frames", [])
    if frames and not all(f.get("accepted") is True and f.get("CG", {}).get("converged") is True
                          and f.get("expected_e_nonnegative") is True for f in frames):
        rec.update(status="SIGNED_SOLVE_FAILED_OR_ABSTAINED", numerical_execution_passed=False)
    if rec.get("metrics") is not None and "temperature_color_K_nominal" in truth:
        nominal = score_prediction(output, {**score_truth, "temperature_color_K": truth["temperature_color_K_nominal"]}, physics)
        nominal_metrics = nominal.get("metrics") or {}
        rec["metrics"].update(temperature_nominal_color_RMSE_K=nominal_metrics.get("temperature_RMSE_K"),
                              temperature_nominal_color_bias_K=nominal_metrics.get("temperature_bias_K"),
                              color_primary_reference="nominal continuum color" if not row.get("stress") else
                              "measurement-equivalent reference color under declared calibration stress; nominal target reported separately")
    if sigma is not None:
        rec["detection"] = detection_diagnostics(output, sigma, threshold)
        if rec.get("metrics") is not None:
            rec["metrics"].update({k: rec["detection"][k] for k in
                                   ("frame_detection_score", "frame_detection_score_kind", "frame_detected", "frame_threshold")})
            if row["coefficient_e"] == 0 and threshold is not None:
                rec["metrics"].update(null_frame_FP=rec["detection"]["frame_detected"],
                                      null_pixel_FP=rec["detection"]["detected_pixel_fraction"],
                                      null_pixel_FP_count=rec["detection"]["detected_pixels"])
    return rec


def load_cached_prediction(record, physics):
    """Reopen hash-bound same-raw output, outside the inference boundary."""
    receipt = record["prediction_cache"]
    path = common.ROOT/receipt["path"]
    assert common.sha(path) == receipt["sha256"]
    values = common.arrays(path)
    prefix = receipt.get("alpha_key")
    if prefix is None:
        result = {k: torch.as_tensor(v, device=physics.M.device,
                   dtype=torch.bool if k == "valid" else physics.M.dtype) for k, v in values.items()}
    else:
        result = {k: torch.as_tensor(values[k], device=physics.M.device,
                   dtype=physics.M.dtype) for k in
                   ("continuum_latent", "continuum_postblur", "temperature_color_K")}
        for key in ("template_latent_e", "template_postblur_e", "beta", "expected_e", "valid"):
            result[key] = torch.as_tensor(values[prefix+"__"+key], device=physics.M.device,
                            dtype=torch.bool if key == "valid" else physics.M.dtype)
    cg = record.get("diagnostics", {}).get("CG")
    result["info"] = {"cache_of_same_raw": True, "raw_sha256": record["raw_sha256"],
                       "same_raw_CG_diagnostics": cg}
    return result


def _selection(records, grid):
    candidates = []
    for alpha in grid:
        normal = filter_records(records, method="v5_physics", group="normal", input_kind="noise", alpha_g=alpha)
        aggregate = aggregate_metrics(normal)
        parents = mother_metric(normal)
        objective = (sum(parents.values())/len(parents)
                     if parents and all(_finite(v) for v in parents.values()) else None)
        feasible = bool(aggregate["complete"] and aggregate["CG"]["all_converged"]
                        and aggregate["temperature_RMSE_K"] <= 20.)
        candidates.append({"alpha_g": alpha, "paired_parent_normal_MSE_e2": objective,
                           "feasible": feasible, "normal_pooled": aggregate})
    feasible = [r for r in candidates if r["feasible"]]
    finite = [r for r in candidates if _finite(r["paired_parent_normal_MSE_e2"])]
    selected = min(feasible or finite or candidates,
                   key=lambda r: (r["paired_parent_normal_MSE_e2"]
                                  if _finite(r["paired_parent_normal_MSE_e2"]) else float("inf"), r["alpha_g"]))
    traditional = []
    for name in ("B1_tikhonov", "v3_physics"):
        rows = filter_records(records, method=name, group="normal", input_kind="noise")
        parents = mother_metric(rows)
        mse = sum(parents.values())/len(parents) if parents and all(_finite(v) for v in parents.values()) else None
        traditional.append({"method": name, "paired_parent_normal_MSE_e2": mse,
                            "pooled": aggregate_metrics(rows)})
    best = min(traditional, key=lambda r: (r["paired_parent_normal_MSE_e2"]
                   if _finite(r["paired_parent_normal_MSE_e2"]) else float("inf"), r["method"]))
    return {"selected_alpha_g": selected["alpha_g"], "selection_feasible": bool(feasible),
            "selection_rule": "normal +/- noisy development mother mean MSE, converged CG and color<=20K; ties use smallest alpha",
            "diagnostic_fallback_if_infeasible": not bool(feasible), "alpha_comparison": candidates,
            "traditional_reference": best["method"], "traditional_comparison": traditional,
            "truth_access_role": "external development-only scoring; no truth in inference",
            "null_or_final_used_to_select_alpha": False}


def _mc_cases(plan):
    """Metadata-only 12-case fixed subset: six geometry pairs x both signs."""
    from .data import TEMPERATURE_GEOMETRIES, TEMPLATE_GEOMETRIES
    chosen = []
    for index, geometry in enumerate(TEMPERATURE_GEOMETRIES):
        matching = [r for r in plan if r["geometry"] == geometry
                    and r["template_geometry"] == TEMPLATE_GEOMETRIES[index]
                    and r["input_kind"] == "noise" and r["group"] == "normal"]
        for coefficient in (180., -180.):
            row = sorted([r for r in matching if r["coefficient_e"] == coefficient], key=lambda r: r["case_id"])[0]
            chosen.append(row["case_id"])
    assert len(chosen) == len(set(chosen)) == 12
    return chosen


def _run_grid(physics, folder, manifest, protocol, baselines, budget):
    records = []
    for row in manifest["rows"]:
        budget.check()
        raw_dn, truth = None, None
        try:
            raw_dn, truth = _inputs(folder, row, physics)
            raw_e = common.raw_electrons(physics, raw_dn)
            common.synchronize(physics.M.device)
            started = time.perf_counter()
            continuum = estimate_continuum(physics, raw_e, protocol["continuum_options"])
            common.synchronize(physics.M.device)
            continuum_seconds = time.perf_counter()-started
            cache = {k: continuum[k] for k in ("continuum_latent", "continuum_postblur", "temperature_color_K")}
            cache["continuum_valid"] = continuum["valid"]
            case_records = []
            for alpha in protocol["alpha_g_development_grid"]:
                options = {**protocol["linear_defaults"], "alpha_g": alpha}
                started = time.perf_counter()
                linear = solve_signed_linear(physics, raw_e, continuum["continuum_latent"].detach().clone(), options)
                common.synchronize(physics.M.device)
                output = _physical_prediction(continuum, linear, raw_e)
                elapsed = continuum_seconds+time.perf_counter()-started
                prefix = ALPHA_KEYS[float(alpha)]
                for key in ("template_latent_e", "template_postblur_e", "beta", "expected_e", "valid"):
                    cache[prefix+"__"+key] = output[key]
                record = _record(row, "v5_physics", output, truth, physics, elapsed, alpha=alpha)
                case_records.append(record)
            receipt = _save_arrays(common.DATA/"development_grid_predictions"/(row["case_id"]+".npz"), cache)
            for record in case_records:
                record["prediction_cache"] = {**receipt, "alpha_key": ALPHA_KEYS[record["alpha_g"]]}
            records.extend(case_records)
        except Exception as error:
            # Do not omit difficult scenes. If part of a row already completed,
            # replace only its missing methods/configurations with failure rows.
            present = {(r["method"], r.get("alpha_g")) for r in records if r["case_id"] == row["case_id"]}
            for alpha in protocol["alpha_g_development_grid"]:
                if ("v5_physics", float(alpha)) not in present:
                    records.append(_failure_record(row, "v5_physics", error, alpha=alpha))
        # A physical inversion failure is no reason to skip an independent
        # traditional/neural reference on a valid observed raw frame.
        if raw_dn is not None:
            try:
                for name, output in baselines.evaluate(raw_dn).items():
                    records.append(_record(row, name, output, truth, physics, output["info"].get("seconds", 0.)))
            except Exception as error:
                for name in protocol["baselines"]:
                    if not any(r["case_id"] == row["case_id"] and r["method"] == name for r in records):
                        records.append(_failure_record(row, name, error))
        else:
            for name in protocol["baselines"]:
                records.append(_failure_record(row, name, ValueError("raw unavailable or input hashes invalid")))
        for record in [r for r in records if r["case_id"] == row["case_id"]]:
            common.append_jsonl(DIRECTORY/"records.jsonl", _plain(record))
    return records


def _run_selected_family(physics, folder, manifest, core, name, budget, threshold=None, baselines=None):
    records = []
    for row in manifest["rows"]:
        budget.check()
        raw_dn, truth = None, None
        try:
            raw_dn, truth = _inputs(folder, row, physics)
            started = time.perf_counter()
            output = reconstruct(raw_dn, {"physics_config": physics.cfg}, core)
            output["raw_e"] = common.raw_electrons(physics, raw_dn)
            common.synchronize(physics.M.device)
            seconds = time.perf_counter()-started
            proxy = raw_noise_proxy(physics, raw_dn, core)
            record = _record(row, "v5_physics", output, truth, physics, seconds,
                             alpha=core["linear_options"]["alpha_g"], sigma=proxy["sigma_S"], threshold=threshold)
            record["noise_proxy"] = proxy["summary"]
            if threshold is not None:
                keep = {key: value for key, value in output.items() if isinstance(value, torch.Tensor)}
                record["prediction_cache"] = _save_arrays(
                    common.DATA/(name+"_physics_predictions")/(row["case_id"]+".npz"), keep)
            records.append(record)
        except Exception as error:
            if not any(r["case_id"] == row["case_id"] and r["method"] == "v5_physics" for r in records):
                records.append(_failure_record(row, "v5_physics", error, alpha=core["linear_options"]["alpha_g"]))
        if baselines is not None:
            try:
                if raw_dn is None:
                    raise ValueError("raw unavailable or input hashes invalid")
                for baseline, value in baselines.evaluate(raw_dn).items():
                    records.append(_record(row, baseline, value, truth, physics, value["info"].get("seconds", 0.)))
            except Exception as error:
                for baseline in ("B1_tikhonov", "v3_physics", "old_B4", "v4.4_final_continuum_fixed"):
                    if not any(r["case_id"] == row["case_id"] and r["method"] == baseline for r in records):
                        records.append(_failure_record(row, baseline, error))
        for record in [r for r in records if r["case_id"] == row["case_id"]]:
            common.append_jsonl(DIRECTORY/(name+"_records.jsonl"), _plain(record))
    return records


def _rescore_selected(physics, folder, manifest, selected_records, core, threshold, budget):
    rows = {r["case_id"]: r for r in manifest["rows"]}
    scored = []
    for rec in selected_records:
        budget.check()
        row = rows[rec["case_id"]]
        try:
            raw_dn, truth = _inputs(folder, row, physics)
            output = load_cached_prediction(rec, physics)
            output["raw_e"] = common.raw_electrons(physics, raw_dn)
            proxy = raw_noise_proxy(physics, raw_dn, core)
            new = _record(row, "v5_physics", output, truth, physics, rec["diagnostics"].get("seconds", 0),
                          alpha=rec["alpha_g"], sigma=proxy["sigma_S"], threshold=threshold)
            new["diagnostics"] = rec["diagnostics"]
            new["prediction_cache"] = rec["prediction_cache"]
            new["noise_proxy"] = proxy["summary"]
        except Exception as error:
            new = _failure_record(row, "v5_physics", error, alpha=rec["alpha_g"])
        scored.append(new)
        common.append_jsonl(DIRECTORY/"selected_records.jsonl", _plain(new))
    return scored


def _mc_audit(physics, folder, manifest, selected_records, core, protocol, case_ids, budget):
    rows = {r["case_id"]: r for r in manifest["rows"]}
    records = {r["case_id"]: r for r in selected_records}
    case_results, parent = [], {}
    for cid in case_ids:
        budget.check()
        row = rows[cid]
        try:
            raw_dn, truth = _inputs(folder, row, physics)
            baseline = load_cached_prediction(records[cid], physics)
            output = monte_carlo_uncertainty(physics, raw_dn, reconstruct,
                {"physics_config": physics.cfg}, core, options={"repeats": protocol["MC"]["repeats"],
                "seed": protocol["MC"]["seed"], "confidence_level": protocol["MC"]["confidence_level"]}, baseline=baseline)
            if output["status"] != "completed":
                raise ValueError("MC abstained: "+output["summary"].get("reason", "unknown"))
            fields = {key: value for key, value in output.items() if isinstance(value, torch.Tensor)}
            receipt = _save_arrays(common.DATA/"mc_uncertainty_predictions"/(cid+".npz"), fields)
            merged = {**baseline, **output, "raw_e": common.raw_electrons(physics, raw_dn)}
            scored = score_prediction(merged, truth, physics)
            metrics = scored.get("metrics") or {}
            s = metrics.get("signed_interval", {})
            t = metrics.get("temperature_interval", {})
            result = {"case_id": cid, "mother_id": row["mother_id"], "sign": records[cid]["sign"],
                      "geometry": row["geometry"], "template_geometry": row["template_geometry"],
                      "status": "completed", "raw_sha256": row["raw_file_sha256"],
                      "original_same_raw_cache": records[cid]["prediction_cache"],
                      "MC": _plain(output["summary"]), "signed_interval": s, "temperature_interval": t,
                      "uncertainty_arrays": receipt}
            domain = torch.as_tensor(truth["evaluation_domain"], device=physics.M.device).bool()
            if domain.ndim == 2:
                domain = domain[None]
            result["bias_variance"] = {
                "baseline_template_bias_e": metrics.get("template_bias_e"),
                "baseline_temperature_bias_K": metrics.get("temperature_bias_K"),
                "MC_empirical_postblur_template_variance_mean_e2": float(output["sigma_S"][domain].square().mean())
                    if bool(torch.isfinite(output["sigma_S"][domain]).all()) else None,
                "MC_empirical_color_temperature_variance_mean_K2": float(output["temperature_mc_sigma_K"][domain].square().mean())
                    if bool(torch.isfinite(output["temperature_mc_sigma_K"][domain]).all()) else None,
                "variance_scope": "finite raw-plugin noise repeats; calibration/model discrepancy omitted"}
        except Exception as error:
            result = {"case_id": cid, "mother_id": row["mother_id"], "status": "failed",
                      "error_type": type(error).__name__, "error": str(error),
                      "signed_interval": {"coverage_full_truth_domain": 0.},
                      "temperature_interval": {"coverage_full_truth_domain": 0.}}
        parent.setdefault(row["mother_id"], []).append(result)
        case_results.append(result)
    mother_s = {mid: sum(r["signed_interval"].get("coverage_full_truth_domain", 0.) or 0. for r in values)/len(values)
                for mid, values in parent.items()}
    mother_t = {mid: sum(r["temperature_interval"].get("coverage_full_truth_domain", 0.) or 0. for r in values)/len(values)
                for mid, values in parent.items()}
    complete = len(case_results) == 12 and all(r["status"] == "completed" for r in case_results)
    return {"schema": "v5-external-nominal-mc-coverage-001", "complete": complete,
            "cases": case_results, "predeclared_case_ids": case_ids,
            "selection_sha256": common.sha(DIRECTORY/"selection.json"),
            "nominal_only": True, "truth_entered_reconstruction": False,
            "coverage_rule": "mean of full truth-domain pixel interval coverage within each mother, then equal mother mean; failed MC counts as zero coverage",
            "independent_parent_count": len(parent), "pixels_assumed_independent": False,
            "signed_parent_mean_coverage": sum(mother_s.values())/len(mother_s) if mother_s else 0.,
            "temperature_parent_mean_coverage": sum(mother_t.values())/len(mother_t) if mother_t else 0.,
            "signed_parent_coverage_bootstrap": mother_bootstrap(mother_s),
            "temperature_parent_coverage_bootstrap": mother_bootstrap(mother_t),
            "gate_coverage": min(sum(mother_s.values())/len(mother_s), sum(mother_t.values())/len(mother_t)) if mother_s else 0.,
            "small_MC_normal_intervals_have_no_coverage_guarantee": True}


def _hessian_audit(physics, folder, manifest, selected_records, core, protocol, cid):
    row = next(r for r in manifest["rows"] if r["case_id"] == cid)
    record = next(r for r in selected_records if r["case_id"] == cid)
    try:
        raw_dn, _ = _inputs(folder, row, physics)
        output = load_cached_prediction(record, physics)
        system = create_linear_system(physics, common.raw_electrons(physics, raw_dn),
                                      output["continuum_latent"], core["linear_options"])
        probes = hessian_probe_diagnostics(system["matvec"], torch.zeros_like(system["rhs"]),
            system["measurement_matvec"], options={"probes": protocol["Hessian"]["probes"],
                "seed": protocol["Hessian"]["seed"], "cg_tolerance": 1e-4, "cg_max_iterations": 200},
            preconditioner=system["precondition"], residual_scale=system["residual_scale"])
        arrays = _save_arrays(common.DATA/"hessian_probes"/(cid+".npz"),
                              {k: probes[k] for k in ("diagonal_estimate", "sigma_proxy", "valid")})
        return {"schema": "v5-packed-latent-hessian-audit-001", "case_id": cid, "raw_sha256": row["raw_file_sha256"],
                "summary": _plain(probes["summary"]), "arrays": arrays,
                "coordinates": {"first_n": system["n_S"], "first_units": "latent S electrons",
                                "last_n": 3, "last_units": "normalized beta coefficients"},
                "postblur_sigma_claimed": False, "continuum_propagated": False,
                "exact_full_image_CRLB_claimed": False}
    except Exception as error:
        return {"schema": "v5-packed-latent-hessian-audit-001", "case_id": cid, "status": "failed",
                "error_type": type(error).__name__, "error": str(error), "postblur_sigma_claimed": False}


def _weak_detectable(records):
    result = {}
    for sign in ("positive", "negative"):
        rows = filter_records(records, group="weak", input_kind="noise", sign=sign)
        detectable = [r["case_id"] for r in rows if r.get("numerical_execution_passed")
            and r.get("detection", {}).get("frame_detected") is True
            and r["detection"]["valid_fraction"] >= .99 and r["detection"]["proxy_valid_fraction"] == 1.]
        result[sign] = {"detectable_case_ids": detectable,
                        "undetectable_case_ids": [r["case_id"] for r in rows if r["case_id"] not in detectable],
                        "available": bool(detectable), "cases": len(rows),
                        "rule": "raw-only full-FOV standardized maximum exceeds frozen independent-null threshold; numerical output passes and valid fraction>=.99",
                        "truth_used_for_detectability": False,
                        "sign_source": "external descriptor grouping only; does not enter detectability rule"}
    return result


def evaluate_physics_gate(records, reference, selection_feasible, null_eval, mc, stage1):
    """Pure decision function for a complete frozen physics evaluation."""
    noisy = filter_records(records, method="v5_physics", input_kind="noise")
    pooled = aggregate_metrics(noisy)
    by_sign, checks = {}, {}
    for sign in ("positive", "negative"):
        physical = aggregate_metrics(filter_records(noisy, group="normal", sign=sign))
        traditional = aggregate_metrics(filter_records(records, method=reference, group="normal", input_kind="noise", sign=sign))
        ratio = (physical["template_RMSE_e"]/traditional["template_RMSE_e"]
                 if physical.get("complete") and traditional.get("complete") and traditional["template_RMSE_e"] > 0 else None)
        passed = _finite(ratio) and ratio <= 1.05
        by_sign[sign] = {"passed": passed, "RMSE_factor": ratio, "maximum": 1.05,
                         "physics": physical, "traditional": traditional}
        checks["signed_normal_"+sign] = passed
    color = pooled.get("temperature_RMSE_K")
    validity = pooled.get("valid_fraction")
    coverage = mc.get("gate_coverage")
    checks.update(selection_feasible=bool(selection_feasible),
                  nominal_color=_finite(color) and color <= 20.,
                  valid_coverage=_finite(validity) and validity >= .99,
                  all_selected_noisy_CG_converged=pooled.get("CG", {}).get("all_converged") is True,
                  null_heldout_FP=null_eval.get("complete") is True and _finite(null_eval.get("frame_FP"))
                                   and null_eval["frame_FP"] <= .05 and null_eval.get("all_numerical_predictions_passed") is True,
                  MC_nominal_coverage=mc.get("complete") is True and _finite(coverage) and coverage >= .90)
    no_hardstop = bool(stage1.get("stage1_evidence_complete") and stage1.get("physical_stage2_eligible")
                       and stage1.get("ml_stop_required") is False)
    return {"schema": "v5-physics-development-gate-001", "physics_eligible": all(checks.values()) and no_hardstop,
            "no_hardstop": no_hardstop, "physical_metrics_passed": all(checks.values()),
            "gates": checks, "failed_gates": [name for name, passed in checks.items() if not passed],
            "signed_normal": by_sign, "nominal_pooled": pooled,
            "MCcoverage": coverage, "MCcoverage_minimum": .90,
            "null_heldout_frame_FP": null_eval.get("frame_FP"),
            "weak_L2_required_below_one": False,
            "final_and_confirmation_sealed": True, "scientific_acceptance_is_not_code_execution": True}


def _write_report(gate, selection, null, mc, hessian, stress):
    path = common.ed_path(DIRECTORY/"PHYSICS_ONLY_V5_REPORT.md")
    if path.exists():
        raise FileExistsError(path)
    def number(v):
        return "unavailable" if v is None else f"{v:.6g}"
    lines = ["# Physics-only v5 development decision", "",
        f"Physics candidate eligible: **{gate['physics_eligible']}**. Stage-1 no-hard-stop: **{gate['no_hardstop']}**.", "",
        f"Frozen alpha_g={selection['selected_alpha_g']}; traditional reference={selection['traditional_reference']}.",
        "One continuum fit per development raw preceded four signed S/beta solves. All four old methods received that same raw. No S feedback changed the continuum.", "",
        "Main field errors use postblur template and continuum color on the complete fixed truth domain; abstained pixels remain in errors. Synthetic negative template coefficients are deviations, not concentration.", "",
        "| Gate | Result |", "|---|---|",
        *[f"| {name} | {passed} |" for name, passed in gate["gates"].items()], "",
        f"Nominal color RMSE: {number(gate['nominal_pooled'].get('temperature_RMSE_K'))} K; valid coverage: {number(gate['nominal_pooled'].get('valid_fraction'))}.",
        *[f"Normal {sign} RMSE/reference factor: {number(gate['signed_normal'][sign]['RMSE_factor'])}." for sign in ("positive", "negative")], "",
        f"Independent null z threshold: {number(null.get('z_threshold'))}; heldout frame FP: {number(null.get('evaluation',{}).get('frame_FP'))}; pixel FP: {number(null.get('evaluation',{}).get('pixel_FP'))}.",
        "Threshold is the .95 higher quantile of the full-FOV maximum absolute standardized postblur template over calibration frames. Null evaluation uses disjoint mothers and never retunes it.", "",
        "The standardized scale is an observed-raw stationary postblur noise proxy. It omits continuum and beta propagation and provides no confidence interval or instrument guarantee. MC uncertainty is reported separately.", "",
        f"MC: 12 predeclared normal +/- cases, 8 repeats per case, rerunning raw-only continuum+signed inversion every repeat. Parent mean full-domain signed interval coverage={number(mc['signed_parent_mean_coverage'])}; color interval coverage={number(mc['temperature_parent_mean_coverage'])}.",
        "Small-sample normal intervals have no guaranteed coverage. Truth enters only the external coverage calculation. Pixel independence is never used for uncertainty on reported scene performance.", "",
        "Hessian: eight fixed Rademacher probes on one selected normal case. Diagonal estimates are in packed latent S/beta coordinates; they are not postblur sigma or a full-image exact CRLB.",
        f"Hessian all probes converged: {hessian.get('summary',{}).get('all_converged',False)}.", "",
        "Calibration stress is a separate fixed gain/emissivity experiment. Nominal acceptance is not extended to those stresses. Reference/nominal color errors, validity and rejection are retained in calibration_stress_records.jsonl.",
        f"Stress records: {len(stress)}; no stress result was used to choose alpha or threshold.", "",
        "records.jsonl retains per-case bias, fixed-domain error, true CG histories, regularization scales, latent gradient energy and route latency. paired_statistics.json resamples whole mothers 2000 times with seed 53130, preserving signed conditions and noise repeats.", "",
        "Confirmation and final arrays remain sealed. An unsuccessful physics gate forbids a learned candidate and preserves a negative result.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    assert path.read_text(encoding="utf-8") == "\n".join(lines)
    return {"path": common.relative(path), "sha256": common.sha(path)}


def run_development(device="cuda", options=None):
    """Root-owned formal phase; finite frozen data and no confirmation access."""
    if options:
        raise ValueError("Formal development options are fully preregistered; no run-time retuning")
    protocol = common.require_registered()
    if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
        raise FileExistsError("Never overwrite a started development experiment")
    stage1_path = common.REPORTS/"DECISION_1.json"
    stage1 = common.read_json(stage1_path)
    if stage1.get("physical_stage2_eligible") is not True or stage1.get("stage1_evidence_complete") is not True:
        raise ValueError("Stage 1 has not authorized physical development; preserve its stop decision")
    budget = common.Budget(protocol["phase_budgets_s"]["development-compare"])
    plans = common.read_json(common.CODE/"family_plans.json")
    assert common.sha(common.CODE/"family_plans.json") == protocol["family_plans_sha256"]
    physics = common.make_physics(device, torch.float32, config=protocol["physics_config"])
    from .baselines import BaselineSet
    baselines = BaselineSet(device, physics.cfg)
    folder, manifest, manifest_receipt = _verified_family(physics, plans, "development", protocol)
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    common.write_json(DIRECTORY/"DEVELOPMENT_STARTED.json", {"time": common.now(),
        "protocol_sha256": common.sha(common.CODE/"protocol.json"), "manifest": manifest_receipt,
        "stage1_path": common.relative(stage1_path), "stage1_sha256": common.sha(stage1_path),
        "source_sha256": _source_receipt(), "device": str(device), "sealed_arrays_opened": 0})
    records = _run_grid(physics, folder, manifest, protocol, baselines, budget)
    selection = _selection(records, protocol["alpha_g_development_grid"])
    core = {"continuum_options": copy.deepcopy(protocol["continuum_options"]),
            "linear_options": {**protocol["linear_defaults"], "alpha_g": selection["selected_alpha_g"]},
            "device": str(device), "dtype": "float32"}
    mc_case_ids = _mc_cases(plans["plans"]["development"])
    selection.update(schema="v5-physics-selection-001", status="FROZEN_BEFORE_NULL_THRESHOLD_CALIBRATION",
                     core_selection=core, uncertainty_proxy=copy.deepcopy(PROXY_SPEC),
                     thresholds_file=common.relative(DIRECTORY/"null_threshold.json"),
                     MC_case_ids=mc_case_ids, development_manifest=manifest_receipt,
                     protocol_sha256=common.sha(common.CODE/"protocol.json"),
                     source_sha256=_source_receipt(), statistics=statistics_plan(),
                     frozen_at=common.now(), confirmation_or_final_used=False)
    selection_hash = common.write_json(DIRECTORY/"selection.json", _plain(selection))
    selected = filter_records(records, method="v5_physics", alpha_g=selection["selected_alpha_g"])
    cal_folder, cal_manifest, cal_receipt = _verified_family(physics, plans, "null_calibration", protocol)
    cal_records = _run_selected_family(physics, cal_folder, cal_manifest, core, "null_calibration", budget)
    threshold = calibrate_null_threshold(cal_records)
    calibration_freeze = {"schema": "v5-null-calibration-threshold-freeze-001", **threshold,
                          "selection_hash": selection_hash, "calibration_manifest": cal_receipt,
                          "frozen_at": common.now(), "evaluation_frames_opened": 0,
                          "protocol_sha256": common.sha(common.CODE/"protocol.json")}
    calibration_path = DIRECTORY/"null_calibration_threshold.json"
    calibration_hash = common.write_json(calibration_path, _plain(calibration_freeze))
    eval_folder, eval_manifest, eval_receipt = _verified_family(physics, plans, "null_evaluation", protocol)
    # A failed calibration is retained; it cannot manufacture a usable threshold.
    z = threshold["z_threshold"] if threshold["complete"] else None
    eval_records = _run_selected_family(physics, eval_folder, eval_manifest, core, "null_evaluation", budget, z)
    if z is None:
        heldout = {"complete": False, "frame_FP": None, "pixel_FP": None,
                   "reason": "independent calibration has no valid threshold"}
    else:
        heldout = _null_summary(eval_records, z)
    assert set(threshold["calibration_mother_ids"]).isdisjoint({r["mother_id"] for r in eval_records})
    null = {"schema": "v5-independent-null-threshold-001", **threshold,
            "selection_hash": selection_hash, "calibration_manifest": cal_receipt,
            "evaluation_manifest": eval_receipt, "evaluation": heldout,
            "evaluation_mother_ids": sorted({r["mother_id"] for r in eval_records}),
            "noise_proxy": copy.deepcopy(PROXY_SPEC),
            "calibration_threshold_freeze": {"path": common.relative(calibration_path), "sha256": calibration_hash},
            "protocol_sha256": common.sha(common.CODE/"protocol.json"), "threshold_frozen_before_evaluation": True,
            "threshold_chosen_at": "from calibration scores before any null-evaluation reconstruction"}
    threshold_hash = common.write_json(DIRECTORY/"null_threshold.json", _plain(null))
    selected_scored = _rescore_selected(physics, folder, manifest, selected, core, z, budget)
    mc = _mc_audit(physics, folder, manifest, selected, core, protocol, mc_case_ids, budget)
    common.write_json(DIRECTORY/"mc_coverage.json", _plain(mc))
    budget.check()
    hessian = _hessian_audit(physics, folder, manifest, selected, core, protocol, mc_case_ids[0])
    common.write_json(DIRECTORY/"hessian_diagnostics.json", _plain(hessian))
    stress_folder, stress_manifest, stress_receipt = _verified_family(physics, plans, "calibration_stress", protocol)
    stress = _run_selected_family(physics, stress_folder, stress_manifest, core, "calibration_stress", budget, z, baselines)
    comparisons = [paired_statistics(records, "v5_physics", selection["traditional_reference"],
                                     group=group, sign=sign, alpha_g=selection["selected_alpha_g"])
                   for group in ("normal", "weak") for sign in (None, "positive", "negative")]
    stats_receipt = {"schema": "v5-development-paired-statistics-001", "plan": statistics_plan(),
                     "selection_hash": selection_hash, "comparisons": comparisons,
                     "frozen_before_final": True}
    stats_hash = common.write_json(DIRECTORY/"paired_statistics.json", _plain(stats_receipt))
    # Use the original complete comparison records for gate losses; detection
    # rescores must never remove an unavailable case or alter selection metrics.
    gate = evaluate_physics_gate([*selected, *filter_records(records, method=selection["traditional_reference"])],
        selection["traditional_reference"], selection["selection_feasible"], heldout, mc, stage1)
    gate.update(selection_hash=selection_hash, selection_path=common.relative(DIRECTORY/"selection.json"),
                threshold_hash=threshold_hash, threshold_path=common.relative(DIRECTORY/"null_threshold.json"),
                weakdetectable_by_sign=_weak_detectable(selected_scored),
                stage1={"path": common.relative(stage1_path), "sha256": common.sha(stage1_path), **stage1},
                manifest_receipts={"development": manifest_receipt, "null_calibration": cal_receipt,
                                   "null_evaluation": eval_receipt, "calibration_stress": stress_receipt},
                statistics={"path": common.relative(DIRECTORY/"paired_statistics.json"), "sha256": stats_hash,
                            "plan": statistics_plan()}, source_sha256=_source_receipt(),
                protocol_sha256=common.sha(common.CODE/"protocol.json"), elapsed_seconds=budget.elapsed(),
                pure_physics_candidate_kind="v5_physics", final_payloads_opened=0, confirmation_payloads_opened=0)
    gate["report"] = _write_report(gate, selection, null, mc, hessian, stress)
    common.write_json(DIRECTORY/"development_gate.json", _plain(gate))
    return gate
