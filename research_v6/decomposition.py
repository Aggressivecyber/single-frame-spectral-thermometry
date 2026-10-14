"""Labelled Stage A error accounting for the unchanged frozen v5 quadratic.

Truth is used here only to explain an independently returned signed estimate.
This module never clips, masks, changes that estimate, or reconstructs raw data.
"""
from __future__ import annotations

from . import common
common.configure_runtime()

import copy
import hashlib
import math
import time
import torch


DEFAULT_IDENTITY_CRITERIA = {
    "postblur_rms_limit_e": 0.05,
    "postblur_relative_l2_limit": 0.005,
    "postblur_relative_l2_scale_floor_e": 1.0,
    "normal_equation_algebra_rtol": 1e-4,
}
_COMPONENTS = ("leakage", "noise", "regularization_bias")


def _hash(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(str((str(a.dtype), a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rms(value):
    if not bool(torch.isfinite(value).all()):
        return None
    return float(value.double().square().mean().sqrt())


def _stats(value):
    finite = bool(torch.isfinite(value).all())
    return {"rms": _rms(value), "mean": float(value.double().mean()) if finite else None,
            "max_abs": float(value.abs().max()) if finite else None,
            "all_finite": finite, "count": value.numel(),
            "finite_count": int(torch.isfinite(value).sum()), "domain": "complete fixed field"}


def _pair(left, right):
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return {"twice_mean_product": None, "cosine": None, "pearson": None,
                "undefined_reason": "NONFINITE_FULL_DOMAIN"}
    a, b = left.double().flatten(), right.double().flatten()
    norm = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    ac, bc = a-a.mean(), b-b.mean()
    centered_norm = torch.linalg.vector_norm(ac) * torch.linalg.vector_norm(bc)
    return {"twice_mean_product": float(2 * (a*b).mean()),
            "cosine": float((a*b).sum()/norm) if float(norm) > 0 else None,
            "pearson": float((ac*bc).sum()/centered_norm) if float(centered_norm) > 0 else None,
            "undefined_reason": "ZERO_RMS_OR_CONSTANT" if float(centered_norm) == 0 else None}


def _options(options, physics, raw):
    frozen = common.frozen_v5("signed_linear_solver")
    supplied = {"alpha_g": 1.0, **({} if options is None else options)}
    cfg = frozen._options(supplied)
    if float(cfg["alpha_g"]) != 1 or float(cfg["rtol"]) != 1e-4 or int(cfg["max_iterations"]) != 200:
        raise ValueError("Stage A requires alpha_g=1, rtol=1e-4, max_iterations=200")
    cfg["weights"] = frozen.fixed_raw_weights(physics, raw[None], cfg.get("weights"))[0]
    return cfg


def _inputs(physics, raw_e, continuum_true, template_latent_true, beta_true):
    shape = tuple(physics.cfg["shape"])
    def tensor(value, wanted, name):
        out = torch.as_tensor(value, dtype=physics.M.dtype, device=physics.M.device).detach()
        if tuple(out.shape) == (1, *wanted):
            out = out[0]
        if tuple(out.shape) != wanted or not bool(torch.isfinite(out).all()):
            raise ValueError(name+" has invalid shape or nonfinite values")
        return out.clone()
    return (tensor(raw_e, shape, "raw_e"), tensor(continuum_true, (7, *shape), "C_true"),
            tensor(template_latent_true, shape, "S_true"), tensor(beta_true, (3,), "beta_true"))


def _regularizer(system, vector):
    frozen = common.frozen_v5("signed_linear_solver")
    s, beta = system["unpack"](vector)
    return system["pack"](system["lambda_g"]*frozen.laplacian(s)+system["lambda0"]*s,
                          None if beta is None else system["lambda_beta"]*beta)


def _solve(system, rhs, name):
    frozen = common.frozen_v5("signed_linear_solver")
    t = time.perf_counter()
    value, report = frozen.pcg_true_residual(
        system["matvec"], rhs, system["precondition"], residual_scale=system["residual_scale"],
        rtol=1e-4, max_iterations=200,
        refresh_interval=int(system["options"]["residual_refresh_interval"]))
    report = {**report, "component": name, "dtype": str(rhs.dtype), "rtol": 1e-4,
              "max_iterations": 200, "all_finite": bool(torch.isfinite(value).all()),
              "wall_s": time.perf_counter()-t}
    return {"value": value, "rhs": rhs, "CG": report}


def _initialize_cache(physics, system, raw, ctrue, strue, btrue, fixed_beta_mode, primary_cache=None):
    linear = system["linear_operator"]
    truth_vector = system["pack"](strue, None if fixed_beta_mode else btrue)
    ctrue_raw = physics.op(ctrue[None])[0]
    expected_true = ctrue_raw+linear.apply_template(strue)+linear.apply_beta(btrue)
    noise_raw = raw-expected_true
    noise_rhs = system["measurement_adjoint"](system["weights"]*noise_raw)
    bias_rhs = -_regularizer(system, truth_vector)
    promoted_rhs_differences = None
    if primary_cache is not None:
        promoted_rhs_differences = {
            "noise": _stats(noise_rhs-primary_cache["noise"]["rhs"].double()),
            "regularization_bias": _stats(bias_rhs-primary_cache["regularization_bias"]["rhs"].double()),
            "noise_raw": _stats(noise_raw-primary_cache["noise_raw"].double()),
        }
        noise_raw = primary_cache["noise_raw"].double()
        noise_rhs = primary_cache["noise"]["rhs"].double()
        bias_rhs = primary_cache["regularization_bias"]["rhs"].double()
    return {"physics": physics, "system": system, "raw": raw, "C_true": ctrue,
            "S_true": strue, "beta_true": btrue, "fixed_beta_mode": bool(fixed_beta_mode),
            "truth_vector": truth_vector, "C_true_raw": ctrue_raw,
            "expected_true_raw": expected_true, "noise_raw": noise_raw,
            "noise": _solve(system, noise_rhs, "noise"),
            "regularization_bias": _solve(system, bias_rhs, "regularization_bias"),
            "weight_sha256": _hash(system["weights"]),
            "physics_versions": tuple((n, b._version) for n, b in physics.named_buffers()),
            "float64_cache": None, "promoted_rhs_differences": promoted_rhs_differences}


@torch.no_grad()
def build_shared_cache(physics, raw_e, continuum_true, template_latent_true, beta_true,
                       options=None, fixed_beta_mode=False):
    """Build one immutable raw/mode K,W,truth/noise/bias snapshot, shared by C sources."""
    raw, ct, st, bt = _inputs(physics, raw_e, continuum_true, template_latent_true, beta_true)
    cfg = _options(options, physics, raw)
    system = common.frozen_v5("signed_linear_solver").create_linear_system(
        physics, raw, ct, cfg, fixed_beta=bt if fixed_beta_mode else None)
    return _initialize_cache(physics, system, raw, ct, st, bt, fixed_beta_mode)


def _validate_cache(cache, physics, raw, ct, st, bt, cfg, mode):
    if cache["physics"] is not physics or cache["fixed_beta_mode"] != bool(mode):
        raise ValueError("Shared decomposition cache is specific to one physics and beta mode")
    for label, value in (("raw", raw), ("C_true", ct), ("S_true", st), ("beta_true", bt)):
        if not torch.equal(cache[label], value):
            raise ValueError("Shared decomposition cache input changed: "+label)
    if cache["physics_versions"] != tuple((n, b._version) for n, b in physics.named_buffers()):
        raise ValueError("Calibration buffers changed after shared cache creation")
    for key in cfg:
        if key == "weights":
            if not torch.equal(cfg[key], cache["system"]["weights"]):
                raise ValueError("Shared cache weights changed")
        elif cfg[key] != cache["system"]["options"][key]:
            raise ValueError("Shared cache linear options changed: "+key)
    if _hash(cache["system"]["weights"]) != cache["weight_sha256"]:
        raise ValueError("Shared cache fixed weights were mutated")


def _closure(cache, cest, returned_vector, actual_report, criteria, primary_rhs=None,
             actual_postblur_error=None):
    p, system = cache["physics"], cache["system"]
    W, Gt = system["weights"], system["measurement_adjoint"]
    cest_raw = p.op(cest[None])[0]
    # Equal C fields produce an exactly zero leakage RHS, without a subtraction noise floor.
    leakage_raw = torch.zeros_like(cache["raw"]) if torch.equal(cest, cache["C_true"]) else p.op((cache["C_true"]-cest)[None])[0]
    promoted_leakage_rhs = Gt(W*leakage_raw)
    leakage_rhs = promoted_leakage_rhs if primary_rhs is None else primary_rhs["leakage"].double()
    components = {"leakage": _solve(system, leakage_rhs, "leakage"),
                  "noise": cache["noise"], "regularization_bias": cache["regularization_bias"]}
    summed = sum(v["value"] for v in components.values())
    error = returned_vector-cache["truth_vector"]
    discrepancy = error-summed
    serror, berror = system["unpack"](error)
    ssum, bsum = system["unpack"](summed)
    sdifference, _ = system["unpack"](discrepancy)
    post_error, post_sum = p.op.object_blur(serror), p.op.object_blur(ssum)
    post_difference = p.op.object_blur(sdifference)
    arrays = {"template_postblur_error_e": post_error,
              "recomposition_residual_postblur_e": post_difference,
              "template_latent_error_e": serror, "recomposition_residual_latent_e": sdifference,
              "leakage_raw_e": leakage_raw, "noise_raw_e": cache["noise_raw"],
              "component_sum_postblur_e": post_sum}
    post_components = {}
    for name, component in components.items():
        s, beta = system["unpack"](component["value"])
        post = p.op.object_blur(s)
        arrays[name+"_postblur_e"] = post
        arrays[name+"_latent_e"] = s
        if beta is not None:
            arrays[name+"_beta"] = beta
        post_components[name] = post
    # Compare the actual returned postblur estimate and the separately blurred
    # three terms. Latent recomposition is recorded independently above.
    post_error = (p.op.object_blur(system["unpack"](returned_vector)[0])-p.op.object_blur(cache["S_true"])
                  if actual_postblur_error is None else actual_postblur_error)
    post_sum = sum(post_components.values())
    post_difference = post_error-post_sum
    arrays.update(template_postblur_error_e=post_error, component_sum_postblur_e=post_sum,
                  recomposition_residual_postblur_e=post_difference)
    if berror is not None:
        arrays["beta_error"] = berror
        arrays["beta_recomposition_residual"] = berror-bsum
    # Source rhs and K*x_true use the original measurement/regularizer actions.
    target = cache["raw"]-cest_raw
    if cache["fixed_beta_mode"]:
        target = target-system["linear_operator"].apply_beta(cache["beta_true"])
    source_rhs = Gt(W*target)
    promoted_source_rhs = source_rhs
    if primary_rhs is not None:
        source_rhs = primary_rhs["source"].double()
    truth_normal = system["matvec"](cache["truth_vector"])
    component_rhs = sum(v["rhs"] for v in components.values())
    algebra_gap = source_rhs-truth_normal-component_rhs
    actual_residual = source_rhs-system["matvec"](returned_vector)
    component_residual_sum = sum(v["rhs"]-system["matvec"](v["value"]) for v in components.values())
    scale = system["residual_scale"].sqrt()
    def algebra_ratio(divisor):
        denominator = max(float(torch.linalg.vector_norm(v/divisor)) for v in (source_rhs, truth_normal, component_rhs))
        numerator = float(torch.linalg.vector_norm(algebra_gap/divisor))
        return numerator/max(denominator, 1e-30)
    algebra_physical, algebra_equilibrated = algebra_ratio(1), algebra_ratio(scale)
    rms_gap, rms_error = _rms(post_difference), _rms(post_error)
    relative_l2 = (None if rms_gap is None or rms_error is None else
                   rms_gap/max(rms_error, float(criteria["postblur_relative_l2_scale_floor_e"])))
    post_pass = bool(rms_gap is not None and relative_l2 is not None
                     and rms_gap <= criteria["postblur_rms_limit_e"]
                     and relative_l2 <= criteria["postblur_relative_l2_limit"])
    algebra_pass = bool(math.isfinite(algebra_physical) and math.isfinite(algebra_equilibrated)
                        and max(algebra_physical, algebra_equilibrated) <= criteria["normal_equation_algebra_rtol"])
    component_cg = all(v["CG"]["converged"] and v["CG"]["all_finite"] for v in components.values())
    actual_cg, accepted = bool(actual_report["CG"]["converged"]), bool(actual_report["accepted"])
    finite = bool(torch.isfinite(returned_vector).all() and all(torch.isfinite(v["value"]).all() for v in components.values()))
    cross = {a+"__"+b: _pair(post_components[a], post_components[b])
             for i, a in enumerate(_COMPONENTS) for b in _COMPONENTS[i+1:]}
    squared = {name: float(value.double().square().mean()) if bool(torch.isfinite(value).all()) else None
               for name, value in post_components.items()}
    mse_accounted = None if any(v is None for v in squared.values()) else sum(squared.values())+sum(v["twice_mean_product"] for v in cross.values())
    summary = {
        "schema": "v6-stage-a-linear-decomposition-001", "dtype": str(W.dtype),
        "fixed_true_beta_oracle": cache["fixed_beta_mode"],
        "coordinate_system": "latent S in e; beta in original normalized coefficients",
        "identity": "xhat-xtrue=K^-1 G^T W A(Ctrue-Cest)+K^-1 G^T W n-K^-1 Lambda xtrue",
        "object_blur_count": 1, "complete_field_scoring": True,
        "weights_sha256": cache["weight_sha256"], "weight_stats": _stats(W),
        "lambda_g": float(system["lambda_g"]), "lambda0": float(system["lambda0"]),
        "lambda_beta": float(system["lambda_beta"]),
        "options": {k: v for k, v in system["options"].items() if k != "weights"},
        "identity_criteria": dict(criteria), "actual_v5": actual_report,
        "component_CG": {k: v["CG"] for k, v in components.items()},
        "full_leak_noise_bias_CG_converged": component_cg,
        "actual_v5_CG_converged": actual_cg, "actual_v5_accepted": accepted,
        "postblur_recomposition_passed": post_pass,
        "normal_equation_algebra_passed": algebra_pass, "all_outputs_finite": finite,
        "numerical_gate_passed": component_cg and actual_cg and accepted and post_pass and algebra_pass and finite,
        "postblur_error": _stats(post_error), "postblur_recomposition_residual": _stats(post_difference),
        "postblur_recomposition_relative_l2": relative_l2,
        "latent_recomposition_residual": _stats(sdifference),
        "normal_equation_algebra_relative_physical": algebra_physical,
        "normal_equation_algebra_relative_equilibrated": algebra_equilibrated,
        "postblur_components": {k: _stats(v) for k, v in post_components.items()},
        "postblur_component_mse_e2": squared, "postblur_cross_terms": cross,
        "postblur_mse_accounted_e2": mse_accounted,
        "postblur_mse_of_component_sum_e2": float(post_sum.double().square().mean()) if finite else None,
        "raw_continuum_noise_correlation": _pair(leakage_raw, cache["noise_raw"]),
        "beta_recomposition_residual": None if berror is None else _stats(berror-bsum),
        "failure_interpretation": "numerical failure remains visible; no information-absence conclusion",
    }
    summary["failure_reasons"] = [key for key in ("full_leak_noise_bias_CG_converged", "actual_v5_CG_converged", "actual_v5_accepted", "postblur_recomposition_passed", "normal_equation_algebra_passed", "all_outputs_finite") if not summary[key]]
    if primary_rhs is not None:
        summary["rhs_rule"] = "exact primary noise/bias/leak/source RHS cast64; no new RHS used in diagnostic solves"
        summary["promoted_minus_primary_rhs"] = {
            **cache["promoted_rhs_differences"],
            "leakage": _stats(promoted_leakage_rhs-leakage_rhs),
            "source": _stats(promoted_source_rhs-source_rhs),
        }
    return {"summary": _safe(summary), "arrays": arrays,
            "_rhs_snapshot": {"source": source_rhs, "leakage": leakage_rhs,
                              "actual_residual": actual_residual,
                              "component_residual_sum": component_residual_sum,
                              "algebra_gap": algebra_gap}}


def _promoted_snapshot(cache):
    """Diagnostic arithmetic only: promote the existing buffers and fixed lambdas."""
    primary = cache["system"]
    p = copy.deepcopy(cache["physics"])
    p._apply(lambda buffer: buffer.to(dtype=torch.complex128 if buffer.is_complex() else torch.float64)
             if buffer.is_floating_point() or buffer.is_complex() else buffer)
    frozen = common.frozen_v5("signed_linear_solver")
    linear = frozen.SignedLinearOperator(p)
    # The frozen linear factory creates these operator constants in float32.
    # Cast its actual constants, including fused-pattern summation roundoff,
    # rather than regenerating different columns from the promoted calibration.
    for name in ("template_pattern", "beta_images", "stationary_normal_symbol", "laplacian_symbol"):
        setattr(linear, name, getattr(primary["linear_operator"], name).double())
    W = primary["weights"].double()
    # Preserve original diagonal/lambda buffers exactly; do not recalibrate the cyclic diagonal.
    linear.template_diagonal = lambda weights, mode: (primary["diag_S"].double(), {**primary["diagonal_info"], "promoted_fixed_primary_diagonal": True})
    raw, ct, st, bt = (cache[k].double() for k in ("raw", "C_true", "S_true", "beta_true"))
    cfg = {**primary["options"], "weights": W}
    system = frozen._create_single_frame_system(p, raw, p.op(ct[None])[0], W, linear, cfg,
                                                bt if cache["fixed_beta_mode"] else None)
    for key in ("lambda_g", "lambda0", "lambda_beta"):
        # Float operations in double need not round identically. Factory products use
        # cast primary diagonal; assert equality of each declared regularizer constant.
        if float(system[key]) != float(primary[key]):
            break
    else:
        system["residual_scale"] = primary["residual_scale"].double()
        return _initialize_cache(p, system, raw, ct, st, bt, cache["fixed_beta_mode"], cache)
    # The factory derives lambda values in its closures. Preserve them using an
    # explicit promoted action when factor multiplication differs by one rounding.
    lambdas = {k: primary[k].double() for k in ("lambda_g", "lambda0", "lambda_beta")}
    system.update(lambdas)
    system["matvec"] = lambda x: system["measurement_matvec"](x)+_regularizer(system, x)
    denominator = (W.mean()*linear.stationary_normal_symbol+lambdas["lambda_g"]*linear.laplacian_symbol+lambdas["lambda0"])
    beta_gram = (linear.beta_images.flatten(1)*W.flatten()[None])@linear.beta_images.flatten(1).T
    chol = torch.linalg.cholesky(beta_gram+lambdas["lambda_beta"]*torch.eye(3, dtype=W.dtype, device=W.device))
    def precondition(x):
        s, b = system["unpack"](x)
        return system["pack"](torch.fft.ifft2(torch.fft.fft2(s)/denominator).real,
                              None if b is None else torch.cholesky_solve(b[:, None], chol).squeeze(-1))
    system["precondition"] = precondition
    # Keep the same primary residual-coordinate definition, promoted without tuning.
    system["residual_scale"] = primary["residual_scale"].double()
    return _initialize_cache(p, system, raw, ct, st, bt, cache["fixed_beta_mode"], cache)


def _reference_estimate(cache, primary_source_rhs):
    system, p = cache["system"], cache["physics"]
    solved = _solve(system, primary_source_rhs.double(), "float64_reference_estimate")
    # This diagnostic is not returned as a scientific estimate and cannot rescue
    # either convergence or objective rejection of the original actual output.
    return solved["value"], {"CG": solved["CG"], "accepted": bool(torch.isfinite(solved["value"]).all()),
                              "role": "diagnostic promoted fixed system; never replaces actual v5"}


@torch.no_grad()
def decompose(physics, raw_e, continuum_est, continuum_true, template_latent_true, beta_true,
              options=None, fixed_beta_mode=False, *, v5_result=None, shared_cache=None,
              identity_criteria=None, float64_reference=False):
    """Explain one frozen actual estimate; return JSON summary, tensors and reusable cache.

    The optional double diagnostic runs only after primary numerical failure and
    never changes the primary stop gate. Cache reuse avoids two solves per C source.
    """
    started = time.perf_counter()
    raw, ct, st, bt = _inputs(physics, raw_e, continuum_true, template_latent_true, beta_true)
    cfg = _options(options, physics, raw)
    cest = torch.as_tensor(continuum_est, dtype=raw.dtype, device=raw.device).detach()
    if tuple(cest.shape) == (1, *ct.shape):
        cest = cest[0]
    if tuple(cest.shape) != tuple(ct.shape) or not bool(torch.isfinite(cest).all()):
        raise ValueError("continuum_est must be finite latent 7,H,W or 1,7,H,W")
    criteria = {**DEFAULT_IDENTITY_CRITERIA, **({} if identity_criteria is None else identity_criteria)}
    if set(criteria) != set(DEFAULT_IDENTITY_CRITERIA) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in criteria.values()):
        raise ValueError("Identity criteria must contain the four finite positive preregistered keys")
    cache = shared_cache or build_shared_cache(physics, raw, ct, st, bt, cfg, fixed_beta_mode)
    _validate_cache(cache, physics, raw, ct, st, bt, cfg, fixed_beta_mode)
    if v5_result is None:
        v5_result = common.frozen_v5("signed_linear_solver").solve_signed_linear(
            physics, raw[None], cest[None], cfg, fixed_beta=bt[None] if fixed_beta_mode else None)
    if not torch.equal(v5_result["weights"][0], cache["system"]["weights"]):
        raise ValueError("Actual v5 result used different raw weights")
    if not torch.equal(v5_result["continuum_raw_e"][0], physics.op(cest[None])[0]):
        raise ValueError("Actual v5 result used a different continuum")
    actual_report = copy.deepcopy(v5_result["info"]["frames"][0])
    if bool(actual_report["fixed_beta_diagnostic_oracle"]) != bool(fixed_beta_mode):
        raise ValueError("Actual v5 result used a different beta mode")
    for key, value in v5_result["info"]["options"].items():
        if cfg[key] != value:
            raise ValueError("Actual v5 result options differ: "+key)
    returned = cache["system"]["pack"](v5_result["template_latent_e"][0],
                None if fixed_beta_mode else v5_result["beta"][0])
    actual_post_error = v5_result["template_postblur_e"][0]-physics.op.object_blur(st)
    result = _closure(cache, cest, returned, actual_report, criteria,
                      actual_postblur_error=actual_post_error)
    result["summary"]["float64_reference"] = {"requested": bool(float64_reference), "ran": False,
                                                 "cannot_rescue_primary_failure": True}
    if float64_reference and not result["summary"]["numerical_gate_passed"]:
        # A failed primary identity gets a bounded explanatory solve, never a
        # corrected scientific estimate. For d=(xhat-xtrue)-sum(x_components),
        # Kd = algebra_gap - actual_residual + sum(component_residuals).
        rhs_snapshot = result["_rhs_snapshot"]
        inverse_actual = _solve(cache["system"], -rhs_snapshot["actual_residual"],
                                "negative_actual_estimate_residual")
        inverse_gap = _solve(cache["system"], rhs_snapshot["algebra_gap"]-rhs_snapshot["actual_residual"]
                             +rhs_snapshot["component_residual_sum"], "recomposition_normal_gap")
        actual_s, _ = cache["system"]["unpack"](inverse_actual["value"])
        gap_s, _ = cache["system"]["unpack"](inverse_gap["value"])
        actual_post, explained_gap_post = physics.op.object_blur(actual_s), physics.op.object_blur(gap_s)
        primary_gap = result["arrays"]["recomposition_residual_postblur_e"]
        result["summary"]["finite_solve_residual_diagnostic"] = {
            "role": "same-budget inverse residual explanation; cannot rescue primary failure",
            "negative_actual_residual_CG": inverse_actual["CG"],
            "combined_normal_gap_CG": inverse_gap["CG"],
            "actual_residual_contribution_postblur": _stats(actual_post),
            "combined_normal_gap_contribution_postblur": _stats(explained_gap_post),
            "unexplained_after_actual_residual": _stats(primary_gap-actual_post),
            "unexplained_after_combined_residual": _stats(primary_gap-explained_gap_post),
        }
        result["arrays"]["actual_solve_residual_contribution_postblur_e"] = actual_post
        result["arrays"]["combined_solve_residual_contribution_postblur_e"] = explained_gap_post
        if cache["float64_cache"] is None:
            cache["float64_cache"] = _promoted_snapshot(cache)
        reference_cache = cache["float64_cache"]
        xr, info = _reference_estimate(reference_cache, result["_rhs_snapshot"]["source"])
        reference = _closure(reference_cache, cest.double(), xr, info, criteria, result["_rhs_snapshot"])
        result["summary"]["float64_reference"] = {"requested": True, "ran": True,
            "cannot_rescue_primary_failure": True,
            "calibration_rule": "same original buffers/W/primary lambda cast64; no reconstruction of calibration",
            "rhs_rule": "exact primary noise/bias/leak/source RHS cast64; promoted differences reported separately",
            "diagnostic": reference["summary"]}
        result["arrays"].update({"float64_reference_"+k: v for k, v in reference["arrays"].items()})
    result["summary"]["wall_s"] = time.perf_counter()-started
    result["shared_cache"] = cache
    return result
