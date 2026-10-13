"""Raw-DN-only plug-in Monte Carlo and finite-probe curvature diagnostics.

The reconstruction callback is ``reconstruct(raw_dn, calibration, selection)``
and returns the canonical postblur template, color temperature, validity, and
raw fitted expectation. No truth, condition, or identity is an input here.
Empirical nominal interval coverage must be assessed outside this module.
"""
import copy
import hashlib
import json
import math
from statistics import NormalDist

import runtime
import torch


MC_DEFAULTS = {
    "repeats": 8,
    "seed": 53125,
    "confidence_level": 0.95,
    "minimum_valid_fraction": 0.75,
    "minimum_successful_repeats": 3,
    "keep_samples": False,
}
PROBE_DEFAULTS = {"probes": 8, "seed": 53126, "cg_tolerance": 1e-4,
                  "cg_max_iterations": 200}


def _options(defaults, supplied):
    supplied = {} if supplied is None else dict(supplied)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError("Unknown uncertainty options: " + ", ".join(sorted(unknown)))
    return dict(defaults, **supplied)


def _tensor_metadata(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().contiguous().cpu().numpy()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _tensor_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    raise TypeError("Calibration/selection must contain serializable values or tensors.")


def _fingerprint(value):
    text = json.dumps(_tensor_metadata(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_mc_seed_plan(seed=53125, repeats=8):
    """Predeclare deterministic independent RNG seeds; default 8, ceiling 16."""
    if isinstance(repeats, bool) or int(repeats) != repeats or not 3 <= repeats <= 16:
        raise ValueError("MC requires 3..16 repeats; recommended fixed plans are 8 or 16.")
    return [int.from_bytes(hashlib.sha256(f"thermometry-v5-mc:{int(seed)}:{i}".encode()).digest()[:8],
                           "little") % (2**63 - 1) for i in range(int(repeats))]


def _raw_batch(raw_dn, physics):
    if raw_dn.ndim == 4 and raw_dn.shape[1] == 1:
        raw_dn = raw_dn[:, 0]
    if raw_dn.ndim != 3 or tuple(raw_dn.shape[-2:]) != tuple(physics.cfg["shape"]):
        raise ValueError("Raw DN must be B x H x W or B x 1 x H x W.")
    if not bool(torch.isfinite(raw_dn).all()):
        raise ValueError("Raw DN must be finite.")
    return raw_dn


def _prediction(result, raw_shape):
    required = ("template_postblur_e", "temperature_color_K", "valid", "expected_e")
    if not isinstance(result, dict) or any(key not in result for key in required):
        raise ValueError("Callback must return canonical fields: " + ", ".join(required))
    extracted = {}
    for key in required:
        field = result[key]
        if not isinstance(field, torch.Tensor) or tuple(field.shape) != tuple(raw_shape):
            raise ValueError(key + " must be a tensor with the same B,H,W shape as raw.")
        extracted[key] = field.detach()
    extracted["valid"] = extracted["valid"].bool()
    for key in ("template_postblur_e", "temperature_color_K"):
        if bool((extracted["valid"] & ~torch.isfinite(extracted[key])).any()):
            raise ValueError("Valid callback pixels cannot contain non-finite " + key)
    return extracted


def _failure(reason, plan, raw_shape, summary_extra=None):
    return {"status": "abstained", "reason": reason,
            "summary": dict({"schema": "v5-raw-plugin-mc-001", "status": "abstained",
                             "reason": reason, "seed_plan": plan, "raw_shape": list(raw_shape),
                             "empirical_nominal_coverage": "not assessed; external nominal validation only"},
                            **(summary_extra or {}))}


@torch.no_grad()
def draw_raw_dn_sample(physics, expected_e, seed):
    """Sample Poisson + independent read noise + actual round/clamp ADC.

    Invalid Poisson expectations are errors, never silently clipped into a
    plausible measurement. Saturation of a generated sample is retained and
    reported; saturation of the source frame causes MC abstention upstream.
    """
    if not bool(torch.isfinite(expected_e).all()) or bool((expected_e < 0).any()):
        raise ValueError("Plug-in Poisson expectation must be finite and nonnegative.")
    if bool((expected_e >= physics.cfg["full_well_e"]).any()):
        raise ValueError("Plug-in expectation reaches detector full well.")
    generator = torch.Generator(device=expected_e.device).manual_seed(int(seed))
    counts = torch.poisson(expected_e, generator=generator)
    read = torch.randn(counts.shape, generator=generator, device=counts.device, dtype=counts.dtype)
    counts = counts + float(physics.cfg["read_noise_e"]) * read
    preclip_dn = (counts / physics.adc_step + physics.cfg["black_level_DN"]).round()
    maximum_dn = 2**physics.cfg["adc_bits"] - 1
    raw_dn = preclip_dn.clamp(0, maximum_dn)
    return raw_dn, {"upper_clipped_pixels": int((preclip_dn >= maximum_dn).sum()),
                    "lower_clipped_pixels": int((preclip_dn <= 0).sum())}


def _sample_statistics(samples, valid_samples, minimum_fraction, minimum_count):
    values = torch.stack(samples).double()
    validity = torch.stack(valid_samples).bool() & torch.isfinite(values)
    count = validity.sum(0)
    denominator = count.clamp_min(1)
    mean = torch.where(validity, values, torch.zeros_like(values)).sum(0) / denominator
    centered = torch.where(validity, values - mean, torch.zeros_like(values))
    variance = centered.square().sum(0) / (count - 1).clamp_min(1)
    fraction = count / len(samples)
    valid = (count >= minimum_count) & (fraction >= minimum_fraction)
    sigma = torch.where(valid, variance.clamp_min(0).sqrt(), torch.full_like(mean, float("nan")))
    return mean, sigma, fraction, valid


@torch.no_grad()
def monte_carlo_uncertainty(physics, raw_dn, reconstruct, calibration=None, selection=None,
                            options=None, baseline=None):
    """Reconstruct fixed noise repeats from an original raw-only fitted mean.

    The baseline may be a saved output of the same callback on this raw frame.
    It is not a new measurement or an added data term. ``expected_e`` must come
    from that original fit. Each repeat reruns both continuum and signed solve;
    a callback that freezes the continuum underreports propagated uncertainty.
    No empirical 95% coverage or unbiased-measurement CRLB is claimed here.
    """
    opt = _options(MC_DEFAULTS, options)
    plan = fixed_mc_seed_plan(opt["seed"], opt["repeats"])
    confidence = float(opt["confidence_level"])
    if not 0 < confidence < 1 or not 0 < opt["minimum_valid_fraction"] <= 1:
        raise ValueError("Confidence and validity fractions must be within their probability ranges.")
    if not 2 <= opt["minimum_successful_repeats"] <= opt["repeats"]:
        raise ValueError("minimum_successful_repeats must be between 2 and repeats.")
    raw = _raw_batch(raw_dn, physics)
    calibration = physics.cfg if calibration is None else calibration
    selection = {} if selection is None else selection
    fingerprints = {"calibration_sha256": _fingerprint(calibration),
                    "selection_sha256": _fingerprint(selection)}
    max_dn = 2**physics.cfg["adc_bits"] - 1
    if bool(((raw >= max_dn) | (raw <= 0)).any()):
        return _failure("original raw is ADC-clipped; plug-in mean is not identifiable from clipped values",
                        plan, raw.shape, fingerprints)
    if baseline is None:
        baseline = reconstruct(raw.detach().clone(), copy.deepcopy(calibration), copy.deepcopy(selection))
    central = _prediction(baseline, raw.shape)
    expected = central["expected_e"]
    if not bool(torch.isfinite(expected).all()) or bool((expected < 0).any()):
        return _failure("fitted original-raw expected_e is non-finite or negative; no Poisson mean clamp applied",
                        plan, raw.shape, fingerprints)
    if bool((expected >= physics.cfg["full_well_e"]).any()):
        return _failure("fitted original-raw expected_e reaches full well; plug-in MC abstained",
                        plan, raw.shape, fingerprints)
    if expected.device != raw.device:
        raise ValueError("Callback expected_e and raw_dn must share a device.")
    templates, temperatures, validities = [], [], []
    repeats = []
    for repeat, seed in enumerate(plan):
        noisy_dn, clipping = draw_raw_dn_sample(physics, expected, seed)
        row = {"repeat": repeat, "seed": seed, **clipping}
        try:
            output = reconstruct(noisy_dn, copy.deepcopy(calibration), copy.deepcopy(selection))
            prediction = _prediction(output, raw.shape)
            if not bool(torch.isfinite(prediction["expected_e"]).all()) or bool((prediction["expected_e"] < 0).any()):
                raise ValueError("repeat fitted expectation is non-finite or negative")
            # Saturated repeat pixels cannot contribute to an interval that
            # models an unclipped observation, even if the callback says valid.
            valid = prediction["valid"] & (noisy_dn > 0) & (noisy_dn < max_dn)
            templates.append(prediction["template_postblur_e"])
            temperatures.append(prediction["temperature_color_K"])
            validities.append(valid)
            row.update(status="completed", valid_pixel_fraction=float(valid.double().mean()))
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            templates.append(torch.full_like(expected, float("nan")))
            temperatures.append(torch.full_like(expected, float("nan")))
            validities.append(torch.zeros_like(expected, dtype=torch.bool))
        repeats.append(row)
    completed = sum(row["status"] == "completed" for row in repeats)
    if completed < opt["minimum_successful_repeats"]:
        return _failure("too few completed MC reconstructions", plan, raw.shape,
                        dict(fingerprints, repeats=repeats, completed_repeats=completed))
    s_mean, s_sigma, s_fraction, s_valid = _sample_statistics(
        templates, validities, opt["minimum_valid_fraction"], opt["minimum_successful_repeats"])
    t_mean, t_sigma, t_fraction, t_valid = _sample_statistics(
        temperatures, validities, opt["minimum_valid_fraction"], opt["minimum_successful_repeats"])
    valid = central["valid"] & s_valid & t_valid
    sigma_valid = valid & torch.isfinite(s_sigma) & (s_sigma > 0)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    s_center, t_center = central["template_postblur_e"].double(), central["temperature_color_K"].double()
    nan_s = torch.full_like(s_sigma, float("nan"))
    snr = torch.where(sigma_valid, s_center.abs() / s_sigma.clamp_min(torch.finfo(s_sigma.dtype).tiny), nan_s)
    undetectable = sigma_valid & (s_center.abs() <= z * s_sigma)
    low_confidence = ~sigma_valid | undetectable
    interval_valid = valid & torch.isfinite(s_sigma) & torch.isfinite(t_sigma)
    summary = {
        "schema": "v5-raw-plugin-mc-001", "status": "completed", "options": opt,
        "seed_plan": plan, "repeats": repeats, "completed_repeats": completed,
        **fingerprints, "raw_shape": list(raw.shape),
        "noise_model": "Poisson(original-raw fitted expected_e) + Gaussian read noise, ADC round/clamp to DN",
        "expected_source": "fit of the same original raw; plug-in, not independent measurement",
        "interval_method": "baseline +/- normal quantile * MC sample sigma; approximate, no bias correction",
        "confidence_level": confidence, "normal_quantile": z,
        "MC_template_target": "signed template_postblur_e",
        "MC_temperature_target": "continuum postblur temperature_color_K",
        "baseline_valid_fraction": float(central["valid"].double().mean()),
        "valid_coverage_fraction": float(valid.double().mean()),
        "interval_available_fraction": float(interval_valid.double().mean()),
        "low_confidence_fraction": float(low_confidence.double().mean()),
        "undetectable_fraction": float(undetectable.double().mean()),
        "empirical_nominal_coverage": "not assessed; external nominal validation only",
        "budget": "8 repeats recommended, 16 maximum; fixed mother subset chosen outside callback",
        "limitations": ["small finite MC sample; sample sigma is itself noisy",
                        "plug-in mean and calibration/model errors can bias uncertainty",
                        "nominal interval coverage requires separate externally scored validation",
                        "normal approximate signed interval does not impose template positivity",
                        "callback must rerun continuum and signed solver for every repeat"],
    }
    result = {"status": "completed", "sigma_S": s_sigma,
              "temperature_mc_sigma_K": t_sigma,
              "signed_ci_lower_e": torch.where(interval_valid, s_center-z*s_sigma, nan_s),
              "signed_ci_upper_e": torch.where(interval_valid, s_center+z*s_sigma, nan_s),
              "temperature_ci_lower_K": torch.where(interval_valid, t_center-z*t_sigma, nan_s),
              "temperature_ci_upper_K": torch.where(interval_valid, t_center+z*t_sigma, nan_s),
              "mc_template_mean_e": s_mean, "mc_temperature_mean_K": t_mean,
              "mc_valid_fraction": torch.minimum(s_fraction, t_fraction),
              "valid": valid, "interval_valid": interval_valid, "SNR": snr,
              "undetectable": undetectable, "low_confidence": low_confidence, "summary": summary}
    if opt["keep_samples"]:
        result.update(template_samples_e=torch.stack(templates),
                      temperature_samples_K=torch.stack(temperatures),
                      validity_samples=torch.stack(validities))
    return result


def _pcg_true_residual(matvec, rhs, preconditioner, tolerance, max_iterations, residual_scale=None):
    """Bounded SPD PCG; each stopping check uses rhs - H @ solution."""
    solution = torch.zeros_like(rhs)
    residual = rhs - matvec(solution)
    norm_rhs = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).tiny)
    scale_root = torch.ones_like(rhs) if residual_scale is None else residual_scale.sqrt()
    norm_equilibrated_rhs = torch.linalg.vector_norm(rhs/scale_root).clamp_min(torch.finfo(rhs.dtype).tiny)

    def residual_norms(value):
        return (float(torch.linalg.vector_norm(value)/norm_rhs),
                float(torch.linalg.vector_norm(value/scale_root)/norm_equilibrated_rhs))

    true_relative, equilibrated = residual_norms(residual)
    if true_relative <= tolerance and equilibrated <= tolerance:
        return solution, {"converged": True, "iterations": 0, "true_relative_residual": true_relative,
                          "equilibrated_true_relative_residual": equilibrated}
    apply_preconditioner = preconditioner if preconditioner is not None else lambda value: value
    z = apply_preconditioner(residual)
    direction = z.clone()
    rz = (residual * z).sum()
    status, iterations = "maximum_iterations", 0
    for iteration in range(int(max_iterations)):
        normal_direction = matvec(direction)
        curvature = (direction * normal_direction).sum()
        if not bool(torch.isfinite(curvature)) or not bool(torch.isfinite(rz)) or float(curvature) <= 0 or float(rz) <= 0:
            status = "nonpositive_or_nonfinite_curvature"
            break
        alpha = rz / curvature
        solution = solution + alpha * direction
        # Recomputed residual, not a recursively propagated stopping estimate.
        residual = rhs - matvec(solution)
        iterations = iteration + 1
        true_relative, equilibrated = residual_norms(residual)
        if not math.isfinite(true_relative) or not math.isfinite(equilibrated):
            status = "nonfinite_true_residual"
            break
        if true_relative <= tolerance and equilibrated <= tolerance:
            status = "converged"
            break
        z = apply_preconditioner(residual)
        next_rz = (residual * z).sum()
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    final_residual = rhs - matvec(solution)
    true_relative, equilibrated = residual_norms(final_residual)
    converged = (math.isfinite(true_relative) and math.isfinite(equilibrated)
                 and true_relative <= tolerance and equilibrated <= tolerance)
    return solution, {"converged": converged,
                      "iterations": iterations, "termination": status,
                      "true_relative_residual": true_relative if math.isfinite(true_relative) else None,
                      "equilibrated_true_relative_residual": equilibrated if math.isfinite(equilibrated) else None}


@torch.no_grad()
def hessian_probe_diagnostics(hessian_matvec, parameter_template, data_normal_matvec=None,
                              options=None, preconditioner=None, residual_scale=None):
    """Finite Hutchinson diagonal probes of H^-1 F H^-1 or inverse curvature.

    H includes the selected regularizers. With data normal F=A^T W A and a
    matching noise covariance, the sandwich estimates local estimator variance.
    Without F, H^-1 is a curvature/prior proxy, not unbiased measurement CRLB.
    Fields use exactly the parameter coordinates of the passed matvecs. Probe
    failures are reported; no finite error bar is produced from failed solves.
    """
    opt = _options(PROBE_DEFAULTS, options)
    count = int(opt["probes"])
    if isinstance(opt["probes"], bool) or count != opt["probes"] or not 1 <= count <= 16:
        raise ValueError("Finite Hessian probe budget is 1..16; default 8.")
    if opt["cg_tolerance"] <= 0 or not 1 <= opt["cg_max_iterations"] <= 200:
        raise ValueError("PCG requires positive tolerance and at most 200 iterations.")
    if not parameter_template.dtype.is_floating_point or not bool(torch.isfinite(parameter_template).all()):
        raise ValueError("parameter_template must be a finite floating point tensor.")
    if residual_scale is not None:
        if (tuple(residual_scale.shape) != tuple(parameter_template.shape)
                or not bool(torch.isfinite(residual_scale).all()) or bool((residual_scale <= 0).any())):
            raise ValueError("residual_scale must match parameters and be positive and finite.")
    generator = torch.Generator(device=parameter_template.device).manual_seed(int(opt["seed"]))
    estimates, rows = [], []
    for probe in range(count):
        signs = torch.randint(0, 2, parameter_template.shape, generator=generator,
                              device=parameter_template.device).to(parameter_template.dtype) * 2 - 1
        first, info_first = _pcg_true_residual(hessian_matvec, signs, preconditioner,
                                              opt["cg_tolerance"], opt["cg_max_iterations"], residual_scale)
        row = {"probe": probe, "inverse_solve": info_first}
        inverse_product = first
        successful = info_first["converged"]
        if data_normal_matvec is not None and successful:
            inverse_product, info_second = _pcg_true_residual(
                hessian_matvec, data_normal_matvec(first), preconditioner,
                opt["cg_tolerance"], opt["cg_max_iterations"], residual_scale)
            row["sandwich_solve"] = info_second
            successful = successful and info_second["converged"]
        row["successful"] = successful
        if successful:
            estimates.append(signs * inverse_product)
        rows.append(row)
    all_converged = len(estimates) == count
    diagonal = (torch.stack(estimates).mean(0) if estimates else
                torch.full_like(parameter_template, float("nan")))
    validity = torch.isfinite(diagonal) & (diagonal > 0) & all_converged
    sigma = torch.where(validity, diagonal.clamp_min(0).sqrt(),
                        torch.full_like(diagonal, float("nan")))
    kind = "H^-1 F H^-1 local penalized-estimator covariance" if data_normal_matvec is not None else "H^-1 inverse curvature/prior proxy"
    return {"diagonal_estimate": diagonal, "sigma_proxy": sigma, "valid": validity,
            "summary": {"schema": "v5-finite-hessian-probes-001", "options": opt,
                        "quantity": kind, "shape": list(parameter_template.shape),
                        "parameter_units": "exactly the caller matvec coordinates; no postblur variance transform inferred",
                        "residual_stopping_checks": "physical and diagonal-equilibrated true residual" if residual_scale is not None else "physical-coordinate true residual",
                        "probe_distribution": "fixed-seed Rademacher",
                        "successful_probes": len(estimates), "all_converged": all_converged,
                        "solves": rows,
                        "nonpositive_diagonal_pixels": int((torch.isfinite(diagonal) & (diagonal <= 0)).sum()),
                        "valid_fraction": float(validity.double().mean()),
                        "limitations": ["finite-probe diagonal approximation; estimates can be negative",
                                        "local linearization in the passed parameter coordinates",
                                        "H contains priors/regularization; not an unbiased measurement CRLB",
                                        "sandwich assumes F matches fixed raw W and noise covariance",
                                        "PCG nonconvergence invalidates reported sigma proxy"]}}
