"""700/905-only continuum initialization and spatially consistent warm fit.

Decoded bands are correlated, approximate observations from the same raw
frame.  They are used only to produce the continuum initialization that is
frozen by the later raw-domain template solve.  Their cost is never added to
the raw likelihood as another independent measurement.
"""
from __future__ import annotations

import copy
import time

import torch

from repair_v3.solver import DEFAULTS as WARM_DEFAULTS, lm_step


DEFAULT_OPTIONS = {
    "warm_iterations": 30,
    "cg_max_iterations": 120,
    "cg_relative_tolerance": 1e-4,
    "LM_initial": 1e-3,
    "LM_minimum": 1e-8,
    "LM_maximum": 1e8,
    "LM_trials": 6,
    "acceptance_ratio_minimum": 0.05,
    "line_search": [1.0, 0.5, 0.25, 0.125, 0.0625],
    "warm_regularization": [1e-6, 1e-6, 0.0, 0.0, 0.0],
}


def _warm_options(options):
    options = {} if options is None else dict(options)
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown continuum warm options: {sorted(unknown)}")
    config = copy.deepcopy(WARM_DEFAULTS)
    config.update(copy.deepcopy(DEFAULT_OPTIONS))
    config.update(options)
    if not 0 <= int(config["warm_iterations"]) <= 30:
        raise ValueError("warm_iterations must be between 0 and 30")
    if not 1 <= int(config["cg_max_iterations"]) <= 200:
        raise ValueError("Continuum warm CG maximum must be 1..200")
    if not 0 < float(config["cg_relative_tolerance"]) <= 1e-4:
        raise ValueError("Continuum warm CG tolerance must be <=1e-4")
    strengths = config["warm_regularization"]
    if len(strengths) != 5 or min(strengths) < 0 or any(v != 0 for v in strengths[2:]):
        raise ValueError("Only T/loga may have warm regularization")
    return config


@torch.no_grad()
def estimate_continuum(physics, raw_e, options=None):
    """Estimate a*C(T) from the two declared template/background-free bands.

    The returned state has five legacy-compatible coordinates, but its final
    three entries are exactly zero.  Beta remains zero throughout this warm
    fit.  The function accepts neither labels nor truth nor any frame identity.
    Reported temperature_color_K is the color temperature of the postblur
    continuum.  The latent parameter temperature is separately named.
    """
    config = _warm_options(options)
    raw_e = torch.as_tensor(raw_e, dtype=physics.M.dtype, device=physics.M.device).detach()
    if raw_e.ndim == 2:
        raw_e = raw_e[None]
    if raw_e.ndim != 3 or tuple(raw_e.shape[-2:]) != tuple(physics.cfg["shape"]):
        raise ValueError("raw_e must have shape B,H,W")
    if not bool(torch.isfinite(raw_e).all()):
        raise ValueError("raw_e must be finite")
    if bool((physics.M[5:7] != 0).any() or (physics.q[5:7] != 0).any()):
        raise ValueError("700/905 must have exactly zero template and background responses")
    decoded = physics.op.decode(raw_e).detach()
    decoded_red = decoded[:, 5:7]
    initial_color, initial_valid = physics.color_temperature(decoded)
    continuum_at_color, _ = physics.continuum(initial_color)
    red_model = continuum_at_color[:, 5:7]
    amplitude = ((decoded_red * red_model).sum(1, keepdim=True)
                 / red_model.square().sum(1, keepdim=True).clamp_min(
                     torch.finfo(raw_e.dtype).tiny))
    amplitude = amplitude.clamp(min=torch.exp(raw_e.new_tensor(-4.0)),
                                max=torch.exp(raw_e.new_tensor(0.3)))
    state = raw_e.new_zeros((len(raw_e), 5, *raw_e.shape[-2:]))
    state[:, 0] = (initial_color - physics.T0) / physics.Tscale
    state[:, 1:2] = amplitude.log()
    beta = raw_e.new_zeros((len(raw_e), 3))
    # These marginal decoded-band weights are an approximate warm-fit metric.
    # They are deliberately absent from the subsequent raw likelihood.
    warm_weights = (decoded_red.clamp_min(0) + physics.extra_variance).reciprocal()
    histories, states, frame_info = [], [], []
    for frame in range(len(raw_e)):
        started = time.perf_counter()
        current = state[frame:frame + 1].clone()
        zero_beta = beta[frame:frame + 1]
        damping = float(config["LM_initial"])
        history = []
        stop = "MAX_WARM_ITERATIONS"
        for iteration in range(1, int(config["warm_iterations"]) + 1):
            current, returned_beta, report = lm_step(
                physics, decoded_red[frame:frame + 1], current, zero_beta,
                warm_weights[frame:frame + 1], config, damping, warm=True)
            if not bool((current[:, 2:] == 0).all() and (returned_beta == 0).all()):
                raise RuntimeError("Continuum warm fit unexpectedly changed template/background")
            report["iteration"] = iteration
            report["objective_role"] = "correlated decoded-band initializer fit only"
            history.append(report)
            damping = float(report["next_damping"])
            if not report["accepted"]:
                stop = "NO_ACCEPTED_WARM_STEP"
                break
        if int(config["warm_iterations"]) == 0:
            stop = "INITIALIZER_ONLY"
        states.append(current[0])
        histories.append(history)
        frame_info.append({
            "frame": frame, "warm_iterations": len(history), "stop": stop,
            "wall_s": time.perf_counter() - started,
            "initial_color_valid_fraction": float(initial_valid[frame].to(torch.float64).mean()),
            "template_state_exactly_zero": True, "beta_exactly_zero": True,
            "accepted_warm_steps": sum(int(r["accepted"]) for r in history),
        })
    state = torch.stack(states)
    continuum_latent = physics.components(state, beta)[0].detach()
    continuum_postblur = physics.op.object_blur(continuum_latent)
    temperature_color, valid = physics.color_temperature(continuum_postblur)
    # A constrained Planck fit can turn an out-of-range decoded ratio into an
    # in-range model ratio.  Preserve the initial measurement validity so that
    # clipping alone never manufactures a valid color-temperature pixel.
    valid = (valid & initial_valid & torch.isfinite(temperature_color)
             & torch.isfinite(continuum_postblur).all(1))
    if not bool(torch.isfinite(state).all() and torch.isfinite(continuum_latent).all()):
        raise RuntimeError("Nonfinite continuum estimate")
    return {
        "state": state, "continuum_latent": continuum_latent,
        "continuum_postblur": continuum_postblur,
        "temperature_color_K": temperature_color,
        "temperature_latent_parameter_K": physics.T0 + physics.Tscale * state[:, 0],
        "temperature_uncertainty": None,
        "initial_color_valid": initial_valid,
        "valid": valid, "history": histories,
        "info": {
            "frames": frame_info,
            "options": {key: copy.deepcopy(config[key]) for key in DEFAULT_OPTIONS},
            "initializer_bands_nm": [700, 905],
            "amplitude_initializer": "two-band continuum least squares at decoded color temperature",
            "spatial_model": "object_blur(exp(loga)*band_integrated_Planck(T))[:,5:7]",
            "decoded_role": "correlated approximate observations used only for continuum initialization/warm fit",
            "decoded_covariance": "full cross-pixel/cross-band covariance not modeled in initializer metric",
            "raw_measurement_likelihood_count": 0,
            "warm_cost_added_to_raw_objective": False,
            "full_spectrum_joint_LM_called": False,
            "legacy_full_spectrum_initializer_called": False,
            "template_feedback_into_continuum": False,
            "temperature_semantics": "color temperature of postblur continuum under frozen synthetic calibration",
            "validity_rule": "postblur model color validity and original decoded 700/905 ratio/denominator validity",
            "uncertainty_status": "not computed here; use independently labelled fixed-parameter Monte Carlo",
        },
    }
