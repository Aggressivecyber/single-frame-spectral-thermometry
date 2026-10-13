"""One fixed width-8 bounded gain control on a frozen physical estimate.

No continuum module, trainable temperature, additive template head, metadata,
or reconstruction feedback exists here. Supervision belongs to the outer
training loop. The preserved raw expectation is the *linear* fit and is never
presented as a raw refit of the modified postblur template.
"""
from __future__ import annotations

import copy
from research_v5 import common

common.configure_runtime()

import torch
from torch import nn


WIDTH = 8
GAIN_BOUND = 0.3
FEATURE_NAMES = (
    "log1p_absolute_linear_S_over_sigma",
    "log1p_absolute_linear_S_over_180e",
    "standardized_linear_raw_residual",
    "local_gradient_of_absolute_linear_S_over_180e",
    "log1p_fixed_postblur_continuum_over_1000e",
    "log1p_sigma_over_180e",
)
FORBIDDEN_INPUT_KEYS = {"truth", "condition", "clean_identity", "case_id", "input_id",
                        "labels", "coefficient_e", "sign", "sign_label", "mother_id"}


def _field(value, name, reference=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(name + " must be a B,H,W tensor")
    if reference is not None and (value.shape != reference.shape
                                  or value.device != reference.device):
        raise ValueError(name + " must share the template shape/device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite; invalid uncertainty requires abstention")
    return value.detach()


@torch.no_grad()
def build_gain_features(raw_dn, physics, *, template_postblur_e, expected_e,
                        continuum_postblur, sigma_S):
    """Build six fixed features from this raw and its detached physical fit.

    ``sigma_S`` is calculated using the frozen uncertainty procedure, never
    truth. All scales are fixed constants; no per-case tuning or sign labels
    are inputs. Raw counts are used once to standardize the diagnostic residual
    and are not counted as another likelihood term.
    """
    s = _field(template_postblur_e, "template_postblur_e")
    expected = _field(expected_e, "expected_e", s)
    sigma = _field(sigma_S, "sigma_S", s)
    if bool((sigma <= 0).any()):
        raise ValueError("sigma_S must be positive")
    c = continuum_postblur
    if (not isinstance(c, torch.Tensor) or c.shape != (len(s), 7, *s.shape[-2:])
            or c.device != s.device or not bool(torch.isfinite(c).all())):
        raise ValueError("continuum_postblur must be finite B,7,H,W on template device")
    raw_e = common.raw_electrons(physics, raw_dn)
    if raw_e.shape != s.shape:
        raise ValueError("raw_dn and physical template shapes differ")
    variance = raw_e.clamp_min(0) + float(physics.cfg["read_noise_e"])**2 + float(physics.adc_step)**2/12
    residual = ((raw_e - expected) / variance.sqrt().clamp_min(1e-6)).clamp(-12, 12) / 12
    magnitude = s.abs()
    gradient = ((magnitude - torch.roll(magnitude, 1, -1)).abs()
                + (magnitude - torch.roll(magnitude, 1, -2)).abs()) / 180
    values = (
        torch.log1p(magnitude / sigma.clamp_min(1e-6)).clamp_max(8)/8,
        torch.log1p(magnitude/180).clamp_max(8)/8,
        residual,
        gradient.clamp_max(4)/4,
        torch.log1p(c.detach().mean(1).clamp_min(0)/1000).clamp_max(8)/8,
        torch.log1p(sigma/180).clamp_max(8)/8,
    )
    features = torch.stack(values, dim=1).to(dtype=s.dtype).detach()
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite raw-only features")
    return features


class SignedBoundedGainDenoiser(nn.Module):
    """Fixed 6 -> 8 -> 8 -> 1 CNN; last head starts exactly at zero."""
    def __init__(self, width=WIDTH, gain_bound=GAIN_BOUND):
        super().__init__()
        if width != WIDTH or float(gain_bound) != GAIN_BOUND:
            raise ValueError("Only the preregistered width=8, gain_bound=.3 model is allowed")
        self.gain_bound = GAIN_BOUND
        self.body = nn.Sequential(
            nn.Conv2d(len(FEATURE_NAMES), WIDTH, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(WIDTH, 1, 3, padding=1, padding_mode="circular")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, template_postblur_e, features):
        s = _field(template_postblur_e, "template_postblur_e")
        if (not isinstance(features, torch.Tensor)
                or features.shape != (len(s), len(FEATURE_NAMES), *s.shape[-2:])
                or features.device != s.device or not bool(torch.isfinite(features).all())):
            raise ValueError("Expected finite B,6,H,W raw-only features")
        # Detach at this explicit boundary: gradients update only g_theta.
        z = features.detach().to(dtype=next(self.parameters()).dtype)
        bounded_gain = 1 + self.gain_bound * torch.tanh(self.head(self.body(z))[:, 0])
        out = s * bounded_gain.to(dtype=s.dtype)
        return out


def apply_denoiser(model, raw_dn, physics, physical, sigma_S):
    """Replace only postblur S, keeping every physical continuum field fixed.

    ``expected_e`` and ``template_latent_e`` continue to describe the original
    linear solution. A postblur-only CNN has no inverse-blur raw refit. The
    distinction is stated explicitly in diagnostics and used by MC.
    """
    if not isinstance(physical, dict) or set(physical) & FORBIDDEN_INPUT_KEYS:
        raise ValueError("Physical prediction cannot include truth, sign labels or identity")
    features = build_gain_features(
        raw_dn, physics, template_postblur_e=physical["template_postblur_e"],
        expected_e=physical["expected_e"], continuum_postblur=physical["continuum_postblur"],
        sigma_S=sigma_S)
    result = {key: (value.detach().clone() if isinstance(value, torch.Tensor)
                    else copy.deepcopy(value)) for key, value in physical.items()}
    s = physical["template_postblur_e"].detach()
    result["template_linear_postblur_e"] = s.clone()
    result["template_postblur_e"] = model(s, features)
    result["sigma_S"] = sigma_S.detach().clone()
    result["detection_sigma_proxy_e"] = sigma_S.detach().clone()
    result["sigma_S_source"] = "raw-derived uniform-W postblur proxy in the registered v5 experiment; not MC sigma"
    result["expected_e_source"] = "frozen linear physical measurement prediction"
    info = copy.deepcopy(physical.get("info", {}))
    info["denoiser"] = {
        "model": "SignedBoundedGainDenoiser", "width": WIDTH, "gain_bound": GAIN_BOUND,
        "feature_names": list(FEATURE_NAMES), "truth_or_identity_access": False,
        "continuum_trainable_state": False, "continuum_feedback": False,
        "zero_preserving": True, "per_pixel_linear_sign_preserved": True,
        "expected_e_source": "unaltered linear fit of the same original raw",
        "expected_e_is_modified_template_raw_refit": False,
        "latent_template_source": "unaltered linear solution; CNN output is postblur only",
        "modified_template_raw_fit_assessed": False,
        "feature_uncertainty_source": "frozen raw-derived uniform-W postblur proxy; not MC sigma",
    }
    result["info"] = info
    return result


def load_denoiser(checkpoint, device="cpu", *, expected_sha256=None):
    path = common.ed_path(checkpoint)
    if expected_sha256 is not None and common.sha(path) != expected_sha256:
        raise ValueError("Frozen neural checkpoint hash mismatch")
    payload = torch.load(path, map_location=device, weights_only=True)
    spec = payload.get("model_spec", {})
    if (spec.get("width") != WIDTH or spec.get("gain_bound") != GAIN_BOUND
            or spec.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Checkpoint is not the one preregistered bounded gain model")
    model = SignedBoundedGainDenoiser().to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload
