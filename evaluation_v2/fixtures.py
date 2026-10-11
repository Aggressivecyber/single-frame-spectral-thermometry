"""Freeze truth separately from raw-only inference; preserve parent and noise identities."""
import copy
from pathlib import Path
import numpy as np
import torch
import runtime
from physics import SpectralPhysics
from .common import array_hash, canonical_hash
from .physics_v2 import adc_quantize
from .protocol import PROTOCOL


def geometry(physics, family, parent_index):
    seed = PROTOCOL["geometry_seed_base"] + PROTOCOL["families"].index(family)*1000+parent_index
    rng = np.random.default_rng(seed)
    x, y = physics.background_basis[1].cpu().numpy(), physics.background_basis[2].cpu().numpy()
    cx, cy = rng.uniform(-.45, .45, 2)
    sigma = rng.uniform(.23, .37)
    def gaussian(x0, y0, width):
        return np.exp(-.5*((x-x0)**2+(y-y0)**2)/width**2)
    pattern = gaussian(cx, cy, sigma)
    if family == "multiple_peaks":
        for _ in range(2):
            pattern += rng.uniform(.4, 1)*gaussian(*rng.uniform(-.6, .6, 2), rng.uniform(.12, .23))
    if family == "thin_filament":
        angle = rng.uniform(-np.pi, np.pi)
        u = np.cos(angle)*(x-cx)+np.sin(angle)*(y-cy)
        v = -np.sin(angle)*(x-cx)+np.cos(angle)*(y-cy)
        pattern = np.exp(-.5*((u/.6)**2+((v-.14*np.sin(5*u))/rng.uniform(.035, .06))**2))
    pattern = pattern/pattern.max()
    # Temperature, amplitude, emission and structured background use independent
    # child streams. Their coordinates are not deliberately made coincident.
    trng = np.random.default_rng(seed+1000000)
    arng = np.random.default_rng(seed+2000000)
    brng = np.random.default_rng(seed+3000000)
    tx, ty = trng.uniform(-.5, .5, 2)
    T = trng.uniform(2800, 3020)+trng.uniform(100, 240)*np.cos(np.pi*(x-tx))*np.cos(np.pi*(y-ty))
    if family == "temperature_front":
        angle = trng.uniform(-.8, .8)
        coordinate = np.cos(angle)*x+np.sin(angle)*y-trng.uniform(-.3, .3)
        T = trng.uniform(2870, 2930)+trng.uniform(250, 330)*np.tanh(12*coordinate)
    amplitude = arng.uniform(.43, .68)+arng.uniform(.03, .1)*np.sin(np.pi*(x-arng.uniform(-.5, .5)))
    bg = gaussian(*brng.uniform(-.5, .5, 2), brng.uniform(.2, .5))
    return {"temperature": T.astype(np.float32), "amplitude": amplitude.astype(np.float32),
            "pattern": pattern.astype(np.float32), "background_pattern": bg.astype(np.float32),
            "geometry_seed": seed, "parent_scene_id": f"{family}_parent{parent_index}"}


def make_frame(nominal, fields, condition, *, family, noise_seed, gain_error=0.,
               continuum_multiplier=None, relative_mismatch=0., absolute_mismatch_e=0.,
               normalization="fixed_peak", target_budget=None):
    cfg = copy.deepcopy(nominal.cfg)
    if gain_error:
        cfg["channel_gain"] = [1+gain_error*(-1 if k % 2 else 1) for k in range(7)]
    acquisition = SpectralPhysics(cfg, nominal.M.dtype).to(nominal.M.device)
    device, dtype = nominal.M.device, nominal.M.dtype
    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    state = torch.zeros((1, 5, *cfg["shape"]), dtype=dtype, device=device)
    state[:, 0] = (tensor(fields["temperature"])-acquisition.T0)/acquisition.Tscale
    state[:, 1] = tensor(fields["amplitude"]).log()
    pattern = tensor(fields["pattern"])
    peak = PROTOCOL["weak_preblur_template_peak_e"] if condition == "signal_weak" else PROTOCOL["nominal_preblur_template_peak_e"]
    emission = pattern*peak if condition.startswith("signal") else torch.zeros_like(pattern)
    post = acquisition.op.object_blur(emission)
    if target_budget is not None and emission.abs().sum() > 0:
        denominator = post.sum() if normalization == "fixed_integral" else post.square().sum().sqrt()
        emission = emission*(target_budget/float(denominator))
    state[:, 4] = emission/acquisition.Sscale
    beta = torch.tensor([[.45, .06, -.04]], device=device, dtype=dtype)
    continuum, species, background, _ = acquisition.components(state, beta)
    reference_continuum = continuum.clone()
    multipliers = continuum_multiplier or [1.]*7
    continuum = continuum*tensor(multipliers)[None, :, None, None]
    extra = torch.zeros_like(continuum)
    extra[:, 2] = relative_mismatch*continuum[:, 2]+absolute_mismatch_e
    if condition == "null_structured_background":
        extra += acquisition.q[None, :, None, None]*tensor(fields["background_pattern"])[None, None]*100
    expected = acquisition.op(continuum+species+background+extra)
    if not torch.isfinite(expected).all() or expected.min() < 0:
        raise ValueError("Invalid expected count")
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    noisy_e = torch.poisson(expected, generator=gen)+torch.randn(expected.shape, generator=gen, device=device, dtype=dtype)*cfg["read_noise_e"]
    raw = adc_quantize(noisy_e, acquisition)
    ref, _ = nominal.color_temperature(acquisition.op.object_blur(reference_continuum))
    obs, truth_valid = nominal.color_temperature(acquisition.op.object_blur(continuum))
    target = acquisition.op.object_blur(emission)[None]
    latent = tensor(fields["temperature"])[None]
    uniform = torch.ones_like(truth_valid) if float(latent.max()-latent.min()) < 1e-4 else torch.zeros_like(truth_valid)
    arrays = {"raw_dn": raw[0].cpu().numpy(), "expected_e": expected[0].cpu().numpy(),
              "temperature_latent_K": latent[0].cpu().numpy(), "temperature_reference_color_K": ref[0].cpu().numpy(),
              "temperature_observed_color_K": obs[0].cpu().numpy(), "evaluation_domain": truth_valid[0].cpu().numpy(),
              "uniform_temperature_domain": uniform[0].cpu().numpy(), "template_e": target[0].cpu().numpy(),
              "saturated": (raw[0] >= 4095).cpu().numpy()}
    metadata = {"family": family, "condition": condition, "parent_scene_id": fields["parent_scene_id"],
                "geometry_seed": fields["geometry_seed"], "noise_seed": noise_seed,
                "nominal_config": nominal.cfg, "true_config": cfg, "continuum_multiplier": multipliers,
                "relative_486_mismatch": relative_mismatch, "absolute_486_mismatch_e": absolute_mismatch_e,
                "actual_mean_486_extra_e": float(extra[:, 2].mean()),
                "normalization_control": normalization, "target_budget": target_budget,
                "actual_raw_expected_mean_e": float(expected.mean()), "actual_raw_expected_max_e": float(expected.max()),
                "actual_band_mean_e": (acquisition.op.object_blur(continuum+species+background+extra).mean((0, 2, 3))).cpu().tolist(),
                "actual_template_peak_preblur_e": float(emission.max()), "actual_template_peak_postblur_e": float(target.max()),
                "actual_template_integral_e": float(target.sum()), "actual_template_L2_energy_e2": float(target.square().sum()),
                "temperature_min_K": float(latent.min()), "temperature_max_K": float(latent.max()),
                "amplitude_min": float(tensor(fields["amplitude"]).min()), "amplitude_max": float(tensor(fields["amplitude"]).max()),
                "scope": "new constructed audit frame; no measured thermodynamic temperature"}
    return arrays, metadata


def freeze_file(output, case_id, arrays, metadata, suite):
    path = Path(output)/"frozen"/suite/(case_id+".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Never replace a frozen fixture: "+str(path))
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as check:
        assert all(np.array_equal(check[k], v, equal_nan=True) for k, v in arrays.items())
    return {"case_id": case_id, "suite": suite, "path": str(path.relative_to(output)).replace("\\", "/"),
            "file_sha256": runtime.sha256(path), "raw_array_sha256": array_hash(arrays["raw_dn"]),
            "array_sha256": {k: array_hash(v) for k, v in arrays.items()}, "metadata": metadata}
