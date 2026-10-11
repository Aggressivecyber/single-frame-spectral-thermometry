"""Disjoint synthetic software fixtures; no measured flame labels."""
import runtime
import torch

SPLITS = {"audit": 404, "train": 101, "validation": 202, "test": 303}


def make_scene(physics, split="audit", scene_id=0, noisy=True, null=False, flat=False, band_background_error_fraction=0.):
    device, dtype = physics.M.device, physics.M.dtype
    gen = torch.Generator(device=device).manual_seed(611401 + SPLITS[split]*100000 + scene_id)
    def rand():
        return torch.rand((), generator=gen, device=device, dtype=dtype)
    yy, xx = physics.background_basis[2], physics.background_basis[1]
    T = torch.full_like(xx, 2950.35)
    amplitude = torch.full_like(xx, .72)
    if not flat:
        T = T + 120*torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy) + 15*(rand()-.5)
        amplitude = amplitude + .08*torch.sin(torch.pi*xx)
    state = torch.zeros(1, 5, *xx.shape, dtype=dtype, device=device)
    state[:, 0] = (T-physics.T0)/physics.Tscale
    state[:, 1] = amplitude.log()
    for j in range(3):
        if physics.active[j] and not null:
            pattern = torch.ones_like(xx) if flat else torch.exp(-((xx-(rand()-.5)*.8)**2+(yy-(rand()-.5)*.8)**2)/(.15+.1*rand()))
            state[:, j+2] = (180+60*rand())/physics.Sscale * pattern
    beta = torch.tensor([[.45, .06, -.04]], device=device, dtype=dtype)
    if flat:
        beta[:, 1:] = 0
    continuum, species, background, _ = physics.components(state, beta)
    background_extra = torch.zeros_like(continuum)
    background_extra[:, 2] = band_background_error_fraction * continuum[:, 2]
    expected = physics.op(continuum+species+background+background_extra)
    if expected.min() < 0:
        raise ValueError("Fixture contains a negative expected photon count.")
    raw_e = expected.clone()
    if noisy:
        raw_e = torch.poisson(expected, generator=gen) + torch.randn(expected.shape, generator=gen, device=device, dtype=dtype)*physics.cfg["read_noise_e"]
    raw_dn = (raw_e/physics.adc_step + physics.cfg["black_level_DN"]).round().clamp(0, 2**physics.cfg["adc_bits"]-1)
    continuum_images = physics.op.object_blur(continuum)
    color, valid = physics.color_temperature(continuum_images)
    truth = {"state": state, "background_beta": beta, "band_images_e": physics.op.object_blur(continuum+species+background+background_extra), "temperature_color_K": color, "species_signed_e": physics.op.object_blur(state[:, 2:]*physics.Sscale*physics.active[None, :, None, None]), "valid": valid}
    return {"raw_dn": raw_dn[:, None], "raw_e_exact": raw_e, "expected_e": expected, "truth": truth, "split": split, "scene_id": scene_id, "band_background_error_fraction": band_background_error_fraction, "scope": "synthetic_engineering_fixture"}

