"""Five physical fields, six shared stages, two bounded CNN residual heads."""
import runtime
import torch
from torch import nn
from config import validate_config
from physics import SpectralPhysics
from solver import data_step


class ParameterUnfoldedNet(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        validate_config(cfg)
        self.cfg = cfg
        self.physics = SpectralPhysics(cfg, dtype)
        self.trunk = nn.Sequential(nn.Conv2d(19, 16, 3, padding=1, padding_mode="circular"), nn.SiLU(), nn.Conv2d(16, 16, 3, padding=1, padding_mode="circular"), nn.SiLU())
        self.continuum_head = nn.Conv2d(16, 2, 3, padding=1, padding_mode="circular")
        self.emission_head = nn.Conv2d(16, 3, 3, padding=1, padding_mode="circular")
        self.trunk.to(dtype=dtype)
        self.continuum_head.to(dtype=dtype)
        self.emission_head.to(dtype=dtype)
        for head in (self.continuum_head, self.emission_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("correction_bounds", torch.tensor(cfg["parameter_correction_bounds"], dtype=dtype)[None, :, None, None])

    def features(self, raw_e, state, beta, stage):
        expectation = self.physics(state, beta)
        variance = expectation.detach().clamp_min(0) + self.physics.extra_variance
        weights = variance.reciprocal()
        residual = weights * (expectation-raw_e)
        g, _ = self.physics.vjp(state, beta, residual)
        diagonal, _ = self.physics.preconditioner_diagonal(state, beta, weights)
        gradient = (g/diagonal).clamp(-5, 5)
        bands = self.physics.op.images_adjoint(residual)
        bands = bands / bands.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        noise = variance.sqrt() / self.cfg["full_well_e"]**.5
        index = torch.full_like(raw_e[:, None], stage/(self.cfg["steps"]-1))
        return torch.cat([state, gradient, bands.clamp(-5, 5), noise[:, None], index], 1)

    def infer_electrons(self, raw_e, use_prior=True):
        state, beta = self.physics.initialize(raw_e)
        for stage in range(self.cfg["steps"]):
            state, beta = data_step(self.physics, raw_e, state, beta)
            if use_prior:
                shared = self.trunk(self.features(raw_e, state, beta, stage))
                correction = torch.cat([self.continuum_head(shared), self.emission_head(shared)], 1).tanh() * self.correction_bounds
                state, beta = self.physics.constrain(state+correction, beta)
            state, beta = data_step(self.physics, raw_e, state, beta)
        continuum, species, background, _ = self.physics.components(state, beta)
        band_images = self.physics.op.object_blur(continuum+species+background)
        continuum_images = self.physics.op.object_blur(continuum)
        temperature, valid = self.physics.color_temperature(continuum_images)
        prediction = self.physics.op.encode_images(band_images)
        valid = valid & torch.isfinite(prediction) & (prediction >= 0) & (band_images.amin(1) >= -1e-3)
        signed_species = self.physics.op.object_blur(state[:, 2:] * self.physics.Sscale * self.physics.active[None, :, None, None])
        return {"state": state, "background_beta": beta, "band_images_e": band_images, "continuum_images_e": continuum_images, "temperature_color_K": temperature, "species_signed_e": signed_species, "expected_e": prediction, "reprojection_residual_e": prediction-raw_e, "valid": valid}

    def forward(self, raw_dn, use_prior=True):
        output = self.infer_electrons(self.physics.raw_to_electrons(raw_dn), use_prior=use_prior)
        raw = raw_dn[:, 0] if raw_dn.ndim == 4 else raw_dn
        saturated = raw >= 2**self.cfg["adc_bits"]-1
        output["saturated"] = saturated
        output["valid"] = output["valid"] & ~saturated
        return output


def loss_function(model, raw_e, output, truth):
    """Truth enters supervision only, never the reconstruction."""
    variance = output["expected_e"].detach().clamp_min(0)+model.physics.extra_variance
    data = (output["reprojection_residual_e"].square()/variance).mean()
    temperature = ((output["temperature_color_K"]-truth["temperature_color_K"])/75).square().mean()
    species_error = (output["species_signed_e"]-truth["species_signed_e"])/60
    active = model.physics.active[None, :, None, None]
    species = (species_error.square()*active).sum() / (active.sum().clamp_min(1)*species_error.shape[0]*species_error.shape[2]*species_error.shape[3])
    band = ((output["band_images_e"]-truth["band_images_e"])/300).square().mean()
    null = (truth["species_signed_e"].abs() < 1e-8).to(raw_e.dtype) * active
    false_positive = (torch.relu(output["species_signed_e"]-60).square()/60**2 * null).sum()/null.sum().clamp_min(1)
    background = output["background_beta"].square().mean()
    total = species + .2*temperature + .2*band + .01*data + false_positive + 1e-5*background
    return total, {"species": species.detach(), "temperature": temperature.detach(), "bands": band.detach(), "data": data.detach(), "null_penalty": false_positive.detach()}

