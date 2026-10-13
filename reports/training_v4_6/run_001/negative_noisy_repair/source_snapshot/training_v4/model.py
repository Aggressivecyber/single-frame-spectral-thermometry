"""Tensor-only post-inverse CNN. LM/PCG are outside the differentiable graph."""
import runtime
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from physics import SpectralPhysics
from config import validate_config


def physical_output(p, state, beta, raw_e):
    continuum, species, background, _ = p.components(state, beta)
    bands = p.op.object_blur(continuum+species+background)
    continuous_bands = p.op.object_blur(continuum)
    color, valid = p.color_temperature(continuous_bands)
    expected = p.op.encode_images(bands)
    signed = p.op.object_blur(state[:, 2:]*p.Sscale*p.active[None, :, None, None])
    return {'state': state, 'beta': beta, 'temperature_color_K': color,
            'template_e': signed[:, 2], 'species_signed_e': signed,
            'band_images_e': bands, 'continuum_images_e': continuous_bands,
            'expected_e': expected, 'raw_residual_e': expected-raw_e,
            'valid': valid & torch.isfinite(expected) & (expected >= 0) & (bands.amin(1) >= -1e-3)}


class PhysicsPostInverseCNN(nn.Module):
    """物理反演后CNN校正器: 19->16->16, six shared bounded residual corrections."""
    def __init__(self, cfg, *, activation_checkpointing=True):
        super().__init__()
        validate_config(cfg)
        self.cfg, self.activation_checkpointing = cfg, activation_checkpointing
        self.physics = SpectralPhysics(cfg)
        self.trunk = nn.Sequential(nn.Conv2d(19, 16, 3, padding=1, padding_mode='circular'), nn.SiLU(),
                                   nn.Conv2d(16, 16, 3, padding=1, padding_mode='circular'), nn.SiLU())
        self.continuum_head = nn.Conv2d(16, 2, 3, padding=1, padding_mode='circular')
        self.emission_head = nn.Conv2d(16, 3, 3, padding=1, padding_mode='circular')
        for head in (self.continuum_head, self.emission_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer('correction_bounds', torch.tensor(cfg['parameter_correction_bounds'])[None, :, None, None])
        assert sum(p.numel() for p in self.parameters()) == 5797

    def features(self, raw_e, state, beta, stage):
        p = self.physics
        expected = p(state, beta)
        variance = expected.detach().clamp_min(0)+p.extra_variance
        weights = variance.reciprocal()
        residual = weights*(expected-raw_e)
        gradient, _ = p.vjp(state, beta, residual)
        diagonal, _ = p.preconditioner_diagonal(state, beta, weights)
        gradient = (gradient/diagonal).clamp(-5, 5)
        bands = p.op.images_adjoint(residual)
        # Clamp before sqrt: identical values to the old normalization, with a
        # finite derivative at an exactly clean zero residual.
        bands = bands/bands.square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        noise = variance.sqrt()/self.cfg['full_well_e']**.5
        index = torch.full_like(raw_e[:, None], stage/5)
        return torch.cat([state, gradient, bands.clamp(-5, 5), noise[:, None], index], 1)

    def correction(self, raw_e, state, beta, stage):
        shared = self.trunk(self.features(raw_e, state, beta, stage))
        delta = torch.cat([self.continuum_head(shared), self.emission_head(shared)], 1).tanh()*self.correction_bounds
        return self.physics.constrain(state+delta, beta)[0]

    def forward(self, raw_dn, cached_state, cached_beta):
        raw_e = self.physics.raw_to_electrons(raw_dn)
        return self.forward_electrons(raw_e, cached_state, cached_beta)

    def forward_electrons(self, raw_e, cached_state, cached_beta):
        # No optimizer path through the cached inverse or any LM/PCG iteration.
        state, beta = cached_state.detach(), cached_beta.detach()
        for stage in range(6):
            if self.training and self.activation_checkpointing:
                state = checkpoint(lambda z, k=stage: self.correction(raw_e, z, beta, k), state, use_reentrant=False)
            else:
                state = self.correction(raw_e, state, beta, stage)
        return physical_output(self.physics, state, beta, raw_e)

    def cnn_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if k.startswith(('trunk.', 'continuum_head.', 'emission_head.'))}

    def load_cnn(self, state):
        own = self.state_dict()
        for name, value in state.items():
            if name.startswith(('trunk.', 'continuum_head.', 'emission_head.')):
                own[name].copy_(value)


def supervised_loss(model, raw_dn, output, truth):
    """Fixed absolute scales, signed blank error, true postblur labels only."""
    raw_e = model.physics.raw_to_electrons(raw_dn)
    template = ((output['template_e']-truth['template_e'])/60).square().mean()
    color = ((output['temperature_color_K']-truth['temperature_color_K'])/75).square().mean()
    bands = ((output['band_images_e']-truth['band_images_e'])/300).square().mean()
    raw = ((output['expected_e']-raw_e).square()/(raw_e.clamp_min(0)+model.physics.extra_variance)).mean()
    is_blank = truth['template_e'].square().sum((-2, -1)) < 1e-8
    blank = ((output['template_e']/60).square().mean((-2, -1))*is_blank).mean()
    loss = template+.2*color+.2*bands+.01*raw+blank
    return loss, {'template': template, 'color': color, 'bands': bands, 'raw': raw, 'signed_blank': blank}
