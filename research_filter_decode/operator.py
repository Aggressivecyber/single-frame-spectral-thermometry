"""Frozen spatial operator copied from physics.py; no spectral parameter model."""
import runtime
import math
import torch
from torch import nn


def filt(image, otf):
    return torch.fft.ifft2(torch.fft.fft2(image) * otf).real


class CodedOperator(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        h, w = cfg["shape"]
        fy, fx = torch.meshgrid(torch.fft.fftfreq(h, dtype=dtype), torch.fft.fftfreq(w, dtype=dtype), indexing="ij")
        radius = (fx.square() + fy.square()).sqrt()
        nu = (radius / cfg["object_cutoff_cpp"]).clamp(0, 1)
        object_otf = 2 / math.pi * (torch.acos(nu) - nu * (1 - nu.square()).clamp_min(0).sqrt())
        relay = torch.exp(-2 * math.pi ** 2 * cfg["relay_psf_sigma_px"] ** 2 * radius.square())
        yy, xx = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
        patterns = []
        self.bins = cfg["carrier_bins_xy"]
        for (kx, ky), gain, depth, phase in zip(self.bins, cfg["channel_gain"], cfg["modulation_depth"], cfg["phase_rad"]):
            patterns.append(gain * 0.5 * (1 + depth * torch.cos(2 * math.pi * (kx * xx / w + ky * yy / h) + phase)))
        self.register_buffer("patterns", torch.stack(patterns))
        self.register_buffer("object_otf", object_otf)
        self.register_buffer("relay_otf", relay)
        self.register_buffer("decode_window", (radius <= cfg["decode_cutoff_cpp"]).to(dtype))
        transfers = []
        for (kx, ky), gain, depth, phase in zip(self.bins, cfg["channel_gain"], cfg["modulation_depth"], cfg["phase_rad"]):
            coefficient = gain * depth / 4 * complex(math.cos(phase), math.sin(phase))
            transfers.append(self.decode_window / (coefficient * torch.roll(relay, (-ky, -kx), (-2, -1))))
        self.register_buffer("decode_transfers", torch.stack(transfers))
        self.register_buffer("pattern_energy", self.patterns.square().mean((-2, -1)))
        self.validate_geometry()

    def validate_geometry(self):
        h, w = self.cfg["shape"]
        centers = [(0., 0.)]
        for kx, ky in self.bins:
            centers += [(kx / w, ky / h), (-kx / w, -ky / h)]
        distances = []
        for i, a in enumerate(centers):
            for b in centers[i+1:]:
                dx, dy = abs(a[0]-b[0]) % 1, abs(a[1]-b[1]) % 1
                distances.append(math.hypot(min(dx, 1-dx), min(dy, 1-dy)))
        separation = min(distances)
        if separation <= 2 * self.cfg["decode_cutoff_cpp"]:
            raise ValueError("DC/carrier windows collide on the sampled frequency torus.")
        self.geometry = {"minimum_center_separation_cpp": separation, "required_separation_cpp": 2*self.cfg["decode_cutoff_cpp"], "carrier_count": len(self.bins), "includes_conjugates_and_DC": True}

    def object_blur(self, radiance):
        return filt(radiance, self.object_otf)

    def encode_images(self, band_images):
        return filt((band_images * self.patterns).sum(-3), self.relay_otf)

    def forward(self, radiance):
        return self.encode_images(self.object_blur(radiance))

    def images_adjoint(self, residual):
        return filt(residual, self.relay_otf).unsqueeze(-3) * self.patterns

    def adjoint(self, residual):
        return self.object_blur(self.images_adjoint(residual))

    def decode(self, raw_e):
        spectrum = torch.fft.fft2(raw_e)
        return torch.stack([torch.fft.ifft2(torch.roll(spectrum, (-ky, -kx), (-2, -1)) * self.decode_transfers[k]).real for k, (kx, ky) in enumerate(self.bins)], dim=-3)

