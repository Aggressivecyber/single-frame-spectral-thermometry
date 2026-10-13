"""Band-integrated radiometry, exact coded forward/adjoint and Jacobians."""
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
        self.geometry = {"minimum_center_separation_cpp": separation, "required_separation_cpp": 2*self.cfg["decode_cutoff_cpp"], "carrier_count": 7, "includes_conjugates_and_DC": True}

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


class SpectralPhysics(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.op = CodedOperator(cfg, dtype)
        temperatures = torch.arange(cfg["temperature_range_K"][0], cfg["temperature_range_K"][1]+cfg["temperature_lut_step_K"]*.5, cfg["temperature_lut_step_K"], dtype=dtype)
        fluxes = []
        for k, (center, width, qe) in enumerate(zip(cfg["centers_nm"], cfg["widths_nm"], cfg["quantum_efficiencies"])):
            sampled = cfg.get("sampled_instrument_response")
            if sampled is None:
                lam = torch.linspace(center-width/2, center+width/2, 81, dtype=dtype) * 1e-9
                response = torch.full_like(lam, qe)
            else:
                lam = torch.tensor(sampled["wavelength_nm"], dtype=dtype) * 1e-9
                response = torch.tensor(sampled["response"][k], dtype=dtype)
                response = response * torch.tensor(sampled.get("fixed_emissivity_ratio", [1.0]*len(lam)), dtype=dtype)
            exponent = 0.01438776877 / (temperatures[:, None] * lam)
            photon_flux = 2 * 299792458.0 / lam.pow(4) / torch.expm1(exponent) * response
            fluxes.append(torch.trapezoid(photon_flux, lam, dim=-1))
        counts = torch.stack(fluxes, -1)
        index = round((cfg["reference_temperature_K"]-float(temperatures[0]))/cfg["temperature_lut_step_K"])
        scale = cfg.get("absolute_radiometric_scale", cfg["reference_905_e"] / counts[index, -1])
        counts = counts * scale
        self.register_buffer("temperatures", temperatures)
        self.register_buffer("continuum_lut", counts)
        self.register_buffer("color_ratio_lut", counts[:, -2]/counts[:, -1])
        self.register_buffer("M", torch.tensor(cfg["template_response"], dtype=dtype))
        self.register_buffer("q", torch.tensor(cfg["background_response"], dtype=dtype))
        h, w = cfg["shape"]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, dtype=dtype), torch.linspace(-1, 1, w, dtype=dtype), indexing="ij")
        self.register_buffer("background_basis", torch.stack([torch.ones_like(xx), xx, yy]))
        self.register_buffer("active", torch.tensor(cfg["active_species"], dtype=dtype))
        self.T0, self.Tscale = cfg["temperature_center_K"], cfg["temperature_scale_K"]
        self.Sscale, self.Bscale = cfg["species_scale_e"], cfg["background_scale_e"]
        self.adc_step = cfg["full_well_e"] / (2**cfg["adc_bits"]-1-cfg["black_level_DN"])
        self.extra_variance = cfg["read_noise_e"]**2 + self.adc_step**2/12

    def continuum(self, temperature):
        grid = (temperature-self.temperatures[0]) / self.cfg["temperature_lut_step_K"]
        index = grid.floor().long().clamp(0, len(self.temperatures)-2)
        fraction = (grid-index).clamp(0, 1)
        low, high = self.continuum_lut[index], self.continuum_lut[index+1]
        values = low + fraction.unsqueeze(-1) * (high-low)
        slope = (high-low) / self.cfg["temperature_lut_step_K"]
        return values.movedim(-1, -3), slope.movedim(-1, -3)

    def components(self, state, beta):
        T = self.T0 + self.Tscale * state[:, 0]
        C, dC = self.continuum(T)
        a = state[:, 1:2].exp()
        continuum = a * C
        species = self.Sscale * torch.einsum("kj,bjhw->bkhw", self.M, state[:, 2:] * self.active[None, :, None, None])
        bgmap = self.Bscale * torch.einsum("bl,lhw->bhw", beta, self.background_basis)
        background = self.q[None, :, None, None] * bgmap[:, None]
        return continuum, species, background, a * dC * self.Tscale

    def radiance(self, state, beta):
        continuum, species, background, _ = self.components(state, beta)
        return continuum + species + background

    def forward(self, state, beta):
        return self.op(self.radiance(state, beta))

    def spectral_jacobian(self, state, beta):
        continuum, _, _, temperature_derivative = self.components(state, beta)
        h, w = state.shape[-2:]
        species_derivative = (self.Sscale * self.M * self.active[None]).T[:, :, None, None].expand(3, 7, h, w)
        return torch.cat([temperature_derivative[:, None], continuum[:, None], species_derivative[None].expand(len(state), -1, -1, -1, -1)], dim=1)

    def jvp(self, state, beta, direction, beta_direction):
        jac = self.spectral_jacobian(state, beta)
        bands = (jac * direction[:, :, None]).sum(1)
        bg = self.Bscale * torch.einsum("bl,lhw->bhw", beta_direction, self.background_basis)
        return self.op(bands + self.q[None, :, None, None] * bg[:, None])

    def vjp(self, state, beta, residual):
        back = self.op.adjoint(residual)
        gradient = (self.spectral_jacobian(state, beta) * back[:, None]).sum(2)
        bgmap = (back * self.q[None, :, None, None]).sum(1)
        beta_gradient = self.Bscale * torch.einsum("bhw,lhw->bl", bgmap, self.background_basis)
        return gradient, beta_gradient

    def preconditioner_diagonal(self, state, beta, weights):
        # Positive scaling only: not an exact coded normal inverse.
        wm = weights.mean((-2, -1))
        jac = self.spectral_jacobian(state, beta)
        diagonal = wm[:, None, None, None] * (jac.square() * self.op.pattern_energy[None, None, :, None, None]).sum(2)
        qb = (self.q.square() * self.op.pattern_energy).sum() * self.Bscale**2
        beta_diag = wm[:, None] * qb * self.background_basis.square().sum((-2, -1))[None]
        return diagonal.clamp_min(1e-7), beta_diag.clamp_min(1e-7)

    def constrain(self, state, beta):
        limits = self.cfg["temperature_range_K"]
        T = state[:, :1].clamp((limits[0]-self.T0)/self.Tscale, (limits[1]-self.T0)/self.Tscale)
        loga = state[:, 1:2].clamp(-4, 0.3)
        signed_species = state[:, 2:].clamp(-20, 20) * self.active[None, :, None, None]
        return torch.cat([T, loga, signed_species], 1), beta.clamp(-20, 20)

    def raw_to_electrons(self, raw_dn):
        if raw_dn.ndim == 4 and raw_dn.shape[1] == 1:
            raw_dn = raw_dn[:, 0]
        if raw_dn.ndim != 3 or tuple(raw_dn.shape[-2:]) != tuple(self.cfg["shape"]):
            raise ValueError("Raw input must be B x 1 x H x W or B x H x W.")
        return (raw_dn-self.cfg["black_level_DN"]) * self.adc_step

    def color_temperature(self, continuum_images):
        denominator = continuum_images[:, -1]
        ratio = continuum_images[:, -2] / denominator.clamp_min(1e-6)
        index = torch.searchsorted(self.color_ratio_lut, ratio.contiguous()).clamp(1, len(self.temperatures)-1)
        low, high = self.color_ratio_lut[index-1], self.color_ratio_lut[index]
        fraction = ((ratio-low)/(high-low)).clamp(0, 1)
        T = self.temperatures[index-1] + fraction * (self.temperatures[index]-self.temperatures[index-1])
        valid = (denominator > 1) & (ratio >= self.color_ratio_lut[0]) & (ratio <= self.color_ratio_lut[-1])
        return T, valid

    def initialize(self, raw_e):
        decoded = self.op.decode(raw_e)
        T, _ = self.color_temperature(decoded)
        C, _ = self.continuum(T)
        # Scaled band fit is an initializer only, not the inverse data term.
        Cnorm = C.square().sum(1, keepdim=True).sqrt().clamp_min(1)
        M = self.M * self.active[None] * self.Sscale
        design = torch.cat([(C/Cnorm)[:, :, None], M[None, :, :, None, None].expand(len(T), -1, -1, *T.shape[-2:]), (self.q*self.Bscale)[None, :, None, None, None].expand(len(T), -1, -1, *T.shape[-2:])], dim=2)
        design = design.permute(0, 3, 4, 1, 2)
        normal = design.transpose(-2, -1) @ design
        diagonal = normal.diagonal(dim1=-2, dim2=-1).clamp_min(1)
        normal = normal + torch.diag_embed(diagonal * 1e-5)
        target = design.transpose(-2, -1) @ decoded.permute(0, 2, 3, 1).unsqueeze(-1)
        fitted = torch.linalg.solve(normal, target).squeeze(-1).movedim(-1, 1)
        amplitude = (fitted[:, :1]/Cnorm).clamp(0.03, 1.3)
        bg = fitted[:, -1]
        flat_basis = self.background_basis.flatten(1)
        beta = torch.linalg.solve(flat_basis @ flat_basis.T, (bg.flatten(1) @ flat_basis.T).T).T
        state = torch.cat([((T-self.T0)/self.Tscale)[:, None], amplitude.log(), fitted[:, 1:4]], 1)
        return self.constrain(state, beta)

