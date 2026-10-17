"""Filter-defined image reconstruction and separate radiometric readouts."""
import runtime
import math
import torch
from torch import nn
from research_filter_decode.operator import CodedOperator, filt


def gradient(x):
    return torch.stack((torch.roll(x, -1, -1)-x, torch.roll(x, -1, -2)-x), -3)


def gradient_adjoint(p):
    return torch.roll(p[..., 0, :, :], 1, -1)-p[..., 0, :, :] + torch.roll(p[..., 1, :, :], 1, -2)-p[..., 1, :, :]


class Imager(CodedOperator):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__(cfg, dtype)
        self.register_buffer("support", (self.object_otf > 0).to(dtype))
        h, w = cfg["shape"]
        fy, fx = torch.meshgrid(torch.fft.fftfreq(h, dtype=dtype), torch.fft.fftfreq(w, dtype=dtype), indexing="ij")
        self.register_buffer("laplacian", 4*(torch.sin(math.pi*fx).square()+torch.sin(math.pi*fy).square()))
        ac = []
        for (kx, ky), gain, depth in zip(self.bins, cfg["channel_gain"], cfg["modulation_depth"]):
            ac.append((gain*depth/4)**2*(torch.roll(self.relay_otf.square(), (ky, kx), (-2, -1)) + torch.roll(self.relay_otf.square(), (-ky, -kx), (-2, -1))))
        self.register_buffer("ac_normal", torch.stack(ac))
        self.register_buffer("dc_normal_vector", torch.tensor(cfg["channel_gain"], dtype=dtype)[:, None, None]*self.relay_otf/2)
        self.register_buffer("white_noise_gain", ((self.decode_transfers.abs().square()*self.support).sum((-2, -1))/(2*h*w)).sqrt())
        self.register_buffer("prior_mu", (self.ac_normal*self.support).sum((-2, -1))/self.support.sum()*0.5)

    def project(self, x):
        return filt(x, self.support)

    def normal(self, x, weight, mu, rho):
        return self.project(self.images_adjoint(weight*self.encode_images(x))) + mu*x + rho*gradient_adjoint(gradient(x))

    def precondition(self, residual, weight, mu, rho):
        mean_weight = weight.mean((-2, -1), keepdim=True).unsqueeze(-3)
        diagonal = mean_weight*self.ac_normal + mu + rho*self.laplacian
        u = mean_weight.sqrt()*self.dc_normal_vector
        r = torch.fft.fft2(residual)
        first = r/diagonal
        answer = first-u/diagonal*(u*first).sum(-3, keepdim=True)/(1+(u.square()/diagonal).sum(-3, keepdim=True))
        return torch.fft.ifft2(answer*self.support).real

    def solve(self, rhs, initial, weight, mu, rho=0.0, steps=10, tolerance=1e-5):
        x = initial
        r = rhs-self.normal(x, weight, mu, rho)
        q = self.precondition(r, weight, mu, rho)
        p = q
        rq = (r*q).sum((-3, -2, -1), keepdim=True)
        initial_norm = r.square().sum((-3, -2, -1), keepdim=True).sqrt()
        if float(initial_norm.detach().max()) == 0:
            return x
        for _ in range(steps):
            ap = self.normal(p, weight, mu, rho)
            alpha = rq/(p*ap).sum((-3, -2, -1), keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
            x = x+alpha*p
            r = r-alpha*ap
            if bool((r.square().sum((-3, -2, -1), keepdim=True).sqrt() <= tolerance*initial_norm).all()):
                break
            q = self.precondition(r, weight, mu, rho)
            new_rq = (r*q).sum((-3, -2, -1), keepdim=True)
            p = q+(new_rq/rq.clamp_min(torch.finfo(x.dtype).tiny))*p
            rq = new_rq
        return self.project(x)

    def initialize(self, raw_e):
        x = self.project(self.decode(raw_e))
        scale = x[..., 3:, :, :].mean((-3, -2, -1), keepdim=True)
        variance = self.encode_images(x).clamp_min(1)+self.cfg["read_noise_e"]**2 + (self.cfg["full_well_e"]/((2**self.cfg["adc_bits"]-1)-self.cfg["black_level_DN"]))**2/12
        weight = variance.mean((-2, -1), keepdim=True)/variance
        noise = variance.mean((-2, -1), keepdim=True).sqrt().unsqueeze(-3)*self.white_noise_gain[:, None, None]/scale
        return x/scale, raw_e/scale.squeeze(-3), scale, weight, noise.expand_as(x)

    def data_step(self, y, weight, proposal, steps=8):
        mu = self.prior_mu[:, None, None]
        rhs = self.project(self.images_adjoint(weight*y))+mu*proposal
        return self.solve(rhs, proposal, weight, mu, steps=steps)


def tv_reconstruct(op, y, weight, initial, strength, cfg, iterations=None):
    x = initial.clone()
    d = gradient(x)
    u = torch.zeros_like(d)
    rhs_data = op.project(op.images_adjoint(weight*y))
    rho = cfg["rho"]
    history = []
    limit = cfg["maximum_iterations"] if iterations is None else iterations
    for step in range(limit):
        before = x
        rhs = rhs_data+rho*gradient_adjoint(d-u)
        x = op.solve(rhs, x, weight, 0.0, rho, cfg["pcg_iterations"], cfg["pcg_tolerance"])
        dx = gradient(x)
        v = dx+u
        d = v*(1-strength/(rho*v.square().sum(-3, keepdim=True).sqrt().clamp_min(torch.finfo(x.dtype).tiny))).clamp_min(0)
        u = u+dx-d
        if (step+1) % 40 == 0 or step+1 == limit:
            change = float((x-before).norm()/x.norm())
            primal = float((dx-d).norm()/dx.norm())
            stationarity = op.project(op.images_adjoint(weight*(op.encode_images(x)-y))+rho*gradient_adjoint(u))
            objective = float((0.5*(weight*(op.encode_images(x)-y).square()).sum()+strength*dx.square().sum(-3).sqrt().sum())/x.shape[0])
            history.append({"iteration": step+1, "relative_change": change, "primal_relative_residual": primal, "stationarity_rms": float(stationarity.square().mean().sqrt()), "objective": objective})
            if iterations is None and change < cfg["relative_iteration_change_target"] and primal < 0.001:
                break
    return x, history


class LearnedReconstruction(nn.Module):
    def __init__(self, cfg, use_data_consistency):
        super().__init__()
        width = cfg["hidden_channels"]
        self.net = nn.Sequential(nn.Conv2d(18, width, 3, padding=1, padding_mode="circular"), nn.GELU(), nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"), nn.GELU(), nn.Conv2d(width, 6, 3, padding=1, padding_mode="circular"))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.stages = cfg["shared_stages"]
        self.use_dc = use_data_consistency

    def forward(self, op, initial, y, weight, noise):
        x = initial
        for _ in range(self.stages):
            x = op.project(x+0.2*torch.tanh(self.net(torch.cat((x, initial, noise), 1))))
            if self.use_dc:
                x = op.data_step(y, weight, x)
        return x


def training_loss(pred, target):
    means = target.mean((-2, -1), keepdim=True)
    image = ((pred-target)/means).square().mean()
    bias = ((pred.mean((-2, -1), keepdim=True)-means)/means).square().mean()
    p = pred[:, 3:].clamp_min(1e-4).log()
    t = target[:, 3:].log()
    ratios = ((p-p[:, 1:2])-(t-t[:, 1:2])).square().mean()
    return image+0.5*bias+5*ratios


class Radiometry:
    def __init__(self, protocol, device, dtype=torch.float32):
        center = torch.tensor(protocol["channel_centers_nm"], device=device, dtype=dtype)
        width = torch.tensor(protocol["channel_widths_nm"], device=device, dtype=dtype)
        self.wavelength_um = (center[:, None]+width[:, None]*torch.linspace(-0.5, 0.5, 9, device=device, dtype=dtype))/1000
        self.quadrature = torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 0.5], device=device, dtype=dtype)[None, :]*width[:, None]/8000
        wl = self.wavelength_um[4]
        self.normalization = (wl.pow(-4)/torch.expm1(14387.76877/(wl*3000))*self.quadrature[4]).sum()

    def curve(self, temperature):
        wl = self.wavelength_um[:, :, None, None]
        t = temperature[None, None]
        u = 14387.76877/(wl*t)
        em1 = torch.expm1(u)
        b = wl.pow(-4)/em1
        q = self.quadrature[:, :, None, None]/self.normalization
        c = (b*q).sum(1)
        derivative = (b*(u/t)*(1+1/em1)*q).sum(1)
        return c, derivative

    def fit(self, images):
        measured = images[3:]
        t = torch.full_like(images[0], 3000)
        for _ in range(10):
            all_c, all_d = self.curve(t)
            c, derivative = all_c[3:], all_d[3:]
            amplitude = ((measured*c).sum(0)/c.square().sum(0)).clamp_min(0)
            j = amplitude*(derivative-c*(c*derivative).sum(0)/c.square().sum(0))
            delta = (j*(measured-amplitude*c)).sum(0)/j.square().sum(0).clamp_min(torch.finfo(images.dtype).tiny)
            t = (t+delta).clamp(1800, 4500)
        c, _ = self.curve(t)
        amplitude = ((measured*c[3:]).sum(0)/c[3:].square().sum(0)).clamp_min(0)
        continuum = amplitude*c
        residual = (measured-continuum[3:]).square().sum(0).sqrt()/measured.square().sum(0).sqrt().clamp_min(torch.finfo(images.dtype).tiny)
        return t, continuum, images[:3]-continuum[:3], residual
