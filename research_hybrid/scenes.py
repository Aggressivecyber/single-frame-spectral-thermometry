"""Scene-field functions copied from training_v4; new seed namespace below."""
import hashlib
import numpy as np
import torch
TEMP_TYPES = ['front','multipeak','ring','stripe','flat','smooth_random']

def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()

def field(p, seed, kind, *, positive=False):
    r = np.random.default_rng(seed)
    xx, yy = p.background_basis[1], p.background_basis[2]
    angle, scale = r.uniform(0, 2*np.pi), r.uniform(.65, 1.4)
    x = (xx-r.uniform(-.35, .35))/scale
    y = (yy-r.uniform(-.35, .35))/scale
    u, v = np.cos(angle)*x+np.sin(angle)*y, -np.sin(angle)*x+np.cos(angle)*y
    if kind == 'front':
        f = torch.tanh((u-.13*torch.sin(r.uniform(2, 6)*v))/r.uniform(.035, .20))
    elif kind in ('multipeak', 'blob'):
        f = torch.zeros_like(xx)
        for _ in range(int(r.integers(3, 7)) if kind == 'multipeak' else 1):
            cx, cy, width, ratio = r.uniform(-.65, .65), r.uniform(-.65, .65), r.uniform(.045, .30), r.uniform(.5, 1.5)
            f += r.uniform(.4, 1)*torch.exp(-.5*((u-cx).square()+((v-cy)/ratio).square())/width**2)
    elif kind == 'ring':
        radius = (u.square()+v.square()).sqrt()
        f = torch.exp(-.5*((radius-r.uniform(.2, .65))/r.uniform(.035, .13)).square())
    elif kind in ('stripe', 'filament'):
        f = torch.exp(-.5*((v-.13*torch.sin(r.uniform(2, 7)*u))/r.uniform(.02, .13)).square())
        if kind == 'stripe':
            f += .7*torch.exp(-.5*((v-r.uniform(.25, .6))/r.uniform(.025, .1)).square())
    elif kind == 'flat':
        return torch.zeros_like(xx)
    else:
        g = torch.Generator(device=p.M.device).manual_seed(seed)
        noise = torch.randn(tuple(p.cfg['shape']), device=p.M.device, generator=g)
        fy, fx = torch.meshgrid(torch.fft.fftfreq(p.cfg['shape'][0], device=p.M.device),
                                torch.fft.fftfreq(p.cfg['shape'][1], device=p.M.device), indexing='ij')
        transfer = torch.exp(-2*np.pi**2*r.uniform(6., 24.)**2*(fx.square()+fy.square()))
        f = torch.fft.ifft2(torch.fft.fft2(noise)*transfer).real
    normalized = (f-f.min())/(f.max()-f.min()).clamp_min(1e-8)
    return normalized if positive else 2*normalized-1


def mother_state(p, desc):
    rt = np.random.default_rng(desc['temperature_seed'])
    ra = np.random.default_rng(desc['amplitude_seed'])
    rb = np.random.default_rng(desc['background_seed'])
    temp_field = field(p, desc['temperature_seed'], desc['temperature_geometry'])
    temperature = rt.uniform(2750, 3050)+rt.uniform(40, 270)*temp_field
    amp_field = field(p, desc['amplitude_seed'], TEMP_TYPES[int(ra.integers(0, len(TEMP_TYPES)))])
    amplitude = ra.uniform(.25, .75)*(1+ra.uniform(0, .2)*amp_field)
    shape = field(p, desc['template_seed'], desc['template_geometry'], positive=True)
    beta = torch.tensor([[rb.uniform(.2, .9), rb.uniform(-.09, .09), rb.uniform(-.09, .09)]],
                        dtype=p.M.dtype, device=p.M.device)
    state = torch.zeros((1, 5, *p.cfg['shape']), dtype=p.M.dtype, device=p.M.device)
    state[:, 0], state[:, 1] = (temperature-p.T0)/p.Tscale, amplitude.log()
    return state, beta, shape, {'temperature_field_sha256': array_hash(temperature.cpu().numpy()),
        'amplitude_field_sha256': array_hash(amplitude.cpu().numpy()),
        'template_shape_sha256': array_hash(shape.cpu().numpy()),
        'streams_independent': True, 'scene_transform_before_encoding': True}

