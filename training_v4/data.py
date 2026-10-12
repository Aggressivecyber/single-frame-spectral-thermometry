"""New independent mother scenes; scene transforms precede the fixed encoder."""
import copy
import subprocess
import numpy as np
import torch
import runtime
from infer import load_model, DEFAULT_CHECKPOINT
from evaluation_v2.common import array_hash, canonical_hash
from training_v4.common import OUT, DATA, setup, read, write, save_npz
from training_v4.protocol import PROTOCOL, BASE

TEMP_TYPES = PROTOCOL['temperature_geometries']
SPECIES_TYPES = PROTOCOL['template_geometries']
CONDITIONS = PROTOCOL['conditions']


def describe(split, index):
    base = {'train': 100000, 'validation': 200000, 'final_unopened': 300000}[split]
    seed = base+index*113
    return {'mother_id': f'v4_{split}_mother_{index:04d}', 'split': split, 'index': index,
            'seed': seed, 'temperature_seed': seed+1, 'amplitude_seed': seed+29,
            'template_seed': seed+53, 'background_seed': seed+71,
            'temperature_geometry': TEMP_TYPES[index % len(TEMP_TYPES)],
            'template_geometry': SPECIES_TYPES[(index//len(TEMP_TYPES)) % len(SPECIES_TYPES)],
            'raw_noise_seed_base': seed+10000000}


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


def generate():
    setup()
    if (OUT/'data_manifest.json').exists():
        manifest = read(OUT/'data_manifest.json')
        assert manifest['generator_source_sha256'] == runtime.sha256(__file__)
        print({'data_reused': True, 'raws': len(manifest['rows'])}, flush=True)
        return manifest
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime.ROOT).decode().strip() == BASE
    p = load_model(device='cuda').physics
    cfg = copy.deepcopy(p.cfg)
    labels, rows, mothers = [], [], []
    with torch.no_grad():
        for split, count in [('train', 256), ('validation', 64)]:
            for index in range(count):
                desc = describe(split, index)
                state, beta, shape, provenance = mother_state(p, desc)
                mothers.append({**desc, **provenance})
                level_rng = np.random.default_rng(desc['template_seed']+301)
                peaks = [0., level_rng.uniform(25, 55), level_rng.uniform(140, 280), level_rng.uniform(320, 520)]
                for ci, (condition, peak) in enumerate(zip(CONDITIONS, peaks)):
                    z = state.clone()
                    z[:, 4] = shape*peak/p.Sscale
                    continuum, species, background, _ = p.components(z, beta)
                    bands = p.op.object_blur(continuum+species+background)
                    color, domain = p.color_temperature(p.op.object_blur(continuum))
                    expected = p.op.encode_images(bands)[0]
                    template = p.op.object_blur(z[:, 4]*p.Sscale)[0]
                    assert torch.isfinite(expected).all() and expected.min() >= 0 and expected.max() < 58000
                    label_path = DATA/'labels'/desc['mother_id']/(condition+'.npz')
                    save_npz(label_path, template_e=template.cpu().numpy(),
                             temperature_color_K=color[0].cpu().numpy(),
                             band_images_e=bands[0].cpu().numpy(), expected_e=expected.cpu().numpy(),
                             evaluation_domain=domain[0].cpu().numpy())
                    label_hash = runtime.sha256(label_path)
                    labels.append({'mother_id': desc['mother_id'], 'condition': condition,
                        'path': str(label_path.relative_to(DATA)).replace('\\', '/'), 'sha256': label_hash,
                        'latent_template_peak_e': float(peak), 'truth_source': 'true synthetic forward fields'})
                    for noise_index in range(2):
                        seed = desc['raw_noise_seed_base']+ci*17+noise_index
                        generator = torch.Generator(device=p.M.device).manual_seed(seed)
                        measured = torch.poisson(expected, generator=generator)+torch.randn(expected.shape, device=p.M.device, generator=generator)*p.cfg['read_noise_e']
                        raw = (measured/p.adc_step+p.cfg['black_level_DN']).round().clamp(0, 4095)
                        assert not (raw >= 4095).any(), 'Retain and diagnose any saturated generated frame'
                        raw_array = raw.to(torch.uint16).cpu().numpy()
                        case = f'{desc["mother_id"]}_{condition}_noise{noise_index}'
                        path = DATA/'raw'/desc['mother_id']/(condition+f'_noise{noise_index}.npz')
                        save_npz(path, raw_dn=raw_array)
                        rows.append({'case_id': case, 'mother_id': desc['mother_id'], 'split': split,
                            'condition': condition, 'noise_index': noise_index, 'noise_seed': seed,
                            'clean_identity': False, 'raw_path': str(path.relative_to(DATA)).replace('\\', '/'),
                            'raw_hash': array_hash(raw_array), 'raw_file_sha256': runtime.sha256(path),
                            'label_path': str(label_path.relative_to(DATA)).replace('\\', '/'), 'label_sha256': label_hash,
                            'physics_config_sha256': canonical_hash(cfg), 'calibration_sha256': canonical_hash(cfg),
                            'shape': cfg['shape']})
                    if ci == index % 4:
                        clean = (expected/p.adc_step+p.cfg['black_level_DN']).cpu().numpy()
                        path = DATA/'raw'/desc['mother_id']/(condition+'_clean_identity.npz')
                        save_npz(path, raw_dn=clean)
                        rows.append({'case_id': desc['mother_id']+'_'+condition+'_clean_identity',
                            'mother_id': desc['mother_id'], 'split': split, 'condition': condition,
                            'noise_index': None, 'noise_seed': None, 'clean_identity': True,
                            'raw_path': str(path.relative_to(DATA)).replace('\\', '/'), 'raw_hash': array_hash(clean),
                            'raw_file_sha256': runtime.sha256(path),
                            'label_path': str(label_path.relative_to(DATA)).replace('\\', '/'), 'label_sha256': label_hash,
                            'physics_config_sha256': canonical_hash(cfg), 'calibration_sha256': canonical_hash(cfg),
                            'shape': cfg['shape'], 'measurement_note': 'exact electron-equivalent floating DN, no noise or ADC quantization'})
                if index % 16 == 15:
                    print({'generated_split': split, 'mothers_completed': index+1}, flush=True)
    final = [describe('final_unopened', i) for i in range(64)]
    all_ids = [m['mother_id'] for m in mothers+final]
    assert len(set(all_ids)) == 384
    manifest = {'schema': PROTOCOL['schema'], 'generator_source_sha256': runtime.sha256(__file__),
        'physics_config': cfg, 'physics_config_sha256': canonical_hash(cfg),
        'old_checkpoint_sha256': runtime.sha256(DEFAULT_CHECKPOINT),
        'train_mothers': 256, 'validation_mothers': 64,
        'train_noisy_raw': 2048, 'validation_noisy_raw': 512,
        'train_clean_identity': 256, 'validation_clean_identity': 64,
        'all_rows': 2880, 'mothers': mothers, 'labels': labels, 'rows': rows,
        'final_mothers_registered_but_not_generated': final,
        'final_test_used_for_tuning_or_selection': False,
        'existing_expanded_audit_repair_used_for_training': False}
    assert len(rows) == 2880
    write(OUT/'data_manifest.json', manifest)
    write(OUT/'final_test_registration.json', {'parents': final, 'parent_hash': canonical_hash(final),
        'generator_sha256': runtime.sha256(__file__), 'physics_config_sha256': canonical_hash(cfg),
        'raw_generated': 0, 'evaluations': 0, 'opened_for_selection': False})
    write(OUT/'protocol.json', PROTOCOL)
    print({'generation_complete': True, 'raw_rows': len(rows), 'final_unopened': 64}, flush=True)
    return manifest
