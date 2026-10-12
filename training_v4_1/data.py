"""Complete factorial protection inputs and independent signed mothers."""
import copy
import numpy as np
import torch
import runtime
from infer import load_model
from training_v4.data import mother_state, TEMP_TYPES, SPECIES_TYPES
from training_v4_1.common import *

CONDITIONS = ['blank', 'weak', 'normal', 'strong']

def select_mothers(mothers, split, n):
    selected = []
    global_counts = {kind: 0 for kind in SPECIES_TYPES}
    for geometry in TEMP_TYPES:
        candidates = [m for m in mothers if m['split'] == split and m['temperature_geometry'] == geometry]
        counts = {kind: 0 for kind in SPECIES_TYPES}
        for _ in range(n//6):
            candidate = min((m for m in candidates if m not in selected),
                            key=lambda m: (counts[m['template_geometry']], global_counts[m['template_geometry']], m['index']))
            selected.append(candidate); counts[candidate['template_geometry']] += 1
            global_counts[candidate['template_geometry']] += 1
    return selected

def imported(row, mothers):
    m = mothers[row['mother_id']]
    return {**row, 'raw_path': rel(OLD_DATA/row['raw_path']), 'label_path': rel(OLD_DATA/row['label_path']),
            'input_kind': 'noise', 'group': row['condition'], 'geometry': m['temperature_geometry'],
            'template_geometry': m['template_geometry'], 'signed_aux': False, 'origin': 'v4_exact_reuse'}

def append_raw(rows, p, label_path, mother, condition, raw, kind, *, group=None, signed=False):
    path = DATA/'raw'/mother['mother_id']/(condition+'_'+kind+'.npz')
    save_npz(path, raw_dn=raw)
    rows.append({'case_id': mother['mother_id']+'_'+condition+'_'+kind, 'mother_id': mother['mother_id'],
        'split': mother['split'], 'condition': condition, 'group': group or condition,
        'geometry': mother['temperature_geometry'], 'template_geometry': mother['template_geometry'],
        'input_kind': kind, 'signed_aux': signed, 'origin': 'v4_1_protection' if not signed else 'independent_signed',
        'raw_path': rel(path), 'raw_hash': array_hash(raw), 'raw_file_sha256': runtime.sha256(path),
        'label_path': rel(label_path), 'label_sha256': runtime.sha256(label_path),
        'physics_config_sha256': canonical_hash(p.cfg), 'calibration_sha256': canonical_hash(p.cfg)})

def coverage(rows, old):
    matrix = []
    for split in ('train', 'validation'):
        for geometry in TEMP_TYPES:
            for condition in CONDITIONS:
                previous = sum(r['split'] == split and r['clean_identity'] and r['condition'] == condition
                    and old['mothers'][int(r['mother_id'].rsplit('_', 1)[1])+(256 if split == 'validation' else 0)]['temperature_geometry'] == geometry
                    for r in old['rows'])
                for kind in ('continuous', 'quantization'):
                    count = sum(r['split'] == split and r['geometry'] == geometry and r['condition'] == condition
                                and r['input_kind'] == kind and not r['signed_aux'] for r in rows)
                    assert count == (8 if split == 'train' else 4)
                    matrix.append({'split': split, 'geometry': geometry, 'condition': condition,
                        'input_kind': kind, 'new_count': count, 'v4_continuous_count': previous})
    return matrix

def generate():
    setup()
    if (OUT/'data_manifest.json').exists():
        return read(OUT/'data_manifest.json')
    old = read(OLD_OUT/'data_manifest.json')
    mothers = {m['mother_id']: m for m in old['mothers']}
    rows = [imported(r, mothers) for r in old['rows'] if not r['clean_identity']]
    selected = select_mothers(old['mothers'], 'train', 48)+select_mothers(old['mothers'], 'validation', 24)
    p = load_model(device='cuda').physics
    signed_mothers, legal = [], []
    with torch.no_grad():
        for mother in selected:
            for condition in CONDITIONS:
                label = OLD_DATA/'labels'/mother['mother_id']/(condition+'.npz')
                expected = torch.as_tensor(npz(label)['expected_e'], device='cuda')
                dn = expected/p.adc_step+p.cfg['black_level_DN']
                append_raw(rows, p, label, mother, condition, dn.cpu().numpy(), 'continuous')
                quant = dn.round().clamp(0, 4095)
                assert not (quant >= 4095).any()
                quant = quant.to(torch.uint16)
                append_raw(rows, p, label, mother, condition, quant.cpu().numpy(), 'quantization')
        for split, n, seed_base in (('train', 12, 400000), ('validation', 6, 500000)):
            for i in range(n):
                seed = seed_base+i*113
                mother = {'mother_id': f'v4_1_signed_{split}_mother_{i:04d}', 'split': split, 'index': i,
                    'seed': seed, 'temperature_seed': seed+1, 'amplitude_seed': seed+29,
                    'template_seed': seed+53, 'background_seed': seed+71,
                    'temperature_geometry': TEMP_TYPES[i % 6], 'template_geometry': SPECIES_TYPES[(i//6+i) % 6]}
                z, beta, shape, provenance = mother_state(p, mother)
                z[:, 0] += 250/p.Tscale
                amplitude_scale = 1.
                while p(z, beta).max() >= 52000:
                    z[:, 1] += np.log(.8); amplitude_scale *= .8
                # One shared amplitude scale for matched +/- coefficients. Its
                # only purpose is legal synthetic expectations before inference.
                scale = 1.
                for attempt in range(24):
                    valid = True
                    for sign in (-1, 1):
                        zz = z.clone(); zz[:, 4] = shape*180*scale*sign/p.Sscale
                        c, s, b, _ = p.components(zz, beta)
                        bands = p.op.object_blur(c+s+b); expected = p.op.encode_images(bands)
                        valid &= bool((bands.min() >= 0) & (expected.min() >= 0) & (expected.max() < 58000))
                    if valid: break
                    scale *= .8
                assert valid, 'Retain and diagnose any physically illegal signed mother'
                signed_mothers.append({**mother, **provenance, 'temperature_shift_K': 250.,
                    'continuum_amplitude_scale': amplitude_scale, 'legal_amplitude_scale': scale})
                for sign in (-1, 1):
                    for strength, peak in (('weak', 45.), ('normal', 180.)):
                        condition = ('negative_' if sign < 0 else 'positive_')+strength
                        zz = z.clone(); zz[:, 4] = shape*peak*scale*sign/p.Sscale
                        c, s, b, _ = p.components(zz, beta)
                        bands = p.op.object_blur(c+s+b); expected = p.op.encode_images(bands)[0]
                        T, domain = p.color_temperature(p.op.object_blur(c))
                        label = DATA/'labels'/mother['mother_id']/(condition+'.npz')
                        save_npz(label, template_e=p.op.object_blur(zz[:, 4]*p.Sscale)[0].cpu().numpy(),
                            temperature_color_K=T[0].cpu().numpy(), band_images_e=bands[0].cpu().numpy(),
                            expected_e=expected.cpu().numpy(), evaluation_domain=domain[0].cpu().numpy())
                        dn = expected/p.adc_step+p.cfg['black_level_DN']
                        group = 'negative' if sign < 0 else strength
                        append_raw(rows, p, label, mother, condition, dn.cpu().numpy(), 'continuous', group=group, signed=True)
                        append_raw(rows, p, label, mother, condition, dn.round().to(torch.uint16).cpu().numpy(), 'quantization', group=group, signed=True)
                        g = torch.Generator(device='cuda').manual_seed(seed+10000000+int(sign > 0)*71+int(strength == 'normal')*17)
                        noisy = torch.poisson(expected, generator=g)+torch.randn(expected.shape, generator=g, device='cuda')*p.cfg['read_noise_e']
                        noisy_dn = (noisy/p.adc_step+p.cfg['black_level_DN']).round().clamp(0, 4095)
                        assert not (noisy_dn >= 4095).any()
                        noisy_dn = noisy_dn.to(torch.uint16)
                        append_raw(rows, p, label, mother, condition, noisy_dn.cpu().numpy(), 'noise', group=group, signed=True)
                        legal.append({'mother_id': mother['mother_id'], 'condition': condition,
                            'actual_preblur_peak_e': sign*peak*scale, 'minimum_expected_e': float(expected.min()),
                            'maximum_expected_e': float(expected.max()), 'minimum_band_e': float(bands.min())})
    matrix = coverage(rows, old)
    parents = {s: {r['mother_id'] for r in rows if r['split'] == s} for s in ('train', 'validation')}
    assert not parents['train'] & parents['validation']
    assert all(not r['mother_id'].startswith('v4_final') for r in rows)
    manifest = {'base': BASE, 'physics_config': p.cfg, 'rows': rows, 'selected_mothers': selected,
        'signed_mothers': signed_mothers, 'generator_sha256': runtime.sha256(__file__),
        'v4_manifest_sha256': runtime.sha256(OLD_OUT/'data_manifest.json'), 'final_test_opened': False,
        'counts': {'old_noisy_reused': 2560, 'factorial_protection': 576, 'signed_aux': 216}}
    write(OUT/'data_manifest.json', manifest); write(OUT/'coverage_matrix.json', matrix)
    write(OUT/'signed_expectation_legality.json', legal); write(OUT/'protocol.json', PROTOCOL)
    print({'data_complete': True, 'rows': len(rows), 'required_coverage_cells': len(matrix), 'all_nonempty': True}, flush=True)
    return manifest
