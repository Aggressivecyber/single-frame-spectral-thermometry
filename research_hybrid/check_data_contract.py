"""Server preflight for every registered scene, before any new full inverse."""
import runtime
import torch
import numpy as np
from .run_pilot import load_physics, SPEC, REPORT
from .scenes import mother_state

torch.set_num_threads(1)
_, p, _ = load_physics('cpu')
geometries = ['front', 'multipeak', 'ring', 'stripe', 'flat', 'smooth_random']
templates = ['blob', 'multipeak', 'ring', 'stripe', 'filament', 'smooth_random']
rows = []
with torch.no_grad():
    for index in range(SPEC['train_mothers']+SPEC['validation_mothers']):
        seed = SPEC['seed_base']+index*1009
        desc = {'temperature_seed': seed+1, 'amplitude_seed': seed+29,
                'background_seed': seed+71, 'template_seed': seed+53,
                'temperature_geometry': geometries[index % 6], 'template_geometry': templates[index % 6]}
        state, beta, shape, _ = mother_state(p, desc)
        for condition, peak in enumerate(SPEC['signed_peaks_e']):
            for variant in range(SPEC['continuum_variants']):
                z = state.clone()
                if variant:
                    z[:, 1] += np.log(.75)
                z[:, 4] = peak*shape/p.Sscale
                radiance = p.radiance(z, beta)
                expected = p.op(radiance)
                row = {'mother_index': index, 'peak_e': peak, 'continuum_variant': variant,
                       'minimum_band_photon_mean_e': float(radiance.min()),
                       'minimum_raw_photon_mean_e': float(expected.min()),
                       'maximum_raw_photon_mean_e': float(expected.max())}
                row['passed'] = bool(torch.isfinite(radiance).all() and torch.isfinite(expected).all()
                                     and radiance.min() >= 0 and expected.min() >= 0 and expected.max() < 58000)
                rows.append(row)
receipt = {'time': runtime.now(), 'scope': 'CPU forward checks of all registered scenes on the server',
           'experimental_inverse_calls': 0, 'rows': rows,
           'all_passed': len(rows) == SPEC['raw_frames'] and all(r['passed'] for r in rows),
           'minimum_band_photon_mean_e': min(r['minimum_band_photon_mean_e'] for r in rows),
           'maximum_raw_photon_mean_e': max(r['maximum_raw_photon_mean_e'] for r in rows)}
runtime.write_json(REPORT/'DATA_CONTRACTS.json', receipt)
print({k: v for k, v in receipt.items() if k != 'rows'}, flush=True)
if not receipt['all_passed']:
    raise ArithmeticError('Registered scene photon-positivity contract failed')
