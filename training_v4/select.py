"""One bounded warm-omission decision and one three-value regularizer selection."""
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import runtime
from infer import load_model
from repair_v3.solver import DEFAULTS
from training_v4.common import OUT, DATA, setup, read, write, npz, save_npz
from training_v4.cache import cache_row
from training_v4.scalar_engine import invert_single
from training_v4.model import physical_output
from evaluation_v2.common import Journal, array_hash, canonical_hash
from evaluation_v2.methods import demod_fit

LOCK = threading.Lock()


def score_tensor(p, raw_dn, output, label):
    def a(t):
        return t.detach().cpu().numpy()[0].astype(np.float64)
    S, T, B = a(output['template_e']), a(output['temperature_color_K']), a(output['band_images_e'])
    error_s, error_t = S-label['template_e'], T-label['temperature_color_K']
    energy = float(np.square(label['template_e'], dtype=np.float64).sum())
    template = float(np.square(error_s).mean()/60**2)
    color = float(np.square(error_t).mean()/75**2)
    bands = float(np.square(B-label['band_images_e']).mean()/300**2)
    raw_e = p.raw_to_electrons(raw_dn)
    raw = float(((output['expected_e']-raw_e).square()/(raw_e.clamp_min(0)+p.extra_variance)).mean())
    return {'template_RMSE_e': float(np.sqrt(np.square(error_s).mean())),
        'color_RMSE_K': float(np.sqrt(np.square(error_t).mean())),
        'template_relative_L2': float(np.sqrt(np.square(error_s).sum()/energy)) if energy else None,
        'blank_FP60': float(np.mean(S > 60)) if energy == 0 else None,
        'supervised_loss': template+.2*color+.2*bands+.01*raw+(template if energy == 0 else 0.),
        'template_SSE': float(np.square(error_s).sum()), 'template_truth_energy': energy,
        'color_SSE': float(np.square(error_t).sum()), 'pixels': S.size}


def physics_validation(p, rows, method, config, journal):
    existing = {r['identity']: r for r in journal.rows()}
    def work(row):
        identity = canonical_hash({'raw': row['raw_hash'], 'method': method, 'config': config})
        if identity in existing and existing[identity].get('status') == 'SUCCESS':
            return existing[identity]
        try:
            state, beta, cache = cache_row(p, row, method, config)
            raw = torch.as_tensor(npz(DATA/row['raw_path'])['raw_dn'].astype(np.float32), device=p.M.device)[None]
            z, b = torch.as_tensor(state, device=p.M.device)[None], torch.as_tensor(beta, device=p.M.device)[None]
            with torch.no_grad():
                out = physical_output(p, z, b, p.raw_to_electrons(raw))
                metrics = score_tensor(p, raw, out, npz(DATA/row['label_path']))
            result = {'identity': identity, 'case_id': row['case_id'], 'raw_hash': row['raw_hash'],
                      'method': method, 'lambda_template': config['regularization'][4],
                      'metrics': metrics, 'cache_key': cache['cache_key'], 'status': 'SUCCESS'}
        except Exception as e:
            result = {'identity': identity, 'case_id': row['case_id'], 'status': 'EXCEPTION',
                      'exception_type': type(e).__name__, 'message': str(e)}
        with LOCK:
            journal.append(result)
            print({'validation_cached': row['case_id'], 'method': method,
                   'lambda': config['regularization'][4], 'status': result['status']}, flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(work, rows))


def development_gate(p, method, config):
    root = runtime.ROOT/'outputs/repair_v3/run_001'
    fixtures = [f for f in read(runtime.ROOT/'reports/repair_v3/run_001/frozen_manifest.json')['cases'] if f['suite'] == 'gate']
    journal = Journal(OUT/'backbone_gate_per_case.jsonl')
    old = {r['identity']: r for r in journal.rows()}
    def work(f):
        identity = canonical_hash({'fixture': f['file_sha256'], 'method': method, 'config': config})
        if identity in old and old[identity]['status'] == 'SUCCESS':
            return old[identity]
        arr = npz(root/f['path'])
        continuous = f['metadata']['variant'] == 'continuous'
        observed = arr['expected_e'] if continuous else arr['raw_dn']
        raw = torch.as_tensor(observed, device=p.M.device)[None]
        raw_e = raw if continuous else p.raw_to_electrons(raw)
        try:
            z, b, history, seconds = invert_single(p, raw_e, method, config)
            with torch.no_grad():
                out = physical_output(p, z, b, raw_e)
                S, T = out['template_e'][0].cpu().numpy(), out['temperature_color_K'][0].cpu().numpy()
            error_s, error_t = S-arr['template_e'], T-arr['temperature_observed_color_K']
            energy = float(np.square(arr['template_e'], dtype=np.float64).sum())
            rmse_s, rmse_t = float(np.sqrt(np.square(error_s, dtype=np.float64).mean())), float(np.sqrt(np.square(error_t, dtype=np.float64).mean()))
            l2 = float(np.sqrt(np.square(error_s, dtype=np.float64).sum()/energy)) if energy else None
            fp = float(np.mean(S > 60)) if not energy else None
            passed = rmse_t <= (1 if continuous else 3) and rmse_s <= (5 if continuous else 6) and (l2 <= .25 if energy else fp == 0)
            key = DATA/'gate_cache'/method/(str(config['regularization'][4])+'/'+f['case_id']+'.npz')
            save_npz(key, state=z[0].cpu().numpy(), beta=b[0].cpu().numpy(), raw_e=raw_e[0].cpu().numpy())
            result = {'identity': identity, 'case_id': f['case_id'], 'status': 'SUCCESS',
                'method': method, 'config': config, 'input_hash': array_hash(observed),
                'metrics': {'color_RMSE_K': rmse_t, 'template_RMSE_e': rmse_s,
                            'template_relative_L2': l2, 'blank_FP60': fp},
                'gate_passed': passed, 'cache_path': str(key.relative_to(DATA)).replace('\\', '/'),
                'cache_sha256': runtime.sha256(key), 'seconds': seconds}
        except Exception as e:
            result = {'identity': identity, 'case_id': f['case_id'], 'status': 'EXCEPTION',
                'message': str(e), 'gate_passed': False, 'method': method, 'config': config}
        with LOCK:
            journal.append(result)
            print({'gate': f['case_id'], 'method': method, 'lambda': config['regularization'][4],
                   'passed': result['gate_passed']}, flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(work, fixtures))
    return {'passed': all(r['gate_passed'] for r in rows), 'frames': len(rows), 'rows': rows}


def run_selection():
    setup()
    if (OUT/'backbone_selection.json').exists():
        print({'selection_reused': True}, flush=True)
        return read(OUT/'backbone_selection.json')
    manifest = read(OUT/'data_manifest.json')
    p = load_model(device='cuda').physics
    old_method, combined = 'old_init__joint_update_v3', 'consistent_init__joint_update_v3'
    gate = development_gate(p, old_method, DEFAULTS)
    rows = [r for r in manifest['rows'] if r['split'] == 'validation' and not r['clean_identity']
            and int(r['mother_id'].rsplit('_', 1)[-1]) in (0, 7, 16, 23)]
    assert len(rows) == 32
    journal = Journal(OUT/'backbone_validation_per_case.jsonl')
    measurements = {m: physics_validation(p, rows, m, DEFAULTS, journal) for m in (combined, old_method)}
    def value(results):
        return float(np.mean([r['metrics']['supervised_loss'] for r in results])) if all(r['status'] == 'SUCCESS' for r in results) else float('inf')
    omit = gate['passed'] and value(measurements[old_method]) <= value(measurements[combined])*1.05
    method = old_method if omit else combined
    candidate_results = []
    for regularization in (1e-3, 1e-2, 1e-1):
        config = copy.deepcopy(DEFAULTS)
        config['regularization'][4] = regularization
        result = physics_validation(p, rows, method, config, journal)
        candidate_results.append({'lambda': regularization, 'validation_loss': value(result), 'config': config})
    candidates = sorted(candidate_results, key=lambda r: r['validation_loss'])
    chosen, selected_gate = None, None
    for candidate in candidates:
        selected_gate = development_gate(p, method, candidate['config'])
        if selected_gate['passed']:
            chosen = candidate
            break
    if chosen is None:
        raise RuntimeError('All finite regularization choices broke required gates; preserve diagnostics')
    # Traditional baseline has the same bounded validation-only three-point grid.
    traditional = []
    with torch.no_grad():
        for lam in (1e-3, 1e-2, 1e-1):
            losses = []
            for row in rows:
                raw = torch.as_tensor(npz(DATA/row['raw_path'])['raw_dn'].astype(np.float32), device=p.M.device)[None]
                out = demod_fit(p, p.raw_to_electrons(raw), regularization=lam)
                label = npz(DATA/row['label_path'])
                pred = out['species_signed_e'][0, 2].cpu().numpy()
                losses.append(float(np.square(pred-label['template_e'], dtype=np.float64).mean()/60**2))
            traditional.append({'regularization': lam, 'validation_template_MSE_normalized': float(np.mean(losses))})
    selection = {'method': method, 'solver_config': chosen['config'], 'physics_config': manifest['physics_config'],
        'warm_start_omitted': omit, 'old_init_gate_passed': gate['passed'],
        'validation_small_subset_cases': [r['case_id'] for r in rows],
        'warm_comparison_loss': {m: value(rs) for m, rs in measurements.items()},
        'finite_template_regularization_grid': candidate_results,
        'selected_lambda': chosen['lambda'], 'selected_gate_passed': selected_gate['passed'],
        'traditional_candidates': traditional,
        'traditional_selected': min(traditional, key=lambda r: r['validation_template_MSE_normalized'])['regularization'],
        'truth_only_used_for_validation_scoring': True, 'final_test_used': False,
        'selection_rule': 'validation loss with fixed deterministic-gate admissibility; no solver search beyond the registered three values'}
    write(OUT/'backbone_selection.json', selection)
    print({'selection_completed': True, 'method': method, 'lambda': chosen['lambda']}, flush=True)
    return selection
