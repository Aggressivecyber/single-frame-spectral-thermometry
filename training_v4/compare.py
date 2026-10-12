"""Same-input six-method comparison; zero-template L2, blanks and failures retained."""
import copy
import time
import numpy as np
import torch
import runtime
from infer import load_model, DEFAULT_CHECKPOINT
from evaluation_v2.methods import demod_fit
from evaluation_v2.common import Journal, canonical_hash
from training_v4.common import OUT, DATA, setup, read, write, npz, save_npz
from training_v4.model import PhysicsPostInverseCNN, physical_output
from training_v4.cache import load_tensors

METHODS = ('zero_template', 'old_B4', 'traditional_validation_selected',
           'repair_physics_validation_selected', 'old_CNN_transfer', 'trained_CNN')


def metrics(target, template, temperature):
    S, y = np.asarray(template, np.float64), np.asarray(target['template_e'], np.float64)
    error = S-y
    energy = float(np.square(y).sum())
    color_error = None if temperature is None else np.asarray(temperature, np.float64)-target['temperature_color_K']
    support = y >= .2*float(y.max()) if energy else np.zeros_like(y, bool)
    positive = S > 60
    return {'template_SSE': float(np.square(error).sum()), 'template_truth_energy': energy,
        'pixels': S.size, 'template_RMSE_e': float(np.sqrt(np.square(error).mean())),
        'template_relative_L2': float(np.sqrt(np.square(error).sum()/energy)) if energy else None,
        'amplitude_projection': float(np.sum(S*y)/energy) if energy else None,
        'color_SSE': float(np.square(color_error).sum()) if color_error is not None else None,
        'color_RMSE_K': float(np.sqrt(np.square(color_error).mean())) if color_error is not None else None,
        'blank_FP60': float(positive.mean()) if not energy else None,
        'frame_score_99pct_e': float(np.quantile(S, .99)),
        'blank_frame_FP60': bool(np.quantile(S, .99) > 60) if not energy else None,
        'recall_support20_at60': float(positive[support].mean()) if support.any() else None,
        'nonfinite_pixels': int(np.sum(~np.isfinite(S))),
        'signed_mean_e': float(S.mean())}


def compare(seed=101):
    setup()
    selection = read(OUT/'backbone_selection.json')
    manifest = read(OUT/'data_manifest.json')
    receipt = read(OUT/f'seed_{seed}/training_receipt.json')
    checkpoint = receipt['selected_checkpoint'] or receipt['final_checkpoint']
    trained_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    checksum = runtime.sha256(checkpoint)
    old = load_model(device='cuda').eval()
    transfer = PhysicsPostInverseCNN(selection['physics_config']).cuda().eval()
    transfer.load_cnn(old.state_dict())
    trained = PhysicsPostInverseCNN(selection['physics_config']).cuda().eval()
    trained.load_cnn(trained_state['cnn_state_dict'])
    journal = Journal(OUT/f'seed_{seed}/paired_validation.jsonl')
    previous = {r['identity_sha256']: r for r in journal.rows() if r['status'] == 'SUCCESS'}
    old_checksum = runtime.sha256(DEFAULT_CHECKPOINT)
    implementation = canonical_hash({p: runtime.sha256(runtime.ROOT/p) for p in
        ('training_v4/compare.py', 'training_v4/model.py', 'network.py', 'solver.py', 'physics.py', 'evaluation_v2/methods.py')})
    rows = [r for r in manifest['rows'] if r['split'] == 'validation']
    with torch.no_grad():
        for i, row in enumerate(rows):
            inputs, cache = load_tensors(row, selection, labels=False)
            raw_dn, z, beta = inputs
            raw_e = old.physics.raw_to_electrons(raw_dn)
            target = npz(DATA/row['label_path'])
            for method in METHODS:
                identity = canonical_hash({'method': method, 'raw_hash': row['raw_hash'],
                    'camera': row['physics_config_sha256'], 'calibration': row['calibration_sha256'],
                    'cache_key': cache['cache_key'], 'checkpoint': checksum,
                    'old_checkpoint': old_checksum, 'implementation': implementation,
                    'label_sha256': row['label_sha256'], 'traditional': selection['traditional_selected']})
                if identity in previous:
                    continue
                base = {'identity_sha256': identity, 'case_id': row['case_id'], 'mother_id': row['mother_id'],
                    'condition': row['condition'], 'clean_identity': row['clean_identity'], 'method': method,
                    'raw_hash': row['raw_hash'], 'label_sha256': row['label_sha256'],
                    'physics_config_sha256': row['physics_config_sha256'], 'calibration_sha256': row['calibration_sha256'],
                    'checkpoint_sha256': checksum if method == 'trained_CNN' else None,
                    'cache_key': cache['cache_key'], 'detection_threshold_e': 60., 'seed': seed}
                started = time.perf_counter()
                try:
                    out = None
                    if method == 'zero_template':
                        S, T = np.zeros_like(target['template_e']), None
                    elif method == 'old_B4':
                        out = old(raw_dn[:, None])
                        S, T = out['species_signed_e'][0, 2].cpu().numpy(), out['temperature_color_K'][0].cpu().numpy()
                    elif method == 'traditional_validation_selected':
                        out = demod_fit(old.physics, raw_e, regularization=selection['traditional_selected'])
                        S, T = out['species_signed_e'][0, 2].cpu().numpy(), out['temperature_color_K'][0].cpu().numpy()
                    else:
                        out = physical_output(old.physics, z, beta, raw_e) if method == 'repair_physics_validation_selected' else (
                            transfer(*inputs) if method == 'old_CNN_transfer' else trained(*inputs))
                        S, T = out['template_e'][0].cpu().numpy(), out['temperature_color_K'][0].cpu().numpy()
                    scores = metrics(target, S, T)
                    if out is not None:
                        scores['invalid_output_pixels'] = int((~out['valid']).sum())
                        scores['valid_output_fraction'] = float(out['valid'].float().mean())
                        scores['raw_consistency_RMSE_e'] = float((out['expected_e']-raw_e).square().mean().sqrt())
                        scores['raw_weighted_consistency'] = float(((out['expected_e']-raw_e).square()/(
                            raw_e.clamp_min(0)+old.physics.extra_variance)).mean())
                        scores['predicted_band_RMSE_e'] = float((out['band_images_e'][0]-torch.as_tensor(
                            target['band_images_e'], device='cuda')).square().mean().sqrt()) if 'band_images_e' in out else None
                    else:
                        scores.update(invalid_output_pixels=None, valid_output_fraction=None,
                                      raw_consistency_RMSE_e=None, raw_weighted_consistency=None, predicted_band_RMSE_e=None)
                    scores['supervised_absolute_loss'] = (scores['template_RMSE_e']**2/60**2
                        +.2*scores['color_RMSE_K']**2/75**2+.2*scores['predicted_band_RMSE_e']**2/300**2
                        +.01*scores['raw_weighted_consistency']
                        +(scores['template_RMSE_e']**2/60**2 if scores['template_truth_energy'] == 0 else 0)
                        ) if scores['predicted_band_RMSE_e'] is not None else None
                    assert np.isfinite(S).all() and (T is None or np.isfinite(T).all())
                    path = DATA/'validation_predictions'/f'seed_{seed}'/row['case_id']/(method+'.npz')
                    save_npz(path, template_e=S, **({'temperature_color_K': T} if T is not None else {}))
                    journal.append({**base, 'status': 'SUCCESS', 'metrics': scores,
                        'seconds': time.perf_counter()-started, 'prediction_path': str(path.relative_to(DATA)).replace('\\', '/'),
                        'prediction_sha256': runtime.sha256(path)})
                except Exception as exc:
                    journal.append({**base, 'status': 'EXCEPTION', 'exception_type': type(exc).__name__,
                                    'message': str(exc), 'seconds': time.perf_counter()-started})
            if i % 16 == 15:
                print({'paired_validation_frames': i+1, 'total': len(rows), 'methods': 6}, flush=True)
    records = list({r['identity_sha256']: r for r in journal.rows()}.values())
    groups = []
    for clean in (False, True):
        for condition in ('blank', 'weak', 'normal', 'strong'):
            for method in METHODS:
                rs = [r for r in records if r['method'] == method and r['condition'] == condition and r['clean_identity'] == clean]
                good = [r for r in rs if r['status'] == 'SUCCESS']
                n = sum(r['metrics']['pixels'] for r in good)
                energy = sum(r['metrics']['template_truth_energy'] for r in good)
                sse = sum(r['metrics']['template_SSE'] for r in good)
                colors = [r for r in good if r['metrics']['color_SSE'] is not None]
                groups.append({'clean_identity': clean, 'condition': condition, 'method': method,
                    'expected_frames': 16 if clean else 128, 'records': len(rs), 'successes': len(good),
                    'failures': len(rs)-len(good), 'pooled_template_RMSE_e': float(np.sqrt(sse/n)) if n else None,
                    'pooled_template_L2': float(np.sqrt(sse/energy)) if energy else None,
                    'pooled_color_RMSE_K': float(np.sqrt(sum(r['metrics']['color_SSE'] for r in colors)/sum(r['metrics']['pixels'] for r in colors))) if colors else None,
                    'mean_supervised_absolute_loss': float(np.mean([r['metrics']['supervised_absolute_loss'] for r in good])) if good and good[0]['metrics']['supervised_absolute_loss'] is not None else None,
                    'mean_valid_output_fraction': float(np.mean([r['metrics']['valid_output_fraction'] for r in good])) if good and good[0]['metrics']['valid_output_fraction'] is not None else None,
                    'mean_blank_FP60': float(np.mean([r['metrics']['blank_FP60'] for r in good])) if condition == 'blank' and good else None,
                    'zero_estimate_L2_reference': 1. if condition != 'blank' else None})
    write(OUT/f'seed_{seed}/paired_summary.json', groups)
    trained_groups = [g for g in groups if g['method'] == 'trained_CNN' and not g['clean_identity']]
    physical_groups = [g for g in groups if g['method'] == 'repair_physics_validation_selected' and not g['clean_identity']]
    weak = next(g for g in trained_groups if g['condition'] == 'weak')
    improvements = [g['pooled_template_RMSE_e']/next(b['pooled_template_RMSE_e'] for b in physical_groups if b['condition'] == g['condition']) for g in trained_groups]
    clear = receipt['selected_checkpoint'] is not None and all(g['failures'] == 0 for g in trained_groups) and weak['pooled_template_L2'] < 1 and float(np.mean(improvements)) < .9
    write(OUT/f'seed_{seed}/benefit_decision.json', {'clear_validation_gain': clear,
        'mean_group_template_RMSE_ratio': float(np.mean(improvements)), 'weak_pooled_L2': weak['pooled_template_L2'],
        'weak_zero_estimate_L2': 1., 'protected_gate_checkpoint_available': receipt['selected_checkpoint'] is not None,
        'final_test_used': False, 'additional_seeds_authorized_by_gain': clear,
        'scope': 'development validation, not final test'})
    print({'paired_comparison_complete': True, 'seed': seed, 'records': len(records),
           'clear_validation_gain': clear}, flush=True)
