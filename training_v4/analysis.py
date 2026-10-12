"""Offline paired differences and parent-level confidence intervals, no selection."""
import math
import numpy as np
from evaluation_v2.common import Journal
from training_v4.common import OUT, read, write


def summarize_validation(folder):
    """Describe every saved validation, including candidates rejected by gates."""
    summaries = []
    for event in Journal(folder/'validation_curve.jsonl').rows():
        rows = Journal(folder/f'validation_update_{event["optimizer_update"]:04d}.jsonl').rows()
        groups = []
        for clean in (False, True):
            for condition in ('blank', 'weak', 'normal', 'strong'):
                group = [r for r in rows if r['clean_identity'] == clean and r['condition'] == condition]
                assert len(group) == (16 if clean else 128)
                sse = sum(r['template_SSE'] for r in group)
                energy = sum(r['template_truth_energy'] for r in group)
                pixels = sum(r['pixels'] for r in group)
                groups.append({'clean_identity': clean, 'condition': condition, 'frames': len(group),
                    'template_RMSE_e': math.sqrt(sse/pixels),
                    'template_L2': math.sqrt(sse/energy) if energy else None,
                    'color_RMSE_K': math.sqrt(sum(r['color_RMSE_K']**2 for r in group)/len(group)),
                    'mean_blank_FP60': float(np.mean([r['blank_FP60'] for r in group])) if not energy else None})
        gates = read(folder/f'gates_update_{event["optimizer_update"]:04d}.json')
        summaries.append({**event, 'gates_passed_count': sum(r['passed'] for r in gates['decisions']),
                          'groups': groups})
    write(folder/'all_checkpoint_validation_summary.json', summaries)
    return summaries


def paired_analysis(seed):
    folder = OUT/f'seed_{seed}'
    summaries = summarize_validation(folder)
    records = Journal(folder/'paired_validation.jsonl').rows()
    by_case = {}
    for record in records:
        assert record['method'] not in by_case.setdefault(record['case_id'], {})
        by_case[record['case_id']][record['method']] = record
    assert len(by_case) == 576
    paired = Journal(folder/'paired_differences.jsonl')
    # A reproducible regeneration writes each pair exactly once.
    if paired.path.exists():
        paired.path.unlink()
    for case_id, methods in by_case.items():
        trained = methods['trained_CNN']
        for name, baseline in methods.items():
            if name == 'trained_CNN':
                continue
            assert trained['raw_hash'] == baseline['raw_hash']
            assert trained['label_sha256'] == baseline['label_sha256']
            assert trained['cache_key'] == baseline['cache_key']
            entry = {k: trained[k] for k in ('case_id', 'mother_id', 'condition', 'clean_identity', 'raw_hash', 'cache_key')}
            success = trained['status'] == baseline['status'] == 'SUCCESS'
            entry.update(baseline=name, trained_status=trained['status'], baseline_status=baseline['status'],
                         paired_success=success)
            if success:
                a, b = trained['metrics'], baseline['metrics']
                entry.update(trained_template_SSE=a['template_SSE'], baseline_template_SSE=b['template_SSE'],
                    truth_energy=a['template_truth_energy'], pixels=a['pixels'],
                    template_MSE_delta=(a['template_SSE']-b['template_SSE'])/a['pixels'],
                    template_RMSE_delta_e=a['template_RMSE_e']-b['template_RMSE_e'],
                    color_RMSE_delta_K=a['color_RMSE_K']-b['color_RMSE_K'] if b['color_RMSE_K'] is not None else None)
            paired.append(entry)
    pairs = paired.rows()
    estimates = []
    rng = np.random.default_rng(7007)
    for clean in (False, True):
        for condition in ('blank', 'weak', 'normal', 'strong'):
            for baseline in sorted({r['baseline'] for r in pairs}):
                rs = [r for r in pairs if r['baseline'] == baseline and r['condition'] == condition and r['clean_identity'] == clean]
                good = [r for r in rs if r['paired_success']]
                parents = sorted({r['mother_id'] for r in good})
                deltas = np.asarray([np.mean([r['template_MSE_delta'] for r in good if r['mother_id'] == p]) for p in parents])
                interval = None
                if len(deltas):
                    samples = deltas[rng.integers(0, len(deltas), size=(2000, len(deltas)))].mean(1)
                    interval = np.quantile(samples, [.025, .975]).tolist()
                estimates.append({'condition': condition, 'clean_identity': clean, 'baseline': baseline,
                    'paired_frames': len(rs), 'paired_successes': len(good), 'failed_pairs_retained': len(rs)-len(good),
                    'mother_clusters': len(parents), 'mean_template_MSE_delta': float(deltas.mean()) if len(deltas) else None,
                    'parent_bootstrap_MSE_delta_CI95': interval, 'bootstrap_resamples': 2000,
                    'direction': 'negative means trained CNN has lower error',
                    'used_for_checkpoint_selection': False, 'final_test_used': False})
    write(folder/'paired_confidence_intervals.json', estimates)
    # The terminal model is diagnostic if gates reject it. Reuse its already
    # recorded full validation instead of inferring or selecting it again.
    terminal = Journal(folder/'validation_update_2000.jsonl').rows()
    diagnostic = Journal(folder/'terminal_2000_paired_diagnostic.jsonl')
    if diagnostic.path.exists():
        diagnostic.path.unlink()
    for row in terminal:
        methods = by_case[row['case_id']]
        for name in ('old_B4', 'repair_physics_validation_selected'):
            baseline = methods[name]
            assert row['raw_hash'] == baseline['raw_hash'] and row['cache_key'] == baseline['cache_key']
            success = row['status'] == baseline['status'] == 'SUCCESS'
            entry = {k: row[k] for k in ('case_id', 'mother_id', 'condition', 'clean_identity', 'raw_hash', 'cache_key')}
            entry.update(method='trained_CNN_terminal_2000_diagnostic', baseline=name,
                optimizer_update=2000, paired_success=success, used_for_model_selection=False,
                source='existing full validation update_2000, no new inference')
            if success:
                entry.update(template_RMSE_delta_e=row['template_RMSE_e']-baseline['metrics']['template_RMSE_e'],
                    template_MSE_delta=row['template_SSE']/row['pixels']-baseline['metrics']['template_SSE']/baseline['metrics']['pixels'],
                    trained_template_relative_L2=row['template_relative_L2'])
            diagnostic.append(entry)
    write(folder/'paired_analysis_receipt.json', {'same_input_cases': len(by_case), 'differences': len(pairs),
        'all_checkpoint_summaries': len(summaries), 'pairing_fields_checked': ['raw_hash', 'label_sha256', 'cache_key'],
        'terminal_diagnostic_pairs': len(diagnostic.rows()),
        'bootstrap_unit': 'mother scene, noise replicates kept in one cluster', 'final_test_used': False})
    return summaries
