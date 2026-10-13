"""Independently check saved v4.5/v4.6 scalar evidence; no arrays or models."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'reports/training_v4_6/run_001'
OUT = Path(__file__).resolve().parent
ARMS = {'v4.5': ROOT / 'reports/training_v4_5/run_001/balanced_fixed_continuum',
        'v4.6': BASE / 'negative_noisy_repair'}
UPDATES = tuple(range(0, 501, 100))
SIGNED = ('negative_weak', 'negative_normal', 'positive_weak', 'positive_normal')
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def blob(path):
    path = Path(path)
    require(path.suffix in ('.json', '.jsonl', '.md', '.py'), 'Only scalar text evidence may be read: ' + str(path))
    value = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(value).hexdigest()
    return value


def read(path):
    return json.loads(blob(path).decode('utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in blob(path).decode('utf-8-sig').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(blob(path)).hexdigest()


def close(actual, expected, label):
    if expected is None:
        require(actual is None, label)
    else:
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), label)


def mean(values):
    return sum(values) / len(values)


def verify_scores(name, folder, update, ids, protocol, gate_variants):
    rows = lines(folder / f'validation_update_{update:04d}.jsonl')
    summary = read(folder / f'validation_summary_{update:04d}.json')
    science = read(folder / f'science_summary_{update:04d}.json')
    require([r['case_id'] for r in rows] == ids and len(set(ids)) == len(rows) == 776, 'Validation case identity')
    require(summary['optimizer_update'] == science['optimizer_update'] == update, 'Summary update identity')
    require(all(r['optimizer_update'] == update and r['status'] == 'SUCCESS' for r in rows), 'Record update/status')
    require(all(math.isfinite(v) for r in rows for v in r['metrics'].values() if v is not None), 'Finite saved metrics')
    require(all(r['invalid_output_fraction'] == r['temperature_correction_RMS_K'] == 0 for r in rows), 'Invalid/temperature audit')
    require(science['validation_sha256'] == digest(folder / f'validation_update_{update:04d}.jsonl')
            and science['protocol_sha256'] == digest(folder.parent / 'protocol.json'), 'Screen source binding')
    for r in rows:
        m = r['metrics']
        close(m['template_SSE'] / m['pixels'], m['template_MSE_e2'], 'Per-case MSE/SSE')
        close(math.sqrt(m['template_MSE_e2']), m['template_RMSE_e'], 'Per-case RMSE')
        if m['truth_energy']:
            close(math.sqrt(m['template_SSE'] / m['truth_energy']), m['template_L2'], 'Per-case L2')
    groups = []
    for saved in summary['groups']:
        rs = [r for r in rows if all(r[k] == saved[k] for k in ('input_kind', 'signed_aux', 'group'))]
        require(len(rs) == saved['frames'] and saved['failures'] == 0, 'Group counts')
        sse, pixels, energy = (sum(r['metrics'][k] for r in rs) for k in ('template_SSE', 'pixels', 'truth_energy'))
        values = {'template_RMSE_e': math.sqrt(sse / pixels), 'template_L2': math.sqrt(sse / energy) if energy else None,
                  'color_RMSE_K': math.sqrt(mean([r['metrics']['temperature_MSE_K2'] for r in rs])),
                  'physical_color_RMSE_K': math.sqrt(mean([r['physical']['temperature_MSE_K2'] for r in rs])),
                  'mean_template_excess_e2': mean([r['metrics']['template_MSE_e2'] - r['physical']['template_MSE_e2'] for r in rs]),
                  'mean_temperature_excess_K2': mean([r['metrics']['temperature_MSE_K2'] - r['physical']['temperature_MSE_K2'] for r in rs])}
        for key, value in values.items():
            close(value, saved[key], 'Group aggregate ' + key)
        groups.append({**saved, **values})
    require(len(groups) == 21, 'All group coverage')
    protected = defaultdict(list)
    for r in rows:
        if r['input_kind'] != 'noise':
            protected[(r['geometry'], r['condition'], r['input_kind'])].append(r)
    saved_protected = {(r['geometry'], r['condition'], r['input_kind']): r for r in summary['protection_groups']}
    require(set(protected) == set(saved_protected) and len(protected) == 96, 'Protection coverage')
    protection_pass = []
    for key, rs in protected.items():
        saved = saved_protected[key]
        ds = mean([r['metrics']['template_MSE_e2'] - r['physical']['template_MSE_e2'] for r in rs])
        dt = mean([r['metrics']['temperature_MSE_K2'] - r['physical']['temperature_MSE_K2'] for r in rs])
        close(ds, saved['mean_template_MSE_delta'], 'Protection template delta')
        close(dt, saved['mean_temperature_MSE_delta'], 'Protection temperature delta')
        passed = ds <= .25 and dt <= .04
        require(saved['frames'] == len(rs) and saved['passed'] == passed, 'Protection decision')
        protection_pass.append(passed)
    gates = read(folder / f'gates_update_{update:04d}.json')
    require(len(gates['decisions']) == len({g['case_id'] for g in gates['decisions']}) == 16, 'Original gate coverage')
    gate_pass = []
    for g in gates['decisions']:
        continuous = gate_variants[g['case_id']] == 'continuous'
        passed = (g['color_RMSE_K'] <= (1 if continuous else 3) and g['template_RMSE_e'] <= (5 if continuous else 6)
                  and (g['template_relative_L2'] <= .25 if g['template_relative_L2'] is not None else g['blank_FP60'] == 0))
        require(passed == g['passed'], 'Original gate decision')
        gate_pass.append(passed)
    weak = next(g for g in groups if g['input_kind'] == 'noise' and not g['signed_aux'] and g['group'] == 'weak')
    normal = next(g for g in groups if g['input_kind'] == 'noise' and not g['signed_aux'] and g['group'] == 'normal')
    temperature_pass = all(g['color_RMSE_K'] ** 2 <= 1.05 * g['physical_color_RMSE_K'] ** 2 + .04 for g in groups if g['input_kind'] == 'noise')
    eligible = all(gate_pass) and all(protection_pass) and temperature_pass and weak['template_L2'] < 1
    require(gates['passed'] == summary['original_gates_pass'] == all(gate_pass), 'Gate conjunction')
    require(summary['original_gates_pass_count'] == sum(gate_pass) and summary['protection_pass'] == all(protection_pass), 'Gate counts')
    require(summary['noisy_temperature_pass'] == temperature_pass and summary['eligible'] == science['old_eligible'] == eligible, 'Old eligible recomputation')
    close(weak['template_L2'], summary['weak_L2'], 'Main weak')
    close(normal['template_RMSE_e'], summary['normal_RMSE_e'], 'Main normal')
    close(mean([r['main_loss'] for r in rows]), summary['main_validation_loss'], 'Main loss mean')
    screen = protocol['v4_4_additional_development_screen']
    require([g['condition'] for g in science['signed_noisy_conditions']] == list(SIGNED), 'Signed condition order')
    signed = []
    for condition, saved in zip(SIGNED, science['signed_noisy_conditions']):
        rs = [r for r in rows if r['signed_aux'] and r['input_kind'] == 'noise' and r['condition'] == condition]
        require(len(rs) == saved['frames'] == 6, 'Signed count')
        mse = mean([r['metrics']['template_MSE_e2'] for r in rs])
        physical = mean([r['physical']['template_MSE_e2'] for r in rs])
        limit = screen['template_relative_MSE_factor'] * physical + screen['template_absolute_MSE_tolerance_e2']
        energy = sum(r['metrics']['truth_energy'] for r in rs)
        l2 = math.sqrt(sum(r['metrics']['template_SSE'] for r in rs) / energy)
        projection = sum(r['metrics']['signed_projection'] * r['metrics']['truth_energy'] for r in rs) / energy
        recall_key = 'negative_recall60' if condition.startswith('negative') else 'positive_recall60'
        recalls = [r['metrics'][recall_key] for r in rs if r['metrics'][recall_key] is not None]
        values = {'template_MSE_e2': mse, 'physical_template_MSE_e2': physical, 'MSE_limit_e2': limit,
                  'template_RMSE_e': math.sqrt(mse), 'template_L2': l2, 'energy_weighted_signed_projection': projection,
                  'macro_recall_at_fixed_60e': mean(recalls) if recalls else None}
        for key, value in values.items():
            close(value, saved[key], 'Signed aggregate ' + key)
        mse_pass = mse <= limit
        weak_pass = not condition.endswith('_weak') or l2 < screen['separate_signed_weak_pooled_L2_strictly_less_than']
        require(saved['MSE_pass'] == mse_pass and saved['weak_L2_pass'] == weak_pass and saved['passed'] == (mse_pass and weak_pass), 'Signed decisions')
        signed.append({**saved, 'RMSE_limit_e': math.sqrt(limit)})
    added = all(g['passed'] for g in signed)
    require(science['new_development_screen_pass'] == added and science['science_candidate'] == (eligible and added), 'Science candidate')
    require(science['all_frames_finite'] and science['maximum_invalid_output_fraction'] == science['maximum_temperature_correction_RMS_K'] == 0, 'Global science audit')
    audits = lines(folder / f'forward_audit_{update:04d}.jsonl')
    require(len(audits) == 776, 'Forward audit coverage')
    for r, a in zip(rows, audits):
        require(all(a[k] == r[k] for k in ('case_id', 'raw_hash', 'condition', 'input_kind')), 'Forward audit identity')
    gate_audits = read(folder / f'original_gate_forward_audit_{update:04d}.json')
    require(len(gate_audits) == 16 and all(all(a[k] is True for k in ('final_C_equals_physics', 'signed_state_equals_scaled_proposal', 'temperature_equals_physics')) for a in audits + gate_audits), 'Recorded fixed-continuum identity')
    return rows, summary, science, {'arm': name, 'update': update, 'original_gates_passed': sum(gate_pass),
        'protection_groups_passed': sum(protection_pass), 'temperature_pass': temperature_pass, 'old_eligible': eligible,
        'main_weak_L2': weak['template_L2'], 'main_normal_RMSE_e': normal['template_RMSE_e'], 'signed': signed,
        'science_candidate': eligible and added, 'saved_forward_identities_true': True}


def main():
    protocol = read(BASE / 'protocol.json')
    require(digest(BASE / 'protocol.json') == '4d49cd48853b20f62960754d07637a3cca3ae454e79e8e4ac67b8a2f7d78b807', 'Frozen protocol')
    old_manifest = read(ROOT / 'reports/training_v4_1/run_001/data_manifest.json')['rows']
    validation_ids = [r['case_id'] for r in old_manifest if r['split'] == 'validation']
    old_by = {r['case_id']: r for r in old_manifest}
    manifest = read(BASE / 'data_manifest.json')['rows']; by = {r['case_id']: r for r in manifest}
    sampling = read(BASE / 'sampling_sequence.json')
    variants = {c['case_id']: c['metadata']['variant'] for c in read(ROOT / 'reports/repair_v3/run_001/frozen_manifest.json')['cases']}
    training = {name: read(folder / 'training_receipt.json') for name, folder in ARMS.items()}
    require(digest(ARMS['v4.6'] / 'training_receipt.json') == 'a6fe06decc38c09133035146cf69c6fde3b597f985b6173c74c73c04acb9ce42', 'Completed training receipt')
    declared_checkpoints = {}
    for name, folder in ARMS.items():
        status = read(folder / 'status.json'); receipt = training[name]
        require(status['stage'] == 'COMPLETE' and status['optimizer_updates'] == receipt['updates'] == 500 and receipt['microbatches'] == 2000, 'Completion counts')
        require(len(receipt['checkpoints']) == 6 and {c['update'] for c in receipt['checkpoints']} == set(UPDATES), 'Checkpoint metadata inventory')
        for c in receipt['checkpoints']:
            require(c['Adam_steps'] == ([] if c['update'] == 0 else [c['update']]), 'Declared native Adam steps')
            declared_checkpoints[c['path']] = c['sha256']
    require(training['v4.6']['initial_parameters_sha256'] == training['v4.5']['initial_parameters_sha256'], 'Shared restart')
    require(training['v4.6']['empty_Adam_initial_state'] and training['v4.6']['actual_unchanged_sample_identity_comparisons'] == 1487, 'Registered start/intervention')
    curves = {name: lines(folder / 'training_curve.jsonl') for name, folder in ARMS.items()}
    unchanged = intervened = changed = signed_draws = 0
    actual_draws = []; noisy = defaultdict(Counter)
    for i, (a, b) in enumerate(zip(curves['v4.5'], curves['v4.6']), 1):
        require(a['optimizer_update'] == b['optimizer_update'] == i and b['status'] == 'SUCCESS' and len(b['samples']) == 4, 'Actual update journal')
        actual_draws.append([r['case_id'] for r in b['samples']])
        for slot, (before, after) in enumerate(zip(a['samples'], b['samples'])):
            ro, rn = old_by[before['case_id']], by[after['case_id']]
            intervention = ro['signed_aux'] and ro['input_kind'] == 'noise' and ro['condition'] in ('negative_normal', 'negative_weak')
            pair = b['paired_source_slots'][slot]
            require(pair['source_sample'] == before and pair['negative_noisy_intervention'] == intervention, 'Paired source slot')
            require(rn['split'] == 'train' and all(ro[k] == rn[k] for k in ('signed_aux', 'condition', 'input_kind')), 'Preserved condition schedule')
            require(after['raw_hash'] == rn['raw_hash'] and pair['label_sha256'] == rn['label_sha256'], 'Recorded training identity')
            signed_draws += rn['signed_aux']
            changed += before['case_id'] != after['case_id']
            if intervention:
                intervened += 1; noisy[rn['condition']][rn['case_id']] += 1
            else:
                require(before == after, 'Unchanged sample identity'); unchanged += 1
    require(len(curves['v4.5']) == len(curves['v4.6']) == len(actual_draws) == 500 and actual_draws == sampling['draws'], 'Actual frozen draw sequence')
    require((unchanged, intervened, changed, signed_draws) == (1487, 513, 502, 575), 'Intervention counts')
    used = {c for b in actual_draws for c in b}; new_used = used - set(old_by)
    require(len(used) == 971 and len(new_used) == 120 and len(set(by) - set(old_by)) == 432, 'Actual unique exposure')
    for condition, counts in noisy.items():
        require(len(counts) == 72 and len({by[c]['mother_id'] for c in counts}) == 36 and set(counts.values()) == {3, 4}, 'Negative noisy coverage')
        # Legacy first-noise rows do not carry noise_index. Preserve that
        # metadata boundary and verify the actual two registered case IDs here.
        require(all(sum(by[c]['mother_id'] == mother for c in counts) == 2 for mother in {by[c]['mother_id'] for c in counts}), 'Two registered noisy case IDs per mother')
    analysis = read(BASE / 'analysis_receipt.json'); comparison = read(BASE / 'comparison_summary.json')
    require(analysis['status'] == 'COMPLETE' and comparison['paired_case_count'] == analysis['paired_cases'] == 4656, 'Analysis completeness')
    entries = {(e['arm'], e['update']): e for e in comparison['entries']}
    require(len(entries) == 12, 'Comparison entry coverage')
    paired_saved = lines(BASE / 'paired_cases.jsonl'); parents_saved = read(BASE / 'paired_parent_summary.json')
    require(len(paired_saved) == 4656, 'Saved paired case count')
    paired_map = {(r['update'], r['case_id']): r for r in paired_saved}; require(len(paired_map) == 4656, 'Unique paired records')
    expected_parents = defaultdict(list); results = []; summaries = {}; sciences = {}
    for u in UPDATES:
        observed = {}
        for name, folder in ARMS.items():
            rows, summary, science, result = verify_scores(name, folder, u, validation_ids, read(folder.parent / 'protocol.json'), variants)
            observed[name] = rows; results.append(result); summaries[(name, u)] = summary; sciences[(name, u)] = science
            require(entries[(name, u)]['summary'] == summary and entries[(name, u)]['science'] == science, 'Analysis uses same saved endpoint')
            if name == 'v4.6':
                require(entries[(name, u)]['checkpoint_sha256'] == declared_checkpoints[entries[(name, u)]['checkpoint_path']], 'Analysis binds declared checkpoint')
        for a, b in zip(observed['v4.5'], observed['v4.6']):
            require(all(a[k] == b[k] for k in ('case_id', 'mother_id', 'condition', 'group', 'geometry', 'input_kind', 'signed_aux', 'raw_hash', 'label_sha256', 'physical')), 'Paired raw/label/cache metadata')
            require(all(a['metrics'][k] == b['metrics'][k] for k in ('pixels', 'truth_energy')), 'Paired truth scalar identity')
            if u == 0: require(a == b, 'All zero-update records identical')
            delta = b['metrics']['template_MSE_e2'] - a['metrics']['template_MSE_e2']
            expected = {'update': u, 'case_id': a['case_id'], 'mother_id': a['mother_id'], 'geometry': a['geometry'],
                'condition': a['condition'], 'input_kind': a['input_kind'], 'signed_aux': a['signed_aux'],
                'v45_MSE_e2': a['metrics']['template_MSE_e2'], 'v46_MSE_e2': b['metrics']['template_MSE_e2'],
                'paired_delta_MSE_e2': delta, 'same_raw_label_cache': True}
            require(paired_map[(u, a['case_id'])] == expected, 'Paired scalar delta')
            expected_parents[(u, a['mother_id'], a['condition'], a['input_kind'])].append(delta)
    require(len(parents_saved) == len(expected_parents) == 3120, 'Parent aggregate coverage')
    seen = set()
    for parent in parents_saved:
        key = tuple(parent[k] for k in ('update', 'mother_id', 'condition', 'input_kind'))
        require(key not in seen and parent['frames'] == len(expected_parents[key]), 'Unique parent aggregate')
        seen.add(key); close(mean(expected_parents[key]), parent['paired_mean_delta_MSE_e2'], 'Parent delta mean')
    selection = read(BASE / 'candidate_selection.json')
    require(sum(r['science_candidate'] for r in results if r['arm'] == 'v4.6') == selection['candidate_count'] == analysis['candidate_count'] == 0, 'Zero candidates independently recomputed')
    require(selection['selected'] is None and selection['eligible_candidates'] == [] and selection['next_action'] == 'close_bounded_data_contrast_no_additional_optimization', 'Registered stop action')
    require(not analysis['final_test_used'] and not analysis['locked_development_used'] and not selection['final_test_used'], 'Saved analysis scope')
    report = blob(BASE / 'V4_6_REPORT.md').decode('utf-8-sig')
    for u in UPDATES:
        s, c = summaries[('v4.6', u)], sciences[('v4.6', u)]
        failed = ', '.join(g['condition'] for g in c['signed_noisy_conditions'] if not g['passed']) or '无'
        row = f"|{u}|{s['eligible']}|{c['new_development_screen_pass']}|{c['science_candidate']}|{s['weak_L2']:.6f}|{s['normal_RMSE_e']:.4f}|{failed}|"
        require(row in report, 'Chinese checkpoint table')
    for name in ARMS:
        for g in sciences[(name, 500)]['signed_noisy_conditions']:
            row = f"|{name}|{g['condition']}|{g['template_RMSE_e']:.4f}|{math.sqrt(g['MSE_limit_e2']):.4f}|{g['template_L2']:.6f}|{g['energy_weighted_signed_projection']:.6f}|{g['passed']}|"
            require(row in report, 'Chinese endpoint table')
    for name, expected in analysis['input_files_sha256'].items():
        if name.endswith('.pt'):
            require(declared_checkpoints[name] == expected, 'Declared checkpoint analysis hash')
        else:
            require(digest(ROOT / name) == expected, 'Saved analysis input drift: ' + name)
    skipped_visuals = []
    for name, expected in analysis['outputs_sha256'].items():
        if Path(name).suffix in ('.png', '.svg'):
            skipped_visuals.append(name)
        else:
            require(digest(BASE / name) == expected, 'Saved analysis output drift: ' + name)
    require(digest(ARMS['v4.6'] / 'validation_update_0000.jsonl') == digest(ARMS['v4.5'] / 'validation_update_0000.jsonl'), 'Zero-update byte identity')
    endpoint = next(r for r in results if r['arm'] == 'v4.6' and r['update'] == 500)
    for relative, expected in list(INPUTS.items()):
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, 'Scalar evidence changed during review: ' + relative)
    require(not {'numpy', 'torch', 'runtime'} & set(sys.modules), 'Numerical/project runtime import')
    result = {'status': 'PASS_INDEPENDENT_SAVED_SCALAR_REVIEW', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'reviewer_source_sha256': digest(Path(__file__)), 'input_files_sha256': dict(sorted(INPUTS.items())),
        'checkpoint_results': results, 'candidate_count': 0, 'recommendation': 'CLOSE_THIS_REGISTERED_FINITE_BRANCH',
        'training_updates': 500, 'training_microbatches': 2000, 'actual_changed_case_ID_positions': changed,
        'negative_noisy_intervention_positions': intervened, 'other_sample_identities_unchanged': unchanged,
        'actual_unique_training_cases': len(used), 'actual_unique_training_mothers': len({by[c]['mother_id'] for c in used}),
        'new_training_cases_used': len(new_used), 'new_training_cases_not_used': 432 - len(new_used),
        'paired_case_records_recomputed': 4656, 'parent_group_records_recomputed': 3120,
        'all_zero_update_case_records_identical': True, 'chinese_report_all_numeric_rows_verified': True,
        'all_12_summary_groups_and_gates_recomputed': True, 'checkpoint_hashes_bound_via_saved_receipts_only': True,
        'checkpoint_files_opened_or_rehashed': 0, 'raw_truth_state_array_files_opened': 0,
        'model_inference_calls': 0, 'physical_inverse_calls': 0, 'GPU_calls': 0,
        'visual_outputs_not_opened_by_this_review': skipped_visuals,
        'scope_limit': 'Recomputes saved scalar aggregation and gates, not original model-output scoring. Native execution, array/weight-byte integrity and visual QA are separate root receipts.',
        'final_test_opened': False, 'locked_development_opened': False}
    target = OUT / 'final_science_review_receipt.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    require(json.loads(target.read_text(encoding='utf-8')) == result, 'Review receipt reopen')
    print(json.dumps({'status': result['status'], 'candidate_count': 0, 'review_receipt_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'endpoint': endpoint}, ensure_ascii=False))


if __name__ == '__main__':
    main()
