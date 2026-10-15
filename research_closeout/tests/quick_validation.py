"""CPU-only controller contracts; no formal inference or old report writes."""
from .. import storage as s
from .. import quick_validation as q
import copy
import json
from pathlib import Path
import uuid
import numpy as np


def run():
    token = uuid.uuid4().hex
    report = q.REPORT_DIR/'_cpu_fixtures'/token
    data = q.DATA_DIR/'_cpu_fixtures'/token
    report.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    def rejects(action, exception=(q.QuickIntegrityError, ValueError)):
        try:
            action()
        except exception:
            return True
        return False

    inventory = s.read(q.INVENTORY)
    selected = q.select_panel(inventory)
    expected = [f'v5_development_selection_mother_{i:04d}_{condition}_noise0' for i, condition in q.PANEL]
    check('fixed_picker_uses_actual_six_development_noise_native256_refs',
          [r['case_id'] for r in selected] == expected
          and all(r['cohort'] == 'development' and r['input_kind'] == 'noise'
                  and r['raw_reference']['shape'] == [256, 256]
                  and (s.LEGACY/r['raw_reference']['path']).is_file() for r in selected),
          cases=expected, raw_payloads_opened=False)
    scrambled = list(reversed(inventory['rows']))
    sealed = copy.deepcopy(selected[0])
    sealed.update(cohort='sealed_final', raw_reference={'path': 'DO_NOT_OPEN_FINAL'})
    check('picker_order_is_frozen_and_final_metadata_cannot_enter',
          q.select_panel(scrambled+[sealed]) == selected)
    check('actual_geometry_fallback_has_source_and_is_not_invented',
          selected[0]['quick_geometry']['temperature_geometry'] == 'front'
          and selected[0]['quick_geometry']['template_geometry'] == 'blob'
          and selected[1]['quick_geometry']['temperature_geometry'] == 'ring'
          and selected[1]['quick_geometry']['template_geometry'] == 'multipeak'
          and selected[0]['quick_geometry']['temperature_geometry_source'].endswith(':geometry')
          and selected[0]['quick_geometry']['temperature_geometry_fallback_declared'])
    check('missing_and_ambiguous_cells_are_rejected',
          rejects(lambda: q.select_panel(selected[1:]))
          and rejects(lambda: q.select_panel(selected+[selected[0]])))
    bad = copy.deepcopy(selected)
    bad[0]['raw_reference']['shape'] = [32, 32]
    check('native256_is_required_even_for_small_panel', rejects(lambda: q.select_panel(bad)))
    check('smoke_is_six_calls_and_panel_requires_explicit_tier',
          len(q.planned_cells(selected, 'smoke')) == 6
          and len(q.planned_cells(selected, 'panel')) == 18
          and q.CONFIG['tiers']['smoke']['inverse_wall_budget_s'] == 18*60
          and q.CONFIG['detector'] is None and q.CONFIG['method_selection'] is False)

    dependency = report/'source_fixture.py'
    dependency.write_text('value = 1\n', encoding='utf8')
    hashes = {'source_fixture.py': s.sha(dependency)}
    q.verify_hashes(hashes, root=report)
    dependency.write_text('value = 2\n', encoding='utf8')
    check('registered_source_hash_detects_edit',
          rejects(lambda: q.verify_hashes(hashes, root=report)))
    check('source_hash_check_rejects_parent_path_escape',
          rejects(lambda: q.verify_hashes({'../source_fixture.py': '0'*64}, root=report)))
    original_dn = np.array([[0, 10], [4095, 65535]], dtype=np.uint16)
    original_ref = {'raw_reference': {'shape': [2, 2], 'dtype': 'uint16',
                                      'array_sha256': q._array_sha(original_dn)}}
    floating_provenance = {'source_input_dtype': 'torch.float32',
                          'inference_input_array_sha256': q._array_sha(original_dn.astype(np.float32))}
    cast_audit = q.verify_initializer_raw(original_ref, original_dn.astype(np.float64)[None], floating_provenance)
    check('camera_uint16_hash_and_legacy_float32_source_hash_are_separate_and_lossless',
          all(cast_audit.values())
          and original_ref['raw_reference']['array_sha256'] != floating_provenance['inference_input_array_sha256'])
    fractional_dn = original_dn.astype(np.float64)
    fractional_dn[0, 1] += .5
    check('fractional_raw_and_wrong_floating_source_hash_are_rejected',
          rejects(lambda: q.verify_initializer_raw(original_ref, fractional_dn[None], floating_provenance))
          and rejects(lambda: q.verify_initializer_raw(original_ref, original_dn.astype(np.float64)[None],
                                                       {**floating_provenance, 'inference_input_array_sha256': '0'*64})))

    lock_a, lock_b = report/'global_phase_fixture.json', report/'local_phase_fixture.json'
    with q.phase_locks((lock_a, lock_b), registration_sha256='fixture'):
        lock_value = s.read(lock_a)
        check('phase_lock_has_real_pid_and_rejects_second_owner',
              lock_value['pid'] == q.os.getpid()
              and rejects(lambda: _enter_lock(lock_a), FileExistsError))
    check('finally_removes_both_owned_locks', not lock_a.exists() and not lock_b.exists())
    foreign = {'pid': -1, 'owner_token': 'another-owner'}
    s.write(lock_b, foreign)
    check('partial_acquisition_failure_preserves_foreign_lock',
          rejects(lambda: _enter_lock(lock_a, lock_b), FileExistsError)
          and not lock_a.exists() and s.read(lock_b) == foreign)
    lock_c = report/'replaced_phase_fixture.json'
    with q.phase_locks((lock_c,), registration_sha256='fixture'):
        s.write(lock_c, foreign, replace=True)
    check('finally_cannot_delete_a_replaced_foreign_lock', s.read(lock_c) == foreign)

    planned = q.planned_cells(selected, 'smoke')
    cell = planned[0]
    digest = 'controller-fixture-registration'
    provenance = {'kind': 'CPU_fixture_raw_only', 'truth_or_identity_used': False,
                  'raw_file_sha256': cell['raw_sha256'], 'new_complete_inverse_calls': 0}
    calls, budgets, scoring = [], [], []
    values = np.arange(4, dtype=np.float64).reshape(1, 2, 2)+1.0000000000000002

    def output(certified=False):
        return {'temperature_color_K': 3000.+values/1000., 'template_postblur_e_ref': values.copy(),
                'continuum_postblur_bands': np.repeat(values[:, None], 3, axis=1),
                'predicted_raw_e': values*17.001, 'valid': np.ones((1, 2, 2), dtype=bool),
                'low_signal': np.zeros((1, 2, 2), dtype=bool), 'numerical_valid': certified,
                'info': {'status': 'FINITE_NUMERICAL_UNCERTIFIED', 'truth_used': False}}

    def action():
        calls.append('action')
        return output()

    common = {'registration_sha256': digest, 'initializer_provenance': provenance,
              'report_dir': report/'cache', 'data_dir': data/'cache', 'action': action,
              'count_call': lambda *args: calls.append(args), 'budget_guard': lambda: budgets.append(True),
              'compact_info': lambda out: s.plain(out['info']), 'output_budget_bytes': 2**20, 'shape': (2, 2)}
    key = q._cache_key(cell, digest)

    def score_action(out):
        scoring.append('external_labels_after_inference')
        assert (report/'cache/cells'/key/'INFERENCE.json').exists()
        assert (data/'cache'/(key+'.npz')).exists()
        if len(scoring) == 1:
            raise RuntimeError('Injected scoring failure after persisted inverse')
        return {'status': 'FINITE_NUMERICALLY_UNCERTIFIED', 'production_numerical_certified': False,
                'metrics': {'domain_pixels': 4, 'template_SSE': 4., 'temperature_SSE_K2': 16.,
                            'frame_detection_v2': {'threshold': None, 'detected': None}}}

    first = q.evaluate_cell(cell, score_action=score_action, **common)
    second = q.evaluate_cell(cell, score_action=score_action, **common)
    check('scoring_failure_preserves_finite_uncertified_inverse_and_retry_never_refits',
          first['score_status'] == 'SCORING_FAILED' and first['finite_output']
          and second['score_status'] == 'SCORED' and second['reused_inference']
          and calls == [('repair_quick', cell['method'], cell['case_id']), 'action']
          and len(budgets) == 1 and len(scoring) == 2
          and not second['production_numerical_certified'])
    saved = s.arrays(second['output_cache']['path'])
    check('canonical_npz_roundtrip_retains_native_float64_exact_values',
          set(saved) == {'T', 'u', 'C', 'predicted_raw_e', 'valid', 'low_signal'}
          and all(saved[k].dtype == np.float64 for k in ('T', 'u', 'C', 'predicted_raw_e'))
          and np.array_equal(saved['u'], values)
          and np.array_equal(saved['predicted_raw_e'], values*17.001),
          bytes=Path(second['output_cache']['path']).stat().st_size)
    identity_for_saved = {k: second[k] for k in
                          ('registration_sha256', 'method', 'raw_sha256', 'cache_key', 'case_id')}
    forged_certificate = {**second, 'production_numerical_certified': True}
    check('output_metadata_cannot_be_silently_recertified_by_an_edited_inference_receipt',
          rejects(lambda: q._load_output(forged_certificate, identity=identity_for_saved, shape=(2, 2))))
    check('same_registration_same_method_same_raw_cache_key_is_stable_and_method_specific',
          q._cache_key(cell, digest) == q._cache_key(copy.deepcopy(cell), digest)
          and q._cache_key(cell, digest) != q._cache_key(planned[1], digest)
          and q._cache_key(cell, digest) != q._cache_key(cell, digest+'-changed'))

    summary = q.summarize_cells(planned, [second])
    method_summary = summary['by_method'][cell['method']]
    check('failure_and_unattempted_cells_remain_in_fixed_denominator',
          summary['planned_cells'] == 6 and len(summary['cells']) == 6
          and summary['attempted_cells'] == 1 and summary['finite_output_cells'] == 1
          and summary['numerically_certified_cells'] == 0
          and method_summary['planned_cases'] == 2 and method_summary['finite_scored_cases'] == 1
          and method_summary['failed_or_unavailable_scoring_cases'] == 1
          and method_summary['complete_denominator_template_RMSE_e'] is None
          and method_summary['conditional_finite_case_template_RMSE_e'] == 1.)

    interrupted = planned[1]
    interrupted_key = q._cache_key(interrupted, digest)
    interrupted_folder = report/'interrupted/cells'/interrupted_key
    identity = {'registration_sha256': digest, 'method': interrupted['method'],
                'raw_sha256': interrupted['raw_sha256'], 'cache_key': interrupted_key,
                'case_id': interrupted['case_id']}
    s.write(interrupted_folder/'STARTED.json', identity)
    before = len(calls)
    row = q.evaluate_cell(interrupted, **{**common, 'report_dir': report/'interrupted',
        'data_dir': data/'interrupted', 'score_action': score_action})
    check('interrupted_started_cell_is_terminal_and_never_automatically_repeated',
          row['inference_status'] == 'INTERRUPTED_WITHOUT_SAVED_OUTPUT'
          and row['inference_attempted'] and not row['finite_output'] and len(calls) == before)

    def fail_action():
        calls.append('failed_action')
        raise ArithmeticError('Injected numerical failure')

    failure_kwargs = {**common, 'report_dir': report/'failed', 'data_dir': data/'failed',
                      'action': fail_action, 'score_action': score_action}
    before = len(calls)
    failed = q.evaluate_cell(cell, **failure_kwargs)
    failed_again = q.evaluate_cell(cell, **failure_kwargs)
    check('failed_inverse_receipt_is_reused_and_does_not_shrink_denominator',
          failed['inference_status'] == 'INFERENCE_OR_PERSISTENCE_FAILED'
          and failed['inference_attempted'] and not failed['finite_output']
          and failed_again['reused_inference'] and len(calls) == before+2
          and q.summarize_cells(planned, [failed])['planned_cells'] == 6)

    recovery_key = q._cache_key(cell, digest+'-recovery')
    recovery_identity = {**identity, 'registration_sha256': digest+'-recovery',
                         'method': cell['method'], 'case_id': cell['case_id'],
                         'raw_sha256': cell['raw_sha256'], 'cache_key': recovery_key}
    q._save_output(data/'recovery'/(recovery_key+'.npz'), output(), recovery_identity,
        compact_info=common['compact_info'], output_budget_bytes=2**20,
        data_dir=data/'recovery', shape=(2, 2))
    before = len(calls)
    recovery_score = lambda out: {'status': 'FINITE_NUMERICALLY_UNCERTIFIED',
                                  'production_numerical_certified': False, 'metrics': None}
    recovered = q.evaluate_cell(cell, **{**common, 'registration_sha256': digest+'-recovery',
        'report_dir': report/'recovery', 'data_dir': data/'recovery', 'score_action': recovery_score})
    check('complete_npz_recovers_after_interrupted_terminal_receipt_without_refit',
          recovered['recovered_after_inverse_receipt_interruption'] and len(calls) == before
          and recovered['finite_output'])
    def broken_compactor(out):
        raise RuntimeError('Injected diagnostic history formatting error')
    compact_identity = {**recovery_identity, 'cache_key': recovery_key+'-compact'}
    compact_receipt = q._save_output(data/'compaction'/'finite.npz', output(), compact_identity,
        compact_info=broken_compactor, output_budget_bytes=2**20,
        data_dir=data/'compaction', shape=(2, 2))
    check('diagnostic_compaction_failure_still_saves_exact_finite_canonical_output',
          compact_receipt['finite_output']
          and compact_receipt['info']['quick_info_compaction_error']['error_type'] == 'RuntimeError'
          and np.array_equal(s.arrays(compact_receipt['output_cache']['path'])['u'], values))
    negative = output(certified=True)
    negative['predicted_raw_e'] = -np.ones((1, 2, 2), dtype=np.float64)
    negative_receipt = q._save_output(data/'negative'/'finite.npz', negative,
        {**compact_identity, 'cache_key': recovery_key+'-negative'},
        compact_info=common['compact_info'], output_budget_bytes=2**20,
        data_dir=data/'negative', shape=(2, 2))
    check('finite_negative_measurement_is_preserved_and_cannot_be_numerically_certified',
          negative_receipt['finite_output'] and not negative_receipt['predicted_measurement_nonnegative']
          and not negative_receipt['production_numerical_certified'])

    def guard_failure():
        raise RuntimeError('Injected two-GiB disk floor')

    before = len(calls)
    check('disk_floor_and_source_edit_fail_before_count_or_inverse',
          rejects(lambda: q.evaluate_cell(cell, **{**common, 'report_dir': report/'disk_floor',
              'data_dir': data/'disk_floor', 'score_action': score_action, 'budget_guard': guard_failure}), RuntimeError)
          and rejects(lambda: q.evaluate_cell(cell, **{**common, 'report_dir': report/'source_edit',
              'data_dir': data/'source_edit', 'score_action': score_action,
              'verify_sources': lambda: q.verify_hashes(hashes, root=report)}), q.QuickIntegrityError)
          and len(calls) == before)
    cache_path = Path(second['output_cache']['path'])
    with cache_path.open('ab') as handle:
        handle.write(b'injected-corruption')
    before = len(calls)
    check('modified_output_cache_is_quarantined_without_refit',
          rejects(lambda: q.evaluate_cell(cell, score_action=score_action, **common), q.QuickIntegrityError)
          and len(calls) == before)

    result = {'time': s.now(), 'all_passed': all(v['passed'] for v in checks), 'checks': checks,
              'source_sha256': {name: s.sha(s.ROOT/name) for name in
                               ('research_closeout/quick_validation.py', 'research_closeout/tests/quick_validation.py')},
              'fixture_report_directory': str(report), 'fixture_data_directory': str(data),
              'formal_complete_inversions': 0, 'GPU_used': False,
              'formal_raw_or_truth_payloads_opened': False, 'final_payloads_opened': False,
              'fixture_inferences_are_numpy_callbacks': True,
              'scientific_performance_or_numerical_qualification_claimed': False}
    if result['all_passed'] and not q.CONTROLLER_CONTRACT.exists():
        destination = q.CONTROLLER_CONTRACT
    else:
        suffix = 'INITIAL' if not result['all_passed'] else 'RECHECK'
        destination = q.REPORT_DIR/f'QUICK_CONTROLLER_CONTRACTS_{suffix}_{token}.json'
    s.write(destination, result)
    print(json.dumps({**result, 'receipt': str(destination)}, ensure_ascii=True, indent=2), flush=True)
    return result


def _enter_lock(*paths):
    with q.phase_locks(paths, registration_sha256='fixture-second-owner'):
        pass


if __name__ == '__main__':
    raise SystemExit(0 if run()['all_passed'] else 2)
