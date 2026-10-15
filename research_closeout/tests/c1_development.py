"""CPU contracts for the finite prescreen; no formal registration or GPU."""
from .. import storage as s
from .. import c1_development as dev
from .. import compact_joint_cache as compact
from .. import c1_small_panel as panel
from .c1_small_panel import FakeStore, FakeCache, fixture_refs
from .compact_joint_cache import FixturePhysics, original_fixture
import argparse
import ast
import copy
import hashlib
import inspect
import io
import contextlib
import json
from pathlib import Path
import uuid
from unittest import mock
import numpy as np
import torch


def certified_info():
    stages = [{'stage': i+1, 'complete': True, 'qualified_endpoint': True,
               'configuration': {'gtol': gtol}} for i, gtol in enumerate((1e-4, 1e-5, 1e-6))]
    return {'numerical_engine': {'numerical_certified': True, 'stages': stages, 'returned_stage': 3,
        'required_stable_comparisons': 2, 'observed_consecutive_stable_comparisons': 2,
        'output_comparisons': [{'from_stage': 1, 'to_stage': 2, 'passed': True},
                              {'from_stage': 2, 'to_stage': 3, 'passed': True}],
        'returned_gradient': {'scaled_projected_linf': 8e-7}}}


class ControllerCache(FakeCache):
    def __init__(self, store, name, events, **kwargs):
        super().__init__(store, name, events, **kwargs)
        self.store = store

    def persist(self, ticket, out, raw):
        result = super().persist(ticket, out, raw)
        self.store.write(self.report/'PREPARED.json', {'original_info': out.get('info', {})})
        return result

    def _read(self, name):
        return self.store.read(self.report/name)


def run(output=None):
    entry = panel.execution_entry(); torch.set_num_threads(2); s.budget_guard()
    token = uuid.uuid4().hex
    root = dev.CPU_REPORT_DIR/'fixtures'/token
    root.mkdir(parents=True, exist_ok=False)
    checks = []
    protected = [path for path in (s.ROOT/'research_closeout').rglob('*.py')
                 if path.relative_to(s.ROOT).as_posix() not in dev.SOURCES]
    protected += [dev.c2.REGISTRATION, dev.c2.REPORT_DIR/'QUICK_RESULT.json',
                  panel.REPORT_DIR/panel.REGISTRATION_NAME, panel.REPORT_DIR/panel.RESULT_NAME,
                  panel.REPORT_DIR/panel.QUALIFICATION_NAME, s.REPORTS/'inverse_calls.jsonl']
    before = {str(path): s.sha(path) for path in protected if path.exists()}

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **s.plain(evidence)})

    def rejects(action):
        try:
            action()
        except (dev.PrescreenIntegrityError, compact.CacheIntegrityError, compact.CacheNotScoreable,
                panel.PanelIntegrityError, FileExistsError, FileNotFoundError, RuntimeError, ValueError):
            return True
        return False

    reference = fixture_refs()[0]
    planned = dev.planned_cells(reference)
    base = [cell for cell in planned if cell['physical_options'] == dev.q.PHYSICAL_OPTIONS]
    origins = [{'row': {**cell, 'registration_sha256': 'old-registration', 'status': 'FINITE_NUMERICALLY_CERTIFIED',
        'score_status': 'SCORED', 'finite_output': True, 'production_numerical_certified': True,
        'metrics': {'temperature_RMSE_K': 99., 'template_RMSE_e': 99.}, 'info': certified_info()},
        'origin_kind': 'fixture-v002'} for cell in base]
    catalog = dev.reuse_catalog(planned, origins+[copy.deepcopy(origins[0])])
    bound = dev.bind_plan(planned, catalog)
    reused = [dev.reuse_row(cell, catalog[cell['reuse_key']], 'new-reg') for cell in bound if cell['execution_role'] == 'REUSE_PRIOR_C1V3']
    check('original_one_raw_nine_pairs_two_initializers_and_only_sixteen_new_calls',
          len(bound) == 18 and len(base) == 2 and len(catalog) == 2
          and sum(cell['execution_role'] == 'NEW_INVERSE' for cell in bound) == 16
          and len({(cell['candidate_id'], cell['initializer']) for cell in bound}) == 18
          and dev.CONFIG['maximum_new_inverse_wall_s_excluding_overhead'] == 2880
          and dev.CONFIG['numerical_options'] == {'cost_limit_s': 180.}
          and {cell['physical_options']['grid'] for cell in bound} == {256})
    check('wrong_raw_cohort_partition_extra_candidate_or_initializer_is_rejected',
          all(rejects(lambda altered=altered: dev.planned_cells({**reference, **altered})) for altered in
              ({'case_id': 'final_payload'}, {'cohort': 'sealed'}, {'partition_role': 'confirmation'},
               {'condition': 'negative_normal'}, {'raw_reference': {'shape': [128, 128]}}))
          and rejects(lambda: dev.method_for({'id': 'tenth'}, 'v3'))
          and rejects(lambda: dev.method_for(dev.CANDIDATES[0], 'truth')))
    mixed = dev.summary(bound, reused+[dict(bound[0], status='INFERENCE_FAILED_NO_RERUN',
        finite_output=False, production_numerical_certified=False, new_inverse_calls_this_run=1)])
    check('all_eighteen_cells_and_nine_candidate_denominators_survive_failure_and_early_stop',
          mixed['planned_cells'] == len(mixed['cells']) == 18 and mixed['returned_cell_rows'] == 3
          and len(mixed['candidate_numerical_qualification']) == 9
          and mixed['not_executed_cells'] == 15 and mixed['reused_prior_cells'] == 2
          and mixed['new_inverse_calls_this_run'] == 1 and not mixed['selection12_or_confirmation24_started'])
    evidence = dev.numerical_evidence(certified_info(), True)
    check('stage3_requested_1e_minus6_with_three_endpoints_two_plateaus_is_not_misreported_as_1e_minus7',
          evidence['passed'] and evidence['actual_requested_endpoint_gtol'] == 1e-6
          and evidence['returned_scaled_projected_linf'] > 1e-7
          and evidence['three_qualified_endpoints'] and evidence['two_successive_output_plateaus'])
    bad = certified_info(); bad['numerical_engine']['output_comparisons'][-1]['passed'] = False
    false = copy.deepcopy(reused)
    false[0].update(production_numerical_certified=False,
        numerical_evidence=dev.numerical_evidence(bad, False), metrics={'temperature_RMSE_K': 0., 'template_RMSE_e': 0.})
    check('beautiful_truth_scores_cannot_replace_both_initializer_numerical_qualification',
          dev.summary(bound, reused)['numerically_qualified_candidates'] == 1
          and dev.summary(bound, false)['numerically_qualified_candidates'] == 0
          and all(not row['truth_scores_used_for_qualification'] for row in mixed['candidate_numerical_qualification']))
    retained = []
    for status in ('FINITE_NUMERICALLY_UNCERTIFIED', 'INFERENCE_FAILED_NO_RERUN', 'INTERRUPTED_OR_PARAMETERS_PENDING'):
        item = copy.deepcopy(origins[0]); item['row'].update(status=status, finite_output='FINITE' in status,
            production_numerical_certified=False, score_status='NOT_SCOREABLE', metrics=None)
        found = dev.reuse_catalog(planned, [item])
        candidate_plan = dev.bind_plan(planned, found)
        chosen = next(cell for cell in candidate_plan if cell['execution_role'] == 'REUSE_PRIOR_C1V3')
        row = dev.reuse_row(chosen, found[chosen['reuse_key']], 'another-registration-name')
        retained.append(row['status'] == status and not row['new_inverse_calls_this_run'] and row['inference_reused'])
    changed = copy.deepcopy(origins[0]); changed['row']['raw_sha256'] = '0'*64
    conflicting = copy.deepcopy(origins[0]); conflicting['row']['status'] = 'FAILED'
    check('all_prior_uncertified_failed_and_interrupted_states_block_renamed_registration_refits',
          all(retained) and rejects(lambda: dev.reuse_catalog(planned, [changed]))
          and rejects(lambda: dev.reuse_catalog(planned, [origins[0], conflicting])))
    call_only = {'phase': dev.PHASE, 'method': planned[0]['method'], 'case_id': dev.CASE_ID, 'reused': False}
    counted = dev.reuse_catalog(planned, origins, [call_only])
    counted_plan = dev.bind_plan(planned, counted)
    check('call_ledger_without_output_is_an_interrupted_attempt_not_permission_to_optimize_again',
          sum(cell['execution_role'] == 'NEW_INVERSE' for cell in counted_plan) == 15
          and any(item.get('origin_kind') == 'counted_interrupted_attempt' for item in counted.values()))

    budget = dev.StudyBudget((root,), free_bytes=2*2**30+100)
    synthetic = {'instance': {'bytes': 0}, 'global_new_study': {'bytes': 4*2**30-100}}
    with mock.patch.object(budget, 'ledger', return_value=synthetic):
        exact = budget.check(100); over = rejects(lambda: budget.check(101))
    check('new_E_budget_counts_all_four_roots_and_exact4GiB_2GiB_boundaries',
          budget.global_roots == tuple(path.resolve() for path in dev.GLOBAL_ROOTS)
          and exact['free_after_reserve_bytes'] == 2*2**30 and over
          and dev.execution_format_contract(dev.EStudyStore())['global_budget_roots'] == dev.EStudyStore().contract()['global_budget_roots']
          and dev.execution_format_contract(dev.EStudyStore())['default_E_owner_release_unchanged']
          and rejects(lambda: dev.EStudyStore().write_path(panel.DATA_DIR/'forbidden_new_D_file')))
    prefix_path = root/'ledger_prefix.jsonl'; prefix_path.write_bytes(b'old\n')
    prefix = {'path': str(prefix_path), 'prefix_bytes': 4, 'prefix_sha256': hashlib.sha256(b'old\n').hexdigest()}
    with prefix_path.open('ab') as handle: handle.write(b'new\n')
    dev.verify_call_prefix(prefix)
    prefix_path.write_bytes(b'bad\nnew\n')
    check('append_only_counter_prefix_accepts_own_new_calls_and_rejects_history_edits',
          rejects(lambda: dev.verify_call_prefix(prefix)))
    from .. import experiment_registry as registry
    with mock.patch.object(dev, 'call_snapshot', return_value=([call_only], {})), mock.patch.object(registry, 'count_call') as actual:
        duplicate = rejects(lambda: dev.counted_once(planned[0]['method'], dev.CASE_ID))
        check('known_attempt_cannot_be_counted_again_and_undeclared_counter_calls_are_refused',
              duplicate and not actual.called and rejects(lambda: dev.counted_once('invented-method', dev.CASE_ID)))
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})), mock.patch.object(registry, 'count_call') as actual:
        dev.counted_once(planned[0]['method'], dev.CASE_ID)
        check('new_call_uses_explicit_prescreen_phase_and_original_method_raw_identity',
              actual.call_args.args == (dev.PHASE, planned[0]['method'], dev.CASE_ID))
    at_cap = [{'phase': dev.PHASE, 'method': f'other-{i}', 'case_id': 'earlier', 'reused': False} for i in range(16)]
    with mock.patch.object(dev, 'call_snapshot', return_value=(at_cap, {})), mock.patch.object(registry, 'count_call') as actual:
        check('prescreen_sixteen_new_call_cap_precedes_global_counter_and_callback',
              rejects(lambda: dev.counted_once(planned[0]['method'], dev.CASE_ID)) and not actual.called)

    store = FakeStore(root/'controller'); events = []; cache = ControllerCache(store, 'retry', events)
    counts = []; scores = []
    def action():
        events.append('inverse'); return {'numerical_valid': False, 'info': {'numerical_certified': False}}
    def score(out):
        scores.append(True); events.append('external_truth_score')
        if len(scores) == 1: raise ArithmeticError('external score failed')
        return {'status': 'FINITE_NUMERICALLY_UNCERTIFIED', 'metrics': {'temperature_RMSE_K': 0.}}
    def count(*args): counts.append(args); events.append('count')
    arguments = dict(cache=cache, raw='raw', action=action, score_action=score, count_call=count,
                     store=store, digest='prescreen-fixture')
    first = dev.evaluate_cell(bound[0], **arguments)
    second = dev.evaluate_cell(bound[0], **arguments, score_only=True)
    third = dev.evaluate_cell(bound[0], **arguments, score_only=True)
    check('parameter_and_byte_cache_precede_truth_and_score_only_retry_never_optimizes_or_recounts',
          len(counts) == 1 and counts[0][0] == dev.PHASE and events.count('inverse') == 1
          and len(scores) == 2 and first['score_status'] == 'SCORING_FAILED'
          and second['score_status'] == third['score_status'] == 'SCORED'
          and events.index('parameters_persisted') < events.index('canonical_byte_verified') < events.index('external_truth_score'))
    absent_store = FakeStore(root/'absent'); absent_events = []; absent = ControllerCache(absent_store, 'absent', absent_events)
    untouched = dev.evaluate_cell(bound[1], cache=absent, raw='raw', action=lambda: absent_events.append('inverse'),
        score_action=lambda out: {}, count_call=lambda *a: absent_events.append('count'), store=absent_store,
        digest='score-only-new', score_only=True)
    check('score_only_on_a_new_cell_does_not_even_consume_a_ticket',
          untouched['status'] == 'NOT_EXECUTED_SCORE_ONLY' and not absent_events and not absent.started)
    reuse_store = FakeStore(root/'reuse_score'); reuse_events = []
    old_cache = ControllerCache(reuse_store, 'old_state', reuse_events)
    old_cache.persist(old_cache.begin('raw'), {'numerical_valid': False, 'info': {'numerical_certified': False}}, 'raw')
    old_item = copy.deepcopy(origins[0])
    old_item['row'].update(production_numerical_certified=False, info={}, score_status='SCORING_FAILED')
    old_cell = next(cell for cell in bound if dev.semantic_key(cell) == dev.semantic_key(old_item['row']))
    reuse_scores = []
    def retry_old_score(out):
        reuse_scores.append(True)
        return {'status': 'FINITE_NUMERICALLY_UNCERTIFIED', 'metrics': {'template_RMSE_e': 7.}}
    prior_retry = dev.evaluate_reused_cell(old_cell, old_item, cache=old_cache, raw='raw',
        score_action=retry_old_score, store=reuse_store, digest='reuse-score')
    prior_cached = dev.evaluate_reused_cell(old_cell, old_item, cache=old_cache, raw='raw',
        score_action=retry_old_score, store=reuse_store, digest='reuse-score')
    check('a_prior_registration_scoring_failure_retries_only_score_and_retains_original_failed_score_evidence',
          prior_retry['score_status'] == prior_cached['score_status'] == 'SCORED'
          and len(reuse_scores) == 1 and not prior_retry['new_inverse_calls_this_run']
          and prior_retry['origin_evidence']['row']['score_status'] == 'SCORING_FAILED'
          and prior_retry['origin_scoring_retry_only'] and 'inverse' not in reuse_events)
    interrupted_store = FakeStore(root/'interrupted'); stopped_events = []
    interrupted = ControllerCache(interrupted_store, 'stopped', stopped_events); interrupted.begin('raw')
    resumed = dev.evaluate_cell(bound[1], cache=interrupted, raw='raw', action=lambda: stopped_events.append('inverse'),
        score_action=lambda out: {}, count_call=lambda *a: stopped_events.append('count'),
        store=interrupted_store, digest='interrupted-fixture')
    check('interrupted_cache_retains_failure_denominator_and_cannot_start_a_second_inverse',
          resumed['controller_stop'] and not resumed['finite_output'] and 'inverse' not in stopped_events and 'count' not in stopped_events)
    with mock.patch.object(dev, 'call_snapshot', return_value=([dict(call_only, method=bound[1]['method'])], {})):
        counted_resume = dev.evaluate_cell(bound[1], cache=interrupted, raw='raw', action=lambda: None,
            score_action=lambda out: {}, count_call=lambda *a: None, store=interrupted_store, digest='interrupted-fixture')
    check('global_count_append_before_local_receipt_crash_remains_billed_without_restarting_callback',
          counted_resume['new_inverse_calls_panel_total'] == 1
          and counted_resume['global_counted_prescreen_attempts_for_cell'] == 1
          and not counted_resume['new_inverse_calls_this_run'])
    source_store = FakeStore(root/'changed_source'); source_events = []; source_cache = ControllerCache(source_store, 'source', source_events)
    def changed_source(): raise dev.PrescreenIntegrityError('source changed')
    stopped = dev.evaluate_cell(bound[1], cache=source_cache, raw='raw', action=lambda: source_events.append('inverse'),
        score_action=lambda out: {}, count_call=lambda *a: source_events.append('count'), store=source_store,
        digest='source-fixture', verify_sources=changed_source)
    check('source_hash_failure_stops_before_begin_count_or_inverse', stopped['controller_stop'] and not source_events)

    raw_dn = np.array([[100, 101], [102, 103]], dtype=np.uint16)
    raw_file = root/'manufactured_raw.npz'; s.save(raw_file, raw_dn=raw_dn)
    raw_ref = {'path': str(raw_file), 'sha256': s.sha(raw_file), 'array_sha256': compact.array_sha(raw_dn),
               'dtype': 'uint16', 'shape': [2, 2]}
    raw = torch.as_tensor(raw_dn.astype(np.float64))[None]
    all_physics = []
    def native_cache(name):
        physics = FixturePhysics(); all_physics.append(physics)
        report, data = root/name/'reports', root/name/'data'
        budget = compact.Budget((report, data), global_roots=(root,))
        instance = dev.ECompactCache(report, data, p=physics,
            identity={'registration_sha256': 'CPU-fixture', 'method': 'CPU-C1v3-fixture', 'case_id': name},
            input_reference=raw_ref, calibration={'fingerprint': 'a'*64, 'physical_objective_version': 'CPU-fixture'},
            config={'grid': 2}, budget=budget)
        return instance, physics, original_fixture(physics, raw)
    real, physics, out = native_cache('qualification_namespace/inference')
    ticket = real.begin(raw); reserve = Path(real._read('RESERVATION.json')['reservation']['path'])
    accepted = real.persist(ticket, out, raw); restored = real.load_for_score(raw)
    check('default_E_nested_owner_cache_releases_own_reserve_and_preserves_every_float64_byte',
          accepted['status'] == 'LOSSLESS_READY' and not reserve.exists()
          and real.storage is s and real.context['storage_contract'] is None
          and all(compact.descriptor(restored[key]) == compact.descriptor(out[key])
                  for key in compact.CANONICAL_KEYS+('continuum_latent',))
          and not real.begin(raw)['run_new_inverse'] and physics.inverse_calls == 0)
    failed, pf, badout = native_cache('fallback')
    badout['predicted_raw_e'] = badout['predicted_raw_e'].clone()
    badout['predicted_raw_e'][0, 0, 0] = torch.nextafter(badout['predicted_raw_e'][0, 0, 0], torch.tensor(float('inf')))
    failed_result = failed.persist(failed.begin(raw), badout, raw)
    check('initial_one_bit_mismatch_keeps_full_original_fallback_and_permanently_forbids_score_or_inverse',
          failed_result['original_fallback_available'] and not failed_result['score_allowed']
          and all(compact.descriptor(failed.load_fallback()[key]) == compact.descriptor(badout[key]) for key in compact.CANONICAL_KEYS)
          and rejects(lambda: failed.load_for_score(raw)) and not failed.begin(raw)['run_new_inverse'])
    crash, pc, crashout = native_cache('parameters_before_exit'); crash_ticket = crash.begin(raw)
    with mock.patch.object(crash, '_verify_forward', side_effect=KeyboardInterrupt('manufactured exit')):
        try: crash.persist(crash_ticket, crashout, raw)
        except KeyboardInterrupt: pass
    check('parameter_first_interruption_recovers_by_forward_only_with_no_new_ticket',
          crash.resume(raw)['status'] == 'LOSSLESS_READY' and not crash.begin(raw)['run_new_inverse'] and pc.inverse_calls == 0)
    blocked, pb, blockedout = native_cache('floor_after_inverse'); blocked_ticket = blocked.begin(raw)
    with mock.patch.object(blocked.budget, 'check', side_effect=compact.CacheIntegrityError('simulated E floor loss')):
        unavailable = blocked.persist(blocked_ticket, blockedout, raw)
    check('post_inverse_storage_failure_is_explicit_memory_only_no_claimed_fallback_and_never_refits',
          unavailable['status'] == 'FAIL_CLOSED_ORIGINAL_PERSISTENCE_UNAVAILABLE'
          and not unavailable['durable_failure_receipt_written'] and not unavailable['original_fallback_available']
          and not unavailable['score_allowed'] and not blocked.begin(raw)['run_new_inverse'] and pb.inverse_calls == 0)

    lock_store = FakeStore(root/'locks'); lock1 = lock_store.report_root/'a.json'; lock2 = lock_store.report_root/'b.json'
    with dev.phase_locks(lock_store, 'fixture', 'CPU', paths=(lock1, lock2)):
        def occupied():
            with dev.phase_locks(lock_store, 'other', 'CPU', paths=(lock1,)): pass
        exclusive = rejects(occupied)
    released = not lock1.exists() and not lock2.exists()
    with dev.phase_locks(lock_store, 'fixture', 'CPU', paths=(lock1,)):
        lock1.write_text('{"pid":-1,"owner_token":"foreign"}', encoding='utf8')
    check('single_GPU_locks_are_exclusive_and_finally_never_deletes_a_foreign_replacement',
          exclusive and released and s.read(lock1)['owner_token'] == 'foreign')
    contract_file = root/'CPU_BINDINGS.json'
    source_hashes = {name: s.sha(s.ROOT/name) for name in (*dev.SOURCES,
        'research_closeout/runtime_entry.py', 'research_closeout/compact_joint_cache.py',
        'research_closeout/c1_small_panel.py', 'research_closeout/tests/compact_joint_cache.py')}
    s.write(contract_file, {'all_passed': True, 'E_compact_persistence_contracts_executed': True,
                          'source_sha256': source_hashes})
    _, hashes = dev.contract_sources([contract_file])
    stale_file = root/'STALE_BINDINGS.json'
    s.write(stale_file, {'all_passed': True, 'E_compact_persistence_contracts_executed': True,
        'source_sha256': {**source_hashes, dev.SOURCES[0]: '0'*64}})
    arithmetic_file = root/'ARITHMETIC_ONLY.json'
    s.write(arithmetic_file, {'all_passed': True, 'E_compact_persistence_contracts_executed': False,
                            'source_sha256': source_hashes})
    check('registration_requires_actual_E_persistence_and_exact_new_source_contracts',
          hashes == source_hashes and rejects(lambda: dev.contract_sources([stale_file]))
          and rejects(lambda: dev.contract_sources([arithmetic_file])))
    qualification = panel.qualification_matrix([{'case_id': str(i), 'method': 'saved'} for i in range(4)],
        lambda origin: {'passed': origin['case_id'] != '1', 'complete_inverse_calls': 0})
    check('four_existing_endpoint_E_qualification_preserves_failed_denominator_and_zero_inverse_scope',
          qualification['audited_existing_endpoints'] == 4 and qualification['passed_existing_endpoints'] == 3
          and not qualification['all_passed'] and qualification['complete_inverse_calls'] == 0)
    tree = ast.parse(inspect.getsource(dev.qualify))
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    check('qualification_has_forward_saved_state_cache_calls_and_no_optimizer_or_counter',
          {'forward_reconstruct', 'load_parameters', 'persist'}.issubset(calls)
          and not calls.intersection({'repaired_infer', 'minimize', 'lsmr', 'count_call', 'cached_v3', 'initialize'}))
    prepared = dev.prepare_plan()
    check('actual_immutable_v002_v003_metadata_gives_exactly_two_prior_and_sixteen_new_cells',
          len(prepared['planned_cells']) == 18
          and sum(cell['execution_role'] == 'REUSE_PRIOR_C1V3' for cell in prepared['planned_cells']) == 2
          and sum(cell['execution_role'] == 'NEW_INVERSE' for cell in prepared['planned_cells']) == 16
          and prepared['reference']['case_id'] == dev.CASE_ID)
    check('all_existing_python_sources_old_results_and_actual_inverse_ledger_are_byte_unchanged',
          before == {str(path): s.sha(path) for path in protected if path.exists()}, protected_files=len(before))
    check('no_native_GPU_truth_or_formal_inverse_was_used_by_the_CPU_fixtures',
          all(physics.inverse_calls == 0 and physics.M.device.type == 'cpu' for physics in all_physics)
          and dev.CONFIG['detector'] is None and not dev.CONFIG['selection12_or_confirmation24_implemented'])
    result = {'schema': dev.VERSION+'-CPU-contracts', 'time': s.now(), 'all_passed': all(row['passed'] for row in checks),
        'checks': checks, 'source_sha256': source_hashes, 'execution_entry': entry,
        'storage_contract': dev.EStudyStore().contract(), 'format_contract': dev.execution_format_contract(dev.EStudyStore()),
        'E_compact_persistence_contracts_executed': True, 'GPU_used': False, 'formal_complete_inversions': 0,
        'truth_loaded': False, 'formal_registration_or_qualification_or_prescreen_executed': False,
        'D_writes': 0, 'old_data_converted_deleted_or_migrated': False,
        'protected_file_sha256_before': before, 'protected_file_sha256_after': {str(path): s.sha(path) for path in protected if path.exists()},
        'fixture_root': str(root), 'fixture_storage_ledger': compact.tree_ledger((root,)),
        'actual_planned_new_calls': sum(cell['execution_role'] == 'NEW_INVERSE' for cell in prepared['planned_cells']),
        'native_GPU_E_cache_qualification_performed': False,
        'native_frame_conservative_plan': compact.frame_plan(shape=(256, 256), bands=7, template_dimension=1153, grid=256)}
    destination = Path(output) if output else dev.CPU_REPORT_DIR/('CONTRACTS_'+token+'.json')
    s.write(destination, result)
    print(json.dumps({'receipt': str(destination), 'sha256': s.sha(destination), 'all_passed': result['all_passed'],
        'checks': len(checks), 'failed': [row for row in checks if not row['passed']],
        'source_sha256': source_hashes, 'formal_complete_inversions': 0, 'GPU_used': False,
        'fixture_bytes': result['fixture_storage_ledger']['bytes']}, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    raise SystemExit(0 if run(parser.parse_args().output)['all_passed'] else 2)
