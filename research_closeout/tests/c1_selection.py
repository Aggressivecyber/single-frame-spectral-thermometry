"""CPU first-stage contracts; saved development inputs, tiny forward fixtures.

Never calls formal registration, native qualification, a GPU or an inverse.
Every fixture/report is new and retained under this task's E report root.
"""
from .. import storage as s
from .. import c1_selection as sel
from .. import c1_development as dev
from .. import c1_small_panel as panel
from .. import compact_joint_cache as compact
from .c1_small_panel import FakeStore
from .c1_development import ControllerCache, certified_info
from .compact_joint_cache import FixturePhysics, original_fixture
import argparse
import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import time
import uuid
from unittest import mock
import numpy as np
import torch


def score_metrics(value=0.):
    return {'domain_pixels': 4, 'template_SSE': value, 'temperature_SSE_K2': value,
            'template_RMSE_e': float(np.sqrt(value/4)), 'temperature_RMSE_K': float(np.sqrt(value/4))}


def run(output=None):
    entry = panel.execution_entry(); torch.set_num_threads(2); s.budget_guard()
    started = time.monotonic(); token = uuid.uuid4().hex
    root = sel.CPU_REPORT_DIR/'fixtures'/token
    root.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **s.plain(evidence)})

    def rejects(action):
        try: action()
        except (RuntimeError, ValueError, KeyError, FileNotFoundError, FileExistsError): return True
        return False

    prepared = sel.prepare_plan()
    frozen = {**prepared['inherited_source_sha256']}
    protected = [s.ROOT/name for name in frozen]
    protected += [panel.c2.REGISTRATION, panel.c2.REPORT_DIR/'QUICK_RESULT.json',
        panel.q.REPORT_DIR/'QUICK_REGISTRATION.json', panel.q.REPORT_DIR/'QUICK_RESULT.json',
        panel.REPORT_DIR/panel.REGISTRATION_NAME, panel.REPORT_DIR/panel.RESULT_NAME,
        panel.REPORT_DIR/panel.QUALIFICATION_NAME, dev.REPORT_DIR/dev.REGISTRATION_NAME,
        dev.REPORT_DIR/dev.RESULT_NAME, dev.REPORT_DIR/dev.QUALIFICATION_NAME,
        s.REPORTS/'inverse_calls.jsonl']
    protected += [Path(ref['path']) for ref in prepared['read_dependencies']]
    before = {str(path): s.sha(path) for path in protected if path.exists()}
    plan = prepared['planned_cells']; refs = prepared['references']
    reg = {**prepared, 'source_sha256': frozen, 'format_contract': dev.execution_format_contract(sel.SelectionStore()),
           'store_contract': sel.SelectionStore().contract()}
    blank = next(ref for ref in refs if ref['mother_id'].endswith('_0000') and ref['condition'] == 'blank')
    blank_cells = [cell for cell in plan if cell['case_id'] == blank['case_id']]
    cell = next(cell for cell in blank_cells if cell['execution_role'] == 'NEW_INVERSE')
    check('actual_complete_Q9_first_stage_is1080_with28_unique_old_cells_and_at_most1052_new',
        len(plan) == 1080 and len(refs) == 60 and len({ref['mother_id'] for ref in refs}) == 12
        and len(prepared['reuse_catalog']) == 28
        and sum(row['execution_role'] == 'NEW_INVERSE' for row in plan) == 1052
        and {sum(row['candidate_id'] == candidate['id'] for row in plan) for candidate in dev.CANDIDATES} == {120}
        and sel.CONFIG['numerical_options'] == {'cost_limit_s': 180.}
        and {row['physical_options']['grid'] for row in plan} == {256})
    check('root_first_blank_raw_keeps_two_original_C1s1_routes_and_only_sixteen_new',
        blank['case_id'] == 'v5_development_selection_mother_0000_blank_noise0'
        and len(blank_cells) == 18 and sum(row['execution_role'] == 'NEW_INVERSE' for row in blank_cells) == 16
        and {row['candidate_id'] for row in blank_cells if row['execution_role'] == 'REUSE_PRIOR_C1V3'} == {'joint_c1_s1'})
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})):
        initial_rows = sel.collect_rows(reg, FakeStore(root/'initial_partial'), 'new-reg')
    units = sel.summary(reg, initial_rows)['independent_sample_units']
    check('partial28_prior_cells_have_observed5_mothers6_raws_separate_from_planned12_and60',
        len(initial_rows) == 28 and units['observed_parent_mothers'] == 5 and units['observed_distinct_raws'] == 6
        and units['planned_parent_mothers'] == 12 and units['planned_distinct_raws'] == 60
        and units['method_routes_are_independent_sample_units'] is False)
    check('mother_or_up_to_six_raw_shards_are_explicit_scheduling_and_no_full_run_default',
        len(sel.choose_shard(reg, mother='0')) == 5 and len(sel.choose_shard(reg, raw_ids=[ref['case_id'] for ref in refs[:6]])) == 6
        and sel.choose_shard(reg, raw_ids=[blank['case_id']]) == [blank]
        and all(rejects(action) for action in (
            lambda: sel.choose_shard(reg), lambda: sel.choose_shard(reg, mother='1'),
            lambda: sel.choose_shard(reg, mother='0', raw_ids=[blank['case_id']]),
            lambda: sel.choose_shard(reg, raw_ids=[blank['case_id']]*2),
            lambda: sel.choose_shard(reg, raw_ids=[ref['case_id'] for ref in refs[:7]]),
            lambda: sel.choose_shard(reg, raw_ids=['sealed_final_raw'])))
        and sel.row_key(cell) == sel.row_key(copy.deepcopy(cell)))
    inventory = s.read(panel.q.INVENTORY); spec = s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    selected, all_dev = sel.references(inventory, spec)
    # Alter a definitely selected development row, leaving unrelated cohorts intact.
    changed = copy.deepcopy(inventory)
    target = next(row for row in changed['rows'] if row['case_id'] == blank['case_id'])
    target['cohort'] = 'sealed'
    wrong_shape = copy.deepcopy(inventory)
    next(row for row in wrong_shape['rows'] if row['case_id'] == blank['case_id'])['raw_reference']['shape'] = [128, 128]
    check('only_original_development_noise180_and_protocol_selection60_with_actual_geometry_sources',
        len(all_dev) == 180 and len(selected) == 60 and rejects(lambda: sel.references(changed, spec))
        and rejects(lambda: sel.references(wrong_shape, spec))
        and all(ref['quick_geometry']['temperature_geometry_source'].startswith('research_v6/input_inventory.json:')
                and ref['quick_geometry']['template_geometry_source'].endswith(':template_geometry') for ref in selected)
        and {ref['mother_id'] for ref in selected} == set(spec['development']['selection_mothers']))

    original = copy.deepcopy(next(iter(prepared['reuse_catalog'].values())))
    canonical_method = original['row']['method']
    alias = copy.deepcopy(original); alias['row']['method'] = 'historical-registration-alias'
    alias = sel.normalize_origin(alias, prepared['objective_fingerprints'], prepared['calibration_contract'])
    alias_catalog = dev.reuse_catalog(plan, [alias]); alias_bound = dev.bind_plan(plan, alias_catalog)
    check('different_registration_or_method_label_cannot_refit_the_same_frozen_physical_objective_raw',
        alias['row']['method'] == canonical_method and alias['original_method_label'] == 'historical-registration-alias'
        and alias['alias_resolved_without_inverse'] and len(alias_catalog) == 1
        and sum(row['execution_role'] == 'REUSE_PRIOR_C1V3' for row in alias_bound) == 1)
    rejected = []
    for field, bad in (('C1v3_adapter_fingerprint', 'wrong-cal'), ('objective_fingerprint', 'wrong-objective')):
        altered = copy.deepcopy(original); altered['row'].setdefault('info', {})[field] = bad
        rejected.append(rejects(lambda altered=altered: sel.normalize_origin(altered, prepared['objective_fingerprints'], prepared['calibration_contract'])))
    no_proof = copy.deepcopy(original); no_proof['row']['info'] = {}; no_proof['locator'] = None
    no_proof['row'].pop('compact_report_root', None); no_proof['row'].pop('compact_data_root', None)
    no_proof.pop('frozen_registration_calibration_fingerprint', None)
    no_proof.pop('frozen_registration_objective_fingerprints', None)
    # An alias with no frozen calibration/objective proof cannot be adopted.
    no_proof['row']['method'] = 'unproven-alias'
    rejected.append(rejects(lambda: sel.normalize_origin(no_proof, prepared['objective_fingerprints'], prepared['calibration_contract'])))
    check('calibration_objective_or_unproved_alias_identity_mismatch_is_fail_closed', all(rejected))
    retained = []
    for status in ('FINITE_NUMERICALLY_UNCERTIFIED', 'INFERENCE_FAILED_NO_RERUN', 'INTERRUPTED_OR_PARAMETERS_PENDING'):
        item = copy.deepcopy(alias); item['row'].update(status=status, finite_output=status.startswith('FINITE'),
            production_numerical_certified=False, score_status='NOT_SCOREABLE', metrics=None)
        catalog = dev.reuse_catalog(plan, [item]); bound = dev.bind_plan(plan, catalog)
        chosen = next(row for row in bound if row['execution_role'] == 'REUSE_PRIOR_C1V3')
        reused = dev.reuse_row(chosen, catalog[chosen['reuse_key']], 'new-registration-name')
        retained.append(reused['status'] == status and not reused['new_inverse_calls_this_run'])
    check('old_success_uncertified_failed_and_interrupted_physical_attempts_are_all_reused', all(retained))

    full = [{**row, 'status': 'FINITE_NUMERICALLY_CERTIFIED', 'finite_output': True,
        'production_numerical_certified': True, 'numerical_evidence': dev.numerical_evidence(certified_info(), True),
        'metrics': score_metrics(), 'score_status': 'SCORED', 'new_inverse_calls_selection_total': 0} for row in plan]
    full[0].update(status='INFERENCE_FAILED_NO_RERUN', finite_output=False, production_numerical_certified=False,
                   metrics=None, score_status='NOT_SCOREABLE')
    mixed = sel.summary(reg, full)
    failing_candidate = next(row for row in mixed['candidate_first_stage'] if row['id'] == full[0]['candidate_id'])
    check('a_numerical_failure_preserves_all_remaining119_candidate_cells_and_all1080_denominator',
        len(mixed['cells']) == 1080 and mixed['terminal_outcome_cells'] == 1080
        and mixed['not_executed_cells'] == 0 and failing_candidate['planned_cells'] == 120
        and failing_candidate['numerically_qualified_cells'] == 119 and not failing_candidate['eligible_for_future_root_selection_review']
        and all(row['ordinary_failure_did_not_prune_remaining_cells'] for row in mixed['candidate_first_stage'])
        and not mixed['method_or_initializer_selected'] and not mixed['confirmation24_started'])
    evidence = dev.numerical_evidence(certified_info(), True)
    partial = sel.summary(reg, full[:1])
    check('actual_stage3_1e_minus6_is_retained_and_truth_scores_never_replace_numerical_gate',
        evidence['passed'] and evidence['actual_requested_endpoint_gtol'] == 1e-6
        and evidence['returned_scaled_projected_linf'] > 1e-7
        and partial['not_executed_cells'] == 1079 and all(not row['eligible_for_future_root_selection_review'] for row in partial['candidate_first_stage']))
    stopped = copy.deepcopy(full); stopped[1].update(controller_stop=True, case_state_durable=False)
    stopped_summary = sel.summary(reg, stopped)
    check('interrupted_or_non_durable_state_cannot_qualify_or_close_the_complete_first_stage',
        stopped_summary['terminal_outcome_cells'] == 1079 and stopped_summary['incomplete_or_integrity_stopped_cells'] == 1
        and not next(row for row in stopped_summary['candidate_first_stage'] if row['id'] == stopped[1]['candidate_id'])['eligible_for_future_root_selection_review'])

    budget = sel.SelectionBudget((root,), free_bytes=2*2**30+100)
    synthetic = {'instance': {'bytes': 2*2**30-100}, 'global_new_study': {'bytes': 4*2**30-100}}
    with mock.patch.object(budget, 'ledger', return_value=synthetic):
        exact = budget.check(100); over = rejects(lambda: budget.check(101))
    check('twoGiB_instance_fourGiB_global_twoGiB_E_floor_and_four_E_D_roots_remain_exact',
        exact['free_after_reserve_bytes'] == 2*2**30 and over and budget.instance_limit == 2*2**30
        and budget.global_roots == tuple(path.resolve() for path in dev.GLOBAL_ROOTS)
        and sel.SelectionStore().contract()['D_writes_supported'] is False
        and rejects(lambda: sel.SelectionStore().write_path(panel.DATA_DIR/'forbidden'))
        and rejects(lambda: sel.SelectionStore().write_path(sel.REPORT_DIR/'..'/'outside.json')))
    native_plan = compact.frame_plan(shape=(256, 256), bands=7, template_dimension=1153, grid=256)
    check('1052_success_plus_failure_qualification_and_metadata_plan_fits_instance_without_reducing_Q9',
        native_plan['successful_compact_bound_bytes']*1052 == sel.CONFIG['success1052_conservative_bytes']
        and sel.CONFIG['combined_conservative_plan_bytes'] == sel.CONFIG['success1052_conservative_bytes']+192*2**20+5542016+16*2**20
        and sel.CONFIG['combined_conservative_plan_bytes'] < sel.INSTANCE_LIMIT
        and sel.CONFIG['planned_candidates'] == 9 and sel.CONFIG['planned_cells'] == 1080)
    from .. import experiment_registry as registry
    call = {'phase': sel.PHASE, 'method': cell['method'], 'case_id': cell['case_id'], 'reused': False}
    with mock.patch.object(dev, 'call_snapshot', return_value=([call], {})), mock.patch.object(registry, 'count_call') as actual:
        check('same_method_raw_true_ledger_attempt_blocks_a_renamed_registration_before_new_count',
            rejects(lambda: sel.counted_once(cell, reg)) and not actual.called)
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})), mock.patch.object(registry, 'count_call') as actual:
        sel.counted_once(cell, reg)
        check('one_new_callback_has_exact_phase_original_method_and_raw_and_no_alias_count',
            actual.call_args.args == (sel.PHASE, cell['method'], cell['case_id'])
            and rejects(lambda: sel.counted_once({**cell, 'method': 'invented'}, reg)))
    known = next(row for row in blank_cells if row['execution_role'] == 'REUSE_PRIOR_C1V3')
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})), mock.patch.object(registry, 'count_call') as actual:
        check('frozen_reuse_role_itself_forbids_new_count_even_if_old_ledger_uses_a_label_alias',
            rejects(lambda: sel.counted_once(known, reg)) and not actual.called)
    for name, calls in (
        ('first_stage1052', [{**call, 'method': 'other-'+str(i)} for i in range(1052)]),
        ('semantic_development3400', [{**call, 'phase': 'development', 'method': 'other-'+str(i)} for i in range(3400)])):
        with mock.patch.object(dev, 'call_snapshot', return_value=(calls, {})), mock.patch.object(registry, 'count_call') as actual:
            check(name+'_cap_is_an_integrity_stop_and_not_candidate_filter', rejects(lambda: sel.counted_once(cell, reg)) and not actual.called)
    prefix_path = root/'ledger_prefix.jsonl'; prefix_path.write_bytes(b'old\n')
    prefix = {'path': str(prefix_path), 'prefix_bytes': 4, 'prefix_sha256': hashlib.sha256(b'old\n').hexdigest()}
    with prefix_path.open('ab') as handle: handle.write(b'new\n')
    dev.verify_call_prefix(prefix); prefix_path.write_bytes(b'bad\nnew\n')
    check('source_and_ledger_hash_verification_preserves_old_prefix_but_allows_future_appends',
        rejects(lambda: dev.verify_call_prefix(prefix)) and rejects(lambda: compact.verify_sources({sel.SOURCES[0]: '0'*64})))

    store = FakeStore(root/'controller'); events = []; cache = ControllerCache(store, 'retry', events)
    counts, scores = [], []
    def action():
        events.append('inverse_fixture_callback'); return {'numerical_valid': False, 'info': {'numerical_certified': False}}
    def score(out):
        scores.append(True); events.append('external_score_fixture')
        if len(scores) == 1: raise ArithmeticError('fixture score failure')
        return {'status': 'FINITE_NUMERICALLY_UNCERTIFIED', 'metrics': score_metrics()}
    arguments = dict(cache=cache, raw='fixture', action=action, score_action=score,
        count_call=lambda *args: counts.append(args), store=store, digest='selection-fixture')
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})):
        first = sel.evaluate_cell(cell, **arguments)
        second = sel.evaluate_cell(cell, **arguments, score_only=True)
        third = sel.evaluate_cell(cell, **arguments, score_only=True)
    check('parameter_bytes_first_and_score_only_retry_reuses_false_certified_finite_output_without_inverse',
        len(counts) == 1 and counts[0] == (sel.PHASE, cell['method'], cell['case_id'])
        and events.count('inverse_fixture_callback') == 1 and len(scores) == 2
        and first['score_status'] == 'SCORING_FAILED' and second['score_status'] == third['score_status'] == 'SCORED'
        and not second['production_numerical_certified']
        and events.index('parameters_persisted') < events.index('canonical_byte_verified') < events.index('external_score_fixture'))
    fresh_store = FakeStore(root/'score_only_absent'); absent_events = []
    absent = ControllerCache(fresh_store, 'fresh', absent_events)
    absent_row = sel.evaluate_cell(cell, cache=absent, raw='fixture', action=lambda: absent_events.append('inverse'),
        score_action=lambda out: {}, count_call=lambda *args: absent_events.append('count'), store=fresh_store,
        digest='absent', score_only=True)
    check('score_only_on_unexecuted_cell_does_not_begin_count_or_invert',
        absent_row['status'] == 'NOT_EXECUTED_SCORE_ONLY' and not absent_events and not absent.started)
    stopped_store = FakeStore(root/'interrupted'); stopped_events = []
    interrupted = ControllerCache(stopped_store, 'interrupted', stopped_events); interrupted.begin('raw')
    with mock.patch.object(dev, 'call_snapshot', return_value=([call], {})):
        no_retry = sel.evaluate_cell(cell, cache=interrupted, raw='raw', action=lambda: stopped_events.append('inverse'),
            score_action=lambda out: {}, count_call=lambda *args: stopped_events.append('count'), store=stopped_store, digest='interrupted')
    check('interrupted_cache_and_count_before_local_receipt_remain_billed_with_zero_reinverse',
        no_retry['controller_stop'] and no_retry['new_inverse_calls_selection_total'] == 1
        and not no_retry['new_inverse_calls_this_run'] and 'inverse' not in stopped_events and 'count' not in stopped_events)
    changed_store = FakeStore(root/'source_before_begin'); changed_events = []
    changed_cache = ControllerCache(changed_store, 'source', changed_events)
    def changed_source(): raise sel.SelectionIntegrityError('injected source drift')
    changed_row = sel.evaluate_cell(cell, cache=changed_cache, raw='raw', action=lambda: changed_events.append('inverse'),
        score_action=lambda out: {}, count_call=lambda *args: changed_events.append('count'), store=changed_store,
        digest='source', verify_sources=changed_source)
    check('source_drift_stops_before_owner_ticket_or_count', changed_row['controller_stop'] and not changed_events)
    failed_store = FakeStore(root/'failed_action'); failed_events = []; failed_counts = []
    failed_cache = ControllerCache(failed_store, 'failed', failed_events)
    def fail_action(): failed_events.append('inverse'); raise ArithmeticError('fixture optimizer exception')
    failed_args = dict(cache=failed_cache, raw='raw', action=fail_action, score_action=lambda out: {},
        count_call=lambda *args: failed_counts.append(args), store=failed_store, digest='failed')
    first_fail = sel.evaluate_cell(cell, **failed_args); second_fail = sel.evaluate_cell(cell, **failed_args)
    check('ordinary_inverse_failure_is_retained_without_controller_pruning_or_duplicate_attempt',
        first_fail['status'] == second_fail['status'] == 'INFERENCE_FAILED_NO_RERUN'
        and not first_fail['controller_stop'] and len(failed_counts) == failed_events.count('inverse') == 1
        and not second_fail['new_inverse_calls_this_run'])
    memory_store = FakeStore(root/'memory_only'); memory_events = []
    memory_cache = ControllerCache(memory_store, 'memory', memory_events, persist_failure={
        'status': 'FAIL_CLOSED_ORIGINAL_PERSISTENCE_UNAVAILABLE', 'score_allowed': False,
        'original_fallback_available': False, 'durable_failure_receipt_written': False})
    with contextlib.redirect_stdout(io.StringIO()):
        lost = sel.evaluate_cell(cell, cache=memory_cache, raw='raw', action=lambda: {'numerical_valid': False, 'info': {}},
            score_action=lambda out: memory_events.append('score'), count_call=lambda *args: None,
            store=memory_store, digest='memory')
    with mock.patch.object(dev, 'call_snapshot', return_value=([call], {})):
        recovered_rows = sel.collect_rows(reg, FakeStore(root/'ledger_only'), 'reg')
        overlaid = sel.merge_transient_rows(reg, recovered_rows, [{**lost, 'case_state_durable': False}])
    memory_summary = sel.summary(reg, overlaid)
    check('unwritten_endpoint_failure_is_honest_visible_and_full_denominator_keeps_the_charged_attempt',
        lost['controller_stop'] and lost['durable_failure_receipt_written'] is False
        and lost['original_finite_endpoint_not_durably_preserved'] and not lost['original_fallback_available']
        and 'score' not in memory_events and len(memory_summary['cells']) == 1080
        and memory_summary['new_inverse_calls_selection_total'] == 1
        and any(row['status'] == 'COUNTED_ATTEMPT_WITHOUT_DURABLE_CASE_STATE' for row in recovered_rows))

    state_store = FakeStore(root/'case_history'); state_row = {**full[1], 'new_inverse_calls_selection_total': 0}
    state_cell = plan[1]
    one = sel.persist_row(state_store, state_cell, state_row, 'state-reg')
    duplicate = sel.persist_row(state_store, state_cell, state_row, 'state-reg')
    state_row['metrics'] = score_metrics(4.)
    two = sel.persist_row(state_store, state_cell, state_row, 'state-reg')
    history = sel.state_files(state_store.report_root/'case_states'/sel.row_key(state_cell))
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})):
        state_rows = sel.collect_rows(reg, state_store, 'state-reg')
    check('immutable_case_history_deduplicates_identical_resume_and_selects_latest_valid_score_state',
        one == duplicate and one != two and len(history) == 2
        and next(row for row in state_rows if dev.semantic_key(row) == dev.semantic_key(state_cell))['metrics'] == score_metrics(4.))
    Path(two['path']).with_name(Path(two['path']).name+'.sha256.json').write_text('{}', encoding='utf8')
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})):
        corrupt_rows = sel.collect_rows(reg, state_store, 'state-reg')
    check('corrupt_or_incompletely_sealed_latest_state_is_explicit_incomplete_never_falls_back_to_nicer_score',
        next(row for row in corrupt_rows if dev.semantic_key(row) == dev.semantic_key(state_cell))['status'] == 'CASE_STATE_RECEIPT_INTERRUPTED_OR_CORRUPT'
        and sel.summary(reg, corrupt_rows)['incomplete_or_integrity_stopped_cells'] > 0)
    incomplete_store = FakeStore(root/'summary_final_gate')
    with mock.patch.object(sel, 'collect_rows', return_value=stopped):
        incomplete = sel.write_summary(incomplete_store, reg, 'fixture')
    check('one_integrity_stop_prevents_final_receipt_even_if_every_cell_has_an_attempted_row',
        incomplete['status'] == 'FIRST_STAGE_PARTIAL' and not (incomplete_store.report_root/sel.RESULT_NAME).exists()
        and (incomplete_store.report_root/'SUMMARY_CURRENT.json').exists())

    raw_dn = np.array([[100, 101], [102, 103]], dtype=np.uint16)
    raw_path = root/'raw_fixture.npz'; s.save(raw_path, raw_dn=raw_dn)
    raw = torch.as_tensor(raw_dn.astype(np.float64))[None]
    raw_ref = {'path': str(raw_path), 'sha256': s.sha(raw_path), 'array_sha256': compact.array_sha(raw_dn),
               'dtype': 'uint16', 'shape': [2, 2], 'fixture_only': True}
    all_physics = []
    def tiny_cache(name):
        p = FixturePhysics(); all_physics.append(p)
        cache = dev.ECompactCache(root/'tiny'/name/'reports', root/'tiny'/name/'data', p=p,
            identity={'registration_sha256': 'CPU-qualified-codec-fixture', 'method': 'CPU-forward-fixture', 'case_id': name},
            input_reference=raw_ref, calibration={'fingerprint': 'fixture-C1', 'physical_objective_version': 'fixture-C1'},
            config={'grid': 2}, source_sha256={sel.SOURCES[0]: s.sha(s.ROOT/sel.SOURCES[0])},
            budget=sel.SelectionBudget((root/'tiny'/name,)))
        return cache, p, original_fixture(p, raw)
    cache0, p0, out0 = tiny_cache('old_read_only')
    ticket = cache0.begin(raw); ready = cache0.persist(ticket, out0, raw)
    locator = {'kind': 'compact', 'report': str(cache0.report), 'data': str(cache0.data),
               'case_id': 'old_read_only', 'method': 'CPU-forward-fixture'}
    old_hashes = {str(path): s.sha(path) for path in cache0.report.rglob('*') if path.is_file()}
    old_hashes.update({str(path): s.sha(path) for path in cache0.data.rglob('*') if path.is_file()})
    rebuilt, proof = sel.read_existing_compact(locator, p0, raw)
    values = s.arrays(cache0.data/'PARAMETERS.npz')
    check('reused_default_E_codec_retains_float64_parameter_and_all_canonical_bytes_with_forward_only',
        ready['status'] == 'LOSSLESS_READY' and proof['all_byte_identical']
        and all(compact.descriptor(rebuilt[name]) == compact.descriptor(out0[name]) for name in compact.CANONICAL_KEYS+('continuum_latent',))
        and all(value.dtype == np.float64 for value in values.values())
        and compact.array_sha(values['z_grid']) == compact.array_sha(out0['z_grid'])
        and bool(np.signbit(values['z_grid'][0, 0, 0, 1])) and not list(cache0.data.glob('FALLBACK_RESERVED_*')))
    p0.op.offset = 1.
    rejected_forward = rejects(lambda: sel.read_existing_compact(locator, p0, raw))
    p0.op.offset = 0.
    actual_read = compact._read_sealed
    def bad_parameter_chain(path):
        value = actual_read(path)
        if Path(path).name == 'RESULT.json': value['parameter_receipt']['sha256'] = '0'*64
        return value
    with mock.patch.object(compact, '_read_sealed', side_effect=bad_parameter_chain):
        bad_chain = rejects(lambda: sel.read_existing_compact(locator, p0, raw))
    check('old_read_only_byte_or_receipt_mismatch_rejects_scoring_without_writing_old_failure',
        rejected_forward and bad_chain and rejects(lambda: sel.read_existing_compact({**locator, 'method': 'wrong'}, p0, raw))
        and old_hashes == {str(path): s.sha(path) for base in (cache0.report, cache0.data) for path in base.rglob('*') if path.is_file()}
        and not (cache0.report/'RECONSTRUCTION_FAILURE.json').exists())
    cache1, p1, out1 = tiny_cache('new_namespace')
    exact = cache1.persist(cache1.begin(raw), rebuilt, raw)
    new_saved = cache1.load_for_score(raw)
    qualification = panel.qualification_matrix([{'case_id': str(i), 'method': 'saved'} for i in range(4)],
        lambda origin: {'passed': origin['case_id'] != '1', 'complete_inverse_calls': 0})
    check('new_E_namespace_forward_persistence_reuses_codec_and_four_qual_denominator_without_inverse',
        exact['status'] == 'LOSSLESS_READY'
        and all(compact.descriptor(new_saved[name]) == compact.descriptor(out0[name]) for name in compact.CANONICAL_KEYS)
        and qualification['audited_existing_endpoints'] == 4 and qualification['passed_existing_endpoints'] == 3
        and not qualification['all_passed'] and qualification['complete_inverse_calls'] == 0)
    failed_cache, pf, badout = tiny_cache('one_bit_fallback')
    badout['predicted_raw_e'] = badout['predicted_raw_e'].clone()
    value = badout['predicted_raw_e'][0, 0, 0]
    badout['predicted_raw_e'][0, 0, 0] = torch.nextafter(value, torch.full_like(value, float('inf')))
    fallback = failed_cache.persist(failed_cache.begin(raw), badout, raw)
    check('byte_mismatch_safely_reserves_dense_original_fallback_and_permanently_refuses_score_and_reinverse',
        fallback['original_fallback_available'] and not fallback['score_allowed']
        and all(compact.descriptor(failed_cache.load_fallback()[key]) == compact.descriptor(badout[key]) for key in compact.CANONICAL_KEYS)
        and not failed_cache.begin(raw)['run_new_inverse'] and rejects(lambda: failed_cache.load_for_score(raw)))
    crash_cache, pc, crashout = tiny_cache('params_before_exit'); crash_ticket = crash_cache.begin(raw)
    with mock.patch.object(crash_cache, '_verify_forward', side_effect=KeyboardInterrupt('fixture exit')):
        try: crash_cache.persist(crash_ticket, crashout, raw)
        except KeyboardInterrupt: pass
    check('persisted_parameters_recover_by_forward_only_and_never_issue_a_second_inverse_ticket',
        crash_cache.resume(raw)['status'] == 'LOSSLESS_READY' and not crash_cache.begin(raw)['run_new_inverse'])
    floor_cache, pb, floorout = tiny_cache('floor_lost'); floor_ticket = floor_cache.begin(raw)
    with mock.patch.object(floor_cache.budget, 'check', side_effect=compact.CacheIntegrityError('fixture floor loss')):
        floor_failure = floor_cache.persist(floor_ticket, floorout, raw)
    check('floor_loss_after_callback_does_not_claim_unwritten_original_fallback_or_durable_failure',
        not floor_failure['original_fallback_available'] and not floor_failure['durable_failure_receipt_written']
        and not floor_failure['score_allowed'] and not floor_cache.begin(raw)['run_new_inverse'])
    own_store = FakeStore(root/'evidence_binding')
    # Copy only receipt references into a new case state; do not copy arrays.
    owned_report = own_store.report_root/'reference.json'; own_store.write(owned_report, {'evidence': True})
    evidence_row = {**full[2], 'evidence_files': [panel.file_ref(owned_report)]}
    with mock.patch.object(sel, 'case_evidence_files', return_value=evidence_row['evidence_files']):
        sel.persist_row(own_store, plan[2], evidence_row, 'evidence-reg')
    owned_report.write_text('{"evidence":false}', encoding='utf8')
    with mock.patch.object(dev, 'call_snapshot', return_value=([], {})):
        corrupt_evidence = sel.collect_rows(reg, own_store, 'evidence-reg')
    check('summary_revalidates_cached_evidence_file_hashes_before_trusting_finite_state',
        next(row for row in corrupt_evidence if dev.semantic_key(row) == dev.semantic_key(plan[2]))['status'] == 'CASE_STATE_RECEIPT_INTERRUPTED_OR_CORRUPT')

    # Exercise the production shard loop with CPU fakes at every numerical,
    # truth and registration boundary. The only "cuda" token is the CLI guard;
    # its physics factory is replaced by the existing CPU2x2 fixture.
    from .. import smooth_calibration, numerical_strategy, development, metrics
    from research_v6 import inputs
    shard_store = FakeStore(root/'mocked_shard_loop'); shard_events = []; shard_caches = {}; shard_rows = []
    mock_reg = {**reg, 'config_fingerprint': 'mock-only',
        'v3_initializer_provenance': {blank['case_id']: {'state_cache': panel.file_ref(raw_path)}}}
    def mocked_cache(store, p, target, registration, digest, **kwargs):
        key = sel.row_key(target)
        if key not in shard_caches: shard_caches[key] = ControllerCache(store, key, shard_events)
        return shard_caches[key]
    attempted, charged, scored = [], [], []
    def mocked_inverse(raw, p, *, initializer, v3_state, physical_options, numerical_options):
        attempted.append((initializer, physical_options, numerical_options))
        if len(attempted) == 1: raise ArithmeticError('one ordinary numerical failure')
        return {'numerical_valid': False, 'info': {}}
    def mocked_persist(store, target, row, digest):
        if row['status'] == 'INFERENCE_FAILED_NO_RERUN':
            # The real cache has a durable STARTED record; the imported tiny
            # FakeCache leaves its result field NEW after a callback exception.
            shard_caches[sel.row_key(target)].result['status'] = 'STARTED'
        shard_rows.append(sel.row_projection(target, row))
        return {'path': 'CPU-fixture-no-formal-case-state', 'sha256': sel.row_key(target)}
    def mocked_summary(store, registration, digest):
        merged = {dev.semantic_key(row): row for row in initial_rows+shard_rows}
        return {'summary': sel.summary(registration, list(merged.values()))}
    with (mock.patch.object(sel, 'verify_registration', return_value=(mock_reg, 'mock-registration')),
          mock.patch.object(dev, 'phase_locks', return_value=contextlib.nullcontext()),
          mock.patch.object(smooth_calibration, 'physics', return_value=p0),
          mock.patch.object(smooth_calibration, 'contract', return_value=reg['calibration_contract']),
          mock.patch.object(sel, 'verify_namespace_qualification', return_value={}),
          mock.patch.object(inputs, 'load_raw', return_value=raw),
          mock.patch.object(inputs, 'load_diagnostic_truth', return_value={'fixture_only': True}),
          mock.patch.object(panel.q, 'verify_initializer_raw', return_value={}),
          mock.patch.object(development, 'cached_v3', return_value=(torch.zeros(1), None)),
          mock.patch.object(numerical_strategy, 'repaired_infer', side_effect=mocked_inverse),
          mock.patch.object(panel.c2, 'wrap_output', side_effect=lambda out, target, metadata: out),
          mock.patch.object(sel, 'read_origin', return_value=({}, {'all_byte_identical': True})),
          mock.patch.object(sel, 'cache_for', side_effect=mocked_cache),
          mock.patch.object(sel, 'counted_once', side_effect=lambda target, registration: charged.append(sel.row_key(target))),
          mock.patch.object(metrics, 'score', side_effect=lambda *args, **kwargs: scored.append(True) or {'metrics': score_metrics()}),
          mock.patch.object(sel, 'persist_row', side_effect=mocked_persist),
          mock.patch.object(sel, 'write_summary', side_effect=mocked_summary),
          mock.patch.object(dev, 'call_snapshot', return_value=([], {})),
          contextlib.redirect_stdout(io.StringIO())):
        first_shard = sel.run_shard('cuda', raw_ids=[blank['case_id']], store=shard_store)
        retried_shard = sel.run_shard('cuda', raw_ids=[blank['case_id']], score_only=True, store=shard_store)
    check('actual_shard_loop_CPU_fakes_execute_all18_after_one_failure_and_resume_zero_inverse',
        first_shard['status'] == retried_shard['status'] == 'FIRST_STAGE_SHARD_COMPLETE'
        and len(attempted) == len(charged) == 16 and len(scored) == 15 and len(shard_rows) == 36
        and shard_rows[0]['status'] == shard_rows[18]['status'] == 'INFERENCE_FAILED_NO_RERUN'
        and first_shard['new_inverse_calls_this_shard'] == 16 and retried_shard['new_inverse_calls_this_shard'] == 0
        and first_shard['global_summary_counts']['planned_cells'] == 1080
        and first_shard['global_summary_counts']['not_executed_cells'] == 1036
        and all(options['grid'] == 256 and numerical == {'cost_limit_s': 180.} for _, options, numerical in attempted))

    lock_store = FakeStore(root/'owner_locks'); lock1 = lock_store.report_root/'global.json'; lock2 = lock_store.report_root/'local.json'
    with dev.phase_locks(lock_store, 'one', 'CPU', paths=(lock1, lock2)):
        def occupied():
            with dev.phase_locks(lock_store, 'two', 'CPU', paths=(lock1,)): pass
        exclusive = rejects(occupied)
    released = not lock1.exists() and not lock2.exists()
    with dev.phase_locks(lock_store, 'one', 'CPU', paths=(lock1,)):
        lock1.write_text('{"pid":-1,"owner_token":"foreign"}', encoding='utf8')
    check('single_GPU_owner_global_and_local_locks_are_exclusive_and_leave_foreign_replacement',
        exclusive and released and s.read(lock1)['owner_token'] == 'foreign')
    sources = {name: s.sha(s.ROOT/name) for name in (*sel.SOURCES, 'research_closeout/runtime_entry.py',
        'research_closeout/c1_development.py', 'research_closeout/compact_joint_cache.py',
        'research_closeout/c1_small_panel.py', 'research_closeout/tests/compact_joint_cache.py')}
    binding = root/'CPU_BINDINGS.json'
    s.write(binding, {'all_passed': True, 'selection_E_codec_CPU_contracts_executed': True, 'source_sha256': sources})
    _, audited_sources = sel.contract_sources([binding])
    stale = root/'STALE_BINDINGS.json'; s.write(stale, {'all_passed': True, 'selection_E_codec_CPU_contracts_executed': True,
        'source_sha256': {**sources, sel.SOURCES[0]: '0'*64}})
    arithmetic = root/'ARITHMETIC_ONLY.json'; s.write(arithmetic, {'all_passed': True, 'source_sha256': sources})
    check('registration_CPU_contract_requires_exact_new_sources_entry_and_actual_E_codec_persistence',
        audited_sources == sources and rejects(lambda: sel.contract_sources([stale])) and rejects(lambda: sel.contract_sources([arithmetic])))
    for function in (sel.qualify, sel.verify_namespace_qualification, sel.read_existing_compact, sel.read_origin):
        tree = ast.parse(inspect.getsource(function))
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        check(function.__name__+'_has_no_optimizer_initializer_truth_or_formal_counter',
            not calls.intersection({'repaired_infer', 'minimize', 'lsmr', 'count_call', 'cached_v3', 'initialize', 'load_diagnostic_truth'}))
    from ..smooth_calibration import physics
    p_audit = physics('cpu')
    from research_v6 import inputs
    with (mock.patch.object(inputs, 'load_diagnostic_truth', side_effect=AssertionError('No truth in initializer audit')),
          mock.patch.object(registry, 'count_call', side_effect=AssertionError('No inverse in initializer audit'))):
        initializer_provenance = sel.audit_initializers(prepared['initializer_references180'], p_audit)
    check('all180_existing_v3_raw_and_state_cache_hashes_are_audited_without_any_new_v3_inverse_or_truth',
        len(initializer_provenance) == 180
        and all(value['state_cache']['sha256'] and value['source_input_array_hash_verified']
                and value['original_dtype_roundtrip_verified'] for value in initializer_provenance.values()))
    after = {str(path): s.sha(path) for path in protected if path.exists()}
    check('forty_frozen_dependencies_all_old_terminal_cache_evidence_and_true_ledger_remain_byte_identical',
        len(frozen) == 40 and before == after, protected_files=len(before), frozen_sources=len(frozen))
    check('CPU_fixtures_never_run_native_GPU_formal_inverse_or_registration_or_final_payloads',
        all(p.inverse_calls == 0 and p.M.device.type == 'cpu' for p in all_physics)
        and rejects(lambda: sel.run_shard('cpu', raw_ids=[blank['case_id']]))
        and rejects(lambda: sel.qualify('cpu')) and not sel.CONFIG['historical_detector_threshold_transferred']
        and sel.CONFIG['detector'] is None and not sel.CONFIG['confirmation24_started'])
    result = {'schema': sel.VERSION+'-CPU-contracts', 'time': s.now(), 'all_passed': all(row['passed'] for row in checks),
        'checks': checks, 'source_sha256': sources, 'execution_entry': entry,
        'selection_E_codec_CPU_contracts_executed': True, 'actual_initializer180_hash_audit_executed': True,
        'GPU_used': False, 'formal_complete_inversions': 0, 'truth_loaded': False,
        'formal_registration_or_namespace_qualification_or_shard_executed': False,
        'mocked_production_shard_loop_executed_on_CPU': True,
        'native_GPU_E_namespace_qualification_performed': False, 'D_writes': 0,
        'old_data_converted_deleted_or_migrated': False, 'old_protected_file_sha256_before': before,
        'old_protected_file_sha256_after': after, 'fixture_root': str(root),
        'fixture_storage_ledger': compact.tree_ledger((root,)), 'elapsed_s': time.monotonic()-started,
        'actual_plan': {'cells': len(plan), 'reused': len(prepared['reuse_catalog']), 'maximum_new': 1052,
                        'first_root_shard_raw': blank['case_id'], 'first_shard_reuse': 2, 'first_shard_new_maximum': 16},
        'storage_contract': sel.SelectionStore().contract(), 'format_contract': dev.execution_format_contract(sel.SelectionStore()),
        'native_frame_conservative_plan': native_plan}
    destination = Path(output) if output else sel.CPU_REPORT_DIR/('CONTRACTS_'+token+'.json')
    s.write(destination, result)
    print(json.dumps({'receipt': str(destination), 'sha256': s.sha(destination), 'all_passed': result['all_passed'],
        'checks': len(checks), 'failed': [row for row in checks if not row['passed']], 'elapsed_s': result['elapsed_s'],
        'source_sha256': sources, 'formal_complete_inversions': 0, 'GPU_used': False,
        'fixture_bytes': result['fixture_storage_ledger']['bytes']}, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    raise SystemExit(0 if run(parser.parse_args().output)['all_passed'] else 2)
