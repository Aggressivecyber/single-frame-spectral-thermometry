"""Full registered stage-1 execution; no training and no sealed data access.

The root calls these functions after the source protocol is frozen.  A bounded
CPU fixture is never accepted by these formal entry points.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

from .common import (CODE, DATA, REPORTS, ROOT, arrays, append_jsonl, canonical_hash,
                     ed_path, jsonl, now, read_json, relative, save_npz, sha,
                     source_hashes, synchronize, write_json)
from .reports import (LEVELS, information_summary, oracle_summary, write_decision_1,
                      write_information_reports, write_oracle_reports)


def _plain(value):
    """Strict JSON has null for a nonfinite diagnostic, never NaN/Infinity."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _guard(physics, protocol):
    assert tuple(physics.cfg['shape']) == (256, 256), 'Formal stage1 uses the frozen 256x256 physics'
    assert canonical_hash(physics.cfg) == protocol['physics_config_sha256']
    assert source_hashes() == protocol['source_sha256'], 'Source changed after freeze'


def _plan(protocol):
    path = CODE / 'diagnostic_plan.json'
    assert sha(path) == protocol['diagnostic_plan_sha256']
    plan = read_json(path)
    assert len(plan) == 180 and len({case['case_id'] for case in plan}) == 180
    assert len({case['mother_id'] for case in plan}) == 12
    assert not any('confirmation' in case['mother_id'] or 'final' in case['mother_id'] for case in plan)
    return plan


def _manifest(plan, physics, *, generate=False):
    folder = DATA / 'oracle_data'
    path = folder / 'data_manifest.json'
    if not path.exists():
        if not generate:
            raise FileNotFoundError('The complete oracle data must exist before information diagnostics')
        from .data import generate_diagnostics
        manifest = generate_diagnostics(physics, plan, folder)
    else:
        manifest = read_json(path)
    assert manifest['fixture_only'] is False
    assert manifest['planned_rows'] == 180 and len(manifest['rows']) == 180
    assert manifest['plan_sha256'] == canonical_hash(plan)
    assert manifest['physics_config_sha256'] == canonical_hash(physics.cfg)
    assert manifest['generator_source_sha256'] == sha(CODE / 'data.py')
    assert manifest['matched_continuum_background_shape_verified'] is True
    assert manifest['final_test_opened'] is False and manifest['confirmation_test_opened'] is False
    assert {row['case_id'] for row in manifest['rows']} == {case['case_id'] for case in plan}
    return folder, manifest


def _case_metadata(row, protocol, manifest_path):
    return {key: copy.deepcopy(row.get(key)) for key in
            ('case_id', 'mother_id', 'condition', 'group', 'coefficient_e', 'input_kind',
             'noise_seed', 'registered_noise_seed', 'temperature_geometry', 'template_geometry',
             'source_manifest_sha256', 'label_sha256', 'raw_file_sha256', 'raw_generation_status')}


def _provenance(row, desc, protocol, manifest_path):
    return {**_case_metadata(row, protocol, manifest_path),
            'source_descriptor_sha256': canonical_hash(desc),
            'source_sha256': protocol['source_sha256'],
            'protocol_sha256': sha(CODE / 'protocol.json'),
            'data_manifest_sha256': sha(manifest_path),
            'raw_path': row.get('raw_path'), 'label_path': row.get('label_path'),
            'raw_array_sha256': row.get('raw_hash'),
            'expectation_diagnostics': row.get('expectation_diagnostics'),
            'ADC_diagnostics': row.get('ADC_diagnostics'),
            'fixture_only': False, 'truth_role': 'diagnostic oracle/scoring only; excluded from O3 inference'}


def _load_case(folder, row):
    from .data import _array_hash
    if not row['raw_available']:
        raise ValueError(row['raw_generation_status'])
    raw_path = ed_path(folder / row['raw_path'])
    label_path = ed_path(folder / row['label_path'])
    assert raw_path.is_relative_to(folder.resolve()) and label_path.is_relative_to(folder.resolve())
    assert sha(raw_path) == row['raw_file_sha256'], 'Raw file changed'
    assert sha(label_path) == row['label_sha256'], 'Diagnostic label file changed'
    raw = arrays(raw_path, raw_only=True)['raw_dn']
    assert _array_hash(raw) == row['raw_hash'], 'Raw array hash differs from manifest'
    truth = arrays(label_path)
    return raw, truth


def _failure_rows(base, status, reason, error=None):
    return [{**base, 'level': level, 'oracle_level': level,
             'diagnostic_oracle': level != 'O3', 'deployable_path': level == 'O3',
             'status': status, 'reason': reason, 'error': error,
             'metrics': None, 'CG': None, 'solver_info': None,
             'raw_dn_sha256': base.get('raw_array_sha256'), 'raw_e_sha256': None,
             'latency_s': None, 'output_selected_by_truth': False}
            for level in LEVELS]


def _prediction_arrays(output, continuum=None):
    import numpy as np
    import torch
    selected = {}
    merged = {**(continuum or {}), **output}
    for key in ('template_latent_e', 'template_postblur_e', 'beta', 'expected_e',
                'temperature_color_K', 'valid'):
        value = merged.get(key)
        if value is not None:
            selected[key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return selected


def run_oracle(protocol, physics, budget):
    """Generate/reopen 180 inputs and record all 720 levels plus 84 baselines."""
    _guard(physics, protocol)
    plan = _plan(protocol)
    journal = REPORTS / 'ORACLE_LADDER.jsonl'
    baseline_journal = REPORTS / 'ORACLE_BASELINES_84.jsonl'
    if journal.exists() or baseline_journal.exists() or (REPORTS / 'ORACLE_SUMMARY.json').exists():
        raise FileExistsError('Oracle evidence already exists; never rerun or overwrite an evaluated case')
    budget.check()
    folder, manifest = _manifest(plan, physics, generate=True)
    manifest_path = folder / 'data_manifest.json'
    row_by_id = {row['case_id']: row for row in manifest['rows']}
    from .oracle import run_oracle_case
    from .common import raw_electrons
    options = {'continuum': protocol['continuum_options'], 'linear': protocol['oracle_linear_options']}
    all_rows = []
    exhausted = False
    for index, desc in enumerate(plan):
        row = row_by_id[desc['case_id']]
        base = _provenance(row, desc, protocol, manifest_path)
        if not exhausted:
            try:
                budget.check()
            except TimeoutError:
                exhausted = True
        if exhausted:
            rows = _failure_rows(base, 'NOT_EXECUTED_BUDGET', 'registered stage budget exhausted before case')
        elif not row['raw_available']:
            rows = _failure_rows(base, 'UNAVAILABLE_INPUT', row['raw_generation_status'])
        else:
            try:
                raw, truth = _load_case(folder, row)
                # Saturated/out-of-range/nonfinite observations are refused;
                # the signed-stress mother is never changed for legality.
                raw_electrons(physics, raw)
                synchronize(physics.M.device)
                started = time.perf_counter()
                result = run_oracle_case(physics, raw, truth, options=options)
                synchronize(physics.M.device)
                case_wall_s = time.perf_counter() - started
                assert len(result['rows']) == 4 and {r['level'] for r in result['rows']} == set(LEVELS)
                rows = []
                for value in result['rows']:
                    level = value['level']
                    out = result['outputs'][level]
                    entry = {**base, **value, 'case_wall_s_shared_four_levels': case_wall_s,
                             'same_raw_and_weights_for_all_levels': result['same_raw_and_weights_for_all_levels'],
                             'reported_latency_scope': 'per-level includes shared continuum estimate; case_wall_s is actual four-level wall'}
                    assert entry['raw_dn_sha256'] == row['raw_hash']
                    if out is not None:
                        saved = DATA / 'oracle_predictions' / (desc['case_id'] + '_' + level + '.npz')
                        estimated = result['o3'] if level == 'O3' else result['continuum_result'] if level == 'O1' else None
                        entry['prediction_path'] = relative(saved)
                        entry['prediction_sha256'] = save_npz(saved, **_prediction_arrays(out, estimated))
                    rows.append(_plain(entry))
            except Exception as error:
                rows = _failure_rows(base, 'FAILED', 'oracle case could not be evaluated',
                                     {'type': type(error).__name__, 'message': str(error)})
        for entry in rows:
            append_jsonl(journal, entry)
            all_rows.append(entry)
        print(f'oracle case {index+1}/180 {desc["case_id"]}: ' + ','.join(r['status'] for r in rows), flush=True)

    baseline_plan = [case for case in plan if case['input_kind'] == 'noise'
                     or case['input_kind'] == 'continuous' and abs(case['coefficient_e']) == 180.]
    assert len(baseline_plan) == protocol['oracle_baseline_subset']['cases'] == 84
    from .baselines import BASELINE_NAMES, BaselineSet
    from .metrics import score_prediction
    baseline_error = None
    baselines = None
    baseline_started = time.perf_counter()
    if not exhausted:
        try:
            budget.check()
            baselines = BaselineSet(device=physics.M.device, cfg=physics.cfg)
            assert tuple(protocol['baselines']) == BASELINE_NAMES
        except Exception as error:
            baseline_error = {'type': type(error).__name__, 'message': str(error)}
    baseline_rows = []
    baseline_budget = protocol['oracle_baseline_subset']['budget_s']
    for index, desc in enumerate(baseline_plan):
        row = row_by_id[desc['case_id']]
        base = _provenance(row, desc, protocol, manifest_path)
        not_executed = exhausted or time.perf_counter() - baseline_started > baseline_budget
        if not not_executed:
            try:
                budget.check()
            except TimeoutError:
                not_executed = exhausted = True
        outputs = None
        error = baseline_error
        raw, truth = None, None
        if not not_executed and baselines is not None and row['raw_available']:
            try:
                raw, truth = _load_case(folder, row)
                outputs = baselines.evaluate(raw)
            except Exception as caught:
                error = {'type': type(caught).__name__, 'message': str(caught)}
        for method in protocol['baselines']:
            entry = {**base, 'method': method, 'baseline_subset_scope': 'registered noise60 + continuous normal24',
                     'raw_dn_sha256': row.get('raw_hash'), 'oracle_truth_passed_to_method': False}
            if not_executed:
                entry.update(status='NOT_EXECUTED_BUDGET', metrics=None, reason='registered baseline/phase budget exhausted')
            elif not row['raw_available']:
                entry.update(status='UNAVAILABLE_INPUT', metrics=None, reason=row['raw_generation_status'])
            elif outputs is None:
                entry.update(status='FAILED', metrics=None, error=error)
            else:
                output = outputs[method]
                score = score_prediction(output, truth, baselines.physics)
                info = _plain(output['info'])
                entry.update(score, solver_info=info, latency_s=info.get('seconds'),
                             shared_inverse_seconds=info.get('shared_inverse_seconds'),
                             CG={'certificate_available': False,
                                 'reason': 'legacy adapter exposes inverse history; no fabricated v5 true residual'})
                if score['metrics'] is not None and score['status'] == 'SUCCESS':
                    saved = DATA / 'oracle_baseline_predictions' / (desc['case_id'] + '_' + method + '.npz')
                    entry['prediction_path'] = relative(saved)
                    entry['prediction_sha256'] = save_npz(saved, **_prediction_arrays(output))
            entry = _plain(entry)
            append_jsonl(baseline_journal, entry)
            baseline_rows.append(entry)
        print(f'baseline subset case {index+1}/84 {desc["case_id"]}: '
              + ','.join(r['status'] for r in baseline_rows[-4:]), flush=True)
    assert jsonl(journal) == all_rows and jsonl(baseline_journal) == baseline_rows
    summary = oracle_summary(all_rows, plan, protocol, baseline_rows=baseline_rows, manifest=manifest)
    summary.update({'phase_wall_s': budget.elapsed(), 'oracle_jsonl_sha256': sha(journal),
                    'baseline_jsonl_sha256': sha(baseline_journal),
                    'data_manifest_sha256': sha(manifest_path),
                    'baseline_provenance': _plain(baselines.provenance) if baselines is not None else None,
                    'execution_budget_exhausted': exhausted})
    write_oracle_reports(summary, all_rows, baseline_rows)
    write_decision_1(protocol)
    return summary


def run_identifiability(protocol, physics, budget):
    """Use exactly the 24 same noisy normal inputs and their diagnostic labels."""
    import torch
    _guard(physics, protocol)
    plan = _plan(protocol)
    oracle = read_json(REPORTS / 'ORACLE_SUMMARY.json')
    assert oracle['complete'] is True, 'Information follows the complete 180-input oracle accounting'
    folder, manifest = _manifest(plan, physics)
    manifest_path = folder / 'data_manifest.json'
    assert sha(manifest_path) == oracle['data_manifest_sha256']
    if (REPORTS / 'INFORMATION_SUMMARY.json').exists() or (REPORTS / 'FISHER_CASES.jsonl').exists():
        raise FileExistsError('Information evidence already exists; do not rerun or overwrite')
    from .identifiability import fisher_identifiability, optical_nullspace_tests
    from .common import raw_electrons
    wanted = [case for case in plan if case['input_kind'] == 'noise' and abs(case['coefficient_e']) == 180.]
    assert len(wanted) == 24
    row_by_id = {row['case_id']: row for row in manifest['rows']}
    journal = REPORTS / 'FISHER_CASES.jsonl'
    rows = []
    exhausted = False
    for index, desc in enumerate(wanted):
        source = row_by_id[desc['case_id']]
        entry = _provenance(source, desc, protocol, manifest_path)
        if not exhausted:
            try:
                budget.check()
            except TimeoutError:
                exhausted = True
        if exhausted:
            entry.update(status='NOT_EXECUTED_BUDGET', reason='registered phase budget exhausted', fisher=None)
        elif not source['raw_available']:
            entry.update(status='UNAVAILABLE_INPUT', reason=source['raw_generation_status'], fisher=None)
        else:
            try:
                raw, truth = _load_case(folder, source)
                raw_e = raw_electrons(physics, raw)
                state = torch.as_tensor(truth['latent_state'], device=physics.M.device, dtype=physics.M.dtype)[None]
                beta = torch.as_tensor(truth['beta'], device=physics.M.device, dtype=physics.M.dtype)[None]
                synchronize(physics.M.device)
                started = time.perf_counter()
                result = fisher_identifiability(physics, state, beta, raw_e, protocol['fisher_options'])
                synchronize(physics.M.device)
                assert len(result['frames']) == 1
                entry.update(status='RETURNED', fisher=result, latency_s=time.perf_counter() - started,
                             raw_dn_sha256=source['raw_hash'],
                             truth_role='diagnostic linearization; never deployment input')
            except Exception as error:
                entry.update(status='FAILED', fisher=None, error={'type': type(error).__name__, 'message': str(error)})
        entry = _plain(entry)
        append_jsonl(journal, entry)
        rows.append(entry)
        print(f'Fisher linearization {index+1}/24 {desc["case_id"]}: {entry["status"]}', flush=True)
    try:
        budget.check()
        optical = optical_nullspace_tests(physics, protocol['optical_options'])
    except Exception as error:
        optical = {'schema': 'v5-optical-nullspace-001', 'all_passed': False, 'status': 'FAILED',
                   'error': {'type': type(error).__name__, 'message': str(error)}}
    assert jsonl(journal) == rows
    summary = information_summary(rows, optical, plan)
    summary.update(phase_wall_s=budget.elapsed(), fisher_jsonl_sha256=sha(journal),
                   oracle_data_manifest_sha256=sha(manifest_path),
                   oracle_summary_sha256=sha(REPORTS / 'ORACLE_SUMMARY.json'))
    write_information_reports(summary, rows, _plain(optical))
    write_decision_1(protocol)
    return summary


def run_linear_solver(protocol, budget):
    """Reopen independent CPU operator controls; do not select a hyperparameter."""
    decision = read_json(REPORTS / 'DECISION_1.json')
    assert decision['physical_stage2_eligible'] is True, 'Physical stage 2 is blocked by stage 1'
    from .tests_adjoint import run_tests
    budget.check()
    result = run_tests()
    budget.check()
    result.update(schema='v5-linear-solver-validation-001',
                  scientific_256x256_acceptance=False,
                  selection_performed=False,
                  stage1_decision_sha256=sha(REPORTS / 'DECISION_1.json'))
    write_json(REPORTS / 'LINEAR_SOLVER_CHECKS.json', result)
    if not result['all_passed']:
        raise RuntimeError('Physical signed linear solver CPU controls failed')
    return result
