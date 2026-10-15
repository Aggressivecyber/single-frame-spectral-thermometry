"""Finite C1v3 development prescreen, reusing the registered inference engine.

Only mother0000 positive_normal_noise0, the original nine multiplier pairs,
and the two deployable initializers are supported. This controller does not
start selection/confirmation stages, tune a detector, or open final inputs.
Registration and a separate four-endpoint forward-only E-cache qualification
must precede root's explicit prescreen command. Cache tickets never authorize
repeating a known method/raw attempt under a new registration name.
"""
from . import storage as s
from . import compact_joint_cache as compact
from . import c1_small_panel as panel
from . import calibration_quick as c2
from . import quick_validation as q
import argparse
from contextlib import contextmanager
import copy
import hashlib
import itertools
import json
from pathlib import Path
import traceback
import uuid
import numpy as np

VERSION = 'C1v3-finite-development-prescreen-v001'
PHASE = 'C1v3_development_prescreen_v001'
REPORT_DIR = s.REPORTS/'repair_001/c1_development_001'
DATA_DIR = s.DATA/'repair_001/c1_development_001'
CPU_REPORT_DIR = REPORT_DIR/'cpu_contracts'
REGISTRATION_NAME = 'PRESCREEN_REGISTRATION.json'
QUALIFICATION_NAME = 'NATIVE_E_COMPACT_QUALIFICATION.json'
RESULT_NAME = 'PRESCREEN_RESULT.json'
CASE_ID = 'v5_development_selection_mother_0000_positive_normal_noise0'
INITIALIZERS = ('v3', 'smooth_demod')
CANDIDATES = tuple({'id': f'joint_c{c:g}_s{t:g}', 'continuum_multiplier': c, 'template_multiplier': t}
                   for c, t in itertools.product((.1, 1., 10.), repeat=2))
D_PROJECT = Path('D:/CodexWork/thermometry/closeout_001')
GLOBAL_ROOTS = (s.DATA, s.REPORTS, D_PROJECT/'outputs/repair_001', D_PROJECT/'reports/repair_001')
SOURCES = ('research_closeout/c1_development.py', 'research_closeout/tests/c1_development.py')
CONFIG = {'version': VERSION, 'stage': 'prescreen_only', 'raw_case_id': CASE_ID,
          'candidates': list(CANDIDATES), 'initializers': list(INITIALIZERS),
          'grid': 256, 'native_shape': [256, 256], 'numerical_options': {'cost_limit_s': 180.},
          'planned_cells': 18, 'maximum_new_inverse_calls': 16,
          'maximum_new_inverse_wall_s_excluding_overhead': 2880,
          'global_new_inverse_call_cap': 12000, 'instance_limit_bytes': 192*2**20,
          'global_new_study_limit_bytes': 4*2**30, 'E_free_floor_bytes': 2*2**30,
          'global_budget_roots': [str(p) for p in GLOBAL_ROOTS],
          'storage_backend': 'unchanged_default_E_compact_with_registered_accounting_guards',
          'qualification_existing_endpoints': 4, 'qualification_complete_inverse_calls': 0,
          'numerical_eligibility': 'both initializers pass the unchanged three-endpoint/two-output-plateau policy',
          'endpoint_gtol': 'actual returned stage tolerance; stage3 can certify at 1e-6; no universal 1e-7 claim',
          'detector': None, 'historical_threshold_transferred': False,
          'truth_used_for_numerical_eligibility': False, 'ranking_or_final_selection_performed': False,
          'selection12_or_confirmation24_implemented': False, 'final_payloads_opened': False}


class PrescreenIntegrityError(RuntimeError):
    pass


class StudyBudget(compact.Budget):
    """E writes debit E free space; every old/new E/D study file is billed."""
    def __init__(self, instance_roots, *, free_bytes=None):
        super().__init__(instance_roots, global_roots=GLOBAL_ROOTS,
                         instance_limit_bytes=CONFIG['instance_limit_bytes'], free_bytes=free_bytes)


class EStudyStore:
    def __init__(self):
        self.report_root, self.data_root = REPORT_DIR.resolve(), DATA_DIR.resolve()

    def write_path(self, path):
        path = s.write_path(path)
        if not any(path == root or root in path.parents for root in (self.report_root, self.data_root)):
            raise PrescreenIntegrityError('Prescreen write escapes its fixed E task roots')
        self.budget().check(0)
        return path

    def budget(self):
        return StudyBudget((self.report_root, self.data_root))

    def contract(self):
        return {'version': VERSION+'-E-accounting', 'storage_backend': 'default_E',
                'report_root': str(self.report_root), 'data_root': str(self.data_root),
                'global_budget_roots': [str(p) for p in GLOBAL_ROOTS],
                'global_limit_bytes': 4*2**30, 'E_free_floor_bytes': 2*2**30,
                'instance_limit_bytes': CONFIG['instance_limit_bytes'],
                'D_writes_supported': False, 'default_storage_whitelist_unchanged': True,
                'all_data_meta_fallback_pending_reserve_cache_log_bytes_counted': True,
                'existing_D_qualification_is_read_only_evidence': True}

    def write(self, path, value, *, replace=False):
        path = self.write_path(path)
        self.budget().check(len((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8'))+1024)
        return s.write(path, value, replace=replace)

    def append(self, path, value):
        path = self.write_path(path)
        self.budget().check(len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8'))+1)
        s.append(path, value)

    read = staticmethod(s.read)


def execution_format_contract(store):
    """Declare the four-root override while retaining the frozen array codec."""
    base = compact.format_contract()
    return {**base, 'execution_format_version': VERSION+'-default-E-four-root-accounting',
            'default_module_budget_roots_before_override': base['global_budget_roots'],
            'global_budget_roots': store.contract()['global_budget_roots'],
            'registered_budget_override_required': True, 'default_E_owner_release_unchanged': True,
            'cache_accounting_guard': 'research_closeout.c1_development.ECompactCache'}


class ECompactCache(compact.CompactJointCache):
    """Accounting guards only; inherit the original default-E owner release."""
    def __init__(self, *args, **kwargs):
        if kwargs.get('storage') is not None:
            raise PrescreenIntegrityError('This controller uses only the unchanged default E cache backend')
        super().__init__(*args, **kwargs)

    def _write(self, path, value):
        self._budget_check(len(compact._text(value).encode('utf8'))+1024)
        return super()._write(path, value)

    def _write_parameters(self, values, prepared):
        requested = 0 if (self.data/'PARAMETERS.npz').exists() else self.plan['parameter_zip_bound_bytes']
        self._budget_check(requested+len(compact._text(prepared).encode('utf8'))+1024)
        return super()._write_parameters(values, prepared)

    def _replace(self, source, target):
        self._budget_check(0)
        return super()._replace(source, target)

    def _fallback(self, arrays, owner):
        self._budget_check(0)
        return super()._fallback(arrays, owner)


@contextmanager
def phase_locks(store, digest, phase, *, paths=None):
    store.budget().check(8192)
    with panel.phase_locks(store, digest, phase, paths=paths) as owner:
        yield owner


def method_for(candidate, initializer):
    if candidate not in CANDIDATES or initializer not in INITIALIZERS:
        raise PrescreenIntegrityError('Only the originally declared nine pairs and two initializers are allowed')
    return (f"joint_profile_C1v3_c{candidate['continuum_multiplier']:g}_s{candidate['template_multiplier']:g}"
            f'_numerics_v2__{initializer}')


def planned_cells(reference):
    if (reference.get('case_id') != CASE_ID or reference.get('cohort') != 'development'
            or reference.get('mother_id') != 'v5_development_selection_mother_0000'
            or reference.get('condition') != 'positive_normal' or reference.get('input_kind') != 'noise'
            or reference.get('partition_role') != 'development_selection'
            or reference.get('fixture_only') or reference['raw_reference'].get('shape') != [256, 256]):
        raise PrescreenIntegrityError('Use the unchanged native prescreen raw from the original development plan')
    return [{**{key: reference[key] for key in ('case_id', 'mother_id', 'condition', 'input_kind', 'quick_geometry')},
             'raw_sha256': reference['raw_reference']['sha256'], 'candidate_id': candidate['id'],
             'family': 'joint_C1v3', 'method': method_for(candidate, initializer), 'initializer': initializer,
             'physical_options': {'grid': 256, **{key: candidate[key] for key in
                                  ('continuum_multiplier', 'template_multiplier')}}}
            for candidate in CANDIDATES for initializer in INITIALIZERS]


def semantic_key(cell):
    return cell['case_id'], cell['method'], cell['initializer'], cell['raw_sha256']


def reuse_catalog(planned, evidence, calls=()):
    """All attempted states block another inverse; never choose the nicer duplicate."""
    expected = {(cell['case_id'], cell['method']): cell for cell in planned}
    catalog = {}
    for item in evidence:
        row = item['row']; target = expected.get((row['case_id'], row['method']))
        if target is None:
            continue
        if semantic_key(row) != semantic_key(target):
            raise PrescreenIntegrityError('Prior same-method/raw identity is inconsistent')
        key = s.canonical(list(semantic_key(target)))
        if key in catalog:
            old = catalog[key]['row']
            for name in ('status', 'finite_output', 'production_numerical_certified', 'score_status', 'metrics'):
                if old.get(name) != row.get(name):
                    raise PrescreenIntegrityError('Conflicting prior states cannot be resolved by favorable selection: '+name)
            catalog[key]['aliases'].append(copy.deepcopy(item))
        else:
            catalog[key] = {**copy.deepcopy(item), 'aliases': []}
    for call in calls:
        target = expected.get((call.get('case_id'), call.get('method')))
        if target is None or call.get('reused'):
            continue
        key = s.canonical(list(semantic_key(target)))
        if key in catalog:
            catalog[key].setdefault('prior_call_records', []).append(copy.deepcopy(call))
        else:
            catalog[key] = {'row': {**target, 'status': 'PRIOR_COUNTED_ATTEMPT_WITHOUT_RECOVERABLE_RESULT',
                'finite_output': False, 'production_numerical_certified': False, 'score_status': 'NOT_SCOREABLE',
                'metrics': None}, 'origin_kind': 'counted_interrupted_attempt', 'prior_call_records': [copy.deepcopy(call)],
                'aliases': [], 'no_inverse_retry': True}
    return catalog


def bind_plan(planned, catalog):
    return [{**cell, 'reuse_key': s.canonical(list(semantic_key(cell))),
             'execution_role': 'REUSE_PRIOR_C1V3' if s.canonical(list(semantic_key(cell))) in catalog else 'NEW_INVERSE'}
            for cell in planned]


def numerical_evidence(info, certified):
    engine = info.get('numerical_engine', info)
    stages = engine.get('stages', [])
    comparisons = engine.get('output_comparisons', [])
    returned = engine.get('returned_stage')
    endpoint = next((stage for stage in stages if stage.get('stage') == returned), {})
    gtol = endpoint.get('configuration', {}).get('gtol')
    gradient = engine.get('returned_gradient', {}).get('scaled_projected_linf')
    last = comparisons[-2:]
    chain = (len(last) == 2 and last[0].get('to_stage') == last[1].get('from_stage')
             and last[1].get('to_stage') == returned)
    endpoints = {number for comparison in last for number in
                 (comparison.get('from_stage'), comparison.get('to_stage'))}
    qualified = sum(stage.get('stage') in endpoints and bool(stage.get('complete'))
                    and bool(stage.get('qualified_endpoint')) for stage in stages)
    passed = bool(certified and engine.get('numerical_certified') and chain and qualified == 3
        and engine.get('required_stable_comparisons') == 2
        and engine.get('observed_consecutive_stable_comparisons', 0) >= 2
        and all(comparison.get('passed') for comparison in last)
        and gtol is not None and gradient is not None and np.isfinite(gradient) and gradient <= gtol)
    return {'passed': passed, 'original_inference_certified_flag': bool(certified),
            'three_qualified_endpoints': qualified == 3, 'two_successive_output_plateaus': bool(chain and all(r.get('passed') for r in last)),
            'returned_stage': returned, 'actual_requested_endpoint_gtol': gtol,
            'returned_scaled_projected_linf': gradient, 'truth_metrics_used': False,
            'criterion': 'audit of the unchanged frozen numerical engine contract; no universal 1e-7 requirement'}


def reuse_row(cell, item, digest):
    origin = item['row']
    if semantic_key(cell) != semantic_key(origin):
        raise PrescreenIntegrityError('A prior output cannot be reassigned to another estimator or raw')
    return {**cell, 'execution_registration_sha256': digest, 'status': origin.get('status', 'PRIOR_INTERRUPTED'),
            'finite_output': bool(origin.get('finite_output')), 'production_numerical_certified': bool(origin.get('production_numerical_certified')),
            'numerical_evidence': numerical_evidence(origin.get('info', {}), origin.get('production_numerical_certified', False)),
            'metrics': copy.deepcopy(origin.get('metrics')), 'score_status': origin.get('score_status', 'NOT_SCOREABLE'),
            'inference_attempted': False, 'inference_reused': True, 'score_reused': origin.get('score_status') == 'SCORED',
            'new_inverse_calls_this_run': 0, 'new_inverse_calls_panel_total': 0,
            'origin_evidence': copy.deepcopy(item), 'historical_result_relabelled': False}


def summary(planned, rows):
    supplied = {}
    for row in rows:
        key = semantic_key(row)
        if key in supplied:
            raise PrescreenIntegrityError('Duplicate prescreen result')
        supplied[key] = row
    if set(supplied)-{semantic_key(cell) for cell in planned}:
        raise PrescreenIntegrityError('Unexpected prescreen cell')
    complete = [supplied.get(semantic_key(cell), {**cell, 'status': 'NOT_EXECUTED',
        'finite_output': False, 'production_numerical_certified': False, 'metrics': None,
        'score_status': 'NOT_EXECUTED', 'inference_attempted': False, 'new_inverse_calls_this_run': 0}) for cell in planned]
    candidates = []
    for candidate in CANDIDATES:
        group = [row for row in complete if row['candidate_id'] == candidate['id']]
        good = [row['initializer'] for row in group if row.get('numerical_evidence', {}).get('passed')
                and row.get('finite_output') and row.get('production_numerical_certified')]
        pending = any(row['status'].startswith('NOT_EXECUTED') for row in group)
        candidates.append({**candidate, 'planned_initializers': list(INITIALIZERS), 'qualified_initializers': good,
            'prescreen_numerically_qualified': len(good) == 2,
            'status': 'NUMERICALLY_QUALIFIED' if len(good) == 2 else 'INCOMPLETE' if pending else 'NUMERICALLY_DISQUALIFIED',
            'truth_scores_used_for_qualification': False})
    return {'planned_raws': 1, 'planned_cells': len(planned), 'returned_cell_rows': len(rows), 'cells': complete,
            'candidate_numerical_qualification': candidates, 'planned_candidates': 9,
            'numerically_qualified_candidates': sum(row['prescreen_numerically_qualified'] for row in candidates),
            'reused_prior_cells': sum(row.get('execution_role') == 'REUSE_PRIOR_C1V3' for row in complete),
            'new_inverse_calls_this_run': sum(row.get('new_inverse_calls_this_run', 0) for row in complete),
            'new_inverse_calls_prescreen_total': sum(row.get('new_inverse_calls_panel_total', 0) for row in complete),
            'finite_output_cells': sum(bool(row.get('finite_output')) for row in complete),
            'numerically_certified_cells': sum(bool(row.get('production_numerical_certified')) for row in complete),
            'not_executed_cells': sum(row['status'].startswith('NOT_EXECUTED') for row in complete),
            'scoring_failed_cells': sum(row.get('score_status') == 'SCORING_FAILED' for row in complete),
            'all_denominators_preserved': True, 'method_selection_performed': False,
            'selection12_or_confirmation24_started': False, 'final_payloads_opened': False}


def read_sealed_origin(path):
    path = Path(path).resolve()
    receipt = s.read(path.with_name(path.name+'.sha256.json'))
    if Path(receipt['path']).resolve() != path:
        raise PrescreenIntegrityError('Origin sidecar points elsewhere')
    panel.verify_ref(receipt)
    return s.read(path)


def call_snapshot():
    path = s.REPORTS/'inverse_calls.jsonl'
    payload = path.read_bytes() if path.exists() else b''
    calls = list(s.rows(path)) if path.exists() else []
    return calls, {'path': str(path), 'prefix_bytes': len(payload), 'prefix_sha256': hashlib.sha256(payload).hexdigest()}


def verify_call_prefix(receipt):
    path = Path(receipt['path'])
    if not path.exists() and receipt['prefix_bytes'] == 0:
        payload = b''
    else:
        with path.open('rb') as handle:
            payload = handle.read(receipt['prefix_bytes'])
    if len(payload) != receipt['prefix_bytes'] or hashlib.sha256(payload).hexdigest() != receipt['prefix_sha256']:
        raise PrescreenIntegrityError('The prior complete-inverse ledger prefix changed')


def origin_bundle():
    v002, digest002 = c2.verify_registration()
    result002 = s.read(c2.REPORT_DIR/'QUICK_RESULT.json')
    if result002.get('registration_sha256') != digest002:
        raise PrescreenIntegrityError('v002 result registration changed')
    origin_rows = result002['summary']['cells']; files = []
    expected = c2.planned_cells(v002['references'])
    if (len(origin_rows) != 4 or {(row['case_id'], row['method']) for row in origin_rows}
            != {(row['case_id'], row['method']) for row in expected}):
        raise PrescreenIntegrityError('Preserve the original four-cell v002 denominator, including failed attempts')
    for row in origin_rows:
        # Failed/interrupted attempts need not have produced parameters or
        # canonical arrays. Their durable attempt still forbids another fit.
        for name in ('inference_receipt', 'score_receipt', 'output_cache', 'output_metadata'):
            if row.get(name):
                panel.verify_ref(row[name]); files.append(row[name])
        if row.get('parameter_cache_receipt'):
            receipt = row['parameter_cache_receipt']
            for name in ('parameter_cache', 'parameter_metadata'):
                panel.verify_ref(receipt[name]); files.append(receipt[name])
            c2.load_parameters(receipt, c2.identity_for(row, row['registration_sha256']), v002)
        if row.get('output_cache'):
            q._load_output(row, identity=c2.identity_for(row, row['registration_sha256']))
        elif row.get('finite_output'):
            raise PrescreenIntegrityError('A prior finite-output claim has no retained canonical evidence')
    registration003 = read_sealed_origin(panel.REPORT_DIR/panel.REGISTRATION_NAME)
    digest003 = s.sha(panel.REPORT_DIR/panel.REGISTRATION_NAME)
    result003 = read_sealed_origin(panel.REPORT_DIR/panel.RESULT_NAME)
    qualification003 = read_sealed_origin(panel.REPORT_DIR/panel.QUALIFICATION_NAME)
    for name, digest in registration003['source_sha256'].items():
        if s.sha(s.ROOT/name) != digest:
            raise PrescreenIntegrityError('A frozen prior source changed: '+name)
    for ref in registration003['read_dependencies']:
        panel.verify_ref(ref)
    if (result003.get('registration_sha256') != digest003
            or qualification003.get('registration_sha256') != digest003
            or registration003['v002_registration_sha256'] != digest002
            or result003['summary']['planned_cells'] != 12
            or len(result003['summary']['cells']) != 12
            or registration003['planned_cells'] != panel.planned_cells(registration003['references'])):
        raise PrescreenIntegrityError('Prior complete denominators or registrations differ')
    files.extend(panel.file_ref(path) for path in (c2.REGISTRATION, c2.REGISTRATION_SHA,
        c2.REPORT_DIR/'QUICK_RESULT.json', panel.REPORT_DIR/panel.REGISTRATION_NAME,
        panel.REPORT_DIR/(panel.REGISTRATION_NAME+'.sha256.json'), panel.REPORT_DIR/panel.RESULT_NAME,
        panel.REPORT_DIR/(panel.RESULT_NAME+'.sha256.json'), panel.REPORT_DIR/panel.QUALIFICATION_NAME,
        panel.REPORT_DIR/(panel.QUALIFICATION_NAME+'.sha256.json')))
    evidence = [{'row': row, 'origin_kind': 'v002', 'origin_registration_sha256': digest002}
                for row in origin_rows]
    evidence.extend({'row': row.get('origin_cell', row), 'origin_kind': 'v003',
                     'origin_registration_sha256': digest003} for row in result003['summary']['cells'])
    return v002, registration003, evidence, files, qualification003


def prepare_plan():
    v002, v003, evidence, files, qualification = origin_bundle()
    reference = copy.deepcopy(v002['references'][0])
    spec = s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    protocol = s.read(s.REPORTS/'STUDY_PROTOCOL.yaml')
    if spec != protocol or spec['development']['candidates'] != list(CANDIDATES):
        raise PrescreenIntegrityError('The original nine-candidate scientific protocol changed')
    if (s.sha(s.REPORTS/'USER_EXECUTION_REQUEST.txt') != spec['user_request_sha256']
            or spec['development']['initializers'] != list(INITIALIZERS)
            or spec['development']['joint_per_frame_cost_limit_s'] != 180.):
        raise PrescreenIntegrityError('Original user request or deployable-initializer/cost contract changed')
    inventory = s.read(q.INVENTORY)['rows']
    matches = [row for row in inventory if row['case_id'] == CASE_ID]
    if len(matches) != 1 or matches[0]['raw_reference'] != reference['raw_reference']:
        raise PrescreenIntegrityError('Original prescreen raw metadata changed')
    calls, prefix = call_snapshot()
    planned = planned_cells(reference)
    catalog = reuse_catalog(planned, evidence, calls)
    planned = bind_plan(planned, catalog)
    base = [cell for cell in planned if cell['physical_options'] == q.PHYSICAL_OPTIONS]
    if len(base) != 2 or any(cell['execution_role'] != 'REUSE_PRIOR_C1V3' for cell in base):
        raise PrescreenIntegrityError('The two c1s1 old attempts must be reused')
    if sum(cell['execution_role'] == 'NEW_INVERSE' for cell in planned) > 16:
        raise PrescreenIntegrityError('Prescreen new-call maximum exceeded')
    return {'reference': reference, 'planned_cells': planned, 'reuse_catalog': catalog,
            'v002': v002, 'v003': v003, 'read_dependencies': files, 'prior_call_prefix': prefix,
            'prior_D_qualification': qualification}


def contract_sources(paths):
    refs, sources, persistence = [], {}, False
    for path in paths:
        value = s.read(path)
        if value.get('all_passed') is not True:
            raise PrescreenIntegrityError('Prescreen CPU contract did not pass')
        persistence |= value.get('E_compact_persistence_contracts_executed') is True
        for name, digest in value.get('source_sha256', {}).items():
            if s.sha(s.ROOT/name) != digest or (name in sources and sources[name] != digest):
                raise PrescreenIntegrityError('CPU contract source changed: '+name)
            sources[name] = digest
        refs.append(panel.file_ref(path))
    if not persistence or not set(SOURCES).issubset(sources):
        raise PrescreenIntegrityError('Current controller and E compact persistence CPU contracts are required')
    return refs, sources


def register(contract_paths, *, store=None):
    entry = panel.execution_entry()
    from .smooth_calibration import physics, contract
    from .numerical_strategy import REPAIRED_NUMERICS
    store = EStudyStore() if store is None else store
    path = Path(store.report_root)/REGISTRATION_NAME
    if path.exists():
        raise FileExistsError('Never replace the prescreen registration')
    contracts, contract_hashes = contract_sources(contract_paths)
    plan = prepare_plan(); old = plan.pop('v002'); prior = plan.pop('v003')
    p = physics('cpu')
    if contract(p) != old['calibration_contract']:
        raise PrescreenIntegrityError('The frozen C1v3 physical adapter changed')
    provenance = q._v3_provenances([plan['reference']])
    s.frozen_bridge()
    from research_v6 import inputs
    raw = inputs.load_raw(plan['reference'], p)
    provenance[CASE_ID].update(q.verify_initializer_raw(plan['reference'], raw, provenance[CASE_ID]))
    cached = s.arrays(provenance[CASE_ID]['state_cache']['path'])
    if set(('state', 'beta'))-set(cached) or not all(np.isfinite(cached[name]).all() for name in ('state', 'beta')):
        raise PrescreenIntegrityError('The unchanged raw-only v3 initializer cache must remain finite')
    sources = {**prior['source_sha256'], **contract_hashes,
               **{name: s.sha(s.ROOT/name) for name in SOURCES}}
    for name, digest in prior['source_sha256'].items():
        if sources[name] != digest or s.sha(s.ROOT/name) != digest:
            raise PrescreenIntegrityError('A frozen prior source cannot be replaced')
    dependencies = plan.pop('read_dependencies')+contracts+[panel.file_ref(path) for path in
        (s.REPORTS/'USER_EXECUTION_REQUEST.txt', s.ROOT/'research_closeout/execution_spec.yaml',
         s.REPORTS/'STUDY_PROTOCOL.yaml', q.INVENTORY, s.LEGACY/plan['reference']['raw_reference']['path'],
         panel.RUNTIME_CONTRACT)]
    dependencies += [provenance[CASE_ID]['state_cache'], provenance[CASE_ID]['receipt_journal']]
    # The old c1s1 objective retains its original fingerprint. Other pairs
    # explicitly identify the declared regularizer change, with no anchor edit.
    objectives = {candidate['id']: old['objective_fingerprint'] if candidate['continuum_multiplier'] == candidate['template_multiplier'] == 1.
        else s.canonical({'base_C1v3_c1s1_objective': old['objective_fingerprint'],
                          'regularizer_multipliers': candidate, 'scientific_sources': old['source_sha256']})
        for candidate in CANDIDATES}
    value = {'schema': VERSION+'-registration', 'time': s.now(), 'config': copy.deepcopy(CONFIG),
        **plan, 'contracts': contracts, 'source_sha256': sources,
        'read_dependencies': list({ref['path']: ref for ref in dependencies}.values()),
        'store_contract': store.contract(), 'format_contract': execution_format_contract(store),
        'calibration_contract': old['calibration_contract'], 'numerical_policy': copy.deepcopy(REPAIRED_NUMERICS),
        'objective_fingerprints': objectives, 'base_c1s1_objective_fingerprint': old['objective_fingerprint'],
        'v002_qualification_origins': s.read(c2.REPORT_DIR/'QUICK_RESULT.json')['summary']['cells'],
        'v3_initializer_provenance': provenance, 'execution_entry': entry,
        'new_complete_inverse_calls': 0, 'truth_loaded': False, 'old_data_migrated': False}
    value['config_fingerprint'] = s.canonical({'config': CONFIG, 'sources': sources,
        'store': value['store_contract'], 'format': value['format_contract'], 'objectives': objectives,
        'planned_cells': value['planned_cells'], 'prior_call_prefix': value['prior_call_prefix']})
    with phase_locks(store, value['config_fingerprint'], VERSION+'-register'):
        panel.seal(store, path, value)
    return value, s.sha(path)


def verify_registration(store):
    path = Path(store.report_root)/REGISTRATION_NAME
    value = panel.read_sealed(store, path)
    if (value['schema'] != VERSION+'-registration' or value['config'] != CONFIG
            or value['store_contract'] != store.contract() or value['format_contract'] != execution_format_contract(store)):
        raise PrescreenIntegrityError('Prescreen source/config/default-E contract changed')
    compact.verify_sources(value['source_sha256'])
    for ref in value['read_dependencies']:
        panel.verify_ref(ref)
    verify_call_prefix(value['prior_call_prefix'])
    expected = bind_plan(planned_cells(value['reference']), value['reuse_catalog'])
    if expected != value['planned_cells']:
        raise PrescreenIntegrityError('Registered semantic reuse plan changed')
    fingerprint = s.canonical({'config': CONFIG, 'sources': value['source_sha256'],
        'store': value['store_contract'], 'format': value['format_contract'], 'objectives': value['objective_fingerprints'],
        'planned_cells': value['planned_cells'], 'prior_call_prefix': value['prior_call_prefix']})
    if fingerprint != value['config_fingerprint']:
        raise PrescreenIntegrityError('Registration fingerprint changed')
    return value, s.sha(path)


def cache_for(store, p, cell, registration, digest, *, reference=None, qualification=False):
    reference = registration['reference'] if reference is None else reference
    return ECompactCache(Path(store.report_root)/('qualification' if qualification else 'inference'),
        Path(store.data_root)/('qualification' if qualification else 'inference'), p=p,
        identity={'registration_sha256': digest, 'method': cell['method'], 'case_id': cell['case_id'],
                  'raw_sha256': cell['raw_sha256'], 'controller_version': VERSION},
        input_reference=reference['raw_reference'], calibration=registration['calibration_contract'],
        config={'grid': 256, 'physical_options': cell.get('physical_options', q.PHYSICAL_OPTIONS),
                'numerical_options': CONFIG['numerical_options'], 'execution_storage_contract': store.contract(),
                'operation': 'saved_endpoint_forward_only' if qualification else 'prescreen_inference'},
        source_sha256=registration['source_sha256'], budget=store.budget())


def qualify(device='cuda', *, store=None):
    if device != 'cuda':
        raise PrescreenIntegrityError('Native E qualification is a separate root-owned GPU operation')
    panel.execution_entry()
    from .smooth_calibration import physics, contract
    import torch
    store = EStudyStore() if store is None else store
    registration, digest = verify_registration(store)
    path = Path(store.report_root)/QUALIFICATION_NAME
    if path.exists():
        return panel.read_sealed(store, path)
    with phase_locks(store, digest, VERSION+'-E-qualification'):
        torch.set_num_threads(2); p = physics(device)
        if p.M.device.type != 'cuda' or contract(p) != registration['calibration_contract']:
            raise PrescreenIntegrityError('Native C1 physics differs')
        s.frozen_bridge()
        from research_v6 import inputs
        old = s.read(c2.REGISTRATION)
        def action(origin):
            reference = next(ref for ref in old['references'] if ref['case_id'] == origin['case_id'])
            raw = inputs.load_raw(reference, p)
            parameters, metadata = c2.load_parameters(origin['parameter_cache_receipt'],
                c2.identity_for(origin, origin['registration_sha256']), old)
            arrays = s.arrays(origin['output_cache']['path'])
            out = compact.forward_reconstruct(p, raw, parameters, metadata['inference_info'], grid=256)
            original_match = panel.byte_matches(out, arrays)
            for name, key in (('T', 'temperature_color_K'), ('u', 'template_postblur_e_ref'),
                    ('C', 'continuum_postblur_bands'), ('predicted_raw_e', 'predicted_raw_e'),
                    ('valid', 'valid'), ('low_signal', 'low_signal')):
                out[key] = torch.as_tensor(arrays[name], device=p.M.device)
            out['numerical_valid'] = bool(origin['production_numerical_certified'])
            cache = cache_for(store, p, origin, registration, digest, reference=reference, qualification=True)
            ticket = cache.begin(raw)  # Owner reservation only; this operation never invokes an inverse.
            result = cache.persist(ticket, out, raw) if ticket.get('run_new_inverse') else cache.resume(raw)
            readback = panel.byte_matches(cache.load_for_score(raw), arrays) if result.get('score_allowed') else {}
            passed = all(original_match.values()) and len(readback) == 6 and all(readback.values())
            return {'passed': passed, 'status': 'QUALIFIED' if passed else 'QUALIFICATION_FAILED',
                'original_canonical_byte_matches': original_match, 'compact_readback_original_byte_matches': readback,
                'compact_result': result, 'compact_report_root': str(cache.report), 'compact_data_root': str(cache.data),
                'complete_inverse_calls': 0, 'old_cache_converted_or_deleted': False}
        value = panel.qualification_matrix(registration['v002_qualification_origins'], action)
        value.update(time=s.now(), registration_sha256=digest, native_GPU=True, runtime=compact.runtime_identity(p),
            source_sha256=registration['source_sha256'], format_contract=registration['format_contract'],
            store_contract=registration['store_contract'], storage_ledger=store.budget().ledger(),
            old_D_qualification_not_relabelled_as_E=True)
        verify_registration(store); panel.seal(store, path, value)
        return value


def evaluate_cell(cell, *, cache, raw, action, score_action, count_call, store, digest,
                  verify_sources=lambda: None, score_only=False):
    if score_only and cache.inspect().get('status') == 'NEW':
        return {**cell, 'status': 'NOT_EXECUTED_SCORE_ONLY', 'finite_output': False,
            'production_numerical_certified': False, 'score_status': 'NOT_EXECUTED', 'metrics': None,
            'new_inverse_calls_this_run': 0, 'inference_attempted': False}
    def count(ignored_phase, method, case_id):
        if score_only:
            raise PrescreenIntegrityError('Score-only execution cannot issue a new inference call')
        return count_call(PHASE, method, case_id)
    row = panel.evaluate_new_cell(cell, cache=cache, raw=raw, action=action, score_action=score_action,
        count_call=count, store=store, digest=digest, verify_sources=verify_sources)
    # A crash can occur after the global append but before CALL_RECORDED.json.
    # Preserve that counted quota even when no inference callback began.
    prior_calls, _ = call_snapshot()
    recorded = sum(not call.get('reused') and call.get('phase') == PHASE
                   and call.get('method') == cell['method'] and call.get('case_id') == cell['case_id']
                   for call in prior_calls)
    if recorded > 1:
        raise PrescreenIntegrityError('The ledger contains repeated prescreen calls for one method/raw')
    row['new_inverse_calls_panel_total'] = max(recorded, row.get('new_inverse_calls_panel_total', 0))
    row['global_counted_prescreen_attempts_for_cell'] = recorded
    info = {}
    if (cache.report/'PREPARED.json').exists():
        info = cache._read('PREPARED.json')['original_info']
    row['numerical_evidence'] = numerical_evidence(info, row.get('production_numerical_certified', False))
    if row.get('production_numerical_certified') and not row['numerical_evidence']['passed']:
        row.update(controller_stop=True, numerical_evidence_integrity_failed=True)
    return row


def evaluate_reused_cell(cell, item, *, cache, raw, score_action, store, digest,
                         verify_sources=lambda: None):
    """Keep historical rows; a failed external score may retry from qualified bytes."""
    original = reuse_row(cell, item, digest)
    if original['score_status'] == 'SCORED' or not original['finite_output']:
        return original
    state = cache.inspect()
    if state.get('status') != 'LOSSLESS_READY' or not state.get('score_allowed'):
        return {**original, 'controller_stop': True, 'origin_score_retry_requires_qualified_E_bytes': True}
    def forbidden(*args):
        raise PrescreenIntegrityError('Prior-method reuse cannot invoke or count another inverse')
    result = evaluate_cell(cell, cache=cache, raw=raw, action=forbidden, score_action=score_action,
        count_call=forbidden, store=store, digest=digest, verify_sources=verify_sources, score_only=True)
    result.update(origin_evidence=copy.deepcopy(item), origin_scoring_retry_only=True,
        historical_result_relabelled=False, inference_reused=True)
    if result.get('new_inverse_calls_this_run') or result.get('inference_attempted'):
        raise PrescreenIntegrityError('Reused scoring unexpectedly consumed an inference ticket')
    return result


def counted_once(method, case_id):
    from . import experiment_registry as registry
    allowed = {method_for(candidate, initializer) for candidate in CANDIDATES for initializer in INITIALIZERS}
    if method not in allowed or case_id != CASE_ID:
        raise PrescreenIntegrityError('Cannot count an undeclared method/raw pair')
    calls, _ = call_snapshot()
    if any(not row.get('reused') and row.get('method') == method and row.get('case_id') == case_id for row in calls):
        raise PrescreenIntegrityError('An earlier same-method/raw attempt exists; another inverse is forbidden')
    if sum(not row.get('reused') and row.get('phase') == PHASE for row in calls) >= 16:
        raise PrescreenIntegrityError('Prescreen16-new-call cap reached')
    return registry.count_call(PHASE, method, case_id)


def verify_qualified_cache_readback(store, p, registration, digest):
    """Before issuing any ticket, recheck the four saved qualification states."""
    from research_v6 import inputs
    old = s.read(c2.REGISTRATION)
    for origin in registration['v002_qualification_origins']:
        reference = next(ref for ref in old['references'] if ref['case_id'] == origin['case_id'])
        raw = inputs.load_raw(reference, p)
        cache = cache_for(store, p, origin, registration, digest, reference=reference, qualification=True)
        out = cache.load_for_score(raw)
        if not all(panel.byte_matches(out, s.arrays(origin['output_cache']['path'])).values()):
            raise PrescreenIntegrityError('E qualification no longer reproduces the original saved bytes')


def prescreen(device='cuda', *, score_only=False, store=None):
    if device != 'cuda':
        raise PrescreenIntegrityError('Formal prescreen is root-owned native GPU only')
    panel.execution_entry()
    from .smooth_calibration import physics, contract
    from .numerical_strategy import repaired_infer
    from .development import cached_v3
    from . import metrics
    import torch
    store = EStudyStore() if store is None else store
    registration, digest = verify_registration(store)
    planned = registration['planned_cells']; rows = []
    with phase_locks(store, digest, VERSION+('-score-only' if score_only else '-prescreen')):
        terminal = Path(store.report_root)/RESULT_NAME
        try:
            torch.set_num_threads(2); p = physics(device)
            if contract(p) != registration['calibration_contract']:
                raise PrescreenIntegrityError('The registered physical calibration changed')
            qualification = panel.read_sealed(store, Path(store.report_root)/QUALIFICATION_NAME)
            panel.require_qualification(qualification, registration, digest, compact.runtime_identity(p))
            s.frozen_bridge()
            from research_v6 import inputs
            reference = registration['reference']; raw = inputs.load_raw(reference, p)
            verify_qualified_cache_readback(store, p, registration, digest)
            if terminal.exists():
                completed = panel.read_sealed(store, terminal)
                for previous in completed['summary']['cells']:
                    if previous.get('execution_role') == 'NEW_INVERSE' and previous.get('score_status') == 'SCORED':
                        cache_for(store, p, previous, registration, digest).load_for_score(raw)
                return completed
            q.verify_initializer_raw(reference, raw, registration['v3_initializer_provenance'][CASE_ID])
            def external_score(out, cell):
                tensors = {key: torch.as_tensor(value, device=p.M.device) if isinstance(value, np.ndarray) else value
                           for key, value in out.items()}
                truth = inputs.load_diagnostic_truth(reference, p)
                return metrics.score(tensors, truth, raw, p, detector=None, method=cell['method'],
                                     application_domain=VERSION+'-single-development-raw')
            for cell in planned:
                if cell['execution_role'] == 'REUSE_PRIOR_C1V3':
                    item = registration['reuse_catalog'][cell['reuse_key']]
                    if item['row'].get('finite_output') and item['row'].get('score_status') != 'SCORED':
                        cache = cache_for(store, p, cell, registration, digest, qualification=True)
                        row = evaluate_reused_cell(cell, item, cache=cache, raw=raw,
                            score_action=lambda out, cell=cell: external_score(out, cell),
                            store=store, digest=digest, verify_sources=lambda: verify_registration(store))
                    else:
                        row = reuse_row(cell, item, digest)
                    rows.append(row)
                    if row.get('controller_stop'):
                        raise PrescreenIntegrityError('Prior-method representation failed; another inverse is forbidden')
            for cell in planned:
                if cell['execution_role'] != 'NEW_INVERSE':
                    continue
                verify_registration(store)
                cache = cache_for(store, p, cell, registration, digest)
                def action(cell=cell):
                    state = cached_v3(reference, p)[0] if cell['initializer'] == 'v3' else None
                    out = repaired_infer(raw, p, initializer=cell['initializer'], v3_state=state,
                        physical_options=copy.deepcopy(cell['physical_options']),
                        numerical_options=copy.deepcopy(CONFIG['numerical_options']))
                    effective = {'calibration_contract': registration['calibration_contract'],
                        'objective_fingerprint': registration['objective_fingerprints'][cell['candidate_id']],
                        'config_fingerprint': registration['config_fingerprint']}
                    return c2.wrap_output(out, cell, effective)
                def score(out, cell=cell):
                    return external_score(out, cell)
                def count(phase, method, case_id):
                    return counted_once(method, case_id)
                print('BEGIN prescreen '+cell['method']+' '+cell['case_id'], flush=True)
                row = evaluate_cell(cell, cache=cache, raw=raw, action=action, score_action=score,
                    count_call=count, store=store, digest=digest, score_only=score_only,
                    verify_sources=lambda: verify_registration(store))
                rows.append(row)
                store.append(Path(store.report_root)/'progress.jsonl', {'time': s.now(), 'row': row})
                print('END prescreen '+cell['method']+' '+row['status'], flush=True)
                if row.get('controller_stop'):
                    raise PrescreenIntegrityError('Preserve all remaining denominator cells; controller stopped')
        except Exception as error:
            existing = {semantic_key(row) for row in rows}
            rows.extend(reuse_row(cell, registration['reuse_catalog'][cell['reuse_key']], digest)
                        for cell in planned if cell['execution_role'] == 'REUSE_PRIOR_C1V3' and semantic_key(cell) not in existing)
            value = {'time': s.now(), 'status': 'PRESCREEN_CONTROLLER_STOP', 'registration_sha256': digest,
                'summary': summary(planned, rows), 'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'storage_ledger': store.budget().ledger(),
                'rerun_inverse_permitted': False}
            return panel.run_receipt(store, Path(store.report_root)/('STOP_'+uuid.uuid4().hex+'.json'), value,
                                     failure_status='PRESCREEN_STOP_RECEIPT_FAILED')
        value = {'time': s.now(), 'status': 'C1V3_PRESCREEN_COMPLETE', 'registration_sha256': digest,
            'summary': summary(planned, rows), 'score_only': score_only,
            'storage_ledger': store.budget().ledger(), 'method_selection_performed': False,
            'selection12_or_confirmation24_started': False, 'historical_threshold_transferred': False,
            'final_payloads_opened': False, 'scientific_acceptance_claimed': False}
        verify_registration(store)
        if value['summary']['scoring_failed_cells'] or value['summary']['not_executed_cells']:
            value['status'] = 'C1V3_PRESCREEN_SCORE_RETRY_OR_INCOMPLETE'
            destination = Path(store.report_root)/('RETRY_'+uuid.uuid4().hex+'.json')
        else:
            destination = terminal
        return panel.run_receipt(store, destination, value, failure_status='PRESCREEN_TERMINAL_RECEIPT_FAILED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    item = sub.add_parser('register'); item.add_argument('--contract', type=Path, action='append', required=True)
    for name in ('qualify', 'prescreen'):
        item = sub.add_parser(name); item.add_argument('--device', choices=('cuda',), default='cuda')
        if name == 'prescreen':
            item.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        value = prepare_plan()
        value = {'status': 'PRESCREEN_PREPARED_NOT_REGISTERED', 'config': CONFIG,
            'planned_cells': value['planned_cells'], 'reuse_catalog': value['reuse_catalog'],
            'new_complete_inverse_calls': 0, 'new_calls_planned': sum(cell['execution_role'] == 'NEW_INVERSE' for cell in value['planned_cells'])}
    elif args.command == 'register':
        _, digest = register(args.contract)
        value = {'status': 'PRESCREEN_REGISTERED', 'registration_sha256': digest, 'new_complete_inverse_calls': 0}
    elif args.command == 'qualify':
        value = qualify(args.device)
    else:
        value = prescreen(args.device, score_only=args.score_only)
    print(json.dumps(s.plain(value), ensure_ascii=True, indent=2), flush=True)
    failed = value.get('all_passed') is False or any(word in value.get('status', '') for word in ('STOP', 'RETRY', 'FAILED'))
    raise SystemExit(2 if failed else 0)


if __name__ == '__main__':
    main()
