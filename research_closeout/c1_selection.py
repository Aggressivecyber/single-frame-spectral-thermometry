"""Registered1080-cell C1v3 first stage, scheduled only in explicit small shards.

The numerical estimator, original nine multiplier pairs, codec and three-
endpoint/two-plateau rule are reused unchanged. Every prescreen-qualified
candidate retains all120 required cells even after a numerical failure.
No truth-based pruning, top-two choice, initializer selection or24-mother
confirmation is performed here. Old attempts, including aliases whose frozen
physical identity agrees, never authorize another inverse.
"""
from . import storage as s
from . import c1_development as dev
from . import c1_small_panel as panel
from . import compact_joint_cache as compact
from . import calibration_quick as c2
from . import quick_validation as q
import argparse
import copy
import json
from pathlib import Path
import re
import traceback
import uuid
import numpy as np

VERSION = 'C1v3-first-stage-1080-v001'
PHASE = 'C1v3_selection_first_stage_v001'
REPORT_DIR = s.REPORTS/'repair_001/c1_selection_001'
DATA_DIR = s.DATA/'repair_001/c1_selection_001'
CPU_REPORT_DIR = REPORT_DIR/'cpu_contracts'
REGISTRATION_NAME = 'SELECTION_REGISTRATION.json'
QUALIFICATION_NAME = 'NAMESPACE_FORWARD_QUALIFICATION.json'
RESULT_NAME = 'FIRST_STAGE_RESULT.json'
CONDITIONS = ('blank', 'positive_weak', 'negative_weak', 'positive_normal', 'negative_normal')
SOURCES = ('research_closeout/c1_selection.py', 'research_closeout/tests/c1_selection.py')
INSTANCE_LIMIT = 2*2**30
DEVELOPMENT_PHASES = ('development', 'repair_quick', 'C1v3_small_panel_v003',
                      'C1v3_development_prescreen_v001', PHASE)
CONFIG = {'version': VERSION, 'stage': 'first_stage_only', 'planned_mothers': 12, 'planned_raws': 60,
          'planned_candidates': 9, 'initializers': list(dev.INITIALIZERS), 'planned_cells': 1080,
          'known_reused_cells': 28, 'maximum_new_inverse_calls': 1052,
          'grid': 256, 'native_shape': [256, 256], 'numerical_options': {'cost_limit_s': 180.},
          'maximum_inverse_wall_s_excluding_overhead': 189360, 'instance_limit_bytes': INSTANCE_LIMIT,
          'global_new_study_limit_bytes': 4*2**30, 'E_free_floor_bytes': 2*2**30,
          'global_budget_roots': [str(path) for path in dev.GLOBAL_ROOTS],
          'success1052_conservative_bytes': 1457550208, 'planned_failure_headroom_bytes': 192*2**20,
          'qualification4_success_conservative_bytes': 5542016, 'extra_controller_metadata_plan_bytes': 16*2**20,
          'combined_conservative_plan_bytes': 1681196032, 'qualification_complete_inverse_calls': 0,
          'maximum_shard_raws': 6, 'shard_is_scheduling_only': True, 'automatic_full_run_supported': False,
          'ordinary_numerical_failure_policy': 'execute all120 cells of every Q9 candidate before all-certification decision',
          'truth_pruning': False, 'resource_stop_is_scientific_exclusion': False,
          'detector': None, 'historical_detector_threshold_transferred': False,
          'top_two_or_initializer_selected': False, 'confirmation24_started': False,
          'global_new_inverse_call_cap': 12000, 'initializer_provenance_raws': 180,
          'current_phase_uses_stricter_original12000_enforcement': True,
          'human_authorized_global_cap': 16000,
          'semantic_development_call_cap': 3400, 'semantic_development_phases': list(DEVELOPMENT_PHASES),
          'codec_or_numerical_arithmetic_changed': False}


class SelectionIntegrityError(RuntimeError):
    pass


class SelectionBudget(compact.Budget):
    def __init__(self, roots, *, free_bytes=None):
        super().__init__(roots, global_roots=dev.GLOBAL_ROOTS,
                         instance_limit_bytes=INSTANCE_LIMIT, free_bytes=free_bytes)


class SelectionStore(dev.EStudyStore):
    def __init__(self):
        self.report_root, self.data_root = REPORT_DIR.resolve(), DATA_DIR.resolve()

    def budget(self):
        return SelectionBudget((self.report_root, self.data_root))

    def contract(self):
        return {**super().contract(), 'version': VERSION+'-E-accounting', 'instance_limit_bytes': INSTANCE_LIMIT,
                'unchanged_guard_class': 'research_closeout.c1_development.ECompactCache',
                'rolling_summary_is_rebuildable_current_index': True,
                'case_states_shard_receipts_and_final_result_are_immutable': True}


def references(inventory, spec):
    rows = inventory['rows'] if isinstance(inventory, dict) else inventory
    all_dev = [copy.deepcopy(row) for row in rows if row.get('cohort') == 'development'
               and row.get('input_kind') == 'noise' and row.get('noise_index') == 0]
    if len(all_dev) != 180 or len({row['case_id'] for row in all_dev}) != 180:
        raise SelectionIntegrityError('The existing180 development/noise raws are required')
    for row in all_dev:
        if row.get('fixture_only') or row['raw_reference']['shape'] != [256, 256]:
            raise SelectionIntegrityError('Only unchanged native256 development inputs are allowed')
        field = 'temperature_geometry' if row.get('temperature_geometry') is not None else 'geometry'
        if not row.get(field) or not row.get('template_geometry'):
            raise SelectionIntegrityError('Preserve actual geometry and its inventory source')
        row['quick_geometry'] = {'temperature_geometry': row[field], 'template_geometry': row['template_geometry'],
            'temperature_geometry_source': 'research_v6/input_inventory.json:'+field,
            'template_geometry_source': 'research_v6/input_inventory.json:template_geometry',
            'temperature_geometry_fallback_declared': field == 'geometry'}
    mothers = spec['development']['selection_mothers']
    if len(mothers) != 12 or len(set(mothers)) != 12 or spec['development']['candidates'] != list(dev.CANDIDATES):
        raise SelectionIntegrityError('The original12 mothers and nine pairs cannot change')
    selected = []
    for mother in mothers:
        for condition in CONDITIONS:
            matches = [row for row in all_dev if row['mother_id'] == mother and row['condition'] == condition]
            if len(matches) != 1:
                raise SelectionIntegrityError('Missing/ambiguous original first-stage raw')
            selected.append(matches[0])
    if len(selected) != 60:
        raise SelectionIntegrityError('First stage requires60 distinct old raws')
    return selected, sorted(all_dev, key=lambda row: row['case_id'])


def planned_cells(refs):
    if len(refs) != 60 or len({row['case_id'] for row in refs}) != 60:
        raise SelectionIntegrityError('Freeze all60 first-stage raws before scheduling shards')
    cells = []
    for ref in refs:
        for candidate in dev.CANDIDATES:
            for initializer in dev.INITIALIZERS:
                cells.append({**{key: ref[key] for key in ('case_id', 'mother_id', 'condition', 'input_kind', 'quick_geometry')},
                    'raw_sha256': ref['raw_reference']['sha256'], 'candidate_id': candidate['id'],
                    'family': 'joint_C1v3', 'method': dev.method_for(candidate, initializer), 'initializer': initializer,
                    'physical_options': {'grid': 256, **{key: candidate[key] for key in
                                         ('continuum_multiplier', 'template_multiplier')}}})
    return cells


def locate_origin(row):
    if row.get('compact_report_root') and row.get('compact_data_root'):
        return {'kind': 'compact', 'report': row['compact_report_root'], 'data': row['compact_data_root'],
                'case_id': row['case_id'], 'method': row['method'], 'raw_sha256': row['raw_sha256']}
    if row.get('parameter_cache_receipt') and row.get('output_cache'):
        return {'kind': 'v002', 'row': copy.deepcopy(row)}
    if row.get('origin_evidence', {}).get('locator'):
        return copy.deepcopy(row['origin_evidence']['locator'])
    if row.get('origin_evidence', {}).get('row'):
        return locate_origin(row['origin_evidence']['row'])
    if row.get('origin_cell'):
        return locate_origin(row['origin_cell'])
    return None


def origin_info(row, locator):
    if row.get('info'):
        return row['info']
    if locator and locator['kind'] == 'compact':
        path = Path(locator['report'])/'PREPARED.json'
        if path.exists():
            return compact._read_sealed(path)['original_info']
    if locator and locator['kind'] == 'v002':
        return locator['row'].get('info', {})
    return {}


def normalize_origin(item, objectives, calibration):
    """Canonical reuse identity is physical; an old label is retained verbatim."""
    original = copy.deepcopy(item['row']); locator = locate_origin(original)
    info = origin_info(original, locator)
    params = original.get('physical_options') or info.get('anchor_regscale_contract', {}).get('parameters') or item.get('physical_options')
    if not params:
        raise SelectionIntegrityError('A prior C1 attempt needs frozen physical multiplier metadata')
    candidate = next((candidate for candidate in dev.CANDIDATES if params.get('grid') == 256
        and all(params.get(key) == candidate[key] for key in ('continuum_multiplier', 'template_multiplier'))), None)
    if candidate is None or original.get('initializer') not in dev.INITIALIZERS:
        raise SelectionIntegrityError('Prior physical identity is outside the original finite grid')
    profile = info.get('anchor_regscale_contract', {}).get('parameters', {})
    if profile and any(profile.get(key) != params.get(key) for key in ('grid', 'continuum_multiplier', 'template_multiplier')):
        raise SelectionIntegrityError('Prior label/config and effective physical objective disagree')
    calibrated = info.get('C1v3_adapter_fingerprint', item.get('frozen_registration_calibration_fingerprint'))
    objective = info.get('objective_fingerprint', item.get('frozen_registration_objective_fingerprints', {}).get(candidate['id']))
    if calibrated != calibration['fingerprint']:
        raise SelectionIntegrityError('A prior calibration differs')
    if objective != objectives[candidate['id']]:
        raise SelectionIntegrityError('A prior regularized objective differs')
    canonical_method = dev.method_for(candidate, original['initializer'])
    if locator and locator['kind'] == 'compact':
        locator.update(expected_physical_options={'grid': 256, **{key: candidate[key] for key in
            ('continuum_multiplier', 'template_multiplier')}}, expected_calibration_fingerprint=calibrated)
    normalized = {**original, 'method': canonical_method, 'candidate_id': candidate['id'], 'info': info,
                  'physical_options': {'grid': 256, **{key: candidate[key] for key in ('continuum_multiplier', 'template_multiplier')}}}
    return {**item, 'row': normalized, 'original_row': original, 'locator': locator,
            'original_method_label': original['method'], 'physical_identity_verified': True,
            'alias_resolved_without_inverse': original['method'] != canonical_method}


def origin_files(locator):
    if not locator:
        return []
    if locator['kind'] == 'v002':
        row = locator['row']; refs = [row[key] for key in ('output_cache', 'output_metadata', 'inference_receipt', 'score_receipt') if row.get(key)]
        refs += [row['parameter_cache_receipt'][key] for key in ('parameter_cache', 'parameter_metadata')]
        for ref in refs: panel.verify_ref(ref)
        return refs
    report = Path(locator['report']); refs = []
    for name in ('STARTED.json', 'PREPARED.json', 'PARAMETERS.json', 'RESULT.json', 'FAILURE.json', 'RECONSTRUCTION_FAILURE.json'):
        path = report/name
        if path.exists():
            value = compact._read_sealed(path)
            refs += [panel.file_ref(path), panel.file_ref(path.with_name(path.name+'.sha256.json'))]
            if name == 'PARAMETERS.json':
                compact.verify_file(value['parameter_cache']); refs.append(value['parameter_cache'])
    return refs


def prepare_plan():
    old, old_digest = dev.verify_registration(dev.EStudyStore())
    result = panel.read_sealed(dev.EStudyStore(), dev.REPORT_DIR/dev.RESULT_NAME)
    if (result['registration_sha256'] != old_digest or result['summary']['planned_cells'] != 18
            or len(result['summary']['cells']) != 18 or result['summary']['numerically_qualified_candidates'] != 9):
        raise SelectionIntegrityError('The complete immutable Q9 prescreen is required')
    spec = s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    if spec != s.read(s.REPORTS/'STUDY_PROTOCOL.yaml') or s.sha(s.REPORTS/'USER_EXECUTION_REQUEST.txt') != spec['user_request_sha256']:
        raise SelectionIntegrityError('Original scientific request/protocol changed')
    refs, all_dev = references(s.read(q.INVENTORY), spec)
    planned = planned_cells(refs)
    v002, v003, prior, files, _ = dev.origin_bundle()
    evidence = [{'row': row, 'origin_kind': 'prescreen', 'origin_registration_sha256': old_digest}
                for row in result['summary']['cells']]
    evidence += prior
    # These identities come from the already verified, frozen registrations,
    # including failed attempts that never returned an inference info object.
    for item in evidence:
        item['frozen_registration_calibration_fingerprint'] = old['calibration_contract']['fingerprint']
        item['frozen_registration_objective_fingerprints'] = old['objective_fingerprints']
    normalized = [normalize_origin(item, old['objective_fingerprints'], old['calibration_contract']) for item in evidence]
    calls, prefix = dev.call_snapshot()
    catalog = dev.reuse_catalog(planned, normalized, calls)
    planned = dev.bind_plan(planned, catalog)
    if sum(cell['execution_role'] == 'NEW_INVERSE' for cell in planned) > 1052:
        raise SelectionIntegrityError('All18 prescreen and10 additional old panel cells must be reused')
    for item in catalog.values(): files += origin_files(item.get('locator'))
    old_qualification = dev.read_sealed_origin(dev.REPORT_DIR/dev.QUALIFICATION_NAME)
    if not old_qualification.get('all_passed') or len(old_qualification.get('cells', [])) != 4:
        raise SelectionIntegrityError('Reuse the already qualified E arithmetic/codec evidence')
    qualification_origins = [{'kind': 'compact', 'report': row['compact_report_root'], 'data': row['compact_data_root'],
        'case_id': row['case_id'], 'method': row['method']} for row in old_qualification['cells']]
    for locator in qualification_origins: files += origin_files(locator)
    paths = (dev.REPORT_DIR/dev.REGISTRATION_NAME, dev.REPORT_DIR/(dev.REGISTRATION_NAME+'.sha256.json'),
        dev.REPORT_DIR/dev.RESULT_NAME, dev.REPORT_DIR/(dev.RESULT_NAME+'.sha256.json'),
        dev.REPORT_DIR/dev.QUALIFICATION_NAME, dev.REPORT_DIR/(dev.QUALIFICATION_NAME+'.sha256.json'),
        q.INVENTORY, s.ROOT/'research_closeout/execution_spec.yaml', s.REPORTS/'STUDY_PROTOCOL.yaml',
        s.REPORTS/'USER_EXECUTION_REQUEST.txt', panel.RUNTIME_CONTRACT)
    paths += (s.REPORTS/'repair_001/SELECTION_STORAGE_AND_BUDGET_001.json',
              s.REPORTS/'repair_001/SELECTION_STORAGE_AND_BUDGET_001.md',
              s.REPORTS/'repair_001/INVERSE_BUDGET_EXTENSION_16000.json')
    files += [panel.file_ref(path) for path in paths]
    return {'references': refs, 'initializer_references180': all_dev, 'planned_cells': planned,
        'reuse_catalog': catalog, 'prior_call_prefix': prefix, 'prior_registration_sha256': old_digest,
        'prior_qualification': old_qualification, 'qualification_origins': qualification_origins,
        'calibration_contract': old['calibration_contract'], 'objective_fingerprints': old['objective_fingerprints'],
        'numerical_policy': old['numerical_policy'], 'inherited_source_sha256': old['source_sha256'],
        'read_dependencies': list({ref['path']: ref for ref in files}.values())}


def audit_initializers(refs, p):
    if len(refs) != 180:
        raise SelectionIntegrityError('Audit each of the existing180 development raw/state caches')
    provenance = q._v3_provenances(refs)
    s.frozen_bridge()
    from research_v6 import inputs
    from .development import cached_v3
    for index, ref in enumerate(refs, 1):
        raw = inputs.load_raw(ref, p)
        provenance[ref['case_id']].update(q.verify_initializer_raw(ref, raw, provenance[ref['case_id']]))
        state, beta = cached_v3(ref, p)
        if (tuple(state.shape) != (1, 5, 256, 256) or tuple(beta.shape) != (1, 3)
                or not np.isfinite(compact.array(state)).all() or not np.isfinite(compact.array(beta)).all()):
            raise SelectionIntegrityError('Frozen v3 state/beta cache is malformed/nonfinite')
        if index % 30 == 0: print('INITIALIZER_AUDIT '+str(index)+'/180', flush=True)
    return provenance


def contract_sources(paths):
    refs, hashes, passed = [], {}, False
    for path in paths:
        value = s.read(path)
        if value.get('all_passed') is not True:
            raise SelectionIntegrityError('Selection CPU contracts did not pass')
        passed |= value.get('selection_E_codec_CPU_contracts_executed') is True
        for name, digest in value.get('source_sha256', {}).items():
            if s.sha(s.ROOT/name) != digest or (name in hashes and hashes[name] != digest):
                raise SelectionIntegrityError('CPU dependency digest differs')
            hashes[name] = digest
        refs.append(panel.file_ref(path))
    if not passed or not set(SOURCES).issubset(hashes):
        raise SelectionIntegrityError('Current selection/controller and reused E codec CPU evidence required')
    return refs, hashes


def fingerprint(value):
    return s.canonical({key: value[key] for key in ('config', 'source_sha256', 'store_contract',
        'format_contract', 'objective_fingerprints', 'planned_cells', 'prior_call_prefix')})


def register(contract_paths, *, store=None):
    entry = panel.execution_entry()
    from .smooth_calibration import physics, contract
    store = SelectionStore() if store is None else store
    path = Path(store.report_root)/REGISTRATION_NAME
    if path.exists(): raise FileExistsError('Never replace the first-stage registration')
    contracts, extra = contract_sources(contract_paths)
    plan = prepare_plan(); p = physics('cpu')
    if contract(p) != plan['calibration_contract']: raise SelectionIntegrityError('C1 calibration changed')
    provenance = audit_initializers(plan['initializer_references180'], p)
    sources = {**plan.pop('inherited_source_sha256'), **extra, **{name: s.sha(s.ROOT/name) for name in SOURCES}}
    compact.verify_sources(sources)
    deps = plan.pop('read_dependencies')+contracts
    for ref in plan['initializer_references180']:
        deps += [panel.file_ref(s.LEGACY/ref['raw_reference']['path']), provenance[ref['case_id']]['state_cache'],
                 provenance[ref['case_id']]['receipt_journal']]
    value = {'schema': VERSION+'-registration', 'time': s.now(), 'config': copy.deepcopy(CONFIG), **plan,
        'source_sha256': sources, 'contracts': contracts, 'read_dependencies': list({ref['path']: ref for ref in deps}.values()),
        'v3_initializer_provenance': provenance, 'store_contract': store.contract(),
        'format_contract': dev.execution_format_contract(store), 'execution_entry': entry,
        'new_complete_inverse_calls': 0, 'truth_loaded': False, 'confirmation24_started': False}
    value['config_fingerprint'] = fingerprint(value)
    with dev.phase_locks(store, value['config_fingerprint'], VERSION+'-register'):
        panel.seal(store, path, value)
    return value, s.sha(path)


def verify_registration(store, *, verify_inputs=True):
    path = Path(store.report_root)/REGISTRATION_NAME; value = panel.read_sealed(store, path)
    if (value.get('schema') != VERSION+'-registration' or value['config'] != CONFIG
            or value['store_contract'] != store.contract() or value['format_contract'] != dev.execution_format_contract(store)
            or value['config_fingerprint'] != fingerprint(value)):
        raise SelectionIntegrityError('Selection config/source/store/codec registration changed')
    compact.verify_sources(value['source_sha256']); dev.verify_call_prefix(value['prior_call_prefix'])
    if verify_inputs:
        for ref in value['read_dependencies']: panel.verify_ref(ref)
    if dev.bind_plan(planned_cells(value['references']), value['reuse_catalog']) != value['planned_cells']:
        raise SelectionIntegrityError('Full semantic1080-cell matrix changed')
    return value, s.sha(path)


def read_existing_compact(locator, p, raw):
    """Read-only old-state verification; never writes a failure into old evidence."""
    report, data = Path(locator['report']), Path(locator['data'])
    names = ('STARTED.json', 'PREPARED.json', 'PARAMETERS.json', 'RESULT.json')
    if any(not (report/name).exists() or not (report/(name+'.sha256.json')).exists() for name in names):
        raise SelectionIntegrityError('Old compact instance is incomplete; no recovery write is permitted')
    if any((report/name).exists() for name in ('FAILURE.json', 'RECONSTRUCTION_FAILURE.json')):
        raise SelectionIntegrityError('A failed old compact instance cannot be used for scoring')
    records = {name: compact._read_sealed(report/name) for name in names}
    started, prepared, parameters, result = (records[name] for name in names)
    context = started['context']; key = started['cache_key']; digest = s.canonical(context)
    if any(record.get('cache_key') != key or record.get('context_fingerprint') != digest or record.get('schema') != compact.SCHEMA
           for record in records.values()) or prepared['context'] != context:
        raise SelectionIntegrityError('Old compact context/key/schema chain differs')
    identity = context['identity']
    if any(locator.get(name) is not None and locator[name] != identity.get(name) for name in ('case_id', 'method', 'raw_sha256')):
        raise SelectionIntegrityError('Old compact identity differs from the frozen origin')
    if (locator.get('expected_physical_options') is not None
            and context['config'].get('physical_options') != locator['expected_physical_options']):
        raise SelectionIntegrityError('Old compact physical objective multipliers differ')
    if (locator.get('expected_calibration_fingerprint') is not None
            and context['calibration']['fingerprint'] != locator['expected_calibration_fingerprint']):
        raise SelectionIntegrityError('Old compact calibration differs from frozen origin')
    if context['canonical_schema'] != list(compact.CANONICAL_KEYS) or set(prepared['canonical_arrays']) != set(compact.CANONICAL_KEYS):
        raise SelectionIntegrityError('Old canonical field schema differs')
    if not result.get('lossless_representation_verified') or result.get('status') != 'LOSSLESS_READY' or not result.get('score_allowed'):
        raise SelectionIntegrityError('Old compact result is not scoreable')
    compact.verify_sources(context['source_sha256']); compact.verify_file(context['input_file'])
    if compact.runtime_identity(p) != context['runtime'] or compact._physics_identity(p) != context['physics']:
        raise SelectionIntegrityError('The original same-runtime/device/physics contract is required')
    native_raw = compact.array(raw).reshape(tuple(context['input_reference']['shape']))
    camera = native_raw.astype(context['input_reference']['dtype'])
    if not np.array_equal(camera.astype(native_raw.dtype), native_raw):
        raise SelectionIntegrityError('Old raw dtype conversion is not lossless')
    if compact.array_sha(camera) != context['input_reference']['array_sha256']:
        raise SelectionIntegrityError('Old forward reconstruction has another raw')
    if Path(parameters['parameter_cache']['path']).resolve() != (data/'PARAMETERS.npz').resolve():
        raise SelectionIntegrityError('Old parameter reference escapes its original instance')
    if Path(parameters['prepared']['path']).resolve() != (report/'PREPARED.json').resolve():
        raise SelectionIntegrityError('Old original-byte metadata reference differs')
    if Path(result['parameter_receipt']['path']).resolve() != (report/'PARAMETERS.json').resolve():
        raise SelectionIntegrityError('Old final parameter receipt reference differs')
    compact.verify_file(result['parameter_receipt'])
    compact.verify_file(parameters['parameter_cache']); compact.verify_file(parameters['prepared'])
    values = s.arrays(parameters['parameter_cache']['path'])
    if set(values) != set(compact.PARAMETER_KEYS) or any(value.dtype != np.float64 for value in values.values()):
        raise SelectionIntegrityError('Old parameters must retain all float64 arrays')
    if {name: compact.descriptor(value) for name, value in values.items()} != prepared['parameter_arrays']:
        raise SelectionIntegrityError('Old parameter dtype/shape/byte descriptor differs')
    out = compact.forward_reconstruct(p, raw, values, prepared['original_info'], grid=context['frame_plan']['grid'])
    out['numerical_valid'] = prepared['original_numerical_valid']
    matches = {name: compact.descriptor(out[name]) == expected for name, expected in prepared['canonical_arrays'].items()}
    matches['continuum_latent'] = compact.descriptor(out['continuum_latent']) == prepared['continuum_latent']
    if set(matches) != set(compact.CANONICAL_KEYS)|{'continuum_latent'} or not all(matches.values()):
        raise SelectionIntegrityError('Old canonical forward reconstruction is not byte exact; old files left untouched')
    return out, {'all_byte_identical': True, 'per_array_byte_identical': matches, 'complete_inverse_calls': 0,
                 'old_evidence_written': False, 'parameter_cache': parameters['parameter_cache']}


def read_origin(locator, p, raw):
    if not locator: raise SelectionIntegrityError('Prior attempt has no preserved forward state; never refit it')
    if locator['kind'] == 'compact': return read_existing_compact(locator, p, raw)
    row = locator['row']; old = s.read(c2.REGISTRATION)
    values, metadata = c2.load_parameters(row['parameter_cache_receipt'], c2.identity_for(row, row['registration_sha256']), old)
    out = compact.forward_reconstruct(p, raw, values, metadata['inference_info'], grid=256)
    out['numerical_valid'] = metadata['numerical_valid']
    matches = panel.byte_matches(out, s.arrays(row['output_cache']['path']))
    if not all(matches.values()): raise SelectionIntegrityError('Old v002 canonical bytes differ; no inverse retry')
    return out, {'all_byte_identical': True, 'per_original_six_byte_identical': matches,
                 'complete_inverse_calls': 0, 'old_evidence_written': False}


def cache_for(store, p, cell, reg, digest, *, qualification=False):
    ref = next(ref for ref in reg['references'] if ref['case_id'] == cell['case_id'])
    return dev.ECompactCache(Path(store.report_root)/('qualification' if qualification else 'inference'),
        Path(store.data_root)/('qualification' if qualification else 'inference'), p=p,
        identity={'registration_sha256': digest, 'method': cell['method'], 'case_id': cell['case_id'],
                  'raw_sha256': cell['raw_sha256'], 'controller_version': VERSION},
        input_reference=ref['raw_reference'], calibration=reg['calibration_contract'],
        config={'grid': 256, 'physical_options': cell['physical_options'], 'numerical_options': CONFIG['numerical_options'],
                'execution_storage_contract': store.contract(), 'operation': 'existing_forward_only' if qualification else 'first_stage'},
        source_sha256=reg['source_sha256'], budget=store.budget())


def qualify(device='cuda', *, store=None):
    if device != 'cuda': raise SelectionIntegrityError('Namespace qualification is a separate root-owned GPU forward operation')
    panel.execution_entry()
    from .smooth_calibration import physics, contract
    import torch
    store = SelectionStore() if store is None else store; reg, digest = verify_registration(store)
    path = Path(store.report_root)/QUALIFICATION_NAME
    if path.exists(): return panel.read_sealed(store, path)
    with dev.phase_locks(store, digest, VERSION+'-qualify'):
        torch.set_num_threads(2); p = physics(device)
        if contract(p) != reg['calibration_contract']: raise SelectionIntegrityError('C1 calibration differs')
        s.frozen_bridge()
        from research_v6 import inputs
        def action(origin):
            ref = next(ref for ref in reg['references'] if ref['case_id'] == origin['case_id'])
            cell = next(cell for cell in reg['planned_cells'] if cell['case_id'] == origin['case_id'] and cell['method'] == origin['method'])
            raw = inputs.load_raw(ref, p); out, proof = read_existing_compact(origin, p, raw)
            cache = cache_for(store, p, cell, reg, digest, qualification=True)
            ticket = cache.begin(raw)
            result = cache.persist(ticket, out, raw) if ticket.get('run_new_inverse') else cache.resume(raw)
            restored = cache.load_for_score(raw) if result.get('score_allowed') else {}
            matches = {name: name in restored and compact.descriptor(restored[name]) == compact.descriptor(out[name])
                       for name in compact.CANONICAL_KEYS+('continuum_latent',)}
            return {'passed': all(matches.values()), 'original_read_only_proof': proof,
                'new_namespace_byte_matches': matches, 'compact_report_root': str(cache.report),
                'compact_data_root': str(cache.data), 'compact_result': result, 'complete_inverse_calls': 0}
        value = panel.qualification_matrix(reg['qualification_origins'], action)
        value.update(time=s.now(), registration_sha256=digest, native_GPU=True, runtime=compact.runtime_identity(p),
            source_sha256=reg['source_sha256'], format_contract=reg['format_contract'], store_contract=reg['store_contract'],
            codec_arithmetic_unchanged=True, old_evidence_written=False)
        verify_registration(store); panel.seal(store, path, value)
        return value


def verify_namespace_qualification(store, p, reg, digest, load_raw):
    """Recheck saved qualification bytes on this runtime before any ticket."""
    value = panel.read_sealed(store, Path(store.report_root)/QUALIFICATION_NAME)
    panel.require_qualification(value, reg, digest, compact.runtime_identity(p))
    for origin in reg['qualification_origins']:
        ref = next(ref for ref in reg['references'] if ref['case_id'] == origin['case_id'])
        cell = next(cell for cell in reg['planned_cells'] if cell['case_id'] == origin['case_id'] and cell['method'] == origin['method'])
        raw = load_raw(ref, p)
        original, _ = read_existing_compact(origin, p, raw)
        saved = cache_for(store, p, cell, reg, digest, qualification=True).load_for_score(raw)
        if any(compact.descriptor(saved[name]) != compact.descriptor(original[name])
               for name in compact.CANONICAL_KEYS+('continuum_latent',)):
            raise SelectionIntegrityError('Qualified namespace no longer reconstructs the original bytes')
    return value


def choose_shard(reg, *, mother=None, raw_ids=None):
    refs = reg['references']; requested = list(raw_ids or [])
    if bool(mother) == bool(requested): raise SelectionIntegrityError('Explicitly choose one mother or1-6 raw IDs; there is no automatic full run')
    if mother:
        if mother.isdigit(): mother = f'v5_development_selection_mother_{int(mother):04d}'
        requested = [ref['case_id'] for ref in refs if ref['mother_id'] == mother]
        if len(requested) != 5: raise SelectionIntegrityError('Choose one original selection mother')
    if len(requested) != len(set(requested)) or not 1 <= len(requested) <= 6:
        raise SelectionIntegrityError('A scheduling shard contains1-6 distinct registered raws')
    selected = [ref for ref in refs if ref['case_id'] in requested]
    if len(selected) != len(requested): raise SelectionIntegrityError('Shard requested an unregistered raw')
    return selected


def counted_once(cell, reg):
    from . import experiment_registry as registry
    target = next((item for item in reg['planned_cells'] if dev.semantic_key(item) == dev.semantic_key(cell)), None)
    if target is None:
        raise SelectionIntegrityError('Undeclared method/raw callback')
    if target['execution_role'] != 'NEW_INVERSE' or target['reuse_key'] in reg['reuse_catalog']:
        raise SelectionIntegrityError('A known physical attempt cannot be counted under another label')
    calls, _ = dev.call_snapshot()
    if any(not row.get('reused') and row.get('method') == cell['method'] and row.get('case_id') == cell['case_id'] for row in calls):
        raise SelectionIntegrityError('A prior same-physical-method/raw attempt exists; another inverse is forbidden')
    if sum(not row.get('reused') and row.get('phase') == PHASE for row in calls) >= 1052:
        raise SelectionIntegrityError('First-stage1052-new-call cap reached')
    if sum(not row.get('reused') and row.get('phase') in DEVELOPMENT_PHASES for row in calls) >= 3400:
        raise SelectionIntegrityError('Original semantic development3400-call cap reached')
    return registry.count_call(PHASE, cell['method'], cell['case_id'])


def evaluate_cell(cell, *, cache, raw, action, score_action, count_call, store, digest,
                  verify_sources=lambda: None, score_only=False):
    def count(ignored_phase, method, case_id): return count_call(PHASE, method, case_id)
    row = dev.evaluate_cell(cell, cache=cache, raw=raw, action=action, score_action=score_action,
        count_call=count, store=store, digest=digest, verify_sources=verify_sources, score_only=score_only)
    calls, _ = dev.call_snapshot()
    recorded = sum(not call.get('reused') and call.get('phase') == PHASE
        and call.get('method') == cell['method'] and call.get('case_id') == cell['case_id'] for call in calls)
    if recorded > 1: raise SelectionIntegrityError('Repeated first-stage method/raw calls in the true ledger')
    row['new_inverse_calls_selection_total'] = max(recorded, row.get('new_inverse_calls_panel_total', 0))
    row.pop('global_counted_prescreen_attempts_for_cell', None)
    return row


def row_key(cell): return s.canonical(list(dev.semantic_key(cell)))


def row_projection(cell, row):
    keys = ('status', 'finite_output', 'production_numerical_certified', 'score_status', 'metrics', 'numerical_evidence',
            'new_inverse_calls_selection_total', 'error_type', 'error', 'score_error_type', 'score_error',
            'controller_stop', 'durable_failure_receipt_written', 'original_finite_endpoint_not_durably_preserved',
            'compact_report_root', 'compact_data_root', 'original_fallback_available',
            'case_state_durable', 'evidence_files')
    return {**cell, **{key: copy.deepcopy(row[key]) for key in keys if key in row},
            'prior_result_reused': cell['execution_role'] == 'REUSE_PRIOR_C1V3'}


def case_evidence_files(store, cell, row, digest):
    refs = []
    report = row.get('compact_report_root')
    if report:
        report = store.write_path(report)
        for path in report.glob('*.json'):
            if path.name.endswith('.sha256.json'): continue
            value = compact._read_sealed(path)
            refs += [panel.file_ref(path), panel.file_ref(path.with_name(path.name+'.sha256.json'))]
            if path.name == 'PARAMETERS.json':
                compact.verify_file(value['parameter_cache']); refs.append(value['parameter_cache'])
            if value.get('fallback'):
                compact.verify_file(value['fallback']); refs.append(value['fallback'])
    receipts = Path(store.report_root)/'cell_receipts'/panel.cell_key(cell, digest)
    for path in receipts.glob('*.json'):
        if path.name.endswith('.sha256.json'): continue
        panel.read_sealed(store, path)
        refs += [panel.file_ref(path), panel.file_ref(path.with_name(path.name+'.sha256.json'))]
    return list({ref['path']: ref for ref in refs}.values())


def state_files(root):
    return sorted(path for path in Path(root).glob('STATE_*.json')
                  if re.fullmatch(r'STATE_[0-9]{6}_[0-9a-f]{64}\.json', path.name))


def persist_row(store, cell, row, digest):
    body = row_projection(cell, {**row, 'case_state_durable': True,
                                'evidence_files': case_evidence_files(store, cell, row, digest)})
    content_sha = s.canonical(s.plain(body))
    root = Path(store.report_root)/'case_states'/row_key(cell)
    files = state_files(root)
    for path in files:
        if path.name.endswith('_'+content_sha+'.json') and path.with_name(path.name+'.sha256.json').exists():
            panel.read_sealed(store, path); return panel.file_ref(path)
    sequence = max((int(path.name.split('_')[1]) for path in files), default=0)+1
    path = root/f'STATE_{sequence:06d}_{content_sha}.json'
    panel.seal(store, path, {'time': s.now(), 'registration_sha256': digest, 'sequence': sequence,
                            'content_sha256': content_sha, 'row': body})
    return panel.file_ref(path)


def collect_rows(reg, store, digest):
    rows = []
    calls, _ = dev.call_snapshot()
    for cell in reg['planned_cells']:
        root = Path(store.report_root)/'case_states'/row_key(cell)
        files = state_files(root)
        recorded = sum(not call.get('reused') and call.get('phase') == PHASE
            and call.get('method') == cell['method'] and call.get('case_id') == cell['case_id'] for call in calls)
        if recorded > 1: raise SelectionIntegrityError('Repeated true-ledger method/raw attempts')
        if files:
            path = max(files, key=lambda path: int(path.name.split('_')[1]))
            try:
                value = panel.read_sealed(store, path)
                if (value['registration_sha256'] != digest or dev.semantic_key(value['row']) != dev.semantic_key(cell)
                        or s.canonical(s.plain(value['row'])) != value['content_sha256']):
                    raise SelectionIntegrityError('Case state identity/content differs')
                for ref in value['row'].get('evidence_files', []): panel.verify_ref(ref)
                rows.append({**value['row'], 'state_receipt': panel.file_ref(path),
                    'new_inverse_calls_selection_total': max(recorded, value['row'].get('new_inverse_calls_selection_total', 0))})
            except Exception as error:
                rows.append({**cell, 'status': 'CASE_STATE_RECEIPT_INTERRUPTED_OR_CORRUPT', 'finite_output': False,
                    'production_numerical_certified': False, 'metrics': None, 'score_status': 'NOT_SCOREABLE',
                    'controller_stop': True, 'case_state_durable': False,
                    'error_type': type(error).__name__, 'error': str(error), 'new_inverse_calls_selection_total': recorded})
        elif cell['execution_role'] == 'REUSE_PRIOR_C1V3':
            row = dev.reuse_row(cell, reg['reuse_catalog'][cell['reuse_key']], digest)
            row['new_inverse_calls_selection_total'] = 0
            rows.append(row_projection(cell, row))
        elif recorded:
            rows.append({**cell, 'status': 'COUNTED_ATTEMPT_WITHOUT_DURABLE_CASE_STATE', 'finite_output': False,
                'production_numerical_certified': False, 'metrics': None, 'score_status': 'NOT_SCOREABLE',
                'controller_stop': True, 'case_state_durable': False, 'new_inverse_calls_selection_total': recorded,
                'inference_rerun_permitted': False})
    return rows


def merge_transient_rows(reg, durable, transient):
    merged = {dev.semantic_key(row): row for row in durable}
    calls, _ = dev.call_snapshot()
    for row in transient:
        recorded = sum(not call.get('reused') and call.get('phase') == PHASE
            and call.get('method') == row['method'] and call.get('case_id') == row['case_id'] for call in calls)
        merged[dev.semantic_key(row)] = {**row, 'new_inverse_calls_selection_total': max(recorded, row.get('new_inverse_calls_selection_total', 0))}
    return list(merged.values())


def terminal_outcome(row):
    if row.get('controller_stop') or row.get('case_state_durable') is False: return False
    if row.get('status') == 'INFERENCE_FAILED_NO_RERUN': return True
    return bool(row.get('finite_output') and row.get('score_status') == 'SCORED'
                and row.get('status') in ('FINITE_NUMERICALLY_CERTIFIED', 'FINITE_NUMERICALLY_UNCERTIFIED'))


def summary(reg, rows):
    supplied = {dev.semantic_key(row): row for row in rows}
    if len(supplied) != len(rows) or set(supplied)-{dev.semantic_key(cell) for cell in reg['planned_cells']}:
        raise SelectionIntegrityError('Duplicate/unplanned first-stage state')
    complete = [supplied.get(dev.semantic_key(cell), {**cell, 'status': 'NOT_EXECUTED', 'finite_output': False,
        'production_numerical_certified': False, 'metrics': None, 'score_status': 'NOT_EXECUTED',
        'new_inverse_calls_selection_total': 0}) for cell in reg['planned_cells']]
    observed = [row for row in complete if row.get('finite_output') or row.get('inference_attempted')
        or row.get('new_inverse_calls_selection_total', 0) > 0
        or (row.get('execution_role') == 'REUSE_PRIOR_C1V3' and not row['status'].startswith('NOT_EXECUTED'))]
    from .metrics import aggregate
    candidates = []
    for candidate in dev.CANDIDATES:
        group = [row for row in complete if row['candidate_id'] == candidate['id']]
        passed = [row for row in group if terminal_outcome(row) and row.get('finite_output') and row.get('production_numerical_certified')
                  and row.get('numerical_evidence', {}).get('passed')]
        all120 = len(group) == 120 and len(passed) == 120
        pooled = {initializer: aggregate([row for row in group if row['initializer'] == initializer]) for initializer in dev.INITIALIZERS}
        candidates.append({**candidate, 'planned_cells': 120, 'planned_mothers': 12, 'planned_raws': 60,
            'numerically_qualified_cells': len(passed), 'all120_numerical_certification_passed': all120,
            'not_executed_cells': sum(row['status'].startswith('NOT_EXECUTED') for row in group),
            'incomplete_or_integrity_stopped_cells': sum(not terminal_outcome(row) for row in group),
            'initializer_aggregates': pooled, 'scientific_scores_complete': all(value['all_cases_finite'] for value in pooled.values()),
            'eligible_for_future_root_selection_review': all120 and all(value['all_cases_finite'] for value in pooled.values()),
            'ordinary_failure_did_not_prune_remaining_cells': True, 'budget_used_for_eligibility': False})
    return {'planned_mothers': 12, 'planned_raws': 60, 'planned_cells': 1080, 'state_rows_present': len(rows),
        'cells': complete, 'candidate_first_stage': candidates,
        'reused_prior_cells': sum(cell['execution_role'] == 'REUSE_PRIOR_C1V3' for cell in reg['planned_cells']),
        'finite_output_cells': sum(bool(row.get('finite_output')) for row in complete),
        'numerically_certified_cells': sum(bool(row.get('production_numerical_certified')) for row in complete),
        'not_executed_cells': sum(row['status'].startswith('NOT_EXECUTED') for row in complete),
        'scoring_failed_cells': sum(row.get('score_status') == 'SCORING_FAILED' for row in complete),
        'terminal_outcome_cells': sum(terminal_outcome(row) for row in complete),
        'incomplete_or_integrity_stopped_cells': sum(not terminal_outcome(row) for row in complete),
        'new_inverse_calls_selection_total': sum(row.get('new_inverse_calls_selection_total', 0) for row in complete),
        'all_denominators_preserved': True, 'method_or_initializer_selected': False, 'top_two_selected': False,
        'confirmation24_started': False, 'final_payloads_opened': False,
        'independent_sample_units': {'planned_parent_mothers': 12, 'planned_distinct_raws': 60,
            'observed_parent_mothers': len({row['mother_id'] for row in observed}),
            'observed_distinct_raws': len({row['case_id'] for row in observed}),
            'observed_definition': 'unique mothers/raws with a finite endpoint, attempted callback, charged attempt, or frozen prior attempted/reused state; future unexecuted plan slots excluded',
            'scheduled_method_routes_per_raw': 18, 'method_routes_are_independent_sample_units': False},
        'resource_limited_incompletion_is_not_scientific_rejection': True}


def write_summary(store, reg, digest):
    value = {'time': s.now(), 'registration_sha256': digest, 'status': 'FIRST_STAGE_PARTIAL',
             'summary': summary(reg, collect_rows(reg, store, digest)), 'storage_ledger': store.budget().ledger()}
    current = Path(store.report_root)/'SUMMARY_CURRENT.json'
    store.write(current, s.plain(value), replace=current.exists())
    # Final only when every required slot has an explicit attempted/reused
    # outcome; interrupted/cache failures remain visible and never qualify.
    if value['summary']['terminal_outcome_cells'] == 1080:
        value['status'] = 'FIRST_STAGE_MATRIX_REPORTED_NO_SELECTION'
        final = Path(store.report_root)/RESULT_NAME
        if final.exists(): return panel.read_sealed(store, final)
        panel.seal(store, final, value)
    return value


def run_shard(device='cuda', *, mother=None, raw_ids=None, score_only=False, store=None):
    if device != 'cuda': raise SelectionIntegrityError('Formal shards are root-owned native GPU only')
    panel.execution_entry()
    from .smooth_calibration import physics, contract
    from .numerical_strategy import repaired_infer
    from .development import cached_v3
    from . import metrics
    import torch
    store = SelectionStore() if store is None else store; reg, digest = verify_registration(store)
    refs = choose_shard(reg, mother=mother, raw_ids=raw_ids)
    shard_id = s.canonical({'registration': digest, 'raws': [ref['case_id'] for ref in refs]})
    touched, counted, transient = [], 0, {}
    with dev.phase_locks(store, digest, VERSION+('-score-only' if score_only else '-shard')):
        try:
            torch.set_num_threads(2); p = physics(device)
            if contract(p) != reg['calibration_contract']: raise SelectionIntegrityError('C1 physical contract differs')
            s.frozen_bridge()
            from research_v6 import inputs
            verify_namespace_qualification(store, p, reg, digest, inputs.load_raw)
            for ref in refs:
                raw = inputs.load_raw(ref, p)
                q.verify_initializer_raw(ref, raw, reg['v3_initializer_provenance'][ref['case_id']])
                cells = [cell for cell in reg['planned_cells'] if cell['case_id'] == ref['case_id']]
                for cell in cells:
                    transient[row_key(cell)] = {**cell, 'status': 'CONTROLLER_INTERRUPTED_BEFORE_DURABLE_CASE_STATE',
                        'finite_output': False, 'production_numerical_certified': False, 'score_status': 'NOT_SCOREABLE',
                        'metrics': None, 'controller_stop': True, 'case_state_durable': False,
                        'new_inverse_calls_selection_total': 0}
                    def verify():
                        verify_registration(store, verify_inputs=False)
                        for receipt in (reg['v3_initializer_provenance'][ref['case_id']]['state_cache'],): panel.verify_ref(receipt)
                    def score(out):
                        tensors = {key: torch.as_tensor(value, device=p.M.device) if isinstance(value, np.ndarray) else value for key, value in out.items()}
                        truth = inputs.load_diagnostic_truth(ref, p)
                        return metrics.score(tensors, truth, raw, p, detector=None, method=cell['method'], application_domain=VERSION+'-12-mother-first-stage')
                    print('BEGIN first-stage '+cell['case_id']+' '+cell['method'], flush=True)
                    if cell['execution_role'] == 'REUSE_PRIOR_C1V3':
                        item = reg['reuse_catalog'][cell['reuse_key']]
                        row = dev.reuse_row(cell, item, digest)
                        if row['finite_output']:
                            _, proof = read_origin(item.get('locator'), p, raw)
                            row['old_read_only_byte_verification'] = proof
                        if row.get('score_status') != 'SCORED' and row['finite_output']:
                            # New namespace receives only a forward reconstruction
                            # of the old endpoint, never a new inference.
                            old_out, _ = read_origin(item.get('locator'), p, raw)
                            cache = cache_for(store, p, cell, reg, digest)
                            ticket = cache.begin(raw)
                            result = cache.persist(ticket, old_out, raw) if ticket.get('run_new_inverse') else cache.resume(raw)
                            if not result.get('score_allowed'): raise SelectionIntegrityError('Reused score recovery failed closed')
                            row = dev.evaluate_reused_cell(cell, item, cache=cache, raw=raw, score_action=score,
                                store=store, digest=digest, verify_sources=verify)
                        row['new_inverse_calls_selection_total'] = 0
                    else:
                        cache = cache_for(store, p, cell, reg, digest)
                        def action():
                            state = cached_v3(ref, p)[0] if cell['initializer'] == 'v3' else None
                            out = repaired_infer(raw, p, initializer=cell['initializer'], v3_state=state,
                                physical_options=copy.deepcopy(cell['physical_options']), numerical_options=copy.deepcopy(CONFIG['numerical_options']))
                            metadata = {'calibration_contract': reg['calibration_contract'], 'objective_fingerprint': reg['objective_fingerprints'][cell['candidate_id']],
                                        'config_fingerprint': reg['config_fingerprint']}
                            return c2.wrap_output(out, cell, metadata)
                        row = evaluate_cell(cell, cache=cache, raw=raw, action=action, score_action=score,
                            count_call=lambda phase, method, case_id: counted_once(cell, reg), store=store,
                            digest=digest, verify_sources=verify, score_only=score_only)
                    counted += row.get('new_inverse_calls_this_run', 0)
                    transient[row_key(cell)] = row_projection(cell, {**row, 'case_state_durable': False})
                    touched.append(persist_row(store, cell, row, digest))
                    transient.pop(row_key(cell), None)
                    print('END first-stage '+cell['case_id']+' '+cell['method']+' '+row['status'], flush=True)
                    if row.get('controller_stop'): raise SelectionIntegrityError('Integrity/storage stop; every future denominator cell remains planned')
            report = write_summary(store, reg, digest)
            value = {'time': s.now(), 'status': 'FIRST_STAGE_SHARD_COMPLETE', 'registration_sha256': digest,
                'shard_id': shard_id, 'raw_ids': [ref['case_id'] for ref in refs], 'planned_shard_cells': len(refs)*18,
                'case_state_receipts': touched, 'new_inverse_calls_this_shard': counted,
                'global_summary_counts': {key: report['summary'][key] for key in ('planned_cells', 'finite_output_cells', 'not_executed_cells', 'new_inverse_calls_selection_total')},
                'score_only': score_only, 'top_two_selected': False, 'confirmation24_started': False}
            panel.seal(store, Path(store.report_root)/'shards'/shard_id/('RUN_'+uuid.uuid4().hex+'.json'), value)
            return value
        except Exception as error:
            value = {'time': s.now(), 'status': 'FIRST_STAGE_SHARD_CONTROLLER_STOP', 'registration_sha256': digest,
                'shard_id': shard_id, 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                'summary': summary(reg, merge_transient_rows(reg, collect_rows(reg, store, digest), transient.values())),
                'case_state_receipts': touched,
                'new_inverse_calls_this_shard': counted, 'inference_rerun_permitted': False,
                'resource_stop_is_scientific_exclusion': False}
            return panel.run_receipt(store, Path(store.report_root)/'shards'/shard_id/('STOP_'+uuid.uuid4().hex+'.json'),
                                     value, failure_status='FIRST_STAGE_STOP_RECEIPT_FAILED')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    item = sub.add_parser('register'); item.add_argument('--contract', type=Path, action='append', required=True)
    item = sub.add_parser('qualify'); item.add_argument('--device', choices=('cuda',), default='cuda')
    item = sub.add_parser('run-shard'); item.add_argument('--device', choices=('cuda',), default='cuda')
    group = item.add_mutually_exclusive_group(required=True); group.add_argument('--mother'); group.add_argument('--raw', action='append')
    item.add_argument('--score-only', action='store_true')
    sub.add_parser('summarize')
    args = parser.parse_args(); store = SelectionStore()
    if args.command == 'plan':
        plan = prepare_plan(); value = {'status': 'PREPARED_NOT_REGISTERED', 'config': CONFIG,
            'planned_cells': len(plan['planned_cells']), 'reused_cells': len(plan['reuse_catalog']),
            'maximum_new_calls': sum(cell['execution_role'] == 'NEW_INVERSE' for cell in plan['planned_cells']),
            'new_complete_inverse_calls': 0}
    elif args.command == 'register':
        _, digest = register(args.contract, store=store); value = {'status': 'SELECTION_REGISTERED', 'registration_sha256': digest, 'new_complete_inverse_calls': 0}
    elif args.command == 'qualify': value = qualify(args.device, store=store)
    elif args.command == 'run-shard': value = run_shard(args.device, mother=args.mother, raw_ids=args.raw, score_only=args.score_only, store=store)
    else:
        reg, digest = verify_registration(store)
        with dev.phase_locks(store, digest, VERSION+'-summary'): value = write_summary(store, reg, digest)
    print(json.dumps(s.plain(value), ensure_ascii=True, indent=2), flush=True)
    raise SystemExit(2 if value.get('all_passed') is False or any(word in value.get('status', '') for word in ('STOP', 'FAILED')) else 0)


if __name__ == '__main__': main()
