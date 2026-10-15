"""Registered, resumable quick diagnostics on six unchanged development raws.

``register --contract PATH ...`` freezes the repair contracts and this route's
source/input/configuration dependencies. ``run --tier smoke`` executes only the
first two raws (six new inversions); ``--tier panel`` is an explicit eighteen
inversion request. Neither tier selects a method or opens final data.
"""
from . import storage as s
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid
import numpy as np


VERSION = 'repair-quick-v001'
REPORT_DIR = s.REPORTS/'repair_001/quick_v001'
DATA_DIR = s.DATA/'repair_001/quick_v001'
INVENTORY = s.ROOT/'research_v6/input_inventory.json'
V3_JOURNAL = s.LEGACY/'reports/research_v6/repair_001/source_inference_receipts.jsonl'
REGISTRATION = REPORT_DIR/'QUICK_REGISTRATION.json'
REGISTRATION_SHA = REPORT_DIR/'QUICK_REGISTRATION_SHA256.json'
CONTROLLER_CONTRACT = REPORT_DIR/'QUICK_CONTROLLER_CONTRACTS.json'
PHYSICAL_OPTIONS = {'grid': 256, 'continuum_multiplier': 1., 'template_multiplier': 1.}
NUMERICAL_OPTIONS = {'cost_limit_s': 180.}
PANEL = ((0, 'positive_normal'), (13, 'negative_normal'), (0, 'blank'),
         (5, 'positive_weak'), (14, 'negative_weak'), (27, 'positive_normal'))
ROUTES = (
    {'method': 'joint_c1_s1_repaired_v2__v3', 'family': 'joint', 'initializer': 'v3'},
    {'method': 'joint_c1_s1_repaired_v2__smooth_demod', 'family': 'joint', 'initializer': 'smooth_demod'},
    {'method': 'v3_objective_refined_v2__same_raw_cached_v3', 'family': 'v3_refined', 'initializer': 'cached_v3'},
)
CONFIG = {
    'version': VERSION, 'physical_options': PHYSICAL_OPTIONS,
    'numerical_options': NUMERICAL_OPTIONS, 'v3_options': {'cost_limit_s': 180.},
    'routes': list(ROUTES), 'tiers': {'smoke': {'raws': 2, 'new_inverse_calls': 6, 'inverse_wall_budget_s': 1080},
                                    'panel': {'raws': 6, 'new_inverse_calls': 18, 'inverse_wall_budget_s': 3240}},
    'smoke_output_budget_bytes': 32*2**20, 'panel_output_budget_bytes': 96*2**20,
    'free_space_floor_bytes': 2*2**30, 'global_new_inverse_call_cap': 12000,
    'cache_dtype': 'float64', 'detector': None, 'method_selection': False,
    'final_data_access': False, 'native_shape': [256, 256],
    'output_names': {'T': 'temperature_color_K', 'u': 'template_postblur_e_ref',
                     'C': 'continuum_postblur_bands', 'predicted_raw_e': 'predicted_raw_e'},
    'scope': 'small development numerical/scientific diagnostic; not final selection or acceptance',
}
# Deliberately scoped to this quick route and its imported implementation. No
# all-project freeze and no dependency on unrelated mutable paper/evaluation files.
SOURCE_FILES = (
    'research_closeout/quick_validation.py', 'research_closeout/tests/quick_validation.py',
    'research_closeout/storage.py', 'research_closeout/development.py',
    'research_closeout/measurement.py', 'research_closeout/experiment_registry.py',
    'research_closeout/numerical_strategy.py', 'research_closeout/joint_estimator.py',
    'research_closeout/linear_profile.py', 'research_closeout/observable_basis.py',
    'research_closeout/reliable_v3.py', 'research_closeout/baselines.py',
    'research_closeout/metrics.py', 'research_closeout/uncertainty.py',
    'research_closeout/independent_renderer.py', 'research_closeout/intervals.py',
    'research_v6/inputs.py', 'research_v6/common.py', 'research_v6/continuum_sources.py', 'research_v5/metrics.py',
    'training_v4/model.py', 'repair_v3/solver.py', 'physics.py', 'runtime.py', 'solver.py', 'config.py',
)


class QuickIntegrityError(RuntimeError):
    pass


def _inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path != parent and parent not in path.parents:
        raise QuickIntegrityError('Path escapes approved directory: '+str(path))
    return path


def select_panel(inventory):
    """Metadata-only deterministic picker; missing/ambiguous cells are errors."""
    rows = inventory['rows'] if isinstance(inventory, dict) else list(inventory)
    selected = []
    for number, condition in PANEL:
        mother = f'v5_development_selection_mother_{number:04d}'
        matches = [r for r in rows if r.get('cohort') == 'development'
                   and r.get('input_kind') == 'noise' and r.get('mother_id') == mother
                   and r.get('condition') == condition and r.get('noise_index') == 0]
        if len(matches) != 1:
            raise QuickIntegrityError(f'Expected one unchanged development/noise cell: {mother} {condition}; got {len(matches)}')
        ref = copy.deepcopy(matches[0])
        if ref.get('fixture_only') or ref['raw_reference'].get('shape') != [256, 256]:
            raise QuickIntegrityError('Quick inputs must be formal native256 development raws')
        if ref.get('partition_role') != 'development_selection':
            raise QuickIntegrityError('Unexpected development role')
        temperature_field = 'temperature_geometry' if ref.get('temperature_geometry') is not None else 'geometry'
        if not ref.get(temperature_field) or not ref.get('template_geometry'):
            raise QuickIntegrityError('Actual geometry metadata is required')
        ref['quick_geometry'] = {
            'temperature_geometry': ref[temperature_field], 'template_geometry': ref['template_geometry'],
            'temperature_geometry_source': 'research_v6/input_inventory.json:'+temperature_field,
            'template_geometry_source': 'research_v6/input_inventory.json:template_geometry',
            'temperature_geometry_fallback_declared': temperature_field == 'geometry',
        }
        selected.append(ref)
    if len({r['case_id'] for r in selected}) != 6:
        raise QuickIntegrityError('Duplicate quick case identifiers')
    return selected


def planned_cells(refs, tier):
    if tier not in CONFIG['tiers']:
        raise ValueError('Unknown quick tier: '+str(tier))
    count = CONFIG['tiers'][tier]['raws']
    if len(refs) != 6:
        raise QuickIntegrityError('Registration must retain the complete fixed six-raw panel')
    return [{'case_id': ref['case_id'], 'mother_id': ref['mother_id'], 'condition': ref['condition'],
             'input_kind': ref['input_kind'], 'raw_sha256': ref['raw_reference']['sha256'],
             'quick_geometry': ref['quick_geometry'], **route}
            for ref in refs[:count] for route in ROUTES]


def scoped_hashes():
    return {name: s.sha(s.ROOT/name) for name in SOURCE_FILES}


def verify_hashes(hashes, root=None):
    root = s.ROOT if root is None else Path(root)
    for name, digest in hashes.items():
        path = _inside(root/name, root)
        if not path.is_file() or s.sha(path) != digest:
            raise QuickIntegrityError('Registered source changed or missing: '+str(path))


def _file_ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': s.sha(path)}


def _verify_file_ref(receipt):
    path = Path(receipt['path'])
    if not path.is_file() or s.sha(path) != receipt['sha256']:
        raise QuickIntegrityError('Registered file changed or missing: '+str(path))


def _v3_provenances(refs):
    by_case = {}
    requested = {r['case_id'] for r in refs}
    journal_sha = s.sha(V3_JOURNAL)
    for line, row in enumerate(s.rows(V3_JOURNAL), 1):
        if row.get('source') == 'C_v3' and row.get('case_id') in requested:
            if row['case_id'] in by_case:
                raise QuickIntegrityError('Ambiguous frozen v3 receipt: '+row['case_id'])
            by_case[row['case_id']] = (line, row)
    result = {}
    for ref in refs:
        if ref['case_id'] not in by_case:
            raise QuickIntegrityError('Missing frozen same-raw v3 initializer: '+ref['case_id'])
        line, row = by_case[ref['case_id']]
        if row['raw_sha256'] != ref['raw_reference']['sha256']:
            raise QuickIntegrityError('Frozen v3 initializer raw SHA mismatch')
        metadata = row.get('source_metadata', {})
        if metadata.get('truth_or_identity_used') is not False:
            raise QuickIntegrityError('Frozen v3 initializer lacks raw-only provenance')
        if metadata.get('dtype') not in ('torch.float32', 'torch.float64'):
            raise QuickIntegrityError('Frozen v3 source input dtype is not declared')
        if not isinstance(metadata.get('raw_array_sha256'), str) or len(metadata['raw_array_sha256']) != 64:
            raise QuickIntegrityError('Frozen v3 source input array hash is missing')
        cache = dict(row['state_cache'])
        path = _inside(s.LEGACY/cache['path'], s.LEGACY/'outputs/research_v6')
        _verify_file_ref({'path': str(path), 'sha256': cache['sha256']})
        result[ref['case_id']] = {
            'kind': 'same_raw_frozen_v3_state_beta', 'new_complete_inverse_calls': 0,
            'raw_file_sha256': row['raw_sha256'], 'raw_array_sha256': ref['raw_reference']['array_sha256'],
            'original_raw_dtype': ref['raw_reference']['dtype'],
            'inference_input_array_sha256': metadata['raw_array_sha256'], 'source_input_dtype': metadata['dtype'],
            'array_hash_scope': 'original camera dtype and source floating dtype are recorded separately',
            'truth_or_identity_used': False, 'state_cache': {**cache, 'path': str(path)},
            'receipt_journal': {'path': str(V3_JOURNAL), 'sha256': journal_sha, 'line': line},
            'source_metadata': metadata,
        }
    return result


def verify_initializer_raw(ref, raw, provenance):
    """Verify original and source-floating array hashes, with lossless casting.

    Legacy source metadata hashes its float32 inference tensor; inventory
    hashes the original uint16 camera array. Those are distinct byte hashes.
    """
    observed = _array(raw)
    shape = tuple(ref['raw_reference']['shape'])
    if observed.shape == (1, *shape):
        observed = observed[0]
    if observed.shape != shape or not np.isfinite(observed).all():
        raise QuickIntegrityError('Invalid raw shape for initializer provenance')
    restored = observed.astype(np.dtype(ref['raw_reference']['dtype']))
    if not np.array_equal(restored.astype(observed.dtype), observed):
        raise QuickIntegrityError('Raw camera dtype roundtrip is not lossless')
    if _array_sha(restored) != ref['raw_reference']['array_sha256']:
        raise QuickIntegrityError('Original camera array hash differs from inventory')
    source_dtype = {'torch.float32': np.float32, 'torch.float64': np.float64}.get(provenance['source_input_dtype'])
    if source_dtype is None:
        raise QuickIntegrityError('Unknown cached source tensor dtype')
    source_array = observed.astype(source_dtype)
    if not np.array_equal(source_array.astype(observed.dtype), observed):
        raise QuickIntegrityError('Raw source dtype roundtrip is not lossless')
    if _array_sha(source_array) != provenance['inference_input_array_sha256']:
        raise QuickIntegrityError('Cached source floating raw array hash mismatch')
    return {'original_dtype_roundtrip_verified': True, 'source_dtype_roundtrip_verified': True,
            'source_input_array_hash_verified': True}


def _contract_receipts(paths):
    paths = [Path(p).resolve() for p in paths]
    # Rechecks are separate immutable receipts. After a controller correction,
    # automatically choose a passed recheck for the exact current source bytes.
    controller_names = ('research_closeout/quick_validation.py', 'research_closeout/tests/quick_validation.py')
    controller_hashes = {name: s.sha(s.ROOT/name) for name in controller_names}
    controllers = []
    for candidate in REPORT_DIR.glob('QUICK_CONTROLLER_CONTRACTS*.json'):
        value = s.read(candidate)
        if value.get('all_passed') is True and value.get('source_sha256') == controller_hashes:
            controllers.append(candidate.resolve())
    if not controllers:
        raise QuickIntegrityError('Run the CPU controller contracts for the current source before registering')
    if not any(path in controllers for path in paths):
        paths.append(max(controllers, key=lambda path: path.stat().st_mtime_ns))
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate repair contract arguments')
    if len(paths) < 3:
        raise ValueError('Register numerical and validation repair contracts plus the controller contract')
    result = []
    for path in paths:
        _inside(path, s.REPORTS/'repair_001')
        value = s.read(path)
        if value.get('all_passed') is not True:
            raise QuickIntegrityError('Repair contracts have not all passed: '+str(path))
        if value.get('source_sha256'):
            verify_hashes(value['source_sha256'])
        result.append(_file_ref(path))
    return result


def register(contract_paths):
    """Read permitted old raws/cache states on CPU, without any inversion."""
    if REGISTRATION.exists() or REGISTRATION_SHA.exists():
        raise FileExistsError('Quick v001 registration already exists; verify/reuse it without overwriting')
    if (s.REPORTS/'PHASE_ACTIVE.json').exists() or (REPORT_DIR/'PHASE_ACTIVE.json').exists():
        raise FileExistsError('A phase is active; finish source edits and phase execution before registration')
    contracts = _contract_receipts(contract_paths)
    refs = select_panel(s.read(INVENTORY))
    provenance = _v3_provenances(refs)
    # Both APIs validate immutable files; neither starts an inverse. No diagnostic
    # truth mapping is loaded until after the relevant inference has completed.
    s.frozen_bridge()
    from . import measurement as m
    from .development import cached_v3
    from research_v6 import inputs
    import torch
    torch.set_num_threads(2)
    p = m.physics('cpu')
    for ref in refs:
        raw = inputs.load_raw(ref, p)
        if tuple(raw.shape) != (1, 256, 256):
            raise QuickIntegrityError('Actual raw shape is not native256')
        provenance[ref['case_id']].update(verify_initializer_raw(ref, raw, provenance[ref['case_id']]))
        state, beta = cached_v3(ref, p)
        if tuple(state.shape) != (1, 5, 256, 256) or tuple(beta.shape) != (1, 3):
            raise QuickIntegrityError('Unexpected frozen v3 state/cache shape')
        if not bool(torch.isfinite(state).all() and torch.isfinite(beta).all()):
            raise QuickIntegrityError('Nonfinite frozen v3 initial state')
    if s.read(s.ROOT/'research_closeout/execution_spec.yaml')['new_complete_inverse_call_cap'] != 12000:
        raise QuickIntegrityError('Quick execution requires the registered global 12000-call cap')
    external_files = [_file_ref(INVENTORY), _file_ref(V3_JOURNAL),
                      _file_ref(s.ROOT/'reports/training_v4/run_001/backbone_selection.json'),
                      _file_ref(s.ROOT/'research_closeout/execution_spec.yaml')]
    external_files.extend(_file_ref(_inside(s.LEGACY/r['raw_reference']['path'],
                                         s.LEGACY/'outputs/research_v5/run_001/development/raw')) for r in refs)
    external_files.extend(provenance[r['case_id']]['state_cache'] for r in refs)
    value = {'schema': VERSION+'-registration', 'time': s.now(), 'source_sha256': scoped_hashes(),
             'contracts': contracts, 'config': copy.deepcopy(CONFIG), 'external_files': external_files,
             'references': refs, 'v3_initializer_provenance': provenance,
             'report_directory': str(REPORT_DIR), 'data_directory': str(DATA_DIR),
             'raw_payloads_verified': 6, 'diagnostic_truth_loaded': False,
             'new_complete_inverse_calls': 0, 'final_payloads_opened': False,
             'script_sha256': s.sha(Path(__file__)), 'not_final_method_freeze': True}
    s.write(REGISTRATION, value)
    s.write(REGISTRATION_SHA, {'path': str(REGISTRATION), 'sha256': s.sha(REGISTRATION),
                              'purpose': 'SHA256 of the exact saved registration bytes'})
    return value


def verify_registration():
    sidecar = s.read(REGISTRATION_SHA)
    if Path(sidecar['path']).resolve() != REGISTRATION.resolve():
        raise QuickIntegrityError('Registration SHA sidecar names another file')
    _verify_file_ref(sidecar)
    value = s.read(REGISTRATION)
    if value['schema'] != VERSION+'-registration' or value['config'] != CONFIG:
        raise QuickIntegrityError('Quick configuration differs from registration')
    verify_hashes(value['source_sha256'])
    for receipt in value['contracts']+value['external_files']:
        _verify_file_ref(receipt)
    if select_panel(s.read(INVENTORY)) != value['references']:
        raise QuickIntegrityError('Registered quick input references changed')
    return value, sidecar['sha256']


@contextmanager
def phase_locks(paths, *, registration_sha256):
    """Never break a preexisting lock, including a stale lock from a dead PID."""
    owner = {'pid': os.getpid(), 'owner_token': uuid.uuid4().hex,
             'phase': VERSION, 'registration_sha256': registration_sha256, 'time': s.now()}
    acquired = []
    try:
        for path in paths:
            path = s.write_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            acquired.append(path)
            with os.fdopen(fd, 'w', encoding='utf8') as handle:
                json.dump(owner, handle, allow_nan=False)
                handle.flush()
        yield owner
    finally:
        for path in reversed(acquired):
            try:
                current = s.read(path)
                if current.get('pid') == owner['pid'] and current.get('owner_token') == owner['owner_token']:
                    path.unlink()
            except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
                # An unreadable or replaced lock is not ours to remove.
                pass


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode()+json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_arrays(out, *, shape=(256, 256)):
    values = {name: _array(out[key]) for name, key in CONFIG['output_names'].items()}
    for name, value in values.items():
        if value.dtype != np.float64:
            raise QuickIntegrityError('Preserve native float64 output without conversion: '+name)
        if value.shape[-2:] != tuple(shape) or (name != 'C' and value.shape not in (shape, (1, *shape))):
            raise QuickIntegrityError('Invalid canonical output shape: '+name)
        if name == 'C' and (value.ndim != 4 or value.shape[0] != 1):
            raise QuickIntegrityError('Continuum cache requires B=1,band,H,W')
    values['valid'] = _array(out.get('valid', np.zeros(shape, dtype=bool))).astype(bool, copy=False)
    values['low_signal'] = _array(out.get('low_signal', np.zeros(shape, dtype=bool))).astype(bool, copy=False)
    return values


def _cache_key(cell, registration_sha256):
    return s.canonical({'registration_sha256': registration_sha256,
                        'method': cell['method'], 'raw_sha256': cell['raw_sha256']})


def _check_identity(receipt, identity):
    for key, value in identity.items():
        if receipt.get(key) != value:
            raise QuickIntegrityError('Cached inference identity mismatch: '+key)


def _save_output(path, out, identity, *, compact_info, output_budget_bytes, data_dir, shape):
    arrays = canonical_arrays(out, shape=shape)
    finite = all(bool(np.isfinite(arrays[name]).all()) for name in CONFIG['output_names'])
    nonnegative = bool(np.isfinite(arrays['predicted_raw_e']).all() and np.min(arrays['predicted_raw_e']) >= 0)
    try:
        info = compact_info(out)
    except Exception as error:
        # Saving the endpoint takes precedence over formatting its history.
        info = s.plain(out.get('info', {}))
        info['quick_info_compaction_error'] = {'error_type': type(error).__name__, 'error': str(error)}
    metadata = {**identity, 'info': info, 'finite_output': finite,
                'production_numerical_certified': bool(out.get('numerical_valid', False)) and finite and nonnegative,
                'predicted_measurement_nonnegative': nonnegative,
                'calibration_status': out.get('calibration_status', 'calibration_assumed_not_verified'),
                'array_sha256': {key: _array_sha(value) for key, value in arrays.items()},
                'array_shapes': {key: list(value.shape) for key, value in arrays.items()},
                'canonical_dtype': 'float64', 'truth_loaded_before_persisting': False}
    # The sidecar is saved before compression, so a later metrics/summary error
    # cannot erase the terminal inverse state. A completed NPZ is recoverable if
    # the process exits before writing the final inference receipt.
    sidecar = path.with_suffix('.output.json')
    s.write(sidecar, s.plain(metadata))
    temporary = s.write_path(path.with_suffix('.npz.pending'))
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or temporary.exists():
        raise FileExistsError('Never overwrite quick inverse outputs: '+str(path))
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    # Native canonical fields are small enough; compression never changes dtype.
    total = sum(p.stat().st_size for p in Path(data_dir).rglob('*.npz'))+temporary.stat().st_size
    temporary.replace(path)
    receipt = {**metadata, 'output_cache': _file_ref(path), 'output_bytes': path.stat().st_size,
               'output_metadata': _file_ref(sidecar), 'total_output_bytes': total}
    if total > output_budget_bytes:
        receipt['output_budget_exceeded'] = True
    return receipt


def _load_output(receipt, *, identity, shape=(256, 256)):
    _check_identity(receipt, identity)
    _verify_file_ref(receipt['output_cache'])
    _verify_file_ref(receipt['output_metadata'])
    metadata = s.read(receipt['output_metadata']['path'])
    for name, value in metadata.items():
        if s.plain(receipt.get(name)) != value:
            raise QuickIntegrityError('Saved output metadata differs from inference receipt: '+name)
    arrays = s.arrays(receipt['output_cache']['path'])
    for name, digest in receipt['array_sha256'].items():
        if name not in arrays or _array_sha(arrays[name]) != digest:
            raise QuickIntegrityError('Canonical output array hash mismatch: '+name)
    out = {key: arrays[name] for name, key in CONFIG['output_names'].items()}
    out.update(valid=arrays['valid'], low_signal=arrays['low_signal'], info=receipt['info'],
               numerical_valid=receipt['production_numerical_certified'],
               calibration_status=receipt['calibration_status'], detection_status='NOT_CALIBRATED')
    canonical_arrays(out, shape=shape)
    return out


def _recover_output(path, identity, *, shape):
    sidecar = path.with_suffix('.output.json')
    if not path.exists() or not sidecar.exists():
        return None
    value = s.read(sidecar)
    _check_identity(value, identity)
    receipt = {**value, 'output_cache': _file_ref(path), 'output_metadata': _file_ref(sidecar),
               'output_bytes': path.stat().st_size, 'recovered_after_inverse_receipt_interruption': True}
    _load_output(receipt, identity=identity, shape=shape)
    return receipt


def evaluate_cell(cell, *, registration_sha256, initializer_provenance, report_dir, data_dir,
                  action, score_action, count_call, budget_guard, compact_info,
                  output_budget_bytes, verify_sources=lambda: None, shape=(256, 256)):
    """Persist inference first; score separately. Dependency injection is CPU-testable."""
    key = _cache_key(cell, registration_sha256)
    report = s.write_path(Path(report_dir)/'cells'/key)
    output = s.write_path(Path(data_dir)/(key+'.npz'))
    inference_path, started_path, score_path = (report/name for name in ('INFERENCE.json', 'STARTED.json', 'SCORE.json'))
    identity = {'registration_sha256': registration_sha256, 'method': cell['method'],
                'raw_sha256': cell['raw_sha256'], 'cache_key': key, 'case_id': cell['case_id']}
    row = {**cell, **identity, 'initializer_provenance': initializer_provenance,
           'labels_used_by_inference': False, 'final_payloads_opened': False}
    print('BEGIN '+cell['method']+' '+cell['case_id'], flush=True)
    new_call = False
    try:
        if inference_path.exists():
            inference = s.read(inference_path)
            _check_identity(inference, identity)
        else:
            recovered = _recover_output(output, identity, shape=shape)
            if recovered is not None:
                inference = {**recovered, 'inference_status': 'RETURNED', 'inference_attempted': True,
                             'time': s.now(), 'initializer_provenance': initializer_provenance}
            elif started_path.exists():
                _check_identity(s.read(started_path), identity)
                inference = {**identity, 'time': s.now(), 'inference_status': 'INTERRUPTED_WITHOUT_SAVED_OUTPUT',
                             'inference_attempted': True, 'finite_output': False,
                             'production_numerical_certified': False,
                             'reason': 'A previously started inverse is never repeated automatically'}
            else:
                verify_sources()
                budget_guard()
                s.write(started_path, {**identity, 'time': s.now(), 'pid': os.getpid(),
                                       'initializer_provenance': initializer_provenance})
                started = time.perf_counter()
                try:
                    count_call('repair_quick', cell['method'], cell['case_id'])
                    new_call = True
                    out = action()
                    inference = _save_output(output, out, identity, compact_info=compact_info,
                        output_budget_bytes=output_budget_bytes, data_dir=data_dir, shape=shape)
                    inference.update(inference_status='RETURNED', inference_attempted=True,
                                     time=s.now(), wall_s=time.perf_counter()-started,
                                     initializer_provenance=initializer_provenance)
                except Exception as error:
                    # Preserve a persisted finite endpoint even if a subsequent
                    # serialization step failed; never rerun the inverse for it.
                    inference = _recover_output(output, identity, shape=shape) or dict(identity)
                    inference.update(time=s.now(), inference_status='INFERENCE_OR_PERSISTENCE_FAILED',
                                     inference_attempted=new_call, finite_output=inference.get('finite_output', False),
                                     production_numerical_certified=inference.get('production_numerical_certified', False),
                                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                                     wall_s=time.perf_counter()-started, initializer_provenance=initializer_provenance)
            s.write(inference_path, s.plain(inference))
        row.update(inference, inference_receipt=_file_ref(inference_path), new_inverse_calls_this_run=int(new_call),
                   reused_inference=not new_call)
        if inference.get('output_budget_exceeded'):
            raise QuickIntegrityError('Quick canonical output budget exceeded; saved outputs retained')
        if 'output_cache' not in inference:
            row.update(status=inference['inference_status'], score_status='UNAVAILABLE_WITHOUT_OUTPUT', metrics=None)
            return s.plain(row)
        out = _load_output(inference, identity=identity, shape=shape)
        if score_path.exists():
            scored = s.read(score_path)
            _check_identity(scored, identity)
        else:
            try:
                # This is the sole point where external labels may be loaded.
                verify_sources()
                scored = {**score_action(out), **identity, 'time': s.now(), 'truth_loaded_after_inference': True}
                s.write(score_path, s.plain(scored))
            except Exception as error:
                failure = {**identity, 'time': s.now(), 'error_type': type(error).__name__,
                           'error': str(error), 'traceback': traceback.format_exc(),
                           'inference_receipt': _file_ref(inference_path)}
                number = len(list(report.glob('SCORE_FAILURE_*.json')))+1
                s.write(report/f'SCORE_FAILURE_{number:04d}.json', failure)
                row.update(status='SCORING_FAILED_INFERENCE_PRESERVED', score_status='SCORING_FAILED',
                           score_error=failure, metrics=None)
                return s.plain(row)
        # A score schema cannot silently convert a finite uncertified endpoint
        # into a numerical certificate or attach another method's detector.
        detector = (scored.get('metrics') or {}).get('frame_detection_v2')
        if detector is not None and (detector.get('threshold') is not None or detector.get('detected') is not None):
            raise QuickIntegrityError('Quick new methods have no frozen detection threshold')
        scored_numeric = bool(scored.get('production_numerical_certified', False))
        if scored_numeric != bool(inference.get('production_numerical_certified', False)):
            raise QuickIntegrityError('Scoring changed the saved numerical certification')
        row.update(scored, finite_output=inference['finite_output'], score_status='SCORED',
                   inference_status=inference['inference_status'], score_receipt=_file_ref(score_path))
        return s.plain(row)
    finally:
        print('END '+cell['method']+' '+cell['case_id']+' new_inverse_calls='+str(int(new_call)), flush=True)


def summarize_cells(planned, rows):
    """Every planned method/raw cell remains in all reported denominators."""
    supplied = {}
    for row in rows:
        key = row['method'], row['case_id']
        if key in supplied:
            raise QuickIntegrityError('Duplicate result cell')
        supplied[key] = row
    planned_keys = {(c['method'], c['case_id']) for c in planned}
    if not set(supplied).issubset(planned_keys):
        raise QuickIntegrityError('Unplanned quick result cell')
    complete = []
    for cell in planned:
        complete.append(supplied.get((cell['method'], cell['case_id']),
            {**cell, 'status': 'NOT_ATTEMPTED', 'inference_attempted': False, 'finite_output': False,
             'production_numerical_certified': False, 'metrics': None, 'score_status': 'NOT_ATTEMPTED'}))
    by_method = {}
    for method in dict.fromkeys(c['method'] for c in planned):
        group = [r for r in complete if r['method'] == method]
        scored = [r['metrics'] for r in group if r.get('metrics') is not None]
        finite_scored = [v for v in scored if v.get('domain_pixels', 0) > 0
                         and all(isinstance(v.get(k), (int, float)) and np.isfinite(v[k])
                                 for k in ('template_SSE', 'temperature_SSE_K2'))]
        pixels = sum(v['domain_pixels'] for v in finite_scored)
        template = float(np.sqrt(sum(v['template_SSE'] for v in finite_scored)/pixels)) if pixels else None
        temperature = float(np.sqrt(sum(v['temperature_SSE_K2'] for v in finite_scored)/pixels)) if pixels else None
        all_scored = len(finite_scored) == len(group)
        by_method[method] = {
            'planned_cases': len(group), 'attempted_cases': sum(bool(r.get('inference_attempted')) for r in group),
            'finite_output_cases': sum(bool(r.get('finite_output')) for r in group),
            'numerically_certified_cases': sum(bool(r.get('production_numerical_certified')) for r in group),
            'finite_scored_cases': len(finite_scored), 'failed_or_unavailable_scoring_cases': len(group)-len(finite_scored),
            'all_cases_finite_scored': all_scored, 'scoring_pixels': pixels,
            'complete_denominator_template_RMSE_e': template if all_scored else None,
            'complete_denominator_temperature_RMSE_K': temperature if all_scored else None,
            'conditional_finite_case_template_RMSE_e': template,
            'conditional_finite_case_temperature_RMSE_K': temperature,
            'frame_detection_available': False,
        }
    return {'planned_cells': len(planned), 'returned_cell_rows': len(rows),
            'attempted_cells': sum(bool(r.get('inference_attempted')) for r in complete),
            'finite_output_cells': sum(bool(r.get('finite_output')) for r in complete),
            'numerically_certified_cells': sum(bool(r.get('production_numerical_certified')) for r in complete),
            'new_inverse_calls_this_run': sum(r.get('new_inverse_calls_this_run', 0) for r in complete),
            'by_method': by_method, 'cells': complete, 'method_selection_performed': False,
            'failure_denominators_preserved': True, 'final_payloads_opened': False}


def _initializer_provenance(ref, route, registration):
    if route['initializer'] == 'smooth_demod':
        return {'kind': 'raw_only_smooth_demodulation', 'raw_file_sha256': ref['raw_reference']['sha256'],
                'raw_array_sha256': ref['raw_reference']['array_sha256'], 'truth_or_identity_used': False,
                'new_complete_initializer_inverse_calls': 0,
                'source': 'research_closeout/joint_estimator.py:initialize(smooth_demod)',
                'calibration_reference': registration['external_files'][2]}
    return registration['v3_initializer_provenance'][ref['case_id']]


def run(tier='smoke', device='cuda'):
    try:
        registration, digest = verify_registration()
    except Exception as error:
        _failure_receipt(REPORT_DIR if tier == 'smoke' else REPORT_DIR/'panel',
            {'time': s.now(), 'status': 'QUICK_REGISTRATION_VERIFICATION_FAILED', 'tier': tier,
             'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
             'new_complete_inverse_calls': 0, 'final_payloads_opened': False})
        raise
    try:
        return _run_registered(registration, digest, tier, device)
    except FileExistsError as error:
        _failure_receipt(REPORT_DIR if tier == 'smoke' else REPORT_DIR/'panel',
            {'time': s.now(), 'status': 'QUICK_PHASE_OR_IMMUTABLE_PATH_REFUSED', 'tier': tier,
             'registration_sha256': digest, 'error_type': type(error).__name__, 'error': str(error),
             'final_payloads_opened': False, 'whole_closeout_complete': False})
        raise


def _run_registered(registration, digest, tier, device):
    planned = planned_cells(registration['references'], tier)
    terminal_dir = REPORT_DIR if tier == 'smoke' else REPORT_DIR/'panel'
    with phase_locks((s.REPORTS/'PHASE_ACTIVE.json', REPORT_DIR/'PHASE_ACTIVE.json'), registration_sha256=digest):
        # A terminal result is immutable. Repeat invocations verify sources and
        # cached cells; they never refresh registration or replace old reports.
        result_path = terminal_dir/'QUICK_RESULT.json'
        if result_path.exists():
            previous = s.read(result_path)
            if previous.get('registration_sha256') != digest or previous.get('tier') != tier:
                raise QuickIntegrityError('Another quick result already occupies this terminal path')
            for row in previous['summary']['cells']:
                for name in ('inference_receipt', 'score_receipt'):
                    if row.get(name):
                        _verify_file_ref(row[name])
                if row.get('output_cache'):
                    identity = {k: row[k] for k in ('registration_sha256', 'method', 'raw_sha256', 'cache_key', 'case_id')}
                    _load_output(row, identity=identity)
            return previous
        rows = []
        try:
            import torch
            from . import measurement as m
            from . import metrics
            from . import experiment_registry as registry
            from .development import cached_v3, compact_info
            from .numerical_strategy import repaired_infer
            from .reliable_v3 import refined_infer
            s.frozen_bridge()
            from research_v6 import inputs
            torch.set_num_threads(2)
            p = m.physics(device)
            refs = {r['case_id']: r for r in registration['references']}
            active_case, raw, state, beta = None, None, None, None
            for cell in planned:
                ref = refs[cell['case_id']]
                if active_case != ref['case_id']:
                    raw = inputs.load_raw(ref, p)
                    state, beta = cached_v3(ref, p)
                    active_case = ref['case_id']
                def action(cell=cell):
                    if cell['family'] == 'joint':
                        return repaired_infer(raw, p, initializer=cell['initializer'],
                            v3_state=state.clone() if cell['initializer'] == 'v3' else None,
                            physical_options=copy.deepcopy(PHYSICAL_OPTIONS),
                            numerical_options=copy.deepcopy(NUMERICAL_OPTIONS))
                    return refined_infer(raw, p, initial_state=state.clone(), initial_beta=beta.clone(),
                                         options={'cost_limit_s': 180.})
                def score_action(out, ref=ref, cell=cell):
                    out = {k: torch.as_tensor(v, device=p.M.device) if isinstance(v, np.ndarray) else v
                           for k, v in out.items()}
                    truth = inputs.load_diagnostic_truth(ref, p)
                    return metrics.score(out, truth, raw, p, detector=None, method=cell['method'],
                                         application_domain='repair_quick_development_native256')
                row = evaluate_cell(cell, registration_sha256=digest,
                    initializer_provenance=_initializer_provenance(ref, cell, registration),
                    report_dir=REPORT_DIR, data_dir=DATA_DIR, action=action, score_action=score_action,
                    count_call=registry.count_call, budget_guard=s.budget_guard, compact_info=compact_info,
                    output_budget_bytes=CONFIG[tier+'_output_budget_bytes'],
                    verify_sources=lambda: verify_registration())
                rows.append(row)
                s.append(terminal_dir/'progress.jsonl', row)
            summary = summarize_cells(planned, rows)
            # Source edits during the final inverse/score cannot silently enter
            # an immutable successful terminal receipt.
            verify_registration()
            result = {'time': s.now(), 'status': 'QUICK_DIAGNOSTIC_COMPLETE', 'tier': tier,
                      'registration_sha256': digest, 'planned_max_new_inverse_calls': len(planned),
                      'inverse_wall_budget_s_excluding_overhead': CONFIG['tiers'][tier]['inverse_wall_budget_s'],
                      'summary': summary, 'whole_closeout_complete': False,
                      'not_a_final_method_selection': True, 'final_payloads_opened': False}
            if any(r.get('score_status') == 'SCORING_FAILED' for r in rows):
                result['status'] = 'QUICK_SCORING_FAILURE_INFERENCE_PRESERVED'
                _failure_receipt(terminal_dir, result)
            else:
                s.write(result_path, result)
            return result
        except Exception as error:
            failure = {'time': s.now(), 'status': 'QUICK_CONTROLLER_FAILURE_INFERENCE_PRESERVED',
                       'tier': tier, 'registration_sha256': digest, 'error_type': type(error).__name__,
                       'error': str(error), 'traceback': traceback.format_exc(),
                       'summary': summarize_cells(planned, rows), 'final_payloads_opened': False,
                       'whole_closeout_complete': False}
            _failure_receipt(terminal_dir, failure)
            raise


def _failure_receipt(folder, value):
    path = Path(folder)/'QUICK_FAILURE.json'
    if path.exists():
        number = len(list(Path(folder).glob('QUICK_FAILURE_RESUME_*.json')))+1
        path = Path(folder)/f'QUICK_FAILURE_RESUME_{number:04d}.json'
    return s.write(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    registration = sub.add_parser('register', help='freeze passed repair contracts, source, config and six old raws')
    registration.add_argument('--contract', action='append', required=True, type=Path)
    execute = sub.add_parser('run', help='execute or resume one explicitly selected tier')
    execute.add_argument('--tier', choices=('smoke', 'panel'), default='smoke')
    execute.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.command == 'register':
        result = register(args.contract)
        print(json.dumps({'status': 'QUICK_REGISTERED', 'path': str(REGISTRATION),
                          'sha256': s.sha(REGISTRATION), 'new_complete_inverse_calls': result['new_complete_inverse_calls']}), flush=True)
    else:
        result = run(args.tier, args.device)
        print(json.dumps({'status': result['status'], 'tier': result['tier'],
                          'registration_sha256': result['registration_sha256'],
                          'planned_cells': result['summary']['planned_cells'],
                          'finite_output_cells': result['summary']['finite_output_cells'],
                          'numerically_certified_cells': result['summary']['numerically_certified_cells']}), flush=True)
        if 'FAILURE' in result['status']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
