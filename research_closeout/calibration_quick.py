"""Independent C1v3 calibration diagnostic: two old raws, two initializers.

Registration freezes this new calibration/objective version. Execution makes
at most four new joint inversions (180 s each), with no v3-refined rerun. Old
v001 results remain immutable six-cell controls, including all six failures of
numerical certification. Latent parameters precede canonical output and score
caches; truth is read only by the latter external scoring callback.
"""
from . import storage as s
from . import quick_validation as q
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import traceback
import uuid
import numpy as np


VERSION = 'calibration-quick-v002-C1v3'
REPORT_DIR = s.REPORTS/'repair_001/quick_v002'
DATA_DIR = s.DATA/'repair_001/quick_v002'
REGISTRATION = REPORT_DIR/'QUICK_REGISTRATION.json'
REGISTRATION_SHA = REPORT_DIR/'QUICK_REGISTRATION_SHA256.json'
V001_RESULT = q.REPORT_DIR/'QUICK_RESULT.json'
CONTROLLER_PREFIX = 'CALIBRATION_QUICK_CONTROLLER_CONTRACTS'
ADAPTER_VERSION = 'continuum-c1-local-hermite-v003'
OBJECTIVE_VERSION = 'joint-profile-v003-shared-anchor-c1-calibration'
EXTRA_SOURCES = ('research_closeout/calibration_quick.py', 'research_closeout/tests/calibration_quick.py',
                 'research_closeout/smooth_calibration.py')
ROUTES = (
    {'method': 'joint_profile_C1v3_c1_s1_numerics_v2__v3', 'family': 'joint_C1v3', 'initializer': 'v3'},
    {'method': 'joint_profile_C1v3_c1_s1_numerics_v2__smooth_demod', 'family': 'joint_C1v3', 'initializer': 'smooth_demod'},
)
CONFIG = {'version': VERSION, 'adapter_family': 'C1v3', 'calibration_version': ADAPTER_VERSION,
          'physical_objective_version': OBJECTIVE_VERSION, 'physical_options': copy.deepcopy(q.PHYSICAL_OPTIONS),
          'numerical_options': {'cost_limit_s': 180.}, 'routes': list(ROUTES),
          'planned_raws': 2, 'planned_new_inverse_calls': 4, 'inverse_wall_budget_s': 720,
          'new_npz_output_budget_bytes': 24*2**20, 'free_space_floor_bytes': 2*2**30,
          'disk_safety_metadata_reserve_bytes': 512*2**10, 'global_new_inverse_call_cap': 12000,
          'parameter_arrays': ['z_grid', 'xi', 'beta'], 'parameter_dtype': 'float64',
          'continuum_latent_storage': 'reconstruct from z_grid with frozen measurement.continuum and C1v3 physics; no inverse',
          'canonical_arrays': ['T', 'u', 'C', 'predicted_raw_e', 'valid', 'low_signal'],
          'detector': None, 'native_shape': [256, 256], 'grid': 256,
          'old_v001_complete_control_denominator': 6, 'method_selection': False,
          'v3_refined_rerun': False, 'final_payloads_opened': False}


def planned_cells(refs):
    """Exactly the first two registered old development raws and two routes."""
    if len(refs) != 2:
        raise q.QuickIntegrityError('C1v3 diagnostic requires exactly two old raw references')
    expected = q.PANEL[:2]
    for ref, (mother, condition) in zip(refs, expected):
        if (ref.get('mother_id') != f'v5_development_selection_mother_{mother:04d}'
                or ref.get('condition') != condition or ref.get('cohort') != 'development'
                or ref.get('input_kind') != 'noise' or ref['raw_reference']['shape'] != [256, 256]):
            raise q.QuickIntegrityError('C1v3 diagnostic raw selection differs from v001 smoke')
    return [{**{key: ref[key] for key in ('case_id', 'mother_id', 'condition', 'input_kind', 'quick_geometry')},
             'raw_sha256': ref['raw_reference']['sha256'], **route} for ref in refs for route in ROUTES]


def adapter_identity(contract):
    required = {'calibration_version': ADAPTER_VERSION, 'physical_objective_version': OBJECTIVE_VERSION,
                'original_buffers_byte_identical': True, 'common_color_lookup_unchanged': True}
    for name, expected in required.items():
        if contract.get(name) != expected:
            raise q.QuickIntegrityError('Unexpected C1v3 calibration contract: '+name)
    if not isinstance(contract.get('fingerprint'), str) or len(contract['fingerprint']) != 64:
        raise q.QuickIntegrityError('An explicit C1v3 adapter fingerprint is required')
    return {'adapter_family': 'C1v3', **required, 'fingerprint': contract['fingerprint']}


def objective_fingerprint(contract, source_sha256):
    adapter_identity(contract)
    return s.canonical({'C1v3_adapter': contract, 'physical_objective_version': OBJECTIVE_VERSION,
                        'fixed_physical_options': CONFIG['physical_options'],
                        'source_sha256': source_sha256, 'initializer_changes_objective': False})


def _verify_v001():
    registration, digest = q.verify_registration()
    result = s.read(V001_RESULT)
    if result.get('registration_sha256') != digest or result.get('tier') != 'smoke':
        raise q.QuickIntegrityError('The completed immutable v001 smoke result is required')
    old_plan = q.planned_cells(registration['references'], 'smoke')
    summary = result['summary']
    if summary['planned_cells'] != 6 or len(summary['cells']) != 6:
        raise q.QuickIntegrityError('All six v001 cells, including failures, are required')
    if {(r['method'], r['case_id']) for r in summary['cells']} != {(r['method'], r['case_id']) for r in old_plan}:
        raise q.QuickIntegrityError('v001 six-cell control matrix changed')
    files = [q._file_ref(q.REGISTRATION), q._file_ref(q.REGISTRATION_SHA), q._file_ref(V001_RESULT)]
    for row in summary['cells']:
        for name in ('inference_receipt', 'score_receipt', 'output_cache', 'output_metadata'):
            if row.get(name):
                q._verify_file_ref(row[name])
                files.append(row[name])
        if row.get('output_cache'):
            identity = {name: row[name] for name in
                        ('registration_sha256', 'method', 'raw_sha256', 'cache_key', 'case_id')}
            q._load_output(row, identity=identity)
    return registration, result, files


def storage_ledger(data_dir=DATA_DIR, report_dir=REPORT_DIR):
    """All new persistent files are charged; NPZ includes parameter caches."""
    def files(folder):
        return [p for p in Path(folder).rglob('*') if p.is_file()] if Path(folder).exists() else []
    data, reports = files(data_dir), files(report_dir)
    npz = sum(p.stat().st_size for p in data if p.suffix == '.npz')
    parameter_npz = sum(p.stat().st_size for p in data if p.suffix == '.npz'
                        and 'parameters' in p.relative_to(Path(data_dir)).parts)
    pending = sum(p.stat().st_size for p in data if p.name.endswith('.pending'))
    other = sum(p.stat().st_size for p in data if p.suffix != '.npz' and not p.name.endswith('.pending'))
    report_bytes = sum(p.stat().st_size for p in reports)
    return {'data_npz_bytes': npz, 'data_parameter_npz_bytes': parameter_npz,
            'data_canonical_npz_bytes': npz-parameter_npz, 'data_pending_bytes': pending, 'data_other_bytes': other,
            'data_bytes': npz+pending+other,
            'report_bytes': report_bytes, 'total_new_bytes': npz+pending+other+report_bytes,
            'new_npz_budget_bytes': CONFIG['new_npz_output_budget_bytes'],
            'parameter_and_canonical_npz_both_counted': True}


def reserve_bytes(*, shape, bands, template_dimension):
    pixels = int(np.prod(shape))
    canonical = (3+bands)*pixels*8+2*pixels
    parameters = 2*pixels*8+int(template_dimension)*8+3*8
    # ZIP members add negligible overhead for these arrays; this conservative
    # allowance also reserves JSON receipts and the terminal result growth.
    return {'canonical_uncompressed_bytes': canonical, 'parameter_uncompressed_bytes': parameters,
            'one_inverse_reserve_bytes': canonical+parameters+CONFIG['disk_safety_metadata_reserve_bytes']}


def ensure_headroom(reserve, *, data_dir=DATA_DIR, report_dir=REPORT_DIR,
                    free_bytes=None, floor_guard=None, budget_bytes=None):
    (s.budget_guard if floor_guard is None else floor_guard)()
    ledger = storage_ledger(data_dir, report_dir)
    free = shutil.disk_usage(s.ROOT).free if free_bytes is None else int(free_bytes)
    needed = int(reserve['one_inverse_reserve_bytes'])
    cap = CONFIG['new_npz_output_budget_bytes'] if budget_bytes is None else int(budget_bytes)
    if ledger['data_bytes']+needed > cap:
        raise q.QuickIntegrityError('C1v3 parameter+canonical output reserve would exceed 24 MiB; preserve all old files')
    if free-needed < CONFIG['free_space_floor_bytes']:
        raise q.QuickIntegrityError('C1v3 next-inverse reserve would cross the 2 GiB disk floor')
    return {**ledger, 'free_bytes_before_inverse': free, 'reserved_next_inverse_bytes': needed,
            'free_after_reserve_bytes': free-needed}


def _current_contracts(paths):
    paths = [Path(path).resolve() for path in paths]
    exact = {name: s.sha(s.ROOT/name) for name in EXTRA_SOURCES[:2]}
    controllers = [path.resolve() for path in REPORT_DIR.glob(CONTROLLER_PREFIX+'*.json')
                   if (value := s.read(path)).get('all_passed') is True
                   and value.get('source_sha256') == exact]
    if not controllers:
        raise q.QuickIntegrityError('Run current C1v3 controller CPU contracts before registration')
    if not any(path in controllers for path in paths):
        paths.append(max(controllers, key=lambda path: path.stat().st_mtime_ns))
    if len(paths) < 2 or len(paths) != len(set(paths)):
        raise ValueError('Supply passed C1 calibration contracts; controller contracts are added automatically')
    receipts, adapter_checked = [], False
    for path in paths:
        q._inside(path, s.REPORTS/'repair_001')
        value = s.read(path)
        if value.get('all_passed') is not True:
            raise q.QuickIntegrityError('C1v3 repair contracts have not passed: '+str(path))
        if value.get('source_sha256'):
            q.verify_hashes(value['source_sha256'])
        adapter_checked |= 'research_closeout/smooth_calibration.py' in value.get('source_sha256', {})
        receipts.append(q._file_ref(path))
    if not adapter_checked:
        raise q.QuickIntegrityError('A passed contract bound to smooth_calibration.py is required')
    return receipts


def register(contract_paths):
    if REGISTRATION.exists() or REGISTRATION_SHA.exists():
        raise FileExistsError('C1v3 registration already exists; never overwrite it')
    for path in (s.REPORTS/'PHASE_ACTIVE.json', REPORT_DIR/'PHASE_ACTIVE.json'):
        if path.exists():
            raise FileExistsError('An active phase must finish before C1v3 registration')
    contracts = _current_contracts(contract_paths)
    old, result, old_files = _verify_v001()
    from .smooth_calibration import physics, contract
    from .observable_basis import RealFourierBasis
    from .development import cached_v3
    s.frozen_bridge()
    from research_v6 import inputs
    import torch
    torch.set_num_threads(2)
    p = physics('cpu')
    adapter = s.plain(contract(p))
    adapter_identity(adapter)
    refs = copy.deepcopy(old['references'][:2])
    planned_cells(refs)
    provenance = {ref['case_id']: old['v3_initializer_provenance'][ref['case_id']] for ref in refs}
    for ref in refs:
        raw = inputs.load_raw(ref, p)
        q.verify_initializer_raw(ref, raw, provenance[ref['case_id']])
        state, beta = cached_v3(ref, p)
        if state.shape != (1, 5, 256, 256) or beta.shape != (1, 3):
            raise q.QuickIntegrityError('Wrong old v3 initializer cache shape')
        if not bool(torch.isfinite(state).all() and torch.isfinite(beta).all()):
            raise q.QuickIntegrityError('Nonfinite old v3 initializer state')
    reserve = reserve_bytes(shape=p.cfg['shape'], bands=p.M.shape[0],
                            template_dimension=RealFourierBasis(p.op.object_otf).dimension)
    space = ensure_headroom(reserve)
    sources = dict(old['source_sha256'])
    sources.update({name: s.sha(s.ROOT/name) for name in EXTRA_SOURCES})
    for receipt in contracts:
        for name, digest in s.read(receipt['path']).get('source_sha256', {}).items():
            if name in sources and sources[name] != digest:
                raise q.QuickIntegrityError('Repair contract source digest conflicts with the scoped freeze: '+name)
            sources[name] = digest
    config_fingerprint = s.canonical({'config': CONFIG, 'C1v3_adapter': adapter, 'sources': sources})
    value = {'schema': VERSION+'-registration', 'time': s.now(), 'source_sha256': sources,
             'contracts': contracts, 'config': copy.deepcopy(CONFIG), 'config_fingerprint': config_fingerprint,
             'calibration_contract': adapter, 'C1v3_adapter_identity': adapter_identity(adapter),
             'objective_fingerprint': objective_fingerprint(adapter, sources),
             'references': refs, 'v3_initializer_provenance': provenance,
             'v001_registration_sha256': result['registration_sha256'], 'v001_files': old_files,
             'v001_summary': result['summary'], 'v001_controls_are_reused': True,
             'v001_failed_numerical_certification_not_relabelled': True,
             'space_reserve': reserve, 'storage_at_registration': space,
             'new_complete_inverse_calls': 0, 'truth_loaded': False, 'final_payloads_opened': False}
    s.write(REGISTRATION, value)
    s.write(REGISTRATION_SHA, q._file_ref(REGISTRATION))
    return value


def verify_registration():
    digest = s.read(REGISTRATION_SHA)
    if Path(digest['path']).resolve() != REGISTRATION.resolve():
        raise q.QuickIntegrityError('C1v3 registration SHA sidecar names another file')
    q._verify_file_ref(digest)
    value = s.read(REGISTRATION)
    if value['schema'] != VERSION+'-registration' or value['config'] != CONFIG:
        raise q.QuickIntegrityError('C1v3 configuration differs from registration')
    q.verify_hashes(value['source_sha256'])
    for receipt in value['contracts']+value['v001_files']:
        q._verify_file_ref(receipt)
    old, result, _ = _verify_v001()
    if old['references'][:2] != value['references'] or result['summary'] != value['v001_summary']:
        raise q.QuickIntegrityError('Frozen v001 references or full six-cell control result changed')
    adapter_identity(value['calibration_contract'])
    if s.canonical({'config': CONFIG, 'C1v3_adapter': value['calibration_contract'],
                    'sources': value['source_sha256']}) != value['config_fingerprint']:
        raise q.QuickIntegrityError('C1v3 configuration/source fingerprint mismatch')
    if objective_fingerprint(value['calibration_contract'], value['source_sha256']) != value['objective_fingerprint']:
        raise q.QuickIntegrityError('C1v3 objective fingerprint mismatch')
    return value, digest['sha256']


@contextmanager
def phase_locks(registration_sha256, paths=None):
    owner = {'pid': os.getpid(), 'owner_token': uuid.uuid4().hex, 'phase': VERSION,
             'registration_sha256': registration_sha256, 'time': s.now()}
    acquired = []
    try:
        for path in ((s.REPORTS/'PHASE_ACTIVE.json', REPORT_DIR/'PHASE_ACTIVE.json') if paths is None else paths):
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
                pass


def identity_for(cell, digest):
    return {'registration_sha256': digest, 'method': cell['method'], 'raw_sha256': cell['raw_sha256'],
            'case_id': cell['case_id'], 'cache_key': q._cache_key(cell, digest)}


def wrap_output(out, cell, registration):
    core = s.plain(out['info'])
    physical = core.get('physical_objective', {})
    if not physical.get('continuum_regscale') or not physical.get('regularizer_fingerprint'):
        raise q.QuickIntegrityError('Underlying shared-anchor profile omitted its fixed regularizer contract')
    wrapper = {'method': cell['method'], 'status': core['status'],
               'physical_objective_version': OBJECTIVE_VERSION,
               'core_reported_physical_objective_version': core.get('physical_objective_version'),
               'core_method_label_scope': 'unchanged v2 numerical implementation; effective calibrated objective is C1v3',
               'calibration_contract': registration['calibration_contract'],
               'C1v3_adapter_fingerprint': registration['calibration_contract']['fingerprint'],
               'objective_fingerprint': registration['objective_fingerprint'],
               'config_fingerprint': registration['config_fingerprint'],
               'anchor_regscale_contract': physical, 'numerical_engine': core,
               'numerical_certified': bool(out.get('numerical_valid', False)),
               'truth_used': False, 'old_results_relabelled': False,
               'continuum_latent_cache_rule': CONFIG['continuum_latent_storage']}
    return {**out, 'info': wrapper}


def _parameter_paths(identity, report_dir, data_dir):
    key = identity['cache_key']
    return (Path(data_dir)/'parameters'/(key+'.npz'),
            Path(data_dir)/'parameters'/(key+'.metadata.json'),
            Path(report_dir)/'cells'/key/'PARAMETERS.json')


def save_parameters(out, identity, registration, *, report_dir=REPORT_DIR, data_dir=DATA_DIR, shape=(256, 256)):
    """Save exact z/xi/beta first; no repeated seven-band latent continuum."""
    arrays = {key: q._array(out[key]) for key in CONFIG['parameter_arrays']}
    if any(value.dtype != np.float64 for value in arrays.values()):
        raise q.QuickIntegrityError('Preserve native float64 latent parameters without conversion')
    if arrays['z_grid'].shape != (1, 2, *shape) or arrays['xi'].ndim != 1 or arrays['beta'].shape != (1, 3):
        raise q.QuickIntegrityError('Malformed native latent parameter cache')
    path, metadata_path, receipt_path = _parameter_paths(identity, report_dir, data_dir)
    metadata = {**identity, 'array_sha256': {k: q._array_sha(v) for k, v in arrays.items()},
                'array_shapes': {k: list(v.shape) for k, v in arrays.items()}, 'dtype': 'float64',
                'parameters_finite': all(bool(np.isfinite(v).all()) for v in arrays.values()),
                'calibration_contract': registration['calibration_contract'],
                'source_sha256': registration['source_sha256'],
                'config_fingerprint': registration['config_fingerprint'],
                'objective_fingerprint': registration['objective_fingerprint'],
                'inference_info': out['info'], 'numerical_valid': bool(out.get('numerical_valid', False)),
                'canonical_array_sha256': {k: q._array_sha(v) for k, v in q.canonical_arrays(out, shape=shape).items()},
                'continuum_latent_stored': False, 'continuum_latent_reconstruction': CONFIG['continuum_latent_storage'],
                'truth_loaded_before_persisting': False}
    s.write(metadata_path, s.plain(metadata))
    temporary = s.write_path(path.with_suffix('.npz.pending'))
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or temporary.exists():
        raise FileExistsError('Never overwrite C1v3 parameters: '+str(path))
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)
    receipt = {**identity, 'parameter_cache': q._file_ref(path), 'parameter_metadata': q._file_ref(metadata_path),
               'parameter_bytes': path.stat().st_size, 'parameters_finite': metadata['parameters_finite'],
               'C1v3_adapter_fingerprint': registration['calibration_contract']['fingerprint'],
               'config_fingerprint': registration['config_fingerprint'],
               'objective_fingerprint': registration['objective_fingerprint']}
    s.write(receipt_path, receipt)
    return receipt


def load_parameters(receipt, identity, registration):
    q._check_identity(receipt, identity)
    for name in ('parameter_cache', 'parameter_metadata'):
        q._verify_file_ref(receipt[name])
    metadata = s.read(receipt['parameter_metadata']['path'])
    q._check_identity(metadata, identity)
    for key in ('config_fingerprint', 'objective_fingerprint', 'source_sha256', 'calibration_contract'):
        if metadata.get(key) != registration[key]:
            raise q.QuickIntegrityError('Parameter reconstruction contract mismatch: '+key)
    arrays = s.arrays(receipt['parameter_cache']['path'])
    if set(arrays) != set(CONFIG['parameter_arrays']):
        raise q.QuickIntegrityError('Missing or extra latent parameter arrays')
    for name, array in arrays.items():
        if array.dtype != np.float64 or q._array_sha(array) != metadata['array_sha256'][name]:
            raise q.QuickIntegrityError('Changed latent parameter precision or bytes: '+name)
    return arrays, metadata


def parameter_receipt(cell, digest, registration, *, report_dir=REPORT_DIR, data_dir=DATA_DIR):
    identity = identity_for(cell, digest)
    path, metadata, receipt = _parameter_paths(identity, report_dir, data_dir)
    if receipt.exists():
        value = s.read(receipt)
        load_parameters(value, identity, registration)
        return value
    if path.exists() and metadata.exists():
        value = {**identity, 'parameter_cache': q._file_ref(path), 'parameter_metadata': q._file_ref(metadata),
                 'parameter_bytes': path.stat().st_size, 'recovered_parameter_receipt': True}
        load_parameters(value, identity, registration)
        s.write(receipt, value)
        return value
    return None


def restore_canonical(cell, digest, registration, p, raw, *, report_dir=REPORT_DIR, data_dir=DATA_DIR,
                      shape=(256, 256), reconstruct=None):
    """Crash recovery from exact parameters only; no optimizer or inner solve."""
    identity = identity_for(cell, digest)
    canonical = Path(data_dir)/(identity['cache_key']+'.npz')
    if canonical.exists():
        return
    receipt = parameter_receipt(cell, digest, registration, report_dir=report_dir, data_dir=data_dir)
    if receipt is None:
        return
    arrays, metadata = load_parameters(receipt, identity, registration)
    info = {**metadata['inference_info'], 'parameter_cache_receipt': receipt}
    if reconstruct is None:
        from . import measurement as m
        from .observable_basis import RealFourierBasis
        import torch
        z, xi, beta = (torch.as_tensor(arrays[k], device=p.M.device, dtype=p.M.dtype) for k in CONFIG['parameter_arrays'])
        C = m.continuum(p, z, grid=CONFIG['grid'])
        u = RealFourierBasis(p.op.object_otf).unpack(xi)[None]
        out = m.output(p, p.raw_to_electrons(m.raw_tensor(raw, p)), C, u, beta, info)
    else:
        out = reconstruct(arrays, metadata)
    out['info'] = info
    out['numerical_valid'] = bool(metadata['numerical_valid'])
    for name, value in q.canonical_arrays(out, shape=shape).items():
        if q._array_sha(value) != metadata['canonical_array_sha256'][name]:
            raise q.QuickIntegrityError('Saved-parameter reconstruction differs from original canonical bytes: '+name)
    metadata_path = canonical.with_suffix('.output.json')
    if metadata_path.exists():
        original_metadata = s.read(metadata_path)
        q._check_identity(original_metadata, identity)
        arrays = q.canonical_arrays(out, shape=shape)
        if {name: q._array_sha(value) for name, value in arrays.items()} != original_metadata['array_sha256']:
            raise q.QuickIntegrityError('Interrupted canonical metadata differs from reconstructed parameter bytes')
        temporary = canonical.with_suffix('.npz.pending')
        if temporary.exists():
            archived = temporary.with_name(temporary.name+'.interrupted.'+uuid.uuid4().hex)
            temporary.replace(archived)
        with temporary.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
        temporary.replace(canonical)
    else:
        q._save_output(canonical, out, identity, compact_info=lambda value: s.plain(value['info']),
                       output_budget_bytes=CONFIG['new_npz_output_budget_bytes'], data_dir=data_dir, shape=shape)


def _old_output_comparisons(out, cell, registration):
    old_method = 'joint_c1_s1_repaired_v2__'+cell['initializer']
    old = next(r for r in registration['v001_summary']['cells']
               if r['case_id'] == cell['case_id'] and r['method'] == old_method)
    old_identity = {key: old[key] for key in ('registration_sha256', 'method', 'raw_sha256', 'cache_key', 'case_id')}
    original = q._load_output(old, identity=old_identity)
    differences = {}
    for name, key in q.CONFIG['output_names'].items():
        delta = q._array(out[key])-q._array(original[key])
        differences[name] = {'RMS': float(np.sqrt(np.mean(delta*delta))), 'maximum_absolute': float(np.max(np.abs(delta)))}
    return {'old_method': old_method, 'old_output_cache': old['output_cache'], 'differences': differences,
            'truth_used_for_comparison': False, 'old_numerical_certification': old['production_numerical_certified']}


def _failure(value):
    return q._failure_receipt(REPORT_DIR, value)


def run(device='cuda'):
    rows = []
    planned = []
    try:
        registration, digest = verify_registration()
        planned = planned_cells(registration['references'])
        with phase_locks(digest):
            result_path = REPORT_DIR/'QUICK_RESULT.json'
            if result_path.exists():
                result = s.read(result_path)
                if result['registration_sha256'] != digest:
                    raise q.QuickIntegrityError('Another registration owns this C1v3 terminal result')
                for row in result['summary']['cells']:
                    for name in ('inference_receipt', 'score_receipt'):
                        if row.get(name):
                            q._verify_file_ref(row[name])
                    if row.get('output_cache'):
                        q._load_output(row, identity=identity_for(row, digest))
                        parameter_receipt(row, digest, registration)
                return result
            from .smooth_calibration import physics, contract
            from .numerical_strategy import repaired_infer
            from .development import cached_v3
            from . import experiment_registry as registry
            from . import metrics
            import torch
            torch.set_num_threads(2)
            s.frozen_bridge()
            from research_v6 import inputs
            p = physics(device)
            if s.plain(contract(p)) != registration['calibration_contract']:
                raise q.QuickIntegrityError('Runtime C1v3 calibration adapter contract differs from registration')
            refs = {ref['case_id']: ref for ref in registration['references']}
            active, raw, state = None, None, None
            for cell in planned:
                ref = refs[cell['case_id']]
                if active != ref['case_id']:
                    raw = inputs.load_raw(ref, p)
                    state, _ = cached_v3(ref, p)
                    q.verify_initializer_raw(ref, raw, registration['v3_initializer_provenance'][ref['case_id']])
                    active = ref['case_id']
                restore_canonical(cell, digest, registration, p, raw)
                before = None
                def guard():
                    nonlocal before
                    before = ensure_headroom(registration['space_reserve'])
                def action(cell=cell):
                    out = repaired_infer(raw, p, initializer=cell['initializer'],
                        v3_state=state.clone() if cell['initializer'] == 'v3' else None,
                        physical_options=copy.deepcopy(CONFIG['physical_options']),
                        numerical_options=copy.deepcopy(CONFIG['numerical_options']))
                    out = wrap_output(out, cell, registration)
                    try:
                        receipt = save_parameters(out, identity_for(cell, digest), registration)
                        out['info']['parameter_cache_receipt'] = receipt
                    except Exception as error:
                        # Still preserve the finite canonical endpoint if its
                        # parameter persistence failed; the controller fails.
                        out['info']['parameter_cache_error'] = {'error_type': type(error).__name__, 'error': str(error)}
                    return out
                def score_action(out, cell=cell, ref=ref):
                    receipt = out['info'].get('parameter_cache_receipt')
                    if receipt is None:
                        raise q.QuickIntegrityError('Finite inverse parameters were not safely persisted')
                    load_parameters(receipt, identity_for(cell, digest), registration)
                    comparison = _old_output_comparisons(out, cell, registration)
                    tensors = {key: torch.as_tensor(value, device=p.M.device) if isinstance(value, np.ndarray) else value
                               for key, value in out.items()}
                    truth = inputs.load_diagnostic_truth(ref, p)
                    return {**metrics.score(tensors, truth, raw, p, detector=None, method=cell['method'],
                            application_domain='C1v3_quick_development_native256'), 'v001_output_comparison': comparison}
                provenance = (registration['v3_initializer_provenance'][ref['case_id']] if cell['initializer'] == 'v3'
                    else {'kind': 'C1v3_raw_only_smooth_demodulation', 'raw_file_sha256': cell['raw_sha256'],
                          'C1v3_adapter_fingerprint': registration['calibration_contract']['fingerprint'],
                          'truth_or_identity_used': False, 'new_complete_initializer_inverse_calls': 0})
                row = q.evaluate_cell(cell, registration_sha256=digest, initializer_provenance=provenance,
                    report_dir=REPORT_DIR, data_dir=DATA_DIR, action=action, score_action=score_action,
                    count_call=registry.count_call, budget_guard=guard, compact_info=lambda out: s.plain(out['info']),
                    output_budget_bytes=CONFIG['new_npz_output_budget_bytes'], verify_sources=verify_registration)
                row['storage_before_new_inverse'] = before
                row['storage_after_cell'] = storage_ledger()
                rows.append(row)
                if row.get('output_cache'):
                    receipt = parameter_receipt(row, digest, registration)
                    if receipt is None:
                        raise q.QuickIntegrityError('Canonical endpoint exists without its required exact parameter cache')
                    row['parameter_cache_receipt'] = receipt
                s.append(REPORT_DIR/'progress.jsonl', row)
            verify_registration()
            summary = q.summarize_cells(planned, rows)
            final_storage = storage_ledger()
            if final_storage['data_bytes'] > CONFIG['new_npz_output_budget_bytes']:
                raise q.QuickIntegrityError('C1v3 total new data exceeded the frozen 24 MiB output budget')
            s.budget_guard()
            result = {'time': s.now(), 'status': 'C1V3_QUICK_DIAGNOSTIC_COMPLETE',
                      'registration_sha256': digest, 'summary': summary,
                      'calibration_contract': registration['calibration_contract'],
                      'physical_objective_version': OBJECTIVE_VERSION,
                      'objective_fingerprint': registration['objective_fingerprint'],
                      'storage_ledger': final_storage, 'v001_summary': registration['v001_summary'],
                      'old_v001_all_six_cells_preserved': True, 'old_failures_relabelled': False,
                      'planned_max_new_inverse_calls': 4, 'inverse_wall_budget_s_excluding_overhead': 720,
                      'not_a_final_method_selection': True, 'whole_closeout_complete': False,
                      'final_payloads_opened': False}
            if any(r.get('score_status') == 'SCORING_FAILED' for r in rows):
                result['status'] = 'C1V3_QUICK_SCORING_FAILURE_INFERENCE_PRESERVED'
                _failure(result)
            else:
                s.write(result_path, result)
            return result
    except Exception as error:
        failure = {'time': s.now(), 'status': 'C1V3_QUICK_CONTROLLER_FAILURE_INFERENCE_PRESERVED',
                   'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                   'summary': q.summarize_cells(planned, rows) if planned else None,
                   'storage_ledger': storage_ledger(), 'whole_closeout_complete': False, 'final_payloads_opened': False}
        _failure(failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    register_parser = sub.add_parser('register', help='freeze new C1 calibration contracts, sources and old controls')
    register_parser.add_argument('--contract', type=Path, action='append', required=True)
    execute = sub.add_parser('run', help='run or resume the fixed four-inverse C1v3 diagnostic')
    execute.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.command == 'register':
        value = register(args.contract)
        result = {'status': 'C1V3_QUICK_REGISTERED', 'path': str(REGISTRATION),
                  'sha256': s.sha(REGISTRATION), 'objective_fingerprint': value['objective_fingerprint'],
                  'new_complete_inverse_calls': 0}
    else:
        value = run(args.device)
        result = {'status': value['status'], 'planned_cells': value['summary']['planned_cells'],
                  'finite_output_cells': value['summary']['finite_output_cells'],
                  'numerically_certified_cells': value['summary']['numerically_certified_cells']}
    print(json.dumps(result, ensure_ascii=True), flush=True)
    if 'FAILURE' in result['status']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
