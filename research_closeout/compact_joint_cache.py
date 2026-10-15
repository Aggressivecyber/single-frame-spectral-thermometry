"""Fail-closed float64 joint-state cache, with byte-verified forward recovery.

This module never invokes an inverse, optimizer, LSMR or truth loader. A future
registered controller calls ``begin(raw)`` before its externally counted
inverse, then ``persist(ticket, output, raw)``. Only a newly issued begin ticket
permits a new inverse. Existing, interrupted and failed instances never issue
another such ticket. ``load_for_score(raw)`` reconstructs using forward calls
only and checks every original canonical array's dtype/shape/bytes hash.

On initial persistence failure the original in-memory canonical output is
written to an already allocated full fallback, and scoring is forbidden. A
later reconstruction failure after verified compaction cannot recreate an
original fallback which was never stored: it retains parameters/hashes,
records that limitation explicitly, and forbids scoring or another inverse.
No existing v001/v002 dataset is converted, rewritten or deleted.
"""
from . import storage as s
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid
import numpy as np


SCHEMA = 'joint-float64-forward-lossless-cache-v001'
REPORT_DIR = s.REPORTS/'repair_001/compact_cache_001'
GLOBAL_LIMIT_BYTES = 4*2**30
FREE_FLOOR_BYTES = 2*2**30
METADATA_ALLOWANCE_BYTES = 256*2**10
ZIP_ALLOWANCE_BYTES = 64*2**10
PARAMETER_KEYS = ('z_grid', 'xi', 'beta')
CANONICAL_KEYS = ('temperature_color_K', 'template_postblur_e_ref', 'continuum_postblur_bands',
                  'predicted_raw_e', 'predicted_raw_DN', 'standardized_residual', 'valid', 'low_signal')
FLOAT_CANONICAL_KEYS = CANONICAL_KEYS[:-2]
FORWARD_SOURCES = ('research_closeout/compact_joint_cache.py', 'research_closeout/storage.py',
                   'research_closeout/measurement.py', 'research_closeout/observable_basis.py',
                   'physics.py', 'runtime.py')


class CacheIntegrityError(RuntimeError):
    pass


class CacheNotScoreable(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt.get('status', 'FAIL_CLOSED'))


def format_contract():
    """Prospective registration contract; no native/GPU qualification claim."""
    return {'schema': SCHEMA, 'parameters': list(PARAMETER_KEYS), 'canonical_arrays': list(CANONICAL_KEYS),
            'parameter_dtype': 'float64', 'canonical_float_dtype': 'float64', 'mask_dtype': 'bool',
            'initial_readback': 'same p/device, exact dtype+shape+bytes SHA256 for every canonical array and C_latent',
            'initial_mismatch': 'save full original canonical fallback in dense owner reservation; FAIL_CLOSED',
            'later_mismatch': 'retain parameters/hashes and refuse scoring/refit; original fallback may not exist',
            'numeric_certification_is_separate': True, 'inverse_calls_by_cache': 0, 'truth_loaded_by_cache': False,
            'global_budget_roots': [str(s.DATA), str(s.REPORTS)], 'global_limit_bytes': GLOBAL_LIMIT_BYTES,
            'free_floor_root': str(s.ROOT), 'free_floor_bytes': FREE_FLOOR_BYTES,
            'report_root_and_data_root_configurable_within_existing_storage_whitelist': True,
            'new_D_root_writes_supported': False, 'storage_permission_source': 'research_closeout/storage.py:write_path',
            'caller_must_own_registered_serial_study_execution': True,
            'existing_v001_v002_migration_or_conversion_supported': False,
            'native_GPU_byte_exact_qualification_performed': False}


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def array_sha(value):
    a = np.ascontiguousarray(array(value))
    h = hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def descriptor(value):
    value = array(value)
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'nbytes': value.nbytes,
            'sha256': array_sha(value)}


def _encode(value):
    """Retain every info field, including nonfinite float64 payloads."""
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if hasattr(value, 'tolist'):
        return _encode(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {'__compact_float64_bits__': format(np.float64(value).view(np.uint64).item(), '016x')}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Info/config must contain JSON values: '+type(value).__name__)


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {'__compact_float64_bits__'}:
            return np.uint64(int(value['__compact_float64_bits__'], 16)).view(np.float64).item()
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _text(value):
    return json.dumps(_encode(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'


def file_ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': s.sha(path), 'bytes': path.stat().st_size}


def verify_file(receipt):
    path = Path(receipt['path'])
    if not path.is_file() or s.sha(path) != receipt['sha256']:
        raise CacheIntegrityError('Registered/cache file changed or missing: '+str(path))
    return path


def _within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path != parent and parent not in path.parents:
        raise CacheIntegrityError('Cache path escapes its declared instance: '+str(path))
    return path


def _write_once(path, value):
    path = s.write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as handle:
        handle.write(_text(value))
        handle.flush()
        os.fsync(handle.fileno())
    sha_path = path.with_name(path.name+'.sha256.json')
    with sha_path.open('x', encoding='utf8', newline='\n') as handle:
        handle.write(_text(file_ref(path)))
        handle.flush()
        os.fsync(handle.fileno())
    return file_ref(path)


def _read_sealed(path):
    path = Path(path).resolve()
    receipt = s.read(path.with_name(path.name+'.sha256.json'))
    if Path(receipt['path']).resolve() != path:
        raise CacheIntegrityError('Receipt SHA sidecar points at another file')
    verify_file(receipt)
    return _decode(s.read(path))


def verify_sources(hashes):
    for name, digest in hashes.items():
        path = _within(s.ROOT/name, s.ROOT)
        if not path.is_file() or s.sha(path) != digest:
            raise CacheIntegrityError('Forward dependency changed or missing: '+name)


def source_hashes(extra=None):
    values = {name: s.sha(s.ROOT/name) for name in FORWARD_SOURCES}
    for name, digest in (extra or {}).items():
        if name in values and values[name] != digest:
            raise CacheIntegrityError('Caller dependency conflicts with the active forward implementation: '+name)
        values[name] = digest
    verify_sources(values)
    return values


def tree_ledger(roots):
    """Count every file once: data, metadata, logs, pending, reserve, fallback."""
    files = {}
    for root in roots:
        root = Path(root).resolve()
        if root.is_file():
            files[str(root)] = root.stat().st_size
        elif root.is_dir():
            for path in root.rglob('*'):
                if path.is_file():
                    files[str(path.resolve())] = path.stat().st_size
    groups = {'parameters': 0, 'fallback': 0, 'reserve': 0, 'pending': 0, 'other_data_meta_cache_log': 0}
    for name, size in files.items():
        filename = Path(name).name
        category = ('reserve' if filename.startswith('FALLBACK_RESERVED_') else
                    'fallback' if filename == 'FALLBACK.npz' else
                    'pending' if '.pending.' in filename else
                    'parameters' if filename == 'PARAMETERS.npz' else 'other_data_meta_cache_log')
        groups[category] += size
    return {'bytes': sum(files.values()), 'files': len(files), 'by_kind_bytes': groups,
            'all_file_kinds_counted': True, 'JSON_references_not_dereferenced_or_copied': True}


class Budget:
    def __init__(self, instance_roots, *, global_roots=None, instance_limit_bytes=GLOBAL_LIMIT_BYTES,
                 global_limit_bytes=GLOBAL_LIMIT_BYTES, free_floor_bytes=FREE_FLOOR_BYTES,
                 free_bytes=None):
        self.instance_roots = tuple(Path(p).resolve() for p in instance_roots)
        self.global_roots = tuple(Path(p).resolve() for p in
                                  ((s.DATA, s.REPORTS) if global_roots is None else global_roots))
        self.instance_limit = int(instance_limit_bytes)
        self.global_limit = int(global_limit_bytes)
        self.floor = int(free_floor_bytes)
        self._free_bytes = free_bytes
        if min(self.instance_limit, self.global_limit, self.floor) < 0:
            raise ValueError('Budget bounds must be nonnegative')

    def ledger(self):
        return {'instance': tree_ledger(self.instance_roots), 'global_new_study': tree_ledger(self.global_roots),
                'instance_limit_bytes': self.instance_limit, 'global_limit_bytes': self.global_limit,
                'free_floor_bytes': self.floor}

    def check(self, requested_bytes=0):
        requested = int(requested_bytes)
        if requested < 0:
            raise ValueError('A nonnegative reservation is required')
        ledger = self.ledger()
        free = (shutil.disk_usage(s.ROOT).free if self._free_bytes is None else
                self._free_bytes() if callable(self._free_bytes) else int(self._free_bytes))
        if ledger['global_new_study']['bytes']+requested > self.global_limit:
            raise CacheIntegrityError('Global 4 GiB new-study DATA+REPORTS budget would be exceeded')
        if ledger['instance']['bytes']+requested > self.instance_limit:
            raise CacheIntegrityError('Declared cache instance budget would be exceeded')
        if free-requested < self.floor:
            raise CacheIntegrityError('Dense fallback/parameter reserve would cross the 2 GiB free-space floor')
        return {**ledger, 'free_bytes': free, 'additional_reserved_bytes': requested,
                'free_after_reserve_bytes': free-requested}


def frame_plan(*, shape, bands, template_dimension, grid=None):
    h, w = map(int, shape)
    g = h if grid is None else int(grid)
    if h < 1 or w < 1 or g < 1 or int(bands) < 1 or int(template_dimension) < 1:
        raise ValueError('Positive frame/basis dimensions are required')
    parameters = (2*g*g+int(template_dimension)+3)*8
    canonical = (int(bands)+5)*h*w*8+2*h*w
    return {'shape': [h, w], 'grid': g, 'bands': int(bands), 'template_dimension': int(template_dimension),
            'parameter_payload_bytes': parameters, 'parameter_zip_bound_bytes': parameters+ZIP_ALLOWANCE_BYTES,
            'canonical_payload_bytes': canonical, 'fallback_zip_bound_bytes': canonical+ZIP_ALLOWANCE_BYTES,
            'metadata_bound_bytes': METADATA_ALLOWANCE_BYTES,
            'successful_compact_bound_bytes': parameters+ZIP_ALLOWANCE_BYTES+METADATA_ALLOWANCE_BYTES,
            'initial_safe_reserve_bytes': parameters+canonical+2*ZIP_ALLOWANCE_BYTES+METADATA_ALLOWANCE_BYTES,
            'canonical_array_keys': list(CANONICAL_KEYS), 'all_bounds_preserve_float64': True}


def runtime_identity(p):
    import torch
    device = p.M.device
    result = {'device': str(device), 'dtype': str(p.M.dtype), 'torch': str(torch.__version__),
              'python': sys.version, 'machine': platform.machine(), 'platform': platform.platform(),
              'threads': torch.get_num_threads(), 'torch_cuda_build': torch.version.cuda,
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result['device_properties'] = {'name': properties.name, 'major': properties.major,
                                       'minor': properties.minor, 'uuid': str(getattr(properties, 'uuid', None))}
    return result


def _physics_identity(p):
    return {'class': type(p).__module__+'.'+type(p).__qualname__, 'config_sha256': s.canonical(_encode(p.cfg)),
            'buffers': {name: descriptor(value) for name, value in p.named_buffers()}}


def forward_reconstruct(p, raw, parameters, info, *, grid):
    """Continuum, Fourier unpack and measurement forward only; no solve."""
    from . import measurement as m
    from .observable_basis import RealFourierBasis
    import torch
    with torch.no_grad():
        z, xi, beta = (torch.as_tensor(parameters[name], device=p.M.device, dtype=p.M.dtype)
                       for name in PARAMETER_KEYS)
        C = m.continuum(p, z, grid=grid)
        u = RealFourierBasis(p.op.object_otf).unpack(xi)[None]
        out = m.output(p, p.raw_to_electrons(m.raw_tensor(raw, p)), C, u, beta, info)
        out.update(z_grid=z, xi=xi, beta=beta, continuum_latent=C)
    return out


class CompactJointCache:
    def __init__(self, report_root, data_root, *, p, identity, input_reference, calibration,
                 config, source_sha256=None, budget=None):
        self.report_root, self.data_root = (s.write_path(path) for path in (report_root, data_root))
        self.p = p
        self.input_reference = _encode(input_reference)
        self.config = _encode(config)
        extra = dict(calibration.get('source_sha256', {}))
        for name, digest in (source_sha256 or {}).items():
            if name in extra and extra[name] != digest:
                raise CacheIntegrityError('Calibration and caller source digests conflict: '+name)
            extra[name] = digest
        for cls in (type(p), type(p.op)):
            class_source = Path(inspect.getfile(cls)).resolve()
            if class_source == s.ROOT or s.ROOT in class_source.parents:
                name = class_source.relative_to(s.ROOT).as_posix()
                digest = s.sha(class_source)
                if name in extra and extra[name] != digest:
                    raise CacheIntegrityError('Active physics class differs from the declared source: '+name)
                extra[name] = digest
        self.sources = source_hashes(extra)
        required = ('registration_sha256', 'method', 'case_id')
        if any(not isinstance(identity.get(k), str) or not identity[k] for k in required):
            raise ValueError('Registered method and case identity are required')
        if not calibration.get('fingerprint') or not calibration.get('physical_objective_version'):
            raise ValueError('An explicit calibrated physical objective identity is required')
        if str(p.M.dtype) != 'torch.float64':
            raise ValueError('Compact joint cache requires native float64 physics')
        input_path = Path(input_reference['path'])
        if not input_path.is_absolute():
            input_path = s.LEGACY/input_path
        if input_path.resolve().drive.upper() not in ('E:', 'D:'):
            raise ValueError('Input references must remain on E/D')
        self.input_file = {'path': str(input_path.resolve()), 'sha256': input_reference['sha256']}
        from .observable_basis import RealFourierBasis
        self.plan = frame_plan(shape=p.cfg['shape'], bands=p.M.shape[0],
                               template_dimension=RealFourierBasis(p.op.object_otf).dimension,
                               grid=config.get('grid'))
        self.context = {'schema': SCHEMA, 'identity': _encode(identity), 'input_reference': self.input_reference,
                        'input_file': self.input_file, 'calibration': _encode(calibration), 'config': self.config,
                        'source_sha256': self.sources, 'runtime': runtime_identity(p),
                        'physics': _physics_identity(p), 'frame_plan': self.plan,
                        'canonical_schema': list(CANONICAL_KEYS),
                        'representation_verification_is_not_numeric_certification': True,
                        'later_mismatch_cannot_fabricate_original_fallback': True}
        self.fingerprint = s.canonical(self.context)
        if len(_text(self.context).encode('utf8')) > self.plan['metadata_bound_bytes']//8:
            raise CacheIntegrityError('Registered cache context exceeds its conservative metadata envelope')
        self.key = s.canonical({'identity': identity, 'raw_sha256': input_reference['sha256'],
                                'calibration': calibration['fingerprint'], 'config': self.config})
        self.report = self.report_root/'cells'/self.key
        self.data = self.data_root/'cells'/self.key
        self.budget = budget or Budget((self.report_root, self.data_root))
        self.owner = None

    def _verify_context(self, raw):
        verify_sources(self.sources)
        verify_file(self.input_file)
        if runtime_identity(self.p) != self.context['runtime'] or _physics_identity(self.p) != self.context['physics']:
            raise CacheIntegrityError('Same runtime, physics buffers, device and dtype are required for reconstruction')
        raw_array = array(raw)
        shape = tuple(self.p.cfg['shape'])
        if raw_array.shape == (1, *shape):
            raw_array = raw_array[0]
        if raw_array.shape != shape or not np.isfinite(raw_array).all():
            raise CacheIntegrityError('Finite registered raw frame required')
        restored = raw_array.astype(np.dtype(self.input_reference['dtype']))
        if not np.array_equal(restored.astype(raw_array.dtype), raw_array):
            raise CacheIntegrityError('Original camera dtype roundtrip is not lossless')
        if array_sha(restored) != self.input_reference['array_sha256']:
            raise CacheIntegrityError('Input array dtype/shape/bytes hash mismatch')
        return array_sha(raw_array)

    def _check_record(self, value):
        if value.get('context_fingerprint') != self.fingerprint or value.get('cache_key') != self.key:
            raise CacheIntegrityError('Cache method/input/calibration/source/config/runtime identity changed')
        return value

    def _record(self, **values):
        return {'schema': SCHEMA, 'cache_key': self.key, 'context_fingerprint': self.fingerprint,
                'time': s.now(), **values}

    def _read(self, name):
        return self._check_record(_read_sealed(self.report/name))

    def inspect(self):
        for name in ('RECONSTRUCTION_FAILURE.json', 'RESULT.json', 'FAILURE.json'):
            if (self.report/name).exists():
                return self._read(name)
        if (self.report/'STARTED.json').exists():
            self._read('STARTED.json')
            return self._record(status='INTERRUPTED_OR_PARAMETERS_PENDING', run_new_inverse=False, score_allowed=False)
        return self._record(status='NEW', run_new_inverse=False, score_allowed=False)

    def begin(self, raw):
        """Issue one ticket after dense fallback reservation; never rerun."""
        self._verify_context(raw)
        previous = self.inspect()
        if previous['status'] != 'NEW':
            return {**previous, 'run_new_inverse': False}
        before = self.budget.check(self.plan['initial_safe_reserve_bytes'])
        self.owner = {'pid': os.getpid(), 'token': uuid.uuid4().hex}
        reserve_path = self.data/('FALLBACK_RESERVED_'+self.owner['token']+'.bin')
        started = self._record(status='STARTED', owner=self.owner, context=self.context,
                               raw_runtime_array_sha256=self._verify_context(raw), budget_before=before,
                               reserve_path=str(reserve_path), reserve_bytes=self.plan['fallback_zip_bound_bytes'],
                               run_new_inverse=False, score_allowed=False)
        try:
            _write_once(self.report/'STARTED.json', started)
        except FileExistsError:
            return {**self.inspect(), 'run_new_inverse': False}
        reserve_path = s.write_path(reserve_path)
        reserve_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with reserve_path.open('xb') as handle:
                remaining = self.plan['fallback_zip_bound_bytes']
                block = bytes(64*2**10)
                while remaining:
                    size = min(len(block), remaining)
                    handle.write(block[:size])
                    remaining -= size
                handle.flush()
                os.fsync(handle.fileno())
            _write_once(self.report/'RESERVATION.json', self._record(owner=self.owner, reservation=file_ref(reserve_path)))
            self.budget.check(0)
        except Exception as error:
            self._fail('FAIL_CLOSED_RESERVATION', error=error, original_fallback_available=False)
            raise
        return self._record(status='NEW_INVERSE_TICKET', owner=self.owner, run_new_inverse=True,
                            score_allowed=False, reservation_bytes=self.plan['fallback_zip_bound_bytes'])

    def _own_reservation(self, owner):
        started, reserved = self._read('STARTED.json'), self._read('RESERVATION.json')
        if owner != started['owner'] or owner != reserved['owner']:
            raise CacheIntegrityError('Only this cache instance owner may use or release its reservation')
        expected = self.data/('FALLBACK_RESERVED_'+owner['token']+'.bin')
        path = _within(reserved['reservation']['path'], self.data)
        if path != expected.resolve():
            raise CacheIntegrityError('Reservation path does not belong to this owner')
        verify_file(reserved['reservation'])
        return path

    def release_reservation(self, owner):
        result = self._read('RESULT.json')
        if result.get('status') != 'LOSSLESS_READY' or not result.get('lossless_representation_verified'):
            raise CacheIntegrityError('A byte-verified compact result is required before releasing its full fallback reserve')
        path = self._own_reservation(owner)
        path.unlink()  # Only the newly allocated, hash-verified zero reservation.

    def _canonical(self, out):
        result = {name: array(out[name]) for name in CANONICAL_KEYS}
        shape = tuple(self.p.cfg['shape'])
        for name, value in result.items():
            if name in FLOAT_CANONICAL_KEYS and value.dtype != np.float64:
                raise CacheIntegrityError('Canonical arrays must preserve native float64: '+name)
            if name not in FLOAT_CANONICAL_KEYS and value.dtype != np.bool_:
                raise CacheIntegrityError('Canonical masks must preserve boolean dtype: '+name)
            expected = ((1, self.plan['bands'], *shape) if name == 'continuum_postblur_bands' else (1, *shape))
            if value.shape != expected:
                raise CacheIntegrityError('Canonical shape differs from registration: '+name)
        return result

    def _parameters(self, out):
        values = {name: array(out[name]) for name in PARAMETER_KEYS}
        expected = {'z_grid': (1, 2, self.plan['grid'], self.plan['grid']),
                    'xi': (self.plan['template_dimension'],), 'beta': (1, 3)}
        for name, value in values.items():
            if value.dtype != np.float64 or value.shape != expected[name]:
                raise CacheIntegrityError('Exact float64 latent parameter shape required: '+name)
        return values

    def _write_parameters(self, values, prepared):
        if (self.report/'PREPARED.json').exists():
            previous = self._read('PREPARED.json')
            for name in ('parameter_arrays', 'canonical_arrays', 'original_info', 'continuum_latent'):
                if _text(previous[name]) != _text(prepared[name]):
                    raise CacheIntegrityError('Previously prepared original endpoint differs: '+name)
        else:
            _write_once(self.report/'PREPARED.json', prepared)
        path = s.write_path(self.data/'PARAMETERS.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = s.arrays(path)
            if {k: descriptor(v) for k, v in existing.items()} != prepared['parameter_arrays']:
                raise CacheIntegrityError('Never overwrite different compact joint parameters')
        else:
            pending = path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
            with pending.open('xb') as handle:
                np.savez_compressed(handle, **values)
                handle.flush()
                os.fsync(handle.fileno())
            if pending.stat().st_size > self.plan['parameter_zip_bound_bytes']:
                raise CacheIntegrityError('Parameter ZIP exceeds its reserved native-precision bound')
            pending.replace(path)
        if not (self.report/'PARAMETERS.json').exists():
            _write_once(self.report/'PARAMETERS.json', self._record(parameter_cache=file_ref(path),
                        prepared=file_ref(self.report/'PREPARED.json')))

    def _load_parameters(self):
        prepared = self._read('PREPARED.json')
        if not (self.report/'PARAMETERS.json').exists():
            path = self.data/'PARAMETERS.npz'
            if not path.exists():
                candidates = list(self.data.glob('PARAMETERS.npz.pending.*'))
                if len(candidates) != 1:
                    raise CacheIntegrityError('No single recoverable parameter archive')
                candidate = candidates[0]
                arrays = s.arrays(candidate)
                self._parameters(arrays)
                if {name: descriptor(value) for name, value in arrays.items()} != prepared['parameter_arrays']:
                    raise CacheIntegrityError('Interrupted parameter archive differs from original bytes')
                candidate.replace(path)
            arrays = s.arrays(path)
            self._parameters(arrays)
            if {name: descriptor(value) for name, value in arrays.items()} != prepared['parameter_arrays']:
                raise CacheIntegrityError('Unreceipted parameter archive differs from original bytes')
            _write_once(self.report/'PARAMETERS.json', self._record(parameter_cache=file_ref(path),
                        prepared=file_ref(self.report/'PREPARED.json'), recovered_receipt=True))
        receipt = self._read('PARAMETERS.json')
        verify_file(receipt['prepared'])
        path = verify_file(receipt['parameter_cache'])
        values = s.arrays(path)
        self._parameters(values)
        if {name: descriptor(value) for name, value in values.items()} != prepared['parameter_arrays']:
            raise CacheIntegrityError('Read-back latent parameter bytes differ from the original')
        return values, prepared, receipt

    def _verify_forward(self, raw):
        values, prepared, receipt = self._load_parameters()
        out = forward_reconstruct(self.p, raw, values, prepared['original_info'], grid=self.plan['grid'])
        out['numerical_valid'] = prepared['original_numerical_valid']
        actual = {name: descriptor(value) for name, value in self._canonical(out).items()}
        derived = descriptor(out['continuum_latent'])
        matches = {name: actual[name] == expected for name, expected in prepared['canonical_arrays'].items()}
        matches['continuum_latent'] = derived == prepared['continuum_latent']
        return out, prepared, receipt, {'per_array_byte_identical': matches, 'all_byte_identical': all(matches.values()),
            'reconstructed_canonical_arrays': actual, 'reconstructed_continuum_latent': derived,
            'forward_only': True, 'complete_inverse_calls': 0, 'optimizer_calls': 0, 'linear_solver_calls': 0}

    def _fallback(self, arrays, owner):
        path = self._own_reservation(owner)
        with path.open('r+b') as handle:
            handle.seek(0)
            np.savez(handle, **arrays)
            end = handle.tell()
            if end > self.plan['fallback_zip_bound_bytes']:
                raise CacheIntegrityError('Full canonical fallback exceeded its dense reserved bound')
            handle.truncate(end)
            handle.flush()
            os.fsync(handle.fileno())
        final = self.data/'FALLBACK.npz'
        if final.exists():
            raise FileExistsError('Never overwrite a full fallback')
        path.replace(final)
        recovered = s.arrays(final)
        if {name: descriptor(value) for name, value in recovered.items()} != {name: descriptor(value) for name, value in arrays.items()}:
            raise CacheIntegrityError('Full fallback read-back failed its original byte hashes')
        return file_ref(final)

    def _fail(self, status, *, error=None, **values):
        receipt = self._record(status=status, run_new_inverse=False, score_allowed=False,
                               lossless_representation_verified=False, **values)
        if error is not None:
            receipt.update(error_type=type(error).__name__, error=str(error))
        name = 'RECONSTRUCTION_FAILURE.json' if (self.report/'RESULT.json').exists() else 'FAILURE.json'
        if (self.report/name).exists():
            return self._read(name)
        _write_once(self.report/name, receipt)
        return receipt

    def persist(self, ticket, out, raw):
        """Parameters first, same-p/device read-back next; fallback on mismatch."""
        previous = self.inspect()
        if (self.report/'RESULT.json').exists() or (self.report/'FAILURE.json').exists():
            self._verify_context(raw)
            return previous
        started = self._read('STARTED.json')
        if not ticket.get('run_new_inverse') or ticket.get('owner') != started['owner']:
            raise CacheIntegrityError('A valid once-only owner ticket is required to persist an inverse endpoint')
        self._check_record(ticket)
        # Keep original arrays available for a full fallback even when their
        # dtype or declared parameter contract turns out to be invalid.
        canonical = {name: array(out[name]) for name in CANONICAL_KEYS}
        original_parameters = {name: array(out[name]) for name in PARAMETER_KEYS}
        prepared = self._record(context=self.context, original_info=out['info'],
            original_numerical_valid=bool(out.get('numerical_valid', False)),
            original_finite_canonical=all(bool(np.isfinite(v).all()) for k, v in canonical.items() if k in FLOAT_CANONICAL_KEYS),
            canonical_arrays={name: descriptor(value) for name, value in canonical.items()},
            continuum_latent=descriptor(out['continuum_latent']),
            parameter_arrays={name: descriptor(value) for name, value in original_parameters.items()},
            original_device_dtype=self.context['runtime'], truth_loaded=False)
        try:
            self._verify_context(raw)
            self._canonical(out)
            values = self._parameters(out)
            # The original info, all canonical hashes, and exact parameters are
            # durable before any canonical fallback or score is written.
            metadata_exceeded = len(_text(prepared).encode('utf8')) > self.plan['metadata_bound_bytes']//2
            if metadata_exceeded:
                # Preserve the original info if the full global/floor bounds
                # permit it; the declared representation envelope still fails.
                self.budget.check(self.plan['parameter_zip_bound_bytes']+len(_text(prepared).encode('utf8'))+64*2**10)
            self._write_parameters(values, prepared)
            if metadata_exceeded:
                fallback = self._fallback(canonical, started['owner'])
                return self._fail('FAIL_CLOSED_METADATA_ENVELOPE', original_fallback_available=True,
                    fallback=fallback, original_canonical_arrays=prepared['canonical_arrays'])
            rebuilt, original, receipt, verification = self._verify_forward(raw)
            if not verification['all_byte_identical']:
                fallback = self._fallback(canonical, started['owner'])
                return self._fail('FAIL_CLOSED_INITIAL_BYTE_MISMATCH', original_fallback_available=True,
                    fallback=fallback, verification=verification, original_canonical_arrays=prepared['canonical_arrays'])
            self.budget.check(0)
            result = self._record(status='LOSSLESS_READY', run_new_inverse=False,
                score_allowed=original['original_finite_canonical'], lossless_representation_verified=True,
                numerical_certification=original['original_numerical_valid'],
                original_fallback_available=False, fallback_retained=False,
                representation_scope='same registered runtime/device; each future load must independently pass all byte hashes',
                write_read_same_physics_instance=True, verification=verification,
                parameter_receipt=file_ref(self.report/'PARAMETERS.json'),
                complete_inverse_calls_by_cache=0, native_GPU_qualification_claimed=False)
            _write_once(self.report/'RESULT.json', result)
            try:
                self.release_reservation(started['owner'])
            except Exception as error:
                _write_once(self.report/'RESERVATION_RELEASE_FAILURE.json', self._record(
                    status='OWN_RESERVATION_NOT_RELEASED', error_type=type(error).__name__, error=str(error)))
                result = {**result, 'reservation_release_error': str(error)}
            return result
        except Exception as error:
            parameter_error = None
            try:
                self._write_parameters(original_parameters, prepared)
            except Exception as persistence_error:
                parameter_error = {'error_type': type(persistence_error).__name__, 'error': str(persistence_error)}
            if (self.data/'FALLBACK.npz').exists():
                fallback = file_ref(self.data/'FALLBACK.npz')
            else:
                fallback = self._fallback(canonical, started['owner'])
            return self._fail('FAIL_CLOSED_INITIAL_PERSISTENCE_OR_FORWARD', error=error,
                original_fallback_available=True, fallback=fallback,
                original_parameter_persistence_error=parameter_error,
                original_canonical_arrays=prepared['canonical_arrays'])

    def resume(self, raw):
        """Recover only from persisted parameters; no callback can run an inverse."""
        self._verify_context(raw)
        state = self.inspect()
        if state['status'].startswith('FAIL_CLOSED'):
            if state.get('fallback'):
                verify_file(state['fallback'])
            return state
        if state['status'] == 'NEW':
            return state
        if (self.data/'FALLBACK.npz').exists():
            prepared = self._read('PREPARED.json')
            arrays = s.arrays(self.data/'FALLBACK.npz')
            if {name: descriptor(value) for name, value in arrays.items()} != prepared['canonical_arrays']:
                raise CacheIntegrityError('Unreceipted original fallback differs from its original byte descriptors')
            return self._fail('FAIL_CLOSED_RECOVERED_FULL_ORIGINAL_FALLBACK', original_fallback_available=True,
                fallback=file_ref(self.data/'FALLBACK.npz'), original_canonical_arrays=prepared['canonical_arrays'])
        try:
            out, original, receipt, verification = self._verify_forward(raw)
            if not verification['all_byte_identical']:
                return self._fail('FAIL_CLOSED_LATER_BYTE_MISMATCH_NO_ORIGINAL_FALLBACK',
                    verification=verification, original_fallback_available=False,
                    original_canonical_unavailable_after_compaction=True,
                    cannot_fabricate_or_approximate_fallback=True)
            if not (self.report/'RESULT.json').exists():
                result = self._record(status='LOSSLESS_READY', run_new_inverse=False,
                    score_allowed=original['original_finite_canonical'], lossless_representation_verified=True,
                    numerical_certification=original['original_numerical_valid'],
                    original_fallback_available=False, recovered_after_interruption=True,
                    verification=verification, complete_inverse_calls_by_cache=0,
                    native_GPU_qualification_claimed=False)
                _write_once(self.report/'RESULT.json', result)
                owner = self._read('STARTED.json')['owner']
                self.release_reservation(owner)
            return self._read('RESULT.json')
        except Exception as error:
            return self._fail('FAIL_CLOSED_RECOVERY_WITHOUT_ORIGINAL_CANONICAL', error=error,
                original_fallback_available=False, original_canonical_unavailable_after_compaction=True,
                cannot_fabricate_or_approximate_fallback=True)

    def load_for_score(self, raw):
        status = self.resume(raw)
        if status['status'] != 'LOSSLESS_READY' or not status['score_allowed']:
            raise CacheNotScoreable(status)
        try:
            out, original, receipt, verification = self._verify_forward(raw)
        except Exception as error:
            failure = self._fail('FAIL_CLOSED_LATER_FORWARD_ERROR_NO_ORIGINAL_FALLBACK', error=error,
                original_fallback_available=False, original_canonical_unavailable_after_compaction=True,
                cannot_fabricate_or_approximate_fallback=True)
            raise CacheNotScoreable(failure) from error
        if not verification['all_byte_identical']:
            failure = self._fail('FAIL_CLOSED_LATER_BYTE_MISMATCH_NO_ORIGINAL_FALLBACK',
                verification=verification, original_fallback_available=False,
                original_canonical_unavailable_after_compaction=True, cannot_fabricate_or_approximate_fallback=True)
            raise CacheNotScoreable(failure)
        return out

    def record_score(self, score_id, value, raw):
        """External evaluator supplies an already computed score, never truth."""
        self.load_for_score(raw)
        if not isinstance(score_id, str) or not score_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in score_id):
            raise ValueError('A simple registered external score identifier is required')
        path = self.report/('SCORE_'+score_id+'.json')
        if path.exists():
            return self._read(path.name)
        result = self._record(status='EXTERNAL_SCORE_SAVED', score_id=score_id, score=value,
                              inference_calls=0, truth_loaded_by_cache=False)
        score_bytes = len(_text(result).encode('utf8'))+4096
        case_ledger = tree_ledger((self.report, self.data))
        if case_ledger['by_kind_bytes']['other_data_meta_cache_log']+score_bytes > self.plan['metadata_bound_bytes']:
            raise CacheIntegrityError('External score exceeds the remaining per-frame metadata envelope; inverse cache retained')
        self.budget.check(score_bytes)
        _write_once(path, result)
        return result

    def load_fallback(self):
        status = self.inspect()
        if not status.get('original_fallback_available') or not status.get('fallback'):
            raise CacheNotScoreable(status)
        values = s.arrays(verify_file(status['fallback']))
        if {name: descriptor(value) for name, value in values.items()} != status['original_canonical_arrays']:
            raise CacheIntegrityError('Original fallback byte descriptors changed')
        return values  # Diagnostic preservation only; status.score_allowed remains false.
