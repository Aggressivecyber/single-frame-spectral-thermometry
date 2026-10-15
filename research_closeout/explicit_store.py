"""Explicit owner-bound D storage for newly registered repair work only.

The historical E whitelist remains unchanged. This adapter never migrates,
converts, cleans up, or reads labels. Every real write still requires the E
2 GiB floor, independently of D's 128 MiB free-space floor. The D project
includes all tasks below the two fixed prefixes, capped together at 256 MiB.
"""
from pathlib import Path
import json
import os
import re
import shutil
import stat
import uuid

from . import storage as s


SCHEMA = 'explicit-D-repair-store-v001'
PROJECT = Path('D:/CodexWork/thermometry/closeout_001')
DATA = PROJECT/'outputs/repair_001'
REPORTS = PROJECT/'reports/repair_001'
GLOBAL_LIMIT_BYTES = 4*2**30
D_PROJECT_LIMIT_BYTES = 256*2**20
D_FREE_FLOOR_BYTES = 128*2**20
E_FREE_FLOOR_BYTES = 2*2**30
OWNER_FILE = 'EXPLICIT_STORE_OWNER.json'
SOURCE = 'research_closeout/explicit_store.py'


class StoreIntegrityError(RuntimeError):
    pass


def _inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def _checked_path(path, roots):
    """Require absolute D paths, fixed resolved roots, and no reparse points."""
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.drive.upper() != 'D:':
        raise StoreIntegrityError('Explicit D storage requires an absolute D path')
    devices = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
               *(f'LPT{i}' for i in range(1, 10))}
    for component in candidate.parts[1:]:
        if (component in ('.', '..') or component.endswith((' ', '.'))
                or any(ord(char) < 32 or char in '<>:"|?*' for char in component)
                or component.split('.')[0].upper() in devices):
            raise StoreIntegrityError('Ambiguous Windows path or alternate data stream is forbidden')
    # Check lexical ancestry as well as the resolved result: links cannot
    # launder an outside path into the permitted task, or vice versa.
    lexical = Path(os.path.abspath(candidate))
    resolved = candidate.resolve()
    fixed = tuple(Path(root).resolve() for root in roots)
    if not _inside(lexical, fixed) or not _inside(resolved, fixed):
        raise StoreIntegrityError('Write escapes the explicit D task roots: '+str(candidate))
    for part in (candidate, *candidate.parents):
        if part.exists() or part.is_symlink():
            attrs = getattr(part.lstat(), 'st_file_attributes', 0)
            if part.is_symlink() or attrs & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024):
                raise StoreIntegrityError('Reparse points are forbidden in explicit D storage: '+str(part))
    return resolved


class ExplicitBudget:
    """Four-root accounting; synthetic free values are for arithmetic tests only."""
    def __init__(self, instance_roots, *, instance_limit_bytes=D_PROJECT_LIMIT_BYTES,
                 free_bytes=None):
        self.instance_roots = tuple(Path(p).resolve() for p in instance_roots)
        self.global_roots = (s.DATA.resolve(), s.REPORTS.resolve(), DATA.resolve(), REPORTS.resolve())
        self.d_project_roots = (DATA.resolve(), REPORTS.resolve())
        self.instance_limit = int(instance_limit_bytes)
        if not 0 <= self.instance_limit <= D_PROJECT_LIMIT_BYTES:
            raise ValueError('Instance limit must fit the fixed 256 MiB D project cap')
        self.global_limit = GLOBAL_LIMIT_BYTES
        self.d_project_limit = D_PROJECT_LIMIT_BYTES
        self.floor = D_FREE_FLOOR_BYTES
        self.e_floor = E_FREE_FLOOR_BYTES
        self._free_bytes = free_bytes

    def ledger(self):
        from .compact_joint_cache import tree_ledger
        return {'instance': tree_ledger(self.instance_roots),
                'D_project': tree_ledger(self.d_project_roots),
                'global_new_study': tree_ledger(self.global_roots),
                'instance_limit_bytes': self.instance_limit,
                'D_project_limit_bytes': self.d_project_limit,
                'global_limit_bytes': self.global_limit,
                'D_free_floor_bytes': self.floor, 'E_free_floor_bytes': self.e_floor,
                'global_budget_roots': [str(p) for p in self.global_roots]}

    def _free(self, drive):
        if self._free_bytes is None:
            return shutil.disk_usage(s.ROOT if drive == 'E' else PROJECT.anchor).free
        values = self._free_bytes() if callable(self._free_bytes) else self._free_bytes
        return int(values[drive])

    def check(self, requested_bytes=0):
        requested = int(requested_bytes)
        if requested < 0:
            raise ValueError('A nonnegative D reservation is required')
        ledger = self.ledger()
        e_free, d_free = self._free('E'), self._free('D')
        for name, limit in (('global_new_study', self.global_limit),
                            ('D_project', self.d_project_limit), ('instance', self.instance_limit)):
            if ledger[name]['bytes']+requested > limit:
                raise StoreIntegrityError(name+' byte budget would be exceeded')
        if e_free < self.e_floor:
            raise StoreIntegrityError('E free space is below its independent 2 GiB floor')
        if d_free-requested < self.floor:
            raise StoreIntegrityError('D reserve would cross its independent 128 MiB floor')
        return {**ledger, 'free_bytes': d_free, 'E_free_bytes': e_free, 'D_free_bytes': d_free,
                'additional_reserved_bytes': requested, 'free_after_reserve_bytes': d_free-requested,
                'E_free_after_D_reserve_bytes': e_free}


class ExplicitDStore:
    """One fixed task's D roots, owned by a stable caller-retained token.

    Construction and contract inspection are read-only. First write enrolls
    exclusive owner markers only after all real disk gates pass. Existing
    tasks require the original token; nonempty unowned directories fail closed.
    No test free-space override is accepted by this store.
    """
    def __init__(self, task_name, *, owner_token=None):
        if not isinstance(task_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', task_name):
            raise ValueError('A simple task name of at most 64 characters is required')
        if task_name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
            raise ValueError('Windows device names cannot be task names')
        token = uuid.uuid4().hex if owner_token is None else owner_token
        if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', token):
            raise ValueError('Retain an owner token containing 16 to 128 safe characters')
        self.task_name, self.owner_token = task_name, token
        self.data_root = _checked_path(DATA/task_name, (DATA,))
        self.report_root = _checked_path(REPORTS/task_name, (REPORTS,))
        self.roots = (self.data_root, self.report_root)
        self._source_sha256 = s.sha(s.ROOT/SOURCE)
        self._owner = {'schema': SCHEMA, 'task_name': task_name, 'owner_token': token,
                       'data_root': str(self.data_root), 'report_root': str(self.report_root),
                       'source_sha256': {SOURCE: self._source_sha256}}
        self._inspect_owners(allow_missing=True)

    def contract(self):
        return {**self._owner, 'fixed_D_project_roots': [str(DATA), str(REPORTS)],
                'global_budget_roots': [str(p) for p in (s.DATA, s.REPORTS, DATA, REPORTS)],
                'global_limit_bytes': GLOBAL_LIMIT_BYTES, 'D_project_limit_bytes': D_PROJECT_LIMIT_BYTES,
                'D_free_floor_bytes': D_FREE_FLOOR_BYTES, 'E_free_floor_bytes': E_FREE_FLOOR_BYTES,
                'E_free_floor_root': str(s.ROOT), 'D_free_floor_root': str(PROJECT.anchor),
                'default_E_whitelist_unchanged': True, 'storage_mode': 'explicit-D-repair',
                'owner_markers_required': True, 'migration_cleanup_or_conversion_supported': False,
                'preexisting_dense_reserve_does_not_bypass_live_E_or_D_floor': True,
                'unexpected_floor_drop_requires_stop_and_explicit_missing_persistence_receipt': True,
                'all_data_meta_fallback_pending_reserve_cache_log_bytes_counted': True,
                'constructor_is_read_only': True}

    def budget(self, *, instance_roots=None, instance_limit_bytes=D_PROJECT_LIMIT_BYTES):
        roots = self.roots if instance_roots is None else tuple(instance_roots)
        roots = tuple(_checked_path(p, self.roots) for p in roots)
        return ExplicitBudget(roots, instance_limit_bytes=instance_limit_bytes)

    def _inspect_owners(self, *, allow_missing=False):
        if s.sha(s.ROOT/SOURCE) != self._source_sha256:
            raise StoreIntegrityError('Explicit D storage source changed after construction')
        for root in self.roots:
            _checked_path(root, self.roots)
            marker = root/OWNER_FILE
            if marker.exists():
                _checked_path(marker, self.roots)
                if s.read(marker) != self._owner:
                    raise StoreIntegrityError('Explicit D task belongs to another owner or source version')
            elif root.exists() and any(root.iterdir()):
                raise StoreIntegrityError('Refusing a nonempty explicit D directory without its owner marker')
            elif not allow_missing:
                raise StoreIntegrityError('Explicit D owner marker is missing')

    def _enroll(self):
        self._inspect_owners(allow_missing=True)
        encoded = self._json(self._owner)
        missing = [root for root in self.roots if not (root/OWNER_FILE).exists()]
        self.budget().check(len(encoded)*len(missing))
        for root in missing:
            _checked_path(root, self.roots).mkdir(parents=True, exist_ok=True)
            marker = _checked_path(root/OWNER_FILE, self.roots)
            try:
                with marker.open('xb') as handle:
                    handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
            except FileExistsError:
                if s.read(marker) != self._owner:
                    raise StoreIntegrityError('Another owner enrolled this D task concurrently')
        self._inspect_owners()

    def write_path(self, path):
        approved = _checked_path(path, self.roots)
        if OWNER_FILE.casefold() in (part.casefold() for part in approved.parts):
            raise StoreIntegrityError('Owner markers are immutable storage metadata')
        self._enroll()
        return approved

    @staticmethod
    def _json(value):
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8')

    def write(self, path, value, *, replace=False):
        path = self.write_path(path)
        encoded = self._json(value)
        self.budget().check(len(encoded))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not replace:
            with path.open('xb') as handle:
                handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        else:
            pending = self.write_path(path.with_name(path.name+'.pending.'+uuid.uuid4().hex))
            with pending.open('xb') as handle:
                handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
            self.replace(pending, path, allow_existing=True)
        return path

    def append(self, path, value):
        path = self.write_path(path)
        encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n').encode('utf8')
        self.budget().check(len(encoded))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab') as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        return path

    def save(self, path, **values):
        import io
        import numpy as np
        path = self.write_path(path)
        arrays = {k: v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v) for k, v in values.items()}
        if any(v.dtype.hasobject for v in arrays.values()):
            raise StoreIntegrityError('Object/pickle arrays are forbidden in the explicit D store')
        encoded = io.BytesIO()
        np.savez_compressed(encoded, **arrays)
        payload = encoded.getbuffer()
        self.budget().check(payload.nbytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        return {'path': str(path), 'sha256': s.sha(path), 'bytes': path.stat().st_size}

    def replace(self, source, target, *, allow_existing=False):
        source, target = self.write_path(source), self.write_path(target)
        if not source.is_file() or (target.exists() and not allow_existing):
            raise StoreIntegrityError('Owned replacement requires its source and an unused target')
        source.replace(target)
        return target

    def release_reservation(self, path, *, owner):
        """Release only this task's hash-checked compact-instance placeholder."""
        from .compact_joint_cache import _read_sealed, file_ref
        path = self.write_path(path)
        token = owner.get('token')
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{32}', token):
            raise StoreIntegrityError('A compact cache instance owner is required')
        if path.name != 'FALLBACK_RESERVED_'+token+'.bin':
            raise StoreIntegrityError('Reservation filename does not belong to the instance owner')
        relative = path.relative_to(self.data_root)
        if len(relative.parts) != 3 or relative.parts[0] != 'cells':
            raise StoreIntegrityError('Only a compact cache instance reservation can be released')
        report = _checked_path(self.report_root/'cells'/relative.parts[1], self.roots)
        for name in ('STARTED.json', 'RESERVATION.json', 'RESULT.json'):
            _checked_path(report/name, self.roots)
            _checked_path(report/(name+'.sha256.json'), self.roots)
        started, reserved, result = (_read_sealed(report/name) for name in
                                     ('STARTED.json', 'RESERVATION.json', 'RESULT.json'))
        records = (started, reserved, result)
        if (started.get('owner') != owner or reserved.get('owner') != owner
                or reserved.get('reservation') != file_ref(path)
                or any(record.get('cache_key') != relative.parts[1] for record in records)
                or not started.get('context_fingerprint')
                or any(record.get('context_fingerprint') != started['context_fingerprint'] for record in records)
                or result.get('status') != 'LOSSLESS_READY'
                or not result.get('lossless_representation_verified')):
            raise StoreIntegrityError('Only the byte-verified instance owner may release its reserve')
        path.unlink()

    read = staticmethod(s.read)
    sha = staticmethod(s.sha)
    arrays = staticmethod(s.arrays)
