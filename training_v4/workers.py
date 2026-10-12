"""Matched cache jobs, with bounded independent scalar frames in four threads."""
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import torch
import runtime
from infer import load_model
from training_v4.common import OUT, DATA, setup, read, write
from training_v4.cache import cache_row, cache_identity
from evaluation_v2.common import Journal


class CachePool:
    def __init__(self, rows, selection, *, workers=4):
        self.selection = selection
        self.physics = load_model(device='cuda').physics
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.journal = Journal(OUT/'cache_per_case.jsonl')
        self.lock, self.done, self.total = threading.Lock(), 0, len(rows)
        self.futures = {}
        for row in rows:
            key = self.key(row)
            if key not in self.futures:
                self.futures[key] = self.pool.submit(self.work, row)

    def key(self, row):
        return cache_identity(row['raw_hash'], self.selection['physics_config'], row['calibration_sha256'],
                              self.selection['method'], self.selection['solver_config'])

    def work(self, row):
        try:
            _, _, meta = cache_row(self.physics, row, self.selection['method'], self.selection['solver_config'])
            result = {'identity_sha256': self.key(row), 'case_id': row['case_id'], 'split': row['split'],
                      'raw_hash': row['raw_hash'], 'cache_key': meta['cache_key'],
                      'cache_sha256': meta['cache_sha256'], 'status': 'SUCCESS',
                      'seconds_original_compute': meta['seconds'], 'labels_used': False}
        except Exception as e:
            result = {'identity_sha256': self.key(row), 'case_id': row['case_id'], 'split': row['split'],
                      'raw_hash': row['raw_hash'], 'status': 'EXCEPTION',
                      'exception_type': type(e).__name__, 'message': str(e)}
            with self.lock:
                self.journal.append(result)
            raise
        with self.lock:
            if not self.journal.succeeded(result['identity_sha256']):
                self.journal.append(result)
            self.done += 1
            if self.done % 16 == 0 or self.done == self.total:
                print({'cache_completed': self.done, 'cache_total': self.total,
                       'split': row['split'], 'last_case': row['case_id']}, flush=True)
        return meta

    def ensure(self, row):
        return self.futures[self.key(row)].result()

    def finish(self):
        for future in self.futures.values():
            future.result()
        self.pool.shutdown(wait=True)
        write(OUT/'cache_receipt.json', {'unique_raw_keys': len(self.futures), 'cases': self.total,
            'completed': self.done, 'labels_used': False, 'cache_key_includes_raw_physics_calibration': True,
            'raw_noise_change_invalidates_cache': True, 'parallel_cross_frame_CG_used': False,
            'engine': 'original v3 scalar functions, memoized fixed Jacobian'})


def cache_all():
    setup()
    manifest = read(OUT/'data_manifest.json')
    selection = read(OUT/'backbone_selection.json')
    rows = sorted(manifest['rows'], key=lambda r: (r['split'] != 'validation', r['mother_id']))
    pool = CachePool(rows, selection)
    pool.finish()
