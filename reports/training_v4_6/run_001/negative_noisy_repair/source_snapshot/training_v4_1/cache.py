"""Read-only reuse of v4 caches; new raw inversions in a new artifact directory."""
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import runtime
from infer import load_model
from training_v4.cache import cache_identity
from training_v4.scalar_engine import invert_single
from training_v4.model import physical_output
from training_v4_1.common import *

def key_for(row, selection):
    return cache_identity(row['raw_hash'], selection['physics_config'], row['calibration_sha256'],
                          selection['method'], selection['solver_config'])

def cache_location(key):
    for root in (OLD_DATA, DATA):
        path = root/'cache'/key[:2]/(key+'.npz')
        if path.with_suffix('.json').exists(): return path
    return DATA/'cache'/key[:2]/(key+'.npz')

def materialize(p, row, selection):
    assert runtime.sha256(file_path(row, 'raw_path')) == row['raw_file_sha256']
    raw_array = npz(file_path(row, 'raw_path'))['raw_dn']
    assert array_hash(raw_array) == row['raw_hash']
    key = key_for(row, selection); path = cache_location(key)
    reused = path.with_suffix('.json').exists()
    raw = torch.as_tensor(raw_array.astype(np.float32), device='cuda')[None]
    raw_e = p.raw_to_electrons(raw)
    if reused:
        meta = read(path.with_suffix('.json'))
        assert meta['raw_hash'] == row['raw_hash'] and meta['cache_key'] == key
        assert meta['solver_config_sha256'] == canonical_hash(selection['solver_config'])
        assert runtime.sha256(path) == meta['cache_sha256']
        arr = npz(path)
        z, beta = torch.as_tensor(arr['state'], device='cuda')[None], torch.as_tensor(arr['beta'], device='cuda')[None]
    else:
        z, beta, history, seconds = invert_single(p, raw_e, selection['method'], selection['solver_config'])
        save_npz(path, state=z[0].cpu().numpy(), beta=beta[0].cpu().numpy())
        meta = {'cache_key': key, 'raw_hash': row['raw_hash'], 'cache_sha256': runtime.sha256(path),
            'physics_config_sha256': row['physics_config_sha256'], 'calibration_sha256': row['calibration_sha256'],
            'solver_config_sha256': canonical_hash(selection['solver_config']), 'labels_used': False,
            'method': selection['method'], 'seconds': seconds, 'history': history[0],
            'engine_sha256': runtime.sha256(runtime.ROOT/'training_v4/scalar_engine.py')}
        write(path.with_suffix('.json'), meta)
    # Truth is used only for this independent scoring sidecar, never inversion.
    score_path = DATA/'physical_scores'/(row['case_id']+'.json')
    identity = canonical_hash({'key': key, 'label': row['label_sha256'], 'scores_schema': 'v4_1_output_MSE'})
    if score_path.exists():
        score = read(score_path); assert score['identity'] == identity
    else:
        target = npz(file_path(row, 'label_path'))
        with torch.no_grad():
            base = physical_output(p, z, beta, raw_e)
            S = torch.as_tensor(target['template_e'], device='cuda')[None]
            T = torch.as_tensor(target['temperature_color_K'], device='cuda')[None]
            score = {'identity': identity, 'case_id': row['case_id'], 'raw_hash': row['raw_hash'],
                'label_sha256': row['label_sha256'], 'cache_key': key, 'cache_path': rel(path),
                'cache_sha256': meta['cache_sha256'], 'cache_metadata_sha256': runtime.sha256(path.with_suffix('.json')),
                'template_MSE_e2': float((base['template_e']-S).square().mean()),
                'temperature_MSE_K2': float((base['temperature_color_K']-T).square().mean()),
                'weighted_raw_RMS': float((base['raw_residual_e'].square()/(raw_e.clamp_min(0)+p.extra_variance)).mean().sqrt()),
                'truth_used_for_scoring_only': True}
        write(score_path, score)
    return {**score, 'reused_v4_cache': reused and path.is_relative_to(OLD_DATA),
            'labels_used_in_inversion': False, 'status': 'SUCCESS'}

class CachePool:
    def __init__(self, rows, selection):
        self.selection = selection; self.p = load_model(device='cuda').physics
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock(); self.done = 0; self.total = len(rows)
        self.journal = Journal(OUT/'cache_per_case.jsonl')
        self.futures = {row['case_id']: self.executor.submit(self.work, row) for row in rows}
    def work(self, row):
        try: result = materialize(self.p, row, self.selection)
        except Exception as exc:
            with self.lock: self.journal.append({'case_id': row['case_id'], 'raw_hash': row['raw_hash'],
                'status': 'EXCEPTION', 'type': type(exc).__name__, 'message': str(exc)})
            raise
        with self.lock:
            if not self.journal.succeeded(result['identity']):
                self.journal.append({**result, 'identity_sha256': result['identity']})
            self.done += 1
            if self.done % 32 == 0 or self.done == self.total:
                print({'cached_scored': self.done, 'total': self.total}, flush=True)
        return result
    def ensure(self, row): return self.futures[row['case_id']].result()
    def finish(self):
        for future in self.futures.values(): future.result()
        self.executor.shutdown()
        write(OUT/'cache_receipt.json', {'cases': self.total, 'completed': self.done,
            'fixed_backbone_sha256': runtime.sha256(OLD_OUT/'backbone_selection.json'),
            'old_cache_files_modified': False, 'labels_enter_inversion': False})

def load(row, selection, labels=True):
    key = key_for(row, selection); path = cache_location(key)
    score = read(DATA/'physical_scores'/(row['case_id']+'.json'))
    assert score['cache_key'] == key and score['label_sha256'] == row['label_sha256']
    raw = npz(file_path(row, 'raw_path'))['raw_dn']; assert array_hash(raw) == row['raw_hash']
    arr = npz(path)
    inputs = (torch.as_tensor(raw.astype(np.float32), device='cuda')[None],
              torch.as_tensor(arr['state'], device='cuda')[None], torch.as_tensor(arr['beta'], device='cuda')[None])
    if not labels: return inputs, score
    target = npz(file_path(row, 'label_path'))
    truth = {k: torch.as_tensor(target[k], device='cuda')[None]
             for k in ('template_e', 'temperature_color_K', 'band_images_e')}
    return inputs, truth, score
