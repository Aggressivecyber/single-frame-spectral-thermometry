"""Matched raw/state/beta cache. Labels never enter inversion or cache keys."""
import copy
from pathlib import Path
import numpy as np
import torch
import runtime
from evaluation_v2.common import canonical_hash, array_hash, Journal
from training_v4.common import DATA, OUT, npz, save_npz, read, write
from training_v4.scalar_engine import invert_single


def cache_identity(raw_hash, physics_config, calibration_hash, method, solver_config):
    return canonical_hash({'raw_hash': raw_hash, 'physics_config_sha256': canonical_hash(physics_config),
        'calibration_sha256': calibration_hash, 'method': method, 'solver_config': solver_config,
        'engine_sha256': runtime.sha256(Path(__file__).with_name('scalar_engine.py')),
        'engine': 'single_frame_cached_Jacobian_v3_math', 'dtype': 'float32', 'torch': torch.__version__})


def cache_raw(p, raw_array, method, solver_config, calibration_hash=None):
    """Accepts only actual raw and camera/solver configuration; no label argument."""
    raw_hash = array_hash(raw_array)
    calibration_hash = calibration_hash or canonical_hash(p.cfg)
    identity = cache_identity(raw_hash, p.cfg, calibration_hash, method, solver_config)
    path = DATA/'cache'/identity[:2]/(identity+'.npz')
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        meta = read(meta_path)
        assert meta['raw_hash'] == raw_hash and meta['cache_key'] == identity
        assert runtime.sha256(path) == meta['cache_sha256']
        values = npz(path)
        return values['state'], values['beta'], meta
    raw = torch.as_tensor(np.array(raw_array, dtype=np.float32, copy=True), device=p.M.device)[None]
    if torch.any(raw >= 4095):
        raise ValueError('Saturated whole frame retained as a cache failure')
    raw_e = p.raw_to_electrons(raw)
    # Cross-frame floating CG differences were detected and recorded in the
    # initial tests. Production deliberately uses the verified scalar path.
    state, beta, histories, seconds = invert_single(p, raw_e, method, solver_config)
    assert len(state) == 1
    z, b = state[0].cpu().numpy(), beta[0].cpu().numpy()
    save_npz(path, state=z, beta=b)
    meta = {'cache_key': identity, 'raw_hash': raw_hash,
        'physics_config_sha256': canonical_hash(p.cfg), 'calibration_sha256': calibration_hash,
        'solver_config_sha256': canonical_hash(solver_config), 'method': method,
        'engine': 'single_frame_cached_Jacobian_v3_math', 'labels_used': False,
        'cache_path': str(path.relative_to(DATA)).replace('\\', '/'), 'cache_sha256': runtime.sha256(path),
        'seconds': seconds, 'history': histories[0],
        'peak_GPU_allocated_MiB': torch.cuda.max_memory_allocated()/2**20}
    write(meta_path, meta)
    return z, b, meta


def cache_row(p, row, method, solver_config):
    path = DATA/row['raw_path']
    assert runtime.sha256(path) == row['raw_file_sha256']
    raw = npz(path)['raw_dn']
    assert array_hash(raw) == row['raw_hash']
    return cache_raw(p, raw, method, solver_config, row['calibration_sha256'])


def load_tensors(row, selection, device='cuda', *, labels=True):
    manifest = read(OUT/'data_manifest.json') if 'physics_config' not in selection else selection
    cfg = manifest['physics_config']
    key = cache_identity(row['raw_hash'], cfg, row['calibration_sha256'], selection['method'], selection['solver_config'])
    path = DATA/'cache'/key[:2]/(key+'.npz')
    meta = read(path.with_suffix('.json'))
    assert meta['raw_hash'] == row['raw_hash'] and meta['cache_key'] == key
    raw = npz(DATA/row['raw_path'])['raw_dn']
    assert array_hash(raw) == row['raw_hash'], 'Noise changed without a matching cache'
    cached = npz(path)
    inputs = (torch.as_tensor(raw.astype(np.float32), device=device)[None],
              torch.as_tensor(cached['state'], device=device)[None],
              torch.as_tensor(cached['beta'], device=device)[None])
    if not labels:
        return inputs, meta
    target = npz(DATA/row['label_path'])
    truth = {k: torch.as_tensor(target[k], device=device)[None]
             for k in ('template_e', 'temperature_color_K', 'band_images_e')}
    return inputs, truth, meta
