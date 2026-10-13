"""E/D storage, immutable inputs, durable records and raw-only conversion."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'research_v5'
REPORTS = ROOT / 'reports/research_v5/run_001'
DATA = ROOT / 'outputs/research_v5/run_001'
BASE = 'ddc4223fd1ecb9c5af82a8d1c489bf9cc54fd2d2'
TORCH_PYTHON = Path('E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe')


def ed_path(path):
    path = Path(path).expanduser().resolve()
    if path.drive.upper() not in ('E:', 'D:'):
        raise ValueError('Generated files must resolve on E/D: ' + str(path))
    return path


def configure_runtime():
    folder = ed_path(DATA / '_runtime')
    for name in ('tmp', 'torch', 'cache', 'mpl', 'cuda'):
        ed_path(folder / name).mkdir(parents=True, exist_ok=True)
    for key in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[key] = str(folder / 'tmp')
    os.environ.update(THERMOMETRY_RUNTIME=str(folder), CUDA_CACHE_PATH=str(folder / 'cuda'),
                     TORCH_HOME=str(folder / 'torch'), XDG_CACHE_HOME=str(folder / 'cache'),
                     MPLCONFIGDIR=str(folder / 'mpl'), PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '2')
    sys.dont_write_bytecode = True
    dependencies = folder/'python_deps'
    if dependencies.is_dir() and str(dependencies) not in sys.path:
        sys.path.insert(0, str(dependencies))
    if sys.flags.optimize:
        raise ValueError('Run ordinary Python with assertions enabled')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path, value, *, replace=False):
    path = ed_path(path)
    if not any(path.is_relative_to(folder.resolve()) for folder in (CODE, REPORTS.parent, DATA.parent)):
        raise ValueError('v5 write escaped the new v5 directories: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError('Never overwrite completed evidence: ' + str(path))
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    if read_json(path) != value:
        raise RuntimeError('JSON reopen differs: ' + str(path))
    return sha(path)


def append_jsonl(path, value):
    path = ed_path(path)
    if not path.is_relative_to(REPORTS.parent.resolve()):
        raise ValueError('Journal must stay in new v5 reports')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save_npz(path, **arrays):
    import numpy as np
    path = ed_path(path)
    if not path.is_relative_to(DATA.parent.resolve()):
        raise ValueError('New arrays must stay in new v5 outputs')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as reopened:
        assert set(reopened.files) == set(arrays)
        assert all(np.array_equal(reopened[key], value, equal_nan=True) for key, value in arrays.items())
    return sha(path)


def arrays(path, *, raw_only=False):
    import numpy as np
    with np.load(path, allow_pickle=False) as value:
        if raw_only and set(value.files) != {'raw_dn'}:
            raise ValueError('Deployment input is raw_dn only')
        return {key: value[key].copy() for key in value.files}


def physics_config():
    return read_json(ROOT / 'reports/training_v4/run_001/backbone_selection.json')['physics_config']


def make_physics(device='cpu', dtype=None, *, config=None):
    import torch
    from physics import SpectralPhysics
    return SpectralPhysics(physics_config() if config is None else config,
                           torch.float32 if dtype is None else dtype).to(device)


def raw_electrons(physics, raw_dn):
    import torch
    raw = torch.as_tensor(raw_dn, dtype=physics.M.dtype, device=physics.M.device)
    if raw.ndim == 2:
        raw = raw[None]
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 3 or tuple(raw.shape[-2:]) != tuple(physics.cfg['shape']):
        raise ValueError('Expected raw DN with configured H,W shape')
    maximum = 2**physics.cfg['adc_bits']-1
    if not bool(torch.isfinite(raw).all() and (raw >= 0).all() and (raw < maximum).all()):
        raise ValueError('Invalid or saturated raw DN; refuse inversion')
    return physics.raw_to_electrons(raw)


def synchronize(device):
    import torch
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def source_hashes():
    return {relative(path): sha(path) for path in sorted(CODE.glob('*.py'))}


def require_registered():
    protocol = read_json(CODE / 'protocol.json')
    assert protocol['baseline_head'] == BASE
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT/name) == digest, 'Frozen v5 source changed: ' + name
    for name, digest in protocol['fixed_input_sha256'].items():
        assert sha(ROOT/name) == digest, 'Frozen old input changed: ' + name
    return protocol


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = float(seconds)

    def check(self):
        if time.monotonic()-self.started > self.seconds:
            raise TimeoutError('Registered stage wall-clock budget exceeded')

    def elapsed(self):
        return time.monotonic()-self.started


def tracked_baseline_inventory():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    return [{'path': name, 'bytes': (ROOT/name).stat().st_size, 'sha256': sha(ROOT/name)}
            for name in names if name]
