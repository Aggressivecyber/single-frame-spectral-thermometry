"""E-drive artifacts, identities, source and data provenance."""
import runtime
import json
import os
from pathlib import Path
import numpy as np
import torch
from evaluation_v2.common import canonical_hash, array_hash, tensor_digest, Journal

OUT = runtime.ROOT/'reports/training_v4/run_001'
DATA = runtime.ROOT/'outputs/training_v4/run_001'


def setup():
    assert runtime.ROOT.drive.upper() in ('E:', 'D:'), 'Use an E/D checkout'
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['CUDA_CACHE_PATH'] = str(runtime.RUNTIME/'cuda')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **values)
    temp.replace(path)


def write(path, value):
    runtime.write_json(path, value)


def source_hashes():
    return {str(p.relative_to(runtime.ROOT)).replace('\\', '/'): runtime.sha256(p)
            for p in sorted((runtime.ROOT/'training_v4').glob('*.py'))}


def parameters_hash(model):
    return tensor_digest(model.named_parameters())
