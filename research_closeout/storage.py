"""Storage boundaries and immutable references, configured before numerical imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path('E:/CodexWork/thermometry/public_repository_001')
REPORTS = ROOT/'reports/closeout_001'
DATA = ROOT/'outputs/closeout_001'
BASE = '6f6c4f19e08e83d1ac0cd49e0c85c1c67dc56b51'
PYTHON = Path('E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe')

def configure():
    if ROOT.drive.upper() not in ('E:', 'D:'):
        raise ValueError('Closeout storage must be on E/D')
    folder = DATA/'_runtime'
    for name in ('tmp','torch','cache','mpl','cuda'):
        (folder/name).mkdir(parents=True, exist_ok=True)
    for key in ('TMP','TEMP','TMPDIR'): os.environ[key] = str(folder/'tmp')
    os.environ.update(THERMOMETRY_RUNTIME=str(folder), CUDA_CACHE_PATH=str(folder/'cuda'),
        TORCH_HOME=str(folder/'torch'), XDG_CACHE_HOME=str(folder/'cache'),
        MPLCONFIGDIR=str(folder/'mpl'), PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '2')
    sys.dont_write_bytecode = True
    extra = LEGACY/'outputs/research_v5/run_001/_runtime/python_deps'
    if extra.is_dir() and str(extra) not in sys.path: sys.path.insert(0, str(extra))
    if sys.flags.optimize: raise ValueError('Assertions must remain enabled')

configure()

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_path(path):
    p = Path(path).resolve()
    permitted = [ROOT/'research_closeout', REPORTS, DATA, ROOT/'paper', ROOT/'release']
    if p.drive.upper() not in ('E:', 'D:') or not any(p==q or q in p.parents for q in permitted):
        raise ValueError('Write escapes closeout directories: '+str(p))
    return p
def write(path, value, *, replace=False):
    p = write_path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace: raise FileExistsError(str(p))
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(p)
    return p
def append(path,value):
    p=write_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def save(path,**values):
    import numpy as np
    p=write_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    values={k:(v.detach().cpu().numpy() if hasattr(v,'detach') else v) for k,v in values.items()}
    with p.open('xb') as f: np.savez_compressed(f,**values)
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def arrays(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def budget_guard():
    if shutil.disk_usage(ROOT).free < 2*2**30: raise RuntimeError('Closeout free-space floor reached')
def plain(value):
    import math
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    if hasattr(value,'tolist'): return plain(value.tolist())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value
def frozen_bridge():
    # A process-local read adapter, never a change to legacy files or settings.
    from research_v6 import common
    common.configure_runtime = configure
    common.ROOT = LEGACY
    common.CODE = LEGACY/'research_v6'
    common.V5_REPORTS = LEGACY/'reports/research_v5/run_001'
    common.V5_DATA = LEGACY/'outputs/research_v5/run_001'
    return common
