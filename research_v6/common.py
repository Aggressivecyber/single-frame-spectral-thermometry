"""v6-only storage, runtime and read-only access to frozen numerical libraries."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'research_v6'
REPORTS=ROOT/'reports/research_v6/run_001'
DATA=ROOT/'outputs/research_v6/run_001'
V5_REPORTS=ROOT/'reports/research_v5/run_001'
V5_DATA=ROOT/'outputs/research_v5/run_001'
BASE='431777df6c1d83fdae55fc54c30776e16ab74706'
PYTHON=Path('E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe')


def ed_path(path):
    p=Path(path).resolve()
    if p.drive.upper() not in ('E:','D:'):
        raise ValueError('All new v6 files must be on E/D: '+str(p))
    allowed=[ROOT/'research_v6',ROOT/'reports/research_v6',ROOT/'outputs/research_v6']
    if not any(p==q or q in p.parents for q in allowed):
        raise ValueError('New files must remain in v6 directories: '+str(p))
    return p


def configure_runtime():
    folder=ed_path(DATA/'_runtime')
    for name in ('tmp','torch','cache','mpl','cuda'):
        (folder/name).mkdir(parents=True,exist_ok=True)
    for key in ('TMP','TEMP','TMPDIR'):
        os.environ[key]=str(folder/'tmp')
    os.environ.update(THERMOMETRY_RUNTIME=str(folder),CUDA_CACHE_PATH=str(folder/'cuda'),TORCH_HOME=str(folder/'torch'),XDG_CACHE_HOME=str(folder/'cache'),MPLCONFIGDIR=str(folder/'mpl'),PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key,'2')
    sys.dont_write_bytecode=True
    # Existing E-drive plotting dependencies are read-only libraries, not v6 caches.
    dependencies=V5_DATA/'_runtime/python_deps'
    if dependencies.is_dir() and str(dependencies) not in sys.path:
        sys.path.insert(0,str(dependencies))
    if sys.flags.optimize:
        raise ValueError('Run normal Python -B with assertions enabled')


def now():return datetime.now(timezone.utc).isoformat()


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value,*,replace=False):
    p=ed_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise FileExistsError(str(p))
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8',newline='\n') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    t.replace(p)
    assert read_json(p)==value
    return p


def append_jsonl(path,value):
    p=ed_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def arrays(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def save_npz(path,**values):
    import numpy as np
    values={k:(v.detach().cpu().numpy() if hasattr(v,'detach') else v) for k,v in values.items()}
    p=ed_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**values)
    return {'path':relative(p),'sha256':sha(p),'bytes':p.stat().st_size}


def frozen_v5(name):
    if name not in {'signed_linear_solver','continuum_solver','metrics','statistics'}:
        raise ValueError('Only standalone read-only v5 numerical libraries are allowed')
    configure_runtime()
    module_name='_v6_frozen_v5_'+name
    if module_name in sys.modules:return sys.modules[module_name]
    path=ROOT/'research_v5'/(name+'.py')
    registered=read_json(ROOT/'research_v5/protocol.json')['source_sha256']
    assert sha(path)==registered[relative(path)],'Frozen library changed'
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    spec.loader.exec_module(module)
    return module


def physics_config():return read_json(ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config']


def make_physics(device='cpu',dtype=None):
    configure_runtime()
    from physics import SpectralPhysics
    p=SpectralPhysics(physics_config()).to(device)
    return p if dtype is None else p.to(dtype=dtype)


def source_hashes():return {relative(p):sha(p) for p in sorted(CODE.rglob('*.py'))}


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def memory_receipt(device):
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                  ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
                  ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
                  ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
                  ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    value=Counters();value.cb=ctypes.sizeof(value)
    get_handle=ctypes.windll.kernel32.GetCurrentProcess;get_handle.restype=wintypes.HANDLE
    get_info=ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes=(wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD)
    handle=get_handle()
    assert get_info(handle,ctypes.byref(value),value.cb)
    result={'process_peak_working_set_MiB':value.PeakWorkingSetSize/2**20,
            'process_current_working_set_MiB':value.WorkingSetSize/2**20,
            'process_peak_scope':'whole current Python process; not a per-frame reset'}
    if str(device).startswith('cuda'):
        import torch
        result.update(GPU_allocated_peak_MiB=torch.cuda.max_memory_allocated(device)/2**20,
                      GPU_reserved_peak_MiB=torch.cuda.max_memory_reserved(device)/2**20,
                      GPU_peak_scope='since last explicit per-raw reset')
    return result


def require_registered():
    p=read_json(CODE/'protocol.json')
    for name,digest in p['source_sha256'].items():
        assert sha(ROOT/name)==digest,'Registered v6 source changed: '+name
    for name,digest in p['frozen_library_sha256'].items():
        assert sha(ROOT/name)==digest,'Frozen original implementation changed: '+name
    assert sha(CODE/'TASK_SPEC.md')==p['task_spec_sha256']
    assert p['base_commit']==BASE
    return p


class Budget:
    def __init__(self,seconds):self.seconds=float(seconds);self.start=time.perf_counter()
    @property
    def elapsed(self):return time.perf_counter()-self.start
    def check(self):
        if self.elapsed>self.seconds:raise TimeoutError('Registered v6 wall budget exceeded')
