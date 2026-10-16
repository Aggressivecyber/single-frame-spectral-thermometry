"""Bound the child process and persist its real exit status, including failures."""
import runtime
from pathlib import Path
import os
import subprocess
import sys
import time
import json

root = Path(__file__).resolve().parents[1]
report = root/'reports/hybrid_001'
protocol = json.loads((root/'research_hybrid/protocol.json').read_text())
env = dict(os.environ)
env.update(THERMOMETRY_RUNTIME=str(root/'_runtime'), CUDA_CACHE_PATH=str(root/'_runtime/cuda'),
           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
started = time.monotonic()
with (report/'pilot.stdout.txt').open('x') as stdout, (report/'pilot.stderr.txt').open('x') as stderr:
    child = subprocess.Popen([sys.executable, '-B', '-m', 'research_hybrid.run_pilot', '--device', 'cuda'],
                             cwd=root, env=env, stdout=stdout, stderr=stderr)
    runtime.write_json(report/'PROCESS.json', {'time': runtime.now(), 'supervisor_pid': os.getpid(),
                       'child_pid': child.pid, 'maximum_wall_seconds': protocol['budgets']['maximum_wall_seconds']})
    try:
        code = child.wait(timeout=protocol['budgets']['maximum_wall_seconds'])
        reason = 'child_exit'
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            code = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
        reason = 'registered_wall_time_limit'
    runtime.write_json(report/'EXIT.json', {'time': runtime.now(), 'returncode': code,
                       'reason': reason, 'wall_s': time.monotonic()-started,
                       'completed': code == 0 and (report/'COMPLETION.json').exists()})
sys.exit(code)
