"""Bounded local CUDA repair launcher; touches only its own child process."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('E:/CodexWork/thermometry/public_repository_001')
PYTHON = Path('E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe')
OUT = ROOT/'reports/repair_v3/run_001'


class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


class PE(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ProcessID', wintypes.DWORD),
                ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG), ('dwFlags', wintypes.DWORD),
                ('szExeFile', wintypes.WCHAR*260)]


def descendants(kernel, root_pid):
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    entries = []
    p = PE()
    p.dwSize = ctypes.sizeof(p)
    ok = kernel.Process32FirstW(snapshot, ctypes.byref(p))
    while ok:
        entries.append((p.th32ProcessID, p.th32ParentProcessID))
        ok = kernel.Process32NextW(snapshot, ctypes.byref(p))
    kernel.CloseHandle(snapshot)
    ids = {root_pid}
    while True:
        more = {pid for pid, parent in entries if parent in ids}
        if more.issubset(ids):
            return ids
        ids.update(more)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=['gates', 'smoke'])
    args = parser.parse_args()
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
                'PYTHONPATH': str(ROOT/'_runtime/audit_deps'),
                'TMP': str(ROOT/'_runtime/tmp'), 'TEMP': str(ROOT/'_runtime/tmp'),
                'CUDA_CACHE_PATH': str(ROOT/'_runtime/cuda')})
    command = [str(PYTHON), '-B', str(ROOT/'repair_next.py'), '--phase', args.suite, '--resume', '--output', str(OUT)]
    start = time.perf_counter()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    peak, reason = 0., 'NORMAL_EXIT'
    log = OUT/(args.suite+'.log')
    with log.open('a', encoding='utf-8') as f:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        observed_pids = set()
        while child.poll() is None:
            tree_memory = 0
            for pid in descendants(kernel, child.pid):
                handle = kernel.OpenProcess(0x0410, False, pid)
                memory = PMC()
                memory.cb = ctypes.sizeof(memory)
                if handle and psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    tree_memory += memory.WorkingSetSize
                    observed_pids.add(pid)
                if handle:
                    kernel.CloseHandle(handle)
            peak = max(peak, tree_memory/2**20)
            if peak > 4096 or time.perf_counter()-start > 2400:
                reason = 'HOST_MEMORY_BUDGET' if peak > 4096 else 'WALL_BUDGET'
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                break
            time.sleep(.5)
        code = child.wait()
    receipt = {'suite': args.suite, 'child_pid': child.pid, 'child_exit_code': code,
               'stop_reason': reason, 'status': 'COMPLETE' if code == 0 else 'PARTIAL',
               'seconds': time.perf_counter()-start, 'peak_observed_host_working_set_MiB': peak,
               'memory_scope': 'sum of live working sets in this repair child process tree, includes venv launcher descendant',
               'observed_process_ids': sorted(observed_pids),
               'wall_limit_s': 2400, 'host_memory_limit_MiB': 4096, 'command': command,
               'log_path': str(log), 'compute': 'local CUDA, user explicitly authorized',
               'optimizer_updates': 0, 'other_processes_modified': False}
    path = OUT/('local_'+args.suite+'_launcher_receipt.json')
    path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
