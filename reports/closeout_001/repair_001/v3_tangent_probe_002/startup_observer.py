"""Read the two owned launch processes through Win32/NT APIs, without WMI."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import datetime
import json
import os
from pathlib import Path


class Basic(c.Structure):
    _fields_ = [('reserved', c.c_void_p), ('peb', c.c_void_p),
                ('reserved2', c.c_void_p*2), ('pid', c.c_void_p),
                ('parent', c.c_void_p)]


class Unicode(c.Structure):
    _fields_ = [('length', w.USHORT), ('maximum', w.USHORT),
                ('buffer', c.c_void_p)]


def process(pid):
    kernel = c.WinDLL('kernel32', use_last_error=True)
    native = c.WinDLL('ntdll')
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    native.NtQueryInformationProcess.argtypes = [w.HANDLE, w.ULONG,
        c.c_void_p, w.ULONG, c.POINTER(w.ULONG)]
    native.NtQueryInformationProcess.restype = c.c_long
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {'pid': pid, 'available': False, 'winerror': c.get_last_error()}
    try:
        basic, used = Basic(), w.ULONG()
        status = native.NtQueryInformationProcess(handle, 0, c.byref(basic),
            c.sizeof(basic), c.byref(used))
        if status:
            raise RuntimeError('Process basic query failed: ' + str(status))
        count = w.ULONG()
        native.NtQueryInformationProcess(handle, 60, None, 0, c.byref(count))
        if not 0 < count.value <= 1024*1024:
            raise RuntimeError('Unexpected process command-line buffer size')
        data = c.create_string_buffer(count.value)
        status = native.NtQueryInformationProcess(handle, 60, data,
            len(data), c.byref(count))
        if status:
            raise RuntimeError('Process command-line query failed: ' + str(status))
        text = Unicode.from_buffer(data)
        if not (c.addressof(data) <= text.buffer and
                text.buffer+text.length <= c.addressof(data)+len(data)):
            raise RuntimeError('Command-line pointer outside returned buffer')
        command = c.wstring_at(text.buffer, text.length//2)
        return {'pid': pid, 'available': True, 'actual_pid': basic.pid,
                'parent_pid': basic.parent, 'command_line': command}
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--folder', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.self_test:
        value = process(os.getpid())
        assert value['available'] and value['actual_pid'] == os.getpid()
        assert Path(__file__).name in value['command_line']
        print(json.dumps({'self_query_passed': True, 'method': 'NtQueryInformationProcess',
                          'no_WMI': True, 'pid': os.getpid()}), flush=True)
        return
    folder = args.folder.resolve()
    receipt = json.loads((folder/'PROCESS.json').read_text(encoding='utf-8-sig'))
    root = folder.parent.parent
    global_lock = json.loads((root/'PHASE_ACTIVE.json').read_text(encoding='utf-8-sig'))
    local_lock = json.loads((folder/'PHASE_ACTIVE.json').read_text(encoding='utf-8-sig'))
    launcher = process(receipt['launcher_pid'])
    child = process(global_lock['pid'])
    expected = 'research_closeout.v3_tangent_probe_v2'
    checks = {
        'launcher_available': launcher.get('available') is True,
        'child_available': child.get('available') is True,
        'both_actual_commands_match': all(expected in v.get('command_line', '') and
            'research_closeout.runtime_entry' in v.get('command_line', '') and
            'run --device cuda' in v.get('command_line', '') for v in (launcher, child)),
        'parent_child_matches': child.get('parent_pid') == launcher['pid'] or
            child['pid'] == launcher['pid'],
        'both_owner_locks_match': global_lock == local_lock,
        'owner_lock_registration_matches': global_lock['registration_sha256'] ==
            receipt['registration_sha256'],
        'run_started_exists': (folder/'RUN_STARTED.json').is_file(),
        'stderr_empty': (folder/'probe_002.stderr.txt').stat().st_size == 0,
    }
    value = {'schema': 'native-tangent-v2-startup-observation-v001',
        'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'method': 'NtQueryInformationProcess; no WMI', 'checks': checks,
        'startup_verified': all(checks.values()), 'launcher': launcher,
        'actual_python': child, 'owner_locks': global_lock,
        'RESULT_exists_at_observation': (folder/'RESULT.json').exists()}
    target = folder/'STARTUP_VERIFICATION.json'
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, indent=2), flush=True)
    assert value['startup_verified'], checks


if __name__ == '__main__':
    main()
