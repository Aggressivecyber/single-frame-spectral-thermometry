"""Explicit Windows metadata fallback before scientific imports.

The installed Python/torch files are untouched. CPython's existing OSError
fallback supplies Windows metadata when WMI is unavailable. This is an entry
point dependency for newly registered work, never a rewrite of old runs.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import platform
import runpy
import sys

VERSION = 'windows-platform-wmi-fallback-v001'


def _unavailable_wmi(*args, **kwargs):
    raise OSError('WMI metadata disabled explicitly by ' + VERSION)


def native_machine():
    """Read the native architecture directly without WMI."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsWow64Process2.argtypes = [wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
    kernel.IsWow64Process2.restype = wintypes.BOOL
    process_arch, native_arch = wintypes.USHORT(), wintypes.USHORT()
    if not kernel.IsWow64Process2(kernel.GetCurrentProcess(),
            ctypes.byref(process_arch), ctypes.byref(native_arch)):
        raise ctypes.WinError(ctypes.get_last_error())
    names = {0x8664: 'AMD64', 0xAA64: 'ARM64', 0x014C: 'x86'}
    if native_arch.value not in names:
        raise RuntimeError('Unrecognized native machine: ' + hex(native_arch.value))
    return names[native_arch.value]


def install_platform_fallback():
    result = {'version': VERSION, 'sys_platform': sys.platform,
        'python_version': sys.version.split()[0], 'installed_files_modified': False,
        'scientific_algorithm_changed': False, 'WMI_disabled': False}
    if sys.platform != 'win32':
        return result
    if not hasattr(platform, '_wmi_query'):
        raise RuntimeError('This Python lacks the inspected platform WMI interface')
    native = native_machine()
    platform._wmi_query = _unavailable_wmi
    platform._uname_cache = None
    observed = platform.machine()
    if observed.upper() != native.upper():
        raise RuntimeError('Fallback/native architecture mismatch: ' +
                           repr((observed, native)))
    result.update(WMI_disabled=True, native_machine=native,
                  platform_machine=observed,
                  windows_version=list(platform.win32_ver()))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True)
    opts, args = parser.parse_known_args(argv)
    if not opts.module.startswith('research_closeout.') or not all(
            part.isidentifier() for part in opts.module.split('.')):
        parser.error('Use an explicitly registered research_closeout module')
    if args[:1] == ['--']:
        args = args[1:]
    sys.dont_write_bytecode = True
    evidence = install_platform_fallback()
    print('[runtime-entry] ' + json.dumps(evidence, sort_keys=True), flush=True)
    sys.argv = [opts.module] + args
    runpy.run_module(opts.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
