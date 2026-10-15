"""Small metadata-only checks; no numerical imports or GPU."""
from .. import runtime_entry as entry
from .. import storage as s
import platform
import sys
import time


def run():
    started = time.perf_counter()
    old_query = getattr(platform, '_wmi_query', None)
    old_cache = platform._uname_cache
    checks = []
    calls = []
    def fail_if_called(*args, **kwargs):
        calls.append(args)
        raise AssertionError('The previous WMI implementation must not be called')
    try:
        platform._wmi_query = fail_if_called
        evidence = entry.install_platform_fallback()
        checks.append({'name': 'fallback_does_not_call_previous_WMI',
                       'passed': not calls})
        checks.append({'name': 'native_and_fallback_architecture_agree',
                       'passed': sys.platform != 'win32' or
                       evidence['native_machine'].upper() == evidence['platform_machine'].upper()})
        again = entry.install_platform_fallback()
        checks.append({'name': 'metadata_fallback_idempotent',
                       'passed': again == evidence})
        checks.append({'name': 'WMI_query_raises_only_OSError', 'passed': False})
        try:
            entry._unavailable_wmi('OS', 'Version')
        except OSError:
            checks[-1]['passed'] = True
        checks.append({'name': 'installed_files_and_science_are_not_edited',
                       'passed': evidence['installed_files_modified'] is False and
                       evidence['scientific_algorithm_changed'] is False})
    finally:
        if old_query is not None:
            platform._wmi_query = old_query
        platform._uname_cache = old_cache
    result = {'schema': 'runtime-entry-metadata-contracts-v001', 'time': s.now(),
        'checks': checks, 'passed': sum(x['passed'] for x in checks),
        'total': len(checks), 'evidence': evidence,
        'source_sha256': {p: s.sha(s.ROOT/p) for p in (
            'research_closeout/runtime_entry.py', 'research_closeout/tests/runtime_entry.py')},
        'wall_s': time.perf_counter()-started, 'GPU_used': False,
        'complete_inverse_calls': 0}
    target = s.REPORTS/'repair_001/runtime_entry_001/CONTRACTS.json'
    s.write(target, result)
    print({'path': str(target), 'passed': result['passed'], 'total': result['total']}, flush=True)
    assert result['passed'] == result['total']


if __name__ == '__main__':
    run()
