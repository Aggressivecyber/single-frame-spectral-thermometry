"""Single read-only startup observation using owned NT process metadata."""
from pathlib import Path
import datetime
import importlib.util
import json

root = Path(__file__).resolve().parents[4]
folder = Path(__file__).resolve().parent
reader = root/'reports/closeout_001/repair_001/v3_tangent_probe_002/startup_observer.py'
spec = importlib.util.spec_from_file_location('owned_NT_process_reader', reader)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
receipt = read(folder/'PRESCREEN_PROCESS.json')
global_lock = read(root/'reports/closeout_001/PHASE_ACTIVE.json')
local_lock = read(folder/'PHASE_ACTIVE.json')
launcher = module.process(receipt['launcher_pid'])
child = module.process(global_lock['pid'])
expected = 'research_closeout.c1_development prescreen --device cuda'
calls = [json.loads(line) for line in (root/'reports/closeout_001/inverse_calls.jsonl').read_text(encoding='utf8').splitlines()]
charged = sum(not row.get('reused') for row in calls)
started = []
for path in (folder/'inference/cells').glob('*/STARTED.json'):
    value = read(path)
    started.append({'path': str(path), 'owner': value.get('owner'),
                    'identity': value.get('identity'), 'runtime': value.get('runtime')})
checks = {
    'launcher_available': launcher.get('available', False),
    'child_available': child.get('available', False),
    'commands_match': all(expected in value.get('command_line', '') for value in (launcher, child)),
    'parent_child_matches': child.get('parent_pid') == receipt['launcher_pid'],
    'owner_locks_match': global_lock == local_lock,
    'registered_owner': global_lock.get('registration_sha256') == receipt['registration_sha256'],
    'stderr_empty': Path(receipt['stderr']).stat().st_size == 0,
    'no_terminal_at_observation': not Path(receipt['terminal']).exists(),
    'charged_call_cap': 5099 <= charged <= 5115,
}
value = {'schema': 'C1-nine-pair-startup-observation-v001',
         'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'method': 'NtQueryInformationProcess; no WMI', 'startup_verified': all(checks.values()),
         'checks': checks, 'launcher': launcher, 'actual_python': child,
         'owner_locks': global_lock, 'started_cells': started,
         'global_charged_calls_at_observation': charged, 'full_denominator': 18,
         'maximum_new_inverse_calls': 16, 'stdout_bytes': Path(receipt['stdout']).stat().st_size,
         'stderr_bytes': Path(receipt['stderr']).stat().st_size,
         'first_ticket_seen': bool(started),
         'four_saved_state_readback_may_still_be_running': not bool(started)}
path = folder/'PRESCREEN_STARTUP_VERIFICATION.json'
with path.open('x', encoding='utf8') as handle:
    json.dump(value, handle, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps(value, indent=2))
assert value['startup_verified'], checks
