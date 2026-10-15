"""Single native startup observation, reading the true nested ticket context."""
from pathlib import Path
import datetime
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[4]
folder = Path(__file__).resolve().parent
reader = ROOT/'reports/closeout_001/repair_001/v3_tangent_probe_002/startup_observer.py'
spec = importlib.util.spec_from_file_location('owned_nt_reader', reader)
nt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nt)
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
receipt = read(folder/'BLANK_SHARD_PROCESS.json')
global_lock = read(ROOT/'reports/closeout_001/PHASE_ACTIVE.json')
local_lock = read(folder/'PHASE_ACTIVE.json')
launcher, child = nt.process(receipt['launcher_pid']), nt.process(global_lock['pid'])
expected = 'research_closeout.c1_selection run-shard --device cuda --raw v5_development_selection_mother_0000_blank_noise0'
calls = [json.loads(line) for line in (ROOT/'reports/closeout_001/inverse_calls.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
charged = sum(not row.get('reused') for row in calls)
started = []
for path in (folder/'inference/cells').glob('*/STARTED.json'):
    value = read(path)
    context = value['context']
    started.append({'path': str(path), 'cache_key': value['cache_key'], 'owner': value.get('owner'),
                    'identity': context['identity'], 'runtime': context['runtime'],
                    'source_hashes_count': len(context['source_sha256']),
                    'registration_sha256': context.get('registration_sha256')})
checks = {'launcher_available': launcher.get('available') is True,
          'child_available': child.get('available') is True,
          'commands_match': all(expected in value.get('command_line', '') for value in (launcher, child)),
          'parent_child_matches': child.get('parent_pid') == receipt['launcher_pid'],
          'owner_locks_match': global_lock == local_lock,
          'owner_registration_matches': global_lock.get('registration_sha256') == receipt['registration_sha256'],
          'stderr_empty': Path(receipt['stderr']).stat().st_size == 0,
          'charged_within_shard_cap': 5115 <= charged <= 5131,
          'first_new_ticket_present': bool(started),
          'native_float64_tickets': bool(started) and all(v['runtime']['device'] == 'cuda:0' and v['runtime']['dtype'] == 'torch.float64' for v in started),
          'registered42_sources': bool(started) and all(v['source_hashes_count'] == 42 for v in started)}
value = {'schema': 'C1-one-raw-first-stage-startup-observation-v001',
         'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'method': 'NtQueryInformationProcess; no WMI', 'startup_verified': all(checks.values()),
         'checks': checks, 'launcher': launcher, 'actual_python': child, 'owner_locks': global_lock,
         'started_cells': started, 'global_charged_calls_at_observation': charged,
         'one_raw_denominator': 18, 'maximum_new_inverse_calls': 16,
         'full_first_stage_denominator': 1080, 'truth_permitted_only_in_external_scoring': True}
with (folder/'BLANK_SHARD_STARTUP_VERIFICATION.json').open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(json.dumps(value, indent=2)+'\n')
print(json.dumps(value, indent=2), flush=True)
assert value['startup_verified'], checks
