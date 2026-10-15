"""One NT startup snapshot for a root-owned explicit raw scheduling shard."""
from pathlib import Path
import argparse
import datetime
import importlib.util
import json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--process',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
for name in (args.process,args.output):
    if Path(name).name!=name or not name.endswith('.json'):raise ValueError('Local receipt names required')
root=Path(__file__).resolve().parents[4];folder=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ownedNT',root/'reports/closeout_001/repair_001/v3_tangent_probe_002/startup_observer.py')
nt=importlib.util.module_from_spec(spec);spec.loader.exec_module(nt)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
receipt=read(folder/args.process)
global_lock=read(root/'reports/closeout_001/PHASE_ACTIVE.json');local_lock=read(folder/'PHASE_ACTIVE.json')
launcher=nt.process(receipt['launcher_pid']);child=nt.process(global_lock['pid'])
expected=' '.join(receipt['arguments'])
calls=[json.loads(line) for line in (root/'reports/closeout_001/inverse_calls.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
charged=sum(not row.get('reused') for row in calls)
started=[]
for p in (folder/'inference/cells').glob('*/STARTED.json'):
    ticket=read(p);context=ticket['context'];identity=context['identity']
    if identity['case_id'] not in receipt['raw_ids']:continue
    started.append({'path':str(p),'identity':identity,'runtime':context['runtime'],
                    'source_hashes_count':len(context['source_sha256']),'owner':ticket['owner']})
checks={'launcher_available':launcher.get('available') is True,'child_available':child.get('available') is True,
        'commands_match':all(expected in v.get('command_line','') for v in (launcher,child)),
        'parent_child_matches':child.get('parent_pid')==receipt['launcher_pid'],
        'owner_locks_match':global_lock==local_lock,
        'owner_registration_matches':global_lock['registration_sha256']==receipt['registration_sha256'],
        'stderr_empty':Path(receipt['stderr']).stat().st_size==0,
        'charged_within_shard_cap':receipt['global_charged_inverse_count_before_launch']<=charged<=receipt['global_charged_inverse_count_maximum'],
        'first_new_ticket_present':bool(started),
        'native_float64':bool(started) and all(v['runtime']['device']=='cuda:0' and v['runtime']['dtype']=='torch.float64' for v in started),
        '42_registered_sources':bool(started) and all(v['source_hashes_count']==42 for v in started),
        'ticket_actual_registration':bool(started) and all(v['identity']['registration_sha256']==receipt['registration_sha256'] for v in started)}
value={'schema':'explicit-selection-shard-startup-observation-v001',
    'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'NtQueryInformationProcess; no WMI',
    'startup_verified':all(checks.values()),'checks':checks,'launcher':launcher,'actual_python':child,
    'owner_locks':global_lock,'started_cells':started,'global_charged_calls_at_observation':charged,
    'one_raw_denominator':receipt['planned_shard_cells'],'maximum_new_inverse_calls':receipt['maximum_new_inverse_calls'],
    'full_first_stage_denominator':1080,'raw_ids':receipt['raw_ids']}
with (folder/args.output).open('x',encoding='utf-8',newline='\n') as handle:handle.write(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2),flush=True)
assert value['startup_verified'],checks
