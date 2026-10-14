"""v6 Stage A CLI; conditional B requires its later preregistered addendum."""
from __future__ import annotations

import argparse
import json
import os
import time
from . import common
common.configure_runtime()


def dispatch(phase,device):
    if phase=='register':
        from .protocol import register
        return register()
    protocol=common.require_registered()
    if phase=='stage-a':
        from .stage_a import run_stage_a
        return run_stage_a(protocol,device)
    if phase=='report':
        from .reports import write_A_report,write_current_decision
        d=common.read_json(common.REPORTS/'A_DECISION.json')
        write_A_report(d,list(common.jsonl(common.REPORTS/'continuum_leakage_decomposition.jsonl')),
                      list(common.jsonl(common.REPORTS/'source_inference_receipts.jsonl')))
        write_current_decision(d)
        return d
    raise ValueError(phase)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('register','stage-a','report'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    active=common.ed_path(common.REPORTS/'PHASE_ACTIVE.json')
    active.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(active,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(descriptor,'w',encoding='utf-8') as f:
        json.dump({'phase':args.phase,'pid':os.getpid(),'started_at':common.now()},f);f.write('\n')
    t=time.perf_counter();stamp=common.now().replace(':','').replace('.','').replace('+','_')
    event={'phase':args.phase,'started_at':common.now(),'pid':os.getpid(),'device':args.device,
           'source_sha256':common.source_hashes(),'status':'STARTED'}
    common.append_jsonl(common.REPORTS/'PHASE_EVENTS.jsonl',event)
    code=1
    try:
        value=dispatch(args.phase,args.device)
        event.update(status='COMPLETED',returned_status=value.get('status'),
                     scientific_stage_A_complete=value.get('A_complete'),B_eligible=value.get('B_eligible'))
        if value.get('status')=='STOP_NUMERICAL_OR_ORACLE_AUDIT':event['status']='COMPLETED_NUMERICAL_STOP'
        code=0 if event['status']=='COMPLETED' else 2
    except Exception as error:
        event.update(status='FAILED',error_type=type(error).__name__,error=str(error))
    finally:
        event.update(finished_at=common.now(),wall_s=time.perf_counter()-t)
        common.append_jsonl(common.REPORTS/'PHASE_EVENTS.jsonl',event)
        common.write_json(common.REPORTS/'phase_receipts'/(args.phase+'_'+stamp+'.json'),event)
        active.unlink()
    print(json.dumps(event,ensure_ascii=False,allow_nan=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
