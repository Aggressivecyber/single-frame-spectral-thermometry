"""Registered pilot retest and gated restored Stage A; no model training."""
from __future__ import annotations
import argparse
import json
import time
import shutil
import gc
from . import storage as s
from . import protocol,solver,diagnostics,sources
from .. import common as c,inputs,scoring,reports
import torch


@torch.no_grad()
def case(ref,p32,config,folder):
    raw=inputs.load_raw(ref,p32);raw_e=p32.raw_to_electrons(raw)
    truth=inputs.load_diagnostic_truth(ref,p32)
    if p32.M.device.type=='cuda':torch.cuda.reset_peak_memory_stats(p32.M.device)
    estimated={'C_true':scoring.continuum_result(p32,truth['continuum_latent'])}
    times={'C_true':0.};src=[];rows=[];decomp=[]
    for name in ('C_v3','C_v5'):
        c.synchronize(p32.M.device);start=time.perf_counter()
        estimated[name]=sources.obtain(ref,name,raw,p32,config)
        c.synchronize(p32.M.device);times[name]=time.perf_counter()-start
        meta={'case_id':ref['case_id'],'source':name,'raw_sha256':ref['raw_reference']['sha256'],
              'source_metadata':scoring.plain(estimated[name]['source_metadata']),'access_seconds':times[name]}
        if 'state' in estimated[name]:
            fields={key:estimated[name][key] for key in ('state','beta') if key in estimated[name]}
            meta['state_cache']=c.save_npz(s.DATA/folder/'source_states'/name/(ref['case_id']+'.npz'),**fields)
        src.append(meta)
    rows.append(scoring.score_v3(ref,estimated['C_v3']['baseline_output'],raw_e,truth,p32,
                    seconds=times['C_v3'],memory=c.memory_receipt(p32.M.device)))
    options={**config['linear_options'],'weights':c.frozen_v5('signed_linear_solver').fixed_raw_weights(p32,raw_e)}
    for mode in config['beta_modes']:
        system=solver.build_system(p32,raw_e,torch.zeros_like(truth['continuum_latent']),options,
               fixed_beta=truth['beta'] if mode=='true_beta' else None)
        shared=diagnostics.shared(system,truth);disk_shared=None
        for name in config['sources']:
            c.synchronize(p32.M.device);start=time.perf_counter()
            out=solver.solve(system,estimated[name]['continuum_latent'])
            c.synchronize(p32.M.device);signed_seconds=time.perf_counter()-start
            d=diagnostics.decompose(system,shared,estimated[name],out,truth)
            row=scoring.score(ref,name,mode,estimated[name],out,raw_e,truth,p32,
                     seconds=times[name]+signed_seconds,memory=c.memory_receipt(p32.M.device))
            row['diagnostics'].update(numeric_dtype='float64',source_access_seconds=times[name],
                linear_seconds=signed_seconds,precision_repair=True,
                source_time_scope='current inference or cache replay, not a cold-route speed claim')
            row['continuum_diagnostics']=scoring.source_diagnostics(p32,estimated[name],truth['continuum_latent'],options['weights'],truth['evaluation_domain'])
            rows.append(row)
            summary={**scoring.metadata(ref,name,mode),**scoring.plain(d['summary'])}
            if disk_shared is None:
                disk_shared=c.save_npz(s.DATA/folder/'shared'/mode/(ref['case_id']+'.npz'),
                    noise_postblur_e=d['arrays']['noise_postblur_e'],regularization_bias_postblur_e=d['arrays']['regularization_bias_postblur_e'])
            saved=c.save_npz(s.DATA/folder/'components'/mode/name/(ref['case_id']+'.npz'),
                **{key:d['arrays'][key] for key in ('template_postblur_error_e','leakage_postblur_e','recomposition_residual_postblur_e')})
            summary['component_arrays']={'per_source':saved,'shared_noise_and_bias':disk_shared}
            decomp.append(summary)
    return {'rows':rows,'decompositions':decomp,'sources':src}


def pilot(config,device):
    once=s.REPORTS/'PILOT_ONCE_STARTED.json';assert not once.exists()
    c.write_json(once,{'time':c.now(),'protocol_sha256':c.sha(s.CODE/'protocol.json')})
    ref=next(r for r in c.read_json(c.CODE/'input_inventory.json')['rows'] if r['case_id']==config['first_failed_case_id'])
    result=case(ref,c.make_physics(device),config,'pilot')
    for key,file in (('rows','PILOT_ROWS.jsonl'),('decompositions','PILOT_DECOMPOSITIONS.jsonl'),('sources','PILOT_SOURCES.jsonl')):
        for row in result[key]:c.append_jsonl(s.REPORTS/file,row)
    passed=all(x['numerical_gate_passed'] for x in result['decompositions']) and all(r['numerical_execution_passed'] for r in result['rows'])
    receipt={'time':c.now(),'case_id':ref['case_id'],'raw_sha256':ref['raw_reference']['sha256'],
      'all_passed':passed,'source_beta_rows':len(result['decompositions']),
      'maximum_closure_RMS_e':max(r['postblur_recomposition_residual']['rms'] for r in result['decompositions']),
      'maximum_closure_relative_l2':max(r['postblur_recomposition_relative_l2'] for r in result['decompositions']),
      'full_A_authorized_by_numerical_gate':passed,'original_failed_run_retained':True,
      'protocol_sha256':c.sha(s.CODE/'protocol.json'),'source_sha256':s.sources(),
      'evidence_sha256':{file:c.sha(s.REPORTS/file) for file in ('PILOT_ROWS.jsonl','PILOT_DECOMPOSITIONS.jsonl','PILOT_SOURCES.jsonl')}}
    c.write_json(s.REPORTS/'PILOT_GATE.json',receipt);return receipt


def stage_a(config,device):
    gate=c.read_json(s.REPORTS/'PILOT_GATE.json');assert gate['all_passed'] and gate['source_beta_rows']==6
    for file,digest in gate['evidence_sha256'].items():assert c.sha(s.REPORTS/file)==digest
    once=s.REPORTS/'A_ONCE_STARTED.json';assert not once.exists()
    c.write_json(once,{'time':c.now(),'protocol_sha256':c.sha(s.CODE/'protocol.json'),'pilot_gate_sha256':c.sha(s.REPORTS/'PILOT_GATE.json')})
    rows=list(c.jsonl(s.REPORTS/'PILOT_ROWS.jsonl'));decomp=list(c.jsonl(s.REPORTS/'PILOT_DECOMPOSITIONS.jsonl'))
    src=list(c.jsonl(s.REPORTS/'PILOT_SOURCES.jsonl'));completed={gate['case_id']};stop=None
    refs=c.read_json(c.CODE/'input_inventory.json')['rows'];p32=c.make_physics(device);budget=c.Budget(config['formal_A_budget_s'])
    for row in rows:c.append_jsonl(s.REPORTS/'continuum_source_comparison.jsonl',row)
    for row in decomp:c.append_jsonl(s.REPORTS/'continuum_leakage_decomposition.jsonl',row)
    for row in src:c.append_jsonl(s.REPORTS/'source_inference_receipts.jsonl',row)
    for ref in sorted(refs,key=lambda r:(r['cohort'],r['case_id'])):
        if ref['case_id'] in completed:continue
        try:
            budget.check();assert shutil.disk_usage(s.ROOT).free>=config['minimum_free_bytes']
            result=case(ref,p32,config,'stage_a')
            for key,file in (('rows','continuum_source_comparison.jsonl'),('decompositions','continuum_leakage_decomposition.jsonl'),('sources','source_inference_receipts.jsonl')):
                for row in result[key]:c.append_jsonl(s.REPORTS/file,row)
            rows.extend(result['rows']);decomp.extend(result['decompositions']);src.extend(result['sources']);completed.add(ref['case_id'])
            gc.collect()
            bad=[d for d in result['decompositions'] if not d['numerical_gate_passed']]
            if bad:stop={'case_id':ref['case_id'],'reason':'Repaired observable numerical gate failed','failed_methods':[d['method'] for d in bad]}
            c.append_jsonl(s.REPORTS/'CASE_EVENTS.jsonl',{'time':c.now(),'case_id':ref['case_id'],'completed_raw':len(completed),
                'wall_s':budget.elapsed,'stop':stop,'memory':c.memory_receipt(device)})
            print(f'repaired A {len(completed)}/720 {ref["case_id"]} wall={budget.elapsed:.1f}s stop={bool(stop)}',flush=True)
        except Exception as e:
            stop={'case_id':ref['case_id'],'reason':'Data/runtime/source exception','error_type':type(e).__name__,'error':str(e)}
        if stop:break
    present={(r['case_id'],r['method']) for r in rows}
    for ref in refs:
        for source,mode in [(n,m) for n in config['sources'] for m in config['beta_modes']]+[('original_v3','joint_original')]:
            if (ref['case_id'],source+'__'+mode) not in present:
                row={**scoring.metadata(ref,source,mode),'status':'NOT_EXECUTED','numerical_execution_passed':False,
                     'metrics':None,'diagnostics':{},'stop_reason':stop};rows.append(row);c.append_jsonl(s.REPORTS/'continuum_source_comparison.jsonl',row)
    integrity=inputs.integrity_receipt(refs,payload_audit=False);c.write_json(s.REPORTS/'INPUT_INTEGRITY_AFTER_A.json',integrity)
    decision=reports.decide_A(rows,decomp,planned_raw=720,processed_raw=len(completed),stop_reason=stop,criteria=config['A_gates'])
    decision.update(precision_repair=True,original_failed_run_retained=True,wall_s=budget.elapsed,
                    protocol_sha256=c.sha(s.CODE/'protocol.json'),pilot_included_without_reconstruction=True)
    c.write_json(s.REPORTS/'A_DECISION.json',decision)
    text='# v6 numerical repair001: Stage A\n\n'+json.dumps({k:decision[k] for k in
          ('status','A_complete','processed_raw','numerical_gate_passed','B_eligible','next_research_direction','stop_reason')},ensure_ascii=False,indent=2)+'\n'
    (s.REPORTS/'V6_A_REPAIR_REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    return decision


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('register','pilot','stage-a'),required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();start=time.perf_counter()
    if a.phase=='register':result=protocol.register()
    else:
        config=protocol.require();result=pilot(config,a.device) if a.phase=='pilot' else stage_a(config,a.device)
    print(json.dumps({'phase':a.phase,'all_passed':result.get('all_passed'),'status':result.get('status'),
          'A_complete':result.get('A_complete'),'maximum_closure_RMS_e':result.get('maximum_closure_RMS_e'),
          'wall_s':time.perf_counter()-start},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
