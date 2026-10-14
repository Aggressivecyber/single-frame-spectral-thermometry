"""One controlled same-raw source replacement and error-decomposition study."""
from __future__ import annotations

import re
import shutil
import time
from . import common
common.configure_runtime()
import torch
from . import scoring


def evaluate_source(physics, raw_e, source, truth, options, beta_mode, shared_cache=None,
                    *, identity_criteria=None, float64_reference=False):
    """Actual frozen v5 output remains independent of component diagnostics."""
    from . import decomposition
    solver=common.frozen_v5('signed_linear_solver')
    beta=truth['beta'] if beta_mode=='true_beta' else None
    common.synchronize(physics.M.device);linear_start=time.perf_counter()
    result=solver.solve_signed_linear(physics,raw_e,source['continuum_latent'],options,fixed_beta=beta)
    common.synchronize(physics.M.device)
    result['_v6_linear_seconds']=time.perf_counter()-linear_start
    d=decomposition.decompose(physics,raw_e,source['continuum_latent'],truth['continuum_latent'],
          truth['template_latent_e'],truth['beta'],options,fixed_beta_mode=beta_mode=='true_beta',
          v5_result=result,shared_cache=shared_cache,identity_criteria=identity_criteria,
          float64_reference=float64_reference)
    return result,d


def _safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('Unsafe immutable case identifier: '+name)
    return name


def _source(ref,name,raw,physics,protocol):
    from . import continuum_sources as c
    cached=ref.get(name+'_cache')
    if name=='C_v3':
        out=c.cached_v3(raw,physics,protocol['v3_selection'],cached) if cached else c.infer_v3(raw,physics,protocol['v3_selection'])
    elif name=='C_v5':
        out=c.cached_v5(raw,physics,cached) if cached else c.infer_v5(raw,physics,protocol['continuum_options'])
    else:raise ValueError(name)
    if out.get('status')=='SOURCE_FAILED':raise RuntimeError(out.get('error','Frozen continuum source failed'))
    if not bool(torch.isfinite(out['continuum_latent']).all()):raise ValueError('Nonfinite continuum source')
    receipt={'case_id':ref['case_id'],'source':name,'raw_sha256':ref['raw_reference']['sha256'],
             'source_metadata':scoring.plain(out.get('source_metadata',{})), 'cache_used':bool(cached),
             'status':out.get('status'),'truth_access_during_inference':False}
    # Latent continuum can be reproduced from this state; do not duplicate the
    # seven-band raw/label/cached arrays. C_v5 development caches stay references.
    if 'state' in out:
        fields={key:out[key].detach().cpu().numpy() for key in ('state','beta') if key in out}
        receipt['state_cache']=common.save_npz(common.DATA/'stage_a/sources'/name/(_safe_name(ref['case_id'])+'.npz'),**fields)
    if cached:receipt['original_cache']=cached
    common.append_jsonl(common.REPORTS/'source_inference_receipts.jsonl',receipt)
    return out,receipt


def _save_components(ref,name,mode,d,shared_disk):
    arrays=d['arrays'];case=_safe_name(ref['case_id'])
    if shared_disk is None:
        shared_disk=common.save_npz(common.DATA/'stage_a/components_shared'/mode/(case+'.npz'),
             noise_postblur_e=arrays['noise_postblur_e'],
             regularization_bias_postblur_e=arrays['regularization_bias_postblur_e'])
    receipt=common.save_npz(common.DATA/'stage_a/components'/mode/name/(case+'.npz'),
          template_postblur_error_e=arrays['template_postblur_error_e'],
          leakage_postblur_e=arrays['leakage_postblur_e'],
          recomposition_residual_postblur_e=arrays['recomposition_residual_postblur_e'])
    return {'per_source':receipt,'shared_noise_and_bias':shared_disk},shared_disk


def _data_bytes():
    return sum(p.stat().st_size for p in (common.DATA/'stage_a').rglob('*') if p.is_file())


@torch.no_grad()
def run_stage_a(protocol,device='cuda'):
    from . import inputs, reports
    common.configure_runtime()
    started=common.REPORTS/'A_ONCE_STARTED.json'
    assert not started.exists(),'Stage A has started already; inspect receipt instead of blindly rerunning'
    common.write_json(started,{'started_at':common.now(),'protocol_sha256':common.sha(common.CODE/'protocol.json'),
                             'one_root_GPU_job':True,'sealed_payload_access':False})
    physics=common.make_physics(device)
    budget=common.Budget(protocol['stage_a_budget_s'])
    refs=common.read_json(common.CODE/'input_inventory.json')['rows']
    # Inspect already seen oracle cases first. No new raw is ever generated.
    refs=sorted(refs,key=lambda r:(0 if r['cohort']=='oracle' else 1,
        0 if r['input_kind']=='continuous' else 1 if r['input_kind']=='quantized' else 2,
        0 if r['group']=='normal' else 1,r['case_id']))
    rows=[];decompositions=[];source_receipts=[];processed=0;stop=None;diagnostic64_used=False
    options=protocol['linear_options'];stored_bytes=0;written=set()
    for ref in refs:
        try:
            budget.check()
            assert stored_bytes<=protocol['stage_a_output_budget_bytes'],'Stage A storage budget reached'
            assert shutil.disk_usage(common.ROOT).free>=protocol['minimum_E_free_bytes'],'Registered E free-space floor reached'
            raw=inputs.load_raw(ref,physics)
            raw_e=physics.raw_to_electrons(raw)
            truth=inputs.load_diagnostic_truth(ref,physics)
            W=common.frozen_v5('signed_linear_solver').fixed_raw_weights(physics,raw_e)
            fixed_options={**options,'weights':W}
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
            sources={'C_true':scoring.continuum_result(physics,truth['continuum_latent'])}
            source_times={'C_true':0.0}
            for name in ('C_v3','C_v5'):
                t=time.perf_counter();common.synchronize(device)
                try:
                    sources[name],receipt=_source(ref,name,raw,physics,protocol)
                    common.synchronize(device)
                    source_times[name]=time.perf_counter()-t
                    source_receipts.append(receipt)
                    stored_bytes+=receipt.get('state_cache',{}).get('bytes',0)
                    if name=='C_v3':
                        rows.append(scoring.score_v3(ref,sources[name]['baseline_output'],raw_e,truth,physics,
                             seconds=source_times[name],memory=common.memory_receipt(device)))
                except Exception as error:
                    common.append_jsonl(common.REPORTS/'failure_records.jsonl',
                          {**scoring.metadata(ref,name,'source_inference'),'failure_kind':'CONTINUUM_SOURCE_FAILED',
                           'error_type':type(error).__name__,'error':str(error)})
                    for mode in protocol['beta_modes']:
                        rows.append(scoring.failure(ref,name,mode,error,kind='CONTINUUM_SOURCE_FAILED'))
                    if name=='C_v3':rows.append(scoring.failure(ref,'original_v3','joint_original',error,kind='CONTINUUM_SOURCE_FAILED'))
            for mode in protocol['beta_modes']:
                shared=None;shared_disk=None
                for name in protocol['sources']:
                    if name not in sources:continue
                    budget.check();t=time.perf_counter()
                    result,d=evaluate_source(physics,raw_e,sources[name],truth,fixed_options,mode,shared,
                           identity_criteria=protocol['identity_criteria'])
                    common.synchronize(device)
                    elapsed=time.perf_counter()-t
                    shared=d['shared_cache']
                    scored=scoring.score(ref,name,mode,sources[name],result,raw_e,truth,physics,
                              seconds=source_times[name]+result['_v6_linear_seconds'],memory=common.memory_receipt(device))
                    scored['diagnostics'].update(source_cached=bool(ref.get(name+'_cache')),
                          source_access_seconds=source_times[name],linear_seconds=result['_v6_linear_seconds'],
                          decomposition_study_seconds=elapsed,
                          seconds_scope='current source inference or cache-validation plus actual signed solve; diagnostics separate')
                    scored['continuum_diagnostics']=scoring.source_diagnostics(physics,sources[name],truth['continuum_latent'],W,truth['evaluation_domain'])
                    rows.append(scored)
                    summary={**scoring.metadata(ref,name,mode),**scoring.plain(d['summary']),
                              'seconds':elapsed,'source_inference_seconds':source_times[name]}
                    first_shared=shared_disk is None
                    summary['component_arrays'],shared_disk=_save_components(ref,name,mode,d,shared_disk)
                    stored_bytes+=summary['component_arrays']['per_source']['bytes']
                    if first_shared:stored_bytes+=shared_disk['bytes']
                    decompositions.append(summary)
                    common.append_jsonl(common.REPORTS/'continuum_leakage_decomposition.jsonl',summary)
                    if summary.get('numerical_gate_passed') is not True:
                        stop={'case_id':ref['case_id'],'source':name,'beta_mode':mode,
                              'reason':'Primary fixed-objective numerical decomposition gate failed; no optimization allowed'}
                        if protocol['float64_diagnostic']['allowed'] and not diagnostic64_used:
                            diagnostic64_used=True
                            try:
                                _,reference=evaluate_source(physics,raw_e,sources[name],truth,fixed_options,mode,shared,
                                     identity_criteria=protocol['identity_criteria'],float64_reference=True)
                                common.write_json(common.REPORTS/'FAILED_CASE_FLOAT64_DIAGNOSTIC.json',
                                    {**scoring.metadata(ref,name,mode),**scoring.plain(reference['summary']),
                                     'cannot_rescue_primary_gate':True})
                            except Exception as error:
                                common.write_json(common.REPORTS/'FAILED_CASE_FLOAT64_DIAGNOSTIC.json',
                                    {**stop,'diagnostic_failed':True,'error':str(error),'error_type':type(error).__name__})
                        break
                if stop:break
            processed+=1
            for row in rows:
                key=(row['case_id'],row['method'])
                if key not in written:
                    common.append_jsonl(common.REPORTS/'continuum_source_comparison.jsonl',row)
                    written.add(key)
            common.append_jsonl(common.REPORTS/'A_CASE_EVENTS.jsonl',{'case_id':ref['case_id'],'processed_raw':processed,
                     'planned_raw':len(refs),'time':common.now(),'wall_s':budget.elapsed,'early_stop':stop,
                     'source_receipts':len(source_receipts),'memory':common.memory_receipt(device)})
            print(f'A raw {processed}/{len(refs)} {ref["case_id"]} wall={budget.elapsed:.1f}s stop={bool(stop)}',flush=True)
        except Exception as error:
            stop={'case_id':ref['case_id'],'reason':'Data/runtime/linear implementation exception',
                  'error_type':type(error).__name__,'error':str(error)}
            common.append_jsonl(common.REPORTS/'failure_records.jsonl',stop)
        if stop:break
    # Explicit unexecuted rows preserve the fixed cohort denominator after a
    # mandatory early stop. They are not mislabeled as returned estimates.
    present={(r['case_id'],r['method']) for r in rows}
    for ref in refs:
        methods=[(s,m) for s in protocol['sources'] for m in protocol['beta_modes']]+[('original_v3','joint_original')]
        for s,m in methods:
            if (ref['case_id'],s+'__'+m) not in present:
                rows.append({**scoring.metadata(ref,s,m),'status':'NOT_EXECUTED','numerical_execution_passed':False,
                      'metrics':None,'diagnostics':{},'stop_reason':stop})
    for row in rows:
        if (row['case_id'],row['method']) not in written:
            common.append_jsonl(common.REPORTS/'continuum_source_comparison.jsonl',row)
    after=inputs.integrity_receipt(refs,payload_audit=False)
    common.write_json(common.REPORTS/'INPUT_INTEGRITY_AFTER_A.json',after)
    decision=reports.decide_A(rows,decompositions,planned_raw=len(refs),processed_raw=processed,
                              stop_reason=stop,criteria=protocol['A_gates'])
    decision.update(wall_s=budget.elapsed,output_bytes=_data_bytes(),source_receipts=len(source_receipts),
                    float64_diagnostic_used=diagnostic64_used,device=device,
                    protocol_sha256=common.sha(common.CODE/'protocol.json'),
                    evidence_sha256={common.relative(common.REPORTS/f):common.sha(common.REPORTS/f) for f in
                      ('continuum_source_comparison.jsonl','continuum_leakage_decomposition.jsonl','source_inference_receipts.jsonl')
                      if (common.REPORTS/f).is_file()})
    common.write_json(common.REPORTS/'A_DECISION.json',decision)
    reports.write_A_report(decision,decompositions,source_receipts)
    reports.write_current_decision(decision)
    return decision
