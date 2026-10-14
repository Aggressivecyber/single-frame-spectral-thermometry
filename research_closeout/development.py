"""Finite root-owned selection, numerical eligibility and strong traditional control."""
from . import storage as s
from . import measurement as m
from . import experiment_registry as registry
from . import metrics
from .baselines import FrozenBaselines,from_state
from .numerical_strategy import recovered_infer,RECOVERY
import argparse
import copy
import gc
import json
import os
import time
import traceback
import numpy as np
import torch

def registration():
    spec=s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    check=s.read(s.REPORTS/'DEVELOPMENT_CONTRACTS.json');assert check['all_passed']
    path=s.REPORTS/'DEVELOPMENT_REGISTRATION.json'
    value={'time':s.now(),'source_sha256':registry.source_hashes(),
        'spec_sha256':s.sha(s.ROOT/'research_closeout/execution_spec.yaml'),
        'numeric_checks_sha256':s.sha(s.REPORTS/'DEVELOPMENT_CONTRACTS.json'),
        'foundation_sha256':s.sha(s.REPORTS/'FOUNDATION_CONTINUED_RESULT.json'),
        'single_numerical_recovery':RECOVERY,
        'selection_mothers':spec['development']['selection_mothers'],
        'confirmation_mothers':spec['development']['confirmation_mothers'],
        'candidates':spec['development']['candidates'],
        'joint_prescreen':'same positive_normal noise raw of selection mother0000, both initializers, all nine combinations',
        'numeric_exclusion':'Any failed prescreen initializer excludes that candidate; all other planned cells remain explicitly unexecuted after numerical exclusion. No failed output is deleted.',
        'first_stage':'all five conditions of12 mothers, both initializers, every prescreen-qualified candidate',
        'confirmation':'top2 numerically complete candidates, all five conditions of remaining24 mothers, both initializers',
        'numeric_qualification':'every executed required initializer passes tighter inner and outer comparisons, finite and nonnegative predicted raw',
        'initializer_selection':'choose one globally by development loss among both deployable initializers; confirmation cannot switch initializer',
        'ranking_loss':'(pooled color RMSE/20K)^2+(pooled template RMSE/17e)^2, complete fixed domains',
        'new_method_acceptance':{'normal_signed_template_ratio_max':1.05,'color_RMSE_max_K':20.,
            'at_least_one_relative_RMSE_improvement':.10,'other_primary_dimension_ratio_max':1.05,
            'mean_normal_projection_absolute_error_max':.10,'each_normal_sign_checked':True},
        'traditional_selection':'lambda10/100/1000 on first12 only; validate winner and preselected100 on other24, no retuning',
        'traditional_history':'original lambda0.1 retained by same-raw old receipts, not relabeled as100',
        'v3_production_reference':{'mothers':12,'conditions':5,'original':'exact frozen cached float32, original12 outer iterations',
            'precision_reference':'float64 same finite algorithm/config',
            'linear_reference':'float64 same12 outer iterations, CG rtol1e-8 max600',
            'certification_budgets':{'template_RMS_e':.17,'color_RMS_K':.2},
            'scope':'sensitivity of frozen finite estimator to arithmetic and inner solve; not proof of nonlinear optimum'},
        'v3_fallback':'always retain frozen algorithm; failed reliability makes it a research baseline, not a qualified deployment estimator',
        'no_final_payloads_opened':True,'no_checkpoint_or_architecture_change':True}
    s.write(path,value)
    return value

def verify_registration():
    reg=s.read(s.REPORTS/'DEVELOPMENT_REGISTRATION.json')
    for path,digest in reg['source_sha256'].items(): assert s.sha(s.ROOT/path)==digest,'Development source changed: '+path
    assert s.sha(s.ROOT/'research_closeout/execution_spec.yaml')==reg['spec_sha256']
    return reg

def references():
    return sorted([r for r in s.read(s.ROOT/'research_v6/input_inventory.json')['rows']
        if r['cohort']=='development' and r['input_kind']=='noise'],key=lambda r:r['case_id'])

def cached_v3(ref,p):
    if not hasattr(cached_v3,'receipts'):
        cached_v3.receipts={r['case_id']:r for r in s.rows(s.LEGACY/'reports/research_v6/repair_001/source_inference_receipts.jsonl') if r['source']=='C_v3'}
    row=cached_v3.receipts[ref['case_id']]
    assert row['raw_sha256']==ref['raw_reference']['sha256']
    receipt=row['state_cache'];path=s.LEGACY/receipt['path'];assert s.sha(path)==receipt['sha256']
    values=s.arrays(path)
    return tuple(torch.as_tensor(values[key],device=p.M.device,dtype=p.M.dtype) for key in ('state','beta'))

def reused_baseline_rows(refs):
    path=s.REPORTS/'development_baseline_reuse.jsonl'
    if path.exists(): return list(s.rows(path))
    by_id={r['case_id']:r for r in refs};rows=[]
    source=s.LEGACY/'reports/research_v5/run_001/development/records.jsonl'
    digest=s.sha(source)
    for line,old in enumerate(s.rows(source),1):
        if old['case_id'] not in by_id or old['method'] not in ('B1_tikhonov','v3_physics','old_B4'): continue
        ref=by_id[old['case_id']];assert old['raw_sha256']==ref['raw_reference']['sha256']
        row={k:ref[k] for k in ('case_id','mother_id','condition','input_kind')}
        row.update(method='B1_lambda0.1' if old['method']=='B1_tikhonov' else old['method'],
            status=old['status'],metrics=old['metrics'],raw_sha256=old['raw_sha256'],
            seconds=old.get('diagnostics',{}).get('seconds'),reused=True,
            provenance={'path':str(source),'sha256':digest,'line':line},
            production_numerical_certified=False)
        s.append(path,row);rows.append(row)
        registry.count_call('development',row['method'],row['case_id'],reused=True)
    assert len(rows)==len(refs)*3
    return rows

def compact_info(out):
    info=dict(out['info']);history=info.pop('outer_history',None)
    if history:
        info['outer_history_summary']={'evaluations':len(history),'first':history[0],'last':history[-1],
            'minimum_recorded_objective':min(r['objective'] for r in history)}
    return s.plain(info)

def loss(rows):
    a=metrics.aggregate(rows)
    if not a['all_cases_finite'] or a['template_RMSE_e'] is None: return float('inf')
    return (a['temperature_RMSE_K']/20.)**2+(a['template_RMSE_e']/17.)**2

def run(device='cuda'):
    torch.set_num_threads(2)
    reg=verify_registration();s.frozen_bridge()
    from research_v6 import inputs
    refs=references();spec=s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    first=[r for r in refs if r['mother_id'] in reg['selection_mothers']]
    remaining=[r for r in refs if r['mother_id'] in reg['confirmation_mothers']]
    assert len(first)==60 and len(remaining)==120
    suite=FrozenBaselines(device);p=suite.p;p64=m.physics(device)
    reused=reused_baseline_rows(refs);v3_rows={r['case_id']:r for r in reused if r['method']=='v3_physics'}
    journal=s.REPORTS/'development_results.jsonl'
    rows=list(s.rows(journal)) if journal.exists() else []
    completed={(r['method'],r['case_id'],r.get('initializer')):r for r in rows}
    def append(row):
        s.append(journal,s.plain(row));rows.append(row)
        completed[(row['method'],row['case_id'],row.get('initializer'))]=row
    def evaluate(ref,method,action,initializer=None,save_parameters=False):
        key=(method,ref['case_id'],initializer)
        if key in completed: return completed[key]
        s.budget_guard();registry.count_call('development',method+('__'+initializer if initializer else ''),ref['case_id'])
        started=time.perf_counter();row={k:ref[k] for k in ('case_id','mother_id','condition','input_kind')}
        row.update(method=method,initializer=initializer,raw_sha256=ref['raw_reference']['sha256'],
            labels_used_by_inference=False,reused=False)
        raw=inputs.load_raw(ref,p)
        try:
            out=action(raw);truth=inputs.load_diagnostic_truth(ref,p)
            row.update(metrics.score(out,truth,raw,p),info=compact_info(out))
            # All finite outputs remain represented, even uncertified cases.
            if save_parameters:
                path=s.DATA/'development_parameters'/method/str(initializer)/(ref['case_id']+'.npz')
                row['parameter_cache']=s.save(path,z_grid=out['z_grid'],xi=out['xi'],beta=out['beta'])
            if not save_parameters and ref['case_id'].endswith('0000_positive_normal_noise0'):
                row['representative_prediction']=s.save(s.DATA/'development_representatives'/(method+'.npz'),
                    temperature_color_K=out['temperature_color_K'],template_postblur_e_ref=out['template_postblur_e_ref'],
                    continuum_postblur_bands=out['continuum_postblur_bands'],predicted_raw_e=out['predicted_raw_e'])
        except Exception as error:
            row.update(status='NUMERICAL_OR_COMPLEXITY_UNCERTIFIED',metrics=None,production_numerical_certified=False,
                error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            s.append(s.REPORTS/'failures.jsonl',s.plain(row))
        row['wall_s']=time.perf_counter()-started;append(row)
        print('development '+method+' '+str(initializer)+' '+ref['case_id']+' '+row['status'],flush=True)
        gc.collect()
        return row
    # Cheap strong traditional grid and baseline configuration correction are
    # independent of profile eligibility and therefore run first.
    for ref in first:
        for lam in (10.,100.,1000.):
            evaluate(ref,f'B1_lambda{lam:g}',lambda raw,lam=lam:suite.traditional(raw,lam))
    first_ids={r['case_id'] for r in first}
    rankings=sorted([{'regularization':lam,'loss':loss([r for r in rows if r['method']==f'B1_lambda{lam:g}' and r['case_id'] in first_ids])}
        for lam in (10.,100.,1000.)],key=lambda x:(x['loss'],x['regularization']))
    traditional_eligible=bool(np.isfinite(rankings[0]['loss']))
    winner=rankings[0]['regularization'] if traditional_eligible else 100.
    for ref in remaining:
        for lam in sorted({winner,100.}):
            evaluate(ref,f'B1_lambda{lam:g}',lambda raw,lam=lam:suite.traditional(raw,lam))
    traditional={'time':s.now(),'selected_regularization':winner,'selection_mothers':12,'confirmation_mothers':24,
        'selection_rankings':s.plain(rankings),'selection_numerically_finite':traditional_eligible,
        'historical_regularization':.1,'v4_1_preselected_regularization':100.,
        'fallback_if_no_complete_configuration':'preselected100 retained as failed research comparator; other methods continue',
        'all_results_kept':True,'no_final_inputs_used':True,
        'confirmation':metrics.aggregate([r for r in rows if r['method']==f'B1_lambda{winner:g}' and r['case_id'] not in first_ids])}
    s.write(s.REPORTS/'TRADITIONAL_SELECTION.json',traditional,replace=True)
    # Every combination gets the same physical numerical screen. A failed
    # method is excluded, rather than consuming the science matrix with an
    # estimator which has already failed numerical/complexity eligibility.
    pilot=next(r for r in first if r['case_id']=='v5_development_selection_mother_0000_positive_normal_noise0')
    eligible=[];candidate_reports=[]
    def joint(ref,candidate,initial):
        def action(raw):
            state=cached_v3(ref,p)[0] if initial=='v3' else None
            return recovered_infer(raw,p64,candidate,initializer=initial,v3_state=state)
        return evaluate(ref,candidate['id'],action,initial,save_parameters=True)
    for candidate in reg['candidates']:
        pair=[joint(pilot,candidate,initial) for initial in ('v3','smooth_demod')]
        valid=all(r.get('production_numerical_certified') for r in pair)
        report={'id':candidate['id'],'prescreen_passed':valid,'planned_first_stage_cells':120,
            'first_stage_executed_cells':2,'reason':None if valid else 'numeric_or_complexity_prescreen_failed',
            'unexecuted_cells_are_not_successes':True}
        if valid:
            for ref in first:
                for initial in ('v3','smooth_demod'): joint(ref,candidate,initial)
            stage=[r for r in rows if r['method']==candidate['id'] and r['case_id'] in first_ids]
            report['first_stage_executed_cells']=len(stage)
            valid=len(stage)==120 and all(r.get('production_numerical_certified') for r in stage)
            report['complete_first_stage_numeric_passed']=valid
            if valid:
                initial=min(('v3','smooth_demod'),key=lambda x:(loss([r for r in stage if r['initializer']==x]),x))
                eligible.append({'candidate':candidate,'initializer':initial,'loss':loss([r for r in stage if r['initializer']==initial])})
        candidate_reports.append(report)
        s.write(s.REPORTS/'DEVELOPMENT_CANDIDATE_PROGRESS.json',{'time':s.now(),'rows':candidate_reports},replace=True)
    finalists=sorted(eligible,key=lambda x:(x['loss'],x['candidate']['id']))[:2]
    confirmation=[]
    for finalist in finalists:
        candidate=finalist['candidate']
        for ref in remaining:
            for initial in ('v3','smooth_demod'): joint(ref,candidate,initial)
        stage=[r for r in rows if r['method']==candidate['id'] and r['case_id'] not in first_ids]
        selected=[r for r in stage if r['initializer']==finalist['initializer']]
        valid=len(stage)==240 and all(r.get('production_numerical_certified') for r in stage)
        confirmation.append({**finalist,'loss':loss(selected),'numeric_passed':valid,
            'aggregate':metrics.aggregate(selected)})
    # Precision/inner-solve sensitivity of the unchanged finite v3 algorithm.
    reliability=s.REPORTS/'v3_production_numerics.jsonl'
    old_reliability=list(s.rows(reliability)) if reliability.exists() else []
    done={r['case_id'] for r in old_reliability}
    for ref in first:
        if ref['case_id'] in done: continue
        raw=inputs.load_raw(ref,p);state,beta=cached_v3(ref,p)
        base=from_state(p,raw,state,beta,{'numerical_certified':False})
        comparisons=[]
        for mode in ('same_finite_float64','tighter_linear_float64'):
            registry.count_call('development','v3_numeric_reference_'+mode,ref['case_id'])
            cfg=copy.deepcopy(suite.selected['solver_config'])
            if mode=='tighter_linear_float64': cfg.update(cg_max_iterations=600,cg_relative_tolerance=1e-8)
            started=time.perf_counter()
            try:
                out=suite.v3(raw,physics=p64,solver_config=cfg)
                du=float((out['template_postblur_e_ref']-base['template_postblur_e_ref']).square().mean().sqrt())
                dt=float((out['temperature_color_K']-base['temperature_color_K']).square().mean().sqrt())
                finite=bool(torch.isfinite(out['predicted_raw_e']).all() and out['predicted_raw_e'].min()>=0)
                history=out['info']['inverse_history'][0]['joint']
                comparisons.append({'reference':mode,'template_RMS_difference_e':du,'color_RMS_difference_K':dt,
                    'all_finite_nonnegative_prediction':finite,'within_output_budgets':finite and du<=.17 and dt<=.2,
                    'outer_iterations':len(history),'last_CG':history[-1]['CG'] if history else None,
                    'wall_s':time.perf_counter()-started})
            except Exception as error:
                comparisons.append({'reference':mode,'within_output_budgets':False,'error_type':type(error).__name__,'error':str(error)})
        row={'case_id':ref['case_id'],'mother_id':ref['mother_id'],'condition':ref['condition'],
            'raw_sha256':ref['raw_reference']['sha256'],'comparisons':comparisons,
            'passed':all(x['within_output_budgets'] for x in comparisons),'frozen_v3_was_not_modified':True}
        s.append(reliability,s.plain(row));old_reliability.append(row)
        print('v3 production reference '+ref['case_id']+' passed='+str(row['passed']),flush=True)
        gc.collect()
    numeric_summary={'time':s.now(),'planned_cases':60,'returned_cases':len(old_reliability),
        'passed_cases':sum(r['passed'] for r in old_reliability),
        'all_passed':len(old_reliability)==60 and all(r['passed'] for r in old_reliability),
        'scope':'arithmetic/linear-accuracy sensitivity of frozen finite12-step estimator; not nonlinear optimum proof',
        'original_v3_outputs_preserved':True}
    s.write(s.REPORTS/'V3_PRODUCTION_NUMERICS.json',numeric_summary,replace=True)
    remaining_ids={r['case_id'] for r in remaining}
    controls=[v3_rows[r['case_id']] for r in remaining]
    control=metrics.aggregate(controls);accepted=[]
    for entry in confirmation:
        selected=[r for r in rows if r['method']==entry['candidate']['id'] and r['initializer']==entry['initializer'] and r['case_id'] in remaining_ids]
        a=entry['aggregate'];gates={'numeric':entry['numeric_passed'],
            'color_absolute':a['temperature_RMSE_K'] is not None and a['temperature_RMSE_K']<=20.,
            'color_noninferiority':a['temperature_RMSE_K'] is not None and a['temperature_RMSE_K']<=1.05*control['temperature_RMSE_K']}
        for condition in ('positive_normal','negative_normal'):
            group=[r for r in selected if r['condition']==condition];base_group=[r for r in controls if r['condition']==condition]
            ag=metrics.aggregate(group);bg=metrics.aggregate(base_group)
            projections=[r['metrics']['signed_projection'] for r in group if r.get('metrics') is not None]
            gates[condition+'_RMSE']=ag['template_RMSE_e'] is not None and ag['template_RMSE_e']<=1.05*bg['template_RMSE_e']
            gates[condition+'_amplitude']=len(projections)==24 and all(x is not None for x in projections) and float(np.mean(np.abs(np.asarray(projections)-1)))<=.10
        gates['key_improvement']=a['template_RMSE_e'] is not None and (a['template_RMSE_e']<=.9*control['template_RMSE_e'] or a['temperature_RMSE_K']<=.9*control['temperature_RMSE_K'])
        entry['acceptance_gates']=gates
        if all(gates.values()): accepted.append(entry)
    best=min(accepted,key=lambda x:(x['loss'],x['candidate']['id'])) if accepted else None
    selected={'time':s.now(),'method':'joint_profile' if best else 'v3_physics',
        'joint_options':best['candidate'] if best else None,'initializer':best['initializer'] if best else None,
        'reason':'registered gates passed' if best else 'No new candidate passed registered numerical and scientific gates',
        'traditional_regularization':winner,'v3_production_numerically_qualified':numeric_summary['all_passed'],
        'deployment_qualified':True if best else numeric_summary['all_passed'],
        'fallback_still_enters_independent_and_final_evaluation':True,
        'not_final_freeze':True,'final_payloads_opened':False,'candidate_reports':candidate_reports,
        'confirmation':confirmation,'v3_confirmation':control}
    s.write(s.REPORTS/'selected_method.json',s.plain(selected),replace=True)
    result={'time':s.now(),'status':'DEVELOPMENT_COMPLETE_CONTINUE_SCIENTIFIC_EVALUATION',
        'selected_method':selected,'traditional':traditional,'v3_numerics':numeric_summary,
        'final_payloads_opened':False,'whole_closeout_complete':False}
    s.write(s.REPORTS/'DEVELOPMENT_RESULT.json',s.plain(result),replace=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.register: result=registration()
    else:
        active=s.write_path(s.REPORTS/'PHASE_ACTIVE.json')
        fd=os.open(active,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w',encoding='utf8') as f: json.dump({'pid':os.getpid(),'phase':'development','time':s.now()},f)
        try: result=run(args.device)
        except Exception as error:
            result={'time':s.now(),'status':'DEVELOPMENT_MODULE_REQUIRES_DIAGNOSIS','error_type':type(error).__name__,
                'error':str(error),'traceback':traceback.format_exc(),'whole_closeout_complete':False,'final_payloads_opened':False}
            s.write(s.REPORTS/'DEVELOPMENT_FAILURE.json',result,replace=True)
            raise
        finally: active.unlink()
    print(json.dumps(s.plain(result),ensure_ascii=True),flush=True)

if __name__=='__main__': main()
