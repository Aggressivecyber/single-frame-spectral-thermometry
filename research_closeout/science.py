"""Root-owned preseal independent, boundary, null and full-inverse MC evaluation."""
from . import storage as s
from . import scientific_plan as planning
from . import experiment_registry as registry
from . import scenarios,metrics,uncertainty as u,intervals
from .deployment import FrozenPrototype
from .independent_renderer import sample_raw
import argparse
import collections
import gc
import json
import os
import shutil
import time
import traceback
import numpy as np
import torch

def verify_registration():
    reg=s.read(s.REPORTS/'SCIENCE_REGISTRATION.json')
    for name,digest in reg['source_sha256'].items(): assert s.sha(s.ROOT/name)==digest,'Registered science source changed: '+name
    assert s.sha(s.REPORTS/'SCIENTIFIC_PLAN.json')==reg['plan_sha256']
    assert s.sha(s.REPORTS/'selected_method.json')==reg['selected_method_sha256']
    return s.read(s.REPORTS/'SCIENTIFIC_PLAN.json')

def persistent_raw(path,dn):
    if path.exists():
        values=s.arrays(path);assert scenarios.array_hash(values['raw_dn'])==scenarios.array_hash(dn),'Immutable generated raw mismatch'
        return {'path':str(path),'sha256':s.sha(path),'bytes':path.stat().st_size}
    return s.save(path,raw_dn=dn)

class Runner:
    def __init__(self,plan,device):
        self.plan=plan;self.engine=FrozenPrototype(device);self.p=self.engine.physics
        self.journal=s.REPORTS/'per_case.jsonl';self.rows=list(s.rows(self.journal)) if self.journal.exists() else []
        self.done={(r['case_id'],r['method']):r for r in self.rows}
        self.raw_journal=s.REPORTS/'new_raw_manifest.jsonl'
        self.raws={r['case_id']:r for r in s.rows(self.raw_journal)} if self.raw_journal.exists() else {}
        self.stat_path=s.REPORTS/'whole_inverse_statistics.jsonl'
        self.stat_cache={(r['method'],r['raw_array_sha256']):r for r in s.rows(self.stat_path)} if self.stat_path.exists() else {}
        self.calls=0
    def guard(self):
        s.budget_guard();self.calls+=1
        if self.calls%20==1:
            roots=(s.DATA,s.REPORTS,s.ROOT/'paper/closeout_001',s.ROOT/'release/closeout_001')
            size=sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file())
            if size+32*2**20>4*2**30: raise RuntimeError('Registered4GiB output budget reached; scientific outputs retained')
    def raw(self,row,frame):
        raw,noise=sample_raw(frame['expected_e'],self.p.cfg,row['noise_seed'])
        digest=scenarios.array_hash(raw);path=s.DATA/'science_raws'/(row['case_id']+'.npz')
        if row['case_id'] in self.raws:
            old=self.raws[row['case_id']];assert old['raw_array_sha256']==digest and s.sha(old['raw']['path'])==old['raw']['sha256']
            return raw,old
        receipt=persistent_raw(path,raw)
        value={'case_id':row['case_id'],'mother_id':row['mother_id'],'family':row['family'],
            'engine':row['engine'],'noise_seed':row['noise_seed'],'raw_array_sha256':digest,'raw':receipt,
            'photon_ratio':row['photon_ratio'],'physically_available':frame['physically_available'],
            'minimum_latent_band_e_ref':frame['minimum_latent_band_e_ref'],'noise':noise,
            'expected_mean_sha256':scenarios.array_hash(frame['expected_e']),
            'truth_sha256':{k:scenarios.array_hash(frame[k]) for k in ('temperature_color_K','template_postblur_e_ref','continuum_postblur_bands')},
            'generator_metadata':frame['metadata']}
        s.append(self.raw_journal,s.plain(value));self.raws[row['case_id']]=value
        return raw,value
    def evaluate(self,row,methods):
        if all((row['case_id'],method) in self.done for method in methods):
            return [self.done[(row['case_id'],method)] for method in methods]
        self.guard()
        try:
            frame=scenarios.render(row,self.p)
        except Exception as error:
            frame={'raw_available':False,'physically_available':False,'generation_error':str(error),'generation_error_type':type(error).__name__}
        if frame['raw_available']:
            raw,receipt=self.raw(row,frame);truth=scenarios.truth_for_score(frame,self.p)
        results=[]
        for method in methods:
            key=(row['case_id'],method)
            if key in self.done: results.append(self.done[key]);continue
            result={k:row[k] for k in ('case_id','mother_id','condition','family','engine','geometry','coefficient_e','photon_ratio','normalization','perturbation')}
            result.update(method=method,physically_available=frame['physically_available'],
                raw_available=frame['raw_available'],truth_passed_to_inference=False)
            started=time.perf_counter()
            if not frame['raw_available']:
                result.update(status='RAW_UNAVAILABLE_NOT_REPAIRED',metrics=None,
                    production_numerical_certified=False,generator_metadata=frame.get('metadata'),generation_error=frame.get('generation_error'))
            else:
                result.update(raw=receipt['raw'],raw_array_sha256=receipt['raw_array_sha256'])
                phase='null' if row['family'].startswith('null_') else 'independent' if row['family']=='independent' else 'uncertainty' if row['family']=='uncertainty' else 'noise_structure_strength_calibration_dynamic'
                registry.count_call(phase,method,row['case_id'])
                try:
                    out=self.engine.infer(raw,method,photon_ratio=row['photon_ratio'])
                    result.update(metrics.score(out,truth,raw,self.p))
                    result['resolution_scores']=metrics.resolution_scores(out,truth,raw,self.p)
                    info={k:v for k,v in out['info'].items() if k!='inverse_history'}
                    history=out['info'].get('inverse_history')
                    if history:
                        info['finite_outer_iterations']=len(history[0]['joint'])
                        info['last_CG']=history[0]['joint'][-1]['CG'] if history[0]['joint'] else None
                    result['info']=s.plain(info)
                    result['statistics']=u.summary_statistics(out).tolist()
                    result['noise_feature']=intervals.feature_from_prediction(out,self.p.cfg)
                    representative=(row['family']=='independent' and row['mother_id'].endswith('_0003') and row['condition']=='nominal') or (row['family']=='structure' and row['geometry']=='double_hotspot' and row['coefficient_e']==180. and row['normalization']=='fixed_peak') or (row['family']=='dynamic' and row['mother_id'].endswith('_0000') and row['condition']=='frame02')
                    if representative or row['family']=='uncertainty':
                        result['prediction_cache']=s.save(s.DATA/'science_predictions'/method/(row['case_id']+'.npz'),
                            **{k:out[k] for k in ('temperature_color_K','template_postblur_e_ref','continuum_postblur_bands','predicted_raw_e','valid')})
                    del out
                except Exception as error:
                    result.update(status='FAILED_RETAINED',metrics=None,production_numerical_certified=False,
                        error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                    s.append(s.REPORTS/'failures.jsonl',s.plain(result))
            result['wall_s']=time.perf_counter()-started
            s.append(self.journal,s.plain(result));self.rows.append(result);self.done[key]=result;results.append(result)
            print('science '+row['family']+' '+row['case_id']+' '+method+' '+result['status'],flush=True)
        gc.collect();return results
    def stats_refit(self,raw,call_id):
        a=np.asarray(raw);digest=scenarios.array_hash(a);key=('v3_physics',digest)
        if key in self.stat_cache: return self.stat_cache[key]
        self.guard();registry.count_call('uncertainty','v3_physics',call_id);started=time.perf_counter()
        try:
            out=self.engine.infer(a,'v3_physics');stats=u.summary_statistics(out)
            if not np.isfinite(stats).all(): raise ArithmeticError('Nonfinite full-inverse statistics')
            row={'method':'v3_physics','raw_array_sha256':digest,'call_id':call_id,'statistics':stats.tolist(),
                'noise_feature':intervals.feature_from_prediction(out,self.p.cfg),'numerical_valid':False,
                'status':'FINITE_RESEARCH_BASELINE','truth_in_inference':False,'wall_s':time.perf_counter()-started}
        except Exception as error:
            row={'method':'v3_physics','raw_array_sha256':digest,'call_id':call_id,'statistics':None,
                'noise_feature':None,'numerical_valid':False,'status':'FAILED_RETAINED','error_type':type(error).__name__,'error':str(error)}
        s.append(self.stat_path,s.plain(row));self.stat_cache[key]=row;return row

def null_calibration(runner):
    path=s.REPORTS/'DETECTOR_THRESHOLDS.json'
    if path.exists(): return s.read(path)
    rows=[r for r in runner.rows if r['family']=='null_calibration'];thresholds={}
    for method in runner.plan['null_and_final_methods']:
        selected=[r for r in rows if r['method']==method]
        try:
            assert len(selected)==200
            value=u.null_threshold([r.get('metrics',{}).get('matched_dictionary_frame_score') if r.get('metrics') else None for r in selected],[r['mother_id'] for r in selected])
            value['available']=True
        except (ValueError,TypeError,AssertionError) as error:
            value={'available':False,'declared_mothers':200,'returned_cases':len(selected),'error':str(error),'failed_nulls_not_deleted':True}
        thresholds[method]=value
    result={'time':s.now(),'methods':thresholds,'calibration_units':200,'evaluation_labels_used':False,
        'definition':'framewise max absolute27-dictionary score; position/scale/sign multiplicity calibrated jointly'}
    s.write(path,result);return result

def null_evaluation(runner,thresholds):
    results={}
    for method in runner.plan['null_and_final_methods']:
        rows=[r for r in runner.rows if r['family']=='null_evaluation' and r['method']==method]
        t=thresholds['methods'][method];finite=[r for r in rows if r.get('metrics') and r['metrics'].get('matched_dictionary_frame_score') is not None]
        if t['available'] and len(finite)==200:
            successes=sum(r['metrics']['matched_dictionary_frame_score']>t['threshold'] for r in finite)
            results[method]=u.binomial_interval(successes,200)
        else: results[method]={'declared_mothers':200,'finite_cases':len(finite),'threshold_available':t['available'],'FPR_unavailable':True}
    result={'time':s.now(),'methods':results,'new_independent_null_evaluation_mothers':200,
        'calibration_mothers_disjoint':True,'thresholds_changed_on_evaluation':False}
    s.write(s.REPORTS/'NULL_EVALUATION.json',result,replace=True);return result

def full_uncertainty(runner):
    cases=[r for r in runner.plan['cases'] if r['family']=='uncertainty']
    path=s.REPORTS/'full_inverse_MC.jsonl';records=list(s.rows(path)) if path.exists() else []
    done={(r['case_id'],r['replicate']) for r in records}
    for row in cases:
        base=runner.evaluate(row,('v3_physics',))[0]
        if not base.get('prediction_cache'):
            # Preserve every planned draw when an initial estimate fails.
            for j in range(row['MC']):
                if (row['case_id'],j) not in done:
                    value={'case_id':row['case_id'],'mother_id':row['mother_id'],'role':row['role'],
                        'replicate':j,'statistics':None,'noise_feature':None,'truth_statistics':None,'status':'BASE_INVERSE_UNAVAILABLE'}
                    s.append(path,value);records.append(value)
            continue
        frame=scenarios.render(row,runner.p)
        truth_stats=u.summary_statistics(frame)
        estimate=s.arrays(base['prediction_cache']['path'])
        linear_path=s.REPORTS/'linearization'/(row['case_id']+'.json')
        if not linear_path.exists():
            calls=[0]
            def refit(dn):
                calls[0]+=1;value=runner.stats_refit(dn,row['case_id']+f'_linear{calls[0]:03d}')
                if value['statistics'] is None: raise ArithmeticError('A whole-inverse directional refit failed')
                return value
            try:
                local=u.legacy_whole_inverse_linearization(refit,estimate,runner.p.cfg,seed=row['linearization_seed'],
                    probes=8,summarize=lambda v:np.asarray(v['statistics']))
                local['step_size_check_passed']=all(e<=.2 for e in local['step_size_relative_discrepancies'])
                local['status']='LOCAL_DIRECTIONAL_DIAGNOSTIC' if local['step_size_check_passed'] else 'NUMERICAL_LINEARIZATION_UNCERTIFIED'
            except Exception as error:
                local={'status':'LOCAL_LINEARIZATION_FAILED_RETAINED','error_type':type(error).__name__,'error':str(error),
                    'planned_full_inverse_calls':32,'actual_callback_calls':calls[0]}
            s.write(linear_path,s.plain(local))
        for j,noise_seed in enumerate(row['MC_noise_seeds']):
            if (row['case_id'],j) in done: continue
            dn,noise=sample_raw(frame['expected_e'],runner.p.cfg,noise_seed)
            raw_id=row['case_id']+f'_MC{j:03d}'
            receipt=persistent_raw(s.DATA/'MC_raws'/(raw_id+'.npz'),dn)
            prediction=runner.stats_refit(dn,raw_id)
            value={**prediction,'case_id':row['case_id'],'mother_id':row['mother_id'],'role':row['role'],
                'replicate':j,'raw':receipt,'noise_seed':noise_seed,'truth_statistics':truth_stats.tolist(),
                'generation_mean':'independently specified nominal synthetic model; outside raw-only inference',
                'selected_and_v3_same_result_reused':True}
            s.append(path,s.plain(value));records.append(value)
            print('science MC '+row['case_id']+' '+str(j+1)+'/'+str(row['MC'])+' '+value['status'],flush=True)
        gc.collect()
    model_path=s.REPORTS/'INTERVAL_CALIBRATION.json'
    if model_path.exists(): model=s.read(model_path)
    else:
        model=intervals.legacy_calibrate([r for r in records if r['role']=='interval_calibration'])
        s.write(model_path,s.plain(model))
    evaluation=intervals.legacy_evaluate(records,model)
    stability=[]
    for row in cases:
        if row['MC']!=128: continue
        values=[r for r in records if r['case_id']==row['case_id']]
        prefixes=[]
        for count in (32,64,128):
            subset=[r for r in values if r['replicate']<count];finite=[r for r in subset if r['statistics'] is not None]
            a=np.asarray([r['statistics'] for r in finite])
            prefixes.append({'count':count,'finite':len(finite),
                'empirical_standard_deviation':np.std(a,axis=0,ddof=1).tolist() if len(a)>1 else None,
                'empirical_quantiles025975':np.quantile(a,[.025,.975],axis=0).tolist() if len(a)>1 else None})
        stability.append({'case_id':row['case_id'],'mother_id':row['mother_id'],'prefixes':prefixes,
            'same_first64_draws_reused':True,'interval_model_not_recalibrated_with128_or_evaluation':True})
    result={'time':s.now(),'status':'FULL_INVERSE_MC_COMPLETE_RESEARCH_BASELINE',
        'representative_mothers':12,'initial_MC_per_mother':64,'declared_total_draws':896,'returned_draws':len(records),
        'finite_draws':sum(r['statistics'] is not None for r in records),
        'operational_interval_evaluation':evaluation,'stability':stability,
        'local_linearization_directory':'reports/closeout_001/linearization',
        'shared_raw_correlation_preserved':True,'production_numerical_qualification_still_failed':True}
    s.write(s.REPORTS/'UNCERTAINTY_RESULT.json',s.plain(result),replace=True);return result

def run(device):
    torch.set_num_threads(2);plan=verify_registration();runner=Runner(plan,device)
    for family in ('null_calibration','null_evaluation','independent','structure','photons','calibration','dynamic'):
        for row in (r for r in plan['cases'] if r['family']==family):
            runner.evaluate(row,plan['null_and_final_methods'] if family.startswith('null_') else plan['methods'])
        if family=='null_calibration': null_calibration(runner)
        if family=='null_evaluation': null_evaluation(runner,s.read(s.REPORTS/'DETECTOR_THRESHOLDS.json'))
    uncertainty=full_uncertainty(runner)
    counts=collections.Counter(r['family'] for r in runner.rows)
    result={'time':s.now(),'status':'PRESEAL_SCIENTIFIC_MATRIX_TERMINAL','case_method_rows':dict(counts),
        'failed_or_unavailable_rows':sum(r.get('metrics') is None for r in runner.rows),
        'full_inverse_MC':{k:v for k,v in uncertainty.items() if k not in ('stability','operational_interval_evaluation')},
        'final_payloads_opened':False,'whole_closeout_complete':False,
        'next':['freeze complete methods/thresholds/targets/statistics/source','final64+36 mother evaluation','seven figures and Chinese manuscript/reproducibility release']}
    s.write(s.REPORTS/'SCIENCE_RESULT.json',s.plain(result));return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.register: result=planning.register()
    else:
        active=s.write_path(s.REPORTS/'PHASE_ACTIVE.json');fd=os.open(active,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w',encoding='utf8') as f: json.dump({'pid':os.getpid(),'phase':'preseal_science','time':s.now()},f)
        try: result=run(args.device)
        except Exception as error:
            s.write(s.REPORTS/'SCIENCE_FAILURE.json',{'time':s.now(),'error_type':type(error).__name__,'error':str(error),
                'traceback':traceback.format_exc(),'whole_closeout_complete':False,'final_payloads_opened':False},replace=True)
            raise
        finally: active.unlink()
    print(json.dumps(s.plain(result),ensure_ascii=True),flush=True)

if __name__=='__main__': main()
