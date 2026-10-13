"""Root closure audit and descriptive summaries of saved v5 results only.

No generator, reconstructor, GPU, optimizer, or sealed data is called here.
The frozen estimator and all END-bound results remain unchanged.
Run once: python -B -m research_v5.delivery.audit_saved_results_001
"""
from research_v5 import common
common.configure_runtime()

from collections import Counter, defaultdict
import csv
import json
import math
import os
import stat
import numpy as np
from research_v5.statistics import aggregate_metrics, mother_bootstrap, paired_statistics

R, D, O = common.REPORTS, common.REPORTS/'development', common.REPORTS/'root_review_001'


def write(name, value):
    common.write_json(O/name, value)


def grouped(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def describe(rows):
    a = aggregate_metrics(rows)
    if not a['complete']:
        return a
    metrics = [r['metrics'] for r in rows]
    n = a['domain_pixels']
    for target in ['reference_color', 'nominal_color', 'latent_truth']:
        key = 'temperature_' + target + '_RMSE_K'
        a[key] = math.sqrt(sum(m[key]**2*m['domain_pixels'] for m in metrics)/n)
    a['raw_weighted_RMSE'] = math.sqrt(sum(m['raw_weighted_SSE'] for m in metrics)/sum(m['raw_pixels'] for m in metrics))
    a['support_pixels'] = sum(m['support_pixels'] for m in metrics)
    a['sign_recall'] = sum(m['sign_correct_pixels'] for m in metrics)/a['support_pixels'] if a['support_pixels'] else None
    a['signed_threshold_recall'] = sum(m['signed_detected_pixels'] for m in metrics)/a['support_pixels'] if a['support_pixels'] else None
    a['frame_threshold_crossings'] = sum(bool(m['frame_detected']) for m in metrics)
    mothers = grouped(rows, lambda r:r['mother_id'])
    mse = {mid:sum(r['metrics']['temperature_SSE_K2'] for r in rr)/sum(r['metrics']['domain_pixels'] for r in rr) for mid,rr in mothers.items()}
    ci = mother_bootstrap(mse)
    a['color_RMSE_mother_CI95_K'] = [math.sqrt(v) for v in ci['CI95']]
    a['color_mother_MSE_bootstrap'] = ci
    det = [r.get('detection') for r in rows]
    if all(det):
        a['mean_abstained_pixel_fraction'] = float(np.mean([v['abstained_pixel_fraction'] for v in det]))
        a['median_detection_limit_e_range'] = [min(v['median_detection_limit_e'] for v in det),max(v['median_detection_limit_e'] for v in det)]
    return a


def main():
    assert not O.exists(), 'Audit is append-only; inspect existing receipt instead of rerunning.'
    O.mkdir(parents=True)
    started = common.now()
    protocol = common.require_registered()
    failures = []
    checks = []

    def check(name, passed, detail=None):
        checks.append({'id':name,'passed':bool(passed),'detail':detail})
        if not passed:
            failures.append(name)

    inventory = common.read_json(R/'baseline_readonly_inventory.json')['files']
    mismatch = [f['path'] for f in inventory if not (common.ROOT/f['path']).is_file() or common.sha(common.ROOT/f['path']) != f['sha256']]
    check('original_1452_file_bytes_unchanged',len(inventory)==1452 and not mismatch,mismatch)
    old = common.read_json(R/'old_array_readonly_inventory.json')['files']
    old_bad=[]
    for f in old:
        path=common.ROOT/f['path']
        s=path.stat()
        if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or bool(s.st_mode & stat.S_IWRITE):
            old_bad.append(f['path'])
    check('old_12869_array_metadata_readonly_unchanged',len(old)==12869 and not old_bad,old_bad)
    check('frozen_33_source_hashes',len(protocol['source_sha256'])==33 and protocol['source_sha256']==common.source_hashes())
    end=common.read_json(R/'execution_recovery_001/END.json')
    end_bad=[v['path'] for v in end['evidence']+[end['registration'],end['original_failure_receipt']] if common.sha(common.ROOT/v['path'])!=v['sha256']]
    check('END_complete_and_15_bindings',end['status']=='COMPLETED_WITH_SUPPLEMENTAL_DIAGNOSTICS' and end['all_planned_diagnostics_accounted_for'] and not end_bad,end_bad)
    registration=common.read_json(R/'execution_recovery_001/registration.json')
    source_bad=[p for p,h in registration['source_sha256'].items() if common.sha(common.ROOT/p)!=h]
    check('supplement_registered_sources',not source_bad,source_bad)

    # Byte hashes of all scientific v5 artifacts, excluding runtime and QA fixtures.
    data_files=[]
    for p in sorted(common.DATA.rglob('*')):
        if p.is_file() and not p.relative_to(common.DATA).parts[0].startswith('_'):
            data_files.append({'path':common.relative(p),'bytes':p.stat().st_size,'sha256':common.sha(p)})
    index={v['path']:v for v in data_files}
    write('DATA_FILE_INVENTORY.json',{'schema':'v5-scientific-payload-inventory-v1','time':common.now(),'files':data_files,'total_bytes':sum(v['bytes'] for v in data_files),'excluded':['runtime dependencies and caches','small implementation fixtures'],'sealed_payloads_read':False})
    manifest_counts={}
    manifest_rows={}
    bad=[]
    for family,n in [('oracle_data',180),('development',540),('null_calibration',96),('null_evaluation',96),('calibration_stress',36)]:
        m=common.read_json(common.DATA/family/'data_manifest.json')
        rows=m['rows']; manifest_rows[family]={r['case_id']:r for r in rows}
        manifest_counts[family]={'rows':len(rows),'mothers':len(m['selected_mothers']),'labels':len(m['labels'])}
        check('manifest_'+family,len(rows)==n and len(manifest_rows[family])==n and all(r['raw_available'] for r in rows))
        for row in rows:
            for field,digestfield in [('raw_path','raw_file_sha256'),('label_path','label_sha256')]:
                rel=common.relative(common.DATA/family/row[field])
                if index[rel]['sha256']!=row[digestfield]:bad.append(rel)
    check('all_raw_label_hashes_match_manifests',not bad,bad)

    records=list(common.jsonl(D/'records.jsonl'))
    selected=list(common.jsonl(D/'selected_records.jsonl'))
    oracle=list(common.jsonl(R/'ORACLE_LADDER.jsonl'))
    bases=list(common.jsonl(R/'ORACLE_BASELINES_84.jsonl'))
    check('oracle_180x4',len(oracle)==720 and {(r['case_id'],r['level']) for r in oracle}=={(c,level) for c in manifest_rows['oracle_data'] for level in ['O0','O1','O2','O3']})
    check('oracle_baseline_84x4',len(bases)==336 and len({r['case_id'] for r in bases})==84 and len({(r['case_id'],r['method']) for r in bases})==336)
    expected={(c,m,a) for c in manifest_rows['development'] for m,aa in [('v5_physics',[0,.01,.1,1])]+[(m,[None]) for m in protocol['baselines']] for a in aa}
    keys=[(r['case_id'],r['method'],r.get('alpha_g')) for r in records]
    check('development_exact_4320_grid',len(keys)==4320 and len(set(keys))==4320 and set(keys)==expected)
    check('selected_exact_540_alpha1',len(selected)==540 and {r['case_id'] for r in selected}==set(manifest_rows['development']) and all(r['alpha_g']==1 for r in selected))
    all_result_refs=[]
    for row in records+selected+oracle+bases:
        if 'prediction_path' in row:
            all_result_refs.append({'path':row['prediction_path'],'sha256':row['prediction_sha256']})
        for k in ['prediction_cache','prediction']:
            v=row.get(k)
            if isinstance(v,dict) and 'path' in v and 'sha256' in v:
                all_result_refs.append(v)
    check('prediction_cache_record_hashes',all(index[v['path']]['sha256']==v['sha256'] for v in all_result_refs),{'references':len(all_result_refs)})
    mc=common.read_json(D/'mc_coverage.json')
    check('MC_exact_12_cases',len(mc['cases'])==12 and {r['case_id'] for r in mc['cases']}==set(mc['predeclared_case_ids']) and all(r['status']=='completed' for r in mc['cases']))
    mc_arrays=[]
    for r in mc['cases']:
        v=r['uncertainty_arrays']; a=common.arrays(common.ROOT/v['path'])
        mc_arrays.append({'case_id':r['case_id'],'hash_matched':index[v['path']]['sha256']==v['sha256'],'fields':list(a),'all_shapes':{k:list(vv.shape) for k,vv in a.items()},'nonfinite_counts':{k:int((~np.isfinite(vv)).sum()) for k,vv in a.items()}})
    check('MC_array_hashes_and_finiteness',all(v['hash_matched'] and not any(v['nonfinite_counts'].values()) for v in mc_arrays))
    write('MC_ARRAY_REOPEN_REVIEW.json',{'cases':mc_arrays,'reran_MC':False,'original_three_repeat_logs_unavailable':True})
    h=common.read_json(D/'hessian_diagnostics.json');hs=h['summary']
    check('Hessian_8_actual_probe_ids',len(hs['solves'])==8 and {r['probe'] for r in hs['solves']}==set(range(8)))
    stress=list(common.jsonl(D/'calibration_stress_records.jsonl'))
    skeys={(r['case_id'],r['method']) for r in stress}
    check('stress_exact_36x5',len(stress)==180 and skeys=={(c,m) for c in manifest_rows['calibration_stress'] for m in ['v5_physics']+protocol['baselines']} and all(r['status']=='SUCCESS' for r in stress))
    cal=list(common.jsonl(D/'null_calibration_records.jsonl'));ev=list(common.jsonl(D/'null_evaluation_records.jsonl'))
    check('null_96_plus_96_disjoint',len(cal)==len(ev)==96 and len({r['mother_id'] for r in cal})==len({r['mother_id'] for r in ev})==12 and not ({r['mother_id'] for r in cal}&{r['mother_id'] for r in ev}))
    paired=common.read_json(D/'paired_statistics.json')
    for oldpair in paired['comparisons']:
        p=oldpair['population']
        fresh=paired_statistics(records,'v5_physics','v3_physics',group=p['group'],input_kind='noise',sign=p['sign'],alpha_g=1)
        check('paired_recomputed_'+p['group']+'_'+str(p['sign']),fresh==oldpair)

    nominal=[r for r in selected if r['input_kind']=='noise']
    summary={'schema':'v5-root-saved-result-summary-v1','time':common.now(),'nominal':describe(nominal),'by_condition':{},'all_five_methods_nominal':{},'stress':{},'grid_statuses':dict(Counter(r['status'] for r in records)),'grid_failed_rows':[{k:r.get(k) for k in ['case_id','method','alpha_g','status','failure']} for r in records if r['status']!='SUCCESS'],'manifest_counts':manifest_counts,'weak_interval_coverage_measured':False}
    for condition,rr in grouped(nominal,lambda r:r['condition']).items():summary['by_condition'][condition]=describe(rr)
    for method in ['v5_physics']+protocol['baselines']:
        rr=nominal if method=='v5_physics' else [r for r in records if r['method']==method and r['input_kind']=='noise']
        summary['all_five_methods_nominal'][method]=describe(rr)
    for (kind,method),rr in grouped(stress,lambda r:(r['stress']['name'],r['method'])).items():
        summary['stress'].setdefault(kind,{})[method]=describe(rr)
    weak_review=[]
    for r in nominal:
        if r['group']!='weak':continue
        row=manifest_rows['development'][r['case_id']]
        p=common.arrays(common.ROOT/r['prediction_cache']['path']);t=common.arrays(common.DATA/'development'/row['label_path'])
        pred=p['a1__template_postblur_e'].squeeze();truth=t['template_postblur_e'].squeeze();valid=p['a1__valid'].squeeze().astype(bool)
        threshold=r['noise_proxy']['frames'][0]['sigma_proxy_e']*r['detection']['frame_threshold']
        domain=t['evaluation_domain'].astype(bool);support=domain&(abs(truth)>=.2*float(abs(truth[domain]).max()))
        detected=(abs(pred)>threshold)&valid
        correct=detected&support&(np.sign(pred)==np.sign(truth))
        item={'case_id':r['case_id'],'mother_id':r['mother_id'],'sign':r['sign'],'detected_pixels':int(detected.sum()),'support_pixels':int(support.sum()),'correct_sign_detected_support_pixels':int(correct.sum()),'off_support_detected_pixels':int((detected&~support).sum()),'wrong_sign_detected_support_pixels':int((detected&support&~correct).sum()),'L2':r['metrics']['template_relative_L2'],'threshold_e':threshold}
        weak_review.append(item)
        check('weak_rescore_'+r['case_id'],item['detected_pixels']==r['detection']['detected_pixels'] and item['correct_sign_detected_support_pixels']==r['metrics']['signed_detected_pixels'])
    summary['weak_detection_array_rescore']=weak_review
    write('RESULT_SUMMARY.json',summary)
    with (O/'PER_MOTHER_NOMINAL.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['mother_id','condition','method','cases','temperature_RMSE_K','temperature_bias_K','template_RMSE_e','template_relative_L2','signed_projection','valid_fraction','raw_weighted_RMSE','signed_threshold_recall']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for (mid,condition,method),rr in sorted(grouped(nominal+[r for r in records if r['method'] in protocol['baselines'] and r['input_kind']=='noise'],lambda r:(r['mother_id'],r['condition'],r['method'])).items()):
            a=describe(rr);writer.writerow({'mother_id':mid,'condition':condition,'method':method,**{k:a[k] for k in fields[3:]}})
    absent=[common.DATA/name for name in ['training','signed_confirmation','final_once']]+[R/'FINAL_ONCE_STARTED.json',R/'FINAL_ONCE_RECEIPT.json',R/'final_protocol.json',R/'candidate_review/candidate_receipt.json',R/'PHASE_ACTIVE.json']
    check('training_and_sealed_evaluation_not_started',all(not p.exists() for p in absent),[common.relative(p) for p in absent])
    write('ROOT_DATA_AUDIT.json',{'schema':'v5-root-actual-evidence-audit-v1','started_at':started,'finished_at':common.now(),'checks':checks,'all_audit_checks_passed':not failures,'audit_failures':failures,'all_scientific_gates_passed':False,'data_payload_files':len(data_files),'data_payload_bytes':sum(v['bytes'] for v in data_files),'old_array_check_scope':'size, nanosecond mtime and read-only attributes only; unopened old/sealed payloads were not read','new_reconstruction_calls':0,'new_optimizer_updates':0,'sealed_payloads_read':0})
    print(json.dumps({'audit_passed':not failures,'audit_failures':failures,'data_files':len(data_files),'data_bytes':sum(v['bytes'] for v in data_files),'report':common.relative(O/'RESULT_SUMMARY.json')}))
    assert not failures, failures


if __name__=='__main__':main()
