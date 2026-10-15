"""Retrospective split-draw scale diagnostic; not new independent coverage.

All inputs are already-opened frozen-v3 MC statistics. First32 draws supply
oracle diagnostic SDs; the other32 supply calibration or evaluation errors.
The old geometry confounding is retained explicitly. No inverse is rerun.
"""
from . import storage as s
from . import intervals, uncertainty as u
import collections
import json
import numpy as np


def run():
    path=s.REPORTS/'full_inverse_MC.jsonl'
    original=list(s.rows(path))
    plan=s.read(s.REPORTS/'SCIENTIFIC_PLAN.json')
    cases={r['mother_id']:r for r in plan['cases'] if r['family']=='uncertainty'}
    legacy=s.read(s.REPORTS/'INTERVAL_CALIBRATION.json')
    records=[];scales=[];legacy_rows=[];split_disjoint=True
    for mid,row in cases.items():
        unit={r['replicate']:r for r in original if r['mother_id']==mid}
        if not set(range(64)).issubset(unit):
            raise ValueError('Original first64 MC draws are incomplete: '+mid)
        first=[unit[j].get('statistics') for j in range(32)]
        value=u.empirical_statistic_scale(first,kind='independent_full_inverse_mc_sd',declared_replicates=32)
        scales.append({'mother_id':mid,'role':row['role'],**value})
        source=set(range(32));targets=set(range(32,64));split_disjoint &= not bool(source&targets)
        legacy_covered=np.zeros(12);legacy_width=[];legacy_finite=0
        for j in range(32,64):
            item=unit[j]
            records.append({'mother_id':mid,'role':row['role'],'replicate':j-32,
                'original_replicate':j,'statistics':item.get('statistics'),
                'truth_statistics':item.get('truth_statistics'),
                'scale':value['scale'],'scale_provenance':value['scale_provenance'],
                'actual_temperature_geometry':row['mother']['temperature_geometry'],
                'actual_template_geometry':row['mother']['template_geometry']})
            if item.get('statistics') is not None:
                ci=intervals.legacy_interval(item['statistics'],item['noise_feature'],legacy)
                if ci is not None:
                    truth=np.asarray(item['truth_statistics'])
                    legacy_covered+=(truth>=ci[:,0])&(truth<=ci[:,1])
                    legacy_width.append(ci[:,1]-ci[:,0]);legacy_finite+=1
        legacy_rows.append({'mother_id':mid,'role':row['role'],'declared_draws':32,
            'finite_draws':legacy_finite,'coverage':(legacy_covered/32).tolist(),
            'mean_width':None if not legacy_width else np.mean(legacy_width,axis=0).tolist()})
    cal=[r for r in records if r['role']=='interval_calibration']
    evaluation=[r for r in records if r['role']=='interval_evaluation']
    counts_cal={mid:32 for mid,row in cases.items() if row['role']=='interval_calibration'}
    counts_eval={mid:32 for mid,row in cases.items() if row['role']=='interval_evaluation'}
    model=intervals.calibrate(cal,declared_draws_by_mother=counts_cal)
    score=intervals.evaluate(evaluation,model,declared_draws_by_mother=counts_eval)
    legacy_eval=[r for r in legacy_rows if r['role']=='interval_evaluation']
    old_geometry={role:dict(collections.Counter(r['mother']['template_geometry'] for r in cases.values() if r['role']==role))
                  for role in ('interval_calibration','interval_evaluation')}
    checks={
        'original896_draws_retained_unchanged':len(original)==896,
        'disjoint_scale_and_error_draw_indices':bool(split_disjoint),
        'calibration_evaluation_mothers_disjoint':not set(counts_cal)&set(counts_eval),
        'six_mothers_and192_error_draws_each_role':len(counts_cal)==len(counts_eval)==6 and len(cal)==len(evaluation)==192,
        'all_new_scales_are_diagnostic_oracle':all(r['scale_provenance']['oracle_diagnostic'] and not r['scale_provenance']['deployment_scale'] for r in scales),
        'evaluation_did_not_recalibrate':not score['model_recalibrated'],
        'v2_interval_model_not_deployment_qualified':not model['deployment_qualified'],
    }
    assert all(checks.values()),checks
    result={'time':s.now(),'schema':'retrospective-MC-scale-diagnostic-v2',
        'interpretation':'already examined data, oracle per-scene SD,32-draw development diagnostic; no new independent coverage evidence',
        'new_complete_inverse_calls':0,'original_MC_draws':len(original),
        'original_journal_sha256':s.sha(path),'original_plan_sha256':s.sha(s.REPORTS/'SCIENTIFIC_PLAN.json'),
        'original_legacy_interval_model_sha256':s.sha(s.REPORTS/'INTERVAL_CALIBRATION.json'),
        'scale_draws_per_mother':32,'error_draws_per_mother':32,
        'source_replicate_ranges':{'scale':[0,31],'errors':[32,63]},
        'local_record_index': 'error replicate index = original replicate minus32; original index retained',
        'extra128_draws':'preserved original prefix-extension stability evidence; outside this diagnostic split',
        'old_geometry_counts':old_geometry,'role_geometry_confounding_repaired_in_this_data':False,
        'new_balanced_data_still_required':True,'output_scale_records':scales,
        'new_diagnostic_model':model,'new_diagnostic_evaluation':score,
        'legacy_on_same_evaluation32_draws':{
            'rows':legacy_eval,'pooled_coverage':np.mean([r['coverage'] for r in legacy_eval],axis=0).tolist(),
            'mean_width':np.mean([r['mean_width'] for r in legacy_eval],axis=0).tolist(),
            'legacy_calibration_draw_count_differs':True,
            'not_a_fair_model_selection_or_superiority_test':True},
        'all_passed':all(checks.values()),'checks':checks,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/mc_interval_diagnostic.py','research_closeout/uncertainty.py','research_closeout/intervals.py')},
        'final_payloads_opened':False,'old_reports_rewritten':False,'scientific_95_coverage_guaranteed':False}
    s.write(s.REPORTS/'repair_001/RETROSPECTIVE_INTERVAL_DIAGNOSTIC.json',s.plain(result))
    print(json.dumps({'all_passed':result['all_passed'],'checks':checks,
        'diagnostic_coverage':score['pooled_evaluation_marginal_coverage'],
        'mean_diagnostic_width':np.mean([r['mean_interval_width'] for r in score['rows']],axis=0).tolist(),
        'legacy_width':result['legacy_on_same_evaluation32_draws']['mean_width'],
        'new_full_inverse_calls':0},indent=2))


if __name__=='__main__':run()
