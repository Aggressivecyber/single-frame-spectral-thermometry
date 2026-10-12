"""Output-only summaries from retained outcomes; no inference or training changes."""
from collections import defaultdict
import numpy as np
from training_v4_1.common import *


def summarize_detection(records, method):
    buckets=defaultdict(list)
    for row in records:
        # Signed strengths stay separate rather than allowing normal targets to
        # dominate a combined negative relative error.
        group=row['condition'] if row['signed_aux'] else row['group']
        buckets[(row['input_kind'],row['signed_aux'],group)].append(row)
    result=[]
    for (kind,signed,group), rows in sorted(buckets.items()):
        good=[r for r in rows if r['status']=='SUCCESS']; metrics=[r['metrics'] for r in good]
        pixels=sum(m['pixels'] for m in metrics); energy=sum(m['truth_energy'] for m in metrics)
        sse=sum(m['template_SSE'] for m in metrics)
        item={'method':method,'input_kind':kind,'signed_aux':signed,'condition':group,'frames':len(rows),
            'failures':len(rows)-len(good),'template_RMSE_e':float(np.sqrt(sse/pixels)) if pixels else None,
            'template_L2':float(np.sqrt(sse/energy)) if energy else None,'detection_threshold_e':60.}
        for key in ('positive_recall60','negative_recall60','positive_blank_FP60','negative_blank_FP60',
                    'gradient_energy','truth_gradient_energy','signed_projection'):
            values=[m[key] for m in metrics if m[key] is not None]
            item['mean_'+key]=float(np.mean(values)) if values else None
        result.append(item)
    return result


def make_diagnostics():
    selection=read(OUT/'candidate_selection.json')
    baseline_records=Journal(OUT/'baseline_per_case.jsonl').rows()
    original=Journal(OUT/'original_16_baseline_per_case.jsonl').rows()
    lookup={(r['case_id'],r['method']):r for r in original}
    detection=[]; protection=[]; original_pairs=[]
    for method in sorted({r['method'] for r in baseline_records}):
        detection.extend(summarize_detection([r for r in baseline_records if r['method']==method],method))
    for run in selection['completed_runs']:
        folder=OUT/run
        for event in Journal(folder/'validation_curve.jsonl').rows():
            update=event['optimizer_update']
            records=Journal(folder/f'validation_update_{update:04d}.jsonl').rows()
            detection.extend(summarize_detection(records,run+'/'+str(update)))
            protected=[r for r in records if r['input_kind']!='noise']
            protection.append({'run':run,'optimizer_update':update,'frames':len(protected),
                'template_degraded_frames':sum(r['metrics']['template_MSE_e2']>r['physical']['template_MSE_e2']+.25 for r in protected),
                'temperature_degraded_frames':sum(r['metrics']['temperature_MSE_K2']>r['physical']['temperature_MSE_K2']+.04 for r in protected),
                'passed_subgroups':sum(g['passed'] for g in event['protection_groups']),
                'total_subgroups':len(event['protection_groups']),
                'small_residual_large_correction_fraction':event['small_residual_large_correction_fraction']})
            gates=read(folder/f'gates_update_{update:04d}.json')['decisions']
            for gate in gates:
                for method in ('zero','old_B4','fixed_physics','v4_update100','v4_update2000'):
                    reference=lookup[(gate['case_id'],method)]
                    m=reference['metrics']
                    original_pairs.append({'run':run,'optimizer_update':update,'case_id':gate['case_id'],
                        'input_hash':reference['input_hash'],'baseline':method,
                        'candidate_passed':gate['passed'],'candidate_template_RMSE_e':gate['template_RMSE_e'],
                        'candidate_color_RMSE_K':gate['color_RMSE_K'],
                        'template_MSE_delta':gate['template_RMSE_e']**2-m['template_MSE_e2'],
                        'color_MSE_delta':gate['color_RMSE_K']**2-m['temperature_MSE_K2'] if m['temperature_MSE_K2'] is not None else None,
                        'source':'candidate gate and baseline use the identical frozen electron input and selected physical cache'})
    write(OUT/'signed_detection_and_detail_summary.json',detection)
    write(OUT/'protection_checkpoint_diagnostics.json',protection)
    destination=OUT/'original_16_paired_checkpoints.jsonl'
    import json
    with destination.open('w',encoding='utf-8') as f:
        for r in original_pairs: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    return detection,protection
