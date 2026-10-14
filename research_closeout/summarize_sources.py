"""Aggregate finished source interventions without executing another inverse."""
from . import storage as s
from collections import defaultdict
import math
import json

def run():
    path=s.REPORTS/'source_accounting.jsonl';rows=list(s.rows(path));assert len(rows)==540
    groups=defaultdict(list)
    for row in rows: groups[(row['source'],row['beta_mode'],row['condition'],row['input_kind'])].append(row)
    grouped=[]
    for (source,beta,condition,kind),values in sorted(groups.items()):
        components=values[0]['four_terms']['components'];cross=values[0]['four_terms']['cross_terms']
        grouped.append({'source':source,'beta_mode':beta,'condition':condition,'input_kind':kind,
            'cases':len(values),'mothers':len({r['mother_id'] for r in values}),
            'actual_template_RMSE_e':math.sqrt(sum(r['actual_output_error_RMS_e']**2 for r in values)/len(values)),
            'component_MSE_e2':{key:sum(r['four_terms']['components'][key]['MSE'] for r in values)/len(values) for key in components},
            'signed_cross_contribution_e2':{key:sum(r['four_terms']['cross_terms'][key] for r in values)/len(values) for key in cross},
            'certified_rows':sum(r['numerical_certified'] for r in values)})
    result={'time':s.now(),'raw_inputs':90,'mothers':6,'source_background_combinations':540,
        'source_journal':{'path':str(path.relative_to(s.ROOT)),'sha256':s.sha(path)},
        'certified_rows':sum(r['numerical_certified'] for r in rows),
        'maximum_corrected_four_term_closure_RMS_e':max(r['corrected_four_term_closure_RMS_e'] for r in rows),
        'maximum_actual_numeric_error_RMS_e':max(r['four_terms']['components']['actual_numerical_error']['RMS'] for r in rows),
        'maximum_component_residual_correction_RMS_e':max(v for r in rows for v in r['component_residual_correction_RMS_e'].values()),
        'closure_is_not_camera_accuracy':True,'oracle_entries_are_mechanism_evidence_only':True,
        'cross_terms_are_signed_contributions_and_not_independent_variances':True,
        'original_K_W_lambda_preserved':all(r['original_K_W_lambda_preserved'] for r in rows),
        'old_A_36_mother_extension_reused':{'path':'reports/closeout_001/A_REUSED_COMPARISON.json',
            'sha256':s.sha(s.REPORTS/'A_REUSED_COMPARISON.json')},'rows':grouped}
    s.write(s.REPORTS/'SOURCE_STUDY_SUMMARY.json',result)
    return result

if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k!='rows'},ensure_ascii=True),flush=True)
