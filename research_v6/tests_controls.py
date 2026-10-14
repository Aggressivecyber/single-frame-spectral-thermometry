"""Bounded gate and fixed-denominator controls with labeled synthetic receipts."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from . import common
common.configure_runtime()
from . import reports,scoring


def _fixture(template_ratio=.9):
    rows=[]
    for i in range(8):
        sign='positive' if i%2==0 else 'negative'
        for source in ('C_true','C_v5','C_v3','original_v3'):
            modes=('joint_original',) if source=='original_v3' else ('true_beta','estimated_beta')
            for mode in modes:
                amplitude={'C_true':.7,'C_v5':2.,'C_v3':template_ratio,'original_v3':1.}[source]
                temperature={'C_true':0.,'C_v5':2.,'C_v3':.8,'original_v3':.8}[source]
                n=4
                metrics={'domain_pixels':n,'accepted_pixels':n,'template_SSE':amplitude**2*n,
                 'temperature_SSE_K2':temperature**2*n,'template_truth_energy':n*100,
                 'template_error_sum':0.,'temperature_error_sum_K':0.,'signed_projection_numerator':n*100}
                rows.append({'case_id':f'fixture_{i}','mother_id':f'fixture_mother_{i//2}',
                  'cohort':'development','input_kind':'noise','group':'normal','sign':sign,
                  'method':source+'__'+mode,'continuum_source':source,'beta_mode':mode,
                  'raw_sha256':f'fixture_raw_{i}','metrics':metrics,'status':'SUCCESS',
                  'numerical_execution_passed':True,'diagnostics':{}})
    return rows


def run():
    criteria={'v3_ratio_max':1.05,'Cv5_improvement_RMSE_fraction_min':.25,
      'temperature_improvement_fraction_for_B':.10,'O0_normal_noise_vs_v3_RMSE_ratio_max':1.05,
      'oracle_noncollapse_projection_min':.8}
    components=[{'numerical_gate_passed':True} for _ in range(48)]
    checks=[]
    def add(name,value):checks.append({'name':name,'passed':bool(value)})
    def decide(rows,decomposition=components,processed=8,stop=None):
        return reports.decide_A(rows,decomposition,planned_raw=8,processed_raw=processed,stop_reason=stop,criteria=criteria)
    sufficient=decide(_fixture())
    add('same_raw_both_signs_continuum_route_sufficient',sufficient['status']=='CONTINUUM_ROUTE_SUPPORTED' and not sufficient['B_eligible'])
    necessary=decide(_fixture(1.2))
    add('coupled_route_only_after_clear_temperature_but_template_failure',necessary['status']=='B_REQUIRED' and necessary['B_eligible'])
    bad=decide(_fixture(1.2),components[:-1]+[{'numerical_gate_passed':False}])
    add('one_failed_identity_blocks_B',bad['status']=='STOP_NUMERICAL_OR_ORACLE_AUDIT' and not bad['B_eligible'])
    bad=decide(_fixture(1.2),processed=7)
    add('incomplete_A_blocks_B',not bad['A_complete'] and not bad['B_eligible'])
    rows=_fixture(1.2);rows[-1]['metrics']=None;rows[-1]['numerical_execution_passed']=False
    bad=decide(rows)
    add('missing_prediction_remains_in_fixed_population',bad['returned_v3_rows']==8 and not bad['B_eligible'])
    a=scoring.aggregate([r for r in rows if r['method']=='original_v3__joint_original'])
    add('aggregate_does_not_drop_failed_frame',a['cases']==8 and a['mothers']==4 and not a['complete'] and a['template_RMSE_e'] is None)
    rows=_fixture();rows[0]['status']='NOT_EXECUTED'
    bad=decide(rows)
    add('unexecuted_rows_are_not_returned_estimates',bad['returned_linear_rows']==47 and not bad['A_complete'])
    paired=scoring.paired(_fixture(),'C_v3__estimated_beta','C_v5__estimated_beta',sign='positive')
    add('paired_unit_is_mother_not_pixels',paired['mother_MSE_difference']['mothers']==4 and paired['case_pairs']==4)
    corrupt=_fixture();next(r for r in corrupt if r['method']=='C_v3__estimated_beta')['raw_sha256']='wrong_fixture_raw'
    try:scoring.paired(corrupt,'C_v3__estimated_beta','C_v5__estimated_beta')
    except AssertionError:add('mismatched_raw_pair_rejected',True)
    else:add('mismatched_raw_pair_rejected',False)
    before=common.source_hashes()
    for name in ('protocol.py','run.py','stage_a.py','reports.py','scoring.py','common.py','tests_controls.py'):
        tree=ast.parse((common.CODE/name).read_text(encoding='utf-8-sig'))
        add('connected_source_parse_'+name,bool(tree))
    from .run import dispatch
    try:dispatch('unknown','cpu')
    except (FileNotFoundError,ValueError):add('missing_registration_rejected_before_scientific_data_access',True)
    else:add('missing_registration_rejected_before_scientific_data_access',False)
    add('fixture_execution_source_unchanged',common.source_hashes()==before)
    return {'schema':'v6-root-gate-controls-001','time':common.now(),'all_passed':all(r['passed'] for r in checks),
      'device':'cpu','fixture_only_not_scientific_acceptance':True,'checks':checks,
      'source_sha256':{common.relative(common.CODE/n):common.sha(common.CODE/n) for n in
           ('protocol.py','run.py','stage_a.py','reports.py','scoring.py','common.py','tests_controls.py')},
      'formal_or_sealed_payloads_opened':False,'formal_raw_generated':False,'training_executed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();r=run();common.write_json(args.output,r)
    print({'all_passed':r['all_passed'],'checks':len(r['checks']),'receipt':str(args.output)},flush=True)
    if not r['all_passed']:
        print([x for x in r['checks'] if not x['passed']],flush=True);raise SystemExit(1)


if __name__=='__main__':main()
