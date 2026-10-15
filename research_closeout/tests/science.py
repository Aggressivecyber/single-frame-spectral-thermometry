"""Exposure units, real disjoint metadata, interval separation and raw-only scope."""
from .. import storage as s
from .. import measurement as m
from .. import scientific_plan as planning
from .. import scenarios,intervals,uncertainty as u
from ..deployment import known_exposure
from ..independent_renderer import sample_raw
import inspect
import json
import numpy as np
import torch

def run():
    torch.set_num_threads(2);p=m.physics('cpu');plan=planning.build();checks=[]
    def check(name,passed,**evidence): checks.append({'name':name,'passed':bool(passed),**evidence})
    counts=plan['case_counts']
    check('full_preregistered_mother_and_case_matrix',counts=={'calibration':45,'dynamic':12,'independent':146,
        'null_calibration':200,'null_evaluation':200,'photons':90,'structure':110,'uncertainty':12},counts=counts)
    check('original_train_development_final_and_new_streams_disjoint',all(v for k,v in plan['disjointness'].items() if isinstance(v,bool)),evidence=plan['disjointness'])
    spec=s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    calls=sum(not r['reused'] for r in s.rows(s.REPORTS/'inverse_calls.jsonl'))
    ceiling=plan['prospective_new_call_ceiling']
    check('complete_matrix_fits_original_call_and_phase_caps',all(n<=spec['prospective_phase_call_caps'][k] for k,n in ceiling.items())
        and calls+sum(ceiling.values())<=12000,existing_new_calls=calls,prospective=ceiling)
    state=torch.zeros((1,5,256,256),dtype=torch.float64);state[:,0]=-.15;state[:,1]=-.6;state[:,4]=-.2
    beta=torch.tensor([[.5,.03,-.01]],dtype=torch.float64);before=p(state,beta);adc=p.adc_step;variance=p.extra_variance;old=p.continuum_lut
    with known_exposure(p,.25):
        difference=float((p(state,beta)-before*.25).abs().max())
        same_units=p.adc_step==adc and p.extra_variance==variance
    check('known_photon_budget_all_components_and_ADC_read_units',difference<1e-8 and same_units,
        raw_maximum_difference_e=difference,ADC_step_e=adc,extra_variance_e2=variance)
    check('calibration_context_restores_original_buffers',p.continuum_lut is old and p.cfg==m.selection()['physics_config'])
    area=2*np.pi*(256/64)**2
    totals=[float(scenarios.structure(p,g,'fixed_total_signal').sum()) for g in planning.GEOMETRIES]
    peaks=[float(scenarios.structure(p,g,'fixed_peak').max()) for g in planning.GEOMETRIES]
    check('structure_peak_and_total_signal_are_distinct_controls',max(abs(v-area) for v in totals)<1e-8 and all(v==1 for v in peaks),fixed_absolute_area_px=area)
    row=next(r for r in plan['cases'] if r['family']=='photons' and r['photon_ratio']==1.)
    a=scenarios.render_core(row,p);b=scenarios.render_core({**row,'photon_ratio':2.},p)
    check('controlled_generation_preserves_reference_electron_target',np.array_equal(b['expected_e'],2*a['expected_e'])
        and np.array_equal(a['template_postblur_e_ref'],b['template_postblur_e_ref']) and a['raw_available'])
    raw,receipt=sample_raw(a['expected_e'],p.cfg,row['noise_seed']);raw2,_=sample_raw(a['expected_e'],p.cfg,row['noise_seed'])
    check('nominal_core_raw_seed_and_quantization_replay',np.array_equal(raw,raw2) and receipt['Poisson_read_ADC_once'])
    rng=np.random.default_rng(4006015);records=[]
    for mother in range(12):
        for replicate in range(64):
            records.append({'mother_id':f'fixture_mother{mother}','replicate':replicate,
                'role':'interval_calibration' if mother<6 else 'interval_evaluation',
                'statistics':rng.normal(size=12).tolist(),'truth_statistics':[0.]*12,'noise_feature':1.})
    model=intervals.legacy_calibrate([r for r in records if r['role']=='interval_calibration'])
    result=intervals.legacy_evaluate(records,model)
    check('intervals_calibrated_then_evaluated_on_different_mothers',model['independent_mothers']==6
        and result['evaluation_independent_mothers']==6 and not model['evaluation_labels_used']
        and all(v>0 for v in model['constants']) and not result['simultaneous_full_image_claimed'],
        empirical_fixture_coverage=result['pooled_evaluation_marginal_coverage'])
    rejected=False
    try: intervals.legacy_calibrate(records)
    except ValueError: rejected=True
    check('evaluation_labels_rejected_by_interval_calibrator',rejected)
    # Same shared-raw derivative, now with persisted sufficient statistics.
    mean=np.full((16,16),3000.);estimate={'predicted_raw_e':mean}
    step=adc
    def stats_refit(dn): return {'statistics':np.full(12,np.asarray(dn)[0,0]*step)}
    # Match the configured shape; no formal inverse or scientific raw is used.
    small_cfg={**p.cfg,'shape':[16,16]}
    local=u.legacy_whole_inverse_linearization(stats_refit,estimate,small_cfg,seed=81342,probes=8,
        summarize=lambda out:out['statistics'])
    variance_reference=3000+p.extra_variance
    covariance_error=abs(local['covariance'][0][0]-variance_reference)/variance_reference
    check('whole_inverse_linearization_accepts_cached_sufficient_statistics',max(local['step_size_relative_discrepancies'])<1e-7
        and covariance_error<1e-7,maximum_step_size_discrepancy=max(local['step_size_relative_discrepancies']),
        covariance_relative_error=covariance_error)
    from ..deployment import FrozenPrototype
    parameters=set(inspect.signature(FrozenPrototype.infer).parameters)
    check('deployment_boundary_has_no_truth_or_identity_parameters',parameters=={'self','raw_dn','method','photon_ratio','detector','application_domain'})
    check('final_signed_factors_and_original_main_counts',len(plan['final_signed_metadata'])==180
        and len(plan['final_main_metadata'])==512 and not plan['sealed_payloads_opened'])
    result={'time':s.now(),'all_passed':all(c['passed'] for c in checks),'tests':checks,
        'formal_raws_saved':False,'formal_inverses_executed':0,'sealed_payloads_opened':False,
        'source_sha256':{path.relative_to(s.ROOT).as_posix():s.sha(path) for path in (s.ROOT/'research_closeout').rglob('*.py')}}
    s.write(s.REPORTS/'SCIENCE_CONTRACTS.json',s.plain(result),replace=True)
    print(json.dumps(s.plain(result),ensure_ascii=True),flush=True);return result

if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
