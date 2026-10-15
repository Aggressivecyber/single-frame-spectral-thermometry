"""Small CPU contracts for repaired statistics; no formal inverse or scene run."""
from .. import storage as s
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .. import intervals, metrics, uncertainty as u
import copy
import json
from types import SimpleNamespace
import numpy as np
import torch


def run():
    torch.set_num_threads(2)
    checks=[];callback_count=[0]

    def test(name,fn):
        try:
            value=fn()
            evidence=value if isinstance(value,dict) else {'passed':bool(value)}
            checks.append({'name':name,**evidence})
        except Exception as error:
            checks.append({'name':name,'passed':False,'error_type':type(error).__name__,'error':str(error)})

    def rejected(fn):
        try: fn()
        except ValueError: return True
        return False

    cfg={'shape':[4,4],'full_well_e':255.,'adc_bits':8,'black_level_DN':0.,'read_noise_e':2.}
    mean=np.full((4,4),100.)
    estimate={'predicted_raw_e':mean}
    sigma2=100.+4.+1./12
    coefficients=np.array([2.,3.,4.,5.,6.,7.,-3.,-4.,-5.,-6.,-7.,-8.])

    def linear_refit(dn):
        callback_count[0]+=1
        return {'statistics':coefficients*np.asarray(dn)[0,0],
                'numerical_valid':True,'new_inverse_calls':0}

    def linear(**kwargs):
        return u.whole_inverse_linearization(linear_refit,estimate,cfg,seed=501003,
            summarize=lambda out:out['statistics'],**kwargs)

    local=linear()
    reference=np.outer(coefficients,coefficients)*sigma2
    error=float(np.max(np.abs(np.asarray(local['diagnostic_covariance'])-reference)))
    test('default_two_probes_three_steps_exact_twelve_callbacks',lambda:{'passed':
        local['actual_callback_calls']==local['planned_full_inverse_calls']==12 and local['new_inverse_calls']==0,
        'actual_callbacks':local['actual_callback_calls'],'new_formal_inverse_calls':local['new_inverse_calls']})
    test('linear_shared_raw_cross_covariance_exact_in_output_units',lambda:{'passed':
        error<1e-7 and np.asarray(local['diagnostic_covariance'])[0,6]<0,
        'maximum_absolute_covariance_error':error,'cross_covariance_K_e':local['diagnostic_covariance'][0][6],
        'analytic_cross_covariance_K_e':float(reference[0,6])})
    test('low_probe_stable_diagnostic_never_becomes_interval_scale',lambda:
        local['step_size_check_passed'] and local['covariance'] is None and local['scale'] is None
        and not local['interval_scale_eligible'] and local['covariance_rank']==1 and local['rank_bound']==2)

    def eligible_linear():
        value=linear(probes=32,max_full_inverse_calls=192,probe_convergence_verified=True)
        maximum=float(np.max(abs(np.asarray(value['covariance'])-reference)))
        return {'passed':value['interval_scale_eligible'] and maximum<1e-7,
                'callbacks':value['actual_callback_calls'],'analytic_covariance_error':maximum,
                'convergence_evidence':'all probe second moments equal the analytic rank-one linear covariance'}
    test('analytically_verified_linear_probe_budget_can_supply_output_scale',eligible_linear)

    def unstable_refit(dn):
        callback_count[0]+=1
        delta=(np.asarray(dn)[0,0]-mean[0,0])/np.sqrt(sigma2)
        value=np.full(12,1e8*delta);value[-1]=1000*delta**3
        return {'statistics':value,'numerical_valid':True,'new_inverse_calls':0}
    unstable=u.whole_inverse_linearization(unstable_refit,estimate,cfg,seed=501003,
        summarize=lambda out:out['statistics'])
    test('large_K_statistics_cannot_hide_unstable_template_statistic',lambda:{'passed':
        all(all(row['all_probes_passed_per_statistic'][:6]) for row in unstable['step_comparisons'])
        and all(not row['all_probes_passed_per_statistic'][-1] for row in unstable['step_comparisons'])
        and not unstable['step_size_check_passed'] and unstable['covariance'] is None and unstable['scale'] is None
        and unstable['diagnostic_covariance'] is not None,
        'per_pair_final_template_statistic_passed':[r['all_probes_passed_per_statistic'][-1] for r in unstable['step_comparisons']]})

    def zero_refit(dn):
        callback_count[0]+=1
        return {'statistics':np.zeros(12),'numerical_valid':True,'new_inverse_calls':0}
    zero=u.whole_inverse_linearization(zero_refit,estimate,cfg,seed=501003,
        summarize=lambda out:out['statistics'])
    test('zero_sensitivity_stable_without_division_by_zero_or_fake_scale',lambda:
        zero['step_size_check_passed'] and zero['covariance_rank']==0 and zero['scale'] is None
        and not zero['scale_provenance']['positive_scale_for_all_statistics']
        and np.isfinite(np.asarray(zero['step_comparisons'][0]['relative_difference_per_probe_statistic'])).all())

    def uncertified_refit(dn):
        value=linear_refit(dn);value['numerical_valid']=False;return value
    uncertified=u.whole_inverse_linearization(uncertified_refit,estimate,cfg,seed=501003,
        probes=32,max_full_inverse_calls=192,probe_convergence_verified=True,
        summarize=lambda out:out['statistics'])
    test('finite_stable_uncertified_refits_never_supply_trusted_scale',lambda:
        uncertified['step_size_check_passed'] and not uncertified['scale_provenance']['refits_numerically_certified']
        and uncertified['scale'] is None and uncertified['covariance'] is None)

    def failing_refit(dn):
        callback_count[0]+=1
        raise ArithmeticError('synthetic callback failure')
    failed=u.whole_inverse_linearization(failing_refit,estimate,cfg,seed=501003,
        summarize=lambda out:out['statistics'])
    test('all_failed_directional_callbacks_keep_all_twelve_attempts',lambda:{'passed':
        failed['actual_callback_calls']==failed['planned_full_inverse_calls']==len(failed['failed_refits'])==12
        and failed['covariance'] is None and failed['diagnostic_covariance'] is None
        and failed['scale'] is None and failed['new_inverse_calls'] is None,
        'attempted_callbacks':failed['actual_callback_calls'],'failed_callbacks':len(failed['failed_refits'])})
    test('directional_call_budget_rejected_before_refitting',lambda:rejected(lambda:linear(max_full_inverse_calls=11)))

    sample_values=np.array([np.arange(1.,13.)*v for v in (-2.,-1.,1.,2.)])
    scale=u.empirical_statistic_scale(sample_values,declared_replicates=4)
    test('bootstrap_scale_is_twelve_output_SDs_with_small_fixture_debug_status',lambda:
        scale['available'] and np.allclose(scale['scale'],np.std(sample_values,axis=0,ddof=1))
        and len(scale['scale'])==12 and scale['scale_provenance']['debug_only']
        and scale['scale_provenance']['deployment_scale'] and not scale['scale_provenance']['coverage_guaranteed'])
    oracle=u.empirical_statistic_scale(sample_values,kind='independent_full_inverse_mc_sd',declared_replicates=4)
    test('independent_scene_MC_scale_explicit_oracle_never_deployment_scale',lambda:
        oracle['available'] and oracle['scale_provenance']['oracle_diagnostic']
        and oracle['scale_provenance']['debug_only'] and not oracle['scale_provenance']['deployment_scale'])
    missing=u.empirical_statistic_scale(sample_values[:2],declared_replicates=4)
    nonfinite=u.empirical_statistic_scale([sample_values[0],None,[float('nan')]*12,sample_values[3]],declared_replicates=4)
    test('empirical_scale_missing_and_failed_draws_do_not_shrink_denominator',lambda:{'passed':
        not missing['available'] and missing['scale'] is None and missing['declared_replicates']==4
        and missing['missing_replicates']==2 and not nonfinite['available']
        and nonfinite['failed_replicates']==[1,2] and nonfinite['declared_replicates']==4,
        'missing_draws':missing['missing_replicates'],'failed_draws':nonfinite['failed_replicates']})

    def image_refit(dn):
        callback_count[0]+=1
        return {'temperature_color_K':2000.+np.asarray(dn),
                'template_postblur_e_ref':.1*np.asarray(dn),
                'predicted_raw_e':np.asarray(dn),'numerical_valid':True}
    image_estimate=image_refit(mean)
    boot=u.conditional_bootstrap(image_refit,image_estimate,cfg,seed=501004,replicates=8)
    test('conditional_bootstrap_refits_whole_output_and_keeps_twelve_statistics',lambda:
        boot['complete'] and len(boot['samples'])==8 and np.asarray(boot['samples']).shape==(8,12)
        and boot['new_refits_numerically_certified']==8 and not boot['truth_in_refit']
        and not boot['independent_coverage_test_performed'])
    boot_failed=u.conditional_bootstrap(failing_refit,image_estimate,cfg,seed=501004,replicates=4)
    resumed=u.conditional_bootstrap(image_refit,image_estimate,cfg,seed=501004,
        replicates=4,prior_samples=[[float('nan')]*12])
    test('bootstrap_failed_and_invalid_resumed_draws_remain_in_declared_denominator',lambda:
        len(boot_failed['samples'])==boot_failed['replicates']==len(boot_failed['failed_replicates'])==4
        and boot_failed['basic_CI95'] is None and boot_failed['completed_finite_replicates']==0
        and len(resumed['samples'])==4 and resumed['samples'][0] is None
        and not resumed['complete'] and resumed['basic_CI95'] is None)

    def record(mid,rep=0,role='interval_calibration',source=scale):
        return {'role':role,'mother_id':mid,'replicate':rep,
            'statistics':np.arange(1.,13.).tolist(),'truth_statistics':[0.]*12,
            'scale':copy.deepcopy(source['scale']),'scale_provenance':copy.deepcopy(source['scale_provenance']),
            'actual_temperature_geometry':'front','actual_template_geometry':'filament'}

    calibration=[record('fixture_cal',i) for i in range(4)]
    model=intervals.calibrate(calibration,declared_draws_by_mother={'fixture_cal':4})
    boundary=intervals.interval(calibration[0]['statistics'],calibration[0]['scale'],model,
        scale_provenance=calibration[0]['scale_provenance'])
    test('empirical_quantile_endpoints_include_exact_boundary_after_rounding',lambda:
        np.all(boundary[:,0]<=0.) and np.all(boundary[:,1]>=0.)
        and np.max(abs(boundary[:,0]))<1e-13)
    evaluation=[record('fixture_eval',i,role='interval_evaluation') for i in range(2)]
    # Keep this denominator check inside the interval. A quantile endpoint
    # exactly equal to truth is a separate floating-point boundary question.
    for row in evaluation: row['statistics']=[0.]*12
    model_hash=s.canonical(model)
    result=intervals.evaluate(evaluation,model,declared_draws_by_mother={'fixture_eval':4})
    test('held_out_intervals_keep_missing_draws_and_never_refit_model',lambda:{'passed':
        model['available'] and result['evaluation_noise_draws']==4 and result['finite_draws']==2
        and result['rows'][0]['missing_draws']==2
        and np.allclose(result['pooled_evaluation_marginal_coverage'],.5)
        and not result['evaluation_labels_used_to_fit'] and not result['model_recalibrated']
        and s.canonical(model)==model_hash,
        'coverage_full_declared_denominator':result['pooled_evaluation_marginal_coverage'],
        'declared_draws':result['evaluation_noise_draws'],'finite_draws':result['finite_draws']})
    test('declared_missing_calibration_draw_makes_model_unavailable',lambda:
        not intervals.calibrate(calibration[:2],declared_draws_by_mother={'fixture_cal':4})['available']
        and intervals.calibrate(calibration[:2],declared_draws_by_mother={'fixture_cal':4})['total_draws']==4)
    test('evaluation_labels_cannot_calibrate',lambda:rejected(lambda:intervals.calibrate(evaluation)))
    overlap=[record('fixture_cal',role='interval_evaluation')]
    test('calibration_mother_cannot_become_held_out_evaluation',lambda:
        rejected(lambda:intervals.evaluate(overlap,model)))
    scalar=copy.deepcopy(calibration);scalar[0]['scale']=12.
    test('scalar_raw_brightness_rejected_as_temperature_or_template_scale',lambda:
        rejected(lambda:intervals.calibrate(scalar))
        and rejected(lambda:intervals.interval([0.]*12,12.,model,scale_provenance=scale['scale_provenance'])))
    missing_geometry=copy.deepcopy(calibration)
    for row in missing_geometry:
        row.pop('actual_template_geometry');row['geometry']='filament'
    conflict=copy.deepcopy(calibration);conflict[1]['actual_template_geometry']='blob'
    test('actual_generated_geometries_required_and_consistent_per_mother',lambda:
        rejected(lambda:intervals.calibrate(missing_geometry)) and rejected(lambda:intervals.calibrate(conflict))
        and model['actual_geometries']['fixture_cal']['actual_template_geometry']=='filament')
    failed_records=[record('fixture_fail',i,role='interval_evaluation') for i in range(2)]
    for row in failed_records: row['statistics']=None;row['scale']=None
    all_failed=intervals.evaluate(failed_records,model,declared_draws_by_mother={'fixture_fail':4})
    failed_calibration=[record('fixture_failed_cal',i) for i in range(2)]
    for row in failed_calibration: row['statistics']=None;row['scale']=None
    failed_model=intervals.calibrate(failed_calibration,declared_draws_by_mother={'fixture_failed_cal':4})
    test('all_failed_evaluation_and_calibration_preserve_denominators',lambda:
        all_failed['finite_draws']==0 and all_failed['evaluation_noise_draws']==4
        and np.allclose(all_failed['pooled_evaluation_marginal_coverage'],0.)
        and all_failed['rows'][0]['mean_interval_width'] is None
        and not failed_model['available'] and failed_model['total_draws']==4
        and len(failed_model['failed_records'])==4 and failed_model['constants'] is None)
    oracle_model=intervals.calibrate([record('fixture_oracle',source=oracle)])
    test('oracle_MC_scale_kept_explicit_in_calibration_result',lambda:
        oracle_model['oracle_scale_diagnostic'] and oracle_model['debug_only']
        and not oracle_model['scale_available_at_raw_only_deployment'] and not oracle_model['deployment_qualified'])
    test('uncertified_local_diagnostic_cannot_be_reused_as_interval_scale',lambda:
        rejected(lambda:intervals.interval([0.]*12,local['diagnostic_scale'],model,
            scale_provenance=local['scale_provenance'])))
    test('replicate_id_outside_declared_range_rejected',lambda:
        rejected(lambda:intervals.evaluate([record('fixture_eval',9,role='interval_evaluation')],model,
            declared_draws_by_mother={'fixture_eval':4})))
    legacy={'schema':intervals.LEGACY_SCHEMA,'available':True,'constants':[2.]*12}
    test('legacy_scalar_intervals_require_explicit_legacy_entrypoint',lambda:
        rejected(lambda:intervals.interval([0.]*12,[1.]*12,legacy,scale_provenance=scale['scale_provenance']))
        and np.asarray(intervals.legacy_interval([0.]*12,3.,legacy)).shape==(12,2))

    output={'template_postblur_e_ref':torch.full((1,4,4),80.,dtype=torch.float64),
            'temperature_color_K':torch.full((1,4,4),2000.,dtype=torch.float64),
            'predicted_raw_e':torch.full((1,4,4),100.,dtype=torch.float64),
            'valid':torch.ones((1,4,4),dtype=torch.bool),'numerical_valid':False}
    detector=u.null_threshold([0.]*200,[f'fixture_null_{i}' for i in range(200)],
        method='fixture_method',calibration_domain='fixture_domain')
    absent=u.attach_frame_detection(output)['frame_detection_v2']
    bound=u.attach_frame_detection(output,detector,method='fixture_method',application_domain='fixture_domain')['frame_detection_v2']
    outside=u.attach_frame_detection(output,detector,method='fixture_method',application_domain='other_domain')['frame_detection_v2']
    test('no_frozen_threshold_has_no_detection_decision',lambda:
        absent['detected'] is None and absent['threshold'] is None and absent['threshold_crossed'] is None)
    test('method_specific_threshold_rejects_different_method',lambda:
        rejected(lambda:u.attach_frame_detection(output,detector,method='other_method',application_domain='fixture_domain')))
    test('outside_calibration_domain_is_diagnostic_without_formal_detection',lambda:
        bound['detected'] is True and outside['detected'] is None and outside['threshold_crossed'] is True
        and outside['calibration_applicability']=='OUTSIDE_DECLARED_DOMAIN'
        and 'DIAGNOSTIC' in outside['status'] and not outside['false_positive_rate_guaranteed'])
    equality=copy.deepcopy(detector);equality['threshold']=bound['score']
    test('dictionary_detection_uses_strict_greater_comparison',lambda:
        u.attach_frame_detection(output,equality,method='fixture_method',application_domain='fixture_domain')['frame_detection_v2']['detected'] is False)
    old_dictionary={key:value for key,value in detector.items() if key not in ('schema','dictionary_id','method','calibration_domain')}
    unbound=u.attach_frame_detection(output,old_dictionary)['frame_detection_v2']
    test('old_dictionary_threshold_explicit_unbound_diagnostic_and_60e_rule_rejected',lambda:
        unbound['detected'] is None and unbound['threshold_crossed'] is True
        and unbound['threshold_schema']=='legacy-closeout-dictionary-threshold'
        and rejected(lambda:u.attach_frame_detection(output,{'threshold':60.,'alpha_target':.05})))
    test('null_calibration_must_use_distinct_mothers',lambda:
        rejected(lambda:u.null_threshold([0.]*200,['same_mother']*200)))

    physics=SimpleNamespace(cfg={**cfg,'temperature_range_K':[1000.,3000.]},
        M=torch.zeros(1,dtype=torch.float64),extra_variance=4.+1./12,
        raw_to_electrons=lambda value:value)
    truth={'evaluation_domain':torch.ones((1,4,4),dtype=torch.bool),
           'template_postblur_e':torch.full((1,4,4),100.,dtype=torch.float64),
           'temperature_color_K':torch.full((1,4,4),2000.,dtype=torch.float64)}
    raw=np.full((4,4),100.)
    high=copy.deepcopy(detector);high['threshold']=1e9
    scored=metrics.score(output,truth,raw,physics,detector=high,
        method='fixture_method',application_domain='fixture_domain')
    test('legacy_60e_detection_and_v2_dictionary_decision_remain_distinct',lambda:
        scored['metrics']['legacy_frame_detected'] is True
        and scored['metrics']['legacy_threshold_e']==60.
        and scored['metrics']['frame_detection_v2']['detected'] is False
        and not any(key in scored['metrics'] for key in ('frame_detected','threshold_e','signed_recall')))
    test('finite_output_finite_scoring_and_production_certification_are_separate',lambda:
        scored['finite_output'] and scored['finite_scoring_passed']
        and not scored['production_numerical_certified'] and scored['status']=='FINITE_NUMERICALLY_UNCERTIFIED'
        and 'numerical_execution_passed' not in scored)
    unscorable=metrics.score(output,{},raw,physics)
    out_of_bounds=copy.deepcopy(output);out_of_bounds['temperature_color_K'].fill_(4000.)
    bounds=metrics.score(out_of_bounds,truth,raw,physics)
    test('finite_output_does_not_require_valid_truth_domain_or_temperature_bounds',lambda:
        unscorable['finite_output'] and not unscorable['finite_scoring_passed'] and unscorable['metrics'] is None
        and bounds['finite_output'] and not bounds['finite_scoring_passed']
        and bounds['status']=='TEMPERATURE_OUT_OF_BOUNDS')
    certified=copy.deepcopy(output);certified['numerical_valid']=True
    certified_score=metrics.score(certified,truth,raw,physics)
    nonfinite_output=copy.deepcopy(output);nonfinite_output['temperature_color_K'][0,0,0]=float('nan')
    nonfinite_score=metrics.score(nonfinite_output,truth,raw,physics)
    test('certified_and_nonfinite_score_statuses_stay_explicit',lambda:
        certified_score['production_numerical_certified'] and certified_score['status']=='FINITE_NUMERICALLY_CERTIFIED'
        and not nonfinite_score['finite_output'] and not nonfinite_score['finite_scoring_passed']
        and nonfinite_score['status']=='NONFINITE_PREDICTION')
    aggregate=metrics.aggregate([{'metrics':None},{'metrics':None}])
    test('all_failed_score_aggregation_keeps_case_denominator',lambda:
        aggregate['planned_or_attempted_cases']==aggregate['failed_or_unavailable_cases']==2
        and aggregate['finite_cases']==0 and aggregate['temperature_RMSE_K'] is None
        and aggregate['template_RMSE_e'] is None and not aggregate['all_cases_finite'])

    result={'time':s.now(),'schema':'statistics-repair-contracts-v2','all_passed':all(t['passed'] for t in checks),
        'tests':checks,'fixture_shape':[4,4],'device':'cpu','CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
        'synthetic_fixture_callback_calls':callback_count[0],
        'formal_inverse_calls':0,'formal_complete_inversions':0,'formal_raw_generated':False,
        'formal_MC_rerun':False,'GPU_used':False,'sealed_payloads_opened':False,
        'scientific_coverage_or_accuracy_claimed':False,
        'source_sha256':{path.relative_to(s.ROOT).as_posix():s.sha(path) for path in
            (s.ROOT/'research_closeout/uncertainty.py',s.ROOT/'research_closeout/intervals.py',
             s.ROOT/'research_closeout/metrics.py',s.ROOT/'research_closeout/tests/statistics_repair.py')}}
    path=s.REPORTS/'repair_001/STATISTICS_REPAIR_CONTRACTS.json'
    s.write(path,s.plain(result),replace=path.exists())
    print(json.dumps(s.plain(result),ensure_ascii=True),flush=True)
    return result


if __name__=='__main__':
    raise SystemExit(0 if run()['all_passed'] else 2)
