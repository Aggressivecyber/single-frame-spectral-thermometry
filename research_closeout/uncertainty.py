"""Full-inverse conditional bootstrap, shared-raw linearization and frame detection."""
from . import storage as s
from .independent_renderer import sample_raw
import numpy as np
from scipy.stats import beta as beta_distribution

PIXEL_FRACTIONS=((.25,.25),(.5,.25),(.5,.5),(.25,.75),(.75,.75))

def summary_statistics(output):
    def array(name):
        v=output[name]
        if hasattr(v,'detach'): v=v.detach().cpu().numpy()
        return np.asarray(v).reshape(np.asarray(v).shape[-2:])
    u=array('template_postblur_e_ref');T=array('temperature_color_K');h,w=u.shape
    points=[(min(h-1,int(y*(h-1))),min(w-1,int(x*(w-1)))) for x,y in PIXEL_FRACTIONS]
    roi=(slice(h//4,3*h//4),slice(w//4,3*w//4))
    return np.array([T[y,x] for y,x in points]+[T[roi].mean()]+[u[y,x] for y,x in points]+[u[roi].mean()])

STATISTIC_NAMES=[f'color_pixel_{i}_K' for i in range(5)]+['color_central_ROI_mean_K']+[f'template_pixel_{i}_e_ref' for i in range(5)]+['template_central_ROI_mean_e_ref']

def conditional_bootstrap(refit,estimate,cfg,*,seed,replicates=64,prior_samples=None):
    """refit(raw_DN) must run the entire frozen inverse without any truth input.

    Fit-predicted Poisson/read/ADC draws provide conditional model intervals.
    External truth is required to evaluate coverage, outside this function.
    Failed replicates remain in the denominator and prevent certification.
    """
    mu=estimate['predicted_raw_e']
    if hasattr(mu,'detach'): mu=mu.detach().cpu().numpy()
    mu=np.asarray(mu).reshape(cfg['shape']);base=summary_statistics(estimate)
    if not isinstance(replicates,int) or replicates<2: raise ValueError('At least two declared bootstrap replicates are required')
    samples=list(prior_samples) if prior_samples is not None else []
    if len(samples)>replicates: raise ValueError('Resumed bootstrap prefix exceeds the declared replicate count')
    failures=[];numeric_status=[]
    for i,value in enumerate(samples):
        try: a=np.asarray(value,dtype=float) if value is not None else None
        except (TypeError,ValueError): a=None
        if a is None or a.shape!=(12,) or not np.isfinite(a).all():
            samples[i]=None
            failures.append({'replicate':i,'error_type':'InvalidResumedStatistic','error':'Missing, nonfinite or wrong-sized resumed bootstrap statistic'})
    for i in range(len(samples),replicates):
        raw,_=sample_raw(mu,cfg,seed+i*1009)
        try:
            out=refit(raw);value=summary_statistics(out)
            if not np.isfinite(value).all(): raise ValueError('Nonfinite bootstrap statistics')
            samples.append(value.tolist());numeric_status.append(bool(out.get('numerical_valid',False)))
        except Exception as error:
            samples.append(None);numeric_status.append(False);failures.append({'replicate':i,'error_type':type(error).__name__,'error':str(error)})
    complete=len(samples)==replicates and all(x is not None for x in samples)
    result={'replicates':replicates,'completed_finite_replicates':sum(x is not None for x in samples),
        'complete':complete,'failed_replicates':failures,'samples':samples,'statistics':STATISTIC_NAMES,
        'interval_object':'marginal pointwise or stated ROI mean; not simultaneous image coverage',
        'model_condition':'fitted mean with declared Poisson/read/ADC model; external calibration assumed',
        'shared_raw_correlation':'preserved by refitting continuum/template/background together',
        'truth_in_refit':False,'base_statistics':base.tolist(),
        'new_refits_numerically_certified':sum(numeric_status),'resumed_prefix_numeric_status':'must retain prior receipt',
        'independently_calibrated':False,'independent_coverage_test_performed':False,
        'complete_means':'all declared draws returned finite statistics; not statistical or production certification'}
    if complete:
        a=np.asarray(samples);q=np.quantile(a,[.025,.975],axis=0)
        intervals=np.stack([2*base-q[1],2*base-q[0]],axis=1)
        result.update(basic_CI95=intervals.tolist(),width=(intervals[:,1]-intervals[:,0]).tolist(),
            bootstrap_bias=(a.mean(0)-base).tolist(),covariance=np.cov(a,rowvar=False).tolist())
    else: result.update(basic_CI95=None,width=None,bootstrap_bias=None,covariance=None)
    return result

def legacy_whole_inverse_linearization(refit,estimate,cfg,*,seed,probes=8,epsilon=.001,summarize=summary_statistics):
    """Matrix-free directional derivatives of the whole programmed estimator.

    This coarse directional covariance is a diagnostic, not a95% interval.
    It includes shared raw effects and finite iteration behavior. A second
    step size exposes local nonlinearity/numerical differentiation errors.
    """
    mu=estimate['predicted_raw_e']
    if hasattr(mu,'detach'): mu=mu.detach().cpu().numpy()
    mu=np.asarray(mu).reshape(cfg['shape']);rng=np.random.default_rng(seed)
    adc=cfg['full_well_e']/(2**cfg['adc_bits']-1-cfg['black_level_DN'])
    sigma=np.sqrt(np.maximum(mu,0)+cfg['read_noise_e']**2+adc**2/12)
    derivatives=[];errors=[]
    for _ in range(probes):
        direction=rng.choice([-1.,1.],mu.shape)*sigma
        def derivative(step):
            plus=(mu+step*direction)/adc+cfg['black_level_DN']
            minus=(mu-step*direction)/adc+cfg['black_level_DN']
            return (summarize(refit(plus))-summarize(refit(minus)))/(2*step)
        first=derivative(epsilon);second=derivative(epsilon/2)
        derivatives.append(second);errors.append(float(np.linalg.norm(first-second)/max(np.linalg.norm(second),1e-12)))
    a=np.asarray(derivatives)
    return {'probes':probes,'epsilon_electron_innovation_scale':epsilon,'covariance':(a.T@a/probes).tolist(),
        'step_size_relative_discrepancies':errors,'statistics':STATISTIC_NAMES,'dense_Hessian_built':False,
        'full_inverse_calls':4*probes,'inference_reads_truth':False,
        'scope':'coarse local whole-program derivative diagnostic, not calibrated tail coverage',
        'shared_raw_correlation':'same perturbation drives every inferred component',
        'ADC_linearization':'variance surrogate only; conditional bootstrap retains actual ADC rounding'}

def matched_dictionary_score(template):
    if hasattr(template,'detach'): template=template.detach().cpu().numpy()
    u=np.asarray(template).squeeze();h,w=u.shape
    y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');scores=[]
    for fy in (.25,.5,.75):
        for fx in (.25,.5,.75):
            for sigma in (8.,16.,32.):
                weight=np.exp(-((x-fx*w)**2+(y-fy*h)**2)/(2*sigma*sigma))
                scores.append(float((u*weight).sum()/np.sqrt((weight*weight).sum())))
    return max(abs(v) for v in scores)

DICTIONARY_ID='gaussian-9centers-3scales-absolute-L2-v1'

def null_threshold(scores,mother_ids,alpha=.05,*,method=None,calibration_domain=None):
    a=np.asarray(scores,dtype=float)
    if a.ndim!=1 or not 0<alpha<1: raise ValueError('One-dimensional scores and 0 < alpha < 1 required')
    if len(a)!=len(mother_ids) or len(set(mother_ids))!=len(a): raise ValueError('Null calibration units must be independent mothers')
    if not np.isfinite(a).all(): raise ValueError('Failed null calibration remains unresolved')
    order=int(np.ceil((len(a)+1)*(1-alpha)))
    if order>len(a): raise ValueError('Insufficient null calibration units for target alpha')
    return {'schema':'frame-dictionary-threshold-v2','dictionary_id':DICTIONARY_ID,
        'method':method,'calibration_domain':calibration_domain,'calibration_mothers':list(mother_ids),
        'threshold':float(np.sort(a)[order-1]),'order_statistic':order,'mothers':len(a),
        'alpha_target':alpha,'calibration_false_positive_fraction':float((a>np.sort(a)[order-1]).mean()),
        'task':'max absolute27-dictionary matched-filter score per frame','rule':'strictly greater than threshold',
        'multiple_comparisons':'all positions/scales/signs calibrated together in null score'}

def attach_frame_detection(output,detector=None,*,method=None,application_domain=None):
    """Attach a raw-only, explicitly named frame decision; never reads labels.

    A frozen max-dictionary threshold is required. Legacy pixel/percentile
    decisions are neither read nor relabeled. A decision under an unmatched
    calibration domain remains a diagnostic threshold crossing.
    """
    result=dict(output)
    score=matched_dictionary_score(output['template_postblur_e_ref'])
    finite=bool(np.isfinite(score))
    decision={'schema':'frame-dictionary-decision-v2','dictionary_id':DICTIONARY_ID,
        'method':method,'score':float(score) if finite else None,'detected':None,'threshold_crossed':None,'threshold':None,
        'status':'THRESHOLD_NOT_ATTACHED' if finite else 'NONFINITE_DICTIONARY_SCORE','inference_reads_truth':False,
        'calibration_applicability':'UNVERIFIED','false_positive_rate_guaranteed':False}
    if detector is not None:
        known=detector.get('dictionary_id')==DICTIONARY_ID
        # The old closeout dictionary threshold is identifiable by its exact
        # task string. This does not accept the historical 60 e rule.
        legacy=detector.get('schema') is None and detector.get('task')=='max absolute27-dictionary matched-filter score per frame'
        if not (known or legacy): raise ValueError('An identified max-dictionary threshold is required')
        if detector.get('rule')!='strictly greater than threshold': raise ValueError('Unsupported detector comparison rule')
        if detector.get('available',True) is False: raise ValueError('Detector calibration is unavailable')
        threshold=float(detector['threshold']);alpha=float(detector['alpha_target'])
        if not np.isfinite(threshold) or threshold<0 or not 0<alpha<1: raise ValueError('Invalid detector threshold or target FPR')
        if detector.get('method') is not None and detector['method']!=method:
            raise ValueError('Method-specific threshold cannot be applied to another method')
        domain=detector.get('calibration_domain')
        applicability='UNVERIFIED' if domain is None or application_domain is None else ('MATCHED_DECLARED_DOMAIN' if domain==application_domain else 'OUTSIDE_DECLARED_DOMAIN')
        method_bound=detector.get('method') is not None and detector['method']==method
        qualified=method_bound and applicability=='MATCHED_DECLARED_DOMAIN'
        crossing=bool(score>threshold) if finite else None
        status=('CALIBRATED_DICTIONARY_THRESHOLD_CROSSING' if qualified else
            ('OUTSIDE_DECLARED_DOMAIN_DIAGNOSTIC_THRESHOLD_CROSSING' if applicability=='OUTSIDE_DECLARED_DOMAIN' else
             'UNVERIFIED_CALIBRATION_DIAGNOSTIC_THRESHOLD_CROSSING'))
        decision.update(threshold=threshold,detected=crossing if qualified else None,threshold_crossed=crossing,
            status=status if finite else 'NONFINITE_DICTIONARY_SCORE',comparison='strict_gt',
            alpha_target=alpha,calibration_mothers=detector.get('mothers'),
            threshold_schema=detector.get('schema','legacy-closeout-dictionary-threshold'),
            calibration_domain=domain,application_domain=application_domain,
            calibration_applicability=applicability,
            method_specific_calibration_bound=method_bound,
            FPR_scope='marginal calibrated null distribution; domain exchangeability required')
    result['frame_detection_v2']=decision
    return result

def empirical_statistic_scale(samples,*,kind='full_inverse_bootstrap_sd',
        declared_replicates=None,minimum_replicates=64):
    """Twelve output SDs from complete whole-inverse refits, without labels.

    Missing/nonfinite draws are retained and make the scale unavailable.
    Small fixtures can supply a debug scale but never a coverage guarantee.
    """
    allowed={'full_inverse_bootstrap_sd','independent_full_inverse_mc_sd'}
    if kind not in allowed: raise ValueError('Scale must describe whole-inverse bootstrap or MC output statistics')
    values=list(samples);declared=len(values) if declared_replicates is None else int(declared_replicates)
    if declared<2 or minimum_replicates<2: raise ValueError('At least two declared repeats are required')
    failures=[];finite=[]
    for i,value in enumerate(values):
        try: a=np.asarray(value,dtype=float) if value is not None else None
        except (TypeError,ValueError): a=None
        if a is None or a.shape!=(12,) or not np.isfinite(a).all(): failures.append(i)
        else: finite.append(a)
    complete=len(values)==declared and not failures
    scale=np.std(np.asarray(finite),axis=0,ddof=1) if complete else None
    zero=[] if scale is None else np.flatnonzero(scale<=0).tolist()
    available=bool(complete and scale is not None and not zero and np.isfinite(scale).all())
    provenance={'kind':kind,'full_inverse':True,'inference_reads_truth':False,
        'statistics':list(STATISTIC_NAMES),'replicates':len(values),'declared_replicates':declared,
        'complete':complete,'minimum_replicates':int(minimum_replicates),
        'debug_only':len(values)<minimum_replicates or kind=='independent_full_inverse_mc_sd',
        'deployment_scale':kind=='full_inverse_bootstrap_sd',
        'oracle_diagnostic':kind=='independent_full_inverse_mc_sd',
        'generation_mean':'fitted raw-only prediction' if kind=='full_inverse_bootstrap_sd' else 'independently specified scene mean; diagnostic only',
        'coverage_guaranteed':False}
    return {'available':available,'scale':scale.tolist() if available else None,
        'scale_provenance':provenance,'failed_replicates':failures,
        'zero_variance_statistics':zero,'finite_replicates':len(finite),
        'declared_replicates':declared,'missing_replicates':max(0,declared-len(values)),
        'status':'OUTPUT_SD_AVAILABLE' if available else 'OUTPUT_SD_UNAVAILABLE'}

def whole_inverse_linearization(refit,estimate,cfg,*,seed,probes=2,
        epsilon_grid=(.1,.03,.01),summarize=summary_statistics,
        relative_tolerance=.2,absolute_output_budgets=None,
        max_full_inverse_calls=24,minimum_probes_for_interval_scale=32,
        probe_convergence_verified=False):
    """Bounded, same-direction step-grid check of the programmed inverse.

    Differences are checked separately for each of the 12 output statistics.
    Both relative self-consistency and an absolute output-unit budget must
    pass. Defaults are a debugging fixture (12 calls), not a CI calculation.
    The covariance/scale fields are unavailable without a stable common
    platform, certified refits and an independently verified probe budget.
    Every diagnostic derivative, failure and callback count is retained.
    """
    eps=np.asarray(epsilon_grid,dtype=float)
    if eps.ndim!=1 or len(eps)<2 or not np.isfinite(eps).all() or np.any(eps<=0) or not np.all(np.diff(eps)<0):
        raise ValueError('epsilon_grid must be finite, positive and strictly decreasing')
    if not isinstance(probes,int) or probes<1: raise ValueError('A positive integer probe count is required')
    planned=2*probes*len(eps)
    if planned>max_full_inverse_calls: raise ValueError('Directional refits exceed the explicit call budget')
    if not 0<relative_tolerance<1 or minimum_probes_for_interval_scale<1:
        raise ValueError('Invalid derivative tolerance or probe requirement')
    budgets=np.asarray([.2]*6+[.17]*6 if absolute_output_budgets is None else absolute_output_budgets,dtype=float)
    if budgets.shape!=(12,) or not np.isfinite(budgets).all() or np.any(budgets<=0):
        raise ValueError('Twelve positive output-unit derivative-error budgets are required')
    mu=estimate['predicted_raw_e']
    if hasattr(mu,'detach'): mu=mu.detach().cpu().numpy()
    mu=np.asarray(mu,dtype=float).reshape(cfg['shape'])
    if not np.isfinite(mu).all() or np.any(mu<0): raise ValueError('Finite nonnegative fitted mean required')
    adc=cfg['full_well_e']/(2**cfg['adc_bits']-1-cfg['black_level_DN'])
    if not np.isfinite(adc) or adc<=0: raise ValueError('Invalid ADC scale')
    sigma=np.sqrt(mu+cfg['read_noise_e']**2+adc**2/12)
    rng=np.random.default_rng(seed);values=np.full((probes,len(eps),12),np.nan)
    directions=[];failures=[];calls=0;new_calls=0;new_calls_known=True;certified=[]
    for probe in range(probes):
        direction=rng.choice([-1.,1.],mu.shape)*sigma;steps=[]
        for j,epsilon in enumerate(eps):
            endpoints=[];endpoint_receipts=[]
            for sign in (1.,-1.):
                calls+=1
                try:
                    output=refit((mu+sign*epsilon*direction)/adc+cfg['black_level_DN'])
                    if not isinstance(output,dict): raise ValueError('Refit callback must return a mapping')
                    if 'new_inverse_calls' in output:
                        n=output['new_inverse_calls']
                        if not isinstance(n,int) or n<0: raise ValueError('Invalid refit inverse-call receipt')
                        new_calls+=n
                    else: new_calls_known=False
                    numeric=bool(output.get('numerical_valid',output.get('production_numerical_certified',False)))
                    statistic=np.asarray(summarize(output),dtype=float)
                    if statistic.shape!=(12,) or not np.isfinite(statistic).all(): raise ValueError('Nonfinite or wrong-sized directional statistics')
                    certified.append(numeric)
                    endpoints.append(statistic);endpoint_receipts.append({'sign':int(sign),'statistics':statistic.tolist(),'numerical_certified':numeric})
                except Exception as error:
                    new_calls_known=False;certified.append(False);endpoints.append(None)
                    failure={'probe':probe,'epsilon':float(epsilon),'sign':int(sign),'callback':calls,
                        'error_type':type(error).__name__,'error':str(error)}
                    failures.append(failure);endpoint_receipts.append(failure)
            if all(v is not None for v in endpoints): values[probe,j]=(endpoints[0]-endpoints[1])/(2*epsilon)
            steps.append({'epsilon_sigma_units':float(epsilon),'endpoints':endpoint_receipts,
                'derivative':values[probe,j].tolist() if np.isfinite(values[probe,j]).all() else None})
        directions.append({'probe':probe,'steps':steps})
    comparisons=[];stable_pairs=[]
    for j in range(len(eps)-1):
        first=values[:,j];second=values[:,j+1];finite=np.isfinite(first)&np.isfinite(second)
        absolute=np.abs(first-second)
        relative=absolute/np.maximum(np.maximum(np.abs(first),np.abs(second)),budgets[None])
        passed=finite&(relative<=relative_tolerance)&(absolute<=budgets[None])
        complete=bool(passed.all())
        if complete: stable_pairs.append(j)
        comparisons.append({'epsilon_pair':eps[j:j+2].tolist(),
            'absolute_difference_per_probe_statistic':s.plain(absolute),
            'relative_difference_per_probe_statistic':s.plain(relative),
            'passed_per_probe_statistic':passed.tolist(),
            'all_probes_passed_per_statistic':passed.all(axis=0).tolist(),
            'stable_common_platform':complete})
    # Prefer the smallest stable step pair; otherwise retain the smallest-step
    # diagnostic only. Failed directions are never omitted from covariance.
    chosen=stable_pairs[-1]+1 if stable_pairs else len(eps)-1
    a=values[:,chosen];complete=bool(np.isfinite(a).all() and not failures)
    diagnostic=a.T@a/probes if complete else None
    stable=bool(stable_pairs and complete)
    all_certified=len(certified)==planned and all(certified)
    enough=probes>=minimum_probes_for_interval_scale
    positive_scale=bool(diagnostic is not None and np.all(np.diag(diagnostic)>0))
    eligible=bool(stable and all_certified and enough and probe_convergence_verified and positive_scale)
    rank=None if diagnostic is None else int(np.linalg.matrix_rank(diagnostic))
    provenance={'kind':'whole_inverse_local_sd','full_inverse':True,'inference_reads_truth':False,
        'statistics':list(STATISTIC_NAMES),'complete':complete,'probes':probes,
        'step_size_check_passed':stable,'refits_numerically_certified':all_certified,
        'minimum_probes_for_interval_scale':minimum_probes_for_interval_scale,
        'probe_convergence_verified':bool(probe_convergence_verified),
        'interval_scale_eligible':eligible,'positive_scale_for_all_statistics':positive_scale,
        'debug_only':not eligible,'coverage_guaranteed':False}
    return {'schema':'whole-inverse-linearization-v2','probes':probes,
        'epsilon_grid_sigma_units':eps.tolist(),'selected_epsilon':float(eps[chosen]),
        'relative_tolerance':relative_tolerance,'absolute_output_budgets':budgets.tolist(),
        'budget_meaning':'difference between derivatives per unit input-SD perturbation, in each output statistic unit',
        'step_comparisons':comparisons,'directions':directions,'failed_refits':failures,
        'step_size_check_passed':stable,'status':'INTERVAL_SCALE_ELIGIBLE' if eligible else ('DEBUG_STABLE_PLATFORM' if stable else 'NUMERICAL_LINEARIZATION_UNCERTIFIED'),
        'covariance':diagnostic.tolist() if eligible else None,
        'scale':np.sqrt(np.diag(diagnostic)).tolist() if eligible else None,
        'diagnostic_covariance':None if diagnostic is None else diagnostic.tolist(),
        'diagnostic_scale':None if diagnostic is None else np.sqrt(np.diag(diagnostic)).tolist(),
        'scale_provenance':provenance,'interval_scale_eligible':eligible,
        'covariance_rank':rank,'rank_bound':min(probes,12),
        'rank_warning':'random directional second moment; rank is at most the number of probes; convergence is separately required',
        'planned_full_inverse_calls':planned,'actual_callback_calls':calls,
        'new_inverse_calls':new_calls if new_calls_known else None,
        'inverse_call_count_scope':'exact new count only when every callback provides its new_inverse_calls receipt',
        'statistics':list(STATISTIC_NAMES),'inference_reads_truth':False,
        'shared_raw_correlation':'same perturbation drives every inferred component',
        'ADC_linearization':'continuous DN derivative with noise-variance surrogate; no ADC rounding in directional calls',
        'coverage_guaranteed':False,'simultaneous_full_image_claimed':False}

def binomial_interval(successes,units,confidence=.95):
    if not 0<=successes<=units or units<1: raise ValueError('Invalid binomial denominator')
    alpha=(1-confidence)/2
    lo=0. if successes==0 else float(beta_distribution.ppf(alpha,successes,units-successes+1))
    hi=1. if successes==units else float(beta_distribution.ppf(1-alpha,successes+1,units-successes))
    return {'estimate':successes/units,'CI95':[lo,hi],'successes':successes,'independent_units':units,
        'method':'Clopper-Pearson','unit':'mother scenario for frame detection; noise replicates are not new mothers'}
