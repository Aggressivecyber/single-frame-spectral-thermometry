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
    samples=list(prior_samples) if prior_samples is not None else [];failures=[];numeric_status=[]
    for i in range(len(samples),replicates):
        raw,_=sample_raw(mu,cfg,seed+i*1009)
        try:
            out=refit(raw);value=summary_statistics(out)
            numeric_status.append(bool(out.get('numerical_valid',False)))
            if not np.isfinite(value).all(): raise ValueError('Nonfinite bootstrap statistics')
            samples.append(value.tolist())
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

def whole_inverse_linearization(refit,estimate,cfg,*,seed,probes=8,epsilon=.001):
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
            return (summary_statistics(refit(plus))-summary_statistics(refit(minus)))/(2*step)
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

def null_threshold(scores,mother_ids,alpha=.05):
    a=np.asarray(scores,dtype=float)
    if len(a)!=len(mother_ids) or len(set(mother_ids))!=len(a): raise ValueError('Null calibration units must be independent mothers')
    if not np.isfinite(a).all(): raise ValueError('Failed null calibration remains unresolved')
    order=int(np.ceil((len(a)+1)*(1-alpha)))
    if order>len(a): raise ValueError('Insufficient null calibration units for target alpha')
    return {'threshold':float(np.sort(a)[order-1]),'order_statistic':order,'mothers':len(a),
        'alpha_target':alpha,'calibration_false_positive_fraction':float((a>np.sort(a)[order-1]).mean()),
        'task':'max absolute27-dictionary matched-filter score per frame','rule':'strictly greater than threshold',
        'multiple_comparisons':'all positions/scales/signs calibrated together in null score'}

def binomial_interval(successes,units,confidence=.95):
    if not 0<=successes<=units or units<1: raise ValueError('Invalid binomial denominator')
    alpha=(1-confidence)/2
    lo=0. if successes==0 else float(beta_distribution.ppf(alpha,successes,units-successes+1))
    hi=1. if successes==units else float(beta_distribution.ppf(1-alpha,successes+1,units-successes))
    return {'estimate':successes/units,'CI95':[lo,hi],'successes':successes,'independent_units':units,
        'method':'Clopper-Pearson','unit':'mother scenario for frame detection; noise replicates are not new mothers'}
