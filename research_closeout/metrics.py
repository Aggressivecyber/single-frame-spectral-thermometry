"""External fixed-domain scoring, additional optical targets and structure metrics."""
from . import storage as s
from . import measurement as m
from .uncertainty import attach_frame_detection
import numpy as np
import torch

def canonical_for_score(out,raw,p):
    return {**out,'template_postblur_e':out['template_postblur_e_ref'],
        'expected_e':out['predicted_raw_e'],'raw_e':p.raw_to_electrons(m.raw_tensor(raw,p))}

LEGACY_DETECTION_FIELDS={
    'threshold_e','z_threshold','sigma_valid_fraction','signed_detected_pixels',
    'signed_recall','signed_recall_unconditional','signed_recall_rule',
    'positive_detected_pixels','positive_recall','negative_detected_pixels','negative_recall',
    'frame_detection_score','frame_detection_score_kind','frame_threshold','frame_detected',
    'null_pixel_FP_count','null_pixel_FP','null_frame_FP','null_FOV_FPR_control_claimed'}

def score(out,truth,raw,p,*,detector=None,method=None,application_domain=None):
    from research_v5.metrics import score_prediction
    finite_output=True
    for name in ('template_postblur_e_ref','temperature_color_K','predicted_raw_e'):
        value=out[name]
        if hasattr(value,'detach'): value=value.detach().cpu().numpy()
        finite_output=finite_output and bool(np.isfinite(value).all())
    result=score_prediction(canonical_for_score(out,raw,p),truth,p)
    if result['metrics'] is not None:
        for key in LEGACY_DETECTION_FIELDS:
            if key in result['metrics']: result['metrics']['legacy_'+key]=result['metrics'].pop(key)
        detected=attach_frame_detection(out,detector,method=method,application_domain=application_domain)
        result['metrics']['frame_detection_v2']=detected['frame_detection_v2']
        result['metrics']['matched_dictionary_frame_score']=detected['frame_detection_v2']['score']
        result['metrics']['predicted_measurement_nonnegative']=bool(out['predicted_raw_e'].min()>=0)
        dn=m.raw_tensor(raw,p);top=2**p.cfg['adc_bits']-1
        result['metrics']['raw_saturated_pixels']=int((dn>=top).sum())
        result['metrics']['raw_lower_clipped_pixels']=int((dn<=0).sum())
        u=out['template_postblur_e_ref'].double();target=truth['template_postblur_e'].double()
        result['metrics']['structure']=structure(u,target)
    result['production_numerical_certified']=bool(out.get('numerical_valid',False))
    result['legacy_metric_status']=result['status']
    result['finite_scoring_passed']=bool(result.pop('numerical_execution_passed',False))
    result['finite_scoring_definition']='all finite, within temperature bounds and not abstained; no solver convergence assertion'
    result['finite_output']=finite_output
    result['finite_output_definition']='all template, color-temperature and predicted-raw values finite; independent of scoring truth, bounds and abstention'
    if result['status']=='SUCCESS':
        result['status']='FINITE_NUMERICALLY_CERTIFIED' if result['production_numerical_certified'] else 'FINITE_NUMERICALLY_UNCERTIFIED'
    result['metric_schema']='closeout-fixed-domain-explicit-status-v2'
    return s.plain(result)

def structure(prediction,target):
    def descriptor(value):
        if hasattr(value,'detach'): value=value.detach().cpu().numpy()
        a=np.asarray(value).squeeze();mass=np.abs(a);total=float(mass.sum());h,w=a.shape
        y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
        if total:
            cx=float((mass*x).sum()/total);cy=float((mass*y).sum()/total)
            width=float(np.sqrt((mass*((x-cx)**2+(y-cy)**2)).sum()/total))
        else: cx=cy=width=None
        return {'absolute_peak_e':float(mass.max()),'signed_integral_e_px':float(a.sum()),
            'absolute_integral_e_px':total,'absolute_mass_RMS_radius_px':width,
            'absolute_mass_centroid_xy_px':[cx,cy]}
    return {'prediction':descriptor(prediction),'target':descriptor(target),
        'width_definition':'absolute-mass radial RMS; noise outside structure contributes and is retained'}

@torch.no_grad()
def resolution_scores(out,truth,raw,p):
    results=[]
    target_C=truth.get('continuum_postblur_bands')
    if target_C is None:
        target_C=p.op.object_blur(truth['continuum_latent'])
    for sigma in (0.,2.,4.):
        current=dict(out);label=dict(truth)
        C=m.extra_smooth(out['continuum_postblur_bands'],sigma)
        Ct=m.extra_smooth(target_C,sigma)
        current['temperature_color_K']=p.color_temperature(C)[0]
        current['template_postblur_e_ref']=m.extra_smooth(out['template_postblur_e_ref'],sigma)
        label['temperature_color_K']=p.color_temperature(Ct)[0]
        label['template_postblur_e']=m.extra_smooth(truth['template_postblur_e'],sigma)
        value=score(current,label,raw,p)
        value.update(extra_Gaussian_sigma_px=sigma,color_target='ratio after smoothing continuum bands')
        results.append(value)
    return results

def aggregate(rows):
    metrics=[r['metrics'] for r in rows if r.get('metrics') is not None]
    finite=[r for r in metrics if r['template_SSE'] is not None and r['temperature_SSE_K2'] is not None]
    pixels=sum(r['domain_pixels'] for r in finite)
    return {'planned_or_attempted_cases':len(rows),'finite_cases':len(finite),
        'failed_or_unavailable_cases':len(rows)-len(finite),'scoring_pixels':pixels,
        'template_RMSE_e':float(np.sqrt(sum(r['template_SSE'] for r in finite)/pixels)) if pixels else None,
        'temperature_RMSE_K':float(np.sqrt(sum(r['temperature_SSE_K2'] for r in finite)/pixels)) if pixels else None,
        'all_cases_finite':len(finite)==len(rows),
        'conditional_aggregate_requires_explicit_failure_denominator':True}
