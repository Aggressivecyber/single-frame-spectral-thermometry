"""Output-statistic scale calibration, with an explicit historical scalar path."""
from . import uncertainty as u
import numpy as np

def feature_from_prediction(out,cfg):
    """Historical raw-electron feature; forbidden as the new 12-vector scale."""
    v=out['predicted_raw_e']
    if hasattr(v,'detach'): v=v.detach().cpu().numpy()
    step=cfg['full_well_e']/(2**cfg['adc_bits']-1-cfg['black_level_DN'])
    return float(np.sqrt(max(float(np.mean(v)),0)+cfg['read_noise_e']**2+step**2/12))

LEGACY_SCHEMA='legacy-scalar-raw-noise-interval-v1'
SCHEMA='marginal-statistic-scale-v2'
legacy_feature_from_prediction=feature_from_prediction

def legacy_calibrate(records):
    if not records or any(r['role']!='interval_calibration' for r in records): raise ValueError('Evaluation labels may not calibrate intervals')
    mothers={r['mother_id'] for r in records};assert len(mothers)==6
    first=[r for r in records if r['replicate']<64]
    assert all(sum(r['mother_id']==mid for r in first)==64 for mid in mothers)
    if any(r.get('statistics') is None for r in first):
        return {'available':False,'reason':'failed calibration draw retained','independent_mothers':6}
    errors=np.asarray([np.abs(np.asarray(r['statistics'])-np.asarray(r['truth_statistics']))/r['noise_feature'] for r in first])
    return {'schema':LEGACY_SCHEMA,'legacy':True,'deployment_qualified':False,
        'available':True,'constants':np.quantile(errors,.95,axis=0).tolist(),
        'statistics':u.STATISTIC_NAMES,'calibration_mothers':sorted(mothers),'independent_mothers':6,
        'draws_per_mother':64,'total_draws':384,'formula':'center +/- constant[stat] * sqrt(fitted raw mean + read variance + ADC variance)',
        'assumption':'empirical nominal synthetic scene and noise distribution; not measured calibration',
        'distribution_free95_guaranteed':False,'nominal_level':.95,
        'coverage_unit':'marginal statistic; mother clusters; not simultaneous full-image coverage',
        'evaluation_labels_used':False}

def legacy_interval(statistics,noise_feature,model):
    if model.get('schema') not in (None,LEGACY_SCHEMA): raise ValueError('Only historical scalar interval models are accepted here')
    if not model['available']: return None
    center=np.asarray(statistics);half=np.asarray(model['constants'])*noise_feature
    return np.stack([center-half,center+half],1)

def legacy_evaluate(records,model):
    rows=[]
    for mid in sorted({r['mother_id'] for r in records}):
        unit=[r for r in records if r['mother_id']==mid and r['replicate']<64]
        covered=np.zeros(12);widths=[];finite=0
        for r in unit:
            ci=legacy_interval(r['statistics'],r['noise_feature'],model) if r.get('statistics') is not None else None
            if ci is None: continue
            target=np.asarray(r['truth_statistics']);covered+=(target>=ci[:,0])&(target<=ci[:,1]);widths.append(ci[:,1]-ci[:,0]);finite+=1
        rows.append({'mother_id':mid,'role':unit[0]['role'],'declared_draws':64,'finite_draws':finite,
            'marginal_coverage_full_denominator':(covered/64).tolist(),
            'mean_interval_width':np.mean(widths,axis=0).tolist() if widths else None})
    evaluation=[r for r in rows if r['role']=='interval_evaluation']
    assert len(evaluation)==6 and not {r['mother_id'] for r in evaluation}&set(model.get('calibration_mothers',[]))
    return {'rows':rows,'evaluation_independent_mothers':6,'evaluation_noise_draws':384,
        'pooled_evaluation_marginal_coverage':np.mean([r['marginal_coverage_full_denominator'] for r in evaluation],axis=0).tolist(),
        'independent_unit_for_generalization':'6 mothers, not384 mothers',
        'statistics':u.STATISTIC_NAMES,'simultaneous_full_image_claimed':False,
        'finite_widths_and_failures_kept':True,'distribution_free_coverage_guaranteed':False}

def _vector(value,name,*,positive=False):
    a=np.asarray(value,dtype=float)
    if a.shape!=(12,) or not np.isfinite(a).all() or (positive and np.any(a<=0)):
        raise ValueError(name+' must be a finite 12-vector'+(' with strictly positive values' if positive else ''))
    return a

def _scale(value,provenance):
    a=_vector(value,'scale',positive=True)
    if not isinstance(provenance,dict): raise ValueError('Scale provenance is required')
    if provenance.get('full_inverse') is not True or provenance.get('inference_reads_truth') is not False:
        raise ValueError('Scales must propagate the whole inverse without truth input')
    if provenance.get('statistics')!=u.STATISTIC_NAMES: raise ValueError('Scale statistic order does not match')
    if provenance.get('complete') is not True: raise ValueError('Incomplete refits cannot supply an interval scale')
    kind=provenance.get('kind')
    if kind in ('full_inverse_bootstrap_sd','independent_full_inverse_mc_sd'):
        n=provenance.get('replicates',0)
        if not isinstance(n,int) or n<2 or n!=provenance.get('declared_replicates'):
            raise ValueError('A complete declared replicate count is required')
    elif kind=='whole_inverse_local_sd':
        if not (provenance.get('interval_scale_eligible') is True and provenance.get('step_size_check_passed') is True
                and provenance.get('refits_numerically_certified') is True and provenance.get('probe_convergence_verified') is True):
            raise ValueError('Uncertified local diagnostic scales cannot calibrate intervals')
        if provenance.get('probes',0)<provenance.get('minimum_probes_for_interval_scale',32):
            raise ValueError('Too few probes for an interval scale')
    else: raise ValueError('Unsupported scale source; scalar raw-noise features are legacy only')
    return a

def _identity(records,role):
    if not records or any(r.get('role')!=role for r in records):
        raise ValueError('Only '+role+' records are permitted')
    identities=[];geometries={}
    for r in records:
        mid=r.get('mother_id');rep=r.get('replicate')
        if not isinstance(mid,str) or not mid or not isinstance(rep,int) or rep<0:
            raise ValueError('Every record requires mother_id and nonnegative replicate')
        identities.append((mid,rep))
        geometry={k:r.get(k) for k in ('actual_temperature_geometry','actual_template_geometry')}
        if any(not isinstance(v,str) or not v for v in geometry.values()):
            raise ValueError('Actual generated geometries are required, not a generic display geometry')
        if mid in geometries and geometries[mid]!=geometry: raise ValueError('Conflicting geometry within one mother')
        geometries[mid]=geometry
    if len(identities)!=len(set(identities)): raise ValueError('Duplicate mother/replicate units')
    return geometries

def _declared_counts(records,geometries,declared_draws_by_mother):
    counts={mid:sum(r['mother_id']==mid for r in records) for mid in geometries}
    declared=counts if declared_draws_by_mother is None else dict(declared_draws_by_mother)
    if set(declared)!=set(counts) or any(not isinstance(n,int) or isinstance(n,bool) or n<counts[mid] or n<1 for mid,n in declared.items()):
        raise ValueError('Declared counts must cover every supplied mother and all its records')
    if any(r['replicate']>=declared[r['mother_id']] for r in records):
        raise ValueError('Replicate index lies outside the declared mother draw range')
    return counts,declared

def calibrate(records,*,schema=SCHEMA,nominal_level=.95,declared_draws_by_mother=None):
    """Calibrate per-output scales using calibration labels only.

    Mothers receive equal total weight even if their replicate counts differ.
    This is an empirical marginal error interval, not a distribution-free
    conformal guarantee. A low-count or oracle-scale run remains diagnostic.
    """
    if schema!=SCHEMA: raise ValueError('Use legacy_calibrate explicitly for the historical scalar model')
    if not 0<nominal_level<1: raise ValueError('Nominal coverage must lie between zero and one')
    records=list(records);geometries=_identity(records,'interval_calibration')
    counts,declared=_declared_counts(records,geometries,declared_draws_by_mother)
    result={'schema':SCHEMA,'available':False,'statistics':list(u.STATISTIC_NAMES),
        'nominal_level':float(nominal_level),'calibration_mothers':sorted(geometries),
        'actual_geometries':geometries,'independent_mothers':len(geometries),
        'draws_by_mother':declared,'returned_draws_by_mother':counts,'total_draws':sum(declared.values()),
        'returned_draws':len(records),'evaluation_labels_used':False,
        'formula':'center[stat] +/- constant[stat] * whole_inverse_output_scale[stat]',
        'endpoint_rounding':'one outward representable float at half-width multiplication and endpoint subtraction/addition',
        'quantile_rule':'mother-balanced empirical absolute standardized error, higher quantile',
        'coverage_unit':'marginal statistic; independent generalization units are mothers',
        'distribution_free95_guaranteed':False,'simultaneous_full_image_claimed':False,
        'deployment_qualified':False,'failed_records':[]}
    for mid,n in declared.items():
        returned={r['replicate'] for r in records if r['mother_id']==mid}
        result['failed_records'].extend({'mother_id':mid,'replicate':i,'reason':'declared draw did not return'} for i in range(n) if i not in returned)
    errors=[];weights=[];kinds=set();debug=False;oracle=False
    for r in records:
        if r.get('statistics') is None or r.get('scale') is None:
            result['failed_records'].append({'mother_id':r['mother_id'],'replicate':r['replicate'],'reason':'missing whole-inverse statistic or scale'})
            continue
        scale=_scale(r['scale'],r.get('scale_provenance'))
        estimate=_vector(r['statistics'],'statistics');truth=_vector(r['truth_statistics'],'truth_statistics')
        errors.append(abs(estimate-truth)/scale);weights.append(1./counts[r['mother_id']])
        provenance=r['scale_provenance'];kinds.add(provenance['kind'])
        debug=debug or bool(provenance.get('debug_only',True))
        oracle=oracle or provenance['kind']=='independent_full_inverse_mc_sd'
    result.update(scale_kinds=sorted(kinds),debug_only=debug or len(geometries)<6,
        oracle_scale_diagnostic=oracle,scale_available_at_raw_only_deployment=not oracle,
        declared_draws_retained=True)
    if result['failed_records']:
        result.update(status='CALIBRATION_UNAVAILABLE_FAILED_DRAW_RETAINED',constants=None)
        return result
    a=np.asarray(errors);w=np.asarray(weights);constants=[]
    for j in range(12):
        order=np.argsort(a[:,j],kind='stable');cum=np.cumsum(w[order]);idx=np.searchsorted(cum,nominal_level*cum[-1],side='left')
        constants.append(float(a[order[min(idx,len(order)-1)],j]))
    result.update(available=True,constants=constants,
        status='DIAGNOSTIC_CALIBRATION' if result['debug_only'] or oracle else 'EMPIRICAL_CALIBRATION_REQUIRES_INDEPENDENT_EVALUATION')
    return result

def interval(statistics,scale,model,*,scale_provenance):
    """Raw-only center and output scales; no evaluation labels or scalar scale."""
    if model.get('schema')!=SCHEMA: raise ValueError('Use legacy_interval explicitly for the historical scalar schema')
    if not model.get('available'): return None
    center=_vector(statistics,'statistics');sd=_scale(scale,scale_provenance)
    if scale_provenance['kind'] not in model.get('scale_kinds',[]):
        raise ValueError('Evaluation scale procedure differs from the frozen calibration model')
    constants=_vector(model['constants'],'constants')
    if np.any(constants<0): raise ValueError('Interval constants cannot be negative')
    # Division in calibration and multiplication here can place an empirical
    # quantile endpoint one rounding step inside its calibrated residual.
    # Outward arithmetic preserves the inclusive endpoint convention.
    half=np.nextafter(constants*sd,np.inf)
    bounds=np.stack([np.nextafter(center-half,-np.inf),np.nextafter(center+half,np.inf)],axis=1)
    if not np.isfinite(bounds).all(): raise ValueError('Interval arithmetic overflowed the finite output range')
    return bounds

def evaluate(records,model,*,declared_draws_by_mother=None):
    """Held-out scoring; fixed model is never fitted or inflated here.

    Pass the registered draw counts to retain absent records in denominators.
    Without that argument the explicit supplied records are the denominator;
    missing records must then be represented by null statistics/scale rows.
    """
    if model.get('schema')!=SCHEMA: raise ValueError('Use legacy_evaluate explicitly for the historical scalar schema')
    records=list(records);geometries=_identity(records,'interval_evaluation')
    if set(geometries)&set(model['calibration_mothers']): raise ValueError('Calibration and evaluation mothers must be disjoint')
    counts,declared=_declared_counts(records,geometries,declared_draws_by_mother)
    rows=[]
    for mid in sorted(geometries):
        unit=[r for r in records if r['mother_id']==mid];covered=np.zeros(12);widths=[];finite=0;failures=[]
        for r in unit:
            if r.get('statistics') is None or r.get('scale') is None:
                failures.append({'replicate':r['replicate'],'reason':'missing statistic or scale'});continue
            ci=interval(r['statistics'],r['scale'],model,scale_provenance=r['scale_provenance'])
            if ci is None:
                failures.append({'replicate':r['replicate'],'reason':'calibration unavailable'});continue
            target=_vector(r['truth_statistics'],'truth_statistics')
            covered+=(target>=ci[:,0])&(target<=ci[:,1]);widths.append(ci[:,1]-ci[:,0]);finite+=1
        rows.append({'mother_id':mid,**geometries[mid],'role':'interval_evaluation',
            'declared_draws':declared[mid],'returned_draws':len(unit),'finite_draws':finite,
            'missing_draws':declared[mid]-len(unit),'failed_records':failures,
            'marginal_coverage_full_denominator':(covered/declared[mid]).tolist(),
            'mean_interval_width':np.mean(widths,axis=0).tolist() if widths else None})
    return {'schema':SCHEMA,'rows':rows,'statistics':list(u.STATISTIC_NAMES),
        'evaluation_independent_mothers':len(rows),'evaluation_noise_draws':sum(declared.values()),
        'returned_draws':len(records),'finite_draws':sum(r['finite_draws'] for r in rows),
        'pooled_evaluation_marginal_coverage':np.mean([r['marginal_coverage_full_denominator'] for r in rows],axis=0).tolist(),
        'pooling_rule':'equal mother weight, using every declared draw in its mother denominator',
        'evaluation_labels_used_to_fit':False,'model_recalibrated':False,
        'independent_unit_for_generalization':str(len(rows))+' mothers',
        'debug_only':bool(model.get('debug_only')) or len(rows)<6,
        'oracle_scale_diagnostic':bool(model.get('oracle_scale_diagnostic')),
        'scale_available_at_raw_only_deployment':bool(model.get('scale_available_at_raw_only_deployment')),
        'deployment_qualified':False,'simultaneous_full_image_claimed':False,
        'finite_widths_and_failures_kept':True,'distribution_free_coverage_guaranteed':False}
