"""Finite empirical interval calibration with disjoint mother-scene evaluation."""
from . import uncertainty as u
import numpy as np

def feature_from_prediction(out,cfg):
    v=out['predicted_raw_e']
    if hasattr(v,'detach'): v=v.detach().cpu().numpy()
    step=cfg['full_well_e']/(2**cfg['adc_bits']-1-cfg['black_level_DN'])
    return float(np.sqrt(max(float(np.mean(v)),0)+cfg['read_noise_e']**2+step**2/12))

def calibrate(records):
    if not records or any(r['role']!='interval_calibration' for r in records): raise ValueError('Evaluation labels may not calibrate intervals')
    mothers={r['mother_id'] for r in records};assert len(mothers)==6
    first=[r for r in records if r['replicate']<64]
    assert all(sum(r['mother_id']==mid for r in first)==64 for mid in mothers)
    if any(r.get('statistics') is None for r in first):
        return {'available':False,'reason':'failed calibration draw retained','independent_mothers':6}
    errors=np.asarray([np.abs(np.asarray(r['statistics'])-np.asarray(r['truth_statistics']))/r['noise_feature'] for r in first])
    return {'available':True,'constants':np.quantile(errors,.95,axis=0).tolist(),
        'statistics':u.STATISTIC_NAMES,'calibration_mothers':sorted(mothers),'independent_mothers':6,
        'draws_per_mother':64,'total_draws':384,'formula':'center +/- constant[stat] * sqrt(fitted raw mean + read variance + ADC variance)',
        'assumption':'empirical nominal synthetic scene and noise distribution; not measured calibration',
        'distribution_free95_guaranteed':False,'nominal_level':.95,
        'coverage_unit':'marginal statistic; mother clusters; not simultaneous full-image coverage',
        'evaluation_labels_used':False}

def interval(statistics,noise_feature,model):
    if not model['available']: return None
    center=np.asarray(statistics);half=np.asarray(model['constants'])*noise_feature
    return np.stack([center-half,center+half],1)

def evaluate(records,model):
    rows=[]
    for mid in sorted({r['mother_id'] for r in records}):
        unit=[r for r in records if r['mother_id']==mid and r['replicate']<64]
        covered=np.zeros(12);widths=[];finite=0
        for r in unit:
            ci=interval(r['statistics'],r['noise_feature'],model) if r.get('statistics') is not None else None
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
