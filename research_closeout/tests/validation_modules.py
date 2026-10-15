"""Small fixtures for independent rendering, shared-raw covariance and null units."""
from .. import storage as s
from .. import independent_renderer as renderer
from .. import uncertainty as u
import copy
import json
import numpy as np

def run():
    cfg=copy.deepcopy(s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config'])
    cfg['shape']=[32,32];cfg['carrier_bins_xy']=[[round(a/8),round(b/8)] for a,b in cfg['carrier_bins_xy']]
    a=renderer.render(cfg,seed=913001,geometry='double_hotspot',coefficient_e=-45.)
    b=renderer.render(cfg,seed=913001,geometry='double_hotspot',coefficient_e=-45.,photon_ratio=2.)
    c=renderer.render(cfg,seed=913001,geometry='double_hotspot',coefficient_e=-45.,wavelength_samples=641)
    dynamic=renderer.render_exposure(cfg,seed=913001,geometry='hotspot_small',coefficient_e=45.,mid_fraction=.5)
    raw,receipt=renderer.sample_raw(a['expected_e'],cfg,913002)
    raw2,_=renderer.sample_raw(a['expected_e'],cfg,913002)
    checks=[]
    def check(name,passed,**evidence): checks.append({'name':name,'passed':bool(passed),**evidence})
    check('independent_renderer_finite_physical_frame',a['physically_available'] and a['expected_e'].shape==(32,32) and raw.dtype==np.uint16)
    budget_error=float(np.max(np.abs(b['expected_e']-2*a['expected_e'])))
    check('single_photon_budget_scales_all_raw_components',budget_error<1e-9 and np.array_equal(a['template_postblur_e_ref'],b['template_postblur_e_ref']),maximum_error_e=budget_error)
    spectral_difference=float(np.sqrt(np.mean((a['expected_e']-c['expected_e'])**2)))
    check('independent321_vs641_spectral_quadrature',spectral_difference<.17,raw_RMS_difference_e=spectral_difference)
    check('Poisson_read_ADC_seed_reproducibility',np.array_equal(raw,raw2) and receipt['Poisson_read_ADC_once'])
    check('exposure_integrates_bands_before_color',dynamic['physically_available'] and dynamic['metadata']['color_target'].startswith('ratio of exposure'))
    adc=cfg['full_well_e']/(2**cfg['adc_bits']-1-cfg['black_level_DN'])
    mean=np.full((32,32),4000.)
    def refit(dn):
        y=(np.asarray(dn)-cfg['black_level_DN'])*adc
        # Both components use this same measurement. Independent variance
        # addition would be wrong even though this fixture is linear.
        continuum=.9*y;template=y-continuum
        return {'temperature_color_K':3000+.05*y,'template_postblur_e_ref':template,
            'predicted_raw_e':y,'numerical_valid':True}
    estimate=refit(mean/adc+cfg['black_level_DN'])
    linear=u.legacy_whole_inverse_linearization(refit,estimate,cfg,seed=913003,probes=8)
    expected=.01*(4000+cfg['read_noise_e']**2+adc**2/12)
    actual=linear['covariance'][6][6]
    check('whole_inverse_shared_raw_variance',abs(actual-expected)/expected<1e-7,actual=actual,reference=expected)
    bootstrap=u.conditional_bootstrap(refit,estimate,cfg,seed=913004,replicates=64)
    check('full_refit64_marginal_bootstrap',bootstrap['complete'] and len(bootstrap['samples'])==64 and all(v>0 for v in bootstrap['width']) and not bootstrap['independent_coverage_test_performed'])
    threshold=u.null_threshold(np.arange(200.),[f'fixture_mother_{i}' for i in range(200)])
    check('frame_null_order_statistic',threshold['order_statistic']==191 and threshold['calibration_false_positive_fraction']==.045)
    rejected=False
    try: u.null_threshold([1.,2.],['same_mother','same_mother'])
    except ValueError: rejected=True
    check('noise_replicates_not_independent_mothers',rejected)
    result={'time':s.now(),'all_passed':all(t['passed'] for t in checks),'tests':checks,
        'fixture_shape':[32,32],'formal_data_generated':False,'scientific_coverage_or_accuracy_claimed':False,
        'source_sha256':{v.relative_to(s.ROOT).as_posix():s.sha(v) for v in
            (s.ROOT/'research_closeout/independent_renderer.py',s.ROOT/'research_closeout/uncertainty.py',s.ROOT/'research_closeout/tests/validation_modules.py')}}
    s.write(s.REPORTS/'VALIDATION_MODULE_CHECKS.json',result)
    print(json.dumps(result,ensure_ascii=True),flush=True)
    return result

if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
