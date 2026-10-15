"""CPU32 forward-only contracts for future C1 known-exposure integration."""
from .. import storage as s
from .. import measurement as m
from ..smooth_calibration import SmoothCalibrationPhysics,tensor_sha
from ..smooth_calibration_exposure import known_exposure,contract
from ..deployment import known_exposure as legacy_known_exposure
from .smooth_calibration import _small_reference
import argparse
import ast
import json
import numpy as np
import torch
import time


def _snapshot(p):
    return {'buffers':dict(p.named_buffers()),'hashes':{name:tensor_sha(value) for name,value in p.named_buffers()},
        'cfg':p.cfg,'Sscale':p.Sscale,'Bscale':p.Bscale,'adc_step':p.adc_step,'extra_variance':p.extra_variance}


def _restored(p,snapshot):
    current=dict(p.named_buffers())
    return (set(current)==set(snapshot['buffers'])
        and all(current[name] is value and tensor_sha(value)==snapshot['hashes'][name]
            for name,value in snapshot['buffers'].items())
        and p.cfg is snapshot['cfg'] and p.Sscale==snapshot['Sscale'] and p.Bscale==snapshot['Bscale']
        and p.adc_step==snapshot['adc_step'] and p.extra_variance==snapshot['extra_variance'])


def _relative_error(actual,expected):
    return float((actual-expected).abs().max()/expected.abs().max().clamp_min(1.))


def _forward(p,state,beta,probe):
    live_state=state.detach().requires_grad_(True);live_beta=beta.detach().requires_grad_(True)
    components=p.components(live_state,live_beta)
    encoded=[p.op(component) for component in components[:3]]
    raw=p(live_state,live_beta)
    gradient=torch.autograd.grad((raw*probe).sum(),(live_state,live_beta))
    return {'components':tuple(value.detach() for value in components),
        'encoded':tuple(value.detach() for value in encoded),'raw':raw.detach(),
        'state_gradient':gradient[0].detach(),'beta_gradient':gradient[1].detach()}


def run(output=None):
    start=time.perf_counter();torch.set_num_threads(2)
    selected=m.selection()['physics_config'];p=SmoothCalibrationPhysics(_small_reference(selected))
    saved=_snapshot(p);checks=[]
    def check(name,passed,**details):
        checks.append({'name':name,'passed':bool(passed),**s.plain(details)})
    yy,xx=torch.meshgrid(torch.arange(32,dtype=torch.float64),torch.arange(32,dtype=torch.float64),indexing='ij')
    T=3300.+.08*torch.cos(2*torch.pi*xx/32)+.03*torch.sin(2*torch.pi*yy/32)
    state=p.M.new_zeros((1,5,32,32));state[:,0]=(T-p.T0)/p.Tscale
    state[:,1]=np.log(.53)+.02*torch.sin(2*torch.pi*xx/32)
    state[:,4]=.17*torch.cos(2*torch.pi*xx/32)-.08*torch.sin(2*torch.pi*yy/32)
    beta=p.M.new_tensor([[.02,.003,-.004]])
    probe=(torch.sin(2*torch.pi*xx/32)+.4*torch.cos(4*torch.pi*yy/32))[None]
    base=_forward(p,state,beta,probe)
    temperature=T[None].detach().requires_grad_(True)
    C,dC=p.continuum(temperature)
    band_probe=torch.linspace(.2,1.4,7,dtype=torch.float64)[None,:,None,None]
    actual_derivative=torch.autograd.grad((C*band_probe).sum(),temperature)[0]
    C=C.detach();dC=dC.detach();actual_derivative=actual_derivative.detach()

    # Reproduce the actual integration defect before applying the wrapper.
    with legacy_known_exposure(p,2.):
        old_C,old_dC=p.continuum(T[None])
        old_raw=p(state,beta)
        old_errors={'continuum_value_maximum_absolute_e':float((old_C-2*C).abs().max()),
            'analytic_derivative_maximum_absolute_e_per_K':float((old_dC-2*dC).abs().max()),
            'encoded_full_raw_maximum_absolute_e':float((old_raw-2*base['raw']).abs().max()),
            'slope_buffer_was_unscaled':p.c1_shared_slopes is saved['buffers']['c1_shared_slopes']}
    check('legacy_exposure_helper_C1_window_inconsistency_is_reproduced',
        old_errors['continuum_value_maximum_absolute_e']>1e-6
        and old_errors['analytic_derivative_maximum_absolute_e_per_K']>1e-6
        and old_errors['encoded_full_raw_maximum_absolute_e']>1e-6 and _restored(p,saved),
        ratio=2.,**old_errors,scope='forward-only synthetic calibration-window fixture; no registered raw')

    summaries=[]
    for ratio in (.25,1.,2.):
        with known_exposure(p,ratio) as context_p:
            scaled=_forward(p,state,beta,probe)
            live_T=T[None].detach().requires_grad_(True);counts,derivative=p.continuum(live_T)
            grad_T=torch.autograd.grad((counts*band_probe).sum(),live_T)[0]
            errors={'continuum':_relative_error(counts,ratio*C),
                'analytic_continuum_derivative':_relative_error(derivative,ratio*dC),
                'actual_continuum_autograd':_relative_error(grad_T,ratio*actual_derivative),
                'full_encoded_raw':_relative_error(scaled['raw'],ratio*base['raw']),
                'actual_forward_state_VJP':_relative_error(scaled['state_gradient'],ratio*base['state_gradient']),
                'actual_forward_background_VJP':_relative_error(scaled['beta_gradient'],ratio*base['beta_gradient'])}
            for index,name in enumerate(('continuum','species','background','temperature_Jacobian')):
                errors['component_'+name]=_relative_error(scaled['components'][index],ratio*base['components'][index])
            for index,name in enumerate(('continuum','species','background')):
                errors['encoded_'+name]=_relative_error(scaled['encoded'][index],ratio*base['encoded'][index])
            fixed=all(p.cfg[name]==saved['cfg'][name] for name in
                ('read_noise_e','full_well_e','adc_bits','black_level_DN'))
            fixed=fixed and p.adc_step==saved['adc_step'] and p.extra_variance==saved['extra_variance']
            fixed=fixed and all(dict(p.named_buffers())[name] is value for name,value in saved['buffers'].items()
                if name not in ('continuum_lut','c1_shared_slopes'))
            meta=all(p.cfg[name]==saved['cfg'][name]*ratio for name in
                ('exposure_s','reference_905_e','species_scale_e','background_scale_e'))
            good=context_p is p and max(errors.values())<1e-10 and fixed and meta
        restored=_restored(p,saved)
        summaries.append({'ratio':ratio,'maximum_normalized_error':max(errors.values()),
            'errors':errors,'ADC_read_and_optics_unchanged':fixed,
            'exposure_metadata_scaled':meta,'same_buffer_and_config_objects_restored':restored})
        check(f'all_components_C1_values_derivatives_and_actual_raw_VJP_scale_at_ratio_{ratio:g}',
            good and restored,**summaries[-1])

    with known_exposure(p,.25):
        outer=_snapshot(p)
        with known_exposure(p,2.):
            nested=_forward(p,state,beta,probe)
            nested_error=max(_relative_error(nested[name],.5*base[name]) for name in
                ('raw','state_gradient','beta_gradient'))
            nested_counts,nested_derivative=p.continuum(T[None])
            nested_error=max(nested_error,_relative_error(nested_counts,.5*C),_relative_error(nested_derivative,.5*dC))
        inner_restore=_restored(p,outer)
    check('nested_contexts_compose_ratios_and_restore_each_previous_buffer_object',
        nested_error<1e-10 and inner_restore and _restored(p,saved),
        ratios=[.25,2.],effective_ratio=.5,maximum_normalized_error=nested_error,
        inner_restored_outer_objects=inner_restore,outer_restored_original_objects=_restored(p,saved))

    inner_exception=False;outer_exception=False;after_inner=False
    try:
        with known_exposure(p,2.):
            outer=_snapshot(p)
            try:
                with known_exposure(p,.25): raise RuntimeError('injected inner-context exception')
            except RuntimeError: inner_exception=True
            after_inner=_restored(p,outer)
            raise RuntimeError('injected outer-context exception')
    except RuntimeError: outer_exception=True
    check('exceptions_restore_inner_and_outer_context_objects_and_bytes',
        inner_exception and outer_exception and after_inner and _restored(p,saved),
        inner_exception_observed=inner_exception,outer_exception_observed=outer_exception,
        inner_restored_outer_objects=after_inner,all_original_objects_restored=_restored(p,saved))

    rejected=[]
    for invalid in (0.,-1.,float('nan'),float('inf'),-float('inf'),'not-a-ratio',None):
        try:
            with known_exposure(p,invalid): pass
            rejected.append(False)
        except ValueError: rejected.append(_restored(p,saved))
    check('nonfinite_nonpositive_and_invalid_ratios_are_rejected_before_any_mutation',
        all(rejected),attempts=len(rejected),rejected_and_unchanged=sum(rejected))

    calibration=s.read(s.REPORTS/'repair_001/calibration_v003/CALIBRATION_CONTRACTS.json')
    core_hashes=calibration['source_sha256']
    core_unchanged=all(s.sha(s.ROOT/name)==digest for name,digest in core_hashes.items())
    registration=s.read(s.REPORTS/'repair_001/quick_v001/QUICK_REGISTRATION.json')
    frozen=registration['source_sha256']
    old_unchanged=all(s.sha(s.ROOT/name)==digest for name,digest in frozen.items())
    controller=s.ROOT/'research_closeout/calibration_quick.py'
    imports=[]
    if controller.exists():
        for node in ast.walk(ast.parse(controller.read_text(encoding='utf8'))):
            if isinstance(node,ast.ImportFrom): imports.append(node.module or '')
            elif isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
    unused=not any('smooth_calibration_exposure' in name for name in imports)
    check('frozen_C1_core_v001_sources_and_v002_controller_are_untouched_and_unintegrated',
        core_unchanged and old_unchanged and unused,
        C1_core_dependency_hashes_verified=len(core_hashes),v001_source_hashes_verified=len(frozen),
        v002_imports_exposure_helper=not unused)

    result={'time':s.now(),'all_passed':all(row['passed'] for row in checks),'tests':checks,
        'scope':'locally authored CPU32 forward-only future exposure helper contracts; no native inversion, GPU, registered raw, scientific thresholds or final payloads',
        'exposure_contract':contract(),'ratios':[.25,1.,2.],'legacy_helper_defect':old_errors,
        'formal_inverse_calls':0,'native256_inverse_calls':0,'GPU_used':False,
        'formal_raws_generated':False,'v002_integration':False,'scientific_validation':False,
        'actual_gradient_scope':'VJP of encoded raw against a fixed probe; electron-domain Jacobian scales once with known exposure',
        'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/smooth_calibration_exposure.py',
            'research_closeout/tests/smooth_calibration_exposure.py',
            'research_closeout/smooth_calibration.py','research_closeout/deployment.py')},
        'wall_s':time.perf_counter()-start}
    s.write(output or s.REPORTS/'repair_001/calibration_v003/EXPOSURE_CONTRACTS.json',result,replace=True)
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output')
    raise SystemExit(0 if run(parser.parse_args().output)['all_passed'] else 2)
