"""Root-only native frozen-float32 implementation control, not a cohort run."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from . import common
common.configure_runtime()
import torch
from . import scoring
from .stage_a import evaluate_source


@torch.no_grad()
def run(device='cuda'):
    t=time.perf_counter();p=common.make_physics(device)
    solver=common.frozen_v5('signed_linear_solver')
    linear=solver.SignedLinearOperator(p)
    h,w=p.cfg['shape'];yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=p.M.dtype),
          torch.arange(w,device=device,dtype=p.M.dtype),indexing='ij')
    state=p.M.new_zeros((1,5,h,w));state[:,0]=(3052.5-p.T0)/p.Tscale
    state[:,1]=-.5+.025*torch.cos(2*torch.pi*xx/w)
    beta=p.M.new_tensor([[.5,.015,-.01]])
    C=p.components(state,beta)[0]
    S=180*(.55+.25*torch.cos(2*torch.pi*xx/w)+.15*torch.sin(4*torch.pi*yy/h))
    noise=13*torch.cos(2*torch.pi*(17*xx/w+7*yy/h))
    clean=p.op(C)+linear.apply_template(S)[None]+linear.apply_beta(beta)
    raw=clean+noise[None]
    probe=1+.1*torch.cos(2*torch.pi*xx/w)
    field=2+.2*torch.sin(4*torch.pi*yy/h)
    check=[]
    def add(name,value,limit):check.append({'name':name,'value':float(value),'limit':limit,'passed':float(value)<=limit})
    left=(linear.apply_template(field)*probe).sum()
    right=(field*linear.adjoint_template(probe)).sum()
    add('native_template_forward_adjoint',abs(float(left-right))/max(abs(float(left)),abs(float(right)),1e-30),2e-5)
    add('native_template_fusion_single_blur',float((linear.apply_template(S)-p.op(p.M[:,2][None,:,None,None]*S[None,None])[0]).square().mean().sqrt()),.005)
    true_post=p.op.object_blur(C);T,valid=p.color_temperature(true_post)
    truth={'continuum_latent':C,'template_latent_e':S[None],'template_postblur_e':p.op.object_blur(S[None]),
           'temperature_color_K':T,'evaluation_domain':valid,'beta':beta}
    altered=state.clone();altered[:,0]+=.004*torch.cos(2*torch.pi*xx/w)
    altered[:,1]+=.001*torch.sin(4*torch.pi*yy/h)
    sources={'true':scoring.continuum_result(p,C),'perturbed':scoring.continuum_result(p,p.components(altered,beta)[0])}
    options={**solver.DEFAULT_OPTIONS,'alpha_g':1.0,'weights':solver.fixed_raw_weights(p,raw)}
    controls=[]
    support=int((p.op.object_otf!=0).sum())
    check.append({'name':'frozen_native_OTF_nonzero_support_1153','value':support,'expected':1153,'passed':support==1153})
    for mode in ('true_beta','estimated_beta'):
        cache=None
        for name,source in sources.items():
            result,d=evaluate_source(p,raw,source,truth,options,mode,cache)
            cache=d['shared_cache']
            summary=scoring.plain(d['summary'])
            diagnosis=None
            if summary.get('numerical_gate_passed') is not True:
                # A relative residual of the large original RHS can leave a
                # visible error compared with small noise/bias components.
                # Explain that remaining solve error without replacing output.
                from .decomposition import _solve
                system=cache['system']
                x=system['pack'](result['template_latent_e'][0],None if mode=='true_beta' else result['beta'][0])
                target=raw[0]-p.op(source['continuum_latent'])[0]
                if mode=='true_beta':target-=linear.apply_beta(beta[0])
                rhs=system['measurement_adjoint'](system['weights']*target)
                residual=rhs-system['matvec'](x)
                correction=_solve(system,residual,'actual_normal_residual_diagnostic')
                correction_s,_=system['unpack'](correction['value'])
                post=p.op.object_blur(correction_s)
                gap=d['arrays']['recomposition_residual_postblur_e']
                after=float((gap+post).double().square().mean().sqrt())
                diagnosis={'original_numerical_gate_passed':False,'cannot_rescue_primary_gate':True,
                    'correction_CG':scoring.plain(correction['CG']),
                    'normal_residual_effect_postblur_RMS_e':float(post.double().square().mean().sqrt()),
                    'recomposition_after_accounting_for_actual_solver_residual_RMS_e':after,
                    'interpretation':'extra term is numerical solve error, not an independent physical noise source'}
            expected_stop_explained=bool(diagnosis and summary['full_leak_noise_bias_CG_converged']
                 and summary['actual_v5_CG_converged'] and summary['actual_v5_accepted']
                 and summary['normal_equation_algebra_passed'] and not summary['postblur_recomposition_passed']
                 and diagnosis['correction_CG']['converged']
                 and diagnosis['recomposition_after_accounting_for_actual_solver_residual_RMS_e']<=.005)
            check.append({'name':'native_'+mode+'_'+name+'_identity_or_correct_explained_stop',
                 'primary_identity_passed':summary.get('numerical_gate_passed') is True,
                 'known_failed_primary_retained':expected_stop_explained,
                 'passed':summary.get('numerical_gate_passed') is True or expected_stop_explained})
            scored=common.frozen_v5('metrics').score_prediction({**source,**result,'raw_e':raw,'valid':source['valid']},truth,p)
            check.append({'name':'native_'+mode+'_'+name+'_finite_external_scoring','passed':scored['metrics'] is not None and scored['metrics']['template_SSE'] is not None})
            controls.append({'beta_mode':mode,'continuum':name,'summary':summary,'external_metrics':scored['metrics'],
                             'numerical_diagnosis':diagnosis})
    return {'schema':'v6-native-frozen-buffer-numerical-control-001','time':common.now(),
       'device':device,'dtype':str(p.M.dtype),'shape':[h,w],'all_passed':all(r['passed'] for r in check),
       'checks':check,'controls':controls,'wall_s':time.perf_counter()-t,
       'all_primary_native_identities_passed':all(c['summary']['numerical_gate_passed'] for c in controls),
       'implementation_control_meaning':'correct operator/score/identity accounting or explicitly diagnosed and retained finite-PCG stop; not scientific acceptance',
       'new_or_sealed_mother_arrays_generated':False,'scientific_cohort_acceptance':False,
       'synthetic_fixture_only':True,'OTF_nonzero_support':support,'OTF_exact_zero_frequencies':h*w-support,
       'source_sha256':{common.relative(common.CODE/n):common.sha(common.CODE/n) for n in
            ('tests_native.py','stage_a.py','scoring.py','common.py','decomposition.py')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',required=True,type=Path);a=parser.parse_args()
    r=run(a.device);common.write_json(a.output,r)
    print({'all_passed':r['all_passed'],'checks':len(r['checks']),'wall_s':r['wall_s'],'receipt':str(a.output)},flush=True)
    if not r['all_passed']:
        print([x for x in r['checks'] if not x['passed']],flush=True);raise SystemExit(1)


if __name__=='__main__':main()
