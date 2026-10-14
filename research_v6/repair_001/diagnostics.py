"""Coherent repaired inverse decomposition with the unchanged observable gates."""
from . import storage as s
from .solver import solve_rhs
from ..decomposition import _stats,_pair,DEFAULT_IDENTITY_CRITERIA
import torch


@torch.no_grad()
def shared(system,truth):
    p=system['physics'];G=system['linear_operator'];W=system['weights'];shape=system['shape']
    ct=truth['continuum_latent'].reshape(1,7,*shape).double()
    st=truth['template_latent_e'].reshape(shape).double();bt=truth['beta'].reshape(3).double()
    xt=system['pack'](st,None if system['fixed_beta'] is not None else bt)
    C_raw=p.op(ct)[0]
    noise=system['raw_e']-C_raw-G.apply_template(st)-G.apply_beta(bt)
    n_rhs=system['measurement_adjoint'](W*noise);r_rhs=-system['regularizer'](xt)
    xn,n_cg=solve_rhs(system,n_rhs,'noise');xr,r_cg=solve_rhs(system,r_rhs,'regularization_bias')
    return {'C_true':ct,'S_true':st,'beta_true':bt,'x_true':xt,'noise_raw':noise,
         'noise_rhs':n_rhs,'bias_rhs':r_rhs,'noise_x':xn,'bias_x':xr,'noise_CG':n_cg,'bias_CG':r_cg}


@torch.no_grad()
def decompose(system,cache,source,result,truth):
    p=system['physics'];W=system['weights'];shape=system['shape'];Gt=system['measurement_adjoint']
    ce=source['continuum_latent'].reshape(1,7,*shape).double()
    leak_raw=p.op(cache['C_true']-ce)[0]
    leak_rhs=Gt(W*leak_raw);xl,l_cg=solve_rhs(system,leak_rhs,'continuum_leakage')
    vectors={'leakage':xl,'noise':cache['noise_x'],'regularization_bias':cache['bias_x']}
    reports={'leakage':l_cg,'noise':cache['noise_CG'],'regularization_bias':cache['bias_CG']}
    parts={key:p.op.object_blur(system['unpack'](value)[0]) for key,value in vectors.items()}
    summed=sum(vectors.values());error=result['vector']-cache['x_true']
    internal_error=p.op.object_blur(system['unpack'](error)[0])
    stored_error=result['template_postblur_e'][0]-truth['template_postblur_e'].reshape(shape).double()
    residual=stored_error-sum(parts.values());internal_residual=internal_error-sum(parts.values())
    algebra=result['rhs']-system['matvec'](cache['x_true'])-leak_rhs-cache['noise_rhs']-cache['bias_rhs']
    scale=system['residual_scale'].sqrt()
    relative=float(torch.linalg.vector_norm(algebra/scale)/
        max(float(torch.linalg.vector_norm(result['rhs']/scale)),float(torch.linalg.vector_norm(system['matvec'](cache['x_true'])/scale)),1e-30))
    rms=_stats(residual)['rms'];err=_stats(stored_error)['rms']
    crit=DEFAULT_IDENTITY_CRITERIA
    l2=rms/max(err,crit['postblur_relative_l2_scale_floor_e'])
    all_cg=all(v['converged'] for v in reports.values()) and result['info']['frames'][0]['CG']['converged']
    actual=result['info']['frames'][0]
    numeric=bool(all_cg and actual['accepted'] and relative<=crit['normal_equation_algebra_rtol']
                 and rms<=crit['postblur_rms_limit_e'] and l2<=crit['postblur_relative_l2_limit'])
    names=tuple(parts)
    cross={a+'__'+b:_pair(parts[a],parts[b]) for i,a in enumerate(names) for b in names[i+1:]}
    mse={k:float(v.square().mean()) for k,v in parts.items()}
    summary={'schema':'v6-repair001-decomposition-001','numeric_dtype':'torch.float64',
       'numerical_gate_passed':numeric,'full_actual_and_component_CG_converged':all_cg,
       'actual_accepted':actual['accepted'],'component_CG':reports,'actual_CG':actual['CG'],
       'postblur_error':_stats(stored_error),'postblur_components':{k:_stats(v) for k,v in parts.items()},
       'postblur_component_mse_e2':mse,'postblur_cross_terms':cross,
       'postblur_recomposition_residual':_stats(residual),'postblur_recomposition_relative_l2':l2,
       'internal_operator_recomposition_residual':_stats(internal_residual),
       'stored_label_minus_internal_target':_stats(stored_error-internal_error),
       'normal_equation_algebra_relative_equilibrated':relative,'identity_criteria':dict(crit),
       'accounted_MSE_e2':sum(mse.values())+sum(v['twice_mean_product'] for v in cross.values()),
       'error_leakage_correlation':_pair(stored_error,parts['leakage']),
       'physics_regularizer_and_weight_preserved':system['precision_contract'],
       'failure_reasons':[] if numeric else [name for name,passed in
          (('CG',all_cg),('actual_acceptance',actual['accepted']),('algebra',relative<=crit['normal_equation_algebra_rtol']),
           ('absolute_closure',rms<=crit['postblur_rms_limit_e']),('relative_closure',l2<=crit['postblur_relative_l2_limit'])) if not passed]}
    arrays={'template_postblur_error_e':stored_error,'recomposition_residual_postblur_e':residual,
            **{k+'_postblur_e':v for k,v in parts.items()}}
    return {'summary':summary,'arrays':arrays}
