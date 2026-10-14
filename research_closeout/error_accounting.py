"""Four-term accounting includes actual and component finite-solver errors."""
from . import storage
import torch

@torch.no_grad()
def propagate_residual(system,rhs,*,rtol=1e-11,max_iterations=1000,block_iterations=200):
    """Diagnostic K inverse using bounded calls to the unchanged frozen PCG.

    The old API caps each call at200. True-residual correction blocks share
    the same K/preconditioner and a single global iteration budget. This
    reference does not alter the actual estimator's200-iteration budget.
    """
    from . import storage as s
    frozen=s.frozen_bridge().frozen_v5('signed_linear_solver')
    if not 0<rtol<=1e-4 or not 1<=block_iterations<=200 or max_iterations<1:
        raise ValueError('Invalid diagnostic residual reference budget')
    x=torch.zeros_like(rhs);scale=system['residual_scale'].sqrt()
    rhs_norm=float((rhs/scale).norm());blocks=[];used=0
    relative=0. if rhs_norm==0 else 1.
    while relative>rtol and used<max_iterations:
        residual=rhs-system['matvec'](x)
        remaining=min(block_iterations,max_iterations-used)
        residual_norm=float((residual/scale).norm())
        local_tolerance=min(1e-4,rtol*rhs_norm/max(residual_norm,1e-300))
        correction,info=frozen.pcg_true_residual(system['matvec'],residual,system['precondition'],
            residual_scale=system['residual_scale'],rtol=local_tolerance,
            max_iterations=remaining,refresh_interval=25)
        x=x+correction
        iterations=int(info.get('iterations',remaining));used+=max(iterations,1)
        relative=float(((rhs-system['matvec'](x))/scale).norm())/rhs_norm
        blocks.append({'requested_max_iterations':remaining,'local_rtol':local_tolerance,
            'global_true_relative_residual_after':relative,'CG':info})
        if not torch.isfinite(x).all(): break
    return x,{'role':'propagated_numerical_residual_reference','converged':bool(relative<=rtol and torch.isfinite(x).all()),
        'relative_residual_true':relative,'rtol':rtol,'iterations':used,'maximum_iterations':max_iterations,
        'frozen_API_call_cap':block_iterations,'actual_estimator_budget_unchanged':True,'blocks':blocks}

def summarize(values):
    keys=tuple(values)
    return {'components':{k:{'RMS':float(v.square().mean().sqrt()),'bias':float(v.mean()),'MSE':float(v.square().mean())} for k,v in values.items()},
        'cross_terms':{a+'__'+b:float(2*(values[a]*values[b]).mean()) for i,a in enumerate(keys) for b in keys[i+1:]},
        'sum_RMS':float(sum(values.values()).square().mean().sqrt()),
        'sum_MSE':float(sum(values.values()).square().mean())}

def same_factor_accounting(A,data_rhs,reg_rhs,x_true,leak_rhs,noise_rhs,actual_x,component_x=None):
    """A is the original augmented physical system on a declared finite subspace.

    The QR factor is formed once. All exact physical components, actual-error
    propagation, component residuals and algebraic roundoff share that factor.
    ``reg_rhs`` is -Lambda*x_true; leak/noise RHS are normal RHS vectors.
    This is a reference contract, not full-image scientific attribution.
    """
    q,r=torch.linalg.qr(A,mode='reduced')
    K=A.T@A;b=A.T@data_rhs
    def inverse(B):
        v=torch.linalg.solve_triangular(r.T,B,upper=False)
        return torch.linalg.solve_triangular(r,v,upper=True)
    B=torch.stack([leak_rhs,noise_rhs,reg_rhs,b-K@actual_x],1)
    solved=inverse(B)
    physical={'continuum_leakage':solved[:,0],'shared_measurement_noise':solved[:,1],
        'regularization_bias':solved[:,2],'actual_numerical_error':-solved[:,3]}
    algebra=b-K@x_true-leak_rhs-noise_rhs-reg_rhs
    physical['normal_algebra_roundoff']=inverse(algebra[:,None])[:,0]
    true_error=actual_x-x_true
    closure=true_error-sum(physical.values())
    result={'four_terms':physical,'actual_error':true_error,'closure':closure,
        'three_term_closure_RMS':float((true_error-sum(list(physical.values())[:3])).square().mean().sqrt()),
        'four_term_closure_RMS':float(closure.square().mean().sqrt()),'reference_factor':'single float64 QR',
        'condition_estimate':float(torch.linalg.cond(r))}
    if component_x is not None:
        comp=torch.stack(component_x,1)
        residuals=torch.stack([leak_rhs,noise_rhs,reg_rhs],1)-K@comp
        corrections=inverse(residuals)
        result['component_numerical_errors']=-corrections
        result['approximate_component_recomposition']=comp.sum(1)-solved[:,3]+corrections.sum(1)+physical['normal_algebra_roundoff']
        result['approximate_component_closure_RMS']=float((true_error-result['approximate_component_recomposition']).square().mean().sqrt())
    return result
