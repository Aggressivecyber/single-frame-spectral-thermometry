"""Four-term accounting includes actual and component finite-solver errors."""
from . import storage
import torch

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
