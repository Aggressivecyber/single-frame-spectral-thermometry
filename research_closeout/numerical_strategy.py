"""The single preregistered recovery of the existing profile objective.

Only numerical coordinates and iteration budget change. The physical objective,
initialization, regularizers, latent bounds and observation remain unchanged.
"""
from . import storage as s
from . import measurement as m
from .joint_estimator import ProfileObjective,initialize,DEFAULTS
import time
import numpy as np
import torch
from scipy.optimize import minimize

RECOVERY={'strategy':'fixed_channel_diagonal_scaling_and_longer_LBFGSB',
    'grid':256,'outer_maxiter':600,'outer_maxfun':1600,'outer_ftol':1e-10,'outer_gtol':1e-3,
    'base_limit_s':120.,'cost_limit_s':180.,'reference_maxiter':300,
    'reference_maxfun':800,'reference_ftol':1e-12,'reference_gtol':1e-5}

class RecoveredObjective(ProfileObjective):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # regscale=.01*multiplier*mean whitened Jacobian diagonal.
        # The scale cancels the regularizer multiplier: no hyperparameter is
        # redefined by numerical equilibration. Fixed once at the initializer.
        hessian=self.regscale/(.01*self.opt['continuum_multiplier'])
        block_scale=hessian.clamp_min(1e-10).rsqrt().clamp(1e-5,1e3)
        self.coordinate_scale=block_scale[None,:,None,None].expand(self.shape).cpu().numpy().ravel()
        self.last_result=None;self.force_tolerance=None;self.deadline=self.started+self.opt['base_limit_s']
    def evaluate(self,*args,**kwargs):
        if time.perf_counter()>self.deadline: raise TimeoutError('Registered outer numerical sub-budget reached')
        self.last_result=super().evaluate(*args,**kwargs)
        return self.last_result
    def scaled_fun(self,v):
        value,gradient,*_=self.evaluate(v*self.coordinate_scale,forced_tolerance=self.force_tolerance)
        return value,gradient*self.coordinate_scale

def recovered_infer(raw_dn,p,options=None,*,initializer='v3',v3_state=None):
    """Raw-only invocation of the same physical estimator, with certification.

    A tighter outer continuation is numerical certification within this one
    inversion, not a separately selected candidate or extra measurement.
    """
    opt={**DEFAULTS,**RECOVERY,**(options or {})};started=time.perf_counter()
    raw=m.raw_tensor(raw_dn,p);raw_e=p.raw_to_electrons(raw)
    z0=initialize(p,raw,initializer,v3_state)
    objective=RecoveredObjective(p,raw_e,z0,opt)
    objective.started=started;objective.deadline=started+opt['base_limit_s']
    scale=objective.coordinate_scale;g=objective.g;limits=p.cfg['temperature_range_K']
    lower=np.r_[np.full(g*g,(limits[0]-p.T0)/p.Tscale),np.full(g*g,-4.)]/scale
    upper=np.r_[np.full(g*g,(limits[1]-p.T0)/p.Tscale),np.full(g*g,.3)]/scale
    bounds=np.column_stack([lower,upper]);base=None;reference=None;errors=[];production=None
    def optimize(start,tighter=False):
        return minimize(objective.scaled_fun,start,method='L-BFGS-B',jac=True,bounds=bounds,
            options={'maxiter':opt['reference_maxiter'] if tighter else opt['outer_maxiter'],
                'maxfun':opt['reference_maxfun'] if tighter else opt['outer_maxfun'],
                'ftol':opt['reference_ftol'] if tighter else opt['outer_ftol'],
                'gtol':opt['reference_gtol'] if tighter else opt['outer_gtol'],'maxls':20,'maxcor':20})
    try:
        base=optimize(objective.z0.cpu().numpy().ravel()/scale)
        objective.deadline=objective.started+opt['cost_limit_s']
        production=objective.evaluate(base.x*scale,forced_tolerance=opt['inner_tolerances'][-1])
        if base.success:
            objective.force_tolerance=opt['reference_inner_tolerance']
            reference=optimize(base.x,tighter=True)
            tighter=objective.evaluate(reference.x*scale,forced_tolerance=opt['reference_inner_tolerance'])
        else: tighter=None
    except TimeoutError as error:
        errors.append(str(error));production=production if production is not None else objective.last_result;tighter=None
    if production is None: raise ArithmeticError('No finite physical profile iterate returned')
    _,gradient,z,C,x,inner=production
    inner_reference,inner_info=objective.linear.solve(raw_e-p.op(C),tolerance=opt['reference_inner_tolerance'],maxiter=2*opt['inner_maxiter'],x0=x)
    inner_delta=float(objective.linear.U.unpack((inner_reference-x)[:-3]).square().mean().sqrt())
    color_delta=template_delta=None
    if tighter is not None:
        _,_,_,Ct,xt,_=tighter
        Tc,_=p.color_temperature(p.op.object_blur(C));Tr,_=p.color_temperature(p.op.object_blur(Ct))
        color_delta=float((Tc-Tr).square().mean().sqrt())
        template_delta=float(objective.linear.U.unpack((x-xt)[:-3]).square().mean().sqrt())
    def fit_info(fit):
        return None if fit is None else {'success':bool(fit.success),'message':str(fit.message),
            'iterations':int(fit.nit),'objective':float(fit.fun)}
    certified=bool(base is not None and base.success and reference is not None and reference.success
        and tighter is not None and inner['numerical_certified'] and inner_info['numerical_certified']
        and inner_delta<=opt['numeric_template_budget_e'] and template_delta<=opt['numeric_template_budget_e']
        and color_delta<=opt['numeric_color_budget_K'] and not errors)
    info={'status':'SUCCESS' if certified else 'NUMERICAL_UNCERTIFIED','numerical_certified':certified,
        'numerical_strategy':opt['strategy'],'initializer':initializer,'options':opt,
        'base_outer':fit_info(base),'tighter_outer':fit_info(reference),'errors':errors,
        'inner':inner,'tighter_inner_reference':inner_info,'template_inner_difference_RMS_e':inner_delta,
        'template_outer_difference_RMS_e':template_delta,'color_outer_difference_RMS_K':color_delta,
        'outer_gradient_norm':float(np.linalg.norm(gradient)),
        'outer_history':objective.history,'coordinate_scaling_fixed_at_initialization':True,
        'regularizer_frozen_during_tighter_outer':True,'truth_used':False,
        'wall_s':time.perf_counter()-started,'fixed_weight':True,'raw_objective_count':1}
    result=m.output(p,raw_e,C,objective.linear.U.unpack(x[:-3])[None],x[-3:][None],info)
    finite=all(bool(torch.isfinite(result[k]).all()) for k in
        ('temperature_color_K','template_postblur_e_ref','predicted_raw_e'))
    nonnegative=bool(result['predicted_raw_e'].min()>=0)
    info.update(finite_output=finite,nonnegative_predicted_measurement=nonnegative)
    if not (finite and nonnegative):
        info.update(status='NUMERICAL_UNCERTIFIED',numerical_certified=False)
        result['numerical_valid']=False
    result.update(z_grid=z,xi=x[:-3],continuum_latent=C)
    return result
