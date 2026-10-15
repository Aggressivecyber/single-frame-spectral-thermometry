"""Historical numerical recovery and the explicitly versioned v2 repair.

``recovered_infer`` retains the old initializer-scaled objective and its original
two-pass protocol.  New repair runs must call ``repaired_infer`` explicitly.
"""
from . import storage as s
from . import measurement as m
from .joint_estimator import (ProfileObjective,SharedProfileObjective,initialize,
    DEFAULTS,SHARED_OBJECTIVE_VERSION,SHARED_PHYSICAL_DEFAULTS)
import copy
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


REPAIRED_SOLVER_VERSION='joint-profile-numerics-v2-two-output-plateaus'
REPAIRED_NUMERICS={
    'cost_limit_s':180.,
    'inner_tolerance':1e-11,
    'reference_inner_tolerance':1e-12,
    'inner_maxiter':1200,
    'numeric_template_budget_e':.17,
    'numeric_color_budget_K':.2,
    'required_stable_comparisons':2,
    'maxls':20,
    'maxcor':20,
    'retain_evaluation_history':False,
    'stages':[
        {'maxiter':280,'maxfun':700,'ftol':1e-12,'gtol':1e-4},
        {'maxiter':200,'maxfun':500,'ftol':1e-13,'gtol':1e-5},
        {'maxiter':160,'maxfun':400,'ftol':1e-14,'gtol':1e-6},
        {'maxiter':160,'maxfun':400,'ftol':2e-15,'gtol':1e-7},
    ],
}


class RepairedObjective(SharedProfileObjective):
    """Shared physical target with separately defined numerical coordinates."""
    def __init__(self,*args,**kwargs):
        kwargs['numerical_options']=_repair_options(kwargs.get('numerical_options'))
        super().__init__(*args,**kwargs)
        # The coordinate metric uses the calibration anchor sensitivity,
        # never the physical regularization multipliers or the initial state.
        channel_scale=self.reference_diagonal.clamp_min(1e-10).rsqrt().clamp(1e-5,1e3)
        self.coordinate_scale=channel_scale[None,:,None,None].expand(self.shape).cpu().numpy().ravel()
        self.coordinate_contract={
            'version':'shared-calibration-anchor-diagonal-v2',
            'channel_scale':channel_scale.detach().cpu().tolist(),
            'depends_on_initializer':False,
            'depends_on_regularization_multipliers':False,
            'changes_physical_objective':False,
        }

    def scaled_fun(self,v):
        value,gradient,*_=self.evaluate(v*self.coordinate_scale,
            forced_tolerance=self.opt['inner_tolerance'])
        return value,gradient*self.coordinate_scale


def _repair_options(options):
    values=copy.deepcopy(REPAIRED_NUMERICS)
    values.update(copy.deepcopy(options or {}))
    unknown=set(values)-set(REPAIRED_NUMERICS)
    if unknown: raise ValueError('Unknown v2 numerical options: '+str(sorted(unknown)))
    if not np.isfinite(values['cost_limit_s']) or not 0<values['cost_limit_s']<=180:
        raise ValueError('v2 inference requires a positive total budget no larger than 180 s')
    if values['required_stable_comparisons']!=2:
        raise ValueError('v2 certification requires two consecutive tighter-output comparisons')
    for name,maximum in (('numeric_template_budget_e',.17),('numeric_color_budget_K',.2)):
        if not np.isfinite(values[name]) or not 0<values[name]<=maximum:
            raise ValueError('The registered output budget cannot be relaxed: '+name)
    for name in ('inner_tolerance','reference_inner_tolerance'):
        if not np.isfinite(values[name]) or not 0<values[name]<=REPAIRED_NUMERICS[name]:
            raise ValueError('The registered inner tolerance cannot be relaxed: '+name)
    if values['reference_inner_tolerance']>=values['inner_tolerance']:
        raise ValueError('The independent inner reference must be tighter')
    for name in ('inner_maxiter','maxls','maxcor'):
        if not isinstance(values[name],int) or values[name]<1:
            raise ValueError('Invalid positive integer numerical option: '+name)
    stages=values['stages']
    if not isinstance(stages,(list,tuple)) or not 3<=len(stages)<=4:
        raise ValueError('v2 requires three or four finite continuation stages')
    for i,stage in enumerate(stages):
        if set(stage)!={'maxiter','maxfun','ftol','gtol'}:
            raise ValueError('Each stage must specify maxiter, maxfun, ftol and gtol')
        for name in ('maxiter','maxfun'):
            if not isinstance(stage[name],int) or stage[name]<1:
                raise ValueError('Invalid stage iteration/evaluation budget')
        for name in ('ftol','gtol'):
            if not np.isfinite(stage[name]) or stage[name]<=0:
                raise ValueError('Stage tolerances must be finite and positive')
            if stage[name]>REPAIRED_NUMERICS['stages'][i][name]:
                raise ValueError('The registered stage tolerance cannot be relaxed: '+name)
            if i and stage[name]>=stages[i-1][name]:
                raise ValueError('Every continuation stage must strictly tighten ftol and gtol')
    return values


def _gradient_record(point,scale,lower,upper):
    gradient=point[1]
    if gradient is None:
        return {'finite':False,'physical_l2':None,'physical_linf':None,
            'scaled_projected_linf':None}
    flat=point[2].detach().cpu().numpy().ravel()
    v=flat/scale;g=gradient*scale
    projected=v-np.clip(v-g,lower,upper)
    finite=bool(np.isfinite(gradient).all() and np.isfinite(projected).all())
    return {'finite':finite,'physical_l2':float(np.linalg.norm(gradient)) if finite else None,
        'physical_linf':float(np.max(np.abs(gradient))) if finite else None,
        'scaled_projected_linf':float(np.max(np.abs(projected))) if finite else None}


def _point_output(p,raw_e,objective,point):
    _,_,_,C,x,inner=point
    out=m.output(p,raw_e,C,objective.linear.U.unpack(x[:-3])[None],
        x[-3:][None],{'numerical_certified':False})
    finite=all(bool(torch.isfinite(out[key]).all()) for key in
        ('temperature_color_K','template_postblur_e_ref','predicted_raw_e'))
    nonnegative=finite and bool(out['predicted_raw_e'].min()>=0)
    return out,finite,nonnegative


def repaired_infer(raw_dn,p,*,initializer='smooth_demod',v3_state=None,
                   physical_options=None,numerical_options=None):
    """v2 raw-only profile solve, returning its tightest certified endpoint.

    Certification requires three consecutive qualified endpoints and two
    consecutive output comparisons under strictly tighter configurations.
    Every endpoint has an independent tighter inner reference and a recomputed
    projected gradient.  An optimizer success flag alone cannot certify it.
    Timeouts retain a finite available state and always remain uncertified.
    Cached v3 initialization performs no extra inverse; call accounting belongs
    to the invoking controller.  No labels are accepted by this interface.
    """
    opt=_repair_options(numerical_options);started=time.perf_counter()
    raw=m.raw_tensor(raw_dn,p);raw_e=p.raw_to_electrons(raw)
    z0=initialize(p,raw,initializer,v3_state)
    objective=RepairedObjective(p,raw_e,z0,physical_options=physical_options,
        numerical_options=opt,deadline=started+opt['cost_limit_s'])
    scale=objective.coordinate_scale;g=objective.g;limits=p.cfg['temperature_range_K']
    lo=np.r_[np.full(g*g,(limits[0]-p.T0)/p.Tscale),np.full(g*g,-4.)]
    hi=np.r_[np.full(g*g,(limits[1]-p.T0)/p.Tscale),np.full(g*g,.3)]
    lower=lo/scale;upper=hi/scale;bounds=np.column_stack([lower,upper])
    current=np.clip(objective.z0.cpu().numpy().ravel(),lo,hi)/scale
    # A finite initializer is retained even if the deadline is reached before
    # the first LSMR call.  It is explicitly not a solved profile endpoint.
    fallback_z=torch.as_tensor(current*scale,device=p.M.device,dtype=p.M.dtype).reshape(objective.shape)
    fallback_C=m.continuum(p,fallback_z,grid=g)
    fallback_x=p.M.new_zeros(objective.linear.n)
    returned=(None,None,fallback_z,fallback_C,fallback_x,
        {'solver':'not executed','numerical_certified':False,'status':'INITIALIZER_ONLY'})
    _,fallback_finite,_=_point_output(p,raw_e,objective,returned)
    if not fallback_finite: raise ArithmeticError('No finite initialized v2 profile state')
    stages=[];comparisons=[];errors=[];stable=0;certified=False;timed_out=False
    previous=None;previous_output=None;returned_stage=None;returned_role='initializer_only'
    returned_inner_delta=None

    for index,config in enumerate(opt['stages']):
        stage_number=index+1;stage_started=time.perf_counter();accepted_point=None;accepted_iterations=0
        fit=None

        def accepted(v):
            nonlocal accepted_point,accepted_iterations
            accepted_iterations+=1
            flat=np.asarray(v)*scale
            if objective.last_flat is not None and np.array_equal(flat,objective.last_flat):
                accepted_point=objective.last_result

        try:
            objective.check_deadline()
            fit=minimize(objective.scaled_fun,current,method='L-BFGS-B',jac=True,
                bounds=bounds,callback=accepted,
                options={**config,'maxls':opt['maxls'],'maxcor':opt['maxcor']})
            point=objective.evaluate(fit.x*scale,forced_tolerance=opt['inner_tolerance'])
            objective.check_deadline()
            _,_,z,C,x,inner=point
            reference,reference_info=objective.linear.solve(raw_e-p.op(C),
                tolerance=opt['reference_inner_tolerance'],maxiter=2*opt['inner_maxiter'],x0=x)
            inner_delta=float(objective.linear.U.unpack((reference-x)[:-3]).square().mean().sqrt())
            # Use the tighter linear result, and recompute the actual gradient
            # of precisely the state which is compared and eventually returned.
            endpoint=objective.evaluate_linear_solution(fit.x*scale,reference,reference_info)
            output,finite,nonnegative=_point_output(p,raw_e,objective,endpoint)
            gradient=_gradient_record(endpoint,scale,lower,upper)
            stationary=bool(gradient['finite'] and gradient['scaled_projected_linf']
                <=config['gtol'])
            qualified=bool(fit.success and stationary and inner['numerical_certified']
                and reference_info['numerical_certified'] and inner_delta<=opt['numeric_template_budget_e']
                and finite and nonnegative)
            record={'stage':stage_number,'complete':True,'configuration':dict(config),
                'optimizer':{'success':bool(fit.success),'message':str(fit.message),
                    'iterations':int(fit.nit),'function_evaluations':int(fit.nfev),
                    'objective':float(fit.fun)},
                'returned_endpoint_objective':endpoint[0],'gradient':gradient,
                'requested_projected_gradient_passed':stationary,'inner':inner,
                'tighter_inner_reference':reference_info,'template_inner_difference_RMS_e':inner_delta,
                'finite_output':finite,'nonnegative_predicted_measurement':nonnegative,
                'qualified_endpoint':qualified,'wall_s':time.perf_counter()-stage_started}
            stages.append(record)
            if finite:
                returned=endpoint;returned_stage=stage_number
                returned_role='completed_tightened_endpoint';returned_inner_delta=inner_delta
            if previous is not None:
                du=float((output['template_postblur_e_ref']-previous_output['template_postblur_e_ref']).square().mean().sqrt())
                dt=float((output['temperature_color_K']-previous_output['temperature_color_K']).square().mean().sqrt())
                within=bool(finite and np.isfinite(du) and np.isfinite(dt)
                    and du<=opt['numeric_template_budget_e'] and dt<=opt['numeric_color_budget_K'])
                passed=bool(within and qualified and previous['qualified_endpoint'])
                stable=stable+1 if passed else 0
                comparisons.append({'from_stage':previous['stage'],'to_stage':stage_number,
                    'template_RMS_difference_e':du,'color_RMS_difference_K':dt,
                    'within_output_budgets':within,'both_endpoints_qualified':bool(qualified and previous['qualified_endpoint']),
                    'passed':passed,'consecutive_passed_comparisons':stable})
            objective.check_deadline()
            if stable>=opt['required_stable_comparisons']:
                certified=True;returned_role='tightest_certified_endpoint'
                break
            previous=record;previous_output=output
            current=np.asarray(fit.x).copy()
        except (TimeoutError,ArithmeticError) as error:
            timed_out=isinstance(error,TimeoutError)
            errors.append({'stage':stage_number,'type':type(error).__name__,'message':str(error)})
            # Prefer the last accepted iterate of the interrupted stage.  If
            # no accepted iterate exists, keep the previous completed endpoint.
            # The very first finite evaluation can still improve on x=0.
            partial=accepted_point
            if partial is None and returned_stage is None: partial=objective.last_result
            if partial is not None:
                _,finite,_=_point_output(p,raw_e,objective,partial)
                if finite:
                    returned=partial;returned_stage=stage_number
                    returned_role='interrupted_stage_finite_iterate';returned_inner_delta=None
            if not stages or stages[-1]['stage']!=stage_number:
                stages.append({'stage':stage_number,'complete':False,'configuration':dict(config),
                    'optimizer':{'success':False,'message':str(error),'iterations':accepted_iterations,
                        'function_evaluations':None,'objective':returned[0]},
                    'gradient':_gradient_record(returned,scale,lower,upper),
                    'qualified_endpoint':False,'wall_s':time.perf_counter()-stage_started})
            certified=False
            break

    gradient=_gradient_record(returned,scale,lower,upper)
    last_comparison=comparisons[-1] if comparisons else None
    output,finite,nonnegative=_point_output(p,raw_e,objective,returned)
    certified=bool(certified and not errors and not timed_out and finite and nonnegative)
    reasons=[]
    if not certified:
        if timed_out: reasons.append('total_time_budget_exhausted')
        if errors and not timed_out: reasons.append('numerical_evaluation_failed')
        if stable<2: reasons.append('two_consecutive_qualified_output_plateaus_not_observed')
        if not finite or not nonnegative: reasons.append('finite_nonnegative_output_not_verified')
    history=objective.history
    summary=None if not history else {'evaluations':len(history),'first':history[0],
        'last':history[-1],'minimum_recorded_objective':min(item['objective'] for item in history)}
    info={
        'method':'joint_profile_repaired_v2','physical_objective_version':SHARED_OBJECTIVE_VERSION,
        'numerical_solver_version':REPAIRED_SOLVER_VERSION,
        'status':'NUMERICAL_CERTIFIED_V2' if certified else 'NUMERICAL_UNCERTIFIED_V2',
        'numerical_certified':certified,'failure_reasons':reasons,'errors':errors,'timed_out':timed_out,
        'initializer':initializer,'initializer_source':'provided_cached_v3_state' if v3_state is not None
            else ('computed_v3_inverse' if initializer=='v3' else 'raw_smooth_demod'),
        'cached_initializer_performed_new_inverse':False if v3_state is not None else None,
        'physical_objective':objective.physical_contract,'numerical_coordinates':objective.coordinate_contract,
        'numerical_options':opt,'stages':stages,'output_comparisons':comparisons,
        'required_stable_comparisons':2,'observed_consecutive_stable_comparisons':stable,
        'returned_stage':returned_stage,'returned_result_role':returned_role,
        'returned_endpoint_objective':returned[0],'returned_gradient':gradient,
        'outer_gradient_norm':gradient['physical_l2'],'inner':returned[5],
        'template_inner_difference_RMS_e':returned_inner_delta,
        'template_outer_difference_RMS_e':None if last_comparison is None else last_comparison['template_RMS_difference_e'],
        'color_outer_difference_RMS_K':None if last_comparison is None else last_comparison['color_RMS_difference_K'],
        'outer_history_summary':summary,'outer_history':history if opt['retain_evaluation_history'] else [],
        'objective_evaluations':objective.calls,'finite_output':finite,
        'nonnegative_predicted_measurement':nonnegative,'truth_used':False,'fixed_weight':True,
        'raw_objective_count':1,'regularizer_frozen_during_all_stages':True,
        'certification_scope':'two successive tighter-output plateaus and endpoint checks; not a global-optimum proof',
        'historical_results_reinterpreted':False,'wall_s':time.perf_counter()-started,
    }
    output.update(info=info,numerical_valid=certified,z_grid=returned[2],xi=returned[4][:-3],
        continuum_latent=returned[3])
    return output
