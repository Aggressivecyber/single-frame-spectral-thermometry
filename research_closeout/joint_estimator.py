"""One all-band physical profile estimator; signed templates stay in observable coordinates."""
from . import storage as s
from . import measurement as m
from .linear_profile import LinearProfile
import time
import numpy as np
import torch
from scipy.optimize import minimize

DEFAULTS={'grid':128,'continuum_multiplier':1.,'template_multiplier':1.,
    'outer_maxiter':30,'outer_maxfun':90,'outer_ftol':1e-10,'outer_gtol':1e-3,
    'inner_tolerances':[1e-7,1e-9,1e-11],'inner_maxiter':1200,
    'reference_inner_tolerance':1e-12,'numeric_template_budget_e':.17,
    'numeric_color_budget_K':.2,'cost_limit_s':180.}

def initialize(p,raw_dn,kind='v3',state=None):
    raw=m.raw_tensor(raw_dn,p)
    if state is not None:
        if kind!='v3': raise ValueError('Cached state is only an audited v3 initializer')
        return state.to(device=p.M.device,dtype=p.M.dtype).detach()[:,:2]
    if kind=='v3':
        from training_v4.scalar_engine import invert_single
        selected=m.selection()
        p32=m.physics(p.M.device,dtype=torch.float32)
        z,_,_=invert_single(p32,p32.raw_to_electrons(raw.float()),selected['method'],selected['solver_config'])[:3]
        return z[:,:2].to(p.M.dtype)
    if kind!='smooth_demod': raise ValueError('Undeclared initializer')
    raw_e=p.raw_to_electrons(raw)
    decoded=p.op.decode(raw_e)
    decoded=m.extra_smooth(decoded,2.)
    T,_=p.color_temperature(decoded);C,_=p.continuum(T)
    a=(decoded[:,-1]/C[:,-1].clamp_min(1)).clamp(.03,1.3)
    return torch.stack([(T-p.T0)/p.Tscale,a.log()],1)

class ProfileObjective:
    def __init__(self,p,raw_e,z0,options):
        self.p=p;self.raw_e=raw_e;self.opt={**DEFAULTS,**options}
        self.g=int(self.opt['grid']);self.shape=(1,2,self.g,self.g)
        self.z0=torch.nn.functional.interpolate(z0,size=(self.g,self.g),mode='bilinear',align_corners=False)
        self.W=m.fixed_weights(p,raw_e)
        self.linear=LinearProfile(p,self.W,self.opt['template_multiplier'])
        with torch.no_grad():
            full=torch.nn.functional.interpolate(self.z0,size=p.cfg['shape'],mode='bilinear',align_corners=False)
            T=p.T0+p.Tscale*full[:,0];C,dC=p.continuum(T);a=full[:,1:2].exp()
            jac=torch.stack([a*dC*p.Tscale,a*C],1)
            scale=(jac.square()*p.op.pattern_energy[None,None,:,None,None]).sum(2).mean((0,2,3))*self.W.mean()
            self.regscale=.01*self.opt['continuum_multiplier']*scale
        self.calls=0;self.last_x=None;self.history=[]
        self.started=time.perf_counter()
    def regularizer(self,z):
        step=self.p.cfg['shape'][0]/self.g
        dx=(z[:,:,:,1:]-z[:,:,:,:-1])/step;dy=(z[:,:,1:]-z[:,:,:-1])/step
        return .5*(self.regscale*((dx.square().sum((0,2,3))+dy.square().sum((0,2,3))))).sum()
    def evaluate(self,flat,*,forced_tolerance=None):
        if time.perf_counter()-self.started>self.opt['cost_limit_s']:
            raise TimeoutError('Registered per-frame profile complexity budget exceeded')
        self.calls+=1
        z=torch.as_tensor(flat,device=self.p.M.device,dtype=self.p.M.dtype).reshape(self.shape).detach().requires_grad_(True)
        C=m.continuum(self.p,z,grid=self.g)
        rawC=self.p.op(C)
        tol=forced_tolerance or self.opt['inner_tolerances'][min(self.calls//8,2)]
        with torch.no_grad():
            x,info=self.linear.solve(self.raw_e-rawC.detach(),tolerance=tol,maxiter=self.opt['inner_maxiter'],x0=self.last_x)
        if not info['numerical_certified']: raise ArithmeticError('NUMERICAL_UNCERTIFIED inner profile: '+str(info))
        residual=rawC+self.linear.apply(x)[None]-self.raw_e
        phi=.5*(self.W*residual.square()).sum()+.5*(self.linear.reg*x).square().sum()+self.regularizer(z)
        grad=torch.autograd.grad(phi,z)[0]
        self.last_x=x.detach()
        self.history.append({'evaluation':self.calls,'objective':float(phi.detach()),'gradient_norm':float(grad.norm()),'inner':info})
        return float(phi.detach()),grad.detach().cpu().numpy().ravel(),z.detach(),C.detach(),x.detach(),info
    def fun(self,v):
        a=self.evaluate(v);return a[0],a[1]

def infer(raw_dn,p,options=None,*,initializer='v3',v3_state=None):
    """No labels, identities, geometry, conditions, or truth are accepted."""
    opt={**DEFAULTS,**(options or {})};started=time.perf_counter()
    raw=m.raw_tensor(raw_dn,p);raw_e=p.raw_to_electrons(raw)
    z0=initialize(p,raw,initializer,v3_state)
    objective=ProfileObjective(p,raw_e,z0,opt)
    g=objective.g;limits=p.cfg['temperature_range_K']
    bounds=[((limits[0]-p.T0)/p.Tscale,(limits[1]-p.T0)/p.Tscale)]*(g*g)+[(-4.,.3)]*(g*g)
    fit=minimize(objective.fun,objective.z0.cpu().numpy().ravel(),method='L-BFGS-B',jac=True,bounds=bounds,
        options={'maxiter':opt['outer_maxiter'],'maxfun':opt['outer_maxfun'],'ftol':opt['outer_ftol'],'gtol':opt['outer_gtol'],'maxls':12})
    _,grad,z,C,x,info=objective.evaluate(fit.x,forced_tolerance=opt['inner_tolerances'][-1])
    # Independent tighter inner solve at exactly the returned outer state.
    reference,reference_info=objective.linear.solve(raw_e-p.op(C),tolerance=opt['reference_inner_tolerance'],
        maxiter=2*opt['inner_maxiter'],x0=x)
    delta=objective.linear.U.unpack((reference-x)[:-3])
    numeric_e=float(delta.square().mean().sqrt())
    certified=bool(info['numerical_certified'] and reference_info['numerical_certified'] and numeric_e<=opt['numeric_template_budget_e'] and fit.success)
    info={'numerical_certified':certified,'status':'SUCCESS' if certified else 'NUMERICAL_UNCERTIFIED',
        'inner':info,'tighter_inner_reference':reference_info,'template_inner_difference_RMS_e':numeric_e,
        'outer_success':bool(fit.success),'outer_message':str(fit.message),'outer_iterations':int(fit.nit),
        'outer_gradient_norm':float(np.linalg.norm(grad)),'initializer':initializer,'options':opt,
        'outer_history':objective.history,'truth_used':False,'fixed_weight':True,'raw_objective_count':1,
        'wall_s':time.perf_counter()-started,'color_numeric_budget_K':opt['numeric_color_budget_K'],
        'color_numeric_budget_scope':'tighter-outer comparison is additionally required before final method certification'}
    result=m.output(p,raw_e,C,objective.linear.U.unpack(x[:-3])[None],x[-3:][None],info)
    result.update(z_grid=z,xi=x[:-3],continuum_latent=C)
    return result
