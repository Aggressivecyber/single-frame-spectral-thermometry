"""Accurate fixed-objective inverse: promoted frozen coefficients, no retuning."""
from __future__ import annotations
from . import storage as s
from .. import common as c
import copy
import time
import torch

RTOL=1e-8
MAXITER=200


def promote(physics):
    p=copy.deepcopy(physics)
    p._apply(lambda b:b.to(dtype=torch.complex128 if b.is_complex() else torch.float64)
             if b.is_floating_point() or b.is_complex() else b)
    return p


@torch.no_grad()
def build_system(physics32,raw32,continuum32,options,*,fixed_beta=None):
    f=c.frozen_v5('signed_linear_solver')
    primary=f.create_linear_system(physics32,raw32,continuum32,options,fixed_beta=fixed_beta)
    p=promote(physics32);linear=f.SignedLinearOperator(p)
    for name in ('template_pattern','beta_images','stationary_normal_symbol','laplacian_symbol'):
        setattr(linear,name,getattr(primary['linear_operator'],name).double())
    linear.template_diagonal=lambda W,mode:(primary['diag_S'].double(),
        {**primary['diagonal_info'],'promoted_frozen_primary_diagonal':True})
    raw=primary['raw_e'].double();C=torch.as_tensor(continuum32,device=raw.device).reshape(1,7,*raw.shape).double()
    C_raw=p.op(C)[0];W=primary['weights'].double()
    beta=None if fixed_beta is None else torch.as_tensor(fixed_beta,device=raw.device).reshape(3).double()
    cfg={**primary['options'],'rtol':RTOL,'max_iterations':MAXITER,'weights':W}
    system=f._create_single_frame_system(p,raw,C_raw,W,linear,cfg,beta)
    for name in ('lambda_g','lambda0','lambda_beta'):system[name]=primary[name].double()
    system['residual_scale']=primary['residual_scale'].double()
    def regularizer(x):
        field,b=system['unpack'](x)
        return system['pack'](system['lambda_g']*f.laplacian(field)+system['lambda0']*field,
                             None if b is None else system['lambda_beta']*b)
    system['regularizer']=regularizer
    system['matvec']=lambda x:system['measurement_matvec'](x)+regularizer(x)
    denominator=W.mean()*linear.stationary_normal_symbol+system['lambda_g']*linear.laplacian_symbol+system['lambda0']
    columns=linear.beta_images.flatten(1);gram=(columns*W.flatten()[None])@columns.T
    chol=torch.linalg.cholesky(gram+system['lambda_beta']*torch.eye(3,dtype=W.dtype,device=W.device))
    def precondition(x):
        field,b=system['unpack'](x)
        return system['pack'](torch.fft.ifft2(torch.fft.fft2(field)/denominator).real,
              None if b is None else torch.cholesky_solve(b[:,None],chol).squeeze(-1))
    system['precondition']=precondition
    def objective(field,b):
        residual=raw-C_raw-linear.apply_template(field)-linear.apply_beta(b)
        terms={'data':.5*(W*residual.square()).sum(),
         'gradient_penalty':.5*system['lambda_g']*((field-torch.roll(field,1,-1)).square()+
                                                  (field-torch.roll(field,1,-2)).square()).sum(),
         'ridge_S':.5*system['lambda0']*field.square().sum(),
         'ridge_beta':.5*system['lambda_beta']*b.square().sum()}
        return sum(terms.values()),{k:float(v) for k,v in terms.items()}
    system['objective']=objective
    system['physics']=p
    system['precision_contract']={'calibration_buffers_promoted_exactly':True,
       'original_G_coefficients_and_W_and_lambda_promoted_exactly':True,
       'RHS_arithmetic':'coherent float64 G, continuum and regularizer actions',
       'rtol':RTOL,'maximum_iterations':MAXITER,'alpha_g':1.0,'no_OTF_division':True}
    return system


def solve_rhs(system,rhs,role):
    f=c.frozen_v5('signed_linear_solver');t=time.perf_counter()
    x,info=f.pcg_true_residual(system['matvec'],rhs,system['precondition'],
         residual_scale=system['residual_scale'],rtol=RTOL,max_iterations=MAXITER,
         refresh_interval=25)
    info.update(role=role,wall_s=time.perf_counter()-t,numeric_dtype=str(rhs.dtype))
    return x,info


@torch.no_grad()
def solve(system,continuum32):
    p=system['physics'];linear=system['linear_operator'];raw=system['raw_e'];W=system['weights']
    C=torch.as_tensor(continuum32,device=raw.device).reshape(1,7,*raw.shape).double();Craw=p.op(C)[0]
    target=raw-Craw
    if system['fixed_beta'] is not None:target=target-linear.apply_beta(system['fixed_beta'])
    rhs=system['measurement_adjoint'](W*target)
    x,cg=solve_rhs(system,rhs,'actual_estimate')
    field,b=system['unpack'](x);beta=system['fixed_beta'] if b is None else b
    # Evaluate the actual source continuum, not the Ctrue factory initializer.
    residual=raw-Craw-linear.apply_template(field)-linear.apply_beta(beta)
    zero_beta=system['fixed_beta'] if b is None else torch.zeros_like(beta)
    before_residual=raw-Craw-linear.apply_beta(zero_beta)
    before=.5*(W*before_residual.square()).sum()+.5*system['lambda_beta']*zero_beta.square().sum()
    gradient=((field-torch.roll(field,1,-1)).square()+(field-torch.roll(field,1,-2)).square()).sum()
    terms={'data':float(.5*(W*residual.square()).sum()),'gradient_penalty':float(.5*system['lambda_g']*gradient),
      'ridge_S':float(.5*system['lambda0']*field.square().sum()),'ridge_beta':float(.5*system['lambda_beta']*beta.square().sum()),
      'gradient_energy_S_e2':float(gradient)}
    after=sum(v for k,v in terms.items() if k!='gradient_energy_S_e2')
    accepted=bool(torch.isfinite(x).all() and after<=float(before)+32*torch.finfo(W.dtype).eps*max(1.,abs(float(before))))
    if not accepted:
        field=torch.zeros_like(field);beta=zero_beta
        x=system['pack'](field,None if b is None else beta)
    prediction=Craw+linear.apply_template(field)+linear.apply_beta(beta)
    frame={'CG':cg,'accepted':accepted,'expected_e_nonnegative':bool((prediction>=0).all()),
       'objective_before':float(before),'objective_after':after,'objective_terms_after':terms,
       **{k:float(system[k]) for k in ('lambda_g','lambda0','lambda_beta')},
       'fixed_beta_diagnostic_oracle':b is None,'precision_contract':system['precision_contract']}
    return {'template_latent_e':field[None],'template_postblur_e':p.op.object_blur(field[None]),
       'beta':beta[None],'expected_e':prediction[None],'weights':W[None],'continuum_raw_e':Craw[None],
       'info':{'frames':[frame],'options':{k:v for k,v in system['options'].items() if k!='weights'}},
       'vector':x,'rhs':rhs}
