"""Conditionally linear postblur-template/background problem as augmented least squares."""
from . import storage as s
from .observable_basis import RealFourierBasis
import time
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator,lsmr

class LinearProfile:
    def __init__(self,p,weights,template_multiplier=1.,*,basis=None):
        self.p=p;self.weights=weights.reshape(p.cfg['shape']);self.sqrtW=self.weights.sqrt()
        self.U=basis or RealFourierBasis(p.op.object_otf)
        self.n=self.U.dimension+3;self.pixels=int(np.prod(p.cfg['shape']))
        self.pattern=(p.op.patterns*p.M[:,2,None,None]).sum(0)
        self.beta_images=torch.stack([p.op.encode_images(p.q[:,None,None]*p.op.object_blur(p.Bscale*v)[None]) for v in p.background_basis])
        # New postblur gradient regularizer, in physical electrons. Fixed units,
        # scaled by the mean whitened template sensitivity, not a legacy alpha.
        sensitivity=float(self.weights.mean()*self.pattern.square().mean()*p.op.relay_otf.square().mean())
        lap=self.U.laplacian; positive=lap[lap>0]
        lap_scale=float(positive.median()) if len(positive) else 1.
        template_penalty=template_multiplier*sensitivity*(1e-6+.01*lap/max(lap_scale,1e-30))
        beta_diag=(self.beta_images.square()*self.weights).sum((-2,-1))
        self.reg=torch.cat([template_penalty,1e-6*beta_diag]).sqrt()
        self.column_scale=torch.cat([torch.full((self.U.dimension,),sensitivity,device=p.M.device,dtype=p.M.dtype),beta_diag]).add(self.reg.square()).clamp_min(1e-30).rsqrt()
        self.contract={'postblur_prior_is_new_estimator':True,'template_multiplier':template_multiplier,
            'template_reference_sensitivity':sensitivity,'gradient_reference_laplacian':lap_scale,
            'template_penalty':'multiplier*sensitivity*(1e-6+0.01*lap/lap_median)',
            'beta_penalty':'1e-6 times whitened beta column squared norm',
            'columns_scaled':True,'raw_objective_count':1,'template_object_blur_count':0}
    def apply(self,x):
        u=self.U.unpack(x[:-3]);return self.p.op.encode_images(self.p.M[:,2,None,None]*u[None])+torch.einsum('l,lhw->hw',x[-3:],self.beta_images)
    def adjoint(self,y):
        relay=torch.fft.ifft2(torch.fft.fft2(y)*self.p.op.relay_otf).real
        return torch.cat([self.U.pack(self.pattern*relay),torch.einsum('hw,lhw->l',y,self.beta_images)])
    def aug(self,v):
        x=v*self.column_scale
        return torch.cat([(self.sqrtW*self.apply(x)).flatten(),self.reg*x])
    def aug_adjoint(self,r):
        return self.column_scale*(self.adjoint(self.sqrtW*r[:self.pixels].reshape(self.p.cfg['shape']))+self.reg*r[self.pixels:])
    def operator(self):
        def mat(v): return self.aug(torch.as_tensor(v,device=self.p.M.device,dtype=self.p.M.dtype)).detach().cpu().numpy()
        def adj(v): return self.aug_adjoint(torch.as_tensor(v,device=self.p.M.device,dtype=self.p.M.dtype)).detach().cpu().numpy()
        return LinearOperator((self.pixels+self.n,self.n),matvec=mat,rmatvec=adj,dtype=np.float64)
    def solve(self,target,*,tolerance=1e-9,maxiter=1200,x0=None):
        start=time.perf_counter();A=self.operator()
        b=torch.cat([(self.sqrtW*target.reshape(self.p.cfg['shape'])).flatten(),self.reg.new_zeros(self.n)]).cpu().numpy()
        initial=None if x0 is None else (x0/self.column_scale).detach().cpu().numpy()
        result=lsmr(A,b,atol=tolerance,btol=tolerance,conlim=1e10,maxiter=maxiter,x0=initial)
        x=torch.as_tensor(result[0],device=self.p.M.device,dtype=self.p.M.dtype)*self.column_scale
        r=torch.as_tensor(b,device=self.p.M.device,dtype=self.p.M.dtype)-self.aug(x/self.column_scale)
        normal=self.aug_adjoint(r)
        normal_rel=float(normal.norm()/max(float(self.aug_adjoint(torch.as_tensor(b,device=self.p.M.device,dtype=self.p.M.dtype)).norm()),1e-30))
        accepted=result[1] in (0,1,2,4,5) and bool(torch.isfinite(x).all()) and normal_rel<=max(1e-8,100*tolerance)
        info={'solver':'column-scaled augmented LSMR','istop':int(result[1]),'iterations':int(result[2]),
            'norm_residual':float(result[3]),'norm_normal_residual':float(result[4]),'condition_estimate':float(result[6]),
            'actual_scaled_normal_relative':normal_rel,'tolerance':tolerance,'maxiter':maxiter,
            'numerical_certified':accepted,'wall_s':time.perf_counter()-start,'regularization':self.contract}
        return x,info
    def factor_reference(self,target):
        # Full observable dimension (~1156) is bounded; this method is only
        # permitted on explicitly selected small-mode subspaces in tests.
        if self.n>80: raise ValueError('Reference must use a declared finite-mode subset')
        eye=torch.eye(self.n,dtype=self.p.M.dtype,device=self.p.M.device)
        A=torch.stack([self.aug(v) for v in eye],1)
        b=torch.cat([(self.sqrtW*target.reshape(self.p.cfg['shape'])).flatten(),self.reg.new_zeros(self.n)])
        q,r=torch.linalg.qr(A,mode='reduced');v=torch.linalg.solve_triangular(r,(q.T@b)[:,None],upper=True)[:,0]
        return v*self.column_scale,A,q,r
