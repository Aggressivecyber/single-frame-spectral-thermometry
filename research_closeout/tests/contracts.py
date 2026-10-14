"""Native-grid finite-mode QR reference, LSMR, envelope gradient and shared-noise checks."""
from .. import storage as s
from .. import measurement as m
from ..observable_basis import RealFourierBasis
from ..linear_profile import LinearProfile
from ..error_accounting import same_factor_accounting
from ..joint_estimator import ProfileObjective
import time
import numpy as np
import torch

class FiniteBasis:
    def __init__(self,U,indices):
        self.U=U;self.indices=indices;self.dimension=len(indices);self.shape=U.shape
        self.laplacian=U.laplacian[indices];self.otf_values=U.otf_values[indices]
    def unpack(self,x):
        full=x.new_zeros(self.U.dimension);full[self.indices]=x
        return self.U.unpack(full)
    def pack(self,image): return self.U.pack(image)[self.indices]

def run():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(406014)
    p=m.physics('cpu');U=RealFourierBasis(p.op.object_otf)
    tests=[]
    def record(name,value,limit):
        tests.append({'check':name,'error':float(value),'limit':limit,'passed':bool(value<=limit)})
    xi=torch.randn(U.dimension,dtype=torch.float64);image=U.unpack(xi)
    record('native256 U_transpose_U',float((U.pack(image)-xi).abs().max()),1e-11)
    target=torch.randn(p.cfg['shape'],dtype=torch.float64)
    independent=torch.fft.ifft2(torch.fft.fft2(target)*U.mask).real
    record('native256 UU_transpose_support',float((U.project(target)-independent).abs().max()),1e-11)
    record('native256 norm_and_real',abs(float(image.norm()-xi.norm())),1e-11)
    indices=torch.linspace(0,U.dimension-1,25).long()
    finite=FiniteBasis(U,indices)
    W=(1000+100*torch.sin(torch.arange(256,dtype=torch.float64)[:,None]/20)+torch.zeros((256,256))).reciprocal()
    system=LinearProfile(p,W,basis=finite)
    x=torch.randn(system.n,dtype=torch.float64);r=torch.randn((256,256),dtype=torch.float64)
    record('native256 physical_forward_adjoint',abs(float((system.apply(x)*r).sum()-x@system.adjoint(r)))/max(abs(float(x@system.adjoint(r))),1),1e-11)
    u=finite.unpack(x[:-3]);nonzero=U.mask
    # Pure coordinate equivalence is an explicit reference test. No inverse
    # OTF is present in the production estimator.
    fft=torch.fft.fft2(u);latent_fft=torch.zeros_like(fft);latent_fft[nonzero]=fft[nonzero]/p.op.object_otf[nonzero]
    latent=torch.fft.ifft2(latent_fft).real
    old=p.op(p.M[:,2,None,None]*latent[None])
    new=p.op.encode_images(p.M[:,2,None,None]*u[None])
    record('latent_to_postblur_single_H_equivalence',float((old-new).abs().max()),1e-9)
    b=torch.randn((256,256),dtype=torch.float64)
    reference,A,q,R=system.factor_reference(b)
    solved,info=system.solve(b,tolerance=1e-12,maxiter=400)
    record('native256 augmented_LSMR_vs_same_QR',float((finite.unpack((solved-reference)[:-3])).square().mean().sqrt()),1e-7)
    xt=torch.randn(system.n,dtype=torch.float64)
    cleak=torch.randn((256,256),dtype=torch.float64)*.1;noise=torch.randn_like(cleak)*2
    datarhs=torch.cat([(system.sqrtW*(system.apply(xt)+cleak+noise)).flatten(),xt.new_zeros(system.n)])
    leak=system.adjoint(W*cleak);nrhs=system.adjoint(W*noise);bias=-system.reg.square()*xt
    # Deliberately perturb the actual returned solution: the omitted fourth
    # term must become visible, rather than being called physical exhaustion.
    exact=reference.new_zeros(system.n)
    exact=torch.linalg.solve_triangular(R,(q.T@datarhs)[:,None],upper=True)[:,0]*system.column_scale
    actual=exact+torch.randn_like(exact)*.03
    # Accounting is performed in unscaled original physical coordinates.
    Aphysical=A/system.column_scale[None]
    acc=same_factor_accounting(Aphysical,datarhs,bias,xt,leak,nrhs,actual,
        [torch.linalg.solve(Aphysical.T@Aphysical,v)+torch.randn_like(v)*.002 for v in (leak,nrhs,bias)])
    record('four_term_same_factor_native256',acc['four_term_closure_RMS'],1e-9)
    record('component_residual_corrections_native256',acc['approximate_component_closure_RMS'],1e-9)
    # Envelope-gradient test with a separable nonlinear continuum. The
    # reference factor is exactly the same for each perturbed nonlinear state.
    rng=np.random.default_rng(406015);D=rng.normal(size=(80,9));C=rng.normal(size=(80,3))
    y=rng.normal(size=80);lam=.03;Az=np.vstack([D,np.sqrt(lam)*np.eye(9)])
    Q,Rz=np.linalg.qr(Az);z=rng.normal(size=3)*.1;direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    def profile(z):
        f=C@np.exp(z);rhs=np.r_[y-f,np.zeros(9)];x=np.linalg.solve(Rz,Q.T@rhs)
        rr=f+D@x-y
        return .5*(rr@rr+lam*x@x),(C*np.exp(z)[None]).T@rr
    eps=1e-5;fd=(profile(z+eps*direction)[0]-profile(z-eps*direction)[0])/(2*eps)
    analytical=profile(z)[1]@direction
    record('profile_envelope_gradient_directional',abs(fd-analytical)/max(abs(fd),1),1e-7)
    # Exercise the actual production objective on the original physical grid,
    # with a bounded linear subspace and coarse nonlinear representation.
    z0=torch.zeros((1,2,4,4),dtype=torch.float64);z0[:,0]=.0025;z0[:,1]=-.4
    C0=m.continuum(p,z0,grid=4);raw0=p.op(C0)+system.apply(torch.zeros(system.n,dtype=torch.float64))[None]+.03
    production=ProfileObjective(p,raw0,z0,{'grid':4,'cost_limit_s':60.})
    production.linear=LinearProfile(p,production.W,basis=finite)
    flat=z0.numpy().ravel();d=rng.normal(size=len(flat));d/=np.linalg.norm(d)
    step=3e-6
    center=production.evaluate(flat,forced_tolerance=1e-12)
    plus=production.evaluate(flat+step*d,forced_tolerance=1e-12)[0]
    minus=production.evaluate(flat-step*d,forced_tolerance=1e-12)[0]
    fd_native=(plus-minus)/(2*step);ad_native=center[1]@d
    record('native256_production_profile_gradient',abs(fd_native-ad_native)/max(abs(fd_native),1),1e-6)
    # Shared raw covariance: estimator and residual measurement are correlated.
    B=rng.normal(size=(4,6))*.1;J=rng.normal(size=(6,4))*.1;L=rng.normal(size=(3,6));variance=np.arange(1,7)
    transform=L@(np.eye(6)-J@B);cov=transform@np.diag(variance)@transform.T
    samples=rng.normal(size=(40000,6))*np.sqrt(variance);errors=samples@transform.T
    empirical=np.cov(errors,rowvar=False)
    record('shared_raw_covariance_40000_MC',np.linalg.norm(empirical-cov)/np.linalg.norm(cov),.04)
    receipt={'schema':'closeout-numeric-contracts-001','time':s.now(),'all_passed':all(t['passed'] for t in tests),
        'native_grid':[256,256],'finite_reference_real_template_modes':25,'unknowns_in_QR':system.n,
        'basis':U.report(),'tests':tests,'LSMR':info,'omitted_actual_numeric_term_RMS':acc['three_term_closure_RMS'],
        'four_term_corrected_RMS':acc['four_term_closure_RMS'],'wall_s':time.perf_counter()-start,
        'source_sha256':{str(v.relative_to(s.ROOT)).replace('\\','/'):s.sha(v) for v in (s.ROOT/'research_closeout').rglob('*.py')},
        'formal_inputs_opened':False,'sealed_inputs_opened':False,'scientific_performance_verified':False,
        'external_attached_test_script_available':False}
    s.write(s.REPORTS/'NUMERICAL_CONTRACTS.json',s.plain(receipt),replace=True)
    print(__import__('json').dumps(s.plain(receipt),ensure_ascii=True),flush=True)
    return receipt

if __name__=='__main__':
    raise SystemExit(0 if run()['all_passed'] else 2)
