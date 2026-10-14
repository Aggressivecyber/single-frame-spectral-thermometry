"""Small dense reference plus actual256 calibration fixtures before formal retest."""
import argparse
import time
from pathlib import Path
from . import storage as s
from .solver import build_system,solve,solve_rhs
from .diagnostics import shared,decompose
from .. import common as c,scoring
import torch


def evaluate(p,raw,Ctrue,Cest,S,beta,options,dense=False):
    pT,pvalid=p.color_temperature(p.op.object_blur(Ctrue[None]))
    truth={'continuum_latent':Ctrue[None],'template_latent_e':S[None],
      'template_postblur_e':p.op.object_blur(S[None]),'temperature_color_K':pT,'evaluation_domain':pvalid,'beta':beta[None]}
    checks=[];cases=[]
    for mode in ('true_beta','estimated_beta'):
        system=build_system(p,raw,torch.zeros_like(Ctrue),options,fixed_beta=beta if mode=='true_beta' else None)
        cache=shared(system,truth)
        for name,C in (('C_true',Ctrue),('C_perturbed',Cest)):
            out=solve(system,C);source={'continuum_latent':C[None]}
            d=decompose(system,cache,source,out,truth)
            checks.append({'name':mode+'_'+name+'_identity','passed':d['summary']['numerical_gate_passed']})
            cases.append({'beta_mode':mode,'source':name,**scoring.plain(d['summary'])})
            if dense:
                n=system['dimension'];eye=torch.eye(n,dtype=torch.float64)
                K=torch.stack([system['matvec'](e) for e in eye],1)
                exact=torch.linalg.solve(K,out['rhs'])
                post_error=system['physics'].op.object_blur(system['unpack'](out['vector']-exact)[0])
                rms=float(post_error.square().mean().sqrt())
                checks.append({'name':mode+'_'+name+'_dense_inverse','value':rms,'limit':1e-4,'passed':rms<=1e-4})
    return checks,cases


def run(device):
    start=time.perf_counter()
    if device=='cpu':
        from ..tests_decomposition import fixture
        p,raw,ce,ct,st,bt,opts=fixture()
        checks,cases=evaluate(p,raw,ct,ce,st,bt,opts,dense=True)
    else:
        p=c.make_physics(device);h,w=p.cfg['shape']
        yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=p.M.dtype),torch.arange(w,device=device,dtype=p.M.dtype),indexing='ij')
        z=p.M.new_zeros((1,5,h,w));z[:,0]=(3052.5-p.T0)/p.Tscale;z[:,1]=-.5+.025*torch.cos(2*torch.pi*xx/w)
        bt=p.M.new_tensor([.5,.015,-.01]);ct=p.components(z,bt[None])[0][0]
        st=180*(.55+.25*torch.cos(2*torch.pi*xx/w)+.15*torch.sin(4*torch.pi*yy/h))
        f=c.frozen_v5('signed_linear_solver');G=f.SignedLinearOperator(p)
        raw=p.op(ct[None])[0]+G.apply_template(st)+G.apply_beta(bt)+13*torch.cos(2*torch.pi*(17*xx/w+7*yy/h))
        altered=z.clone();altered[:,0]+=.004*torch.cos(2*torch.pi*xx/w);altered[:,1]+=.001*torch.sin(4*torch.pi*yy/h)
        ce=p.components(altered,bt[None])[0][0]
        opts={**f.DEFAULT_OPTIONS,'alpha_g':1.0}
        checks,cases=evaluate(p,raw,ct,ce,st,bt,opts)
    result={'schema':'v6-repair001-numerical-fixtures-001','time':c.now(),'device':device,'shape':p.cfg['shape'],
       'all_passed':all(r['passed'] for r in checks),'checks':checks,'cases':cases,'wall_s':time.perf_counter()-start,
       'source_sha256':s.sources(),'scientific_acceptance':False,'formal_or_sealed_inputs_read':False}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    r=run(a.device);c.write_json(a.output,scoring.plain(r))
    print({'all_passed':r['all_passed'],'checks':len(r['checks']),'wall_s':r['wall_s'],'output':str(a.output)},flush=True)
    if not r['all_passed']:
        print([x for x in r['checks'] if not x['passed']],flush=True);raise SystemExit(1)


if __name__=='__main__':main()
