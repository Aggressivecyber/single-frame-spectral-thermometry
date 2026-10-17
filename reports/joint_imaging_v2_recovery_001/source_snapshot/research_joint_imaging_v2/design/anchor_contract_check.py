"""A small convex analogue of TV anchoring, not a thermometry experiment."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(23)
n=24;m=48
A=rng.normal(size=(m,n))/np.sqrt(m)
truth=np.r_[np.ones(8),np.ones(8)*2,np.ones(8)*.7]
y=A@truth+rng.normal(scale=.2,size=m)
D=np.roll(np.eye(n),-1,axis=1)-np.eye(n)
lam=.15;eps=.025;mu=.7

def evaluate(x,center=None,tv=True):
    r=A@x-y;f=.5*(r@r);g=A.T@r;H=A.T@A
    if tv:
        d=D@x;z=np.sqrt(d*d+eps*eps)
        f+=lam*z.sum();g+=lam*D.T@(d/z)
        H+=lam*(D.T*(eps*eps/z**3))@D
    if center is not None:
        h=x-center;f+=.5*mu*(h@h);g+=mu*h;H+=mu*np.eye(n)
    return f,g,H

def solve(initial,center=None,tv=True):
    x=initial.copy()
    for it in range(100):
        f,g,H=evaluate(x,center,tv)
        if np.linalg.norm(g)<1e-9:break
        step=np.linalg.solve(H,-g);t=1.
        for _ in range(45):
            z=x+t*step
            if evaluate(z,center,tv)[0]<=f+1e-4*t*(g@step):break
            t*=.5
        x=z
    return x,it,float(np.linalg.norm(evaluate(x,center,tv)[1]))
base,it,gn=solve(np.linalg.lstsq(A,y,rcond=None)[0])
anchor,_,ga=solve(base,base,True)
dconly,_,gd=solve(base,base,False)
# The log-brightness/log-ratio coordinate transform is invertible, not rank reduction.
I=np.exp(rng.normal(size=(3,7,9)))
l=np.log(I);q0=l.mean(0);q1=l[0]-l[1];q2=l[2]-l[1]
li=np.stack([q0+(2*q1-q2)/3,q0-(q1+q2)/3,q0+(-q1+2*q2)/3])
rec=np.exp(li)
results={
 'scope':'toy smooth-TV convex contract and coordinate algebra only',
 'TV_optimum_gradient_norm':gn,
 'anchored_TV_zero_proposal_relative_change':float(np.linalg.norm(anchor-base)/np.linalg.norm(base)),
 'data_only_zero_proposal_relative_change':float(np.linalg.norm(dconly-base)/np.linalg.norm(base)),
 'log_coordinate_roundtrip_max_abs_error':float(np.max(abs(rec-I))),
 'check_passed':bool(np.linalg.norm(anchor-base)<1e-7 and np.linalg.norm(dconly-base)>1e-3 and np.max(abs(rec-I))<1e-12),
 'not_tested':['camera dataset','published weights','new trained network','speed/memory','actual ADMM backward']}
Path(__file__).with_name('anchor_contract_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
assert results['check_passed']
