"""CPU equivalence and bounded-backend contracts; no CUDA qualification."""
from .. import storage as s
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .. import coupled_preconditioner as v1
from .. import coupled_preconditioner_v2 as v2
from config import default_config
from physics import SpectralPhysics
from solver import pack,unpack
from unittest.mock import patch
import json
import time
import torch

REPORT=s.REPORTS/'repair_001/v3_preconditioner_002'
SOURCES=('research_closeout/coupled_preconditioner_v2.py','research_closeout/tests/coupled_preconditioner_v2.py')


def run():
    started=time.perf_counter();torch.set_num_threads(2);tests=[]
    old=s.read(s.REPORTS/'repair_001/v3_tangent_probe_001/REGISTRATION.json')['source_sha256']
    before={name:s.sha(s.ROOT/name) for name in old}
    def check(name,passed,**evidence): tests.append({'name':name,'passed':bool(passed),**s.plain(evidence)})
    generator=torch.Generator(device='cpu').manual_seed(20261015)
    A=torch.randn((133,3,3),dtype=torch.float64,generator=generator)
    A=A@A.transpose(-1,-2)+torch.eye(3,dtype=torch.float64)
    e=v2.eigvalsh_bounded(A);L,info=v2.cholesky_ex_bounded(A)
    rhs=torch.randn((1,133,3,1),dtype=torch.float64,generator=generator)
    answer=v2.cholesky_solve_bounded(rhs,L[None])
    check('block_eigenvalues_Cholesky_and_solve_equal_dense_CPU_reference',
        torch.allclose(e,torch.linalg.eigvalsh(A),atol=1e-14,rtol=1e-14)
        and not bool(info.any()) and torch.allclose(L,torch.linalg.cholesky(A),atol=1e-14,rtol=1e-14)
        and torch.allclose(answer,torch.linalg.solve(A[None],rhs),atol=2e-14,rtol=2e-14),batch=133)
    scalar=A[0];scalar_rhs=rhs[0,0]
    scalarL,scalarinfo=v2.cholesky_ex_bounded(scalar)
    check('unbatched_six_or_three_block_shape_preserved',scalarL.shape==(3,3) and scalarinfo.shape==()
        and torch.allclose(v2.cholesky_solve_bounded(scalar_rhs,scalarL),torch.linalg.solve(scalar,scalar_rhs)))

    #The backend is replaced, so65535 native blocks test dispatch only. No
    #native eigensystem, Hessian, inverse or GPU computation occurs here.
    calls={'eig':[],'chol':[],'solve':[]}
    def eig_fake(value): calls['eig'].append(value.shape[0]);return torch.ones((len(value),3),dtype=value.dtype)
    def chol_fake(value): calls['chol'].append(value.shape[0]);return value.clone(),torch.zeros(len(value),dtype=torch.int32)
    def solve_fake(value,factor): calls['solve'].append(value.shape[0]);return value.clone()
    blocks=torch.eye(3,dtype=torch.float64).expand(65535,3,3)
    with patch.object(torch.linalg,'eigvalsh',side_effect=eig_fake),\
         patch.object(torch.linalg,'cholesky_ex',side_effect=chol_fake),\
         patch.object(torch,'cholesky_solve',side_effect=solve_fake):
        native_e=v2.eigvalsh_bounded(blocks)
        native_L,native_info=v2.cholesky_ex_bounded(blocks)
        native_answer=v2.cholesky_solve_bounded(torch.ones((1,65535,3,1),dtype=torch.float64),native_L[None])
    check('native65535_fake_backends_never_receive_more_than64_blocks',v2.BATCH_LIMIT==64
        and all(max(value)==64 and sum(value)==65535 and len(value)==1024 for value in calls.values())
        and native_e.shape==(65535,3) and native_answer.shape==(1,65535,3,1)
        and not bool(native_info.any()),dispatch_counts={name:len(value) for name,value in calls.items()},
        maximum_backend_batch=max(max(value) for value in calls.values()),native_block_count=65535,
        scope='fake backend dispatch only; not native CUDA acceptance')
    del blocks,native_e,native_L,native_info,native_answer

    p=SpectralPhysics(default_config(32),dtype=torch.float64)
    state=p.M.new_zeros((1,5,32,32));state[:,0]=.1023;state[:,1]=-.63
    beta=p.M.new_tensor([[.02,.001,-.002]])
    W=(p(state,beta).clamp_min(0)+p.extra_variance).reciprocal()
    strength=state.new_tensor([1e-5,1e-5,0.,0.,.1])
    pre1,D1,m1=v1.build(p,state,beta,W,strength,1e-3)
    pre2,D2,m2=v2.build(p,state,beta,W,strength,1e-3)
    a=torch.randn((1,5123),dtype=state.dtype,generator=generator)
    z,b=unpack(a,(32,32));z[:,2:4]=0.;a=pack(z,b)
    b=torch.randn(a.shape,dtype=a.dtype,generator=generator);z,bb=unpack(b,(32,32));z[:,2:4]=0.;b=pack(z,bb)
    ma=pre2(a);mb=pre2(b);relative=float((ma-pre1(a)).norm()/pre1(a).norm())
    check('CPU32_v2_preconditioner_equals_immutable_v1_reference',torch.equal(D1,D2) and relative<2e-12,
        apply_relative_error=relative)
    ignored={'schema','backend_batch_limit','same_frequency_symbol_as_v1','GPU_qualification',
        'nonzero_frequency_block_eigenvalue_min','nonzero_frequency_block_eigenvalue_max','low_mode_equilibrated_eigenvalues'}
    check('frequency_symbol_prior_D_and_zero_beta6_block_unchanged',
        {k:value for k,value in m1.items() if k not in ignored}=={k:value for k,value in m2.items() if k not in ignored}
        and torch.allclose(torch.tensor(m1['low_mode_block']),torch.tensor(m2['low_mode_block']),atol=0,rtol=0))
    lhs=float((a*mb).sum());rhs_value=float((ma*b).sum())
    error=abs(lhs-rhs_value)/max(abs(lhs),abs(rhs_value),1.)
    check('active_subspace_positive_and_self_adjoint',float((a*ma).sum())>0 and error<1e-11,
        self_adjoint_relative_error=error)
    mask=torch.ones_like(a,dtype=torch.bool);mask[:,0:7]=False
    masked=lambda v:mask*pre2(mask*v)
    aa=a*mask;bb=b*mask
    mask_error=abs(float((aa*masked(bb)).sum()-(masked(aa)*bb).sum()))/max(abs(float((aa*masked(bb)).sum())),1.)
    check('free_mask_is_two_sided_and_positive',torch.count_nonzero(masked((~mask).double()))==0
        and float((aa*masked(aa)).sum())>0 and mask_error<1e-11,self_adjoint_relative_error=mask_error)
    high=state.new_zeros(state.shape);high[:,0]=torch.cos(2*torch.pi*7*torch.arange(32,dtype=state.dtype)/32)[None,None,:]
    p1zero,_,_=v1.build(p,state,beta,W,strength,0.)
    p2zero,_,_=v2.build(p,state,beta,W,strength,0.)
    high_vector=pack(high,torch.zeros_like(beta))
    check('H_null_original_prior_scale_unchanged_without_damping',
        torch.allclose(p1zero(high_vector),p2zero(high_vector),atol=1e-8,rtol=2e-12))
    check('all_v1_registered_dependencies_and_receipts_preserved',before=={name:s.sha(s.ROOT/name) for name in old}
        and all(s.sha(s.ROOT/name)==digest for name,digest in old.items()))
    result={'schema':v2.VERSION+'-CPU-contracts','time':s.now(),'all_passed':all(t['passed'] for t in tests),'tests':tests,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in SOURCES},'v1_source_sha256':before,
        'native_GPU_qualified':False,'GPU_used':False,'whole_inverse_calls':0,'scientific_data_read':False,
        'batch_limit':64,'native_dispatch_fixture_blocks':65535,'CPU_wall_s':time.perf_counter()-started}
    REPORT.mkdir(parents=True,exist_ok=True)
    number=len(list(REPORT.glob('ATTEMPT_*.json')))+1;s.write(REPORT/f'ATTEMPT_{number:03d}.json',result)
    if result['all_passed']: s.write(REPORT/'CONTRACTS.json',result,replace=(REPORT/'CONTRACTS.json').exists())
    print(json.dumps({'all_passed':result['all_passed'],'tests':len(tests),'failures':[t for t in tests if not t['passed']],
        'CPU_wall_s':result['CPU_wall_s']},indent=2),flush=True)
    return result


if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
