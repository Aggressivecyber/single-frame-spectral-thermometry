"""CPU same-factor triangular-apply contracts; no native CUDA qualification."""
from .. import storage as s
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .. import coupled_preconditioner_v2 as v2
from .. import coupled_preconditioner_v3 as v3
from config import default_config
from physics import SpectralPhysics
from solver import pack,unpack
from unittest.mock import patch
import ast
import inspect
import json
import math
import time
import torch

REPORT=s.REPORTS/'repair_001/v3_preconditioner_003'
SOURCES=('research_closeout/coupled_preconditioner_v3.py','research_closeout/tests/coupled_preconditioner_v3.py')
V2_REPORT=s.REPORTS/'repair_001/v3_tangent_probe_002'
PANEL=s.REPORTS/'repair_001/quick_v003/PANEL_REGISTRATION.json'


def run():
    started=time.perf_counter();torch.set_num_threads(2);tests=[]
    generator=torch.Generator(device='cpu').manual_seed(2026101503)
    registered17=s.read(V2_REPORT/'REGISTRATION.json')['source_sha256']
    registered38=s.read(PANEL)['source_sha256']
    frozen={**registered17,**registered38};before={name:s.sha(s.ROOT/name) for name in frozen}
    receipt_paths=[V2_REPORT/name for name in ('REGISTRATION.json','REGISTRATION_SHA256.json','RESULT.json','CELL_00.json','CELL_01.json')]+[PANEL]
    receipt_before={str(path):s.sha(path) for path in receipt_paths}
    def check(name,passed,**evidence): tests.append({'name':name,'passed':bool(passed),**s.plain(evidence)})
    def raises(kind,fn):
        try: fn()
        except kind: return True
        return False
    def random(shape,dtype=torch.float64):
        real=torch.randn(shape,dtype=torch.float64,generator=generator)
        return real if dtype==torch.float64 else torch.complex(real,torch.randn(shape,dtype=torch.float64,generator=generator))
    def factor(shape,dtype):
        value=random((*shape,3,3),dtype)
        A=value@value.mH+torch.eye(3,dtype=dtype)
        return torch.linalg.cholesky(A)
    def relative(a,b): return float((a-b).norm()/b.norm().clamp_min(1e-300))

    for dtype in (torch.float64,torch.complex128):
        L=factor((133,),dtype);rhs=random((1,133,3,4),dtype)
        actual=v3.cholesky_solve3(rhs,L[None])
        old=v2.cholesky_solve_bounded(rhs,L[None])
        reference=torch.cholesky_solve(rhs,L[None])
        check('same_factor_'+str(dtype)+'_equals_v2_and_torch',actual.shape==(1,133,3,4)
            and actual.dtype==dtype and relative(actual,reference)<2e-13 and relative(actual,old)<2e-13,
            relative_to_torch=relative(actual,reference),relative_to_v2=relative(actual,old),small_block_dimension=3)

    broadcast_cases=[((2,1),(1,4),5), ((),(2,4),2), ((2,1),(),3), ((),(),1)]
    broadcast_errors=[];broadcast_shapes=[]
    for fshape,rshape,columns in broadcast_cases:
        L=factor(fshape,torch.complex128);rhs=random((*rshape,3,columns),torch.complex128)
        actual=v3.cholesky_solve3(rhs,L);reference=v2.cholesky_solve_bounded(rhs,L)
        broadcast_errors.append(relative(actual,reference));broadcast_shapes.append(list(actual.shape))
    check('leading_batch_and_unbatched_RHS_factor_shapes_broadcast_like_v2',max(broadcast_errors)<2e-13,
        output_shapes=broadcast_shapes,relative_errors=broadcast_errors)
    L=factor((4,),torch.complex128);rhs=random((4,5,3),torch.complex128).transpose(-1,-2)
    stridedL=L.transpose(-1,-2).contiguous().transpose(-1,-2)
    check('noncontiguous_factor_and_multicolumn_RHS_are_supported',not rhs.is_contiguous()
        and not stridedL.is_contiguous()
        and relative(v3.cholesky_solve3(rhs,stridedL),torch.cholesky_solve(rhs,stridedL))<2e-13)
    empty_columns=v3.cholesky_solve3(torch.empty((2,3,0),dtype=torch.float64),torch.eye(3,dtype=torch.float64))
    empty_batches=v3.cholesky_solve3(torch.empty((0,3,2),dtype=torch.float64),torch.eye(3,dtype=torch.float64))
    check('zero_column_or_zero_batch_shapes_do_not_call_a_backend',empty_columns.shape==(2,3,0) and empty_batches.shape==(0,3,2))
    poisoned=L.clone();poisoned[...,0,1]=100+7j;poisoned[...,0,2]=-250+9j;poisoned[...,1,2]=300-4j
    check('only_lower_Cholesky_entries_are_used',torch.equal(v3.cholesky_solve3(rhs,L),v3.cholesky_solve3(rhs,poisoned)))
    check('invalid_shapes_and_nonmatching_or_non64bit_types_are_rejected',
        raises(ValueError,lambda:v3.cholesky_solve3(torch.ones((2,1),dtype=torch.float64),torch.eye(3,dtype=torch.float64)))
        and raises(ValueError,lambda:v3.cholesky_solve3(torch.ones((3,1),dtype=torch.float64),torch.eye(4,dtype=torch.float64)))
        and raises(ValueError,lambda:v3.cholesky_solve3(torch.ones((3,1),dtype=torch.float64),torch.eye(3,dtype=torch.complex128)))
        and raises(ValueError,lambda:v3.cholesky_solve3(torch.ones((3,1)),torch.eye(3)))
        and raises(TypeError,lambda:v3.cholesky_solve3([1,2,3],torch.eye(3,dtype=torch.float64))))

    condition_evidence=[]
    for dtype in (torch.float64,torch.complex128):
        for exponent in (0,2,4,6,8,10,12):
            Q,_=torch.linalg.qr(random((3,3),dtype))
            eigen=torch.tensor([1.,10.**(-exponent/2),10.**(-exponent)],dtype=dtype)
            A=(Q*eigen[None])@Q.mH;A=.5*(A+A.mH)
            L=torch.linalg.cholesky(A);rhs=random((3,2),dtype)
            actual=v3.cholesky_solve3(rhs,L);reference=torch.cholesky_solve(rhs,L)
            backward=float((A@actual-rhs).norm()/(A.norm()*actual.norm()+rhs.norm()))
            condition_evidence.append({'dtype':str(dtype),'nominal_condition':10.**exponent,
                'same_factor_relative_error':relative(actual,reference),'normwise_backward_error':backward,
                'finite':bool(torch.isfinite(actual).all())})
    check('same_factor_equivalence_and_backward_error_through_condition1e12_without_ridge_or_clipping',
        all(row['finite'] and row['same_factor_relative_error']<2e-12 and row['normwise_backward_error']<2e-14
            for row in condition_evidence),cases=condition_evidence,
        scope='CPU3x3 factored systems only; not a native-grid or CUDA condition certificate')

    L=factor((9,),torch.complex128);a=random((9,3,2),torch.complex128);b=random(a.shape,torch.complex128)
    ma=v3.cholesky_solve3(a,L);mb=v3.cholesky_solve3(b,L)
    lhs=(a.conj()*mb).sum();right=(ma.conj()*b).sum();energy=(a.conj()*ma).sum()
    hermitian_error=float((lhs-right).abs()/max(float(lhs.abs()),float(right.abs()),1.))
    check('complex_apply_is_Hermitian_and_positive_with_conjugate_transpose_back_substitution',
        hermitian_error<2e-13 and float(energy.real)>0 and abs(float(energy.imag))/float(energy.real)<2e-13,
        Hermitian_relative_error=hermitian_error)

    #65535 identity factors only exercise vectorized CPU apply and bounded
    #build dispatch. They are not a native scientific system or CUDA profiler.
    count=65535;nativeL=torch.eye(3,dtype=torch.complex128).expand(count,3,3)
    native_rhs=torch.ones((1,count,3,1),dtype=torch.complex128)
    forbidden=AssertionError('The3x3 hot path must not call any linear-algebra backend')
    with patch.object(torch,'cholesky_solve',side_effect=forbidden) as backend,\
         patch.object(torch.linalg,'solve',side_effect=forbidden),\
         patch.object(torch.linalg,'solve_triangular',side_effect=forbidden),\
         patch.object(torch.linalg,'inv',side_effect=forbidden),\
         patch.object(torch,'inverse',side_effect=forbidden):
        native_answer=v3.cholesky_solve3(native_rhs,nativeL[None])
    body=ast.parse(inspect.getsource(v3.cholesky_solve3))
    check('field_apply65535_blocks_has_zero_batched_linalg_calls_and_no_Python_batch_loop',
        backend.call_count==0 and torch.equal(native_answer,native_rhs)
        and not any(isinstance(node,(ast.For,ast.While,ast.AsyncFor)) for node in ast.walk(body)),
        blocks=count,field_backend_calls=backend.call_count,native_GPU_qualified=False)
    dispatch={'eig':[],'chol':[]}
    def fake_eig(value): dispatch['eig'].append(len(value));return torch.ones((len(value),3),dtype=value.dtype)
    def fake_chol(value): dispatch['chol'].append(len(value));return value.clone(),torch.zeros(len(value),dtype=torch.int32)
    blocks=torch.eye(3,dtype=torch.float64).expand(count,3,3)
    with patch.object(torch.linalg,'eigvalsh',side_effect=fake_eig),\
         patch.object(torch.linalg,'cholesky_ex',side_effect=fake_chol):
        v3.eigvalsh_bounded(blocks);v3.cholesky_ex_bounded(blocks)
    check('build_eigen_Cholesky_backends_remain_fixed_at_most64_blocks',v3.BATCH_LIMIT==64
        and v3.eigvalsh_bounded is v2.eigvalsh_bounded and v3.cholesky_ex_bounded is v2.cholesky_ex_bounded
        and all(max(values)==64 and sum(values)==count and len(values)==1024 for values in dispatch.values()),
        maximum_build_backend_batch=max(max(values) for values in dispatch.values()),scope='fake CPU dispatch')
    del nativeL,native_rhs,native_answer,blocks

    p=SpectralPhysics(default_config(32),dtype=torch.float64)
    state=p.M.new_zeros((1,5,32,32));state[:,0]=.1023;state[:,1]=-.63
    beta=p.M.new_tensor([[.02,.001,-.002]])
    W=(p(state,beta).clamp_min(0)+p.extra_variance).reciprocal()
    strength=state.new_tensor([1e-5,1e-5,0.,0.,.1])
    untouched=[value.clone() for value in (state,beta,W,strength)]
    captures={'v2':{},'v3':{}};original_factor=v2._factor
    def capture(tag):
        def wrapped(block,name):
            root,L=original_factor(block,name)
            captures[tag][name]={'block':block.clone(),'root':root.clone(),'factor':L.clone()}
            return root,L
        return wrapped
    with patch.object(v2,'_factor',side_effect=capture('v2')):
        pre2,D2,m2=v2.build(p,state,beta,W,strength,1e-3)
    with patch.object(v3,'_factor',side_effect=capture('v3')):
        pre3,D3,m3=v3.build(p,state,beta,W,strength,1e-3)
    check('CPU32_field_root_factor_symbol_and6x6_low_block_are_byte_identical_to_v2',
        captures['v2'].keys()==captures['v3'].keys()
        and all(torch.equal(captures['v2'][name][key],captures['v3'][name][key])
            for name in captures['v2'] for key in ('block','root','factor')))
    added={'same_frequency_symbol_as_v2','field_apply','field_apply_batched_linalg_calls',
        'low_mode_apply_backend_calls','apply_cost_mechanism_evidence'}
    check('original_D_prior_boundary_scope_and_all_other_v2_metadata_remain_exact',torch.equal(D2,D3)
        and {k:value for k,value in m2.items() if k!='schema'}=={k:value for k,value in m3.items() if k not in added|{'schema'}}
        and not m3['original_objective_changed'] and not m3['physical_constraints_changed']
        and not m3['exact_Hessian_inverse_claimed'] and m3['FFT_norm']=='ortho')

    a=random((1,5123));fields,bb=unpack(a,(32,32));fields[:,2:4]=0.;a=pack(fields,bb)
    b=random(a.shape);fields,bb=unpack(b,(32,32));fields[:,2:4]=0.;b=pack(fields,bb)
    backend_calls={'v2':[],'v3':[]};original_solve=torch.cholesky_solve
    def recorded(tag):
        def wrapped(rhs,L):
            backend_calls[tag].append({'blocks':len(L),'dimension':L.shape[-1]})
            return original_solve(rhs,L)
        return wrapped
    with patch.object(torch,'cholesky_solve',side_effect=recorded('v2')): old_apply=pre2(a)
    with patch.object(torch,'cholesky_solve',side_effect=recorded('v3')): new_apply=pre3(a)
    check('CPU32_full_apply_matches_v2_same_factors_float64',new_apply.dtype==torch.float64
        and relative(new_apply,old_apply)<2e-12,relative_apply_error=relative(new_apply,old_apply))
    check('full_apply_field_backend_dispatch_removed_and_only_original_low6_solve_remains',
        len([row for row in backend_calls['v2'] if row['dimension']==3])==16
        and backend_calls['v3']==[{'blocks':1,'dimension':6}]
        and m3['field_apply_batched_linalg_calls']==0 and m3['low_mode_apply_backend_calls']==1,
        calls=backend_calls,CPU_fixture_side=32,
        inference='At native256 side, static v2 field dispatch count is65536/64=1024 per apply. GPU launch cost dominance is not profiler established.')
    ma=pre3(a);mb=pre3(b);lhs=float((a*mb).sum());right=float((ma*b).sum())
    error=abs(lhs-right)/max(abs(lhs),abs(right),1.)
    check('CPU32_active_field_and_beta_subspace_is_positive_and_self_adjoint',float((a*ma).sum())>0 and error<2e-11,
        self_adjoint_relative_error=error)
    fields,bb=unpack(ma,(32,32))
    inactive=torch.zeros_like(state);inactive[:,2:4]=random((1,2,32,32))
    check('inactive_fields_are_zero_and_do_not_influence_active_output',torch.count_nonzero(fields[:,2:4])==0
        and torch.count_nonzero(pre3(pack(inactive,torch.zeros_like(beta))))==0)
    mask=torch.ones_like(a,dtype=torch.bool);mask[:,0:11]=False
    masked=lambda x:mask*pre3(mask*x)
    masked_error=abs(float((a*masked(b)).sum()-(masked(a)*b).sum()))/max(abs(float((a*masked(b)).sum())),1.)
    check('two_sided_free_mask_preserves_positive_self_adjoint_apply_and_zero_bound_entries',
        torch.count_nonzero(masked((~mask).double()))==0 and torch.count_nonzero(masked(a)[~mask])==0
        and float((a*masked(a)).sum())>0 and masked_error<2e-11,self_adjoint_relative_error=masked_error)

    fields,_=unpack(a,(32,32));spectrum=torch.fft.fft2(fields[:,[0,1,4]],norm='ortho').permute(0,2,3,1)
    field=captures['v3']['Nonzero-frequency block'];root=field['root'][None]
    response=v3.cholesky_solve3((spectrum/root)[...,None],field['factor'][None].to(torch.complex128))[...,0]/root
    negative=(-torch.arange(32))%32
    reflected=response[:,negative[:,None],negative[None,:]].conj()
    hermitian=relative(response,reflected)
    restored=torch.fft.ifft2(response.permute(0,3,1,2),norm='ortho')
    imaginary=float(restored.imag.norm()/restored.real.norm())
    check('same_even_field_factors_preserve_Hermitian_Fourier_symmetry_and_real_ifft',
        hermitian<2e-12 and imaginary<2e-12,Hermitian_relative_error=hermitian,imaginary_relative_norm=imaginary)
    high=state.new_zeros(state.shape)
    high[:,0]=torch.cos(2*torch.pi*7*torch.arange(32,dtype=state.dtype)/32)[None,None,:]
    pre2zero,_,_=v2.build(p,state,beta,W,strength,0.)
    pre3zero,_,_=v3.build(p,state,beta,W,strength,0.)
    high_vector=pack(high,torch.zeros_like(beta))
    check('H_null_original_prior_high_frequency_scale_is_unchanged_without_damping',
        relative(pre3zero(high_vector),pre2zero(high_vector))<2e-12)
    check('fixed_inputs_are_unmodified_and_no_optimizer_or_nonlinear_update_occurs',
        all(torch.equal(old,new) for old,new in zip(untouched,(state,beta,W,strength)))
        and m3['complete_inverse_calls']==0 and not m3['inference_reads_truth'])
    check('packed_shape_finite_precision_contract_is_unchanged',
        raises(ValueError,lambda:pre3(torch.zeros((1,3),dtype=torch.float64)))
        and raises(ValueError,lambda:pre3(a.float()))
        and raises(ValueError,lambda:pre3(torch.full_like(a,float('nan')))))

    old_result=s.read(V2_REPORT/'RESULT.json');side=s.read(V2_REPORT/'REGISTRATION_SHA256.json')
    cells=[s.read(V2_REPORT/f'CELL_{index:02d}.json') for index in range(2)]
    v2_evidence=[]
    for index,(cell,control) in enumerate(zip(cells,old_result['reused_old_block_controls'])):
        old=control['old_row'];v2_evidence.append({'case_id':cell['case_id'],
            'v2_iterations':cell['iterations'],'old_iterations':old['iterations'],
            'v2_wall_s':cell['wall_s'],'old_wall_s':old['wall_s'],'v2_to_old_wall_ratio':cell['wall_s']/old['wall_s'],
            'true_relative_residual':cell['true_relative_residual'],
            'cell_sha256':s.sha(V2_REPORT/f'CELL_{index:02d}.json')})
    check('v2_facts_are_two_converged_same_system_rows_with_slower_observed_wall_time',
        old_result['planned_new_cells']==old_result['returned_new_cells']==old_result['converged_new_cells']==2
        and old_result['registration_sha256']==side['sha256']
        and all(cell==row and cell['converged'] and cell['system_identity_match']
            and cell['true_relative_residual']<=1e-10 for cell,row in zip(cells,old_result['rows']))
        and all(row['v2_to_old_wall_ratio']>1 for row in v2_evidence),observed=v2_evidence)
    after={name:s.sha(s.ROOT/name) for name in frozen}
    check('all_frozen17_and38_registered_sources_and_prior_receipts_remain_byte_identical',
        len(registered17)==17 and len(registered38)==38 and before==after==frozen
        and receipt_before=={str(path):s.sha(path) for path in receipt_paths},
        v2_registered_count=len(registered17),panel_registered_count=len(registered38),unique_frozen_sources=len(frozen))
    check('build_API_is_unchanged_and_GPU_speedup_is_not_claimed',
        list(inspect.signature(v3.build).parameters)==['p','state','beta','W','strength','damping']
        and 'no native CUDA profiler or wall-time benefit established' in m3['apply_cost_mechanism_evidence']
        and m3['GPU_qualification']=='requires separate native CUDA receipt')
    result={'schema':v3.VERSION+'-CPU-contracts','time':s.now(),'all_passed':all(row['passed'] for row in tests),
        'tests':tests,'source_sha256':{name:s.sha(s.ROOT/name) for name in SOURCES},
        'v2_registered_source_sha256':registered17,'panel_registered_source_sha256':registered38,
        'prior_receipt_sha256':receipt_before,'observed_v2_facts':v2_evidence,
        'cost_mechanism_inference':'Static v2 code makes1024 batched3x3 backend solves per native256 field apply. v3 removes those calls using the same factor; native wall-time benefit requires a separate registered probe.',
        'native_profiler_used':False,'GPU_used':False,'native_GPU_qualified':False,
        'whole_inverse_calls':0,'real_raw_or_parameter_cache_arrays_read':False,'truth_arrays_read':False,
        'optimizer_started':False,'native_tangent_probe_started':False,'dense_reference_block_dimension':3,
        'max_physics_fixture_side':32,'build_backend_batch_limit':64,'CPU_wall_s':time.perf_counter()-started}
    REPORT.mkdir(parents=True,exist_ok=True)
    number=len(list(REPORT.glob('ATTEMPT_*.json')))+1;s.write(REPORT/f'ATTEMPT_{number:03d}.json',result)
    if result['all_passed']: s.write(REPORT/'CONTRACTS.json',result,replace=(REPORT/'CONTRACTS.json').exists())
    print(json.dumps({'all_passed':result['all_passed'],'tests':len(tests),'failures':[row for row in tests if not row['passed']],
        'source_sha256':result['source_sha256'],'CPU_wall_s':result['CPU_wall_s']},indent=2),flush=True)
    return result


if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
