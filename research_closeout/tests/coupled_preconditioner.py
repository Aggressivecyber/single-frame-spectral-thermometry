"""CPU32 fixtures with at most24-dimensional dense tangent references."""
from .. import storage as s
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from ..coupled_preconditioner import build,VERSION
from config import default_config
from physics import SpectralPhysics
from solver import pack,unpack
from repair_v3.solver import laplacian,pcg,objective,block_preconditioner
import copy
import inspect
import json
import math
import time
import torch


def run():
    started=time.perf_counter();torch.set_num_threads(2)
    registration=s.read(s.REPORTS/'repair_001/quick_v001/QUICK_REGISTRATION.json')
    frozen_before={name:s.sha(s.ROOT/name) for name in registration['source_sha256']}
    tests=[];jvp_calls=[0];vjp_calls=[0]
    def check(name,passed,**evidence):
        tests.append({'name':name,'passed':bool(passed),**s.plain(evidence)})
    def rejected(fn):
        try: fn()
        except ValueError: return True
        return False
    cfg=default_config(32);p=SpectralPhysics(cfg,dtype=torch.float64)
    state=p.M.new_zeros((1,5,32,32));beta=p.M.new_tensor([[.02,.001,-.002]])
    yy,xx=torch.meshgrid(torch.arange(32,dtype=state.dtype),torch.arange(32,dtype=state.dtype),indexing='ij')
    state[:,0]=.1023+.001*torch.sin(2*torch.pi*xx/32)
    state[:,1]=math.log(.53)+.015*torch.cos(2*torch.pi*yy/32)
    state[:,4]=.1*torch.sin(2*torch.pi*(xx+yy)/32)
    raw=p(state,beta)+.2*torch.sin(2*torch.pi*xx[None]/32)
    W=(raw.clamp_min(0)+p.extra_variance).reciprocal()
    strength=state.new_tensor([1e-5,1e-5,0.,0.,.1]);damping=1e-3
    original_jvp=p.jvp;original_vjp=p.vjp
    def counted_jvp(*args): jvp_calls[0]+=1;return original_jvp(*args)
    def counted_vjp(*args): vjp_calls[0]+=1;return original_vjp(*args)
    p.jvp=counted_jvp;p.vjp=counted_vjp
    inputs=[v.clone() for v in (state,beta,W,strength)]
    buffers={name:value.clone() for name,value in p.named_buffers()};cfg_copy=copy.deepcopy(p.cfg)
    before=float(objective(p,raw,state,beta,W,strength)[0])
    trial_state=state+10.;trial_beta=beta+30.
    constrained_before=p.constrain(trial_state,trial_beta)
    precondition,scaling,meta=build(p,state,beta,W,strength,damping)
    check('build_six_exact_low_mode_columns_only',jvp_calls[0]==6 and meta['low_mode_jvp_columns']==6,
        JVP_calls_during_build=jvp_calls[0])
    d,db=p.preconditioner_diagonal(state,beta,W)
    check('returns_original_damping_diagonal_in_five_field_pack',torch.equal(scaling,pack(d,db))
        and scaling.shape==(1,5123) and meta['active_field_indices']==[0,1,4])
    after=float(objective(p,raw,state,beta,W,strength)[0]);constrained_after=p.constrain(trial_state,trial_beta)
    check('fixed_objective_inputs_calibration_and_constraints_unchanged',before==after
        and all(torch.equal(a,b) for a,b in zip(inputs,(state,beta,W,strength)))
        and all(torch.equal(buffers[name],value) for name,value in p.named_buffers())
        and p.cfg==cfg_copy and all(torch.equal(a,b) for a,b in zip(constrained_before,constrained_after)),
        objective_absolute_difference=abs(after-before))
    check('raw_calibration_only_API_without_truth_or_outer_solver',list(inspect.signature(build).parameters)==
        ['p','state','beta','W','strength','damping'] and meta['inference_reads_truth'] is False
        and not meta['exact_Hessian_inverse_claimed'] and not meta['original_objective_changed'])

    generator=torch.Generator(device='cpu').manual_seed(20261015)
    def active_random():
        value=torch.randn((1,5123),dtype=state.dtype,generator=generator)
        fields,b=unpack(value,(32,32));fields[:,2:4]=0.
        return pack(fields,b)
    a=active_random();b=active_random();ma=precondition(a);mb=precondition(b)
    lhs=float((a*mb).sum());rhs=float((ma*b).sum())
    symmetry=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1.)
    minimum=min(float((a*ma).sum()),float((b*mb).sum()))
    check('active_subspace_positive_and_self_adjoint',minimum>0 and symmetry<5e-12,
        minimum_tested_quadratic_form=minimum,self_adjoint_relative_error=symmetry)
    inactive=torch.zeros_like(a);inactive_fields,inactive_beta=unpack(inactive,(32,32));inactive_fields[:,2]=1.
    inactive=pack(inactive_fields,inactive_beta)
    check('inactive_field_vectors_project_to_zero',torch.count_nonzero(precondition(inactive))==0
        and torch.count_nonzero(unpack(ma,(32,32))[0][:,2:4])==0)

    # Six unit-norm low coordinates. FFT0 of each constant field is1.
    basis=[]
    for index in (0,1,4):
        z=torch.zeros_like(state);z[:,index]=1/32
        basis.append(pack(z,torch.zeros_like(beta))[0])
    for index in range(3):
        bdir=torch.zeros_like(beta);bdir[:,index]=1.
        basis.append(pack(torch.zeros_like(state),bdir)[0])
    Qlow=torch.stack(basis)
    columns=[]
    for q in Qlow:
        dz,d_beta=unpack(q[None],(32,32))
        columns.append(p.jvp(state,beta,dz,d_beta).reshape(-1))
    Jlow=torch.stack(columns)
    low_exact=(Jlow*W.reshape(-1)[None])@Jlow.T+damping*(Qlow*scaling)@Qlow.T
    low_report=state.new_tensor(meta['low_mode_block'])
    cross=float(low_report[:3,3:].abs().max())
    low_error=float((low_exact-low_report).abs().max())
    low_rhs=torch.randn(6,dtype=state.dtype,generator=generator)
    low_reference=torch.linalg.solve(low_exact,low_rhs)
    low_actual=precondition((low_rhs@Qlow)[None])@Qlow.T
    low_solve_error=float(torch.linalg.vector_norm(low_actual[0]-low_reference)/torch.linalg.vector_norm(low_reference))
    low_residual=float(torch.linalg.vector_norm(low_exact@low_actual[0]-low_rhs)/torch.linalg.vector_norm(low_rhs))
    check('low_mode_beta_coupling_uses_exact_six_column_Gram_and_ortho_units',cross>0 and low_error<1e-8
        and low_solve_error<1e-10 and low_residual<1e-9,
        maximum_low_Gram_error=low_error,largest_field_beta_cross_term=cross,
        dense6_solution_relative_error=low_solve_error,true_dense6_residual=low_residual)

    # Independent constant-state Fourier-Rayleigh calculation checks the
    # shared DC term against the actual coded tangent operator, not its formula.
    constant=state.clone();constant[:,0]=.1023;constant[:,1]=math.log(.53);constant[:,4]=0.
    constant_W=torch.full_like(W,float(W.mean()))
    const_pre,const_D,const_meta=build(p,constant,beta,constant_W,strength,damping)
    cosine=math.sqrt(2/1024)*torch.cos(2*torch.pi*xx/32)
    Qmode=[]
    for index in (0,1,4):
        z=torch.zeros_like(state);z[:,index]=cosine
        Qmode.append(pack(z,torch.zeros_like(beta))[0])
    Qmode=torch.stack(Qmode)
    mode_columns=[]
    for q in Qmode:
        dz,d_beta=unpack(q[None],(32,32))
        mode_columns.append(p.jvp(constant,beta,dz,d_beta).reshape(-1))
    Jmode=torch.stack(mode_columns)
    lap_mode=4*math.sin(math.pi/32)**2
    mode_exact=(Jmode*constant_W.reshape(-1)[None])@Jmode.T+torch.diag(strength[[0,1,4]]*lap_mode)
    mode_exact+=damping*(Qmode*const_D)@Qmode.T
    mode_inverse=torch.stack([(const_pre(q[None])@Qmode.T)[0] for q in Qmode],1)
    mode_residual=float(torch.linalg.matrix_norm(mode_exact@mode_inverse-torch.eye(3,dtype=state.dtype)))
    check('common_DC_cross_band_and_both_carriers_match_true_constant_tangent_mode',mode_residual<2e-9
        and const_meta['shared_data_DC_cross_band_terms_retained'],
        dense3_tangent_times_preconditioner_minus_identity_norm=mode_residual,
        tangent_mode=[1,0])

    # H-null frequency must follow the original nearest-neighbor prior, with
    # no hidden sensitivity floor or new template regularization.
    zero_pre,zero_D,zero_meta=build(p,constant,beta,constant_W,strength,0.)
    high=math.sqrt(2/1024)*torch.cos(2*torch.pi*(7*xx+5*yy)/32)
    high_lap=4*math.sin(7*math.pi/32)**2+4*math.sin(5*math.pi/32)**2
    high_errors=[]
    for index in (0,1,4):
        z=torch.zeros_like(state);z[:,index]=high;vector=pack(z,torch.zeros_like(beta))
        expected=vector/(strength[index]*high_lap)
        high_errors.append(float(torch.linalg.vector_norm(zero_pre(vector)-expected)/torch.linalg.vector_norm(expected)))
    check('object_null_high_frequency_has_original_lambda_laplacian_scale',float(p.op.object_otf[5,7])==0
        and max(high_errors)<2e-10,relative_errors_by_active_field=high_errors,
        original_coefficients=strength[[0,1,4]].tolist(),frequency=[7,5],damping=0.)
    singular=strength.clone();singular[0]=0.
    check('singular_null_mode_rejected_without_silent_ridge',rejected(lambda:build(p,constant,beta,constant_W,singular,0.)))
    extra_active_cfg=copy.deepcopy(cfg);extra_active_cfg['active_species']=[True,False,True]
    extra_active=SpectralPhysics(extra_active_cfg,dtype=torch.float64)
    check('bad_weights_damping_and_active_count_rejected',rejected(lambda:build(p,state,beta,-W,strength,damping))
        and rejected(lambda:build(p,state,beta,W,strength,-1.))
        and rejected(lambda:build(extra_active,state,beta,W,strength,damping)))

    #24 real coordinates: six low modes + three real spatial modes x3 fields.
    # Each nonzero frequency supplies cosine/sine, so choose three frequency
    # pairs:6+18=24, safely below the100-coordinate dense-reference ceiling.
    Qrows=list(Qlow)
    for kx,ky in ((1,0),(1,1),(7,5)):
        for trigonometric in (torch.cos,torch.sin):
            spatial=math.sqrt(2/1024)*trigonometric(2*torch.pi*(kx*xx+ky*yy)/32)
            for index in (0,1,4):
                z=torch.zeros_like(state);z[:,index]=spatial
                Qrows.append(pack(z,torch.zeros_like(beta))[0])
    Q=torch.stack(Qrows);dimension=Q.shape[0]
    check('dense_reference_bounded_to24_orthonormal_tangent_coordinates',dimension==24
        and torch.allclose(Q@Q.T,torch.eye(dimension,dtype=state.dtype),atol=1e-13,rtol=0.),
        dense_reference_dimension=dimension,full_active_dimension=3075)
    def true_normal(vector):
        dz,d_beta=unpack(vector,(32,32))
        forward=p.jvp(state,beta,dz,d_beta)
        g,gb=p.vjp(state,beta,W*forward)
        g+=strength[None,:,None,None]*laplacian(dz);g[:,2:4]=0.
        return pack(g,gb)+damping*scaling*vector
    def restricted_normal(vector): return true_normal(vector@Q)@Q.T
    def restricted_pre(vector): return precondition(vector@Q)@Q.T
    dense=torch.stack([restricted_normal(torch.eye(dimension,dtype=state.dtype)[i:i+1])[0]
                       for i in range(dimension)],dim=1)
    dense_symmetry=float((dense-dense.T).abs().max())/max(float(dense.abs().max()),1.)
    known=torch.randn((1,dimension),dtype=state.dtype,generator=generator)
    target=restricted_normal(known)
    reference=torch.linalg.solve(dense,target.T).T
    answer,pcg_receipt=pcg(restricted_normal,target,restricted_pre,rtol=1e-11,max_iterations=80)
    actual_residual=float(torch.linalg.vector_norm(restricted_normal(answer)-target)/torch.linalg.vector_norm(target))
    solution_error=float(torch.linalg.vector_norm(answer-reference)/torch.linalg.vector_norm(reference))
    old_pre,old_scale=block_preconditioner(p,state,beta,W,
        state.new_tensor([1.,1.,0.,0.,1.]),strength,damping)
    old_answer,old_receipt=pcg(restricted_normal,target,lambda v:old_pre(v@Q)@Q.T,
        rtol=1e-11,max_iterations=80)
    check('true_tangent_PCG_matches_dense24_reference',dense_symmetry<1e-13
        and pcg_receipt['converged'] and actual_residual<=1e-11 and solution_error<2e-7,
        dense_matrix_self_adjoint_relative_error=dense_symmetry,
        true_relative_residual=actual_residual,dense_reference_solution_relative_error=solution_error,
        PCG=pcg_receipt)
    check('old_and_new_linear_comparison_keeps_one_fixed_true_system',torch.equal(scaling,pack(d,db))
        and before==float(objective(p,raw,state,beta,W,strength)[0]),
        old_PCG=old_receipt,new_PCG=pcg_receipt,
        new_iterations=pcg_receipt['iterations'],old_iterations=old_receipt['iterations'],
        fixture_iteration_benefit=pcg_receipt['iterations']<old_receipt['iterations'],
        scope='fixed synthetic24-dimensional tangent fixture; not a real-data speed claim')
    frozen_after={name:s.sha(s.ROOT/name) for name in registration['source_sha256']}
    check('all26_v001_registered_sources_unchanged',len(frozen_before)==26
        and frozen_before==frozen_after and all(frozen_after[k]==v for k,v in registration['source_sha256'].items()))
    result={'schema':'v3-coupled-preconditioner-contracts-v1','time':s.now(),
        'all_passed':all(row['passed'] for row in tests),'tests':tests,'helper_version':VERSION,
        'CPU_shape':[32,32],'dense_reference_dimension_max':dimension,
        'JVP_calls':jvp_calls[0],'VJP_calls':vjp_calls[0],
        'whole_inverse_calls':0,'outer_optimizer_runs':0,'formal_experiment_started':False,
        'GPU_used':False,'scientific_raw_or_final_payloads_read':False,
        'truth_inputs_used':False,'coverage_accuracy_or_native_speed_claimed':False,
        'fixture_damping':damping,'damping_or_prior_tuned_after_results':False,
        'metadata':meta,'constant_state_metadata':const_meta,
        'registered_source_sha256':frozen_after,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in
            ('research_closeout/coupled_preconditioner.py','research_closeout/tests/coupled_preconditioner.py')},
        'CPU_wall_s':time.perf_counter()-started}
    folder=s.REPORTS/'repair_001/v3_preconditioner_001'
    folder.mkdir(parents=True,exist_ok=True)
    number=len(list(folder.glob('CPU_TEST_ATTEMPT_*.json')))+1
    s.write(folder/f'CPU_TEST_ATTEMPT_{number:03d}.json',s.plain(result))
    if result['all_passed']:
        # Every immutable attempt remains available; this task-local pointer
        # records the latest fully passed source hashes.
        s.write(folder/'CONTRACTS.json',s.plain(result),replace=(folder/'CONTRACTS.json').exists())
    print(json.dumps({'all_passed':result['all_passed'],'tests':len(tests),
        'failures':[row for row in tests if not row['passed']],
        'linear_comparison':{'new':pcg_receipt,'old':old_receipt},
        'report_folder':str(folder),'CPU_wall_s':result['CPU_wall_s']},indent=2),flush=True)
    return result


if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
