"""Coupled Fourier approximation for unchanged v3 tangent step systems.

This is an SPD preconditioner on the three active fields and three fixed beta
coefficients, not the inverse of the spatially varying exact Hessian. The caller
must continue to use the exact JVP/VJP, original prior, bounds and objective.
"""
from . import storage as _storage
from solver import pack, unpack
import hashlib
import json
import math
import torch

VERSION='v3-coupled-fourier-preconditioner-v2-batch64'
BATCH_LIMIT=64


def eigvalsh_bounded(block):
    """Exactly the same eigenproblem, dispatched in batches of at most64."""
    n=block.shape[-1];flat=block.reshape(-1,n,n)
    result=torch.empty((len(flat),n),device=block.device,dtype=block.real.dtype)
    for first in range(0,len(flat),BATCH_LIMIT):
        last=min(first+BATCH_LIMIT,len(flat))
        result[first:last]=torch.linalg.eigvalsh(flat[first:last])
    return result.reshape(*block.shape[:-2],n)


def cholesky_ex_bounded(block):
    """No ridge, clipping or change of block normalization."""
    n=block.shape[-1];flat=block.reshape(-1,n,n)
    result=torch.empty_like(flat);info=torch.empty(len(flat),device=block.device,dtype=torch.int32)
    for first in range(0,len(flat),BATCH_LIMIT):
        last=min(first+BATCH_LIMIT,len(flat))
        result[first:last],info[first:last]=torch.linalg.cholesky_ex(flat[first:last])
    return result.reshape(block.shape),info.reshape(block.shape[:-2])


def cholesky_solve_bounded(rhs,factor):
    """Bound all broadcast batch dimensions, including native-grid apply."""
    n=factor.shape[-1];columns=rhs.shape[-1]
    batch=torch.broadcast_shapes(rhs.shape[:-2],factor.shape[:-2])
    flat_rhs=rhs.expand(*batch,n,columns).reshape(-1,n,columns)
    flat_factor=factor.expand(*batch,n,n).reshape(-1,n,n)
    result=torch.empty_like(flat_rhs)
    for first in range(0,len(flat_rhs),BATCH_LIMIT):
        last=min(first+BATCH_LIMIT,len(flat_rhs))
        result[first:last]=torch.cholesky_solve(flat_rhs[first:last],flat_factor[first:last])
    return result.reshape(*batch,n,columns)


def _digest(value):
    a=value.detach().contiguous().cpu().numpy()
    h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _factor(block,name):
    """Diagonal equilibration changes arithmetic, not the represented block."""
    diagonal=block.diagonal(dim1=-2,dim2=-1)
    if not torch.isfinite(block).all() or not bool((diagonal>0).all()):
        raise ValueError(name+' has a nonfinite or nonpositive diagonal')
    root=diagonal.sqrt()
    normalized=block/root[..., :,None]/root[..., None,:]
    factor,info=cholesky_ex_bounded(normalized)
    if bool((info!=0).any()):
        raise ValueError(name+' is not positive definite; use declared numerical damping for a singular tangent system')
    return root,factor


@torch.no_grad()
def build(p,state,beta,W,strength,damping):
    """Return (precondition, original_diagonal_scaling, JSON-safe metadata).

    ``precondition`` consumes/returns solver.pack's B=1 five-field-plus-beta
    vector. Inactive fields are projected to zero, so positivity is asserted
    only on the active subspace. ``scaling`` is the original data diagonal D;
    a future exact step system must use the same damping*D, not a new prior.
    No optimizer, inverse, truth input or file read occurs in this function.
    """
    shape=tuple(p.cfg['shape']);h,w=shape;count=h*w
    if state.shape!=(1,5,h,w) or beta.shape!=(1,3):
        raise ValueError('One five-field frame and exactly three beta coefficients are required')
    if state.dtype!=torch.float64 or beta.dtype!=state.dtype or p.M.dtype!=state.dtype:
        raise ValueError('The coupled v3 preconditioner requires float64 physics/state/beta')
    if state.device!=p.M.device or beta.device!=state.device:
        raise ValueError('Physics, state and beta must share one device')
    if not isinstance(damping,(int,float)) or isinstance(damping,bool) or not math.isfinite(damping) or damping<0:
        raise ValueError('Numerical damping must be finite and nonnegative')
    weights=torch.as_tensor(W,device=state.device,dtype=state.dtype).detach()
    if weights.shape==shape: weights=weights[None]
    prior=torch.as_tensor(strength,device=state.device,dtype=state.dtype).detach()
    if weights.shape!=(1,h,w) or prior.shape!=(5,):
        raise ValueError('W must be one frame and strength must contain the original five prior coefficients')
    if not all(bool(torch.isfinite(v).all()) for v in (state,beta,weights,prior)):
        raise ValueError('All linearization inputs must be finite')
    if not bool((weights>0).all()) or not bool((prior>=0).all()):
        raise ValueError('Positive data weights and nonnegative original prior coefficients are required')
    indices=[0,1]+[2+i for i,v in enumerate(p.active) if bool(v)]
    if len(indices)!=3:
        raise ValueError('This version supports exactly one active signed species (three active fields)')

    diagonal_fields,diagonal_beta=p.preconditioner_diagonal(state,beta,weights)
    scaling=pack(diagonal_fields,diagonal_beta).detach().clone()
    field_D=diagonal_fields[0,indices].mean((-2,-1))
    beta_D=diagonal_beta[0]
    if not bool(torch.isfinite(scaling).all()) or not bool((scaling>0).all()):
        raise ValueError('The original damping diagonal must be finite and positive')
    jac=p.spectral_jacobian(state,beta)[0,indices].mean((-2,-1))
    bands=jac.shape[1]
    if bands!=len(p.cfg['carrier_bins_xy']) or bands!=len(p.cfg['channel_gain']):
        raise ValueError('Spectral Jacobian and carrier calibration disagree')
    H2=p.op.object_otf.abs().square()
    R2=p.op.relay_otf.abs().square()
    if H2.shape!=shape or R2.shape!=shape:
        raise ValueError('Native optical transfer shapes disagree')
    gains=state.new_tensor(p.cfg['channel_gain'])
    depths=state.new_tensor(p.cfg['modulation_depth'])
    dc=(jac*(gains/2)[None]).sum(1)
    common_dc=R2[...,None,None]*torch.outer(dc,dc)
    symbol=common_dc.clone()
    for band,((kx,ky),gain,depth) in enumerate(zip(p.cfg['carrier_bins_xy'],gains,depths)):
        if not isinstance(kx,int) or not isinstance(ky,int):
            raise ValueError('The native Fourier carrier bins must be integers')
        sideband_power=(torch.roll(R2,(-ky,-kx),(-2,-1))+
                        torch.roll(R2,(ky,kx),(-2,-1)))
        symbol+=(gain*depth/4).square()*sideband_power[...,None,None]*torch.outer(jac[:,band],jac[:,band])
    symbol*=weights.mean()*H2[...,None,None]
    fy,fx=torch.meshgrid(torch.fft.fftfreq(h,device=state.device,dtype=state.dtype),
                         torch.fft.fftfreq(w,device=state.device,dtype=state.dtype),indexing='ij')
    lap=4*torch.sin(torch.pi*fx).square()+4*torch.sin(torch.pi*fy).square()
    symbol+=lap[...,None,None]*torch.diag(prior[indices])
    symbol+=float(damping)*torch.diag(field_D)
    # Explicitly pair k and -k. The resulting real-even SPD multiplier maps
    # Hermitian spectra to Hermitian spectra and is self-adjoint on real fields.
    neg_y=(-torch.arange(h,device=state.device))%h
    neg_x=(-torch.arange(w,device=state.device))%w
    reflected=symbol[neg_y[:,None],neg_x[None,:]]
    symmetry_error=float((symbol-reflected).abs().max())
    symbol=.5*(symbol+reflected)
    symbol=.5*(symbol+symbol.transpose(-1,-2))
    nonzero=symbol.reshape(-1,3,3)[1:]
    eigen=eigvalsh_bounded(nonzero)
    if not bool(torch.isfinite(eigen).all()) or not bool((eigen>0).all()):
        raise ValueError('A nonzero-frequency block is singular; no extra physical regularizer is added')
    # Zero frequency is represented by the six-dimensional exact Gram below.
    symbol[0,0]=torch.eye(3,device=state.device,dtype=state.dtype)
    field_root,field_factor=_factor(symbol,'Nonzero-frequency block')

    columns=[]
    for index in indices:
        direction=torch.zeros_like(state);direction[:,index]=count**-.5
        columns.append(p.jvp(state,beta,direction,torch.zeros_like(beta)).reshape(-1))
    for index in range(3):
        beta_direction=torch.zeros_like(beta);beta_direction[:,index]=1.
        columns.append(p.jvp(state,beta,torch.zeros_like(state),beta_direction).reshape(-1))
    columns=torch.stack(columns)
    gram=(columns*weights.reshape(-1)[None])@columns.T
    low_D=torch.cat([field_D,beta_D])
    low_block=.5*(gram+gram.T)+float(damping)*torch.diag(low_D)
    low_root,low_factor=_factor(low_block,'Field-zero-mode/beta block')
    field_factor=field_factor.to(torch.complex128)
    vector_shape=(1,5*count+3)

    @torch.no_grad()
    def precondition(vector):
        if not isinstance(vector,torch.Tensor) or vector.shape!=vector_shape:
            raise ValueError('Expected solver.pack vector with one five-field frame and three beta entries')
        if vector.device!=state.device or vector.dtype!=state.dtype or not bool(torch.isfinite(vector).all()):
            raise ValueError('Preconditioner vector must be finite and share the built dtype/device')
        fields,beta_rhs=unpack(vector,shape)
        spectrum=torch.fft.fft2(fields[:,indices],norm='ortho')
        rhs=spectrum.permute(0,2,3,1)/field_root[None]
        answer=cholesky_solve_bounded(rhs[...,None],field_factor[None])[...,0]/field_root[None]
        answer=answer.permute(0,3,1,2)
        low_rhs=torch.cat([spectrum[0,:,0,0].real,beta_rhs[0]])
        low_answer=cholesky_solve_bounded((low_rhs/low_root)[:,None],low_factor)[:,0]/low_root
        answer[0,:,0,0]=low_answer[:3].to(answer.dtype)
        result=torch.zeros_like(fields)
        result[:,indices]=torch.fft.ifft2(answer,norm='ortho').real
        return pack(result,low_answer[3:][None])

    metadata={'schema':VERSION,'backend_batch_limit':BATCH_LIMIT,'same_frequency_symbol_as_v1':True,'GPU_qualification':'requires separate native CUDA receipt','active_field_indices':indices,'packed_field_count':5,'beta_count':3,
        'shape':list(shape),'dtype':str(state.dtype),'device':str(state.device),
        'mean_spectral_jacobian':jac.cpu().tolist(),'mean_raw_weight':float(weights.mean()),
        'field_damping_diagonal_mean':field_D.cpu().tolist(),'beta_damping_diagonal':beta_D.cpu().tolist(),
        'original_prior_coefficients':prior.cpu().tolist(),'numerical_damping':float(damping),
        'scaling_definition':'original p.preconditioner_diagonal, packed without changing the prior or adding damping',
        'frequency_symbol':'mean(W)*|H(k)|^2*(|R(k)|^2*c_DC*c_DC^T + sum_band(gain*depth/4)^2*(|R(k+carrier)|^2+|R(k-carrier)|^2)*j_band*j_band^T) + Lambda*L(k) + damping*mean(D)',
        'shared_data_DC_cross_band_terms_retained':True,'relay_shifts_on_native_torus':True,
        'FFT_norm':'ortho','frequency_even_pairing_max_original_difference':symmetry_error,
        'nonzero_frequency_block_eigenvalue_min':float(eigen.min()),
        'nonzero_frequency_block_eigenvalue_max':float(eigen.max()),
        'object_OTF_zero_frequency_count':int((H2==0).sum()),
        'low_mode_basis':'three orthonormal constant fields (1/sqrt(HW)), then three original beta coordinate vectors',
        'low_mode_jvp_columns':6,'low_mode_undamped_Gram':gram.cpu().tolist(),
        'low_mode_block':low_block.cpu().tolist(),'low_mode_equilibrated_eigenvalues':eigvalsh_bounded(low_block/low_root[:,None]/low_root[None,:]).cpu().tolist(),
        'factorization':'per-block diagonal equilibration and Cholesky; no added ridge or eigenvalue clipping',
        'self_adjoint_positive_scope':'three active fields and all three beta coordinates; inactive field vectors map to zero',
        'bound_active_mask':'caller must restrict the exact step system and preconditioner to free variables',
        'exact_Hessian_inverse_claimed':False,'original_objective_changed':False,
        'physical_constraints_changed':False,'inference_reads_truth':False,
        'complete_inverse_calls':0,'full_parameter_linearization_frozen_at_build':True,
        'input_sha256':{name:_digest(value) for name,value in
            (('state',state),('beta',beta),('W',weights),('strength',prior),('mean_spectral_jacobian',jac))},
        'physics_config_sha256':hashlib.sha256(json.dumps(p.cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    return precondition,scaling,metadata
