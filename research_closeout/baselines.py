"""Raw-only frozen v3, old B4 and explicitly named traditional configurations."""
from . import storage as s
from . import measurement as m
import copy
import time
import torch

def from_state(p,raw,state,beta,info):
    from training_v4.model import physical_output
    raw_e=p.raw_to_electrons(m.raw_tensor(raw,p))
    out=physical_output(p,state,beta,raw_e)
    out.update(template_postblur_e_ref=out['template_e'],predicted_raw_e=out['expected_e'],
        continuum_postblur_bands=out['continuum_images_e'],info=info,
        numerical_valid=bool(info.get('numerical_certified',False)),
        calibration_status='calibration_assumed_not_verified',low_signal=out['continuum_images_e'][:,-1]<=1)
    return out

class FrozenBaselines:
    def __init__(self,device='cuda'):
        self.p=m.physics(device,dtype=torch.float32);self.selected=m.selection();self.old_model=None
    @torch.no_grad()
    def v3(self,raw,*,physics=None,solver_config=None):
        from training_v4.scalar_engine import invert_single
        p=physics or self.p;cfg=copy.deepcopy(solver_config or self.selected['solver_config'])
        state,beta,history,seconds=invert_single(p,p.raw_to_electrons(m.raw_tensor(raw,p)),self.selected['method'],cfg)
        return from_state(p,raw,state,beta,{'method':'v3_physics','status':'FINITE_FROZEN_ITERATION_OUTPUT',
            'numerical_certified':False,'solver_config':cfg,'inverse_history':history,'seconds':seconds,
            'numerical_scope':'Finite estimator. Production differences are evaluated externally; CG success alone is insufficient.'})
    @torch.no_grad()
    def old_B4(self,raw):
        from infer import load_model,CHECKPOINT_SHA256
        if self.old_model is None:
            checkpoint=s.LEGACY/'checkpoints/selected_checkpoint.pt'
            assert s.sha(checkpoint)==CHECKPOINT_SHA256
            self.old_model=load_model(checkpoint,device=self.p.M.device)
        p=self.old_model.physics;raw_e=p.raw_to_electrons(m.raw_tensor(raw,p));started=time.perf_counter()
        out=self.old_model.infer_electrons(raw_e,use_prior=True)
        return from_state(p,raw,out['state'],out.get('background_beta',out.get('beta')),
            {'method':'old_B4','status':'FINITE_FROZEN_OUTPUT','numerical_certified':False,
             'checkpoint_sha256':CHECKPOINT_SHA256,'seconds':time.perf_counter()-started,'training':False})
    @torch.no_grad()
    def traditional(self,raw,regularization):
        from evaluation_v2.methods import demod_fit
        p=self.p;raw_e=p.raw_to_electrons(m.raw_tensor(raw,p));started=time.perf_counter()
        out=demod_fit(p,raw_e,float(regularization))
        # Recover the already fitted continuum column for resolution analysis;
        # it is in the postblur domain. Never add a second object blur here.
        C,_=p.continuum(out['temperature_color_K']);norm=C.square().sum(1,keepdim=True).sqrt().clamp_min(1)
        design=torch.stack([C/norm,(p.M[:,2]*p.Sscale)[None,:,None,None].expand_as(C),
            (p.q*p.Bscale)[None,:,None,None].expand_as(C)],dim=-1).permute(0,2,3,1,4)
        gram=design.transpose(-2,-1)@design
        gram+=torch.diag_embed(gram.diagonal(dim1=-2,dim2=-1).clamp_min(1)*1e-5)
        rhs=design.transpose(-2,-1)@out['decoded_bands_e'].permute(0,2,3,1).unsqueeze(-1)
        coefficients=torch.linalg.solve(gram,rhs).squeeze(-1)
        Cpost=C/norm*coefficients[...,0][:,None]
        out.update(template_postblur_e_ref=out['species_signed_e'][:,2],
            predicted_raw_e=out['expected_e'],continuum_postblur_bands=Cpost,
            numerical_valid=False,calibration_status='calibration_assumed_not_verified',
            low_signal=Cpost[:,-1]<=1,info={'method':f'B1_lambda{regularization:g}',
                'regularization':float(regularization),'seconds':time.perf_counter()-started,
                'status':'FINITE_FROZEN_OUTPUT','numerical_certified':False,
                'target_domain':'postblur local band fit; no second object blur'})
        return out
