"""Frozen, measurement-derived attenuation of a post-inverse CNN proposal.

This is an empirical correction rule, not a proof of observable no-harm.
The coefficient uses only the original raw and the frozen physical state.
"""
import math
import runtime
import torch
from training_v4.model import PhysicsPostInverseCNN,physical_output
from training_v4.common import read,write
from training_v4_1.common import OUT as OLD,DATA as OLD_DATA
from evaluation_v2.common import canonical_hash

ROOT=runtime.ROOT/'reports/training_v4_3/run_001/residual_scaled'


def calibrate():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'coefficient_calibration.json'
    if path.exists(): return read(path)
    rows=[r for r in read(OLD/'data_manifest.json')['rows'] if r['split']=='train']
    records=[]; small=[]; noisy=[]
    for row in rows:
        score=read(OLD_DATA/'physical_scores'/(row['case_id']+'.json'))
        assert score['raw_hash']==row['raw_hash']
        rho=score['weighted_raw_RMS']; assert math.isfinite(rho) and rho>=0
        (noisy if row['input_kind']=='noise' else small).append(rho)
        records.append({'case_id':row['case_id'],'raw_hash':row['raw_hash'],'cache_key':score['cache_key'],
                        'training_input_kind':row['input_kind'],'weighted_raw_RMS':rho})
    protected_max,noisy_min=max(small),min(noisy)
    assert 0<protected_max<noisy_min,'No training residual separation; do not tune against validation truth'
    tau=math.sqrt(protected_max*noisy_min); epsilon=1e-4
    power=max(2,math.ceil(math.log(epsilon/(1-epsilon))/math.log(protected_max/tau)))
    def coefficient(r):
        ratio=(r/tau)**power
        return ratio/(1+ratio)
    assert coefficient(protected_max)<=epsilon and coefficient(noisy_min)>=1-epsilon
    value={'tau':tau,'power':power,'design_leakage_target':epsilon,
        'training_protection_max_RMS':protected_max,'training_noise_min_RMS':noisy_min,
        'protected_count':len(small),'noise_count':len(noisy),'training_cases':len(rows),
        'protected_upper_coefficient':coefficient(protected_max),'noise_lower_coefficient':coefficient(noisy_min),
        'formula':'alpha = sigmoid(power * log(rho / tau)); rho from cached physical raw residual',
        'calibration_inputs':'training raw/cached-state residuals and training noise-kind metadata only; no truth MSE or validation scores',
        'inference_inputs':'raw_dn,cached_state,cached_beta only',
        'records_sha256':canonical_hash(records),'data_manifest_sha256':runtime.sha256(OLD/'data_manifest.json'),
        'local_only':True,'validation_motivated_diagnostic':True,'final_test_used':False}
    write(ROOT/'calibration_training_residuals.json',records); write(path,value)
    return value


class ResidualScaledCNN(PhysicsPostInverseCNN):
    def __init__(self,cfg,calibration):
        super().__init__(cfg)
        self.gate_tau=float(calibration['tau']); self.gate_power=int(calibration['power'])
        assert self.gate_tau>0 and self.gate_power>=2

    def coefficient(self,raw_e,cached_state,cached_beta):
        expected=self.physics(cached_state.detach(),cached_beta.detach())
        variance=raw_e.clamp_min(0)+self.physics.extra_variance
        rho=((expected-raw_e).square()/variance).mean((-2,-1)).sqrt().detach()
        alpha=torch.sigmoid(self.gate_power*(rho.clamp_min(1e-20).log()-math.log(self.gate_tau)))
        return alpha,rho

    def forward_electrons(self,raw_e,cached_state,cached_beta):
        base,beta=cached_state.detach(),cached_beta.detach()
        alpha,rho=self.coefficient(raw_e,base,beta)
        # The six-stage CNN and its physical features remain unchanged.
        proposal=super().forward_electrons(raw_e,base,beta)
        state=base+alpha[:,None,None,None]*(proposal['state']-base)
        result=physical_output(self.physics,state,beta,raw_e)
        result.update(correction_coefficient=alpha,initial_weighted_raw_RMS=rho,
                      unrestricted_proposal_state=proposal['state'])
        return result
