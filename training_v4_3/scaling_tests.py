import inspect
import torch
from config import default_config
from training_v4.model import physical_output,supervised_loss
from training_v4_1.common import setup
from training_v4_3.residual_scaling import ResidualScaledCNN,calibrate


def test_training_only_calibration_and_coefficient_bounds():
    c=calibrate()
    assert c['training_cases']==2576 and c['protected_count']==480 and c['noise_count']==2096
    assert c['protected_upper_coefficient']<=1e-4 and c['noise_lower_coefficient']>=1-1e-4
    assert not c['final_test_used']


def test_exactly_fitted_input_preserves_signed_physics_with_nonzero_CNN_heads():
    setup(); model=ResidualScaledCNN(default_config(64),calibrate()).cuda().eval()
    assert sum(p.numel() for p in model.parameters())==5797
    assert list(inspect.signature(model.forward).parameters)==['raw_dn','cached_state','cached_beta']
    z=torch.zeros(1,5,64,64,device='cuda'); z[:,1]=-.3; z[:,4]=-.12
    beta=z.new_tensor([[.4,.02,-.03]])
    with torch.no_grad():
        model.emission_head.bias[2]=.1; model.continuum_head.bias[0]=.1
        raw=model.physics(z,beta); base=physical_output(model.physics,z,beta,raw)
        out=model.forward_electrons(raw,z,beta)
    assert out['correction_coefficient'].item()==0
    assert not torch.equal(out['unrestricted_proposal_state'],z)
    assert all(torch.equal(out[k],base[k]) for k in ('state','template_e','temperature_color_K','band_images_e','expected_e'))
    assert out['template_e'].mean()<0,'Signed template must remain nonzero and negative'


def test_noisy_input_keeps_live_physical_output_to_CNN_gradient_path():
    setup(); torch.manual_seed(73)
    model=ResidualScaledCNN(default_config(64),calibrate()).cuda().train()
    z=torch.zeros(1,5,64,64,device='cuda'); z[:,1]=-.3; z[:,4]=-.12
    beta=z.new_tensor([[.4,.02,-.03]])
    with torch.no_grad():
        expected=model.physics(z,beta)
        raw=expected+torch.randn_like(expected)*(expected.clamp_min(0)+model.physics.extra_variance).sqrt()
        dn=raw/model.physics.adc_step+model.cfg['black_level_DN']
        truth=physical_output(model.physics,z,beta,expected)
        truth={k:truth[k].clone() for k in ('template_e','temperature_color_K','band_images_e')}
        truth['template_e']-=3; truth['temperature_color_K']+=1
    out=model(dn,z,beta)
    assert .999<out['correction_coefficient'].item()<=1
    loss,_=supervised_loss(model,dn,out,truth); loss.backward()
    assert model.emission_head.weight.grad.norm()>0 and model.continuum_head.weight.grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
