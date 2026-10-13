"""Same v4 model and unchanged residual features for the initial two controls."""
import torch
from training_v4.model import PhysicsPostInverseCNN, supervised_loss
from training_v4_1.common import PROTOCOL

def protective_loss(model, raw, prediction, truth, physical_score, safe_weight):
    main, terms = supervised_loss(model, raw, prediction, truth)
    s_mse = (prediction['template_e']-truth['template_e']).square().mean()
    t_mse = (prediction['temperature_color_K']-truth['temperature_color_K']).square().mean()
    s_base, t_base = physical_score['template_MSE_e2'], physical_score['temperature_MSE_K2']
    safe_s = (s_mse-s_base-PROTOCOL['safe_template_tolerance_e2']).relu()/(s_base+1.)
    safe_t = (t_mse-t_base-PROTOCOL['safe_temperature_tolerance_K2']).relu()/(t_base+1.)
    return main+safe_weight*(safe_s+safe_t), {**terms, 'main': main, 'safe_template': safe_s,
        'safe_temperature': safe_t, 'safe_total': safe_s+safe_t}

class AmplitudeFeaturesCNN(PhysicsPostInverseCNN):
    """Conditional, single feature ablation; never used for safe1/safe5."""
    def __init__(self, cfg, reference_rms):
        super().__init__(cfg); self.reference_rms = reference_rms
    def features(self, raw_e, state, beta, stage):
        p = self.physics; expected = p(state, beta)
        variance = expected.detach().clamp_min(0)+p.extra_variance; weights = variance.reciprocal()
        residual = weights*(expected-raw_e)
        g, _ = p.vjp(state, beta, residual); d, _ = p.preconditioner_diagonal(state, beta, weights)
        bands = p.op.images_adjoint(residual)/self.reference_rms
        noise = variance.sqrt()/self.cfg['full_well_e']**.5
        index = torch.full_like(raw_e[:, None], stage/5)
        return torch.cat([state, (g/d).clamp(-5,5), bands.clamp(-5,5), noise[:,None], index],1)

def new_model(selection, variant='unchanged', reference_rms=None):
    if variant == 'unchanged': return PhysicsPostInverseCNN(selection['physics_config']).cuda()
    assert variant == 'amplitude' and reference_rms > 0
    return AmplitudeFeaturesCNN(selection['physics_config'], reference_rms).cuda()
