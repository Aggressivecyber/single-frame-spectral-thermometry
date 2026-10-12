"""Meaningful tensor, gradient, identity and per-frame solver checks."""
import runtime
import copy
import torch
import pytest
from config import default_config
from physics import SpectralPhysics
from training_v4.model import PhysicsPostInverseCNN, physical_output, supervised_loss
from training_v4.scalar_engine import invert_single
from training_v4.common import parameters_hash
from repair_v3.solver import DEFAULTS, joint_data_step, consistent_initialize


def sample(device='cuda', dtype=torch.float32):
    p = SpectralPhysics(default_config(64), dtype).to(device)
    xx, yy = p.background_basis[1], p.background_basis[2]
    z = torch.zeros((1, 5, 64, 64), device=device, dtype=dtype)
    z[:, 0] = .1*torch.sin(2*xx)
    z[:, 1] = -.4+.02*yy
    z[:, 4] = 2*torch.exp(-8*(xx.square()+yy.square()))
    beta = z.new_tensor([[.4, .04, -.03]])
    raw = p(z, beta)
    return p, z, beta, raw


def test_zero_heads_are_identity_at_six_stages_and_parameters_match_old_size():
    torch.manual_seed(17)
    p, state, beta, raw = sample()
    model = PhysicsPostInverseCNN(p.cfg).cuda().eval()
    dn = raw/p.adc_step+p.cfg['black_level_DN']
    with torch.no_grad():
        before = physical_output(model.physics, state, beta, model.physics.raw_to_electrons(dn))
        after = model(dn, state, beta)
    assert sum(t.numel() for t in model.parameters()) == 5797
    for key in ('state', 'beta', 'template_e', 'temperature_color_K', 'band_images_e', 'expected_e'):
        assert torch.equal(before[key], after[key])


@pytest.mark.parametrize('checkpointed', [False, True])
def test_physics_to_active_head_gradients_and_real_adam_update(checkpointed):
    torch.manual_seed(19)
    p, state, beta, raw = sample()
    model = PhysicsPostInverseCNN(p.cfg, activation_checkpointing=checkpointed).cuda().train()
    observed = raw+8*torch.cos(p.background_basis[1]*23)
    dn = observed/p.adc_step+p.cfg['black_level_DN']
    initial = state.clone()
    initial[:, 4] += .3*torch.sin(14*p.background_basis[1])
    initial[:, 0] += .01
    truth_output = physical_output(p, state, beta, raw)
    truth = {k: truth_output[k].detach() for k in ('template_e', 'temperature_color_K', 'band_images_e')}
    before = parameters_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    out = model(dn, initial, beta)
    loss, _ = supervised_loss(model, dn, out, truth)
    loss.backward()
    assert model.continuum_head.weight.grad.abs().sum() > 0
    assert model.emission_head.weight.grad[2].abs().sum() > 0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    # With zero heads the first trunk gradient is correctly zero, not a defect.
    assert all(t.grad is not None and torch.count_nonzero(t.grad) == 0 for t in model.trunk.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    assert parameters_hash(model) != before
    optimizer.zero_grad()
    second, _ = supervised_loss(model, dn, model(dn, initial, beta), truth)
    second.backward()
    assert sum(float(t.grad.abs().sum()) for t in model.trunk.parameters()) > 0


@pytest.mark.parametrize('method', ['old_init__joint_update_v3', 'consistent_init__joint_update_v3'])
def test_scalar_cached_linearization_matches_v3_in_float64(method):
    p, _, _, raw = sample(dtype=torch.float64)
    config = copy.deepcopy(DEFAULTS)
    config.update(warm_iterations=3, outer_iterations=2, cg_max_iterations=15)
    inputs = torch.cat([raw, raw*1.04])
    with torch.no_grad():
        for i in range(2):
            z, b, _, _ = invert_single(p, inputs[i:i+1], method, config)
            if method.startswith('consistent'):
                zs, bs, _ = consistent_initialize(p, inputs[i:i+1], config)
            else:
                zs, bs = p.initialize(inputs[i:i+1])
            damping = config['LM_initial']
            for _ in range(2):
                zs, bs, info = joint_data_step(p, inputs[i:i+1], zs, bs, config, damping)
                damping = info['next_damping']
                if not info['accepted']:
                    break
            torch.testing.assert_close(z, zs, atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(b, bs, atol=1e-7, rtol=1e-7)


def test_noise_change_requires_another_cache_and_labels_are_absent_from_key():
    import inspect
    import numpy as np
    from evaluation_v2.common import array_hash, canonical_hash
    from training_v4.cache import cache_identity, cache_raw
    cfg = default_config(64)
    a = np.full((64, 64), 1200, dtype=np.uint16)
    b = a.copy()
    b[17, 23] += 1
    key1 = cache_identity(array_hash(a), cfg, canonical_hash(cfg), 'old_init__joint_update_v3', DEFAULTS)
    key2 = cache_identity(array_hash(b), cfg, canonical_hash(cfg), 'old_init__joint_update_v3', DEFAULTS)
    assert key1 != key2
    different = copy.deepcopy(cfg)
    different['channel_gain'][5] = 1.03
    key3 = cache_identity(array_hash(a), different, canonical_hash(different), 'old_init__joint_update_v3', DEFAULTS)
    assert key1 != key3
    assert not any('label' in n or 'truth' in n for n in inspect.signature(cache_raw).parameters)
