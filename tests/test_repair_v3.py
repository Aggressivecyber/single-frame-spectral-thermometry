import inspect
import torch
import pytest
from config import default_config
from physics import SpectralPhysics
from solver import data_step
from repair_v3.solver import (DEFAULTS, pcg, regularization, linearization,
                              consistent_initialize, joint_data_step,
                              old_step_with_subblocks)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def small(device, dtype=torch.float64):
    p = SpectralPhysics(default_config(64), dtype).to(device)
    z = torch.zeros(1, 5, 64, 64, dtype=dtype, device=device)
    z[:, 0] = -.12+.08*torch.cos(torch.pi*p.background_basis[1])
    z[:, 1] = -.5
    z[:, 4] = .6*torch.cos(torch.pi*p.background_basis[2])
    b = z.new_tensor([[.1, .03, -.02]])
    return p, z, b


def test_cg_stops_on_relative_true_residual():
    diagonal = torch.tensor([[1., 3., 11., 37.]], dtype=torch.float64)
    rhs = torch.tensor([[4., -2., 3., 9.]], dtype=torch.float64)
    x, info = pcg(lambda v: diagonal*v, rhs, lambda r: r, rtol=1e-10, max_iterations=30)
    assert info['converged'] and info['iterations'] < 30
    assert info['relative_residual'] <= 1e-10
    torch.testing.assert_close(x, rhs/diagonal)
    _, capped = pcg(lambda v: diagonal*v, rhs, lambda r: r, rtol=1e-15, max_iterations=1)
    assert not capped['converged'] and capped['stop'] == 'MAX_ITERATIONS'


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('warm', [False, True])
def test_scaled_jvp_vjp_are_adjoint_with_object_blur(device, warm):
    p, z, b = small(device)
    j, v, mask = linearization(p, z, b, warm=warm)
    gen = torch.Generator(device=device).manual_seed(912)
    direction = torch.randn(1, z.numel()+3, device=device, dtype=z.dtype, generator=gen)
    residual = torch.randn((1, 2, 64, 64) if warm else (1, 64, 64),
                           device=device, dtype=z.dtype, generator=gen)
    a = (j(direction)*residual).sum()
    c = (direction*v(residual)).sum()
    assert float((a-c).abs()/(a.abs()+c.abs()).clamp_min(1e-20)) < 2e-12


@pytest.mark.parametrize('device', DEVICES)
def test_instrumented_old_blocks_preserve_exact_old_result(device):
    p, z, b = small(device, torch.float32)
    raw = p(z, b)
    z0, b0 = p.initialize(raw)
    za, ba = data_step(p, raw, z0, b0)
    zb, bb, trace = old_step_with_subblocks(p, raw, z0, b0)
    assert torch.equal(za, zb) and torch.equal(ba, bb)
    assert [step[2] for step in trace] == ['old_spectral_subblock', 'old_temperature_subblock']


@pytest.mark.parametrize('device', DEVICES)
def test_continuum_warm_start_fits_postblur_ir_without_truth_interface(device):
    p, z, b = small(device)
    raw = p(z, b)
    z0, b0 = p.initialize(raw)
    target = p.op.decode(raw)[:, 5:7]
    before = (p.op.object_blur(p.components(z0, b0)[0])[:, 5:7]-target).square().mean()
    config = dict(DEFAULTS, warm_iterations=8, cg_max_iterations=80)
    repaired, beta, history = consistent_initialize(p, raw, config)
    after = (p.op.object_blur(p.components(repaired, beta)[0])[:, 5:7]-target).square().mean()
    assert after < before*.02
    assert torch.equal(repaired[:, 2:], z0[:, 2:]) and torch.equal(beta, b0)
    assert all(h['objective_after'] <= h['objective_before']+1e-8 for h in history)
    assert 'truth' not in inspect.signature(consistent_initialize).parameters


@pytest.mark.parametrize('device', DEVICES)
def test_joint_acceptance_includes_regularizer_and_excludes_lm_penalty(device):
    p, z, b = small(device)
    raw = p(z, b)
    state, beta = p.initialize(raw)
    strength = [1., .2, 0., 0., .3]
    config = dict(DEFAULTS, regularization=strength, cg_max_iterations=80)
    updated, bb, info = joint_data_step(p, raw, state, beta, config)
    assert info['accepted'] and info['actual_predicted_ratio'] >= config['acceptance_ratio_minimum']
    weights = (raw.clamp_min(0)+p.extra_variance).reciprocal()
    cost = .5*((p(updated, bb)-raw).square()*weights).sum()
    penalty, gradient = regularization(updated, updated.new_tensor(strength))
    assert abs(info['objective_after']-float(cost+penalty)) < 1e-7
    assert info['objective_after'] < info['objective_before']
    assert info['predicted_reduction'] > 0 and info['CG']['relative_residual'] is not None


def test_regularizer_gradient_and_signed_template_are_not_zero_or_relu():
    p, z, b = small('cpu')
    z[:, 4] -= 1.
    constrained, _ = p.constrain(z, b)
    assert constrained[:, 4].min() < 0
    strength = z.new_tensor(DEFAULTS['regularization'])
    x = z.clone().requires_grad_(True)
    cost, gradient = regularization(x, strength)
    observed, = torch.autograd.grad(cost, x)
    torch.testing.assert_close(observed, gradient, atol=1e-12, rtol=1e-12)
    assert 'truth' not in inspect.signature(joint_data_step).parameters


def test_raw_interface_and_whole_frame_saturation_rejection():
    import numpy as np
    from evaluation_v2.common import MeasurementInput
    from repair_v3.infer import PhysicalReconstructor
    p, _, _ = small('cpu')
    recon = PhysicalReconstructor(p)
    with pytest.raises(TypeError):
        recon.reconstruct({'raw_dn': np.zeros((64, 64)), 'truth': np.zeros((64, 64))})
    raw = np.full((64, 64), 1000.)
    raw[12, 13] = 4095
    p.initialize = lambda *args: pytest.fail('Saturated input entered initialization')
    result = recon.reconstruct(MeasurementInput(raw))
    assert result['status'] == 'SATURATION_REJECTED'


def test_original_physics_buffers_stay_unchanged_during_repair():
    from evaluation_v2.common import tensor_digest
    p, z, b = small('cpu')
    before = tensor_digest(p.named_buffers())
    raw = p(z, b)
    config = dict(DEFAULTS, warm_iterations=2, cg_max_iterations=12)
    start, beta, _ = consistent_initialize(p, raw, config)
    joint_data_step(p, raw, start, beta, config)
    assert tensor_digest(p.named_buffers()) == before
    # Warm start explicitly predicts postblur IR bands; no inverse OTF path.
    assert 'object_otf' not in inspect.getsource(consistent_initialize)
