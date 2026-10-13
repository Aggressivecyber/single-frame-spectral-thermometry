"""Small tensor checks for the final-continuum constraint; no optimizer steps.

Run from the repository with ``python -B -m training_v4_4.model_tests`` for CPU,
or pass ``--device cuda`` when a GPU run is separately authorized.  These checks
use synthetic 64 x 64 tensors and the existing frozen v4.3 calibration; they do
not load training, validation, or final-test cases or invoke calibration.
"""
import argparse
import inspect
import json
import time

import runtime
import torch

from config import default_config
from training_v4.model import physical_output
from training_v4_3.residual_scaling import ResidualScaledCNN
from training_v4_4.model import FixedContinuumCNN


PHYSICAL_FIELDS = (
    'state', 'beta', 'temperature_color_K', 'template_e', 'species_signed_e',
    'band_images_e', 'continuum_images_e', 'expected_e', 'raw_residual_e', 'valid',
)
CALIBRATION_PATH = (
    runtime.ROOT / 'reports/training_v4_3/run_001/residual_scaled'
    / 'coefficient_calibration.json'
)


def _pair(device, *, all_species=False):
    torch.manual_seed(4401)
    cfg = default_config(64)
    if all_species:
        cfg['active_species'] = [True, True, True]
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))
    original = ResidualScaledCNN(cfg, calibration).to(device).eval()
    # Nonzero heads exercise the constraint and the internal continuum path.
    with torch.no_grad():
        original.continuum_head.weight.uniform_(-.004, .004)
        original.emission_head.weight.uniform_(-.004, .004)
        original.continuum_head.bias.copy_(
            original.continuum_head.bias.new_tensor([.04, -.03]))
        original.emission_head.bias.copy_(
            original.emission_head.bias.new_tensor([-.02, .01, -.03]))
    fixed = FixedContinuumCNN(cfg, calibration).to(device).eval()
    fixed.load_cnn(original.cnn_state_dict())
    return original, fixed


def _inputs(model, *, batch=2, noisy=True):
    p = model.physics
    xx, yy = p.background_basis[1], p.background_basis[2]
    state = xx.new_zeros((batch, 5, *model.cfg['shape']))
    state[:, 0] = .04 * torch.sin(2 * xx)
    state[:, 1] = -.3 + .01 * torch.cos(2 * yy)
    state[:, 2] = -.04 * p.active[0]
    state[:, 3] = .03 * p.active[1]
    state[:, 4] = -.12 - .02 * torch.cos(2 * xx)
    if batch > 1:
        state[1, 4].neg_()
    beta = xx.new_tensor([[.4, .02, -.03]]).repeat(batch, 1)
    with torch.no_grad():
        raw = p(state, beta)
        if noisy:
            torch.manual_seed(4402)
            raw = raw + torch.randn_like(raw) * (
                raw.clamp_min(0) + p.extra_variance).sqrt()
    return raw, state, beta


def test_api_weights_and_parameter_count(device):
    original, fixed = _pair(device)
    assert sum(p.numel() for p in fixed.parameters()) == 5797
    assert dict(fixed.named_parameters()).keys() == dict(original.named_parameters()).keys()
    assert dict(fixed.named_buffers()).keys() == dict(original.named_buffers()).keys()
    assert inspect.signature(FixedContinuumCNN) == inspect.signature(ResidualScaledCNN)
    assert list(inspect.signature(fixed.forward).parameters) == [
        'raw_dn', 'cached_state', 'cached_beta']
    assert list(inspect.signature(fixed.forward_electrons).parameters) == [
        'raw_e', 'cached_state', 'cached_beta']
    assert FixedContinuumCNN.load_cnn is ResidualScaledCNN.load_cnn
    assert FixedContinuumCNN.coefficient is ResidualScaledCNN.coefficient
    assert all(torch.equal(value, original.state_dict()[key])
               for key, value in fixed.state_dict().items())
    assert all(p.requires_grad for p in fixed.continuum_head.parameters())


def test_same_weights_preserve_physical_continuum_and_scaled_signed(device):
    original, fixed = _pair(device, all_species=True)
    raw, state, beta = _inputs(fixed)
    untouched = [value.clone() for value in (raw, state, beta)]
    with torch.no_grad():
        baseline = physical_output(fixed.physics, state, beta, raw)
        control = original.forward_electrons(raw, state, beta)
        output = fixed.forward_electrons(raw, state, beta)
    assert (output['correction_coefficient'] > .99).all()
    assert not torch.equal(output['unrestricted_proposal_state'][:, :2], state[:, :2])
    assert not torch.equal(control['state'][:, :2], state[:, :2])
    assert torch.equal(output['state'][:, :2], state[:, :2])
    for key in ('temperature_color_K', 'continuum_images_e'):
        assert torch.equal(output[key], baseline[key]), key
    for key in ('template_e', 'species_signed_e', 'correction_coefficient',
                'initial_weighted_raw_RMS', 'unrestricted_proposal_state'):
        assert torch.equal(output[key], control[key]), key
    assert torch.equal(output['state'][:, 2:], control['state'][:, 2:])
    assert torch.equal(output['unrestricted_scaled_state'], control['state'])
    assert torch.equal(output['beta'], beta)
    # The two frames include a strictly negative and a strictly positive template.
    assert (output['template_e'][0] < 0).all()
    assert (output['template_e'][1] > 0).all()
    assert all(torch.equal(value, prior)
               for value, prior in zip((raw, state, beta), untouched))


def test_exact_zero_raw_residual_protects_signed_physics(device):
    _, fixed = _pair(device)
    raw, state, beta = _inputs(fixed, batch=1, noisy=False)
    with torch.no_grad():
        baseline = physical_output(fixed.physics, state, beta, raw)
        output = fixed.forward_electrons(raw, state, beta)
    assert output['initial_weighted_raw_RMS'].item() == 0
    assert output['correction_coefficient'].item() == 0
    assert not torch.equal(output['unrestricted_proposal_state'], state)
    for key in PHYSICAL_FIELDS:
        assert torch.equal(output[key], baseline[key]), key
    assert (output['template_e'] < 0).all()


def test_raw_valid_and_dn_interface_recompute_final_physics(device):
    original, fixed = _pair(device)
    raw, state, beta = _inputs(fixed)
    raw_dn = raw / fixed.physics.adc_step + fixed.cfg['black_level_DN']
    with torch.no_grad():
        converted = fixed.physics.raw_to_electrons(raw_dn)
        output = fixed(raw_dn, state, beta)
        direct = fixed.forward_electrons(converted, state, beta)
        expected = physical_output(fixed.physics, output['state'], beta, converted)
        control = original.forward_electrons(converted, state, beta)
    for key in PHYSICAL_FIELDS:
        assert torch.equal(output[key], expected[key]), key
        assert torch.equal(output[key], direct[key]), key
    assert output['valid'].dtype == torch.bool
    assert torch.equal(output['raw_residual_e'], output['expected_e'] - converted)
    # Diagnostics must refer to the final constrained output, not the old scaled one.
    assert not torch.equal(output['expected_e'], control['expected_e'])
    assert not torch.equal(output['raw_residual_e'], control['raw_residual_e'])


def test_signed_loss_reaches_cnn_through_internal_proposal(device):
    for checkpointed in (False, True):
        _, fixed = _pair(device)
        fixed.train()
        fixed.activation_checkpointing = checkpointed
        raw, state, beta = _inputs(fixed, batch=1)
        with torch.no_grad():
            target = physical_output(fixed.physics, state, beta, raw)['template_e'] - 3
        state.requires_grad_()
        beta.requires_grad_()
        output = fixed.forward_electrons(raw, state, beta)
        signed_loss = ((output['template_e'] - target) / 60).square().mean()
        assert signed_loss.requires_grad and torch.isfinite(signed_loss)
        signed_loss.backward()
        for group in (fixed.trunk, fixed.emission_head, fixed.continuum_head):
            gradients = [p.grad for p in group.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        assert state.grad is None and beta.grad is None
        assert torch.equal(output['state'][:, :2], state.detach()[:, :2])


def run(device='cpu'):
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    tests = (
        test_api_weights_and_parameter_count,
        test_same_weights_preserve_physical_continuum_and_scaled_signed,
        test_exact_zero_raw_residual_protects_signed_physics,
        test_raw_valid_and_dn_interface_recompute_final_physics,
        test_signed_loss_reaches_cnn_through_internal_proposal,
    )
    started = time.monotonic()
    for test in tests:
        test(device)
        print(f'PASS {test.__name__}', flush=True)
    print(json.dumps({'passed': len(tests), 'device': device,
                      'shape': [64, 64], 'optimizer_updates': 0,
                      'elapsed_s': time.monotonic() - started,
                      'model_source_sha256': runtime.sha256(runtime.ROOT / 'training_v4_4/model.py'),
                      'tests_source_sha256': runtime.sha256(__file__),
                      'coefficient_calibration_sha256': runtime.sha256(CALIBRATION_PATH),
                      'final_test_used': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    run(parser.parse_args().device)
