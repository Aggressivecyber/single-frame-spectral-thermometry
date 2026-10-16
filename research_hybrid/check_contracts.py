"""Short CPU numerical contracts, without training or experimental raw inputs."""
import runtime
from pathlib import Path
import json
import time
import torch
from physics import SpectralPhysics
from .observable_basis import RealFourierBasis
from .linear_profile import LinearProfile
from .model import ConditionalProblem, Proximal, ResidualNet, tangent_features
from .run_pilot import load_physics

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'research_hybrid'
REPORT = ROOT/'reports/hybrid_001'


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(617)
    config = json.loads((CODE/'backbone_selection.json').read_text())['physics_config']
    calibration = CODE/'calibration.pt'
    if not calibration.exists():
        torch.save(SpectralPhysics(config).state_dict(), calibration)
    _, p, _ = load_physics('cpu')
    dtype = torch.float64
    basis = RealFourierBasis(p.op.object_otf)
    coefficients = torch.randn(basis.dimension, dtype=dtype)
    im = basis.unpack(coefficients)
    parseval = float(abs(im.square().sum()-coefficients.square().sum())/coefficients.square().sum())
    roundtrip = float((basis.pack(im)-coefficients).norm()/coefficients.norm())
    outside = float(torch.fft.fft2(im)[~basis.mask].abs().max())
    bands = torch.randn((1, 7, *config['shape']), dtype=dtype)
    raw = torch.randn((1, *config['shape']), dtype=dtype)
    left = (p.op(bands)*raw).sum()
    right = (bands*p.op.adjoint(raw)).sum()
    adjoint = float(abs(left-right)/max(abs(float(left)), abs(float(right)), 1.))
    z = torch.zeros((1, 2, *config['shape']), dtype=dtype)
    z[:, 0] = (2962.3-p.T0)/p.Tscale
    z[:, 1] = -.7
    direction = .01*torch.randn(z.shape, dtype=dtype)
    C, dC = p.continuum(p.T0+p.Tscale*z[:, 0])
    jac = torch.stack([dC*p.Tscale*z[:, 1:2].exp(), C*z[:, 1:2].exp()], 1)
    analytic = p.op((jac*direction[:, :, None]).sum(1))
    eps = 1e-4
    high = z+eps*direction
    low = z-eps*direction
    finite = (p.op(p.continuum(p.T0+p.Tscale*high[:, 0])[0]*high[:, 1:2].exp())-
              p.op(p.continuum(p.T0+p.Tscale*low[:, 0])[0]*low[:, 1:2].exp()))/(2*eps)
    jac_error = float((analytic-finite).norm()/analytic.norm())
    back = (jac*p.op.adjoint(raw)[:, None]).sum(2)
    jac_adjoint = float(abs((analytic*raw).sum()-(direction*back).sum())/abs((analytic*raw).sum()))
    weights = (torch.full(config['shape'], 5000., dtype=dtype)+p.extra_variance).reciprocal()
    effective, tangent_info = tangent_features(p, z, raw, weights)

    # Direct finite-mode QR uses the same native sampled forward operator.
    mask = torch.zeros_like(p.op.object_otf)
    for y in (0, 1, -1):
        for x in (0, 1, -1):
            mask[y, x] = p.op.object_otf[y, x]
    small = RealFourierBasis(mask)
    linear = LinearProfile(p, weights, 1., basis=small)
    target = .1*torch.randn(config['shape'], dtype=dtype)
    base, base_info = linear.solve(target, tolerance=1e-12)
    reference, A, _, _ = linear.factor_reference(target)
    qr_error = float((base-reference).norm()/reference.norm())
    problem = ConditionalProblem(p, weights, target, base, basis=small)
    proposal = (base[:-3]+torch.randn(small.dimension, dtype=dtype)).requires_grad_()
    result = Proximal.apply(proposal, problem)
    mu = problem.mu
    prox_rows = torch.zeros((small.dimension, linear.n), dtype=dtype)
    prox_rows[:, :small.dimension] = torch.eye(small.dimension, dtype=dtype)*mu**.5
    augmented = torch.cat([A, prox_rows*linear.column_scale[None]], 0)
    rhs = torch.cat([(weights.sqrt()*target).flatten(), linear.reg.new_zeros(linear.n),
                     proposal.detach()*mu**.5])
    q, r = torch.linalg.qr(augmented, mode='reduced')
    prox_ref = torch.linalg.solve_triangular(r, (q.T@rhs)[:, None], upper=True)[:, 0]*linear.column_scale
    prox_qr_error = float((result.detach()-prox_ref).norm()/prox_ref.norm())
    test_gradient = torch.randn(linear.n, dtype=dtype)
    (result*test_gradient).sum().backward()
    test_direction = torch.randn(small.dimension, dtype=dtype)
    step = 1e-3
    fd = ((problem.solve(proposal.detach()+step*test_direction)-
           problem.solve(proposal.detach()-step*test_direction))*test_gradient).sum()/(2*step)
    ad = (proposal.grad*test_direction).sum()
    implicit_error = float(abs(fd-ad)/abs(ad))

    # Native passband solve and network-off matched-step invariance.
    full = LinearProfile(p, weights, 1.)
    native_base, native_info = full.solve(target, tolerance=1e-11)
    native = ConditionalProblem(p, weights, target, native_base)
    unchanged = native.solve(native_base[:-3])
    unchanged_e = float(full.U.unpack((unchanged-native_base)[:-3]).abs().max())
    network = ResidualNet()
    features = torch.randn((1, 8, *config['shape']))
    zero_head = bool((network(features) == 0).all())
    result = {'schema': 'hybrid-engineering-contracts-001', 'device': 'cpu',
              'training_updates': 0, 'experimental_raw_inverse_calls': 0,
              'seconds': time.perf_counter()-started, 'passband_dimension': basis.dimension,
              'parseval_relative': parseval, 'basis_roundtrip_relative': roundtrip,
              'outside_passband_max_FFT': outside, 'forward_adjoint_relative': adjoint,
              'continuum_JVP_finite_difference_relative': jac_error,
              'continuum_JVP_adjoint_relative': jac_adjoint,
              'regularized_tangent': tangent_info,
              'tangent_residual_norm_ratio': float(effective.norm()/raw.norm()),
              'small_QR_profile_relative': qr_error, 'small_QR_prox_relative': prox_qr_error,
              'implicit_gradient_directional_relative': implicit_error,
              'native_profile': native_info, 'native_prox': native.last_solve,
              'matched_step_max_change_e': unchanged_e, 'zero_initialized_head': zero_head,
              'network_parameters': sum(v.numel() for v in network.parameters()),
              'calibration_sha256': runtime.sha256(calibration),
              'thresholds': {'algebra_relative': 1e-10, 'finite_difference_relative': 1e-6,
                             'QR_relative': 1e-6, 'implicit_gradient_relative': 1e-5,
                             'matched_step_e': .001}}
    result['all_passed'] = bool(max(parseval, roundtrip, adjoint, jac_adjoint) < 1e-10 and
                                outside < 1e-10 and jac_error < 1e-6 and qr_error < 1e-6 and
                                prox_qr_error < 1e-6 and implicit_error < 1e-5 and
                                unchanged_e < .001 and native_info['numerical_certified'] and zero_head)
    runtime.write_json(REPORT/'CPU_CONTRACTS.json', result)
    print(json.dumps(result), flush=True)
    if not result['all_passed']:
        raise ArithmeticError('Hybrid CPU contract failed')


if __name__ == '__main__':
    main()
