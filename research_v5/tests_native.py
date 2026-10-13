"""Root-owned native-resolution control before source/protocol freeze."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from . import common

common.configure_runtime()
import torch
from .continuum_solver import estimate_continuum
from .signed_linear_solver import SignedLinearOperator, solve_signed_linear


@torch.no_grad()
def run(device='cuda'):
    started = time.perf_counter()
    p = common.make_physics(device, torch.float64)
    h, w = p.cfg['shape']
    generator = torch.Generator(device=device).manual_seed(510133)
    checks = []

    def record(name, value, limit):
        checks.append({'name': name, 'value': float(value), 'limit': float(limit),
                       'passed': float(value) <= limit})

    def relative(a, b):
        return float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-30))

    def random(*shape):
        return torch.randn(shape, dtype=p.M.dtype, device=device, generator=generator)

    bands, probe = random(1, 7, h, w), random(1, h, w)
    lhs, rhs = (p.op(bands)*probe).sum(), (bands*p.op.adjoint(probe)).sum()
    record('native_full_forward_adjoint', abs(float(lhs-rhs))/max(abs(float(lhs)), abs(float(rhs)), 1e-30), 1e-10)
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=p.M.dtype),
                            torch.arange(w, device=device, dtype=p.M.dtype), indexing='ij')
    state = torch.zeros((1, 5, h, w), device=device, dtype=p.M.dtype)
    state[:, 0] = (3052.5-p.T0)/p.Tscale
    state[:, 1] = -.5 + .025*torch.cos(2*torch.pi*xx/w)
    beta = p.M.new_tensor([[.5, .015, -.01]])
    direction, beta_direction = random(1, 5, h, w), random(1, 3)
    jvp = p.jvp(state, beta, direction, beta_direction)
    gz, gb = p.vjp(state, beta, probe)
    lhs, rhs = (jvp*probe).sum(), (gz*direction).sum()+(gb*beta_direction).sum()
    record('native_JVP_VJP', abs(float(lhs-rhs))/max(abs(float(lhs)), abs(float(rhs)), 1e-30), 1e-10)
    epsilon = 1e-7
    finite_difference = (p(state+epsilon*direction, beta+epsilon*beta_direction)
                         -p(state-epsilon*direction, beta-epsilon*beta_direction))/(2*epsilon)
    record('native_JVP_finite_difference', relative(jvp, finite_difference), 1e-6)
    linear = SignedLinearOperator(p)
    C = p.components(state, beta)[0]
    unsigned = 180*(.55 + .25*torch.cos(2*torch.pi*xx/w) + .15*torch.sin(4*torch.pi*yy/h))
    true_postblur = p.op.object_blur(unsigned[None])
    controls = []
    for sign in (1, -1):
        raw = p.op(C)+linear.apply_beta(beta)+sign*linear.apply_template(unsigned)[None]
        for background in ('fixed_beta', 'joint_beta'):
            common.synchronize(device)
            t0 = time.perf_counter()
            out = solve_signed_linear(p, raw, C, fixed_beta=beta if background=='fixed_beta' else None)
            common.synchronize(device)
            frame = out['info']['frames'][0]
            target = sign*true_postblur
            rmse = float((out['template_postblur_e']-target).square().mean().sqrt())
            l2 = relative(out['template_postblur_e'], target)
            prefix = f'{background}_{sign:+d}'
            record(prefix+'_CG_true_residual', frame['CG']['relative_true_residual'], 1e-4)
            record(prefix+'_CG_equilibrated_true_residual', frame['CG']['equilibrated_relative_true_residual'], 1e-4)
            record(prefix+'_postblur_RMSE_e', rmse, .5)
            record(prefix+'_postblur_L2', l2, .02)
            controls.append({'sign': sign, 'background': background, 'wall_s': time.perf_counter()-t0,
                             'template_RMSE_e': rmse, 'template_L2': l2, 'info': out['info']})
        warm = estimate_continuum(p, raw)
        record(f'continuum_{sign:+d}_template_fields_remain_zero', float(warm['state'][:, 2:].abs().max()), 0.)
        record(f'continuum_{sign:+d}_single_blur', relative(warm['continuum_postblur'], p.op.object_blur(warm['continuum_latent'])), 1e-12)
    return {'schema': 'v5-native-resolution-control-v1', 'shape': [h, w], 'device': device,
            'dtype': 'float64', 'gpu': torch.cuda.get_device_name() if device=='cuda' else None,
            'all_passed': all(c['passed'] for c in checks), 'checks': checks, 'controls': controls,
            'wall_s': time.perf_counter()-started, 'scientific_cohort_acceptance': False,
            'new_or_sealed_mother_arrays_generated': False, 'synthetic_control_seed': 510133,
            'physics_config_sha256': common.canonical_hash(p.cfg),
            'sources': {common.relative(common.CODE/name): common.sha(common.CODE/name)
                        for name in ('tests_native.py', 'continuum_solver.py', 'signed_linear_solver.py')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.device)
    common.write_json(args.output, report)
    print({'all_passed': report['all_passed'], 'checks': len(report['checks']), 'wall_s': report['wall_s'],
           'receipt': str(args.output)}, flush=True)
    if not report['all_passed']:
        print([c for c in report['checks'] if not c['passed']], flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
