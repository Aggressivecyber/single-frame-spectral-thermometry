"""Physical objective/gradient and false-plateau regression contracts."""
from .. import storage as s
from .. import measurement as m
from ..reliable_v3 import V3Objective, plateau_accepts, validated_options
from repair_v3.solver import objective as old_objective
import argparse
import json
import numpy as np
import torch


def run(output=None):
    torch.set_num_threads(2)
    p = m.physics('cpu')
    state = p.M.new_zeros((1, 5, *p.cfg['shape']))
    state[:, 0] = (3102.3-p.T0)/p.Tscale
    state[:, 1] = np.log(.53)
    yy, xx = torch.meshgrid(torch.arange(256, dtype=p.M.dtype), torch.arange(256, dtype=p.M.dtype), indexing='ij')
    state[:, 4] = .1*torch.cos(2*torch.pi*xx/256)*torch.cos(2*torch.pi*yy/256)
    beta = p.M.new_tensor([[.015, .001, -.002]])
    raw_e = p(state, beta)+.2*torch.sin(2*torch.pi*xx[None]/256)
    obj = V3Objective(p, raw_e)
    flat = obj.pack(state, beta)
    value, gradient, _, _ = obj.evaluate(flat)
    original = old_objective(p, raw_e, state, beta, obj.W, obj.strength)[0]
    direction = np.random.default_rng(930015).normal(size=flat.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-4
    finite_difference = (obj.evaluate(flat+eps*direction)[0]-obj.evaluate(flat-eps*direction)[0])/(2*eps)
    analytic = float(gradient@direction)
    relative = abs(finite_difference-analytic)/max(abs(analytic), 1e-12)
    checks = [
        {'name': 'unchanged_v3_physical_objective', 'passed': abs(value-float(original)) < 1e-10,
         'absolute_difference': abs(value-float(original))},
        {'name': 'actual_native256_analytic_gradient', 'passed': relative < 2e-5,
         'relative_error': relative, 'analytic': analytic, 'finite_difference': finite_difference},
        {'name': 'optimizer_success_is_not_output_convergence', 'passed': not plateau_accepts([
            {'both_optimizer_success': True, 'color_RMS_K': 1., 'template_RMS_e': .3}]*2)},
        {'name': 'requires_two_tighter_output_comparisons', 'passed': not plateau_accepts([
            {'both_optimizer_success': True, 'color_RMS_K': .01, 'template_RMS_e': .01}])},
        {'name': 'failed_optimizer_cannot_certify_plateau', 'passed': not plateau_accepts([
            {'both_optimizer_success': False, 'color_RMS_K': 0., 'template_RMS_e': 0.}]*2)},
        {'name': 'shared_raw_calibration_scale_is_deterministic', 'passed': bool(np.array_equal(obj.scale, V3Objective(p, raw_e).scale))},
    ]
    rejected = []
    for options in ({'cost_limit_s': 181.}, {'numeric_color_budget_K': 1.},
                    {'minimum_stable_comparisons': 1}, {'unknown': True}):
        try:
            validated_options(options)
            rejected.append(False)
        except ValueError:
            rejected.append(True)
    checks.append({'name': 'cannot_relax_registered_numerical_budget', 'passed': all(rejected)})
    result = {'time': s.now(), 'all_passed': all(row['passed'] for row in checks), 'checks': checks,
              'native_shape': [256, 256], 'complete_inversions': 0, 'scientific_validation': False,
              'source_sha256': {name: s.sha(s.ROOT/name) for name in (
                  'research_closeout/reliable_v3.py', 'research_closeout/tests/v3_repair.py')}}
    if output:
        s.write(output, result)
    print(json.dumps(result, indent=2))
    assert result['all_passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    run(parser.parse_args().output)
