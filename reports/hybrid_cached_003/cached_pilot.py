"""Registered full pilot using frozen raw-derived caches, with no new baselines."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text())
SOURCE = Path(CONFIG['frozen_source_root'])
sys.path.insert(0, str(SOURCE))
os.environ.update(THERMOMETRY_RUNTIME=str(ROOT/'_runtime'), CUDA_CACHE_PATH=str(ROOT/'_runtime/cuda'),
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
import runtime
import numpy as np
import torch
from research_hybrid.run_pilot import load_physics
from research_hybrid.model import ConditionalProblem, ResidualNet, infer


class SkipResidualNet(ResidualNet):
    def forward(self, features):
        value = self.body[1](self.body[0](features))
        for start in (2, 6, 10):
            value = value+self.body[start:start+4](value)
        return 60.*self.head(value)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    registration = json.loads((ROOT/'REGISTRATION.json').read_text())
    for path, digest in registration['frozen_source_sha256'].items():
        if runtime.sha256(SOURCE/path) != digest:
            raise ValueError('Frozen source changed: '+path)
    for name, digest in registration['source_sha256'].items():
        if runtime.sha256(ROOT/name) != digest:
            raise ValueError('New source changed: '+name)
    manifest = [json.loads(x) for x in (SOURCE/'reports/hybrid_002/manifest.jsonl').read_text().splitlines()]
    for row in manifest:
        for kind in ('input', 'label'):
            path = SOURCE/'outputs/hybrid_002'/(row['case_id']+'_'+kind+'.pt')
            if runtime.sha256(path) != row[kind+'_sha256']:
                raise ValueError('Cached input changed: '+path.name)
    _, p, _ = load_physics('cuda')
    calls = 0

    def account(method, case):
        nonlocal calls
        calls += 1
        if calls > CONFIG['reserved_calls']:
            raise RuntimeError('Complete inverse reservation exhausted')
        with (ROOT/'inverse_calls.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': runtime.now(), 'phase': CONFIG['phase'],
                         'method': method, 'case_id': case, 'reused': False,
                         'run_call_number': calls, 'new_call_number': CONFIG['prior_calls']+calls})+'\n')

    def load(case):
        data = torch.load(SOURCE/'outputs/hybrid_002'/(case+'_input.pt'), map_location='cuda', weights_only=True)
        label = torch.load(SOURCE/'outputs/hybrid_002'/(case+'_label.pt'), map_location='cuda', weights_only=True)
        problem = ConditionalProblem(p, data['weights'], data['target'], data['x0'])
        mu = CONFIG['mu_relative_to_sensitivity']*problem.linear.contract['template_reference_sensitivity']
        problem.penalty[:-3] += mu-problem.mu
        problem.mu = mu
        return data, label, problem

    train = [r for r in manifest if r['split'] == 'train' and r['continuum_variant'] == 0]
    generator = np.random.default_rng(CONFIG['seed'])
    schedule = []
    while len(schedule) < CONFIG['updates']:
        schedule.extend(generator.permutation(len(train)).tolist())
    model = SkipResidualNet().cuda()
    initial_digest = hashlib.sha256()
    for value in model.state_dict().values():
        initial_digest.update(value.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for step, choice in enumerate(schedule[:CONFIG['updates']], 1):
        optimizer.zero_grad(set_to_none=True)
        predictions, losses = [], []
        for variant in (0, 1):
            case = train[choice]['case_id'][:-1]+str(variant)
            data, label, problem = load(case)
            account('hybrid_training_forward', case+f'_update{step:03d}')
            prediction, _, _ = infer(model, problem, data['features'])
            predictions.append(prediction)
            losses.append(((prediction-label['template'])/60.).square().mean())
        loss = torch.stack(losses).mean()+.25*((predictions[0]-predictions[1])/60.).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError('Nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
               'mother_id': train[choice]['mother_id'], 'condition': train[choice]['condition']}
        with (ROOT/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if step == 1 or step % 25 == 0:
            runtime.write_json(ROOT/'STATUS.json', {'stage': 'TRAINING', 'pid': os.getpid(),
                               'update': step, 'calls': calls, 'elapsed_s': time.monotonic()-started})
            print(row, flush=True)
    torch.save(model.state_dict(), ROOT/'last_checkpoint.pt')
    model.eval()
    metrics = []
    max_normal = 0.
    with torch.no_grad():
        for row in [r for r in manifest if r['split'] == 'validation']:
            case = row['case_id']
            data, label, problem = load(case)
            base = problem.linear.U.unpack(data['x0'][:-3])
            account('matched_extra_physics', case)
            control = problem.linear.U.unpack(problem.solve(data['x0'][:-3])[:-3])
            if not torch.equal(control, base):
                raise ArithmeticError('Matched zero-head solve changed the converged baseline')
            account('hybrid_evaluation', case)
            prediction, x, proposal = infer(model, problem, data['features'])
            max_normal = max(max_normal, problem.last_solve['relative_residual'])
            account('same_network_proposal_no_DC', case)
            outputs = {'v3': data['v3_template'], 'fixed_profile': base, 'matched_control': control,
                       'same_network_proposal_no_DC': problem.linear.U.unpack(proposal), 'hybrid': prediction}
            for method, value in outputs.items():
                error = value-label['template']
                metrics.append({'case_id': case, 'mother_id': row['mother_id'], 'peak_e': row['peak_e'],
                                'method': method, 'pixels': value.numel(),
                                'template_squared_error_sum': float(error.square().sum()),
                                'template_bias_e': float(error.mean()),
                                'prediction_truth_inner_product': float((value*label['template']).sum()),
                                'truth_squared_sum': float(label['template'].square().sum()),
                                'temperature_squared_error_sum': float((data['temperature']-label['temperature']).square().sum())})
            torch.save({'predictions': {k: v.cpu() for k, v in outputs.items()},
                        'temperature_fixed': data['temperature'].cpu(),
                        'temperature_hybrid': data['temperature'].cpu(),
                        'numerical_solve': problem.last_solve}, ROOT/(case+'_evaluation.pt'))
    summary = []
    for method in outputs:
        for group, levels in [('all', [0., 30., -30., 60., -60.]), ('null', [0.]),
                              ('weak', [30., -30.]), ('larger_signed', [60., -60.])]:
            subset = [v for v in metrics if v['method'] == method and v['peak_e'] in levels]
            pixels = sum(v['pixels'] for v in subset)
            truth_norm = sum(v['truth_squared_sum'] for v in subset)
            summary.append({'method': method, 'group': group, 'frames': len(subset), 'mothers': 2,
                            'pixels': pixels, 'template_RMSE_e': (sum(v['template_squared_error_sum'] for v in subset)/pixels)**.5,
                            'color_RMSE_K': (sum(v['temperature_squared_error_sum'] for v in subset)/pixels)**.5,
                            'signed_gain': sum(v['prediction_truth_inner_product'] for v in subset)/truth_norm if truth_norm else None,
                            'mean_bias_e': sum(v['template_bias_e'] for v in subset)/len(subset)})
    runtime.write_json(ROOT/'EVALUATION.json', {'rows': metrics, 'summary': summary,
                       'validation_mothers': 2, 'temperature_unchanged': True,
                       'detector': 'NOT_CALIBRATED', 'final_payloads_opened': False,
                       'scientific_acceptance': 'NOT_ASSESSED'})
    if calls != CONFIG['reserved_calls']:
        raise ArithmeticError('Actual call count differs from reservation')
    receipt = {'time': runtime.now(), 'stage': 'COMPLETED_CACHED_HYBRID_PILOT', 'calls': calls,
               'optimizer_updates': CONFIG['updates'], 'train_mothers': 8, 'validation_mothers': 2,
               'initial_weight_sha256': initial_digest.hexdigest(),
               'max_scaled_normal_relative': max_normal, 'temperature_unchanged': True,
               'scientific_acceptance': False, 'final_payloads_opened': False,
               'wall_s': time.monotonic()-started, 'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20,
               'checkpoint_sha256': runtime.sha256(ROOT/'last_checkpoint.pt'),
               'evaluation_sha256': runtime.sha256(ROOT/'EVALUATION.json')}
    runtime.write_json(ROOT/'COMPLETION.json', receipt)
    runtime.write_json(ROOT/'STATUS.json', {'stage': receipt['stage'], 'pid': os.getpid(), 'calls': calls})
    print(receipt, flush=True)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
