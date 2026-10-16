"""One budgeted warm-start diagnostic of signed response and null preservation."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text())
SOURCE = Path(CONFIG['frozen_source_root'])
PRIOR = Path(CONFIG['previous_run_root'])
sys.path.insert(0, str(SOURCE))
os.environ.update(THERMOMETRY_RUNTIME=str(ROOT/'_runtime'), CUDA_CACHE_PATH=str(ROOT/'_runtime/cuda'),
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
import runtime
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
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory)
    registration = json.loads((ROOT/'REGISTRATION.json').read_text())
    for base, key in [(SOURCE, 'frozen_source_sha256'), (ROOT, 'source_sha256'),
                      (PRIOR, 'previous_artifact_sha256')]:
        for name, digest in registration[key].items():
            if runtime.sha256(base/name) != digest:
                raise ValueError('Registered file changed: '+str(base/name))
    manifest = [json.loads(x) for x in (SOURCE/'reports/hybrid_002/manifest.jsonl').read_text().splitlines()]
    cases = {row['case_id']: row for row in manifest}
    for row in manifest:
        for kind in ('input', 'label'):
            path = SOURCE/'outputs/hybrid_002'/(row['case_id']+'_'+kind+'.pt')
            if runtime.sha256(path) != row[kind+'_sha256']:
                raise ValueError('Cached file changed: '+str(path))
    _, physics, _ = load_physics('cuda')
    calls = 0
    maximum_normal = 0.

    def predict(case, method):
        nonlocal calls, maximum_normal
        data = torch.load(SOURCE/'outputs/hybrid_002'/(case+'_input.pt'), map_location='cuda', weights_only=True)
        label = torch.load(SOURCE/'outputs/hybrid_002'/(case+'_label.pt'), map_location='cuda', weights_only=True)
        problem = ConditionalProblem(physics, data['weights'], data['target'], data['x0'])
        mu = CONFIG['mu_relative_to_sensitivity']*problem.linear.contract['template_reference_sensitivity']
        problem.penalty[:-3] += mu-problem.mu
        problem.mu = mu
        if calls >= CONFIG['reserved_calls'] or CONFIG['prior_calls']+calls >= CONFIG['running_call_guard']:
            raise RuntimeError('Registered inverse budget exhausted')
        calls += 1
        with (ROOT/'inverse_calls.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': runtime.now(), 'phase': CONFIG['phase'], 'method': method,
                         'case_id': case, 'reused': False, 'run_call_number': calls,
                         'new_call_number': CONFIG['prior_calls']+calls})+'\n')
        value, _, _ = infer(model, problem, data['features'])
        maximum_normal = max(maximum_normal, problem.last_solve['relative_residual'])
        return value, data, label, problem.last_solve

    model = SkipResidualNet().cuda()
    model.load_state_dict(torch.load(PRIOR/'last_checkpoint.pt', map_location='cuda', weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    for step, entry in enumerate(CONFIG['schedule'], 1):
        optimizer.zero_grad(set_to_none=True)
        predictions, labels, squared_errors, gains = [], [], [], []
        for case in entry['signed_cases']:
            if cases[case]['split'] != 'train':
                raise ValueError('Training schedule crosses a split')
            value, _, label, _ = predict(case, 'signed_training_forward')
            truth = label['template']
            predictions.append(value)
            labels.append(truth)
            squared_errors.append(((value-truth)/60.).square().mean())
            gains.append((value*truth).sum()/truth.square().sum())
        truth_difference = labels[0]-labels[1]
        paired_gain = ((predictions[0]-predictions[1])*truth_difference).sum()/truth_difference.square().sum()
        supervised = torch.stack(squared_errors).mean()
        response = .5*torch.stack([(gain-1.).square() for gain in gains]).mean()+.5*(paired_gain-1.).square()
        null_loss = supervised.new_zeros(())
        if entry['null_case'] is not None:
            case = entry['null_case']
            if cases[case]['split'] != 'train' or cases[case]['peak_e'] != 0:
                raise ValueError('Null training case is not a training null')
            value, _, label, _ = predict(case, 'null_training_forward')
            null_loss = ((value-label['template'])/60.).square().mean()
        loss = supervised+CONFIG['response_weight']*response+CONFIG['null_weight']*null_loss
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError('Nonfinite response training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        record = {'update': step, **entry, 'loss': float(loss.detach()),
                  'supervised_loss': float(supervised.detach()), 'response_loss': float(response.detach()),
                  'null_loss': float(null_loss.detach()), 'individual_gains': [float(g.detach()) for g in gains],
                  'paired_gain': float(paired_gain.detach()), 'gradient_norm': float(norm)}
        with (ROOT/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        runtime.write_json(ROOT/'STATUS.json', {'stage': 'TRAINING', 'update': step, 'calls': calls})
    torch.save(model.state_dict(), ROOT/'last_checkpoint.pt')
    model.eval()
    metrics = []
    # Prior evaluation arrays are opened only after the final checkpoint is frozen.
    with torch.no_grad():
        for row in [v for v in manifest if v['split'] == 'validation']:
            case = row['case_id']
            value, data, label, numerical = predict(case, 'hybrid_evaluation')
            previous = torch.load(PRIOR/(case+'_evaluation.pt'), map_location='cuda', weights_only=True)
            if not torch.equal(data['temperature'], previous['temperature_hybrid']):
                raise ArithmeticError('Frozen temperature changed')
            outputs = {'v3': previous['predictions']['v3'],
                       'previous_hybrid': previous['predictions']['hybrid'], 'response_hybrid': value}
            for method, prediction in outputs.items():
                error = prediction-label['template']
                metrics.append({'case_id': case, 'mother_id': row['mother_id'], 'peak_e': row['peak_e'],
                                'continuum_variant': row['continuum_variant'], 'method': method,
                                'pixels': prediction.numel(), 'template_squared_error_sum': float(error.square().sum()),
                                'template_bias_e': float(error.mean()),
                                'prediction_truth_inner_product': float((prediction*label['template']).sum()),
                                'truth_squared_sum': float(label['template'].square().sum()),
                                'temperature_squared_error_sum': float((data['temperature']-label['temperature']).square().sum())})
            torch.save({'template': value.cpu(), 'temperature': data['temperature'].cpu(),
                        'numerical_solve': numerical}, ROOT/(case+'_evaluation.pt'))
    summary = []
    for method in outputs:
        for group, levels in [('all', [0., 30., -30., 60., -60.]), ('null', [0.]),
                              ('weak', [30., -30.]), ('larger_signed', [60., -60.])]:
            selected = [r for r in metrics if r['method'] == method and r['peak_e'] in levels]
            pixels = sum(r['pixels'] for r in selected)
            truth_norm = sum(r['truth_squared_sum'] for r in selected)
            summary.append({'method': method, 'group': group, 'frames': len(selected), 'mothers': 2,
                            'pixels': pixels, 'template_RMSE_e': (sum(r['template_squared_error_sum'] for r in selected)/pixels)**.5,
                            'signed_gain': sum(r['prediction_truth_inner_product'] for r in selected)/truth_norm if truth_norm else None,
                            'color_RMSE_K': (sum(r['temperature_squared_error_sum'] for r in selected)/pixels)**.5})
    runtime.write_json(ROOT/'EVALUATION.json', {'rows': metrics, 'summary': summary,
                       'validation_mothers': 2, 'temperature_unchanged': True,
                       'detector': 'NOT_CALIBRATED', 'final_payloads_opened': False})
    if calls != CONFIG['reserved_calls']:
        raise ArithmeticError('Completed call count differs from reservation')
    runtime.write_json(ROOT/'COMPLETION.json', {'time': runtime.now(), 'stage': 'COMPLETED_RESPONSE_DIAGNOSTIC',
                       'calls': calls, 'optimizer_updates': len(CONFIG['schedule']),
                       'max_scaled_normal_relative': maximum_normal, 'temperature_unchanged': True,
                       'wall_s': time.monotonic()-started, 'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20,
                       'checkpoint_sha256': runtime.sha256(ROOT/'last_checkpoint.pt'),
                       'evaluation_sha256': runtime.sha256(ROOT/'EVALUATION.json'),
                       'scientific_acceptance': False, 'final_payloads_opened': False})
    runtime.write_json(ROOT/'STATUS.json', {'stage': 'COMPLETED_RESPONSE_DIAGNOSTIC', 'calls': calls})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
