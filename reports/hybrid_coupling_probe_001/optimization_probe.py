"""Matched-capacity optimization diagnostic on one cached training pair."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'probe_config.json').read_text())
SOURCE = Path(CONFIG['frozen_source_root'])
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
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    registration = json.loads((ROOT/'REGISTRATION.json').read_text())
    for path, digest in registration['frozen_source_sha256'].items():
        if runtime.sha256(SOURCE/path) != digest:
            raise ValueError('Frozen source changed: '+path)
    for name, digest in registration['probe_source_sha256'].items():
        if runtime.sha256(ROOT/name) != digest:
            raise ValueError('Probe source changed: '+name)
    _, physics, _ = load_physics('cuda')
    records = []
    for case in CONFIG['training_pair']:
        for kind in ('input', 'label'):
            path = SOURCE/'outputs/hybrid_002'/(case+'_'+kind+'.pt')
            if runtime.sha256(path) != registration['input_sha256'][path.name]:
                raise ValueError('Cached training input changed: '+path.name)
        data = torch.load(SOURCE/'outputs/hybrid_002'/(case+'_input.pt'), map_location='cuda', weights_only=True)
        label = torch.load(SOURCE/'outputs/hybrid_002'/(case+'_label.pt'), map_location='cuda', weights_only=True)
        problem = ConditionalProblem(physics, data['weights'], data['target'], data['x0'])
        new_mu = CONFIG['mu_relative_to_sensitivity']*problem.linear.contract['template_reference_sensitivity']
        problem.penalty[:-3] += new_mu-problem.mu
        problem.mu = new_mu
        records.append((case, data, label, problem))
    calls = 0
    results = []
    initial_hashes = []
    constructors = {'serial_DSConv': ResidualNet, 'residual_DSConv': SkipResidualNet}
    for arm in CONFIG['arms']:
        constructor = constructors[arm]
        torch.manual_seed(CONFIG['seed'])
        model = constructor().cuda()
        digest = hashlib.sha256()
        for _, value in model.state_dict().items():
            digest.update(value.detach().cpu().numpy().tobytes())
        initial_hashes.append(digest.hexdigest())
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        for step in range(1, CONFIG['updates_per_arm']+1):
            optimizer.zero_grad(set_to_none=True)
            predictions, errors = [], []
            for case, data, label, problem in records:
                calls += 1
                if calls > CONFIG['reserved_calls']:
                    raise RuntimeError('Probe inverse budget exhausted')
                with (ROOT/'inverse_calls.jsonl').open('a') as ledger:
                    ledger.write(json.dumps({'time': runtime.now(), 'phase': CONFIG['phase'], 'method': arm,
                                 'case_id': case+f'_update{step:03d}', 'reused': False,
                                 'probe_call_number': calls, 'new_call_number': CONFIG['prior_calls']+calls})+'\n')
                value, _, _ = infer(model, problem, data['features'])
                predictions.append(value)
                errors.append(value-label['template'])
            supervised = torch.stack([(v/60.).square().mean() for v in errors]).mean()
            paired = .25*((predictions[0]-predictions[1])/60.).square().mean()
            loss = supervised+paired
            if not bool(torch.isfinite(loss)):
                raise ArithmeticError('Nonfinite diagnostic loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row = {'arm': arm, 'update': step, 'loss': float(loss.detach()),
                   'supervised_loss': float(supervised.detach()), 'paired_loss': float(paired.detach()),
                   'centered_MSE_e2': float(torch.stack([(v-v.mean()).square().mean() for v in errors]).mean().detach()),
                   'gradient_norm': float(norm)}
            with (ROOT/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if step == 1 or step % 25 == 0:
                runtime.write_json(ROOT/'STATUS.json', {'stage': 'TRAINING', 'pid': os.getpid(),
                                   'arm': arm, 'step': step, 'calls': calls, 'elapsed_s': time.monotonic()-started})
                print(row, flush=True)
        torch.save(model.state_dict(), ROOT/(arm+'_last.pt'))
        with torch.no_grad():
            for case, data, label, problem in records:
                calls += 1
                with (ROOT/'inverse_calls.jsonl').open('a') as ledger:
                    ledger.write(json.dumps({'time': runtime.now(), 'phase': CONFIG['phase'],
                                 'method': arm+'_postupdate_evaluation', 'case_id': case, 'reused': False,
                                 'probe_call_number': calls, 'new_call_number': CONFIG['prior_calls']+calls})+'\n')
                value, x, proposal = infer(model, problem, data['features'])
                base = problem.linear.U.unpack(data['x0'][:-3])
                base_error = base-label['template']
                error = value-label['template']
                correction = problem.linear.U.unpack(proposal)-base
                before = float((base_error-base_error.mean()).square().mean())
                after = float((error-error.mean()).square().mean())
                results.append({'arm': arm, 'case_id': case, 'pixels': value.numel(),
                                'baseline_RMSE_e': float(base_error.square().mean().sqrt()),
                                'trained_RMSE_e': float(error.square().mean().sqrt()),
                                'baseline_centered_MSE_e2': before, 'trained_centered_MSE_e2': after,
                                'centered_MSE_reduction_fraction': 1-after/before,
                                'proposal_correction_mean_e': float(correction.mean()),
                                'proposal_correction_spatial_std_e': float(correction.std(correction=0)),
                                'numerical_solve': problem.last_solve})
                torch.save({'template': value.cpu(), 'baseline_template': base.cpu(),
                            'proposal': proposal.cpu(), 'temperature_fixed': data['temperature'].cpu()},
                           ROOT/(arm+'_'+case+'_output.pt'))
    if calls != CONFIG['reserved_calls'] or len(set(initial_hashes)) != 1:
        raise ArithmeticError('Unmatched parameter initialization or incorrect accounting')
    passed = all(r['centered_MSE_reduction_fraction'] >= .2 for r in results if r['arm'] == 'residual_DSConv')
    receipt = {'time': runtime.now(), 'stage': 'COMPLETED_OPTIMIZATION_DIAGNOSTIC',
               'calls': calls, 'arms': len(CONFIG['arms']), 'updates_per_arm': CONFIG['updates_per_arm'],
               'mu_relative_to_sensitivity': CONFIG['mu_relative_to_sensitivity'],
               'identical_initial_weight_sha256': initial_hashes[0], 'parameters_each': 1457,
               'rows': results, 'residual_arm_training_centered_error_gate_passed': passed,
               'validation_or_final_used': False, 'scientific_acceptance': False,
               'wall_s': time.monotonic()-started, 'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20}
    runtime.write_json(ROOT/'COMPLETION.json', receipt)
    runtime.write_json(ROOT/'STATUS.json', {'stage': receipt['stage'], 'calls': calls, 'pid': os.getpid()})
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
