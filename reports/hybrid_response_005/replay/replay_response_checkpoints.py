"""Replay each frozen arm on one null and one negative 60-electron frame."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
sys.path.insert(0, str(STUDY/'response'))
import signed_response_pilot as experiment
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
_, physics, _ = experiment.load_physics('cuda')
rows = []
calls = 0
for arm in ('control', 'response'):
    out = STUDY/arm
    config = json.loads((out/'config.json').read_text())
    completed = json.loads((out/'COMPLETION.json').read_text())
    assert experiment.runtime.sha256(out/'last_checkpoint.pt') == completed['checkpoint_sha256']
    model = experiment.SkipResidualNet().cuda()
    model.load_state_dict(torch.load(out/'last_checkpoint.pt', map_location='cuda', weights_only=True))
    model.eval()
    with torch.no_grad():
        for level in (0, 4):
            case = f'hybrid_002_validation_008_level{level}_continuum0'
            data = torch.load(experiment.SOURCE/'outputs/hybrid_002'/(case+'_input.pt'),
                              map_location='cuda', weights_only=True)
            problem = experiment.ConditionalProblem(physics, data['weights'], data['target'], data['x0'])
            mu = config['mu_relative_to_sensitivity']*problem.linear.contract['template_reference_sensitivity']
            problem.penalty[:-3] += mu-problem.mu
            problem.mu = mu
            calls += 1
            with (ROOT/'inverse_calls.jsonl').open('a') as ledger:
                ledger.write(json.dumps({'time': experiment.runtime.now(), 'phase': 'hybrid_response_005_replay',
                             'method': arm+'_checkpoint_replay', 'case_id': case, 'reused': False,
                             'run_call_number': calls, 'new_call_number': 6929+calls})+'\n')
            value, _, _ = experiment.infer(model, problem, data['features'])
            saved = torch.load(out/(case+'_evaluation.pt'), map_location='cpu', weights_only=True)
            actual = value.cpu()
            assert torch.equal(actual, saved['template']), 'Checkpoint replay is not bitwise identical'
            rows.append({'arm': arm, 'case_id': case, 'pixels': actual.numel(), 'bitwise_equal': True,
                         'prediction_sha256': hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                         'relative_normal_residual': problem.last_solve['relative_residual']})
receipt = {'time': experiment.runtime.now(), 'rows': rows, 'all_four_bitwise_equal': True,
           'new_complete_inverse_calls': calls, 'python': platform.python_version(),
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'labels_opened': False}
experiment.runtime.write_json(ROOT/'CHECKPOINT_REPLAY.json', receipt)
print(json.dumps(receipt))
