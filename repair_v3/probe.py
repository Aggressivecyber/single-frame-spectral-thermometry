"""Bounded development probe on the unchanged continuous-electron front."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import numpy as np
import torch
from infer import load_model
from evaluation_v2.common import Journal
from evaluation_v2.methods import state_output
from repair_v3.solver import DEFAULTS, consistent_initialize, joint_data_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(device='cuda')
    manifest_path = runtime.ROOT/'reports/audit_v2/run_001/frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fixture = next(f for f in manifest['cases'] if f['case_id'] == 'front_continuous_e')
    path = runtime.ROOT/'outputs/audit_v2/run_001'/fixture['path']
    assert runtime.sha256(path) == fixture['file_sha256']
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    raw = torch.as_tensor(arrays['expected_e'], device='cuda')[None]
    sources = {str(p.relative_to(runtime.ROOT)): runtime.sha256(p) for p in (runtime.ROOT/'repair_v3').glob('*.py')}
    runtime.write_json(args.output/'config.json', {'config': DEFAULTS, 'sources': sources, 'fixture_sha256': fixture['file_sha256']})
    journal = Journal(args.output/'steps.jsonl')
    start = time.perf_counter()
    def record(state, beta, label, info=None):
        out = state_output(model, state, beta, raw)
        S = out['species_signed_e'][0, 2].cpu().numpy()
        T = out['temperature_color_K'][0].cpu().numpy()
        row = {'stage': label, 'elapsed_s': time.perf_counter()-start,
               'template_RMSE_e': float(np.sqrt(np.mean((S-arrays['template_e'])**2))),
               'color_RMSE_K': float(np.sqrt(np.mean((T-arrays['temperature_observed_color_K'])**2))),
               'historical60_FP': float((S>60).mean()), 'solver': info}
        journal.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'solver'}), flush=True)
    with torch.no_grad():
        state, beta = model.physics.initialize(raw)
        record(state, beta, 'old_initialization')
        state, beta, warm = consistent_initialize(model.physics, raw, deadline=start+DEFAULTS['per_frame_wall_s'])
        record(state, beta, 'consistent_initialization', warm)
        damping = DEFAULTS['LM_initial']
        for i in range(DEFAULTS['outer_iterations']):
            state, beta, info = joint_data_step(model.physics, raw, state, beta, damping=damping)
            damping = info['next_damping']
            record(state, beta, f'joint_{i+1}', info)
            if time.perf_counter()-start > DEFAULTS['per_frame_wall_s']:
                raise TimeoutError('Development probe budget exhausted')
        torch.cuda.synchronize()
    print('bounded_probe_finished', flush=True)


if __name__ == '__main__':
    main()
