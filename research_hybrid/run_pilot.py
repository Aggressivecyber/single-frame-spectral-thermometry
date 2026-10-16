"""One registered server run; no historical/final payload paths are opened."""
import runtime
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import numpy as np
import torch
from physics import SpectralPhysics
from training_v4.scalar_engine import invert_single
from .scenes import mother_state
from .model import prepare, ConditionalProblem, ResidualNet, infer

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'research_hybrid'
SPEC = json.loads((CODE/'protocol.json').read_text())
OUT = ROOT/'outputs'/SPEC['run_id']
REPORT = ROOT/'reports'/SPEC['run_id']


def load_physics(device):
    selection = json.loads((CODE/'backbone_selection.json').read_text())
    p32 = SpectralPhysics(selection['physics_config'])
    p32.load_state_dict(torch.load(CODE/'calibration.pt', map_location='cpu', weights_only=True))
    p64 = SpectralPhysics(selection['physics_config'])
    p64.load_state_dict(p32.state_dict())
    for module in p64.modules():
        for key, value in module._buffers.items():
            if value is not None:
                module._buffers[key] = value.to(torch.complex128 if value.is_complex() else torch.float64)
    return p32.to(device), p64.to(device), selection


def append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False)+'\n')
        handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SPEC['training']['seed'])
    torch.cuda.manual_seed_all(SPEC['training']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(SPEC['budgets']['GPU_memory_limit_MiB']*2**20/torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    calls = 0
    registration = json.loads((REPORT/'REGISTRATION.json').read_text())
    for name, digest in registration['source_sha256'].items():
        if runtime.sha256(ROOT/name) != digest:
            raise ValueError('Source/calibration hash mismatch: '+name)
    for name in ('CUDA_CONTRACTS.json', 'DATA_CONTRACTS.json'):
        if not json.loads((REPORT/name).read_text())['all_passed']:
            raise ArithmeticError('Required server preflight failed: '+name)

    def status(stage, **details):
        value = {'time': runtime.now(), 'stage': stage, 'pid': os.getpid(),
                 'calls': calls, 'elapsed_s': time.monotonic()-started, **details}
        runtime.write_json(REPORT/'STATUS.json', value)
        print(json.dumps(value), flush=True)

    def account(method, case):
        nonlocal calls
        if calls >= SPEC['budgets']['reservation_calls']:
            raise RuntimeError('Registered pilot call reservation exhausted')
        calls += 1
        append(REPORT/'inverse_calls.jsonl', {'time': runtime.now(), 'phase': SPEC['run_id'],
               'method': method, 'case_id': case, 'reused': False,
               'pilot_call_number': calls,
               'new_call_number': SPEC['budgets']['prior_charged_calls']+calls})

    def budget():
        if time.monotonic()-started > SPEC['budgets']['maximum_wall_seconds']:
            raise TimeoutError('Registered pilot time budget reached')
        if shutil.disk_usage(ROOT).free < SPEC['budgets']['remote_free_floor_bytes']:
            raise RuntimeError('Remote free-space floor reached')
        total = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
        if total > SPEC['budgets']['remote_new_output_limit_bytes']:
            raise RuntimeError('Pilot output budget reached')

    p32, p, selection = load_physics(args.device)
    status('STARTED', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
           sealed_final_opened=False, source_verified=True)
    rows = []
    geometries = ['front', 'multipeak', 'ring', 'stripe', 'flat', 'smooth_random']
    templates = ['blob', 'multipeak', 'ring', 'stripe', 'filament', 'smooth_random']
    for index in range(SPEC['train_mothers']+SPEC['validation_mothers']):
        split = 'train' if index < SPEC['train_mothers'] else 'validation'
        seed = SPEC['seed_base']+index*1009
        desc = {'mother_id': f'{SPEC["run_id"]}_{split}_{index:03d}', 'split': split,
                'temperature_seed': seed+1, 'amplitude_seed': seed+29,
                'background_seed': seed+71, 'template_seed': seed+53,
                'temperature_geometry': geometries[index % len(geometries)],
                'template_geometry': templates[index % len(templates)]}
        base_state, beta, shape, provenance = mother_state(p, desc)
        for condition, peak in enumerate(SPEC['signed_peaks_e']):
            for variant in range(SPEC['continuum_variants']):
                budget()
                case = f'{desc["mother_id"]}_level{condition}_continuum{variant}'
                with torch.no_grad():
                    state = base_state.clone()
                    if variant:
                        state[:, 1] += np.log(.75)
                    state[:, 4] = peak*shape/p.Sscale
                    radiance = p.radiance(state, beta)
                    if not bool(torch.isfinite(radiance).all() and (radiance >= 0).all()):
                        raise ArithmeticError('Negative or nonfinite total per-band photon expectation: '+case)
                    expected = p.op(radiance)[0]
                    if not bool(torch.isfinite(expected).all() and (expected >= 0).all()):
                        raise ArithmeticError('Invalid physical photon mean: '+case)
                    noise_seed = seed+100000+condition*101+variant
                    generator = torch.Generator(device=args.device).manual_seed(noise_seed)
                    measured = torch.poisson(expected, generator=generator)
                    measured += torch.randn(expected.shape, device=args.device, dtype=expected.dtype,
                                            generator=generator)*p.cfg['read_noise_e']
                    quantized = (measured/p.adc_step+p.cfg['black_level_DN']).round()
                    if bool((quantized >= 4095).any() or (quantized < 0).any()):
                        raise ArithmeticError('ADC clipping in generated frame: '+case)
                    raw = quantized.to(torch.uint16)
                    raw_e = p.raw_to_electrons(raw.double()[None])
                    account('frozen_v3', case)
                    v3_state, v3_beta, history, wall = invert_single(
                        p32, p32.raw_to_electrons(raw.float()[None]), selection['method'], selection['solver_config'])
                    if wall > selection['solver_config']['per_frame_wall_s']:
                        raise TimeoutError('Frozen v3 per-frame budget exceeded: '+case)
                    account('fixed_continuum_profile', case)
                    prepared = prepare(p, raw_e, v3_state)
                    prepared = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in prepared.items()}
                    torch.save(prepared, OUT/(case+'_input.pt'))
                    truth_C = p.components(state, beta)[0]
                    truth_T, domain = p.color_temperature(p.op.object_blur(truth_C))
                    labels = {'template': p.op.object_blur(state[:, 4]*p.Sscale)[0].cpu(),
                              'temperature': truth_T[0].cpu(), 'domain': domain[0].cpu()}
                    torch.save(labels, OUT/(case+'_label.pt'))
                    torch.save({'raw_dn': raw.cpu(), 'v3_state': v3_state.cpu(), 'v3_beta': v3_beta.cpu()},
                               OUT/(case+'_raw_baseline.pt'))
                    append(REPORT/'baseline_trace.jsonl', {'case_id': case, 'wall_s': wall,
                           'history': history, 'profile': prepared['profile_info'], 'tangent': prepared['tangent_info']})
                    row = {'case_id': case, **desc, 'condition': condition, 'peak_e': peak,
                           'continuum_variant': variant, 'noise_seed': noise_seed, **provenance,
                           'input_sha256': runtime.sha256(OUT/(case+'_input.pt')),
                           'raw_baseline_sha256': runtime.sha256(OUT/(case+'_raw_baseline.pt')),
                           'label_sha256': runtime.sha256(OUT/(case+'_label.pt'))}
                    rows.append(row)
                    append(REPORT/'manifest.jsonl', row)
                status('PREPARING_DATA', frames=len(rows), mother=index+1, case_id=case)

    train = [r for r in rows if r['split'] == 'train' and r['continuum_variant'] == 0]
    generator = np.random.default_rng(SPEC['training']['seed'])
    # Balanced signs/nulls: each block of 40 visits every train mother/level.
    schedule = []
    while len(schedule) < SPEC['training']['updates']:
        schedule.extend(generator.permutation(len(train)).tolist())
    model = ResidualNet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SPEC['training']['learning_rate'])

    def open_input(case):
        data = torch.load(OUT/(case+'_input.pt'), map_location=args.device, weights_only=True)
        problem = ConditionalProblem(p, data['weights'], data['target'], data['x0'])
        return data, problem

    for step, choice in enumerate(schedule[:SPEC['training']['updates']], 1):
        budget()
        row = train[choice]
        optimizer.zero_grad(set_to_none=True)
        outputs, losses = [], []
        for variant in range(2):
            case = row['case_id'][:-1]+str(variant)
            data, problem = open_input(case)
            label = torch.load(OUT/(case+'_label.pt'), map_location=args.device, weights_only=True)
            account('hybrid_training_forward', f'{case}_update{step:03d}')
            prediction, _, _ = infer(model, problem, data['features'])
            outputs.append(prediction)
            losses.append(((prediction-label['template'])/60.).square().mean())
        loss = torch.stack(losses).mean()+.25*((outputs[0]-outputs[1])/60.).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError('Nonfinite training loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SPEC['training']['gradient_norm_limit'],
                                                 error_if_nonfinite=True)
        optimizer.step()
        append(REPORT/'training.jsonl', {'update': step, 'loss': float(loss.detach()),
               'gradient_norm': float(grad_norm), 'mother_id': row['mother_id'], 'condition': row['condition']})
        if step == 1 or step % 10 == 0:
            status('TRAINING', update=step, loss=float(loss.detach()),
                   peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20)
    torch.save({'state_dict': model.state_dict(), 'updates': SPEC['training']['updates'],
                'source_registration_sha256': runtime.sha256(REPORT/'REGISTRATION.json')}, OUT/'last_checkpoint.pt')
    model.eval()
    metrics = []
    with torch.no_grad():
        for row in [r for r in rows if r['split'] == 'validation']:
            budget()
            case = row['case_id']
            data, problem = open_input(case)
            label = torch.load(OUT/(case+'_label.pt'), map_location=args.device, weights_only=True)
            base = problem.linear.U.unpack(data['x0'][:-3])
            account('matched_extra_physics', case)
            control = problem.linear.U.unpack(problem.solve(data['x0'][:-3])[:-3])
            account('hybrid_evaluation', case)
            result, x, proposal = infer(model, problem, data['features'])
            account('same_network_proposal_no_DC', case)
            outputs = {'v3': data['v3_template'], 'fixed_profile': base,
                       'matched_control': control, 'same_network_proposal_no_DC': problem.linear.U.unpack(proposal),
                       'hybrid': result}
            for method, value in outputs.items():
                error = value-label['template']
                metrics.append({'case_id': case, 'mother_id': row['mother_id'], 'peak_e': row['peak_e'],
                                'continuum_variant': row['continuum_variant'], 'method': method,
                                'pixels': value.numel(), 'template_squared_error_sum': float(error.square().sum()),
                                'template_bias_e': float(error.mean()), 'template_max_abs_e': float(value.abs().max()),
                                'temperature_squared_error_sum': float((data['temperature']-label['temperature']).square().sum()),
                                'prediction_truth_inner_product': float((value*label['template']).sum()),
                                'truth_squared_sum': float(label['template'].square().sum())})
            max_control_change = float((control-base).abs().max())
            if max_control_change > .001:
                raise ArithmeticError('Matched extra solve unexpectedly changes the converged baseline')
            torch.save({'predictions': {k: v.cpu() for k, v in outputs.items()},
                        'temperature_fixed': data['temperature'].cpu(),
                        'temperature_hybrid': data['temperature'].cpu(),
                        'predicted_conditional_raw': problem.linear.apply(x).cpu(),
                        'numerical_info': problem.last_solve}, OUT/(case+'_evaluation.pt'))
    summary = []
    for method in outputs:
        for group, peaks in [('all', SPEC['signed_peaks_e']),
                             ('null', [0.]), ('weak', [30., -30.]), ('larger_signed', [60., -60.])]:
            subset = [r for r in metrics if r['method'] == method and r['peak_e'] in peaks]
            pixels = sum(r['pixels'] for r in subset)
            truth_norm = sum(r['truth_squared_sum'] for r in subset)
            summary.append({'method': method, 'group': group, 'frames': len(subset), 'mothers': 2,
                            'pixels': pixels, 'template_RMSE_e': (sum(r['template_squared_error_sum'] for r in subset)/pixels)**.5,
                            'color_RMSE_K': (sum(r['temperature_squared_error_sum'] for r in subset)/pixels)**.5,
                            'signed_gain': sum(r['prediction_truth_inner_product'] for r in subset)/truth_norm if truth_norm else None,
                            'mean_bias_e': sum(r['template_bias_e'] for r in subset)/len(subset)})
    runtime.write_json(REPORT/'EVALUATION.json', {'rows': metrics, 'summary': summary,
                       'temperature_unchanged': True, 'final_test_opened': False,
                       'detection': 'NOT_CALIBRATED', 'scientific_acceptance': 'NOT_ASSESSED',
                       'same_network_no_DC_is_independently_trained_ablation': False})
    budget()
    if calls != SPEC['budgets']['reservation_calls']:
        raise ArithmeticError('Actual call count differs from the registered completed pilot')
    runtime.write_json(REPORT/'COMPLETION.json', {'status': 'COMPLETED_ENGINEERING_PILOT',
                       'time': runtime.now(), 'calls': calls, 'optimizer_updates': SPEC['training']['updates'],
                       'raw_frames': len(rows), 'validation_mothers': 2,
                       'temperature_unchanged': True, 'final_test_opened': False,
                       'checkpoint_sha256': runtime.sha256(OUT/'last_checkpoint.pt'),
                       'evaluation_sha256': runtime.sha256(REPORT/'EVALUATION.json'),
                       'wall_s': time.monotonic()-started,
                       'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20})
    status('COMPLETED_ENGINEERING_PILOT')


if __name__ == '__main__':
    main()
