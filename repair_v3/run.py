"""Frozen-input orchestration, four-combination ablation and guarded smoke."""
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import runtime
from infer import load_model, DEFAULT_CHECKPOINT
from physics import SpectralPhysics
from evaluation_v2.common import (Journal, MeasurementInput, array_hash, canonical_hash,
                                 model_fingerprints, execute_record, case_identity, tensor_digest)
from evaluation_v2.audit import load_fixture
from evaluation_v2.fixtures import freeze_file
from evaluation_v2.physics_v2 import adc_quantize
from evaluation_v2.metrics import score_prediction, calibrate_detector
from repair_v3.infer import METHODS, SELECTED, PhysicalReconstructor, prediction
from repair_v3.solver import old_step_with_subblocks, regularization
from repair_v3.protocol import PROTOCOL
from repair_v3 import SCHEMA

OLD = runtime.ROOT/'reports/audit_v2/run_001'
OLD_DATA = runtime.ROOT/'outputs/audit_v2/run_001'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    names = ['repair_next.py', 'tests/test_repair_v3.py']
    names += [str(p.relative_to(runtime.ROOT)).replace('\\', '/') for p in (runtime.ROOT/'repair_v3').glob('*.py')]
    return {p: runtime.sha256(runtime.ROOT/p) for p in sorted(names)}


def configure(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Authorized local GPU unavailable')


def original_preservation(reg):
    changed = [p for p, h in reg['original_tracked_sha256'].items() if runtime.sha256(runtime.ROOT/p) != h]
    if changed:
        raise RuntimeError('Original files changed: '+str(changed))
    return True


def guard(args):
    reg = read(args.output/'registration.json')
    current = sources()
    if reg['sources'] != current:
        # A post-gate orchestration correction excludes unregistered audit
        # resume probes. Preserve the original registration and numeric code.
        compatibility = read(args.output/'orchestration_compatibility_v3_001.json')
        assert compatibility['registration_sha256'] == runtime.sha256(args.output/'registration.json')
        assert compatibility['current_sources'] == current
        assert [p for p in current if current[p] != reg['sources'].get(p)] == ['repair_v3/run.py']
        snapshot = args.output/compatibility['original_source_snapshot']
        assert runtime.sha256(snapshot) == reg['sources']['repair_v3/run.py']
        assert compatibility['changed_function_names'] == ['guard', 'old_rows']
    assert reg['protocol_sha256'] == canonical_hash(PROTOCOL)
    assert reg['audit_v2_journal_sha256'] == runtime.sha256(OLD/'per_case.jsonl')
    assert reg['audit_v2_manifest_sha256'] == runtime.sha256(OLD/'frozen_manifest.json')
    original_preservation(reg)
    return reg


def prepare(args):
    if (args.output/'registration.json').exists():
        raise FileExistsError('Registration is immutable; reuse existing run')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime.ROOT).decode().strip() == PROTOCOL['baseline_commit']
    configure(args.device)
    model = load_model(device=args.device)
    p = model.physics
    before = model_fingerprints(model, DEFAULT_CHECKPOINT)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=runtime.ROOT).decode('utf-8').strip('\0').split('\0')
    preserved = {name: runtime.sha256(runtime.ROOT/name) for name in tracked}
    tree = ET.parse(args.output/'tests.xml').getroot()
    suites = tree.findall('testsuite') or [tree]
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    assert sum(int(s.attrib['tests']) for s in suites) >= 12
    args.data_root.mkdir(parents=True, exist_ok=True)
    old_manifest = read(OLD/'frozen_manifest.json')
    cases = []
    for case_id in ('front_continuous_e', 'front_quantization_only', 'front_old_noisy_quantized'):
        original = next(f for f in old_manifest['cases'] if f['case_id'] == case_id)
        a = load_fixture(OLD_DATA, original)
        destination = args.data_root/'fixtures'/Path(original['path']).name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(OLD_DATA/original['path'], destination)
        f = copy.deepcopy(original)
        f['path'] = str(destination.relative_to(args.data_root)).replace('\\', '/')
        f['suite'] = 'gate' if case_id != 'front_old_noisy_quantized' else 'diagnostic'
        f['metadata']['input_kind'] = 'oracle_continuous_e' if case_id == 'front_continuous_e' else 'raw_DN'
        f['metadata']['variant'] = 'continuous' if case_id == 'front_continuous_e' else ('quantization_only' if case_id == 'front_quantization_only' else 'old_noisy')
        f['metadata']['source_audit_v2_case'] = case_id
        cases.append(f)
    old_front = load_fixture(args.data_root, cases[0])
    xx, yy = p.background_basis[1], p.background_basis[2]
    pattern = torch.exp(-.5*((xx-.18).square()+((yy+.13)/.8).square())/.25**2)
    pattern /= pattern.max()
    generators = [('same_front_signal', 'old', 210.), ('same_front_weak', 'old', 45.),
                  ('same_front_negative', 'old', -120.), ('rotated70_null', 70., 0.),
                  ('rotated25_signal', 25., 210.), ('flat_null', 'flat', 0.),
                  ('flat_signal', 'flat', 210.)]
    with torch.no_grad():
        for name, geometry, peak in generators:
            z = torch.zeros((1, 5, *p.cfg['shape']), device=p.M.device, dtype=p.M.dtype)
            if geometry == 'old':
                T = torch.as_tensor(old_front['temperature_latent_K'], device=p.M.device)
            elif geometry == 'flat':
                T = torch.full_like(xx, 3000.)
            else:
                angle = geometry*np.pi/180
                coordinate = np.cos(angle)*xx+np.sin(angle)*yy
                transverse = -np.sin(angle)*xx+np.cos(angle)*yy
                T = 2950.35+300*torch.tanh(14*(coordinate-.12*torch.sin(3*transverse)))
            z[:, 0] = (T-p.T0)/p.Tscale
            z[:, 1] = (.72+.08*torch.sin(torch.pi*xx)).log()
            z[:, 4] = pattern*peak/p.Sscale
            beta = z.new_tensor([[.45, .06, -.04]])
            expected = p(z, beta)[0]
            C = p.op.object_blur(p.components(z, beta)[0])
            color, valid = p.color_temperature(C)
            target = p.op.object_blur(z[:, 4]*p.Sscale)[0]
            quantized = adc_quantize(expected, p)
            arrays = {'raw_dn': quantized.cpu().numpy(), 'expected_e': expected.cpu().numpy(),
                      'template_e': target.cpu().numpy(), 'temperature_latent_K': T.cpu().numpy(),
                      'temperature_observed_color_K': color[0].cpu().numpy(),
                      'temperature_reference_color_K': color[0].cpu().numpy(),
                      'evaluation_domain': valid[0].cpu().numpy(),
                      'uniform_temperature_domain': np.full(tuple(p.cfg['shape']), geometry == 'flat'),
                      'saturated': (quantized >= 4095).cpu().numpy()}
            for variant in ('continuous', 'quantization_only'):
                meta = {'family': name, 'condition': 'signal' if peak else 'null_clean',
                        'parent_scene_id': name, 'geometry': geometry, 'template_peak_preblur_e': peak,
                        'input_kind': 'oracle_continuous_e' if variant == 'continuous' else 'raw_DN',
                        'variant': variant, 'truth_enters_generation_and_scoring_only': True}
                cases.append(freeze_file(args.data_root, name+'_'+variant, arrays, meta, 'gate'))
    assert model_fingerprints(model, DEFAULT_CHECKPOINT) == before
    runtime.write_json(args.output/'protocol.json', PROTOCOL)
    runtime.write_json(args.output/'frozen_manifest.json', {'cases': cases,
        'counts': {'gate': 16, 'diagnostic': 1}, 'old_inputs_exact_copies': 3,
        'prior_smoke_and_validation_manifest': str(OLD/'frozen_manifest.json'),
        'audit_v2_manifest_sha256': runtime.sha256(OLD/'frozen_manifest.json')})
    registration = {'schema': SCHEMA, 'utc': runtime.now(), 'protocol_sha256': canonical_hash(PROTOCOL),
                    'sources': sources(), 'original_tracked_sha256': preserved,
                    'fingerprints': before, 'audit_v2_journal_sha256': runtime.sha256(OLD/'per_case.jsonl'),
                    'audit_v2_manifest_sha256': runtime.sha256(OLD/'frozen_manifest.json'),
                    'new_frozen_manifest_sha256': runtime.sha256(args.output/'frozen_manifest.json'),
                    'tests_xml_sha256': runtime.sha256(args.output/'tests.xml'),
                    'tests_passed': sum(int(s.attrib['tests']) for s in suites),
                    'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU',
                    'optimizer_updates': 0}
    runtime.write_json(args.output/'registration.json', registration)
    print(json.dumps({'prepared': True, 'gate_frames': 16, 'original_files_preserved': len(preserved)}), flush=True)


HISTORICAL_DETECTOR = {'pixel_threshold_e': 60., 'frame_threshold_e': 60.}


def scored_snapshot(physics, arrays, state, beta, label, solver=None):
    pred = prediction(physics, state, beta)
    raw = torch.as_tensor(arrays['expected_e'] if arrays.get('_kind') == 'oracle_continuous_e' else arrays['raw_dn'], device=physics.M.device)[None]
    if arrays.get('_kind') != 'oracle_continuous_e':
        raw = physics.raw_to_electrons(raw)
    expectation = torch.as_tensor(pred['expected_e'], device=physics.M.device)[None]
    weights = (raw.clamp_min(0)+physics.extra_variance).reciprocal()
    reg, _ = regularization(state, state.new_tensor(PROTOCOL['config']['regularization']))
    return {'label': label, 'solver': solver, 'metrics': score_prediction(arrays, pred, HISTORICAL_DETECTOR)['metrics'],
            'raw_objective': float(.5*((expectation-raw).square()*weights).sum()),
            'regularization': float(reg), 'prediction': pred}


def score_work(args, fixture, method, physics, *, trace=False, detector=HISTORICAL_DETECTOR, data_root=None):
    root = args.data_root if data_root is None else data_root
    arrays = load_fixture(root, fixture)
    kind = fixture['metadata'].get('input_kind', 'raw_DN')
    recon = PhysicalReconstructor(physics, method, PROTOCOL['config'])
    out = recon.diagnose_continuous_electrons(arrays['expected_e'], trace=trace) if kind == 'oracle_continuous_e' else recon.reconstruct(MeasurementInput(arrays['raw_dn']), trace=trace)
    if out['status'] != 'SUCCESS':
        return out
    pred = out.pop('prediction')
    snapshots = out.pop('trace')
    scored = score_prediction(arrays, pred, detector)
    filename = fixture['case_id']+'/'+method+('_known_calibration' if physics.cfg != read(args.output/'registration.json')['fingerprints']['config'] else '')
    path = args.data_root/'predictions'/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while path.with_name(path.name+f'_attempt{attempt:03d}.npz').exists():
        attempt += 1
    path = path.with_name(path.name+f'_attempt{attempt:03d}.npz')
    np.savez_compressed(path, **pred)
    trace_arrays, trace_summary = {}, []
    for i, step in enumerate(snapshots):
        step_pred = step.pop('prediction')
        trace_summary.append({**step, 'metrics': score_prediction(arrays, step_pred, detector)['metrics']})
        if i < 5 or i == len(snapshots)-1:
            for key in ('temperature_K', 'template_e', 'latent_state', 'beta', 'expected_e'):
                trace_arrays[f'{i:02d}_{key}'] = step_pred[key]
    trace_path = path.with_name(path.stem+'_trace.npz')
    if trace_arrays:
        np.savez_compressed(trace_path, **trace_arrays)
    expected = pred['expected_e']
    raw_e = arrays['expected_e'] if kind == 'oracle_continuous_e' else (arrays['raw_dn']-physics.cfg['black_level_DN'])*physics.adc_step
    return {**out, **scored, 'trace': trace_summary, 'prediction_path': str(path.relative_to(args.data_root)).replace('\\', '/'),
            'prediction_sha256': runtime.sha256(path),
            'trace_path': str(trace_path.relative_to(args.data_root)).replace('\\', '/') if trace_arrays else None,
            'trace_sha256': runtime.sha256(trace_path) if trace_arrays else None,
            'raw_RMSE_e': float(np.sqrt(np.mean((expected-raw_e)**2))),
            'physics_buffers_sha256': tensor_digest(physics.named_buffers()),
            'method_physics_config_sha256': canonical_hash(physics.cfg)}


def evaluate(args, journal, fixture, method, physics, reg, *, trace=False, detector=HISTORICAL_DETECTOR,
             data_root=None, known=False):
    root = args.data_root if data_root is None else data_root
    arrays = load_fixture(root, fixture)
    kind = fixture['metadata'].get('input_kind', 'raw_DN')
    input_hash = array_hash(arrays['expected_e'] if kind == 'oracle_continuous_e' else arrays['raw_dn'])
    numerical_config = canonical_hash({'registration_sha256': runtime.sha256(args.output/'registration.json'),
                                      'method': method, 'physics_config': physics.cfg,
                                      'detector': detector, 'known_calibration': known})
    identity = case_identity(config_hash=numerical_config, case_id=fixture['case_id'],
                             fixture_hash=fixture['file_sha256'], method=method,
                             input_kind=kind, input_hash=input_hash, run_id=args.output.name)
    if args.resume and journal.succeeded(identity):
        row = next(r for r in reversed(journal.rows()) if r['identity_sha256'] == identity and r['status'] == 'SUCCESS')
        assert runtime.sha256(args.data_root/row['prediction_path']) == row['prediction_sha256']
        return row
    base = {'schema': SCHEMA, 'case_id': fixture['case_id'], 'parent_scene_id': fixture['metadata']['parent_scene_id'],
            'method': method, 'suite': fixture['suite'], 'scene': fixture['metadata']['family'],
            'condition': fixture['metadata']['condition'], 'metadata': fixture['metadata'],
            'input_kind': kind, 'input_hash': input_hash, 'fixture_sha256': fixture['file_sha256'],
            'identity_sha256': identity, 'config_sha256': numerical_config, 'known_calibration_oracle': known}
    row = execute_record(journal, base, lambda: score_work(args, fixture, method, physics, trace=trace, detector=detector, data_root=root))
    print(json.dumps({'case': row['case_id'], 'method': method, 'known': known, 'status': row['status'],
                      'seconds': row.get('seconds'), 'T_RMSE': (row.get('metrics') or {}).get('temperature_observed_RMSE_K'),
                      'S_RMSE': (row.get('metrics') or {}).get('template_RMSE_e')}), flush=True)
    return row


def old_rows():
    from evaluation_v2.methods import METHODS as registered_methods
    active = read(OLD/'active_configuration.json')['config_sha256']
    return list({r['identity_sha256']: r for r in Journal(OLD/'per_case.jsonl').rows()
                 if r['config_sha256'] == active and r['method'] in registered_methods}.values())


def gates(args):
    reg = guard(args)
    configure(args.device)
    model = load_model(device=args.device)
    before = model_fingerprints(model, DEFAULT_CHECKPOINT)
    manifest = read(args.output/'frozen_manifest.json')
    assert runtime.sha256(args.output/'frozen_manifest.json') == reg['new_frozen_manifest_sha256']
    journal = Journal(args.output/'per_case.jsonl')
    baseline = Journal(args.output/'reused_audit_v2.jsonl')
    for fixture in manifest['cases']:
        methods = METHODS if fixture['case_id'].startswith('front_') else (SELECTED,)
        for method in methods:
            evaluate(args, journal, fixture, method, model.physics, reg, trace=True)
        if fixture['case_id'].startswith('front_'):
            arrays = load_fixture(args.data_root, fixture)
            for name in ('B2_six_stage_no_CNN', 'B4_published'):
                old = next(r for r in old_rows() if r['case_id'] == fixture['case_id'] and r['method'] == name)
                assert old['input_hash'] == array_hash(arrays['expected_e'] if old['input_kind'] == 'oracle_continuous_e' else arrays['raw_dn'])
                assert runtime.sha256(OLD_DATA/old['prediction_path']) == old['prediction_sha256']
                if not any(r['identity_sha256'] == old['identity_sha256'] for r in baseline.rows()):
                    baseline.append({**old, 'reused_from_audit_v2': True, 'old_result_file_sha256': reg['audit_v2_journal_sha256']})
            # Only the first old data-step is recomputed, solely to expose its
            # two missing subblocks. The old 12-step final result stays reused.
            subblock_path = args.output/(fixture['case_id']+'_old_first_step.json')
            if not subblock_path.exists():
                raw_array = arrays['expected_e'] if fixture['metadata']['input_kind'] == 'oracle_continuous_e' else arrays['raw_dn']
                raw = torch.as_tensor(raw_array, device=model.physics.M.device)[None]
                if fixture['metadata']['input_kind'] != 'oracle_continuous_e':
                    raw = model.physics.raw_to_electrons(raw)
                with torch.no_grad():
                    state, beta = model.physics.initialize(raw)
                    _, _, blocks = old_step_with_subblocks(model.physics, raw, state, beta)
                    scoring = {**arrays, '_kind': fixture['metadata']['input_kind']}
                    details = [scored_snapshot(model.physics, scoring, state, beta, 'initialization')]
                    details += [scored_snapshot(model.physics, scoring, z, b, label, info) for z, b, label, info in blocks]
                    for d in details:
                        d.pop('prediction')
                    runtime.write_json(subblock_path, {'case_id': fixture['case_id'], 'input_hash': array_hash(raw_array),
                                                       'old_full_final_reused': True, 'subblocks': details})
    rows = list({r['identity_sha256']: r for r in journal.rows()}.values())
    decisions = []
    for fixture in manifest['cases']:
        if fixture['suite'] != 'gate':
            continue
        row = next(r for r in rows if r['case_id'] == fixture['case_id'] and r['method'] == SELECTED)
        limits = PROTOCOL['gate'][fixture['metadata']['variant']]
        metrics = row.get('metrics') or {}
        signal = float(np.abs(load_fixture(args.data_root, fixture)['template_e']).max()) > 0
        passed = row['status'] == 'SUCCESS' and metrics.get('temperature_observed_RMSE_K', float('inf')) <= limits['temperature_RMSE_K_max'] and metrics.get('template_RMSE_e', float('inf')) <= limits['template_RMSE_e_max']
        passed = passed and (metrics.get('template_relative_L2', float('inf')) <= PROTOCOL['gate']['nonzero_template_relative_L2_max'] if signal else metrics.get('historical60_null_FP_fixed_domain') == 0.)
        decisions.append({'case_id': fixture['case_id'], 'passed': bool(passed), 'signal': signal,
                          'status': row['status'], 'T_RMSE_K': metrics.get('temperature_observed_RMSE_K'),
                          'template_RMSE_e': metrics.get('template_RMSE_e'), 'template_L2': metrics.get('template_relative_L2'),
                          'historical60_FP': metrics.get('historical60_null_FP_fixed_domain'), 'input_hash': row['input_hash']})
    unchanged = model_fingerprints(model, DEFAULT_CHECKPOINT) == before
    assert unchanged and original_preservation(reg)
    passed = len(decisions) == 16 and all(d['passed'] for d in decisions)
    receipt = {'all_physical_development_gates_passed': passed, 'required': 16,
               'passed': sum(d['passed'] for d in decisions), 'decisions': decisions,
               'method': SELECTED, 'new_method_frame_combinations': 23,
               'new_successes': sum(r['status'] == 'SUCCESS' for r in rows),
               'failed_records_retained': [r['identity_sha256'] for r in rows if r['status'] != 'SUCCESS'],
               'new_smoke_authorized_by_gate': passed, 'original_fingerprints_unchanged': unchanged,
               'optimizer_updates': 0, 'scope': 'development gate, not hardware or scientific acceptance'}
    runtime.write_json(args.output/'gate_receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('all_physical_development_gates_passed', 'passed', 'required', 'new_successes')}), flush=True)


def smoke(args):
    reg = guard(args)
    gate = read(args.output/'gate_receipt.json')
    if gate['all_physical_development_gates_passed'] is not True:
        raise RuntimeError('Physical repair gate failed; smoke is prohibited')
    configure(args.device)
    model = load_model(device=args.device)
    before = model_fingerprints(model, DEFAULT_CHECKPOINT)
    old_manifest = read(OLD/'frozen_manifest.json')
    validation = [f for f in old_manifest['cases'] if f['suite'] == 'validation' and f['metadata']['condition'].startswith('null')]
    val_journal = Journal(args.output/'validation_per_case.jsonl')
    arrays_for_detector = []
    for fixture in validation:
        row = evaluate(args, val_journal, fixture, SELECTED, model.physics, reg, data_root=OLD_DATA)
        if row['status'] != 'SUCCESS':
            raise RuntimeError('New validation detector incomplete; keep partial records')
        with np.load(args.data_root/row['prediction_path'], allow_pickle=False) as f:
            arrays_for_detector.append(f['template_e'].copy())
    detector = calibrate_detector(arrays_for_detector)
    detector['selected_only_on'] = [f['case_id'] for f in validation]
    runtime.write_json(args.output/'detector_calibration.json', detector)
    journal = Journal(args.output/'per_case.jsonl')
    reused = Journal(args.output/'reused_audit_v2.jsonl')
    fixtures = [f for f in old_manifest['cases'] if f['suite'] == 'smoke']
    old = old_rows()
    for fixture in fixtures:
        evaluate(args, journal, fixture, SELECTED, model.physics, reg, detector=detector, data_root=OLD_DATA)
        if fixture['metadata']['family'] == 'channel_gain_3pct':
            known = SpectralPhysics(fixture['metadata']['true_config']).to(args.device)
            evaluate(args, journal, fixture, SELECTED, known, reg, detector=detector, data_root=OLD_DATA, known=True)
        array = load_fixture(OLD_DATA, fixture)['raw_dn']
        for row in [r for r in old if r['case_id'] == fixture['case_id'] and not r.get('known_calibration_oracle')]:
            assert row['input_hash'] == array_hash(array)
            assert runtime.sha256(OLD_DATA/row['prediction_path']) == row['prediction_sha256']
            if not any(r['identity_sha256'] == row['identity_sha256'] for r in reused.rows()):
                reused.append({**row, 'reused_from_audit_v2': True, 'old_result_file_sha256': reg['audit_v2_journal_sha256']})
    rows = list({r['identity_sha256']: r for r in journal.rows() if r['suite'] == 'smoke'}.values())
    unchanged = model_fingerprints(model, DEFAULT_CHECKPOINT) == before
    assert unchanged and original_preservation(reg)
    receipt = {'new_smoke_nominal_and_unknown_configurations': sum(not r['known_calibration_oracle'] for r in rows),
               'new_known_calibration_oracles': sum(r['known_calibration_oracle'] for r in rows),
               'new_successful': sum(r['status'] == 'SUCCESS' for r in rows),
               'failed_or_incomplete': sum(r['status'] != 'SUCCESS' for r in rows),
               'old_smoke_method_inputs_reused': sum(r['suite'] == 'smoke' for r in reused.rows()),
               'old_full_audit_inferences_rerun': 0, 'validation_null_frames': len(validation),
               'original_fingerprints_unchanged': unchanged, 'optimizer_updates': 0}
    runtime.write_json(args.output/'smoke_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
