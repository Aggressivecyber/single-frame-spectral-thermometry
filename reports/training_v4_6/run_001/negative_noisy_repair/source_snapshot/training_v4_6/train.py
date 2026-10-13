"""One frozen 500-update negative-noisy parent/repeat contrast.

The public import and --self-test use only the standard library.  Formal
execution requires the exact frozen protocol hash and completed generation/cache
receipts before runtime, torch, arrays or a CUDA context are opened.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/training_v4_6/run_001'
RUN = OUT / 'negative_noisy_repair'
CHECKPOINTS = ROOT / 'training_v4_6/checkpoints/negative_noisy_repair'
PREVIOUS = ROOT / 'reports/training_v4_5/run_001/balanced_fixed_continuum'
UPDATES = (0, 100, 200, 300, 400, 500)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_intervention(row):
    return row['signed_aux'] and row['condition'].startswith('negative_') and row['input_kind'] == 'noise'


def verify_position(before, after):
    """Keep every nonintervened case and all condition positions paired."""
    require(before['split'] == after['split'] == 'train', 'Only TRAIN may enter an optimizer update')
    fields = ('signed_aux', 'condition', 'input_kind')
    require(all(before[key] == after[key] for key in fields), 'Condition/input-type position changed')
    intervention = is_intervention(before)
    require(intervention or before == after, 'A nonintervened row changed')
    return intervention


def sampling_checks(rows, original_rows, sampling, reference):
    by_id = {row['case_id']: row for row in rows}
    old_by_id = {row['case_id']: row for row in original_rows}
    draws = sampling['draws']
    require(len(draws) == len(reference) == 500 and all(len(batch) == 4 for batch in draws),
            'The paired experiment requires exactly 500 by 4 draws')
    intervention_count = signed_count = unchanged_count = id_changed_count = 0
    signed_exposure = {}
    for update, (batch, old_update) in enumerate(zip(draws, reference), 1):
        require(old_update['optimizer_update'] == update and old_update['status'] == 'SUCCESS'
                and len(old_update['samples']) == 4, 'v4.5 actual training journal is incomplete')
        for case, old_sample in zip(batch, old_update['samples']):
            require(case in by_id and old_sample['case_id'] in old_by_id, 'Unknown draw case')
            before, after = old_by_id[old_sample['case_id']], by_id[case]
            require(old_sample['raw_hash'] == before['raw_hash']
                    and old_sample['signed_aux'] == before['signed_aux']
                    and old_sample['input_kind'] == before['input_kind'], 'v4.5 sample identity changed')
            changed_slot = verify_position(before, after)
            intervention_count += changed_slot
            unchanged_count += not changed_slot
            signed_count += after['signed_aux']
            id_changed_count += case != old_sample['case_id']
            if changed_slot:
                signed_exposure.setdefault(after['condition'], []).append(after)
    require(intervention_count == sampling['negative_noisy_positions_replaced'] == 513,
            'The only intervention is the 513 negative noisy positions')
    require(unchanged_count == 1487 and sampling['all_other_1487_case_ids_unchanged'] is True,
            'The other 1487 positions must stay identical')
    require(signed_count == sampling['signed_microbatches'] == 575, 'Signed total must stay 575, not 513')
    require(id_changed_count == sampling['positions_with_actual_case_id_change'] == 502
            and intervention_count - id_changed_count == sampling['resampled_positions_returning_original_case_id'] == 11,
            'Distinguish 513 resampled positions from the 502 actually different case IDs')
    exposure = []
    require(set(signed_exposure) == {'negative_weak', 'negative_normal'}, 'Intervention conditions changed')
    for condition, group in sorted(signed_exposure.items()):
        counts = Counter(row['case_id'] for row in group)
        mothers = {row['mother_id'] for row in group}
        noise_per_mother = {mother: {row.get('noise_index', 0) for row in group if row['mother_id'] == mother}
                            for mother in mothers}
        pairs = {(row['geometry'], row['template_geometry']) for row in group}
        require(len(counts) == 72 and len(mothers) == len(pairs) == 36 and min(counts.values()) >= 3,
                'Negative TRAIN coverage must be 72 cases / 36 mothers / 36 geometry pairs')
        require(all(indices == {0, 1} for indices in noise_per_mother.values()),
                'Each exposed negative mother must use both noise realizations')
        exposure.append({'condition': condition, 'draws': len(group), 'unique_cases': len(counts),
                         'mothers': len(mothers), 'geometry_pairs': len(pairs),
                         'minimum_draws_per_case': min(counts.values()), 'noise_realizations_per_mother': 2})
    return {'total_microbatches': 2000, 'signed_microbatches': signed_count,
            'negative_noisy_intervention_positions': intervention_count,
            'other_exactly_preserved_positions': unchanged_count, 'actually_different_case_ids': id_changed_count,
            'all_condition_input_kind_positions_preserved': True, 'negative_exposure': exposure}


def preflight(protocol_sha256):
    from training_v4_6 import data
    require(ROOT.resolve().drive.upper() in ('E:', 'D:'), 'Use an E/D checkout')
    for path in (OUT, RUN, CHECKPOINTS):
        require(path.resolve().is_relative_to(ROOT.resolve()) and path.resolve().drive.upper() in ('E:', 'D:'),
                'Training artifacts must remain inside the E/D checkout')
    protocol, bundle, gate = data.verify_execution('train_one_paired_arm', protocol_sha256)
    require(data.read(OUT / 'protocol.json') == protocol, 'Execution and scoring protocols differ')
    for relative, expected in protocol['frozen_inputs_sha256'].items():
        require(data.file_hash(data.rooted(relative)) == expected, 'Frozen input changed: ' + relative)
    for relative, expected in protocol['sources'].items():
        require(data.file_hash(data.rooted(relative)) == expected, 'Frozen v4.6 source changed: ' + relative)
    old_protocol = data.read(PREVIOUS.parent / 'protocol.json')
    fixed = ('seed', 'lr', 'accumulation', 'clip', 'safe_weight', 'train_updates', 'checkpoint_interval',
             'parameters', 'initial_sha256', 'calibration_sha256', 'validation_ids_sha256',
             'selection_rule', 'v4_4_additional_development_screen')
    require(all(protocol[key] == old_protocol[key] for key in fixed), 'An inherited training or gate choice changed')
    require(protocol['train_updates'] == 500 and protocol['checkpoint_interval'] == 100
            and protocol['accumulation'] == 4 and protocol['parameters'] == 5797,
            'The bounded model/update count changed')
    require(protocol['restart_from_old_initial_not_continuation'] is True and not protocol['final_test_used']
            and not protocol['locked_development_generated'], 'Execution scope changed')
    require(not RUN.exists() and not CHECKPOINTS.exists(), 'Never overwrite or resume this finite experiment')
    require(shutil.disk_usage(ROOT).free >= protocol['minimum_free_E_bytes'], 'Insufficient registered free disk budget')
    manifest_path, new_path = OUT / 'data_manifest.json', OUT / 'new_rows.json'
    generation = data.read(OUT / 'generation_receipt.json')
    cache = data.read(OUT / 'cache_receipt.json')
    manifest_hash, new_hash = data.file_hash(manifest_path), data.file_hash(new_path)
    for receipt in (generation, cache):
        require(receipt['stage'] == 'COMPLETE', 'Data generation and selected cache must both be COMPLETE')
        require(receipt['protocol_sha256'] == protocol_sha256 and receipt['merged_manifest_sha256'] == manifest_hash
                and receipt['new_rows_sha256'] == new_hash
                and receipt['sampling_sequence_sha256'] == protocol['sampling_sequence_sha256'],
                'Generation/cache receipt does not match frozen inputs')
    require((generation['new_train_rows'], generation['merged_train_rows'], generation['old_validation_rows'])
            == (432, 3008, 776), 'Generated row counts differ')
    require(cache['all_selected_new_cases_materialized'] is True and cache['physical_buffers_unchanged'] is True
            and cache['labels_enter_inversion'] is False, 'Cache provenance or scope check failed')
    rows = data.read(manifest_path)['rows']
    new_rows = data.read(new_path)['rows']
    old_rows = bundle['old']['rows']
    require(rows[:3352] == old_rows and rows[3352:] == new_rows and len(new_rows) == 432,
            'Merged manifest must retain all 3352 old rows exactly before the 432 new TRAIN rows')
    require(len(rows) == len({row['case_id'] for row in rows}) == 3784,
            'Merged count or case identity differs')
    require(all(row['split'] == 'train' and row['origin'].startswith('v4_6_') for row in new_rows),
            'New cases must all be TRAIN with explicit v4.6 origins')
    validation = [row for row in rows if row['split'] == 'validation']
    require(validation == [row for row in old_rows if row['split'] == 'validation'] and len(validation) == 776,
            'Old development rows changed')
    require(data.canonical_hash([row['case_id'] for row in validation]) == protocol['validation_ids_sha256'],
            'Old development order changed')
    require(data.file_hash(data.OLD_MANIFEST) == protocol['manifest_sha256'], 'Inherited anchor manifest changed')
    sampling = data.read(OUT / 'sampling_sequence.json')
    require(data.canonical_hash(sampling['draws']) == protocol['draws_sha256'] == sampling['draw_sha256'],
            'Frozen sampling sequence changed')
    with (PREVIOUS / 'training_curve.jsonl').open(encoding='utf-8-sig') as handle:
        reference = [json.loads(line) for line in handle if line.strip()]
    require(data.canonical_hash([[sample['case_id'] for sample in row['samples']] for row in reference])
            == protocol['source_draws_sha256'] == sampling['source_draws_sha256'], 'Source actual draws changed')
    paired = sampling_checks(rows, old_rows, sampling, reference)
    new_ids = {row['case_id'] for row in new_rows}
    selected = {case for batch in sampling['draws'] for case in batch} & new_ids
    require(len(selected) == cache['selected_case_count'] == cache['cache_record_count']
            == protocol['planned_new_unique_cache_count'] == 120, 'Exactly 120 drawn new cases must be cached')
    require(selected == set(cache['selected_case_ids']) == set(sampling['new_unique_cases_to_cache'])
            and len(cache['selected_case_ids']) == 120, 'Cache case set differs from actual draws')
    require(data.canonical_hash(cache['selected_case_ids']) == cache['selected_case_ids_sha256'],
            'Cache selected-case list changed')
    cache_records_path = OUT / 'cache_per_case.jsonl'
    require(data.file_hash(cache_records_path) == cache['cache_per_case_sha256'], 'Cache journal changed')
    with cache_records_path.open(encoding='utf-8-sig') as handle:
        cache_records = [json.loads(line) for line in handle if line.strip()]
    require(len(cache_records) == 120 and {row['case_id'] for row in cache_records} == selected,
            'Cache success journal is incomplete or duplicated')
    by_id = {row['case_id']: row for row in rows}
    for record in cache_records:
        row = by_id[record['case_id']]
        require(record['raw_hash'] == row['raw_hash'] and record['label_sha256'] == row['label_sha256'],
                'Cached raw/label identity differs')
        cache_path = data.rooted(record['cache_path'])
        require(cache_path.is_relative_to(data.DATA / 'cache'), 'A new cache is outside the v4.6 data tree')
        require(data.file_hash(cache_path) == record['cache_sha256']
                and data.file_hash(cache_path.with_suffix('.json')) == record['cache_metadata_sha256']
                and data.file_hash(data.rooted(record['score_path'])) == record['score_sha256'],
                'A materialized cache or score file changed')
    predecessor = data.read(PREVIOUS / 'training_receipt.json')
    require(predecessor['updates'] == 500 and predecessor['microbatches'] == 2000,
            'v4.5 training did not complete its registered exposure')
    checks = {**gate, **paired, 'merged_manifest_sha256': manifest_hash, 'new_rows_sha256': new_hash,
              'generation_receipt_sha256': data.file_hash(OUT / 'generation_receipt.json'),
              'cache_receipt_sha256': data.file_hash(OUT / 'cache_receipt.json'),
              'cache_per_case_sha256': cache['cache_per_case_sha256'],
              'reference_training_curve_sha256': data.file_hash(PREVIOUS / 'training_curve.jsonl'),
              'old_validation_rows_unchanged': True, 'selected_new_cache_cases': len(selected),
              'cache_and_score_byte_hashes_verified': True, 'final_test_used': False}
    return protocol, rows, validation, sampling, reference, predecessor, checks


def run(*, protocol_sha256):
    # Every operation above this boundary is metadata / file-byte verification.
    protocol, rows, validation, sampling, reference, predecessor, checks = preflight(protocol_sha256)
    runtime_root = ROOT / '_runtime'
    require(runtime_root.resolve().drive.upper() in ('E:', 'D:'), 'Runtime must remain on E/D')
    os.environ['THERMOMETRY_RUNTIME'] = str(runtime_root)
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[key] = str(runtime_root / 'tmp')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    import runtime
    import torch
    from training_v4.common import read, write, parameters_hash
    from training_v4.train import checkpoint_save
    from training_v4_1.common import setup, OLD_OUT, rel
    from training_v4_1.model import protective_loss
    from training_v4_1.evaluate import score_arrays
    from evaluation_v2.common import Journal
    from training_v4_4.evaluate import AuditedFixedContinuumCNN, evaluate
    from training_v4_6.cache import load
    from training_v4_6.protocol import INITIAL, CALIBRATION
    setup()
    require(torch.cuda.is_available(), 'This registered execution requires the root-controlled CUDA worker')
    RUN.mkdir(parents=True, exist_ok=False)
    CHECKPOINTS.mkdir(parents=True, exist_ok=False)
    write(RUN / 'input_verification.json', checks)
    selection, calibration = read(OLD_OUT / 'backbone_selection.json'), read(CALIBRATION)
    by_id = {row['case_id']: row for row in rows}
    old_by_id = {row['case_id']: row for row in rows[:3352]}
    require(runtime.sha256(INITIAL) == protocol['initial_sha256']
            and runtime.sha256(CALIBRATION) == protocol['calibration_sha256'], 'Initial/calibration bytes changed')
    snapshot = RUN / 'source_snapshot'
    source_hashes = {relative: expected for relative, expected in protocol['frozen_inputs_sha256'].items()
                     if relative.endswith('.py')}
    for relative, expected in source_hashes.items():
        source, destination = ROOT / relative, snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require(runtime.sha256(source) == runtime.sha256(destination) == expected, 'Source snapshot drift: ' + relative)
    extra = {'schema': 'physical-inverse-post-CNN-v4.6-negative-noisy-parent-repair',
             'model_name': 'FixedContinuumCNN', 'coefficient_calibration': calibration,
             'protocol': protocol, 'protocol_sha256': runtime.sha256(OUT / 'protocol.json'),
             'execution_protocol_sha256': protocol_sha256, 'training_source_sha256': runtime.sha256(__file__),
             'model_source_sha256': runtime.sha256(ROOT / 'training_v4_4/model.py'),
             'source_hashes_at_start': source_hashes, 'sampling_sequence_sha256': protocol['sampling_sequence_sha256'],
             'draws_sha256': protocol['draws_sha256'], 'merged_manifest_sha256': checks['merged_manifest_sha256'],
             'new_rows_sha256': checks['new_rows_sha256'], 'final_test_used': False}
    write(RUN / 'run_protocol.json', {**extra, 'registered_at': runtime.now(), 'paired_exposure': checks})
    curve, valcurve = Journal(RUN / 'training_curve.jsonl'), Journal(RUN / 'validation_curve.jsonl')
    started, completed, phase = time.monotonic(), 0, 'initialization'
    model = optimizer = None

    def status(stage, **values):
        write(RUN / 'status.json', {'stage': stage, 'time': runtime.now(), 'optimizer_updates': completed,
                                  'limit': 500, **values})

    def budget():
        if time.monotonic() - started > protocol['maximum_gpu_seconds_per_training_arm']:
            raise TimeoutError('Registered total GPU-arm time budget exceeded')

    def checkpoint_and_evaluate(update, initial=False):
        path = CHECKPOINTS / ('initial.pt' if initial else f'update_{update:04d}.pt')
        checkpoint_hash = checkpoint_save(path, model, optimizer, update, extra)
        status('zero_training_audit' if initial else 'validation')
        audit_started = time.monotonic()
        summary, screen = evaluate(model, validation, selection, RUN, update, protocol_path=OUT / 'protocol.json')
        if initial:
            require(time.monotonic() - audit_started <= protocol['maximum_gpu_seconds_zero_audit'],
                    'Registered zero-update audit budget exceeded')
            require(summary == read(PREVIOUS / 'validation_summary_0000.json'),
                    'Zero-update full 776-frame summary must equal v4.5 exactly')
        valcurve.append({**summary, 'science_candidate': screen['science_candidate'],
                         'checkpoint_path': rel(path), 'checkpoint_sha256': checkpoint_hash})
        budget()

    try:
        torch.manual_seed(protocol['seed'])
        model = AuditedFixedContinuumCNN(selection['physics_config'], calibration).cuda()
        require(sum(parameter.numel() for parameter in model.parameters()) == protocol['parameters'],
                'Trainable parameter count changed')
        initial = torch.load(INITIAL, map_location='cpu', weights_only=False)
        model.load_cnn(initial['cnn_state_dict'])
        initial_hash = parameters_hash(model)
        require(initial_hash == initial['parameters_sha256'] == predecessor['initial_parameters_sha256'],
                'Restart must use exactly the old shared CNN initial weights')
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol['lr'])
        require(not optimizer.state, 'Adam must start with empty state')
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        phase = 'training'
        checkpoint_and_evaluate(0, initial=True)
        preserved_samples = 0
        for update, batch in enumerate(sampling['draws'], 1):
            budget()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals, samples, pairs = {}, [], []
            for slot, case_id in enumerate(batch):
                row = by_id[case_id]
                old_sample = reference[update - 1]['samples'][slot]
                before = old_by_id[old_sample['case_id']]
                changed_slot = verify_position(before, row)
                inputs, truth, base = load(row, selection)
                prediction = model(*inputs)
                loss, terms = protective_loss(model, inputs[0], prediction, truth, base, protocol['safe_weight'])
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss: ' + case_id)
                (loss / protocol['accumulation']).backward()
                for key, value in {'total': loss, **terms}.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) / protocol['accumulation']
                sample = {'case_id': case_id, 'raw_hash': row['raw_hash'], 'cache_key': base['cache_key'],
                          'input_kind': row['input_kind'], 'signed_aux': row['signed_aux']}
                if not changed_slot:
                    require(sample == old_sample, 'Unchanged slot must retain exact raw/cache/sample identity')
                    preserved_samples += 1
                samples.append(sample)
                pairs.append({'slot': slot, 'negative_noisy_intervention': changed_slot,
                              'source_sample': old_sample, 'mother_id': row['mother_id'], 'condition': row['condition'],
                              'label_sha256': row['label_sha256'], 'cache_sha256': base['cache_sha256'],
                              'cache_metadata_sha256': base['cache_metadata_sha256']})
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['clip'])
            require(bool(torch.isfinite(gradient_norm)), 'Nonfinite gradient norm')
            optimizer.step()
            completed = update
            curve.append({'optimizer_update': update, 'samples': samples, 'paired_source_slots': pairs,
                          'loss': totals, 'gradient_norm_before_clip': float(gradient_norm), 'lr': protocol['lr'],
                          'elapsed_s': time.monotonic() - started, 'status': 'SUCCESS'})
            if update % 25 == 0:
                status('training')
                print({'run': 'negative_noisy_repair', 'update': update, 'loss': totals['total']}, flush=True)
            if update % 100 == 0:
                checkpoint_and_evaluate(update)
        require(preserved_samples == 1487, 'Did not preserve all 1487 actual sample identities')
        final_hash = parameters_hash(model)
        require(final_hash != initial_hash, 'CNN weights did not train')
        require(all(torch.equal(value, buffers[key]) for key, value in model.named_buffers()),
                'A frozen physical/model buffer changed')
        # Record two representative new TRAIN outputs before reloading the final
        # checkpoint. These checks never choose or modify weights.
        model.eval()
        new_references = {}
        selected_ids = set(sampling['new_unique_cases_to_cache'])
        with torch.no_grad():
            for condition in ('negative_weak', 'negative_normal'):
                row = next(row for row in rows if row['case_id'] in selected_ids and row['condition'] == condition)
                inputs, truth, base = load(row, selection)
                prediction = model(*inputs)
                new_references[row['case_id']] = {'inputs': [value.clone() for value in inputs],
                    'template_e': prediction['template_e'].clone(),
                    'temperature_color_K': prediction['temperature_color_K'].clone(), 'base': base}
        phase = 'checkpoint_verification'
        native = []
        for update in UPDATES:
            budget()
            path = CHECKPOINTS / ('initial.pt' if update == 0 else f'update_{update:04d}.pt')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            steps = sorted({int(value['step']) for value in checkpoint['optimizer_state']['state'].values()})
            require(checkpoint['optimizer_updates'] == update and steps == ([] if update == 0 else [update]),
                    'Native Adam step/checkpoint update mismatch')
            model.load_cnn(checkpoint['cnn_state_dict'])
            require(parameters_hash(model) == checkpoint['parameters_sha256'], 'Native CNN weight hash mismatch')
            native.append({'update': update, 'path': rel(path), 'sha256': runtime.sha256(path),
                           'Adam_steps': steps, 'parameters_sha256': checkpoint['parameters_sha256']})
        require(parameters_hash(model) == final_hash, 'Final checkpoint did not restore the final CNN')
        model.eval()
        final_rows = {row['case_id']: row for row in Journal(RUN / 'validation_update_0500.jsonl').rows()}
        replay, new_replays = [], []
        with torch.no_grad():
            for condition, kind in [('negative_normal', 'noise'), ('weak', 'noise'), ('blank', 'quantization')]:
                row = next(row for row in validation if row['condition'] == condition and row['input_kind'] == kind)
                inputs, truth, base = load(row, selection)
                prediction = model(*inputs)
                metrics = score_arrays(prediction['template_e'][0].cpu().numpy(),
                    prediction['temperature_color_K'][0].cpu().numpy(), {key: value[0].cpu().numpy() for key, value in truth.items()})
                difference = {key: abs(metrics[key] - final_rows[row['case_id']]['metrics'][key])
                              for key in ('template_RMSE_e', 'color_RMSE_K')}
                require(max(difference.values()) < 1e-5, 'Old validation native replay changed')
                require(all(base[key] == final_rows[row['case_id']]['physical'][key]
                            for key in ('raw_hash', 'label_sha256', 'cache_key', 'cache_sha256', 'cache_metadata_sha256')),
                        'Old validation replay cache identity changed')
                replay.append({'case_id': row['case_id'], 'differences': difference, 'cache_identity_unchanged': True})
            for case_id, expected in new_references.items():
                inputs, _, base = load(by_id[case_id], selection)
                prediction = model(*inputs)
                require(all(torch.equal(value, other) for value, other in zip(inputs, expected['inputs'])),
                        'New training raw/state/beta replay changed')
                require(base == expected['base'], 'New training scoring/cache identity changed')
                require(all(torch.equal(prediction[key], expected[key]) for key in ('template_e', 'temperature_color_K')),
                        'New training output changed after native checkpoint reload')
                new_replays.append({'case_id': case_id, 'raw_state_beta_exact': True, 'outputs_exact': True,
                                    'cache_key': base['cache_key'], 'cache_sha256': base['cache_sha256']})
        budget()
        for relative, expected in protocol['frozen_inputs_sha256'].items():
            require(runtime.sha256(ROOT / relative) == expected, 'Frozen input changed during training: ' + relative)
        require(runtime.sha256(OUT / 'execution_protocol.json') == protocol_sha256
                and read(OUT / 'protocol.json') == protocol, 'Frozen protocol changed during training')
        require(runtime.sha256(OUT / 'data_manifest.json') == checks['merged_manifest_sha256']
                and runtime.sha256(OUT / 'new_rows.json') == checks['new_rows_sha256']
                and runtime.sha256(OUT / 'sampling_sequence.json') == protocol['sampling_sequence_sha256'],
                'Training data/sampling descriptors changed during training')
        require(len(curve.rows()) == 500 and [row['optimizer_update'] for row in valcurve.rows()] == list(UPDATES),
                'Actual update/checkpoint journals are incomplete')
        receipt = {'updates': 500, 'microbatches': 2000, **checks,
            'same_frozen_negative_noisy_parent_repair_draws': True,
            'actual_unchanged_sample_identity_comparisons': preserved_samples,
            'initial_full_development_summary_equals_v4_5': True, 'empty_Adam_initial_state': True,
            'initial_parameters_sha256': initial_hash, 'final_parameters_sha256': final_hash,
            'checkpoints': native, 'native_replays': replay, 'new_cache_native_replays': new_replays,
            'physical_buffers_unchanged': True, 'source_hashes_at_start': source_hashes,
            'source_hashes_unchanged_at_end': True, 'source_sha256': runtime.sha256(__file__),
            'elapsed_s': time.monotonic() - started, 'final_test_used': False,
            'locked_development_generated': False, 'local_only': True,
            'claim_limit': sampling['claim_limit']}
        write(RUN / 'training_receipt.json', receipt)
        require(read(RUN / 'training_receipt.json') == receipt, 'Training receipt did not reopen exactly')
        status('COMPLETE')
        print({'training_complete': True, 'updates': 500, 'signed_microbatches': 575,
               'intervened_negative_noisy_positions': 513, 'other_unchanged_positions': 1487,
               'receipt_sha256': runtime.sha256(RUN / 'training_receipt.json')}, flush=True)
        return receipt
    except BaseException as exc:
        # Earlier-weight reloads may not be paired with the final optimizer.
        interrupted = phase == 'training' and model is not None and optimizer is not None
        checkpoint_error = None
        if interrupted:
            try:
                checkpoint_save(CHECKPOINTS / 'interrupted.pt', model, optimizer, completed, extra)
            except BaseException as save_error:
                interrupted = False
                checkpoint_error = repr(save_error)
        write(RUN / 'failure_record.json', {'type': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc(), 'updates': completed, 'phase': phase,
            'interrupted_checkpoint_written': interrupted, 'interrupted_checkpoint_error': checkpoint_error,
            'deleted_records': 0, 'protocol_sha256': protocol_sha256, 'elapsed_s': time.monotonic() - started})
        status('FAILED', message=str(exc))
        raise


def self_test():
    """Exercise scope errors with tiny descriptors; no files or arrays are used."""
    normal = {'case_id': 'old', 'split': 'train', 'signed_aux': False, 'condition': 'weak',
              'input_kind': 'noise', 'raw_hash': 'raw', 'label_sha256': 'label'}
    require(verify_position(normal, dict(normal)) is False, 'Unchanged position fixture failed')
    negative = {**normal, 'case_id': 'negative', 'signed_aux': True, 'condition': 'negative_weak'}
    changed = {**negative, 'case_id': 'new_negative', 'raw_hash': 'new_raw', 'label_sha256': 'new_label'}
    require(verify_position(negative, changed) is True, 'Allowed intervention fixture failed')
    failures = [(normal, {**normal, 'raw_hash': 'changed'}),
                ({**negative, 'condition': 'positive_weak'}, {**changed, 'condition': 'positive_weak'}),
                ({**negative, 'input_kind': 'continuous'}, {**changed, 'input_kind': 'continuous'}),
                (negative, {**changed, 'condition': 'negative_normal'}),
                (negative, {**changed, 'split': 'validation'})]
    for before, after in failures:
        try:
            verify_position(before, after)
        except ValueError:
            continue
        raise AssertionError('A prohibited position change was accepted')
    result = {'fixture_only': True, 'unchanged_identity_accepted': True, 'negative_noisy_replacement_accepted': True,
              'prohibited_mutations_rejected': len(failures), 'torch_imported': 'torch' in sys.modules,
              'runtime_imported': 'runtime' in sys.modules, 'model_inference_calls': 0, 'files_written': 0,
              'GPU_calls': 0, 'formal_execution_authorized': False}
    require(not result['torch_imported'] and not result['runtime_imported'], 'Fixture imported the training runtime')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        require(not args.execute and args.protocol_sha256 is None, 'Fixture checking is separate from execution')
        self_test()
    else:
        require(args.execute, 'Formal training requires --execute and the exact --protocol-sha256')
        run(protocol_sha256=args.protocol_sha256)
