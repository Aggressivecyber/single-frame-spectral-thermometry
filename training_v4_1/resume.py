"""Resume a stopped trial from its validated Adam checkpoint, retaining all attempts.

The numerical training implementation and frozen sampling sequence are unchanged.
Updates after the last durable checkpoint are archived and deterministically replayed.
"""
import argparse
import shutil
import time
import numpy as np
import torch
import runtime
from training_v4.train import checkpoint_save
from training_v4_1.common import *
from training_v4_1.model import new_model, protective_loss
from training_v4_1.cache import load
from training_v4_1.sampling import build_draws
from training_v4_1.evaluate import wait_ready, evaluate


def resume(name='safe1'):
    setup()
    folder = OUT/name
    assert not (folder/'training_receipt.json').exists(), 'Trial is already complete'
    directory = DATA/'checkpoints'/name
    path = max(directory.glob('update_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    update_start = saved['optimizer_updates']
    assert all(int(s['step']) == update_start for s in saved['optimizer_state']['state'].values())
    numeric = ['common.py', 'data.py', 'cache.py', 'sampling.py', 'model.py', 'evaluate.py', 'train.py']
    assert all(runtime.sha256(runtime.ROOT/'training_v4_1'/p) == saved['sources_v4_1']['training_v4_1/'+p] for p in numeric)
    selection = read(OLD_OUT/'backbone_selection.json')
    manifest = read(OUT/'data_manifest.json')
    assert runtime.sha256(OUT/'data_manifest.json') == saved['manifest_sha256']
    rows = manifest['rows']; by_id = {r['case_id']: r for r in rows}
    draws = build_draws(rows)
    assert runtime.sha256(OUT/'sampling_sequence.json') == saved['sampling_sequence_sha256']
    validation = [r for r in rows if r['split'] == 'validation']
    original_rows = Journal(folder/'training_curve.jsonl').rows()
    original_end = len(original_rows)
    assert [r['optimizer_update'] for r in original_rows] == list(range(1, original_end+1))
    assert original_end >= update_start
    archive = folder/('interruption_'+str(len(list(folder.glob('interruption_*')))+1).zfill(2))
    archive.mkdir()
    archive_hashes = {}
    for filename in ('training_curve.jsonl', 'validation_curve.jsonl', 'failure_record.json'):
        source = folder/filename
        if source.exists():
            shutil.copy2(source, archive/filename)
            archive_hashes[filename] = runtime.sha256(source)
    # Preserve partial/newer validation outputs before replaying their checkpoint.
    for pattern in ('validation_update_*.jsonl', 'validation_summary_*.json', 'gates_update_*.json'):
        for source in folder.glob(pattern):
            if int(source.stem.split('_')[-1]) > update_start:
                shutil.move(str(source), str(archive/source.name))
                archive_hashes[source.name] = runtime.sha256(archive/source.name)
    old_validations = Journal(folder/'validation_curve.jsonl').rows()
    with (folder/'training_curve.jsonl').open('w', encoding='utf-8') as f:
        for r in original_rows[:update_start]: f.write(__import__('json').dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    with (folder/'validation_curve.jsonl').open('w', encoding='utf-8') as f:
        for r in old_validations:
            if r['optimizer_update'] <= update_start: f.write(__import__('json').dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    recovery = {'status': 'RESUMING', 'checkpoint_path': rel(path), 'checkpoint_sha256': runtime.sha256(path),
        'checkpoint_update': update_start, 'previous_logged_updates': original_end,
        'replayed_updates': original_end-update_start, 'archived_attempts': archive_hashes,
        'reason': 'External process interruption; no Python exception receipt; system reboot verified on 2026-10-12 at 19:47 Asia/Singapore',
        'Adam_state_restored': True, 'Adam_reset_again': False, 'sampling_sequence_unchanged': True,
        'numeric_sources_unchanged': True, 'failure_frames_deleted': False, 'max_replay_loss_relative_difference': 0.}
    write(archive/'recovery.json', recovery)
    torch.manual_seed(PROTOCOL['seed'])
    model = new_model(selection, saved['variant'], saved.get('reference_rms'))
    model.load_cnn(saved['cnn_state_dict'])
    assert parameters_hash(model) == saved['parameters_sha256']
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['lr'])
    optimizer.load_state_dict(saved['optimizer_state'])
    excluded = {'cnn_state_dict', 'optimizer_updates', 'optimizer_state', 'parameters_sha256', 'validation'}
    extra = {k:v for k,v in saved.items() if k not in excluded}
    extra['recovery_source_checkpoint'] = rel(path)
    extra['recovery_implementation_sha256'] = runtime.sha256(runtime.ROOT/'training_v4_1/resume.py')
    curve = Journal(folder/'training_curve.jsonl'); valcurve = Journal(folder/'validation_curve.jsonl')
    admissible = [r for r in valcurve.rows() if r['eligible']]
    best_event = min(admissible, key=lambda r:r['main_validation_loss']) if admissible else None
    best_path = runtime.ROOT/best_event['checkpoint_path'] if best_event else None
    best_loss = best_event['main_validation_loss'] if best_event else float('inf')
    elapsed_before = original_rows[update_start-1]['elapsed_s']
    start = time.monotonic()
    try:
        for update in range(update_start+1, PROTOCOL['updates']+1):
            batch = draws[update-1]
            model.train(); optimizer.zero_grad(set_to_none=True); totals = {}; samples = []
            for case_id in batch:
                row = by_id[case_id]; wait_ready(row)
                inputs, truth, base = load(row, selection); prediction = model(*inputs)
                loss, terms = protective_loss(model, inputs[0], prediction, truth, base, saved['safe_weight'])
                assert torch.isfinite(loss), case_id
                (loss/4).backward()
                for k,v in {'total':loss, **terms}.items(): totals[k] = totals.get(k,0)+float(v.detach())/4
                samples.append({'case_id':case_id, 'raw_hash':row['raw_hash'], 'cache_key':base['cache_key'],
                    'input_kind':row['input_kind'], 'signed_aux':row['signed_aux']})
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(norm)
            optimizer.step()
            if update <= original_end:
                previous = original_rows[update-1]
                assert samples == previous['samples']
                difference = max(abs(v-previous['loss'][k])/(1+abs(previous['loss'][k])) for k,v in totals.items())
                recovery['max_replay_loss_relative_difference'] = max(recovery['max_replay_loss_relative_difference'], difference)
                assert difference < 1e-4, (update, difference, 'Replay is not numerically consistent')
            curve.append({'optimizer_update':update, 'samples':samples, 'loss':totals,
                'gradient_norm_before_clip':float(norm), 'lr':3e-5,
                'elapsed_s':elapsed_before+time.monotonic()-start, 'status':'SUCCESS'})
            if update % 25 == 0: print({'run':name, 'resumed':True, 'updates':update, 'loss':totals['total']}, flush=True)
            if update % 100 == 0:
                result = evaluate(model, validation, selection, folder, update, saved['safe_weight'])
                checkpoint = directory/f'update_{update:04d}.pt'
                checksum = checkpoint_save(checkpoint, model, optimizer, update, {**extra, 'validation':result})
                valcurve.append({**result, 'checkpoint_path':rel(checkpoint), 'checkpoint_sha256':checksum})
                if result['eligible'] and result['main_validation_loss'] < best_loss:
                    best_path = checkpoint; best_loss = result['main_validation_loss']
                print({'run':name, 'validated':update, 'gates':result['original_gates_pass_count'],
                    'protection':result['protection_pass'], 'weak_L2':result['weak_L2'], 'eligible':result['eligible']}, flush=True)
                recovery['status'] = 'RUNNING'; recovery['last_validated_update'] = update
                write(archive/'recovery.json', recovery)
    except BaseException as exc:
        checkpoint_save(directory/'interrupted.pt', model, optimizer, len(curve.rows()), extra)
        write(folder/'failure_record.json', {'type':type(exc).__name__, 'message':str(exc), 'updates':len(curve.rows()), 'failed_frames_deleted':False})
        raise
    recovery['status'] = 'COMPLETE'; write(archive/'recovery.json', recovery)
    write(folder/'training_receipt.json', {'name':name, 'safe_weight':saved['safe_weight'], 'variant':saved['variant'],
        'updates':1500, 'microbatches':6000, 'validation_events':15, 'validation_cases':len(validation),
        'initial_parameters_sha256':saved['initial_parameters_sha256'], 'final_parameters_sha256':parameters_hash(model),
        'real_parameters_changed':parameters_hash(model) != saved['initial_parameters_sha256'], 'Adam_reset':True,
        'sampling_sequence_sha256':saved['sampling_sequence_sha256'],
        'final_checkpoint':rel(directory/'update_1500.pt'), 'final_sha256':runtime.sha256(directory/'update_1500.pt'),
        'selected_checkpoint':rel(best_path) if best_path else None, 'selected_main_loss':best_loss if best_path else None,
        'qualified_candidate_available':best_path is not None, 'elapsed_s':elapsed_before+time.monotonic()-start,
        'interruption_recovery':rel(archive/'recovery.json'), 'additional_replayed_updates':original_end-update_start})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--name', default='safe1')
    resume(p.parse_args().name)
