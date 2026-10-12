"""Actual Adam training, checkpointed CNN activations and validation-only choice."""
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
import runtime
from training_v4.common import OUT, DATA, setup, read, write, parameters_hash, source_hashes
from training_v4.model import PhysicsPostInverseCNN, supervised_loss, physical_output
from training_v4.cache import load_tensors
from training_v4.workers import CachePool
from training_v4.protocol import PROTOCOL
from evaluation_v2.common import Journal


def checkpoint_save(path, model, optimizer, update, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'schema': PROTOCOL['schema'], 'model_name': PROTOCOL['model_name'],
                'unfolded_training': False, 'cnn_state_dict': model.cnn_state_dict(),
                'physics_config': model.cfg, 'optimizer_updates': update,
                'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
                'parameters_sha256': parameters_hash(model), **extra}, temporary)
    temporary.replace(path)
    return runtime.sha256(path)


def evaluate(model, rows, selection, pool=None, *, output=None, update=None):
    model.eval()
    losses, records = [], []
    with torch.no_grad():
        for row in rows:
            if pool:
                pool.ensure(row)
            inputs, truth, cache = load_tensors(row, selection)
            prediction = model(*inputs)
            loss, components = supervised_loss(model, inputs[0], prediction, truth)
            s_error = prediction['template_e']-truth['template_e']
            t_error = prediction['temperature_color_K']-truth['temperature_color_K']
            energy = float(truth['template_e'].square().sum())
            record = {'case_id': row['case_id'], 'mother_id': row['mother_id'], 'condition': row['condition'],
                'clean_identity': row['clean_identity'], 'raw_hash': row['raw_hash'], 'cache_key': cache['cache_key'],
                'optimizer_update': update, 'loss': float(loss), 'template_SSE': float(s_error.square().sum()),
                'template_truth_energy': energy, 'pixels': s_error.numel(),
                'template_RMSE_e': float(s_error.square().mean().sqrt()),
                'template_relative_L2': float((s_error.square().sum()/energy).sqrt()) if energy else None,
                'color_RMSE_K': float(t_error.square().mean().sqrt()),
                'blank_FP60': float((prediction['template_e'] > 60).float().mean()) if energy == 0 else None,
                'components': {k: float(v) for k, v in components.items()}, 'status': 'SUCCESS'}
            assert np.isfinite(record['loss'])
            records.append(record)
            losses.append(float(loss))
    if output:
        journal = Journal(output)
        for record in records:
            journal.append(record)
    return float(np.mean(losses)), records


def protect_development_gates(model, selection):
    from training_v4.common import npz
    from evaluation_v2.common import Journal
    records = Journal(OUT/'backbone_gate_per_case.jsonl').rows()
    chosen = [r for r in records if r['method'] == selection['method'] and r['config'] == selection['solver_config'] and r['status'] == 'SUCCESS']
    chosen = list({r['case_id']: r for r in chosen}.values())
    assert len(chosen) == 16
    fixtures = read(runtime.ROOT/'reports/repair_v3/run_001/frozen_manifest.json')['cases']
    out = []
    with torch.no_grad():
        for r in chosen:
            source = next(f for f in fixtures if f['case_id'] == r['case_id'])
            label = npz(runtime.ROOT/'outputs/repair_v3/run_001'/source['path'])
            cache = npz(DATA/r['cache_path'])
            e = torch.as_tensor(cache['raw_e'], device='cuda')[None]
            z, beta = torch.as_tensor(cache['state'], device='cuda')[None], torch.as_tensor(cache['beta'], device='cuda')[None]
            pred = model.forward_electrons(e, z, beta)
            S, T = pred['template_e'][0].cpu().numpy(), pred['temperature_color_K'][0].cpu().numpy()
            se = np.square(S-label['template_e'], dtype=np.float64)
            te = np.square(T-label['temperature_observed_color_K'], dtype=np.float64)
            energy = float(np.square(label['template_e'], dtype=np.float64).sum())
            rs, rt = float(np.sqrt(se.mean())), float(np.sqrt(te.mean()))
            continuous = source['metadata']['variant'] == 'continuous'
            l2 = float(np.sqrt(se.sum()/energy)) if energy else None
            fp = float(np.mean(S > 60)) if not energy else None
            passed = rt <= (1 if continuous else 3) and rs <= (5 if continuous else 6) and (l2 <= .25 if energy else fp == 0)
            out.append({'case_id': r['case_id'], 'color_RMSE_K': rt, 'template_RMSE_e': rs,
                        'template_relative_L2': l2, 'blank_FP60': fp, 'passed': passed})
    return all(r['passed'] for r in out), out


def overfit_eight(selection):
    manifest = read(OUT/'data_manifest.json')
    rows = [r for r in manifest['rows'] if r['split'] == 'train' and not r['clean_identity']
            and r['noise_index'] == 0 and int(r['mother_id'].rsplit('_', 1)[-1]) in (0, 7)]
    assert len(rows) == 8
    pool = CachePool(rows, selection)
    for row in rows:
        pool.ensure(row)
    torch.manual_seed(901)
    model = PhysicsPostInverseCNN(selection['physics_config']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    initial = parameters_hash(model)
    before, _ = evaluate(model, rows, selection)
    curve = Journal(OUT/'overfit_eight_curve.jsonl')
    for update in range(1, 201):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for j in range(4):
            row = rows[(4*(update-1)+j) % 8]
            inputs, truth, _ = load_tensors(row, selection)
            out = model(*inputs)
            loss, _ = supervised_loss(model, inputs[0], out, truth)
            assert torch.isfinite(loss)
            (loss/4).backward()
            total += float(loss.detach())/4
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        if update % 20 == 0:
            score, _ = evaluate(model, rows, selection)
            curve.append({'optimizer_update': update, 'training_loss': total, 'eight_sample_loss': score,
                          'gradient_norm_before_clip': float(norm)})
            print({'eight_sample_overfit_update': update, 'loss': score}, flush=True)
    after, records = evaluate(model, rows, selection, output=OUT/'overfit_eight_per_case.jsonl', update=200)
    new_hash = parameters_hash(model)
    saved = DATA/'checkpoints/overfit_eight.pt'
    checksum = checkpoint_save(saved, model, optimizer, 200, {'purpose': 'gradient/overfit diagnostic only, not formal initialization'})
    receipt = {'samples': [r['case_id'] for r in rows], 'before_loss': before, 'after_loss': after,
               'loss_ratio': after/before, 'real_updates': 200, 'parameters_changed': initial != new_hash,
               'new_checkpoint_path': str(saved), 'checkpoint_sha256': checksum,
               'passed': after < before*.75 and initial != new_hash,
               'used_for_formal_initialization': False}
    write(OUT/'overfit_eight_receipt.json', receipt)
    pool.finish()
    if not receipt['passed']:
        raise RuntimeError('Eight-sample diagnostic failed; retain trained checkpoint and diagnose tensor path')
    return receipt


def train_seed(seed=101):
    """One complete 2,000-update trial; matching inversions are prefetched, not trained."""
    setup()
    selection = read(OUT/'backbone_selection.json')
    manifest = read(OUT/'data_manifest.json')
    train_rows = [r for r in manifest['rows'] if r['split'] == 'train']
    val_rows = [r for r in manifest['rows'] if r['split'] == 'validation']
    assert len(train_rows) == 2304 and len(val_rows) == 576
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Predeclare all sample draws. Ordering cache work by first use allows real
    # training to start while the remaining fixed inversions are being written.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(train_rows), size=(2000, 4))
    priority, seen = [], set()
    first_draws = list(draws[:100].ravel())
    rest_draws = list(draws[100:].ravel())
    for row in [train_rows[i] for i in first_draws]+val_rows+[train_rows[i] for i in rest_draws]+train_rows:
        if row['case_id'] not in seen:
            seen.add(row['case_id'])
            priority.append(row)
    pool = CachePool(priority, selection)
    model = PhysicsPostInverseCNN(selection['physics_config']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    initial_hash = parameters_hash(model)
    directory = DATA/'checkpoints'/f'seed_{seed}'
    report = OUT/f'seed_{seed}'
    report.mkdir(parents=True, exist_ok=True)
    updates = Journal(report/'training_curve.jsonl')
    validations = Journal(report/'validation_curve.jsonl')
    best_score, best_path, best_update = float('inf'), None, None
    start = time.perf_counter()
    extra = {'seed': seed, 'backbone_selection': selection,
             'data_manifest_sha256': runtime.sha256(OUT/'data_manifest.json'),
             'sources': source_hashes(), 'initial_parameters_sha256': initial_hash,
             'formal_training': True, 'training_mothers': 256, 'validation_mothers': 64,
             'final_test_evaluations': 0, 'activation_checkpointing': True}
    checkpoint_save(directory/'initial_zero_heads.pt', model, None, 0, extra)
    try:
        for update in range(1, 2001):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals, batches = {}, []
            for micro, index in enumerate(draws[update-1]):
                row = train_rows[int(index)]
                pool.ensure(row)
                inputs, truth, cache = load_tensors(row, selection)
                out = model(*inputs)
                loss, components = supervised_loss(model, inputs[0], out, truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss, frame retained: '+row['case_id'])
                (loss/4).backward()
                for name, value in {'total': loss, **components}.items():
                    totals[name] = totals.get(name, 0.)+float(value.detach())/4
                batches.append({'case_id': row['case_id'], 'raw_hash': row['raw_hash'], 'cache_key': cache['cache_key']})
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('Nonfinite gradient; no failed batch removed')
            optimizer.step()
            updates.append({'optimizer_update': update, 'microbatches': 4, 'lr': 1e-4,
                'loss': totals, 'gradient_norm_before_clip': float(grad_norm),
                'elapsed_s': time.perf_counter()-start, 'samples': batches,
                'status': 'SUCCESS', 'parameters_changed_from_initial': parameters_hash(model) != initial_hash if update == 1 else True})
            if update % 20 == 0:
                print({'seed': seed, 'optimizer_update': update, 'training_loss': totals['total'],
                       'cached_cases': pool.done, 'elapsed_s': time.perf_counter()-start}, flush=True)
            if update % 100 == 0:
                validation_path = report/f'validation_update_{update:04d}.jsonl'
                score, records = evaluate(model, val_rows, selection, pool, output=validation_path, update=update)
                gates_pass, gates = protect_development_gates(model, selection)
                path = directory/f'update_{update:04d}.pt'
                checksum = checkpoint_save(path, model, optimizer, update,
                    {**extra, 'validation_loss': score, 'development_gates_passed': gates_pass})
                if gates_pass and score < best_score:
                    best_score, best_path, best_update = score, path, update
                write(report/f'gates_update_{update:04d}.json', {'passed': gates_pass, 'decisions': gates})
                validations.append({'optimizer_update': update, 'validation_loss': score,
                    'validation_cases': 576, 'development_gates_passed': gates_pass,
                    'checkpoint_path': str(path), 'checkpoint_sha256': checksum,
                    'accepted_for_selection': gates_pass, 'final_test_used': False})
                print({'seed': seed, 'validated_update': update, 'validation_loss': score,
                       'gates_pass': gates_pass, 'best_update': best_update}, flush=True)
        pool.finish()
    except BaseException as exc:
        path = directory/'interrupted_last.pt'
        checksum = checkpoint_save(path, model, optimizer, len(updates.rows()), extra)
        write(report/'failure_record.json', {'exception_type': type(exc).__name__, 'message': str(exc),
            'optimizer_updates_saved': len(updates.rows()), 'checkpoint': str(path), 'sha256': checksum,
            'failed_frames_deleted': False, 'final_test_used': False})
        raise
    write(report/'training_receipt.json', {'seed': seed, 'optimizer_updates': 2000,
        'microbatch_forwards': 8000, 'validation_events': 20, 'validation_cases_each_event': 576,
        'real_parameters_changed': parameters_hash(model) != initial_hash,
        'initial_parameters_sha256': initial_hash, 'final_parameters_sha256': parameters_hash(model),
        'final_checkpoint': str(directory/'update_2000.pt'),
        'final_checkpoint_sha256': runtime.sha256(directory/'update_2000.pt'),
        'selected_checkpoint': str(best_path) if best_path else None, 'selected_update': best_update,
        'selected_validation_loss': best_score if best_path else None,
        'selection_constraints': 'full validation loss with required development-gate admissibility',
        'all_training_and_validation_raw_cached': pool.done == len(priority),
        'final_test_used_for_selection': False, 'old_954_rerun': False,
        'model_name': PROTOCOL['model_name'], 'unfolded_training': False,
        'elapsed_s': time.perf_counter()-start})
    print({'formal_training_complete': True, 'seed': seed, 'updates': 2000,
           'selected_update': best_update, 'best_path': str(best_path)}, flush=True)
    return best_path
