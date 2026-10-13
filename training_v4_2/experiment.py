"""One-variable, 500-update restart using the frozen v4.1 data and sampler."""
import argparse
import json
import subprocess
import time
from collections import defaultdict

import runtime
import numpy as np
import torch
from training_v4.common import read, write, parameters_hash
from training_v4.model import physical_output
from training_v4.train import checkpoint_save
from training_v4_1.common import setup, rel, OUT as OLD, DATA as OLD_DATA, OLD_OUT, PROTOCOL as OLD_PROTOCOL
from training_v4_1.cache import load, key_for
from training_v4_1.model import new_model, protective_loss
from training_v4_1.evaluate import evaluate, score_arrays
from evaluation_v2.common import Journal, canonical_hash

OUT = runtime.ROOT/'reports/training_v4_2/run_001'
WEIGHTS = runtime.ROOT/'training_v4_2/checkpoints'
BASE = '281a81b975662c6485def535d0634d22f8b1b0b0'
INITIAL = runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
LIMIT = 500


def status(stage, **kwargs):
    write(OUT/'status.json', {'stage': stage, 'time': runtime.now(), **kwargs})


def reset_heads(model):
    with torch.no_grad():
        for head in (model.continuum_head, model.emission_head):
            head.weight.zero_()
            head.bias.zero_()


def traced_outputs(model, inputs):
    """Observe the original six steps; preserve their original stage indices."""
    raw, state, beta = inputs
    raw_e = model.physics.raw_to_electrons(raw)
    state, beta = state.detach(), beta.detach()
    yield 0, physical_output(model.physics, state, beta, raw_e)
    for stage in range(6):
        state = model.correction(raw_e, state, beta, stage)
        yield stage+1, physical_output(model.physics, state, beta, raw_e)


def choose_probes(rows):
    # Selection depends only on registered metadata, never errors/predictions.
    val = [r for r in rows if r['split'] == 'validation']
    cells = defaultdict(list)
    for r in val:
        cells[(r['signed_aux'], r['geometry'], r['condition'], r['input_kind'])].append(r)
    stage_ids = [min(rs, key=lambda r: r['case_id'])['case_id'] for _, rs in sorted(cells.items())]
    assert len(stage_ids) == 144
    train = [r for r in rows if r['split'] == 'train']
    gradient_ids = []
    for signed, conditions in [(False, ('blank', 'weak')), (True, ('negative_normal', 'positive_normal'))]:
        for condition in conditions:
            for kind in ('noise', 'continuous', 'quantization'):
                pool = [r for r in train if r['signed_aux'] == signed and r['condition'] == condition and r['input_kind'] == kind]
                gradient_ids.append(min(pool, key=lambda r: r['case_id'])['case_id'])
    assert len(gradient_ids) == 12 and not set(stage_ids) & set(gradient_ids)
    return {'stage_validation_ids': stage_ids, 'gradient_train_ids': gradient_ids,
            'selection': 'lexicographically first per metadata stratum, no score-based selection'}


def prepare():
    setup(); OUT.mkdir(parents=True, exist_ok=True); WEIGHTS.mkdir(parents=True, exist_ok=True)
    selection = read(OLD_OUT/'backbone_selection.json')
    manifest = read(OLD/'data_manifest.json')
    rows = manifest['rows']; by_id = {r['case_id']: r for r in rows}
    draws = read(OLD/'sampling_sequence.json')['draws'][:LIMIT]
    assert len(draws) == LIMIT and all(len(b) == 4 for b in draws)
    reference = Journal(OLD/'safe1/training_curve.jsonl').rows()[:LIMIT]
    assert len(reference) == LIMIT
    for update, (draw, old) in enumerate(zip(draws, reference), 1):
        assert old['optimizer_update'] == update
        assert draw == [s['case_id'] for s in old['samples']]
        for case, sample in zip(draw, old['samples']):
            row = by_id[case]
            assert row['split'] == 'train' and sample['raw_hash'] == row['raw_hash']
            assert sample['cache_key'] == key_for(row, selection)
        assert [by_id[i]['input_kind'] for i in draw] == ['noise','noise','continuous','quantization']
    probes = choose_probes(rows)
    used = sorted({i for batch in draws for i in batch} | set(probes['gradient_train_ids']) |
                  {r['case_id'] for r in rows if r['split'] == 'validation'})
    inventory_path = OUT/'input_cache_hashes.json'
    if not inventory_path.exists():
        status('hash_verification', cases=len(used))
        inventory = []
        checked = {}
        for i, case in enumerate(used):
            row = by_id[case]; score_path = OLD_DATA/'physical_scores'/(case+'.json')
            score = read(score_path)
            assert score['raw_hash'] == row['raw_hash'] and score['cache_key'] == key_for(row, selection)
            assert score['label_sha256'] == row['label_sha256']
            files = {row['raw_path']: row['raw_file_sha256'], row['label_path']: row['label_sha256'],
                     score['cache_path']: score['cache_sha256'],
                     str((runtime.ROOT/score['cache_path']).with_suffix('.json').relative_to(runtime.ROOT)).replace('\\','/'): score['cache_metadata_sha256']}
            for path, expected in files.items():
                if path not in checked: checked[path] = runtime.sha256(runtime.ROOT/path)
                assert checked[path] == expected, path
            inventory.append({'case_id':case, 'split':row['split'], 'raw_hash':row['raw_hash'],
                              'cache_key':score['cache_key'], 'files':files, 'physical_score_sha256':runtime.sha256(score_path)})
            if (i+1)%250 == 0: print({'hashed_cases': i+1, 'total':len(used)}, flush=True)
        write(inventory_path, inventory)
    if not (OUT/'preserved_files.json').exists():
        tracked = subprocess.check_output(['git','ls-tree','-r','--name-only',BASE], cwd=runtime.ROOT).decode().splitlines()
        write(OUT/'preserved_files.json', {p:runtime.sha256(runtime.ROOT/p) for p in tracked})
    protocol = {'base':BASE, 'updates':LIMIT, 'initial_checkpoint':rel(INITIAL),
        'initial_sha256':runtime.sha256(INITIAL), 'change':'zero both residual heads; retain v4 update2000 trunk',
        'architecture_parameters':5797, 'features':19, 'shared_corrections':6,
        'lr':3e-5, 'accumulation':4, 'clip':1., 'safe_weight':1., 'Adam_reset':True,
        'seed':OLD_PROTOCOL['seed'], 'sampling_sequence_sha256':runtime.sha256(OLD/'sampling_sequence.json'),
        'draw_prefix_sha256':canonical_hash(draws), 'data_manifest_sha256':runtime.sha256(OLD/'data_manifest.json'),
        'backbone_sha256':runtime.sha256(OLD_OUT/'backbone_selection.json'),
        'input_inventory_sha256':runtime.sha256(inventory_path), 'validation_events':[0,100,200,300,400,500],
        'gradient_diagnostic_updates':[0,100,300,500], 'final_test_used':False, 'old_954_rerun':False,
        'training_extended_automatically':False, 'probes':probes}
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json') == protocol
    else: write(OUT/'protocol.json', protocol)
    return selection, rows, by_id, draws, protocol


def gradients(model, rows, selection, name, update):
    """Read-only autograd probes. No optimizer step, no gradient accumulation."""
    path = OUT/f'gradients/{name}_{update:04d}.jsonl'
    if path.exists():
        assert len(Journal(path).rows()) == len(rows)
        return
    journal = Journal(path); old_mode = model.training; model.train()
    params = tuple(model.parameters())
    for row in rows:
        inputs, truth, base = load(row, selection)
        out = model(*inputs); _, terms = protective_loss(model, inputs[0], out, truth, base, 1.)
        ga = torch.autograd.grad(terms['main'], params, retain_graph=True, allow_unused=True)
        gb = torch.autograd.grad(terms['safe_total'], params, allow_unused=True)
        a = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,ga)])
        b = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gb)])
        an,bn = float(a.norm()),float(b.norm()); combined=float((a+b).norm())
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        journal.append({'case_id':row['case_id'], 'split':row['split'], 'condition':row['condition'],
            'input_kind':row['input_kind'], 'raw_hash':row['raw_hash'], 'checkpoint':name, 'update':update,
            'main_loss':float(terms['main'].detach()), 'safe_loss':float(terms['safe_total'].detach()),
            'main_gradient_norm':an, 'safe_gradient_norm':bn, 'combined_gradient_norm':combined,
            'cosine':float(torch.dot(a,b)/(a.norm()*b.norm())) if an>0 and bn>0 else None,
            'single_sample_clip_factor':min(1.,1./combined) if combined else 1.,
            'note':'individual diagnostic gradients, distinct from actual four-sample accumulated clipping'})
    model.train(old_mode)


def stages(model, rows, selection, name):
    path = OUT/f'stages/{name}.jsonl'
    if path.exists():
        assert len(Journal(path).rows()) == len(rows)*7
        return
    journal = Journal(path); model.eval()
    with torch.no_grad():
        for i, row in enumerate(rows):
            inputs, truth, base = load(row, selection)
            targets = {k:v[0].cpu().numpy() for k,v in truth.items()}
            first = previous = None
            for stage, out in traced_outputs(model, inputs):
                if first is None: first = out
                record = {k:row[k] for k in ('case_id','mother_id','condition','geometry','input_kind','signed_aux','raw_hash')}
                record.update(checkpoint=name, stage=stage,
                    metrics=score_arrays(out['template_e'][0].cpu().numpy(),out['temperature_color_K'][0].cpu().numpy(),targets),
                    template_correction_RMS_e=float((out['template_e']-first['template_e']).square().mean().sqrt()),
                    temperature_correction_RMS_K=float((out['temperature_color_K']-first['temperature_color_K']).square().mean().sqrt()),
                    raw_weighted_RMS=float((out['raw_residual_e'].square()/(model.physics.raw_to_electrons(inputs[0]).clamp_min(0)+model.physics.extra_variance)).mean().sqrt()),
                    state_increment_RMS=[float(t) for t in (out['state']-(previous or out)['state']).square().mean((0,2,3)).sqrt()],
                    physical_score_identity=base['identity'], truth_used_for_scoring_only=True)
                journal.append(record); previous = out
            native=model(*inputs)
            assert torch.equal(out['state'],native['state']), 'Trace must match original six-stage forward'
            if (i+1)%48 == 0: print({'stage_probe':name,'cases':i+1},flush=True)


def initialization_checks(model, rows, selection, initial):
    receipt = {'parameters':sum(p.numel() for p in model.parameters()),'trunk_equal':True,'zero_heads':True,'identity_cases':[]}
    for name, value in model.cnn_state_dict().items():
        if name.startswith('trunk.'): assert torch.equal(value, initial['cnn_state_dict'][name])
        else: assert torch.count_nonzero(value) == 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            inputs, _, _ = load(row, selection)
            expected = physical_output(model.physics,inputs[1],inputs[2],model.physics.raw_to_electrons(inputs[0]))
            out = model(*inputs)
            errors={k:float((out[k]-expected[k]).abs().max()) for k in ('state','temperature_color_K','template_e','band_images_e','expected_e')}
            assert max(errors.values()) == 0, (row['case_id'], errors)
            receipt['identity_cases'].append({'case_id':row['case_id'],'max_abs_difference':errors})
    write(OUT/'initialization_checks.json',receipt)


def load_old(selection, update=None):
    path = INITIAL if update is None else runtime.ROOT/f'training_v4_1/checkpoints/safe1/update_{update:04d}.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=new_model(selection); model.load_cnn(checkpoint['cnn_state_dict'])
    assert parameters_hash(model)==checkpoint['parameters_sha256']
    return model, checkpoint


def verify_reference(selection, by_id, draw):
    model,_=load_old(selection); model.train(); model.zero_grad(set_to_none=True)
    totals={}
    for case in draw:
        inputs,truth,base=load(by_id[case],selection)
        loss,terms=protective_loss(model,inputs[0],model(*inputs),truth,base,1.)
        (loss/4).backward()
        for k,v in {'total':loss,**terms}.items(): totals[k]=totals.get(k,0)+float(v.detach())/4
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf')))
    expected=Journal(OLD/'safe1/training_curve.jsonl').rows()[0]
    differences={k:abs(v-expected['loss'][k])/max(abs(v),abs(expected['loss'][k]),1e-12) for k,v in totals.items()}
    differences['gradient_norm']=abs(norm-expected['gradient_norm_before_clip'])/max(norm,1e-12)
    assert max(differences.values())<1e-4,differences
    write(OUT/'warm_control_reuse_check.json',{'first_batch_loss_relative_differences':differences,
        'matched':True,'optimizer_steps':0,'full_prefix_draws_raw_cache_hashes_matched':True})
    del model; torch.cuda.empty_cache()


def run():
    selection, rows, by_id, draws, protocol = prepare()
    assert not (OUT/'training_curve.jsonl').exists(), 'Use explicit recovery; do not overwrite prior updates'
    val=[r for r in rows if r['split']=='validation']
    stage_rows=[by_id[i] for i in protocol['probes']['stage_validation_ids']]
    gradient_rows=[by_id[i] for i in protocol['probes']['gradient_train_ids']]
    status('pretraining_diagnostics')
    verify_reference(selection,by_id,draws[0])
    for name,update in [('v4_update2000',None),('safe1_update1500',1500),('safe1_update0500',500)]:
        old,_=load_old(selection,update)
        stages(old,stage_rows,selection,name)
        if update != 1500: gradients(old,gradient_rows,selection,'warm_control',update or 0)
        del old; torch.cuda.empty_cache()
    torch.manual_seed(OLD_PROTOCOL['seed'])
    model,initial=load_old(selection); reset_heads(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5); assert not optimizer.state
    initial_hash=parameters_hash(model)
    initialization_checks(model,stage_rows,selection,initial)
    extra={'schema':'physical-inverse-post-CNN-v4.2-identity-restart', 'base_commit':BASE,
        'name':'identity_heads_safe1','initial_parameters_sha256':initial_hash,'protocol':protocol,
        'sources':{rel(p):runtime.sha256(p) for p in sorted((runtime.ROOT/'training_v4_2').glob('*.py'))}}
    checkpoint_save(WEIGHTS/'initial.pt',model,optimizer,0,extra)
    gradients(model,gradient_rows,selection,'identity_restart',0)
    stages(model,stage_rows,selection,'identity_initial')
    initial_result=evaluate(model,val,selection,OUT,0,1.)
    assert initial_result['original_gates_pass_count']==16 and initial_result['protection_pass']
    curve=Journal(OUT/'training_curve.jsonl'); valcurve=Journal(OUT/'validation_curve.jsonl')
    valcurve.append({**initial_result,'checkpoint_path':rel(WEIGHTS/'initial.pt'),'checkpoint_sha256':runtime.sha256(WEIGHTS/'initial.pt')})
    start=time.monotonic(); completed=0
    try:
        for update,batch in enumerate(draws,1):
            model.train(); optimizer.zero_grad(set_to_none=True); totals={}; samples=[]
            for case in batch:
                row=by_id[case]; inputs,truth,base=load(row,selection)
                out=model(*inputs); loss,terms=protective_loss(model,inputs[0],out,truth,base,1.)
                assert torch.isfinite(loss),case
                (loss/4).backward()
                for k,v in {'total':loss,**terms}.items(): totals[k]=totals.get(k,0)+float(v.detach())/4
                samples.append({'case_id':case,'raw_hash':row['raw_hash'],'cache_key':base['cache_key'],
                                'input_kind':row['input_kind'],'signed_aux':row['signed_aux']})
            if update==1:
                grad_check={name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters()}
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert all((v or 0)==0 for k,v in grad_check.items() if k.startswith('trunk.'))
                assert grad_check['continuum_head.weight']>0 and grad_check['emission_head.weight']>0
                write(OUT/'first_update_gradients.json',{'norms':grad_check,'zero_trunk_at_zero_heads_expected':True})
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
            optimizer.step(); completed=update
            curve.append({'optimizer_update':update,'samples':samples,'loss':totals,'gradient_norm_before_clip':float(norm),
                          'clip_factor':min(1.,1./float(norm)) if float(norm) else 1.,'lr':3e-5,'elapsed_s':time.monotonic()-start,'status':'SUCCESS'})
            if update%25==0:
                status('training',optimizer_updates=update,limit=LIMIT)
                print({'update':update,'loss':totals['total'],'gradient_norm':float(norm)},flush=True)
            if update%100==0:
                # Durable weights first, then validation; interruption cannot discard completed training.
                path=WEIGHTS/f'update_{update:04d}.pt'; checksum=checkpoint_save(path,model,optimizer,update,extra)
                status('validation',optimizer_updates=update)
                result=evaluate(model,val,selection,OUT,update,1.)
                valcurve.append({**result,'checkpoint_path':rel(path),'checkpoint_sha256':checksum})
                print({'validated':update,'gates':result['original_gates_pass_count'],
                       'protection':sum(g['passed'] for g in result['protection_groups']),
                       'weak_L2':result['weak_L2'],'eligible':result['eligible']},flush=True)
                if update in (100,300,500):
                    gradients(model,gradient_rows,selection,'identity_restart',update)
                    if update!=500:
                        old,_=load_old(selection,update)
                        gradients(old,gradient_rows,selection,'warm_control',update)
                        del old; torch.cuda.empty_cache()
        stages(model,stage_rows,selection,'identity_update0500')
        steps=sorted({int(v['step']) for v in optimizer.state.values()})
        assert steps==[LIMIT] and parameters_hash(model)!=initial_hash
        write(OUT/'training_receipt.json',{'updates':LIMIT,'microbatches':LIMIT*4,'Adam_steps':steps,
            'initial_parameters_sha256':initial_hash,'final_parameters_sha256':parameters_hash(model),
            'final_checkpoint':rel(WEIGHTS/'update_0500.pt'),'final_sha256':runtime.sha256(WEIGHTS/'update_0500.pt'),
            'old_control_updates_retrained':0,'same_draws_as_safe1':True,'elapsed_s':time.monotonic()-start,
            'final_test_used':False,'limit_respected':True})
        status('training_complete',optimizer_updates=LIMIT)
    except BaseException as exc:
        checkpoint_save(WEIGHTS/'interrupted.pt',model,optimizer,completed,extra)
        write(OUT/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),'completed_updates':completed,'failed_frames_deleted':False})
        status('failed',optimizer_updates=completed,message=str(exc)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only: prepare()
    else: run()
