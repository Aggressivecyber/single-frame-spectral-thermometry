"""One finite 500-update paired final-continuum intervention, no final-test access."""
import shutil
import time
import runtime
import torch
from training_v4.common import read, write, parameters_hash
from training_v4.train import checkpoint_save
from training_v4_1.common import setup, OUT as OLD, OLD_OUT, rel
from training_v4_1.cache import load
from training_v4_1.model import protective_loss
from training_v4_1.evaluate import score_arrays
from evaluation_v2.common import Journal, canonical_hash
from training_v4_4.protocol import OUT, WEIGHTS, INITIAL, CALIBRATION, register
from training_v4_4.evaluate import AuditedFixedContinuumCNN, evaluate

RUN = OUT/'fixed_continuum'
CHECKPOINTS = WEIGHTS/'fixed_continuum'


def status(stage, **values):
    write(RUN/'status.json', {'stage': stage, 'time': runtime.now(), **values})


def run():
    setup()
    protocol = register()
    assert read(OUT/'zero_audit/status.json')['stage']=='COMPLETE'
    assert not (RUN/'training_curve.jsonl').exists(), 'Never overwrite a trial'
    cuda_tests = (OUT/'gpu_model_tests.log').read_text(encoding='utf-8-sig')
    assert '"passed": 5' in cuda_tests and '"device": "cuda"' in cuda_tests
    selection = read(OLD_OUT/'backbone_selection.json')
    rows = read(OLD/'data_manifest.json')['rows']; by_id = {r['case_id']:r for r in rows}
    validation = [r for r in rows if r['split']=='validation']
    draws = read(OLD/'sampling_sequence.json')['draws'][:500]
    reference = Journal(OLD/'safe1/training_curve.jsonl').rows()[:500]
    assert canonical_hash(draws)==protocol['draws_sha256']
    assert all(batch==[s['case_id'] for s in r['samples']] for batch,r in zip(draws,reference))
    assert all(by_id[i]['split']=='train' for batch in draws for i in batch)
    assert runtime.sha256(INITIAL)==protocol['initial_sha256']
    assert runtime.sha256(CALIBRATION)==protocol['calibration_sha256']
    calibration = read(CALIBRATION)
    torch.manual_seed(protocol['seed'])
    model = AuditedFixedContinuumCNN(selection['physics_config'], calibration).cuda()
    initial = torch.load(INITIAL, map_location='cpu', weights_only=False)
    model.load_cnn(initial['cnn_state_dict'])
    initial_hash = parameters_hash(model); assert initial_hash==initial['parameters_sha256']
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['lr']); assert not optimizer.state
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    snapshot = RUN/'source_snapshot'; snapshot.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for path in sorted((runtime.ROOT/'training_v4_4').glob('*.py')):
        shutil.copy2(path, snapshot/path.name)
        source_hashes[rel(path)] = runtime.sha256(path)
        assert runtime.sha256(snapshot/path.name)==source_hashes[rel(path)]
    extra = {'schema': 'physical-inverse-post-CNN-v4.4-fixed-final-continuum',
        'model_name': 'FixedContinuumCNN', 'coefficient_calibration': calibration,
        'protocol': protocol, 'protocol_sha256': runtime.sha256(OUT/'protocol.json'),
        'training_source_sha256': runtime.sha256(__file__),
        'model_source_sha256': runtime.sha256(runtime.ROOT/'training_v4_4/model.py'),
        'source_hashes_at_start': source_hashes, 'final_test_used': False}
    write(RUN/'run_protocol.json', {**extra, 'registered_at': runtime.now()})
    started = time.monotonic(); completed = 0; phase = 'training'
    curve = Journal(RUN/'training_curve.jsonl'); valcurve = Journal(RUN/'validation_curve.jsonl')
    try:
        status('zero_training_audit', optimizer_updates=0, limit=500)
        sha = checkpoint_save(CHECKPOINTS/'initial.pt', model, optimizer, 0, extra)
        summary, screen = evaluate(model, validation, selection, RUN, 0)
        valcurve.append({**summary, 'science_candidate': screen['science_candidate'],
                         'checkpoint_path': rel(CHECKPOINTS/'initial.pt'), 'checkpoint_sha256': sha})
        for update,batch in enumerate(draws, 1):
            if time.monotonic()-started > protocol['maximum_gpu_seconds_per_training_arm']:
                raise TimeoutError('Registered 1800-second training budget exceeded')
            model.train(); optimizer.zero_grad(set_to_none=True); totals = {}; samples = []
            for case_id in batch:
                row = by_id[case_id]; inputs,truth,base = load(row,selection)
                prediction = model(*inputs)
                loss,terms = protective_loss(model,inputs[0],prediction,truth,base,protocol['safe_weight'])
                assert torch.isfinite(loss),case_id
                (loss/protocol['accumulation']).backward()
                for key,value in {'total':loss,**terms}.items():
                    totals[key] = totals.get(key,0)+float(value.detach())/protocol['accumulation']
                samples.append({'case_id':case_id,'raw_hash':row['raw_hash'],'cache_key':base['cache_key'],
                                'input_kind':row['input_kind'],'signed_aux':row['signed_aux']})
            assert samples==reference[update-1]['samples']
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip'])
            assert torch.isfinite(norm)
            optimizer.step(); completed=update
            curve.append({'optimizer_update':update,'samples':samples,'loss':totals,
                'gradient_norm_before_clip':float(norm),'lr':protocol['lr'],
                'elapsed_s':time.monotonic()-started,'status':'SUCCESS'})
            if update%25==0:
                status('training',optimizer_updates=update,limit=500)
                print({'run':'fixed_continuum','update':update,'loss':totals['total']},flush=True)
            if update%100==0:
                path=CHECKPOINTS/f'update_{update:04d}.pt'; sha=checkpoint_save(path,model,optimizer,update,extra)
                status('validation',optimizer_updates=update,limit=500)
                summary,screen=evaluate(model,validation,selection,RUN,update)
                valcurve.append({**summary,'science_candidate':screen['science_candidate'],
                    'checkpoint_path':rel(path),'checkpoint_sha256':sha})
        final_hash=parameters_hash(model); assert final_hash!=initial_hash
        assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
        phase='checkpoint_verification'
        native=[]
        for update in range(0,501,100):
            path=CHECKPOINTS/('initial.pt' if update==0 else f'update_{update:04d}.pt')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            steps=sorted({int(s['step']) for s in ck['optimizer_state']['state'].values()})
            assert steps==([] if update==0 else [update])
            model.load_cnn(ck['cnn_state_dict']); assert parameters_hash(model)==ck['parameters_sha256']
            native.append({'update':update,'path':rel(path),'sha256':runtime.sha256(path),
                           'Adam_steps':steps,'parameters_sha256':ck['parameters_sha256']})
        model.eval(); reference_final={r['case_id']:r for r in Journal(RUN/'validation_update_0500.jsonl').rows()}
        replay=[]
        with torch.no_grad():
            for condition,kind in [('negative_normal','noise'),('weak','noise'),('blank','quantization')]:
                row=next(r for r in validation if r['condition']==condition and r['input_kind']==kind)
                inputs,truth,_=load(row,selection); prediction=model(*inputs)
                metrics=score_arrays(prediction['template_e'][0].cpu().numpy(),prediction['temperature_color_K'][0].cpu().numpy(),
                                     {k:v[0].cpu().numpy() for k,v in truth.items()})
                differences={k:abs(metrics[k]-reference_final[row['case_id']]['metrics'][k])
                             for k in ('template_RMSE_e','color_RMSE_K')}
                assert max(differences.values())<1e-5
                replay.append({'case_id':row['case_id'],'differences':differences})
        write(RUN/'training_receipt.json', {'updates':500,'microbatches':2000,'same_original_500_draws':True,
            'initial_parameters_sha256':initial_hash,'final_parameters_sha256':final_hash,
            'checkpoints':native,'native_replays':replay,'physical_buffers_unchanged':True,
            'elapsed_s':time.monotonic()-started,'source_hashes_at_start':source_hashes,
            'source_sha256':runtime.sha256(__file__),'final_test_used':False,'local_only':True})
        status('COMPLETE',optimizer_updates=500,limit=500)
    except BaseException as exc:
        # Reload verification can temporarily pair an earlier model with the
        # final optimizer.  Such a pair must never become a resume checkpoint.
        if phase=='training':
            checkpoint_save(CHECKPOINTS/'interrupted.pt',model,optimizer,completed,extra)
        write(RUN/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),'updates':completed,
            'phase':phase,'interrupted_checkpoint_written':phase=='training','deleted_records':0})
        status('FAILED',optimizer_updates=completed,message=str(exc)); raise


if __name__=='__main__':
    run()
