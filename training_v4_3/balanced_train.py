"""Same residual-scaled 500-update run; change only one sampling slot."""
from datetime import datetime, timezone
import time
import runtime
import torch
from training_v4.common import read, write, parameters_hash
from training_v4.train import checkpoint_save
from training_v4_1.common import setup, OUT as OLD, OLD_OUT, rel, PROTOCOL
from training_v4_1.cache import load
from training_v4_1.model import protective_loss
from training_v4_1.evaluate import evaluate, score_arrays
from evaluation_v2.common import Journal
from training_v4_3.residual_scaling import ResidualScaledCNN
from training_v4_3.balanced_sampling import prepare, OUT
import training_v4_3.scaled_train as shared_diagnostics
import training_v4_2.experiment as gradient_diagnostics

CONTROL=runtime.ROOT/'reports/training_v4_3/run_001/residual_scaled'
WEIGHTS=runtime.ROOT/'training_v4_3/checkpoints/negative_balanced'
INITIAL=runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
CUTOFF=datetime(2026,10,13,3,40,tzinfo=timezone.utc)


def status(stage, **values):
    write(OUT/'status.json', {'stage':stage,'time':runtime.now(),**values})


def run():
    setup()
    assert read(CONTROL/'status.json')['stage']=='COMPLETE'
    control_final=read(CONTROL/'validation_summary_0500.json')
    failures=[(g['signed_aux'],g['group']) for g in control_final['groups']
              if g['input_kind']=='noise' and g['color_RMSE_K']**2>1.05*g['physical_color_RMSE_K']**2+.04]
    assert control_final['original_gates_pass'] and control_final['protection_pass'] and control_final['weak_L2']<1
    assert not control_final['eligible'] and failures==[(True,'negative')], 'Registered condition not met'
    sampling=prepare(); draws=sampling['draws']
    assert not (OUT/'training_curve.jsonl').exists(), 'Never overwrite a run'
    calibration=read(CONTROL/'coefficient_calibration.json')
    assert not (OUT/'coefficient_calibration.json').exists()
    write(OUT/'coefficient_calibration.json',calibration)
    assert runtime.sha256(OUT/'coefficient_calibration.json')==runtime.sha256(CONTROL/'coefficient_calibration.json')
    selection=read(OLD_OUT/'backbone_selection.json')
    manifest=read(OLD/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in manifest}
    validation=[r for r in manifest if r['split']=='validation']
    reference=Journal(CONTROL/'training_curve.jsonl').rows(); assert len(reference)==500
    torch.manual_seed(PROTOCOL['seed'])
    model=ResidualScaledCNN(selection['physics_config'],calibration).cuda()
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model.load_cnn(initial['cnn_state_dict']); initial_hash=parameters_hash(model)
    assert initial_hash==read(CONTROL/'training_receipt.json')['initial_parameters_sha256']
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5); assert not optimizer.state
    buffers={k:v.clone() for k,v in model.named_buffers()}
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    protocol={**read(CONTROL/'protocol.json'),'name':'negative_balanced',
        'one_changed_variable':'slot 1 of each training update is an existing negative noisy training case',
        'paired_control':'residual_scaled at same update; identical initialization and all settings except sampling',
        'control_protocol_sha256':runtime.sha256(CONTROL/'protocol.json'),
        'draw_prefix_sha256':sampling['draw_sha256'],
        'sampling_sequence_sha256':runtime.sha256(OUT/'sampling_sequence.json'),
        'unchanged_slots_per_update':[0,2,3],
        'coefficient_calibration_sha256':runtime.sha256(OUT/'coefficient_calibration.json'),
        'registered_conditional_plan_sha256':runtime.sha256(runtime.ROOT/'reports/local_session_20261013/conditional_sampling_plan.json'),
        'new_training_cases':0,'validation_used_for_sampling':False,
        'interpretation':'Development-motivated sampling contrast on a small existing signed training pool; not independent validation.'}
    write(OUT/'protocol.json',protocol)
    extra={'schema':'physical-inverse-post-CNN-v4.3-negative-balanced','protocol':protocol,
        'coefficient_calibration':calibration,'training_source_sha256':runtime.sha256(__file__),
        'model_source_sha256':runtime.sha256(runtime.ROOT/'training_v4_3/residual_scaling.py'),
        'sampler_source_sha256':runtime.sha256(runtime.ROOT/'training_v4_3/balanced_sampling.py')}
    checksum=checkpoint_save(WEIGHTS/'initial.pt',model,optimizer,0,extra)
    # Reuse read-only diagnostic calculations, changing only their destination.
    shared_diagnostics.OUT=OUT
    status('zero_training_audit')
    shared_diagnostics.coefficients(model,validation,selection)
    result=evaluate(model,validation,selection,OUT,0,1.)
    assert result==read(CONTROL/'validation_summary_0000.json'), 'Initial control must match exactly'
    curve=Journal(OUT/'training_curve.jsonl'); valcurve=Journal(OUT/'validation_curve.jsonl')
    valcurve.append({**result,'checkpoint_path':rel(WEIGHTS/'initial.pt'),'checkpoint_sha256':checksum})
    print({'initial_audit_exact_control_match':True},flush=True)
    started=time.monotonic(); completed=0
    try:
        for update,batch in enumerate(draws,1):
            if datetime.now(timezone.utc)>=CUTOFF: raise TimeoutError('Time reserved for saving reached')
            model.train(); optimizer.zero_grad(set_to_none=True); totals={}; samples=[]
            for case in batch:
                row=by_id[case]; assert row['split']=='train'
                inputs,truth,base=load(row,selection)
                pred=model(*inputs); loss,terms=protective_loss(model,inputs[0],pred,truth,base,1.)
                assert torch.isfinite(loss),case
                (loss/4).backward()
                for k,v in {'total':loss,**terms}.items(): totals[k]=totals.get(k,0)+float(v.detach())/4
                samples.append({'case_id':case,'raw_hash':row['raw_hash'],'cache_key':base['cache_key'],
                                'input_kind':row['input_kind'],'signed_aux':row['signed_aux']})
            assert all(samples[j]==reference[update-1]['samples'][j] for j in (0,2,3))
            assert by_id[samples[1]['case_id']]['condition']==('negative_weak' if update%2 else 'negative_normal')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
            optimizer.step(); completed=update
            curve.append({'optimizer_update':update,'samples':samples,'loss':totals,
                'gradient_norm_before_clip':float(norm),'lr':3e-5,'elapsed_s':time.monotonic()-started,'status':'SUCCESS'})
            if update%25==0:
                status('training',optimizer_updates=update,limit=500)
                print({'run':'negative_balanced','update':update,'loss':totals['total']},flush=True)
            if update%100==0:
                path=WEIGHTS/f'update_{update:04d}.pt'; sha=checkpoint_save(path,model,optimizer,update,extra)
                status('validation',optimizer_updates=update)
                result=evaluate(model,validation,selection,OUT,update,1.)
                valcurve.append({**result,'checkpoint_path':rel(path),'checkpoint_sha256':sha})
                print({'validated':update,'gates':result['original_gates_pass_count'],
                    'protection':sum(g['passed'] for g in result['protection_groups']),
                    'weak_L2':result['weak_L2'],'normal_RMSE_e':result['normal_RMSE_e'],
                    'temperature_pass':result['noisy_temperature_pass'],'eligible':result['eligible']},flush=True)
        final_hash=parameters_hash(model); assert final_hash!=initial_hash
        assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
        probes=read(runtime.ROOT/'reports/training_v4_2/run_001/protocol.json')['probes']
        gradient_diagnostics.OUT=OUT
        gradient_diagnostics.gradients(model,[by_id[i] for i in probes['gradient_train_ids']],selection,'negative_balanced',500)
        shared_diagnostics.proposal_diagnostics(model,[by_id[i] for i in probes['stage_validation_ids']],selection)
        native=[]
        for update in range(0,501,100):
            path=WEIGHTS/('initial.pt' if not update else f'update_{update:04d}.pt')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            steps=sorted({int(s['step']) for s in ck['optimizer_state']['state'].values()})
            assert steps==([] if not update else [update])
            model.load_cnn(ck['cnn_state_dict']); assert parameters_hash(model)==ck['parameters_sha256']
            native.append({'update':update,'path':rel(path),'sha256':runtime.sha256(path),'Adam_steps':steps})
        targets={r['case_id']:r for r in Journal(OUT/'validation_update_0500.jsonl').rows()}
        replay=[]; model.eval()
        with torch.no_grad():
            for cond,kind in [('negative_normal','continuous'),('negative_normal','noise'),('weak','noise')]:
                row=next(r for r in validation if r['condition']==cond and r['input_kind']==kind)
                inputs,truth,_=load(row,selection); pred=model(*inputs)
                metrics=score_arrays(pred['template_e'][0].cpu().numpy(),pred['temperature_color_K'][0].cpu().numpy(),
                    {k:v[0].cpu().numpy() for k,v in truth.items()})
                diff={k:abs(metrics[k]-targets[row['case_id']]['metrics'][k]) for k in ('template_RMSE_e','color_RMSE_K')}
                assert max(diff.values())<1e-5
                replay.append({'case_id':row['case_id'],'differences':diff})
        write(OUT/'training_receipt.json',{'updates':500,'microbatches':2000,
            'unchanged_slot_comparisons':1500,'same_training_manifest_as_control':True,
            'initial_audit_exact_control_match':True,'initial_parameters_sha256':initial_hash,
            'final_parameters_sha256':final_hash,'checkpoints':native,'native_replays':replay,
            'original_physics_buffers_unchanged':True,'local_only':True,'elapsed_s':time.monotonic()-started,
            'source_sha256':runtime.sha256(__file__),'final_test_used':False})
        status('COMPLETE',optimizer_updates=500)
    except BaseException as exc:
        checkpoint_save(WEIGHTS/'interrupted.pt',model,optimizer,completed,extra)
        write(OUT/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),'updates':completed,'failed_frames_deleted':False})
        status('FAILED',message=str(exc),optimizer_updates=completed)
        raise


if __name__=='__main__':
    run()
