"""Change only the inherited trunk initialization relative to v4.2, for 500 updates."""
from datetime import datetime,timezone
import time
import runtime
import numpy as np
import torch
from training_v4.common import read,write,parameters_hash
from training_v4.train import checkpoint_save
from training_v4_1.common import setup,OUT as OLD,OLD_OUT,rel,PROTOCOL
from training_v4_1.cache import load
from training_v4_1.model import new_model,protective_loss
from training_v4_1.evaluate import evaluate,score_arrays
from evaluation_v2.common import Journal,canonical_hash
import training_v4_2.experiment as diagnostic

BASE='cca766c65814f89d6f8228f3e511974fa603455e'
OUT=runtime.ROOT/'reports/training_v4_3/run_001/fresh_trunk'
WEIGHTS=runtime.ROOT/'training_v4_3/checkpoints/fresh_trunk'
REFERENCE=runtime.ROOT/'reports/training_v4_2/run_001'
LIMIT=500
CUTOFF=datetime(2026,10,13,3,40,tzinfo=timezone.utc)


def state(stage,**values):
    write(OUT/'status.json',{'stage':stage,'time':runtime.now(),**values})


def run():
    setup(); OUT.mkdir(parents=True,exist_ok=True); WEIGHTS.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'training_curve.jsonl').exists(),'Never overwrite a trained trial'
    selection=read(OLD_OUT/'backbone_selection.json')
    rows=read(OLD/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in rows}
    reference=read(REFERENCE/'protocol.json'); draws=read(OLD/'sampling_sequence.json')['draws'][:LIMIT]
    assert canonical_hash(draws)==reference['draw_prefix_sha256']
    assert runtime.sha256(OLD/'data_manifest.json')==reference['data_manifest_sha256']
    assert runtime.sha256(OLD_OUT/'backbone_selection.json')==reference['backbone_sha256']
    assert runtime.sha256(REFERENCE/'input_cache_hashes.json')==reference['input_inventory_sha256']
    previous_curve=Journal(REFERENCE/'training_curve.jsonl').rows(); assert len(previous_curve)==500
    for draw,r in zip(draws,previous_curve):
        assert draw==[s['case_id'] for s in r['samples']]
        assert all(by_id[i]['split']=='train' for i in draw)
    stages=[by_id[i] for i in reference['probes']['stage_validation_ids']]
    probes=[by_id[i] for i in reference['probes']['gradient_train_ids']]
    validation=[r for r in rows if r['split']=='validation']
    # Redirect only diagnostic artifact paths inside this isolated process.
    # No model/physics/config/training function is monkey-patched.
    diagnostic.OUT=OUT
    torch.manual_seed(PROTOCOL['seed']); model=new_model(selection)
    initial_dict=model.cnn_state_dict(); initial_hash=parameters_hash(model)
    torch.manual_seed(PROTOCOL['seed']); repeat=new_model(selection)
    assert parameters_hash(repeat)==initial_hash
    del repeat; torch.cuda.empty_cache()
    old_initial=torch.load(runtime.ROOT/'training_v4_2/checkpoints/initial.pt',map_location='cpu',weights_only=False)
    assert all(torch.count_nonzero(v)==0 for k,v in initial_dict.items() if not k.startswith('trunk.'))
    assert all(not torch.equal(v,old_initial['cnn_state_dict'][k]) for k,v in initial_dict.items() if k.startswith('trunk.'))
    frozen_buffers={k:v.clone() for k,v in model.named_buffers()}
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5); assert not optimizer.state
    protocol={'base':BASE,'name':'fresh_trunk','one_changed_variable':'random trunk instead of inherited v4 update2000 trunk',
        'updates':500,'seed':PROTOCOL['seed'],'lr':3e-5,'accumulation':4,'clip':1.,'safe_weight':1.,
        'parameters':5797,'features':19,'shared_corrections':6,'zero_residual_heads':True,
        'reference_protocol_sha256':runtime.sha256(REFERENCE/'protocol.json'),
        'input_inventory_sha256':reference['input_inventory_sha256'],
        'draw_prefix_sha256':reference['draw_prefix_sha256'],'backbone_sha256':reference['backbone_sha256'],
        'maximum_new_updates_utc':CUTOFF.isoformat(),'local_only':True,'final_test_used':False,'old_954_rerun':False,
        'estimated_seconds':1200,'automatic_extension':False}
    write(OUT/'protocol.json',protocol)
    extra={'schema':'physical-inverse-post-CNN-v4.3-fresh-trunk','base_commit':BASE,'protocol':protocol,
        'initial_parameters_sha256':initial_hash,'training_source_sha256':runtime.sha256(__file__)}
    checkpoint_save(WEIGHTS/'initial.pt',model,optimizer,0,extra)
    state('initialization_checks')
    diagnostic.initialization_checks(model,stages,selection,{'cnn_state_dict':initial_dict})
    diagnostic.gradients(model,probes,selection,'fresh_trunk',0)
    curve=Journal(OUT/'training_curve.jsonl'); valcurve=Journal(OUT/'validation_curve.jsonl')
    initial_result=evaluate(model,validation,selection,OUT,0,1.)
    assert initial_result['original_gates_pass_count']==16 and initial_result['protection_pass']
    valcurve.append({**initial_result,'checkpoint_path':rel(WEIGHTS/'initial.pt'),'checkpoint_sha256':runtime.sha256(WEIGHTS/'initial.pt')})
    completed=0; started=time.monotonic()
    try:
        for update,batch in enumerate(draws,1):
            if datetime.now(timezone.utc)>=CUTOFF: raise TimeoutError('Predeclared saving cutoff reached')
            model.train(); optimizer.zero_grad(set_to_none=True); totals={}; samples=[]
            for case in batch:
                row=by_id[case]; inputs,truth,base=load(row,selection)
                out=model(*inputs); loss,terms=protective_loss(model,inputs[0],out,truth,base,1.)
                assert torch.isfinite(loss),case
                (loss/4).backward()
                for k,v in {'total':loss,**terms}.items(): totals[k]=totals.get(k,0)+float(v.detach())/4
                samples.append({'case_id':case,'raw_hash':row['raw_hash'],'cache_key':base['cache_key'],
                                'input_kind':row['input_kind'],'signed_aux':row['signed_aux']})
            assert samples==previous_curve[update-1]['samples']
            if update==1:
                norms={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}
                assert all((v or 0)==0 for k,v in norms.items() if k.startswith('trunk.'))
                assert norms['continuum_head.weight']>0 and norms['emission_head.weight']>0
                write(OUT/'first_update_gradients.json',norms)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm); optimizer.step(); completed=update
            curve.append({'optimizer_update':update,'samples':samples,'loss':totals,'gradient_norm_before_clip':float(norm),
                          'lr':3e-5,'elapsed_s':time.monotonic()-started,'status':'SUCCESS'})
            if update%25==0:
                state('training',optimizer_updates=update,limit=500)
                print({'run':'fresh_trunk','update':update,'loss':totals['total']},flush=True)
            if update%100==0:
                path=WEIGHTS/f'update_{update:04d}.pt'; sha=checkpoint_save(path,model,optimizer,update,extra)
                state('validation',optimizer_updates=update)
                result=evaluate(model,validation,selection,OUT,update,1.)
                valcurve.append({**result,'checkpoint_path':rel(path),'checkpoint_sha256':sha})
                print({'validated':update,'gates':result['original_gates_pass_count'],
                    'protection':sum(d['passed'] for d in result['protection_groups']),
                    'weak_L2':result['weak_L2'],'normal_RMSE_e':result['normal_RMSE_e'],'eligible':result['eligible']},flush=True)
        diagnostic.gradients(model,probes,selection,'fresh_trunk',500)
        diagnostic.stages(model,stages,selection,'fresh_trunk_update0500')
        assert all(torch.equal(v,frozen_buffers[k]) for k,v in model.named_buffers())
        final_hash=parameters_hash(model); assert final_hash!=initial_hash
        # Native checkpoint/Adam re-opening and same-input replay.
        reopened=[]
        for update in range(0,501,100):
            path=WEIGHTS/('initial.pt' if not update else f'update_{update:04d}.pt')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            steps=sorted({int(s['step']) for s in ck['optimizer_state']['state'].values()})
            assert steps==([] if update==0 else [update])
            model.load_cnn(ck['cnn_state_dict']); assert parameters_hash(model)==ck['parameters_sha256']
            reopened.append({'update':update,'path':rel(path),'sha256':runtime.sha256(path),'Adam_steps':steps})
        records={r['case_id']:r for r in Journal(OUT/'validation_update_0500.jsonl').rows()}
        replays=[]; model.eval()
        with torch.no_grad():
            for condition,kind in [('negative_normal','continuous'),('weak','noise'),('blank','quantization')]:
                row=next(r for r in validation if r['condition']==condition and r['input_kind']==kind)
                inputs,truth,_=load(row,selection); prediction=model(*inputs)
                metrics=score_arrays(prediction['template_e'][0].cpu().numpy(),prediction['temperature_color_K'][0].cpu().numpy(),
                                     {k:v[0].cpu().numpy() for k,v in truth.items()})
                diff={k:abs(metrics[k]-records[row['case_id']]['metrics'][k]) for k in ('template_RMSE_e','color_RMSE_K')}
                assert max(diff.values())<1e-5
                replays.append({'case_id':row['case_id'],'differences':diff})
        write(OUT/'training_receipt.json',{'updates':500,'microbatches':2000,'same_draws_and_cache_ids_as_v4_2':True,
            'initial_parameters_sha256':initial_hash,'final_parameters_sha256':final_hash,'checkpoints':reopened,
            'native_replays':replays,'original_physics_buffers_unchanged':True,'local_only':True,
            'elapsed_s':time.monotonic()-started,'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'source_sha256':runtime.sha256(__file__),'final_test_used':False})
        state('COMPLETE',optimizer_updates=500)
    except BaseException as exc:
        checkpoint_save(WEIGHTS/'interrupted.pt',model,optimizer,completed,extra)
        write(OUT/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),'completed_updates':completed,'failed_frames_deleted':False})
        state('FAILED',optimizer_updates=completed,message=str(exc)); raise


if __name__=='__main__':run()
