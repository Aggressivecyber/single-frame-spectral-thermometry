"""Same 500-update warm safe1 experiment, with only a frozen residual coefficient added."""
from datetime import datetime,timezone
import time
import runtime
import torch
from training_v4.common import read,write,parameters_hash
from training_v4.train import checkpoint_save
from training_v4_1.common import setup,OUT as OLD,OLD_OUT,rel,PROTOCOL
from training_v4_1.cache import load
from training_v4_1.model import protective_loss
from training_v4_1.evaluate import evaluate,score_arrays
from evaluation_v2.common import Journal,canonical_hash
from training_v4.model import physical_output
import training_v4_2.experiment as diagnostic
from training_v4_3.residual_scaling import ResidualScaledCNN,calibrate,ROOT as OUT

WEIGHTS=runtime.ROOT/'training_v4_3/checkpoints/residual_scaled'
INITIAL=runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
CUTOFF=datetime(2026,10,13,3,40,tzinfo=timezone.utc)


def status(stage,**values):
    write(OUT/'status.json',{'stage':stage,'time':runtime.now(),**values})


def coefficients(model,rows,selection):
    log=Journal(OUT/'validation_coefficients.jsonl'); assert not log.path.exists()
    with torch.no_grad():
        for row in rows:
            inputs,score=load(row,selection,labels=False)
            alpha,rho=model.coefficient(model.physics.raw_to_electrons(inputs[0]),inputs[1],inputs[2])
            assert abs(float(rho)-score['weighted_raw_RMS'])<1e-5
            log.append({'case_id':row['case_id'],'raw_hash':row['raw_hash'],'cache_key':score['cache_key'],
                'condition_for_reporting_only':row['condition'],'input_kind_for_reporting_only':row['input_kind'],
                'rho':float(rho),'alpha':float(alpha),'truth_used':False,'fixed_for_every_checkpoint':True})


def proposal_diagnostics(model,rows,selection):
    model.eval(); journal=Journal(OUT/'proposal_vs_scaled.jsonl')
    with torch.no_grad():
        for row in rows:
            inputs,truth,score=load(row,selection); raw_e=model.physics.raw_to_electrons(inputs[0])
            result=model(*inputs)
            proposal=physical_output(model.physics,result['unrestricted_proposal_state'],inputs[2],raw_e)
            baseline=physical_output(model.physics,inputs[1],inputs[2],raw_e)
            targets={k:v[0].cpu().numpy() for k,v in truth.items()}
            for label,pred in [('six_stage_unrestricted',proposal),('final_scaled',result)]:
                journal.append({'case_id':row['case_id'],'raw_hash':row['raw_hash'],'condition':row['condition'],
                    'input_kind':row['input_kind'],'signed_aux':row['signed_aux'],'output':label,
                    'alpha':float(result['correction_coefficient']),
                    'initial_weighted_raw_RMS':float(result['initial_weighted_raw_RMS']),
                    'valid_fraction':float(pred['valid'].float().mean()),
                    'signed_template_correction_mean_e':float((pred['template_e']-baseline['template_e']).mean()),
                    'template_correction_RMS_e':float((pred['template_e']-baseline['template_e']).square().mean().sqrt()),
                    'temperature_correction_RMS_K':float((pred['temperature_color_K']-baseline['temperature_color_K']).square().mean().sqrt()),
                    'metrics':score_arrays(pred['template_e'][0].cpu().numpy(),pred['temperature_color_K'][0].cpu().numpy(),targets)})


def run():
    setup(); c=calibrate(); WEIGHTS.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'training_curve.jsonl').exists(),'Do not overwrite any trial'
    selection=read(OLD_OUT/'backbone_selection.json')
    rows=read(OLD/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in rows}
    validation=[r for r in rows if r['split']=='validation']
    draws=read(OLD/'sampling_sequence.json')['draws'][:500]
    reference=Journal(OLD/'safe1/training_curve.jsonl').rows()[:500]
    assert all(b==[s['case_id'] for s in r['samples']] for b,r in zip(draws,reference))
    old_protocol=read(runtime.ROOT/'reports/training_v4_2/run_001/protocol.json')
    assert canonical_hash(draws)==old_protocol['draw_prefix_sha256']
    torch.manual_seed(PROTOCOL['seed']); model=ResidualScaledCNN(selection['physics_config'],c).cuda()
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model.load_cnn(initial['cnn_state_dict']); initial_hash=parameters_hash(model)
    assert initial_hash==initial['parameters_sha256']
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5); assert not optimizer.state
    physics_buffers={k:v.clone() for k,v in model.named_buffers()}
    protocol={'name':'residual_scaled','base':'cca766c65814f89d6f8228f3e511974fa603455e',
        'one_changed_variable':'fixed input-residual coefficient applied to the final six-stage CNN state increment',
        'paired_control':'v4.1 safe1 at the same updates','updates':500,'seed':4101,'lr':3e-5,'clip':1.,'accumulation':4,
        'safe_weight':1.,'parameters':5797,'features':19,'six_shared_corrections':True,
        'initial_sha256':runtime.sha256(INITIAL),'initial_parameters_sha256':initial_hash,
        'draw_prefix_sha256':canonical_hash(draws),'input_inventory_sha256':old_protocol['input_inventory_sha256'],
        'coefficient_calibration':c,'coefficient_calibration_sha256':runtime.sha256(OUT/'coefficient_calibration.json'),
        'detection_threshold_e':60,'regression_gates_changed':False,'backbone_changed':False,'local_only':True,
        'maximum_new_updates_utc':CUTOFF.isoformat(),'estimated_seconds':1200,'final_test_used':False,
        'interpretation':'Empirical protection on this synthetic development distribution; raw fit alone does not guarantee truth error.'}
    write(OUT/'protocol.json',protocol)
    extra={'schema':'physical-inverse-post-CNN-v4.3-residual-scaled','protocol':protocol,
        'coefficient_calibration':c,'training_source_sha256':runtime.sha256(__file__),
        'model_source_sha256':runtime.sha256(runtime.ROOT/'training_v4_3/residual_scaling.py')}
    checksum=checkpoint_save(WEIGHTS/'initial.pt',model,optimizer,0,extra)
    status('zero_training_audit'); coefficients(model,validation,selection)
    valcurve=Journal(OUT/'validation_curve.jsonl')
    result=evaluate(model,validation,selection,OUT,0,1.)
    valcurve.append({**result,'checkpoint_path':rel(WEIGHTS/'initial.pt'),'checkpoint_sha256':checksum})
    print({'initial_audit':True,'gates':result['original_gates_pass_count'],
           'protection':sum(g['passed'] for g in result['protection_groups']),
           'weak_L2':result['weak_L2'],'temperature_pass':result['noisy_temperature_pass'],'eligible':result['eligible']},flush=True)
    curve=Journal(OUT/'training_curve.jsonl'); completed=0; started=time.monotonic()
    try:
        for update,batch in enumerate(draws,1):
            if datetime.now(timezone.utc)>=CUTOFF: raise TimeoutError('Time reserved for saving reached')
            model.train(); optimizer.zero_grad(set_to_none=True); totals={}; samples=[]
            for case in batch:
                row=by_id[case]; inputs,truth,base=load(row,selection)
                result=model(*inputs); loss,terms=protective_loss(model,inputs[0],result,truth,base,1.)
                assert torch.isfinite(loss),case
                (loss/4).backward()
                for k,v in {'total':loss,**terms}.items(): totals[k]=totals.get(k,0)+float(v.detach())/4
                samples.append({'case_id':case,'raw_hash':row['raw_hash'],'cache_key':base['cache_key'],
                                'input_kind':row['input_kind'],'signed_aux':row['signed_aux']})
            assert samples==reference[update-1]['samples']
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
            optimizer.step(); completed=update
            curve.append({'optimizer_update':update,'samples':samples,'loss':totals,'gradient_norm_before_clip':float(norm),
                          'lr':3e-5,'elapsed_s':time.monotonic()-started,'status':'SUCCESS'})
            if update%25==0:
                status('training',optimizer_updates=update,limit=500)
                print({'run':'residual_scaled','update':update,'loss':totals['total']},flush=True)
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
        assert all(torch.equal(v,physics_buffers[k]) for k,v in model.named_buffers())
        diagnostic.OUT=OUT
        probes=[by_id[i] for i in old_protocol['probes']['gradient_train_ids']]
        diagnostic.gradients(model,probes,selection,'residual_scaled',500)
        proposal_diagnostics(model,[by_id[i] for i in old_protocol['probes']['stage_validation_ids']],selection)
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
            for cond,kind in [('negative_normal','continuous'),('weak','noise'),('blank','quantization')]:
                row=next(r for r in validation if r['condition']==cond and r['input_kind']==kind)
                inputs,truth,_=load(row,selection); out=model(*inputs)
                metrics=score_arrays(out['template_e'][0].cpu().numpy(),out['temperature_color_K'][0].cpu().numpy(),
                                     {k:v[0].cpu().numpy() for k,v in truth.items()})
                diff={k:abs(metrics[k]-targets[row['case_id']]['metrics'][k]) for k in ('template_RMSE_e','color_RMSE_K')}
                assert max(diff.values())<1e-5
                replay.append({'case_id':row['case_id'],'differences':diff})
        write(OUT/'training_receipt.json',{'updates':500,'microbatches':2000,'same_sampling_and_raw_cache_ids_as_safe1':True,
            'initial_parameters_sha256':initial_hash,'final_parameters_sha256':final_hash,'checkpoints':native,'native_replays':replay,
            'original_physics_buffers_unchanged':True,'local_only':True,'elapsed_s':time.monotonic()-started,
            'source_sha256':runtime.sha256(__file__),'final_test_used':False})
        status('COMPLETE',optimizer_updates=500)
    except BaseException as exc:
        checkpoint_save(WEIGHTS/'interrupted.pt',model,optimizer,completed,extra)
        write(OUT/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),'updates':completed,'failed_frames_deleted':False})
        status('FAILED',message=str(exc),optimizer_updates=completed); raise


if __name__=='__main__':run()
