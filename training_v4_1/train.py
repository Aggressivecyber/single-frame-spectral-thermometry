import time
import numpy as np
import torch
import runtime
from training_v4.train import checkpoint_save
from training_v4_1.common import *
from training_v4_1.model import new_model,protective_loss
from training_v4_1.cache import load
from training_v4_1.sampling import build_draws
from training_v4_1.evaluate import wait_ready,evaluate

def train(name='safe1',safe_weight=1.,variant='unchanged',reference_rms=None):
    setup(); selection=read(OLD_OUT/'backbone_selection.json')
    manifest=read(OUT/'data_manifest.json'); rows=manifest['rows']; by_id={r['case_id']:r for r in rows}
    draws=build_draws(rows); validation=[r for r in rows if r['split']=='validation']
    folder=OUT/name; folder.mkdir(exist_ok=True)
    directory=DATA/'checkpoints'/name; directory.mkdir(parents=True,exist_ok=True)
    assert not (folder/'training_curve.jsonl').exists(), 'Use a new controlled-run name, never overwrite trained trials'
    torch.manual_seed(PROTOCOL['seed']); model=new_model(selection,variant,reference_rms)
    initial_checkpoint=runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
    initial=torch.load(initial_checkpoint,map_location='cpu',weights_only=False)
    model.load_cnn(initial['cnn_state_dict']); initial_hash=parameters_hash(model)
    assert initial_hash==initial['parameters_sha256']
    optimizer=torch.optim.Adam(model.parameters(),lr=PROTOCOL['lr']); assert not optimizer.state
    extra={'schema':'physical-inverse-post-CNN-v4.1-protective','name':name,'safe_weight':safe_weight,'variant':variant,'reference_rms':reference_rms,
        'base_commit':BASE,'training_v4_1':True,'initial_checkpoint_sha256':runtime.sha256(initial_checkpoint),
        'initial_parameters_sha256':initial_hash,'sampling_sequence_sha256':runtime.sha256(OUT/'sampling_sequence.json'),
        'manifest_sha256':runtime.sha256(OUT/'data_manifest.json'),'backbone_selection':selection,'sources_v4_1':sources()}
    checkpoint_save(directory/'initial.pt',model,optimizer,0,extra)
    curve=Journal(folder/'training_curve.jsonl'); valcurve=Journal(folder/'validation_curve.jsonl')
    best_path=None; best_loss=float('inf'); start=time.monotonic()
    try:
        for update,batch in enumerate(draws,1):
            model.train(); optimizer.zero_grad(set_to_none=True); totals={}; samples=[]
            for case_id in batch:
                row=by_id[case_id]; wait_ready(row)
                inputs,truth,base=load(row,selection); prediction=model(*inputs)
                loss,terms=protective_loss(model,inputs[0],prediction,truth,base,safe_weight)
                assert torch.isfinite(loss), case_id
                (loss/4).backward()
                for k,v in {'total':loss,**terms}.items(): totals[k]=totals.get(k,0)+float(v.detach())/4
                samples.append({'case_id':case_id,'raw_hash':row['raw_hash'],'cache_key':base['cache_key'],
                                'input_kind':row['input_kind'],'signed_aux':row['signed_aux']})
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
            optimizer.step()
            curve.append({'optimizer_update':update,'samples':samples,'loss':totals,
                'gradient_norm_before_clip':float(norm),'lr':3e-5,'elapsed_s':time.monotonic()-start,'status':'SUCCESS'})
            if update%25==0: print({'run':name,'updates':update,'loss':totals['total']},flush=True)
            if update%100==0:
                result=evaluate(model,validation,selection,folder,update,safe_weight)
                path=directory/f'update_{update:04d}.pt'
                checksum=checkpoint_save(path,model,optimizer,update,{**extra,'validation':result})
                valcurve.append({**result,'checkpoint_path':rel(path),'checkpoint_sha256':checksum})
                if result['eligible'] and result['main_validation_loss']<best_loss:
                    best_path=path; best_loss=result['main_validation_loss']
                print({'run':name,'validated':update,'gates':result['original_gates_pass_count'],
                    'protection':result['protection_pass'],'weak_L2':result['weak_L2'],'eligible':result['eligible']},flush=True)
    except BaseException as exc:
        checkpoint_save(directory/'interrupted.pt',model,optimizer,len(curve.rows()),extra)
        write(folder/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),'updates':len(curve.rows()),'failed_frames_deleted':False})
        raise
    write(folder/'training_receipt.json',{'name':name,'safe_weight':safe_weight,'variant':variant,
        'updates':1500,'microbatches':6000,'validation_events':15,'validation_cases':len(validation),
        'initial_parameters_sha256':initial_hash,'final_parameters_sha256':parameters_hash(model),
        'real_parameters_changed':parameters_hash(model)!=initial_hash,'Adam_reset':True,
        'sampling_sequence_sha256':extra['sampling_sequence_sha256'],
        'final_checkpoint':rel(directory/'update_1500.pt'),'final_sha256':runtime.sha256(directory/'update_1500.pt'),
        'selected_checkpoint':rel(best_path) if best_path else None,'selected_main_loss':best_loss if best_path else None,
        'qualified_candidate_available':best_path is not None,'elapsed_s':time.monotonic()-start})
    return best_path

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--name',default='safe1'); p.add_argument('--safe-weight',type=float,default=1.)
    p.add_argument('--variant',default='unchanged'); p.add_argument('--reference-rms',type=float)
    a=p.parse_args(); train(a.name,a.safe_weight,a.variant,a.reference_rms)
