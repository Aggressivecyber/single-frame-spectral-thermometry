"""Same-input baseline bank and pairing of every separate trained checkpoint."""
import numpy as np
import torch
import json
import os
from infer import load_model,DEFAULT_CHECKPOINT
from evaluation_v2.methods import demod_fit
from training_v4.model import PhysicsPostInverseCNN,physical_output
from training_v4_1.common import *
from training_v4_1.cache import load
from training_v4_1.evaluate import wait_ready,score_arrays

BASELINES=['zero','old_B4','fixed_physics','v4_update100','v4_update2000','tikhonov_validation_selected']

def baseline_bank():
    setup(); selection=read(OLD_OUT/'backbone_selection.json')
    rows=[r for r in read(OUT/'data_manifest.json')['rows'] if r['split']=='validation']
    lam=read(OUT/'tikhonov_selection.json')['selected_lambda_decoder']
    old=load_model(device='cuda').eval(); m100=PhysicsPostInverseCNN(selection['physics_config']).cuda().eval()
    m2000=PhysicsPostInverseCNN(selection['physics_config']).cuda().eval()
    m100.load_cnn(torch.load(runtime.ROOT/'training_v4/checkpoints/seed_101_validation_selected.pt',map_location='cpu',weights_only=False)['cnn_state_dict'])
    m2000.load_cnn(torch.load(runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt',map_location='cpu',weights_only=False)['cnn_state_dict'])
    previous_old={ (r['case_id'],r['method']):r for r in Journal(OLD_OUT/'seed_101/paired_validation.jsonl').rows() if r['status']=='SUCCESS'}
    mapping={'zero':'zero_template','old_B4':'old_B4','fixed_physics':'repair_physics_validation_selected','v4_update100':'trained_CNN'}
    journal=Journal(OUT/'baseline_per_case.jsonl'); previous={r['identity']:r for r in journal.rows()}
    with torch.no_grad():
        for i,row in enumerate(rows):
            wait_ready(row); inputs,score=load(row,selection,labels=False)
            raw,z,beta=inputs; raw_e=old.physics.raw_to_electrons(raw); target=npz(file_path(row,'label_path'))
            for method in BASELINES:
                identity=canonical_hash({'raw':row['raw_hash'],'label':row['label_sha256'],'cache':score['cache_key'],
                    'method':method,'decoder_lambda':lam,'version':BASE})
                if identity in previous: continue
                base={k:row[k] for k in ('case_id','mother_id','condition','group','geometry','input_kind','signed_aux','raw_hash','label_sha256')}
                try:
                    reused=False; old_row=previous_old.get((row['case_id'],mapping.get(method)))
                    if old_row and old_row['raw_hash']==row['raw_hash'] and old_row['label_sha256']==row['label_sha256'] and old_row['cache_key']==score['cache_key']:
                        path=OLD_DATA/old_row['prediction_path']; assert runtime.sha256(path)==old_row['prediction_sha256']
                        out=npz(path); S=out['template_e']; T=out.get('temperature_color_K'); reused=True
                    elif method=='zero': S=np.zeros_like(target['template_e']); T=None
                    else:
                        if method=='old_B4': out=old(raw[:,None]); S=out['species_signed_e'][0,2]
                        elif method=='fixed_physics': out=physical_output(old.physics,z,beta,raw_e); S=out['template_e'][0]
                        elif method=='tikhonov_validation_selected': out=demod_fit(old.physics,raw_e,regularization=lam); S=out['species_signed_e'][0,2]
                        else: out=(m100 if method=='v4_update100' else m2000)(*inputs); S=out['template_e'][0]
                        S=S.cpu().numpy(); T=out['temperature_color_K'][0].cpu().numpy()
                    assert np.isfinite(S).all() and (T is None or np.isfinite(T).all())
                    journal.append({**base,'identity':identity,'method':method,'status':'SUCCESS',
                        'metrics':score_arrays(S,T,target),'cache_key':score['cache_key'],
                        'reused_exact_v4_prediction':reused,'detection_threshold_e':60.,
                        'lambda_decoder':lam if method=='tikhonov_validation_selected' else None})
                except Exception as exc:
                    journal.append({**base,'identity':identity,'method':method,'status':'EXCEPTION','message':str(exc),'type':type(exc).__name__})
            if (i+1)%32==0: print({'baseline_frames':i+1,'total':len(rows)},flush=True)
    return journal.rows()

def pair_all():
    baselines=Journal(OUT/'baseline_per_case.jsonl').rows()
    lookup={(r['case_id'],r['method']):r for r in baselines}
    receipts=[]
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder/'training_receipt.json').exists(): continue
        destination=folder/'paired_all_checkpoints.jsonl'
        temporary=destination.with_suffix('.tmp'); count=0
        # These are derived comparisons, atomically regenerated from immutable
        # per-frame outcomes. Avoid quadratic journal copies for 69,840 pairs.
        with temporary.open('w',encoding='utf-8') as stream:
            for path in sorted(folder.glob('validation_update_*.jsonl')):
                for r in Journal(path).rows():
                    for method in BASELINES:
                        baseline=lookup[(r['case_id'],method)]
                        assert r['raw_hash']==baseline['raw_hash'] and r['label_sha256']==baseline['label_sha256']
                        assert r['physical']['cache_key']==baseline['cache_key']
                        record={k:r[k] for k in ('case_id','mother_id','condition','group','input_kind','signed_aux','raw_hash','label_sha256','optimizer_update')}
                        record.update(run=folder.name,baseline=method,cache_key=r['physical']['cache_key'],paired_success=baseline['status']==r['status']=='SUCCESS')
                        if record['paired_success']:
                            record.update(template_MSE_delta=r['metrics']['template_MSE_e2']-baseline['metrics']['template_MSE_e2'],
                                template_RMSE_delta=r['metrics']['template_RMSE_e']-baseline['metrics']['template_RMSE_e'],
                                color_MSE_delta=r['metrics']['temperature_MSE_K2']-baseline['metrics']['temperature_MSE_K2'] if baseline['metrics']['temperature_MSE_K2'] is not None else None)
                        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); count+=1
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(destination)
        receipts.append({'run':folder.name,'paired_rows':count,'sha256':runtime.sha256(destination),'checkpoints_separate':True})
    write(OUT/'paired_receipt.json',{'runs':receipts,'baseline_records':len(baselines),'baseline_count':6,'all_raws_identical_within_pairs':True})

def original_gate_baselines():
    selection=read(OLD_OUT/'backbone_selection.json'); old=load_model(device='cuda').eval()
    models={}
    for name,filename in (('v4_update100','seed_101_validation_selected.pt'),('v4_update2000','seed_101_final_2000.pt')):
        model=PhysicsPostInverseCNN(selection['physics_config']).cuda().eval()
        model.load_cnn(torch.load(runtime.ROOT/'training_v4/checkpoints'/filename,map_location='cpu',weights_only=False)['cnn_state_dict'])
        models[name]=model
    fixtures={f['case_id']:f for f in read(runtime.ROOT/'reports/repair_v3/run_001/frozen_manifest.json')['cases']}
    gates=[r for r in Journal(OLD_OUT/'backbone_gate_per_case.jsonl').rows() if r.get('method')==selection['method'] and r.get('config')==selection['solver_config']]
    gates=list({r['case_id']:r for r in gates}.values()); assert len(gates)==16
    journal=Journal(OUT/'original_16_baseline_per_case.jsonl')
    completed={(r['case_id'],r['method']) for r in journal.rows() if r['status']=='SUCCESS'}
    with torch.no_grad():
        for row in gates:
            cache_path=OLD_DATA/row['cache_path']
            fixture=fixtures[row['case_id']]
            fixture_path=runtime.ROOT/'outputs/repair_v3/run_001'/fixture['path']
            assert runtime.sha256(cache_path)==row['cache_sha256']
            assert runtime.sha256(fixture_path)==fixture['file_sha256']
            arr=npz(cache_path); labels=npz(fixture_path)
            observed=labels['expected_e'] if fixture['metadata']['variant']=='continuous' else labels['raw_dn']
            assert array_hash(observed)==row['input_hash']
            raw=torch.as_tensor(arr['raw_e'],device='cuda')[None]
            z=torch.as_tensor(arr['state'],device='cuda')[None]; beta=torch.as_tensor(arr['beta'],device='cuda')[None]
            target={'template_e':labels['template_e'],'temperature_color_K':labels['temperature_observed_color_K']}
            for method in ('zero','old_B4','fixed_physics','v4_update100','v4_update2000'):
                if (row['case_id'],method) in completed: continue
                if method=='zero': S=np.zeros_like(target['template_e']); T=None
                else:
                    if method=='old_B4': out=old.infer_electrons(raw); S=out['species_signed_e'][0,2]
                    elif method=='fixed_physics': out=physical_output(old.physics,z,beta,raw); S=out['template_e'][0]
                    else: out=models[method].forward_electrons(raw,z,beta); S=out['template_e'][0]
                    S=S.cpu().numpy(); T=out['temperature_color_K'][0].cpu().numpy()
                journal.append({'case_id':row['case_id'],'input_hash':row['input_hash'],'method':method,
                    'metrics':score_arrays(S,T,target),'status':'SUCCESS','source':'unchanged original16 gate cache',
                    'fixture_sha256':fixture['file_sha256'],'physical_cache_sha256':row['cache_sha256']})

if __name__=='__main__': baseline_bank(); original_gate_baselines(); pair_all()
