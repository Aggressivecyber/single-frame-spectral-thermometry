import shutil
import sys
import torch
from training_v4_1.common import *
from training_v4_1.model import new_model
from training_v4_1.cache import load
from infer import load_model

def verify():
    setup(); selection=read(OLD_OUT/'backbone_selection.json'); manifest=read(OUT/'data_manifest.json')
    old=load_model(device='cuda').physics; receipts=[]
    for name in read(OUT/'candidate_selection.json')['completed_runs']:
        receipt=read(OUT/name/'training_receipt.json')
        assert receipt['updates']==1500 and receipt['real_parameters_changed']
        curve=Journal(OUT/name/'training_curve.jsonl').rows()
        assert [r['optimizer_update'] for r in curve]==list(range(1,1501))
        initial=torch.load(DATA/'checkpoints'/name/'initial.pt',map_location='cpu',weights_only=False)
        assert initial['optimizer_state']['state']=={} and initial['optimizer_updates']==0
        final=torch.load(runtime.ROOT/receipt['final_checkpoint'],map_location='cpu',weights_only=False)
        assert set(int(s['step']) for s in final['optimizer_state']['state'].values())=={1500}
        assert sum(t.numel() for t in final['cnn_state_dict'].values())==5797
        assert initial['initial_parameters_sha256']==final['initial_parameters_sha256']
        model=new_model(selection,final['variant'],final.get('reference_rms')).eval()
        model.load_cnn(final['cnn_state_dict']); assert parameters_hash(model)==final['parameters_sha256']
        assert all(torch.equal(v,model.physics.state_dict()[k]) for k,v in old.state_dict().items())
        row=next(r for r in manifest['rows'] if r['split']=='validation' and r['input_kind']=='continuous' and r['condition']=='weak' and r['geometry']=='front' and not r['signed_aux'])
        inputs,score=load(row,selection,labels=False)
        with torch.no_grad(): out=model(*inputs)
        target=npz(file_path(row,'label_path'))
        from training_v4_1.evaluate import score_arrays
        metrics=score_arrays(out['template_e'][0].cpu().numpy(),out['temperature_color_K'][0].cpu().numpy(),target)
        recorded=next(r for r in Journal(OUT/name/'validation_update_1500.jsonl').rows() if r['case_id']==row['case_id'])
        assert abs(metrics['template_MSE_e2']-recorded['metrics']['template_MSE_e2'])<1e-5
        export=runtime.ROOT/'training_v4_1/checkpoints'; export.mkdir(exist_ok=True)
        checkpoint_inventory=[]
        for event in Journal(OUT/name/'validation_curve.jsonl').rows():
            source=runtime.ROOT/event['checkpoint_path']
            assert runtime.sha256(source)==event['checkpoint_sha256']
            item=torch.load(source,map_location='cpu',weights_only=False)
            assert item['optimizer_updates']==event['optimizer_update']
            assert set(int(s['step']) for s in item['optimizer_state']['state'].values())=={event['optimizer_update']}
            destination=export/name/source.name; destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source,destination)
            assert runtime.sha256(destination)==event['checkpoint_sha256']
            checkpoint_inventory.append({'optimizer_update':event['optimizer_update'],'path':rel(destination),
                'sha256':event['checkpoint_sha256'],'original_gates_pass_count':event['original_gates_pass_count'],
                'protection_pass':event['protection_pass'],'weak_L2':event['weak_L2'],'eligible':event['eligible']})
        path=export/(name+'_final_1500.pt'); shutil.copy2(runtime.ROOT/receipt['final_checkpoint'],path)
        assert runtime.sha256(path)==receipt['final_sha256']
        if receipt['selected_checkpoint']:
            source=runtime.ROOT/receipt['selected_checkpoint']; destination=export/(name+'_validation_selected.pt')
            shutil.copy2(source,destination); assert runtime.sha256(source)==runtime.sha256(destination)
        numeric=['common.py','data.py','cache.py','sampling.py','model.py','evaluate.py','train.py']
        assert all(runtime.sha256(runtime.ROOT/'training_v4_1'/p)==initial['sources_v4_1']['training_v4_1/'+p] for p in numeric)
        recovery = None
        if receipt.get('interruption_recovery'):
            recovery=read(runtime.ROOT/receipt['interruption_recovery'])
            assert recovery['status']=='COMPLETE' and recovery['Adam_state_restored'] and not recovery['Adam_reset_again']
            assert recovery['max_replay_loss_relative_difference']<1e-4
            archive=(runtime.ROOT/receipt['interruption_recovery']).parent
            assert all(runtime.sha256(archive/filename)==checksum for filename,checksum in recovery['archived_attempts'].items())
            assert runtime.sha256(runtime.ROOT/'training_v4_1/resume.py')==final['recovery_implementation_sha256']
        receipts.append({'run':name,'actual_Adam_steps':1500,'parameters_reopened_and_verified':True,
            'original_physics_buffers_equal':True,'same_initial_weights_hash':initial['initial_parameters_sha256'],
            'sampling_hash':receipt['sampling_sequence_sha256'],'native_front_weak_protection_repeat_verified':True,
            'final_export':rel(path),'sha256':runtime.sha256(path),'numeric_training_sources_unchanged':True,
            'recovered_interruption':recovery,'all_100_update_checkpoints_exported':checkpoint_inventory})
    assert receipts
    write(OUT/'native_checkpoint_receipt.json',{'runs':receipts,'torch':str(torch.__version__),
        'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'Python':sys.version,
        'inference_arguments':['raw_dn','cached_state','cached_beta'],'flags_or_truth_enter_inference':False})
    print({'native_checkpoints_verified':len(receipts)},flush=True)

if __name__=='__main__': verify()
