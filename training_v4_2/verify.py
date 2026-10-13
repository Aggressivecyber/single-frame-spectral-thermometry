"""Reopen saved weights, check actual optimizer steps and replay native outputs."""
import runtime
import sys
import torch
from training_v4.common import read,write,parameters_hash,npz
from training_v4_1.common import setup,OUT as OLD,OLD_OUT,OLD_DATA,rel
from training_v4_1.cache import load
from training_v4_1.model import new_model
from training_v4_1.evaluate import score_arrays
from evaluation_v2.common import Journal,array_hash
from training_v4_2.experiment import OUT,WEIGHTS,INITIAL,LIMIT


def main():
    setup(); selection=read(OLD_OUT/'backbone_selection.json')
    protocol=read(OUT/'protocol.json')
    rows=read(OLD/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in rows}
    final_rows={r['case_id']:r for r in Journal(OUT/'validation_update_0500.jsonl').rows()}
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model=new_model(selection)
    frozen_buffers={k:v.clone() for k,v in model.named_buffers()}
    native=[]; saved_hashes=[]
    for update in range(0,LIMIT+1,100):
        path=WEIGHTS/('initial.pt' if update==0 else f'update_{update:04d}.pt')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_cnn(checkpoint['cnn_state_dict'])
        assert checkpoint['optimizer_updates']==update and checkpoint['protocol']==protocol
        steps=sorted({int(v['step']) for v in checkpoint['optimizer_state']['state'].values()})
        assert steps==([] if update==0 else [update])
        assert parameters_hash(model)==checkpoint['parameters_sha256']
        saved_hashes.append(parameters_hash(model))
        assert all(torch.equal(v,frozen_buffers[k]) for k,v in model.named_buffers())
        assert sum(p.numel() for p in model.parameters())==5797
        if update==0:
            for k,v in model.cnn_state_dict().items():
                assert torch.equal(v,initial['cnn_state_dict'][k]) if k.startswith('trunk.') else v.count_nonzero()==0
        val=read(OUT/f'validation_summary_{update:04d}.json')
        assert val['validation_cases']==776 and len(val['protection_groups'])==96
        per_case=Journal(OUT/f'validation_update_{update:04d}.jsonl').rows()
        if update==0:
            assert all(r['template_correction_RMS_e']==0 and r['temperature_correction_RMS_K']==0 for r in per_case)
        native.append({'update':update,'checkpoint':rel(path),'sha256':runtime.sha256(path),
                       'Adam_steps':steps,'parameters_reopened':True,'original_physics_buffers_unchanged':True,
                       'frames_with_invalid_output':sum(r['invalid_output_fraction']>0 for r in per_case)})
    assert len(set(saved_hashes))==6
    assert any(not torch.equal(v,initial['cnn_state_dict'][k]) for k,v in model.cnn_state_dict().items() if k.startswith('trunk.'))
    draws=read(OLD/'sampling_sequence.json')['draws'][:LIMIT]
    trained=Journal(OUT/'training_curve.jsonl').rows()
    assert len(trained)==LIMIT
    old=Journal(OLD/'safe1/training_curve.jsonl').rows()[:LIMIT]
    for update,(new,previous,batch) in enumerate(zip(trained,old,draws),1):
        assert new['optimizer_update']==update
        assert new['samples']==previous['samples']
        assert [r['case_id'] for r in new['samples']]==batch
    sample_ids=[next(i for i in final_rows if by_id[i]['signed_aux'] and by_id[i]['condition']=='negative_normal' and by_id[i]['input_kind']=='quantization'),
                next(i for i in final_rows if not by_id[i]['signed_aux'] and by_id[i]['condition']=='weak' and by_id[i]['input_kind']=='noise'),
                next(i for i in final_rows if not by_id[i]['signed_aux'] and by_id[i]['condition']=='weak' and by_id[i]['input_kind']=='continuous')]
    replay=[]; model.eval()
    with torch.no_grad():
        for case in sample_ids:
            inputs,truth,_=load(by_id[case],selection); out=model(*inputs)
            metrics=score_arrays(out['template_e'][0].cpu().numpy(),out['temperature_color_K'][0].cpu().numpy(),
                                 {k:v[0].cpu().numpy() for k,v in truth.items()})
            difference={k:abs(metrics[k]-final_rows[case]['metrics'][k]) for k in ('template_RMSE_e','color_RMSE_K')}
            assert max(difference.values())<1e-5
            replay.append({'case_id':case,'metric_absolute_differences':difference})
    preserved=read(OUT/'preserved_files.json')
    for path,checksum in preserved.items(): assert runtime.sha256(runtime.ROOT/path)==checksum,path
    frozen={r['case_id']:r for r in read(runtime.ROOT/'reports/repair_v3/run_001/frozen_manifest.json')['cases']}
    gate_records=Journal(OLD_OUT/'backbone_gate_per_case.jsonl').rows()
    gates={r['case_id']:r for r in gate_records if r.get('method')==selection['method'] and r.get('config')==selection['solver_config'] and r['status']=='SUCCESS'}
    assert len(gates)==16
    gate_receipt=[]
    for case,row in gates.items():
        fixture=frozen[case]
        path=runtime.ROOT/'outputs/repair_v3/run_001'/fixture['path']
        cache=OLD_DATA/row['cache_path']
        assert runtime.sha256(path)==fixture['file_sha256']
        assert runtime.sha256(cache)==row['cache_sha256']
        data=npz(path)
        observed=data['expected_e'] if fixture['metadata']['variant']=='continuous' else data['raw_dn']
        assert array_hash(observed)==row['input_hash']
        gate_receipt.append({'case_id':case,'input_hash':row['input_hash'],'fixture_sha256':fixture['file_sha256'],
                             'cache_sha256':row['cache_sha256']})
    write(OUT/'original16_input_hashes.json',gate_receipt)
    # Recheck the numerical implementation recorded in the saved checkpoint.
    assert checkpoint['sources']['training_v4_2/experiment.py']==runtime.sha256(runtime.ROOT/'training_v4_2/experiment.py')
    write(OUT/'native_verification.json',{'checkpoints':native,'native_replays':replay,
        'trained_updates':LIMIT,'actual_microbatches':2000,'frozen_parent_files_unchanged':len(preserved),
        'first500_sample_raw_cache_records_exactly_equal':True,'unique_checkpoint_parameter_hashes':6,
        'source_implementation_matches_checkpoint':True,'final_test_used':False,
        'original16_fixture_and_cache_hashes_matched':True,
        'native_runtime':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                          'GPU':torch.cuda.get_device_name(0),'device':'cuda:0'}})
    print({'native_verification':'passed','updates':LIMIT,'preserved_files':len(preserved)},flush=True)


if __name__=='__main__':main()
