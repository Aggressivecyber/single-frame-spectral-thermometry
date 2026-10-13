"""Audit the raw-only entry point against a cache path; not a final-test evaluation."""
import runtime
import numpy as np
import torch
from training_v4.common import read, write, npz
from training_v4_1.common import OUT as OLD, OLD_OUT, setup
from training_v4_1.cache import load
from training_v4_4.infer import run as raw_infer, load_backbone, load_model, DEFAULT_BACKBONE

OUT=runtime.ROOT/'reports/training_v4_4/run_001/native_replay'
CHECKPOINT=runtime.ROOT/'training_v4_4/checkpoints/fixed_continuum/update_0500.pt'


def run():
    setup()
    assert read(runtime.ROOT/'reports/training_v4_4/run_001/fixed_continuum/status.json')['stage']=='COMPLETE'
    assert not (OUT/'receipt.json').exists()
    rows=read(OLD/'data_manifest.json')['rows']
    selected=[next(r for r in rows if r['split']=='validation' and r['condition']==c and r['input_kind']==k)
              for c,k in [('negative_normal','noise'),('weak','noise'),('blank','quantization')]]
    protocol={'case_ids':[r['case_id'] for r in selected],'checkpoint_sha256':runtime.sha256(CHECKPOINT),
        'raw_interface_inputs':'raw_dn-only NPZ and explicit checkpoint; no labels or caches',
        'qa_reference':'Separate outer audit reads frozen development caches after independent raw inversion',
        'maximum_pixel_template_difference_e':.02,'maximum_pixel_color_difference_K':.02,
        'optimizer_updates':0,'final_test_used':False,'registered_at':runtime.now()}
    write(OUT/'protocol.json',protocol)
    selection=read(OLD_OUT/'backbone_selection.json')
    model,_=load_model(CHECKPOINT,load_backbone(DEFAULT_BACKBONE),'cuda')
    records=[]
    for row in selected:
        folder=runtime.ROOT/'outputs/training_v4_4/run_001/native_replay'/row['case_id']
        raw_infer(runtime.ROOT/row['raw_path'],CHECKPOINT,folder,device='cuda')
        receipt=read(folder/'inference_receipt.json')
        assert receipt['labels_loaded'] is False and receipt['condition_loaded'] is False
        assert receipt['development_cache_loaded'] is False and receipt['inverse_recomputed_from_raw']
        actual=npz(folder/'prediction.npz'); inverse=npz(folder/'physical_inverse.npz')
        inputs,cache=load(row,selection,labels=False)
        with torch.no_grad():
            expected=model(*inputs)
        differences={k:float(np.max(np.abs(actual[k]-expected[k].cpu().numpy())))
                     for k in ('template_e','temperature_color_K','state','expected_e')}
        record={'case_id':row['case_id'],'raw_hash':row['raw_hash'],'checkpoint_sha256':runtime.sha256(CHECKPOINT),
            'differences':differences,'state_difference_vs_cache':float(np.max(np.abs(inverse['state']-inputs[1].cpu().numpy()))),
            'beta_difference_vs_cache':float(np.max(np.abs(inverse['beta']-inputs[2].cpu().numpy()))),
            'physical_inverse_seconds':receipt['physical_inverse_seconds'],'cnn_seconds':receipt['cnn_seconds'],
            'valid_fraction':receipt['valid_fraction'],'raw_inference_receipt_sha256':runtime.sha256(folder/'inference_receipt.json')}
        records.append(record)
        write(OUT/'progress.json',{'completed':len(records),'records':records,'final_test_used':False})
        assert differences['template_e']<=protocol['maximum_pixel_template_difference_e']
        assert differences['temperature_color_K']<=protocol['maximum_pixel_color_difference_K']
        assert receipt['valid_fraction']==1.
    write(OUT/'receipt.json',{'passed':True,'records':records,'case_count':3,'optimizer_updates':0,
        'source_sha256':runtime.sha256(__file__),'inference_source_sha256':runtime.sha256(runtime.ROOT/'training_v4_4/infer.py'),
        'final_test_used':False})
    print({'raw_native_replays':3,'passed':True,'records':records},flush=True)


if __name__=='__main__':
    run()
