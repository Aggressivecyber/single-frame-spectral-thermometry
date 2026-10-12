"""One bounded safe-weight control, then at most one conditional feature ablation."""
import numpy as np
import torch
from training_v4_1.common import *
from training_v4_1.train import train
from training_v4_1.cache import load
from training_v4_1.model import new_model

def decision():
    setup(); receipts=[]
    first=read(OUT/'safe1/training_receipt.json')
    if not first['qualified_candidate_available']:
        if not (OUT/'safe5/training_receipt.json').exists(): train('safe5',5.)
        second=read(OUT/'safe5/training_receipt.json')
        assert first['initial_parameters_sha256']==second['initial_parameters_sha256']
        assert first['sampling_sequence_sha256']==second['sampling_sequence_sha256']
        events=Journal(OUT/'safe5/validation_curve.jsonl').rows()
        safest=min(events,key=lambda e:(-e['original_gates_pass_count'],sum(not g['passed'] for g in e['protection_groups']),e['main_validation_loss']))
        trigger=not second['qualified_candidate_available'] and safest['small_residual_large_correction_fraction']>.05
        write(OUT/'feature_ablation_decision.json',{'both_safe_controls_fail':not second['qualified_candidate_available'],
            'diagnostic_checkpoint_update':safest['optimizer_update'],'small_residual_large_correction_fraction':safest['small_residual_large_correction_fraction'],
            'triggered':trigger,'selection_rule':'largest original gate count, fewest failed protection subgroups, then validation loss',
            'new_module_count':0,'final_test_used':False})
        if trigger and not (OUT/'amplitude_safe5/training_receipt.json').exists():
            selection=read(OLD_OUT/'backbone_selection.json'); model=new_model(selection).eval()
            rows=[r for r in read(OUT/'data_manifest.json')['rows'] if r['split']=='train' and r['input_kind']=='noise' and not r['signed_aux']][:64]
            values=[]
            with torch.no_grad():
                for row in rows:
                    inputs,_=load(row,selection,labels=False); raw,z,beta=inputs
                    raw_e=model.physics.raw_to_electrons(raw); expected=model.physics(z,beta)
                    residual=(expected-raw_e)/(expected.clamp_min(0)+model.physics.extra_variance)
                    values.append(float(model.physics.op.images_adjoint(residual).square().mean().sqrt()))
            reference=max(float(np.median(values)),1e-6)
            write(OUT/'amplitude_feature_reference.json',{'training_raw_hashes':[r['raw_hash'] for r in rows],
                'median_weighted_adjoint_RMS':reference,'truth_used':False,'validation_used':False,
                'only_change':'replace per-frame/per-band RMS division with a fixed training-derived scalar'})
            del model
            train('amplitude_safe5',5.,'amplitude',reference)
    else:
        write(OUT/'feature_ablation_decision.json',{'triggered':False,'reason':'safe1 has an admissible candidate; no further control authorized by protocol'})
    candidates=[]
    for name in ('safe1','safe5','amplitude_safe5'):
        if not (OUT/name/'training_receipt.json').exists(): continue
        receipt=read(OUT/name/'training_receipt.json'); receipts.append(receipt)
        for event in Journal(OUT/name/'validation_curve.jsonl').rows():
            if event['eligible']: candidates.append({'run':name,**event})
    best=min(candidates,key=lambda r:r['main_validation_loss']) if candidates else None
    write(OUT/'candidate_selection.json',{'selected':best,'completed_runs':[r['name'] for r in receipts],
        'all_checkpoints_kept_separate':True,'combined_different_checkpoint_benefits':False,
        'final_test_opened':False,'qualified_model_available':best is not None})
    print({'controls_complete':True,'completed_runs':[r['name'] for r in receipts],'selected_run':best['run'] if best else None},flush=True)

if __name__=='__main__': decision()
