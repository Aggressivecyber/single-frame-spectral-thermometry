import inspect
import numpy as np
import torch
from collections import Counter
from training_v4_1.common import *
from training_v4_1.model import protective_loss, new_model
from training_v4_1.sampling import build_draws
from training_v4.model import physical_output

def test_complete_factorial_and_signed_legality():
    matrix = read(OUT/'coverage_matrix.json')
    assert len(matrix) == 96 and all(r['new_count'] > 0 for r in matrix)
    for split in ('train','validation'):
        rows = [r for r in matrix if r['split'] == split and r['input_kind'] == 'continuous']
        assert len(rows) == 24 and sum(r['v4_continuous_count'] == 0 for r in rows) == 12
    legal = read(OUT/'signed_expectation_legality.json')
    assert len(legal) == 72 and all(r['minimum_expected_e'] >= 0 and r['minimum_band_e'] >= 0 for r in legal)
    data = read(OUT/'data_manifest.json')
    parents = {s:{r['mother_id'] for r in data['rows'] if r['split']==s} for s in ('train','validation')}
    assert not parents['train'] & parents['validation']

def test_balanced_draws_exactly_four_roles_and_same_sequence():
    rows = read(OUT/'data_manifest.json')['rows']; by_id = {r['case_id']:r for r in rows}
    draws = build_draws(rows); assert draws == build_draws(rows) and len(draws) == 1500
    signed = 0
    for slot, kind in enumerate(('noise','noise','continuous','quantization')):
        conditions = Counter()
        for batch in draws:
            row=by_id[batch[slot]]; assert row['split']=='train' and row['input_kind']==kind
            signed += row['signed_aux']
            if not row['signed_aux']: conditions[row['condition']] += 1
        assert max(conditions.values())-min(conditions.values()) <= 1
    assert signed == 300

def test_no_harm_is_output_domain_and_has_live_parameter_gradients():
    from config import default_config
    selection={'physics_config':default_config(64)}
    model=new_model(selection)
    assert sum(p.numel() for p in model.parameters()) == 5797
    assert list(inspect.signature(model.forward).parameters) == ['raw_dn','cached_state','cached_beta']
    z=torch.zeros((1,5,64,64),device='cuda'); z[:,1]=-.3
    beta=z.new_tensor([[.4,.02,-.03]])
    raw=model.physics(z,beta); dn=raw/model.physics.adc_step+model.cfg['black_level_DN']
    with torch.no_grad():
        base=physical_output(model.physics,z,beta,raw)
        truth={k:base[k].clone() for k in ('template_e','temperature_color_K','band_images_e')}
        model.emission_head.bias[2]=.02
    out=model(dn,z,beta)
    score={'template_MSE_e2':0.,'temperature_MSE_K2':0.}
    loss,terms=protective_loss(model,dn,out,truth,score,1.)
    assert terms['safe_template']>0
    loss.backward()
    assert model.emission_head.bias.grad[2].abs()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert 'latent' not in terms

def test_signed_support_and_zero_reference_do_not_use_positive_only_thresholds():
    from training_v4_1.evaluate import score_arrays
    y=np.zeros((16,16),np.float32); y[4:12,4:12]=-100
    target={'template_e':y,'temperature_color_K':np.full_like(y,3000)}
    exact=score_arrays(y,target['temperature_color_K'],target)
    assert exact['negative_recall60']==1 and exact['positive_recall60'] is None
    zero=score_arrays(np.zeros_like(y),None,target)
    assert zero['template_L2']==1 and zero['negative_recall60']==0
    blank={'template_e':np.zeros_like(y),'temperature_color_K':target['temperature_color_K']}
    false=score_arrays(y,target['temperature_color_K'],blank)
    assert false['negative_blank_FP60']==.25 and false['positive_blank_FP60']==0


def test_original_development_gate_sources_never_enter_gradient_sampling():
    rows=read(OUT/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in rows}
    gate_ids={r['case_id'] for r in read(runtime.ROOT/'reports/repair_v3/run_001/frozen_manifest.json')['cases']}
    sampled={case_id for batch in build_draws(rows) for case_id in batch}
    assert not sampled & gate_ids
    assert all('outputs/repair_v3/' not in by_id[i]['raw_path'] and
               'outputs/repair_v3/' not in by_id[i]['label_path'] for i in sampled)


def test_independent_noise_has_distinct_matched_physical_cache():
    from training_v4_1.cache import key_for
    rows=read(OUT/'data_manifest.json')['rows']; selection=read(OLD_OUT/'backbone_selection.json')
    first=next(r for r in rows if r['split']=='train' and r['case_id'].endswith('blank_noise0'))
    second=next(r for r in rows if r['mother_id']==first['mother_id'] and r['condition']=='blank' and r['case_id'].endswith('noise1'))
    assert first['raw_hash']!=second['raw_hash']
    assert key_for(first,selection)!=key_for(second,selection)
    for row in (first,second):
        assert row['raw_hash']==array_hash(npz(file_path(row,'raw_path'))['raw_dn'])
        if not (OUT/'cache_receipt.json').exists():
            # The same test is also run before cache generation. The final
            # pipeline, after its completion receipt, requires every sidecar.
            continue
        score=read(DATA/'physical_scores'/(row['case_id']+'.json'))
        assert score['raw_hash']==row['raw_hash']
        assert score['cache_key']==key_for(row,selection)
        meta=read((runtime.ROOT/score['cache_path']).with_suffix('.json'))
        assert meta['cache_key']==score['cache_key'] and not meta['labels_used']
