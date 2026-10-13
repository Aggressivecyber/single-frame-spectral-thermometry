import inspect
import torch
from config import default_config
from training_v4.model import physical_output
from training_v4_1.model import new_model, protective_loss
from training_v4_1.common import OUT as OLD, OLD_OUT, read, setup
from training_v4_1.cache import key_for
from evaluation_v2.common import Journal
from training_v4_2.experiment import reset_heads, traced_outputs, choose_probes, LIMIT


def fixture():
    setup(); model=new_model({'physics_config':default_config(64)})
    z=torch.zeros((1,5,64,64),device='cuda'); z[:,1]=-.3; z[:,4]=-.1
    beta=z.new_tensor([[.4,.02,-.03]])
    raw=model.physics(z,beta); dn=raw/model.physics.adc_step+model.cfg['black_level_DN']
    return model,(dn.detach(),z,beta)


def test_reset_retains_trunk_and_exact_identity_in_original_physical_outputs():
    model,inputs=fixture()
    with torch.no_grad():
        model.continuum_head.bias.fill_(.1); model.emission_head.bias.fill_(-.1)
    trunk={k:v.clone() for k,v in model.trunk.state_dict().items()}
    reset_heads(model); model.eval()
    assert sum(p.numel() for p in model.parameters())==5797
    assert list(inspect.signature(model.forward).parameters)==['raw_dn','cached_state','cached_beta']
    assert all(torch.equal(v,trunk[k]) for k,v in model.trunk.state_dict().items())
    with torch.no_grad():
        out=model(*inputs)
        base=physical_output(model.physics,inputs[1],inputs[2],model.physics.raw_to_electrons(inputs[0]))
    assert all(torch.equal(out[k],base[k]) for k in ('state','template_e','temperature_color_K','band_images_e','expected_e'))


def test_stage_trace_is_original_six_step_forward_including_stage_index():
    model,inputs=fixture(); model.eval()
    with torch.no_grad():
        model.emission_head.weight.normal_(0,.001); model.continuum_head.weight.normal_(0,.001)
        records=list(traced_outputs(model,inputs)); native=model(*inputs)
    assert [i for i,_ in records]==list(range(7))
    assert torch.equal(records[-1][1]['state'],native['state'])
    assert not torch.equal(records[0][1]['state'],native['state'])


def test_zero_heads_have_live_output_gradients_and_parameters_update():
    model,inputs=fixture(); model.train()
    with torch.no_grad():
        truth=physical_output(model.physics,inputs[1],inputs[2],model.physics.raw_to_electrons(inputs[0]))
        truth={k:truth[k].clone() for k in ('template_e','temperature_color_K','band_images_e')}
        truth['template_e']-=5; truth['temperature_color_K']+=2
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5)
    out=model(*inputs); loss,_=protective_loss(model,inputs[0],out,truth,{'template_MSE_e2':25,'temperature_MSE_K2':4},1.)
    loss.backward()
    assert model.emission_head.weight.grad.norm()>0 and model.continuum_head.weight.grad.norm()>0
    assert all(p.grad is not None and p.grad.count_nonzero()==0 for p in model.trunk.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step(); assert model.emission_head.weight.count_nonzero()>0


def test_probe_selection_and_training_prefix_are_disjoint_and_hash_matched():
    rows=read(OLD/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in rows}
    probes=choose_probes(rows)
    assert len(probes['stage_validation_ids'])==144 and len(probes['gradient_train_ids'])==12
    assert all(by_id[i]['split']=='validation' for i in probes['stage_validation_ids'])
    assert all(by_id[i]['split']=='train' for i in probes['gradient_train_ids'])
    draws=read(OLD/'sampling_sequence.json')['draws'][:LIMIT]
    old=Journal(OLD/'safe1/training_curve.jsonl').rows()[:LIMIT]
    selection=read(OLD_OUT/'backbone_selection.json')
    for b,r in zip(draws,old):
        assert b==[x['case_id'] for x in r['samples']]
        for i,x in zip(b,r['samples']):
            assert by_id[i]['split']=='train' and by_id[i]['raw_hash']==x['raw_hash']
            assert key_for(by_id[i],selection)==x['cache_key']
