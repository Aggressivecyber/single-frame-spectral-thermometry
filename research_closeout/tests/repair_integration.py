"""Forward-only replay and raw-interface compatibility across repair modules."""
from .. import storage as s
from .. import measurement as m
from .. import scenarios
from .. import independent_renderer as renderer
from ..deployment import FrozenPrototype
from ..baselines import from_state
from ..repair_protocol import build
import inspect
import json
import numpy as np
import torch


def run():
    torch.set_num_threads(2)
    s.frozen_bridge()
    p=m.physics('cpu')
    old=s.read(s.REPORTS/'SCIENTIFIC_PLAN.json')
    row=next(r for r in old['cases'] if r['family']=='independent' and r['condition']=='nominal')
    receipt=next(r for r in s.rows(s.REPORTS/'new_raw_manifest.jsonl') if r['case_id']==row['case_id'])
    original=scenarios.render(row,p)
    original_dn,_=renderer.sample_raw(original['expected_e'],p.cfg,row['noise_seed'])
    matched=scenarios.render({**row,'relay_model':renderer.MATCHED_RELAY},p)
    # Exercise the public raw interface with a forward-only finite initializer,
    # not an unlogged inverse. The numerical flag must stay research-baseline.
    state=p.M.new_zeros((1,5,*p.cfg['shape']))
    state[:,1]=np.log(.5)
    beta=p.M.new_zeros((1,3))
    raw=p(state,beta)/p.adc_step+p.cfg['black_level_DN']
    class Suite:
        old_model=None
        def v3(self,dn):
            return from_state(p,dn,state,beta,{'numerical_certified':False})
    engine=object.__new__(FrozenPrototype)
    engine.suite=Suite();engine.physics=p;engine.qualified=False
    out=engine.infer(raw)
    new=build()
    checks=[
        {'name':'historical_independent_expected_mean_exact_replay',
         'passed':scenarios.array_hash(original['expected_e'])==receipt['expected_mean_sha256']},
        {'name':'historical_independent_raw_exact_replay',
         'passed':scenarios.array_hash(original_dn)==receipt['raw_array_sha256']},
        {'name':'historical_plan_selects_explicit_old_relay',
         'passed':original['metadata']['relay_model']==renderer.LEGACY_RELAY_STRESS},
        {'name':'matched_relay_requires_explicit_new_plan_field',
         'passed':matched['metadata']['relay_model']==renderer.MATCHED_RELAY and not np.array_equal(original['expected_e'],matched['expected_e'])},
        {'name':'raw_interface_has_no_label_or_scene_identity_parameters',
         'passed':set(inspect.signature(FrozenPrototype.infer).parameters)=={'self','raw_dn','method','photon_ratio','detector','application_domain'}},
        {'name':'raw_interface_separates_finite_numeric_and_detection',
         'passed':out['finite_output'] and not out['numerical_valid'] and out['frame_detection_v2']['detected'] is None
             and out['frame_detection_v2']['threshold'] is None},
        {'name':'repaired_protocol_keeps_original_final_metadata_sealed',
         'passed':not new['final_payloads_opened'] and len(old['final_main_metadata'])==512 and len(old['final_signed_metadata'])==180},
    ]
    result={'time':s.now(),'all_passed':all(r['passed'] for r in checks),'tests':checks,
        'complete_inverse_calls':0,'forward_only':True,'new_scientific_accuracy_claimed':False,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/scenarios.py','research_closeout/deployment.py','research_closeout/tests/repair_integration.py')}}
    s.write(s.REPORTS/'repair_001/REPAIR_INTEGRATION_CONTRACTS.json',result)
    print(json.dumps(result,indent=2))
    assert result['all_passed']


if __name__=='__main__':run()
