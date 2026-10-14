"""An additive, one-repair registration; previous stop evidence is immutable."""
from . import storage as s
from .solver import RTOL,MAXITER
from .. import common as c,decomposition
import subprocess


def register():
    prior=c.require_registered()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT).decode().strip()==s.PRIOR_HEAD
    assert not (s.CODE/'protocol.json').exists()
    receipts=[]
    for device in ('cpu','cuda'):
        name='cpu_001.json' if device=='cpu' else 'gpu_001.json'
        path=s.REPORTS/'checks'/name;r=c.read_json(path)
        assert r['all_passed'] and r['device']==device
        for key in ('solver.py','diagnostics.py','storage.py','tests.py'):
            rel=c.relative(s.CODE/key);assert r['source_sha256'][rel]==c.sha(s.CODE/key)
        receipts.append({'path':c.relative(path),'sha256':c.sha(path)})
    old=c.read_json(s.OLD_REPORTS/'A_DECISION.json')
    assert old['status']=='STOP_NUMERICAL_OR_ORACLE_AUDIT'
    p={'schema':'v6-precision-repair001-registration','registered_at':c.now(),
       'authorization':'User continuation after published required stop; one targeted numerical repair/retest',
       'prior_head':s.PRIOR_HEAD,'original_v6_protocol_sha256':c.sha(c.CODE/'protocol.json'),
       'preserved_stop_decision_sha256':c.sha(s.OLD_REPORTS/'A_DECISION.json'),
       'source_sha256':s.sources(),'original_source_sha256':prior['source_sha256'],
       'first_failed_case_id':old['stop_reason']['case_id'],'all_input_inventory_sha256':c.sha(c.CODE/'input_inventory.json'),
       'linear_options':prior['linear_options'],'continuum_options':prior['continuum_options'],
       'v3_selection':prior['v3_selection'],'identity_criteria':decomposition.DEFAULT_IDENTITY_CRITERIA,
       'A_gates':prior['A_gates'],'sources':prior['sources'],'beta_modes':prior['beta_modes'],
       'arithmetic':'float64 with exact promotion of original float32 coefficients, W and lambda',
       'rtol':RTOL,'max_iterations':MAXITER,'alpha_g':1.0,
       'scientific_and_identity_thresholds_changed':False,'raw_likelihood_count':1,
       'source_algorithms_weights_calibration_changed':False,'no_OTF_division':True,
       'controlled_retest':'all6 source/beta rows on original failed raw, original v3 scored separately',
       'resume_rule':'full720 A only after all6 retest rows converge and satisfy unchanged observable identity',
       'numerical_repairs_remaining_after_this':0,'formal_A_budget_s':43200,'output_budget_bytes':8*2**30,
       'minimum_free_bytes':2*2**30,'CNN_training_allowed':False,'signed36_final64_access':False,
       'implementation_controls':receipts}
    c.write_json(s.CODE/'protocol.json',p)
    return p


def require():
    p=c.read_json(s.CODE/'protocol.json');c.require_registered()
    for rel,digest in p['source_sha256'].items():assert c.sha(c.ROOT/rel)==digest,'Repair source changed: '+rel
    assert c.sha(c.CODE/'input_inventory.json')==p['all_input_inventory_sha256']
    return p
