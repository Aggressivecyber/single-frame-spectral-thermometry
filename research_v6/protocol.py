"""Append-only Stage A registration before any formal v6 reconstruction."""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from . import common


def check_receipts():
    sources=common.source_hashes()
    accepted=[];coverage=set()
    directory=common.REPORTS.parent/'implementation_checks'
    for path in sorted(directory.rglob('*.json')):
        r=common.read_json(path)
        hashes=r.get('source_sha256',r.get('source_hashes',r.get('sources',{})))
        success=r.get('all_passed') is True or (r.get('passed') is True and r.get('failures',0)==0 and r.get('errors',0)==0)
        if not success or not hashes: continue
        normalized={('research_v6/'+k if '/' not in k else k):v for k,v in hashes.items()}
        if any(not (common.ROOT/k).is_file() or common.sha(common.ROOT/k)!=v for k,v in normalized.items()):continue
        covered=set(normalized)&set(sources)
        coverage.update(covered)
        accepted.append({'path':common.relative(path),'sha256':common.sha(path),'source_coverage':sorted(covered),
                         'device':r.get('device'), 'scientific_cohort_acceptance':False})
    for path,digest in sources.items():
        ast.parse((common.ROOT/path).read_text(encoding='utf-8-sig'),filename=path)
    required={'research_v6/decomposition.py','research_v6/continuum_sources.py','research_v6/inputs.py',
              'research_v6/scoring.py','research_v6/stage_a.py','research_v6/reports.py',
              'research_v6/protocol.py','research_v6/run.py','research_v6/common.py'}
    assert required<=coverage,'Missing current-source QA: '+str(sorted(required-coverage))
    native=[]
    for item in accepted:
        r=common.read_json(common.ROOT/item['path'])
        if r.get('shape')==[256,256] and r.get('device')=='cuda' and r.get('new_or_sealed_mother_arrays_generated') is False:
            native.append(item)
    assert native,'Actual frozen-buffer float32 native-resolution numerical control is required'
    return {'all_passed':True,'source_sha256':sources,'accepted_current_source_checks':accepted,
            'native_resolution_controls':native,'scientific_cohort_acceptance':False}


def baseline_inventory():
    changed=subprocess.check_output(['git','diff',common.BASE,'--name-only'],cwd=common.ROOT).decode().splitlines()
    assert all(x.startswith(('research_v6/','reports/research_v6/')) for x in changed),changed
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',common.BASE],cwd=common.ROOT).decode().splitlines()
    assert not any(x.startswith('outputs/') for x in names),'Sealed output payloads are forbidden in baseline content audit'
    return [{'path':name,'sha256':common.sha(common.ROOT/name),'bytes':(common.ROOT/name).stat().st_size} for name in names]


def register():
    common.configure_runtime()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=common.ROOT).decode().strip()=='research/v6-continuum-profile'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=common.ROOT).decode().strip()==common.BASE
    assert not (common.CODE/'protocol.json').exists(),'Frozen registration cannot be replaced'
    from . import inputs, continuum_sources, decomposition
    qa=check_receipts()
    inventory=inputs.inventory_stage_a()
    assert len(inventory['rows'])==720
    assert len({r['case_id'] for r in inventory['rows']})==720,'Case identifiers must be globally unique'
    assert sum(r['cohort']=='development' for r in inventory['rows'])==540
    assert sum(r['cohort']=='oracle' for r in inventory['rows'])==180
    assert len({r['mother_id'] for r in inventory['rows'] if r['cohort']=='development'})==36
    assert len({r['mother_id'] for r in inventory['rows'] if r['cohort']=='oracle'})==12
    integrity=inputs.integrity_receipt(inventory['rows'],payload_audit=True)
    assert integrity.get('all_passed') is True or integrity.get('passed') is True,integrity
    old=baseline_inventory()
    previous=common.read_json(common.ROOT/'research_v5/protocol.json')
    weight_paths={'old_B4':'checkpoints/selected_checkpoint.pt',
                  'v4.4_final_continuum_fixed':'training_v4_4/checkpoints/fixed_continuum/update_0500.pt'}
    weight_hashes={key:common.sha(common.ROOT/name) for key,name in weight_paths.items()}
    assert weight_hashes==previous['old_weights_sha256'],'An old weight changed'
    linear={**previous['linear_defaults'],'alpha_g':1.0}
    frozen_names=('physics.py','runtime.py','repair_v3/solver.py','training_v4/scalar_engine.py',
                  'research_v5/signed_linear_solver.py','research_v5/continuum_solver.py',
                  'research_v5/metrics.py','research_v5/statistics.py',
                  'reports/training_v4/run_001/backbone_selection.json')
    frozen={x:common.sha(common.ROOT/x) for x in frozen_names}
    p={'schema':'v6-fixed-raw-continuum-profile-registration-001','registered_at':common.now(),
       'base_commit':common.BASE,'branch':'research/v6-continuum-profile',
       'task_spec_sha256':common.sha(common.CODE/'TASK_SPEC.md'),
       'source_sha256':qa['source_sha256'],'frozen_library_sha256':frozen,
       'old_weight_sha256':weight_hashes,
       'physics_config':common.physics_config(),'physics_construction':'original float32 buffers unchanged',
       'linear_options':linear,'continuum_options':previous['continuum_options'],
       'v3_selection':continuum_sources.frozen_selection(),
       'sources':['C_true','C_v5','C_v3'],'beta_modes':['true_beta','estimated_beta'],
       'input_counts':{'development_mothers':36,'development_raw':540,'oracle_mothers':12,'oracle_raw':180,
                       'all_raw':720,'linear_rows':4320,'original_v3_rows':720},
       'weights':'one positive detached reciprocal(observed raw_e clamped at zero + fixed read and ADC variance) per raw',
       'identity_criteria':decomposition.DEFAULT_IDENTITY_CRITERIA,
       'known_pre_registration_numerical_failure':{
          'native_estimated_beta_Ctrue':True,'primary_float32_recomposition_gap_e':0.2122237797450025,
          'same_budget_actual_residual_explains_gap':True,
          'policy':'Implementation self-check verifies correct detection; formal gate remains unchanged and first failing real case stops A'},
       'numerical_gate':'every actual/component CG converges at both true residual criteria and observable identity passes',
       'float64_diagnostic':{'allowed':True,'max_failed_raw_cases':1,'trigger':'primary float32 identity/numerical failure',
           'same_frozen_primary_buffers_W_lambdas_promoted':True,'can_rescue_primary_gate':False},
       'A_gates':{'v3_ratio_max':1.05,'v3_ratio_upper_CI95_max':1.05,'Cv5_improvement_RMSE_fraction_min':0.25,
          'paired_Cv3_minus_Cv5_MSE_CI95_upper_max':0.0,'temperature_improvement_fraction_for_B':0.10,
          'oracle_noncollapse_projection_min':0.8,
          'O0_normal_noise_vs_v3_RMSE_ratio_max':1.05,
          'O0_noiseless_criterion':'actual observable error explained by regularization bias and exact n within identity tolerance; no alpha0 threshold transplant',
          'B_requires_complete_A_and_all_numerical_checks':True},
       'statistics':{'unit':'mother_scene','draws':2000,'seed':53130,'confidence_level':0.95,
                     'same_raw_pairing':True,'failed_or_invalid_cases_remain_in_error_denominator':True},
       'stage_a_budget_s':43200,'stage_a_output_budget_bytes':8*2**30,'minimum_E_free_bytes':2*2**30,
       'sealed_confirmation_mothers':36,'sealed_final_mothers':64,'sealed_payload_access_allowed':False,
       'CNN_training_allowed':False,'Stage_B_registration':'only after completed eligible A; separate immutable addendum',
       'historical_O0_O1_O2_O3':'frozen v5 alpha0 oracle context, not identical-alpha comparisons',
       'prospective_confirmations':'preregister future suite only; no sealed data access this round'}
    common.write_json(common.CODE/'input_inventory.json',inventory)
    common.write_json(common.REPORTS/'INPUT_INTEGRITY_BEFORE.json',integrity)
    common.write_json(common.REPORTS/'BASELINE_INVENTORY_BEFORE.json',{'base_commit':common.BASE,'files':old})
    common.write_json(common.REPORTS/'SOURCE_REGISTRATION_CHECKS.json',qa)
    common.write_json(common.CODE/'protocol.json',p)
    return {'status':'REGISTERED','protocol_sha256':common.sha(common.CODE/'protocol.json'),'input_raw':720,
            'all_passed':True,'sealed_payloads_opened':False}
