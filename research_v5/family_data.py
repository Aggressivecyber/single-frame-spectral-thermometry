"""Explicitly authorized fresh development/train and sealed final families.

Generation has no selection logic.  Guard every descriptor before drawing a
field or noise sample.  Preserve original final amplitude/noise rules and keep
all truth in separate labels, never in raw containers or deployment arguments.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys

from . import common
from .data import (_array_hash, _expectation_diagnostics, _save_npz,
                   TEMPERATURE_GEOMETRIES, TEMPLATE_GEOMETRIES)

ROLE_KEYS = {'development_selection':'development', 'null_calibration':'null_calibration',
             'null_evaluation':'null_evaluation', 'training':'training',
             'calibration_stress':'calibration_stress', 'signed_confirmation':'signed_confirmation',
             'final_once':'final_once'}
SEALED = {'signed_confirmation','final_once'}


def _guard(physics, plan, output_root, options):
    options = dict(options or {})
    if set(options)-{'fixture_only','authorization'}:
        raise ValueError('Unknown family generation options')
    fixture = bool(options.get('fixture_only', False))
    root = common.ed_path(output_root)
    if not root.is_relative_to(common.DATA.parent.resolve()):
        raise ValueError('New families must stay under outputs/research_v5')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Never overwrite an existing family: '+str(root))
    if not plan or not isinstance(plan,list):
        raise ValueError('An explicit nonempty registered plan is required')
    roles = {row.get('partition_role') for row in plan}
    if len(roles)!=1 or not roles.issubset(ROLE_KEYS):
        raise ValueError('One registered partition role per family')
    role, = roles
    cases = [row.get('case_id') for row in plan]
    if len(set(cases))!=len(cases) or any(not isinstance(c,str) or re.fullmatch(r'[A-Za-z0-9_-]+',c) is None for c in cases):
        raise ValueError('Unique safe case names are required')
    mothers = {}
    for row in plan:
        mother = row.get('mother',{})
        mid = row.get('mother_id')
        if not isinstance(mid,str) or mid!=mother.get('mother_id') or re.fullmatch(r'[A-Za-z0-9_-]+',mid) is None:
            raise ValueError('Unsafe or conflicting mother identity')
        digest = common.canonical_hash(mother)
        if mid in mothers and mothers[mid]!=digest:
            raise ValueError('A mother has conflicting fields')
        mothers[mid] = digest
        if mother['temperature_geometry'] not in TEMPERATURE_GEOMETRIES or mother['template_geometry'] not in TEMPLATE_GEOMETRIES:
            raise ValueError('Unregistered geometry')
        if row.get('input_kind') not in ('continuous','quantization','noise'):
            raise ValueError('Unregistered raw input kind')
        if row['input_kind']=='noise' and (not isinstance(row.get('noise_seed'),int) or row['noise_seed']<0):
            raise ValueError('Noise seed must be explicitly registered')
        if row.get('coefficient_rule')=='original_v4_final_levels':
            if role!='final_once' or row.get('condition') not in ('blank','weak','normal','strong'):
                raise ValueError('Original final amplitude rules are final-only')
        elif row.get('coefficient_e') not in (0.,45.,-45.,180.,-180.):
            raise ValueError('Unknown signed coefficient')
        stress = row.get('stress')
        if stress:
            if role!='calibration_stress' or stress.get('kind') not in ('channel_gain','continuum_emissivity'):
                raise ValueError('Stress perturbation in a nominal family')
            values = stress.get('band_multipliers')
            if not isinstance(values,list) or len(values)!=7 or min(values)<=0:
                raise ValueError('Stress spectral multipliers must be positive seven-vectors')
    if fixture:
        if role in SEALED or physics.M.device.type!='cpu':
            raise ValueError('Fixtures are CPU-only and cannot touch sealed partitions')
        return root, role, True, None
    auth = dict(options.get('authorization') or {})
    protocol = common.require_registered()
    protocol_digest = common.sha(common.CODE/'protocol.json')
    plan_digest = common.canonical_hash(plan)
    if (auth.get('partition_role')!=role or auth.get('protocol_sha256')!=protocol_digest
            or auth.get('plan_sha256')!=plan_digest
            or protocol['family_plan_hashes'][ROLE_KEYS[role]]!=plan_digest
            or common.canonical_hash(physics.cfg)!=protocol['physics_config_sha256']):
        raise ValueError('Generation does not match the registered role/plan/calibration')
    allowed_phase = {'development_selection':'development-compare','null_calibration':'development-compare',
                     'null_evaluation':'development-compare','calibration_stress':'development-compare',
                     'training':'train-denoiser','signed_confirmation':'final-once','final_once':'final-once'}
    if auth.get('phase')!=allowed_phase[role]:
        raise ValueError('Family generation at the wrong stage')
    if role in SEALED:
        frozen_path = Path(auth.get('final_protocol_path','')).resolve()
        required_path = (common.REPORTS/'final_protocol.json').resolve()
        if frozen_path!=required_path or not frozen_path.is_file():
            raise ValueError('No frozen final protocol at the registered path')
        frozen_digest = common.sha(frozen_path)
        frozen = common.read_json(frozen_path)
        if (auth.get('final_protocol_sha256')!=frozen_digest
                or frozen.get('status')!='FROZEN_CANDIDATE_THRESHOLDS_STATISTICS'
                or not frozen.get('thresholds') or not frozen.get('statistics')
                or auth.get('candidate_receipt_sha256')!=frozen.get('candidate_receipt_sha256')):
            raise ValueError('Candidate, thresholds and statistics are not frozen')
        candidate = (common.REPORTS/'candidate_review'/'candidate_receipt.json').resolve()
        if not candidate.is_file() or common.sha(candidate)!=frozen['candidate_receipt_sha256']:
            raise ValueError('Frozen candidate receipt does not match')
        start = common.REPORTS/'FINAL_ONCE_STARTED.json'
        if not start.is_file() or (common.REPORTS/'FINAL_ONCE_RECEIPT.json').exists():
            raise ValueError('Final generation must occur inside the single registered execution')
        started = common.read_json(start)
        if started.get('final_protocol_sha256')!=frozen_digest:
            raise ValueError('One-time start guard binds another final protocol')
    return root, role, False, protocol_digest


def _actual_coefficient(row, cache):
    if row.get('coefficient_rule')!='original_v4_final_levels':
        return float(row['coefficient_e'])
    import numpy as np
    mid = row['mother_id']
    if mid not in cache:
        random = np.random.default_rng(row['mother']['template_seed']+301)
        cache[mid] = [0.,float(random.uniform(25.,55.)),float(random.uniform(140.,280.)),float(random.uniform(320.,520.))]
    index = row['original_condition_index']
    value = cache[mid][index]
    low,high = row['coefficient_range_e']
    if not low<=value<=high:
        raise RuntimeError('Original final amplitude left its registered range')
    return value


def generate_family(physics, plan, output_root, options=None):
    root,role,fixture,protocol_digest = _guard(physics,plan,output_root,options)
    # All sealed guards precede even importing the numerical mother generator.
    import numpy as np
    import torch
    from training_v4.data import mother_state
    sys.dont_write_bytecode = True
    root.mkdir(parents=True, exist_ok=True)
    mother_cache,amplitude_cache,label_cache = {},{},{}
    rows,labels,legality = [],[],[]
    maximum_dn = 2**int(physics.cfg['adc_bits'])-1
    with torch.no_grad():
        for descriptor in plan:
            row = copy.deepcopy(descriptor)
            mid = row['mother_id']
            if mid not in mother_cache:
                state,beta,shape,provenance = mother_state(physics,row['mother'])
                mother_cache[mid] = (state.detach().clone(),beta.detach().clone(),shape.detach().clone(),provenance)
            state,beta,shape,provenance = mother_cache[mid]
            coefficient = _actual_coefficient(row,amplitude_cache)
            row['registered_coefficient_e'] = row.get('coefficient_e')
            row['coefficient_e'] = coefficient
            stress = row.get('stress')
            label_key = (mid,coefficient,None if stress is None else stress['name'])
            if label_key not in label_cache:
                z = state.clone()
                z[:,4] = shape*coefficient/physics.Sscale
                if not torch.equal(z[:,:4],state[:,:4]):
                    raise RuntimeError('Signed injection modified a protected mother field')
                C,species,background,_ = physics.components(z,beta)
                nominal_C = C.clone()
                physical_C = C.clone()
                effective_C = C.clone()
                radiance = C+species+background
                if stress:
                    multipliers = C.new_tensor(stress['band_multipliers'])[None,:,None,None]
                    effective_C = nominal_C*multipliers
                    if stress['kind']=='channel_gain':
                        radiance = radiance*multipliers
                    else:
                        physical_C = effective_C
                        radiance = physical_C+species+background
                bands = physics.op.object_blur(radiance)
                expected = physics.op.encode_images(bands)[0]
                continuum_postblur = physics.op.object_blur(physical_C)
                effective_postblur = physics.op.object_blur(effective_C)
                nominal_postblur = physics.op.object_blur(nominal_C)
                reference_color,domain = physics.color_temperature(effective_postblur)
                nominal_color,_ = physics.color_temperature(nominal_postblur)
                latent_S = z[:,4]*physics.Sscale
                postblur_S = physics.op.object_blur(latent_S)
                diagnostic = _expectation_diagnostics(physics,expected,radiance,bands)
                values = {'latent_state':z[0].cpu().numpy(),
                    'latent_temperature_K':(physics.T0+physics.Tscale*z[0,0]).cpu().numpy(),
                    'latent_log_amplitude':z[0,1].cpu().numpy(), 'beta':beta[0].cpu().numpy(),
                    'template_shape':shape.cpu().numpy(),'coefficient_e':np.asarray(coefficient,dtype=np.float64),
                    'continuum_latent':physical_C[0].cpu().numpy(),
                    'continuum_latent_nominal':nominal_C[0].cpu().numpy(),
                    'continuum_latent_reference':effective_C[0].cpu().numpy(),
                    'continuum_postblur':continuum_postblur[0].cpu().numpy(),
                    'continuum_postblur_nominal':nominal_postblur[0].cpu().numpy(),
                    'continuum_postblur_reference':effective_postblur[0].cpu().numpy(),
                    'template_latent_e':latent_S[0].cpu().numpy(),
                    'template_postblur_e':postblur_S[0].cpu().numpy(),'template_e':postblur_S[0].cpu().numpy(),
                    'temperature_color_K':reference_color[0].cpu().numpy(),
                    'temperature_color_K_nominal':nominal_color[0].cpu().numpy(),
                    'temperature_color_K_reference':reference_color[0].cpu().numpy(),
                    'evaluation_domain':domain[0].cpu().numpy(),'band_images_e':bands[0].cpu().numpy(),
                    'expected_e':expected.cpu().numpy()}
                suffix = row['condition'] + ('' if stress is None else '_'+stress['name'])
                path = root/'diagnostic_labels'/mid/(suffix+'.npz')
                _save_npz(path,values)
                label = {'mother_id':mid,'condition':row['condition'],'coefficient_e':coefficient,
                    'path':path.relative_to(root).as_posix(),'sha256':common.sha(path),
                    'partition_role':role,'diagnostics':diagnostic,'replayed_mother_provenance':provenance,
                    'protected_field_hashes':{key:_array_hash(values[key]) for key in
                        ('latent_temperature_K','latent_log_amplitude','beta','template_shape')},
                    'stress':copy.deepcopy(stress),
                    'color_truth_kind':'nominal postblur continuum color' if stress is None else
                        'measurement-equivalent reference color under declared calibration stress; nominal color is separate',
                    'signed_semantics':'synthetic deviation coefficient; negative is not concentration'}
                labels.append(label)
                label_cache[label_key] = (expected,label)
                legality.append({'mother_id':mid,'condition':row['condition'],'stress':stress,'coefficient_e':coefficient,**diagnostic})
            expected,label = label_cache[label_key]
            row.pop('mother',None)
            row.update(label_path=label['path'],label_sha256=label['sha256'],
                raw_available=False,raw_path=None,raw_file_sha256=None,raw_hash=None,
                expectation_diagnostics=label['diagnostics'],fixture_only=fixture,
                raw_generation_status='PENDING',physics_config_sha256=common.canonical_hash(physics.cfg))
            if not label['diagnostics']['expected_all_finite']:
                row['raw_generation_status']='UNAVAILABLE_NONFINITE_EXPECTATION'
                rows.append(row);continue
            if row['input_kind']=='noise' and not label['diagnostics']['expected_nonnegative']:
                row['raw_generation_status']='UNAVAILABLE_NEGATIVE_POISSON_MEAN'
                rows.append(row);continue
            if row['input_kind']=='noise':
                generator = torch.Generator(device=physics.M.device).manual_seed(row['noise_seed'])
                sampled = torch.poisson(expected,generator=generator)+torch.randn(expected.shape,
                    generator=generator,device=physics.M.device,dtype=expected.dtype)*physics.cfg['read_noise_e']
                unbounded = (sampled/physics.adc_step+physics.cfg['black_level_DN']).round()
                raw = unbounded.clamp(0,maximum_dn).to(torch.uint16)
            elif row['input_kind']=='quantization':
                unbounded = (expected/physics.adc_step+physics.cfg['black_level_DN']).round()
                raw = unbounded.clamp(0,maximum_dn).to(torch.uint16)
            else:
                unbounded = expected/physics.adc_step+physics.cfg['black_level_DN']
                raw = unbounded
            raw_path = root/'raw'/mid/(row['case_id']+'.npz')
            raw_array = raw.cpu().numpy()
            _save_npz(raw_path,{'raw_dn':raw_array})
            row.update(raw_available=True,raw_path=raw_path.relative_to(root).as_posix(),
                raw_file_sha256=common.sha(raw_path),raw_hash=_array_hash(raw_array),
                raw_generation_status='GENERATED',shape=list(raw_array.shape),dtype=str(raw_array.dtype),
                ADC_diagnostics={'upper_clipped_pixels':int((unbounded>=maximum_dn).sum()),
                                 'lower_clipped_pixels':int((unbounded<0).sum()),
                                 'saturated_pixels':int((raw.to(physics.M.dtype)>=maximum_dn).sum())})
            rows.append(row)
    by_mother={}
    for label in labels:
        by_mother.setdefault(label['mother_id'],set()).add(common.canonical_hash(label['protected_field_hashes']))
    assert all(len(values)==1 for values in by_mother.values()), 'Matched family changed protected fields'
    manifest={'schema':'v5-explicit-authorized-family-v1','fixture_only':fixture,'partition_role':role,
        'generator_source_sha256':common.sha(Path(__file__)),'plan_sha256':common.canonical_hash(plan),
        'protocol_sha256':protocol_digest,'physics_config_sha256':common.canonical_hash(physics.cfg),
        'planned_rows':len(plan),'generated_raw_rows':sum(r['raw_available'] for r in rows),
        'unavailable_rows':sum(not r['raw_available'] for r in rows),'labels':labels,'rows':rows,
        'selected_mothers':[copy.deepcopy(next(r['mother'] for r in plan if r['mother_id']==mid)) for mid in mother_cache],
        'matched_T_loga_beta_shape_verified':True,'mother_fields_modified_for_legality':False,
        'same_noise_realization_claimed':False,'labels_accepted_by_deployment':False,
        'original_final_definition_preserved':role=='final_once',
        'confirmation_test_opened':role=='signed_confirmation','final_test_opened':role=='final_once'}
    common.write_json(root/'expectation_legality.json',legality)
    common.write_json(root/'data_manifest.json',manifest)
    return manifest
