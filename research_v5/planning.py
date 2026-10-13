"""Metadata-only, disjoint mother registration; no array generation."""
import copy
import hashlib
from .common import ROOT, read_json, canonical_hash
from .data import TEMPERATURE_GEOMETRIES, TEMPLATE_GEOMETRIES, COEFFICIENTS_E, CONDITIONS


def noise_seed(case_id):
    return int.from_bytes(hashlib.sha256(('research-v5-frozen-noise:' + case_id).encode()).digest()[:8], 'little') % (2**63-1)


def new_mothers(role, count, seed_base, *, full_factorial=False):
    mothers = []
    for index in range(count):
        t = index//6 if full_factorial else index % 6
        s = index % 6 if full_factorial else (index//6 + index) % 6
        seed = seed_base + index*1009
        mothers.append({'mother_id': f'v5_{role}_mother_{index:04d}',
            'split': 'train' if role == 'training' else 'validation', 'partition_role': role,
            'index': index, 'seed': seed, 'temperature_seed': seed+1, 'amplitude_seed': seed+29,
            'template_seed': seed+53, 'background_seed': seed+71,
            'temperature_geometry': TEMPERATURE_GEOMETRIES[t], 'template_geometry': TEMPLATE_GEOMETRIES[s],
            'fields_from': 'unchanged training_v4.data.mother_state with fresh registered seed streams'})
    return mothers


def family_plan(mothers, *, coefficients=COEFFICIENTS_E, kinds=('noise',), noise_repeats=1, role=None, stress=None):
    plan = []
    for mother in mothers:
        role_name = role or mother['partition_role']
        for coefficient in coefficients:
            condition = CONDITIONS[COEFFICIENTS_E.index(float(coefficient))]
            for kind in kinds:
                for repeat in range(noise_repeats if kind == 'noise' else 1):
                    case = f"{mother['mother_id']}_{condition}_{kind}{repeat if kind=='noise' else ''}"
                    if stress:
                        case += '_' + stress['name']
                    plan.append({'case_id': case, 'mother_id': mother['mother_id'], 'mother': copy.deepcopy(mother),
                        'partition_role': role_name, 'condition': condition,
                        'group': 'blank' if coefficient == 0 else 'weak' if abs(coefficient)==45 else 'normal',
                        'coefficient_e': float(coefficient), 'input_kind': kind,
                        'noise_index': repeat if kind == 'noise' else None,
                        'noise_seed': noise_seed(case) if kind == 'noise' else None,
                        'geometry': mother['temperature_geometry'], 'template_geometry': mother['template_geometry'],
                        'stress': copy.deepcopy(stress), 'same_noise_realization_claimed': False,
                        'truth_role': 'separate diagnostic labels; never passed to deployment'})
    return plan


def build_family_plans():
    train = new_mothers('training', 36, 2000000, full_factorial=True)
    development = new_mothers('development_selection', 36, 3000000, full_factorial=True)
    null_cal = new_mothers('null_calibration', 12, 4000000)
    null_eval = new_mothers('null_evaluation', 12, 5000000)
    registry = read_json(ROOT/'reports/training_v4_4/run_001/data_audit/repair_parent_registry.json')
    locked = [copy.deepcopy(m) for m in registry['new_parents'] if m['partition_role']=='development_confirmation_locked']
    for m in locked:
        m['partition_role'] = 'signed_confirmation'
    final = copy.deepcopy(read_json(ROOT/'reports/training_v4/run_001/final_test_registration.json')['parents'])
    for m in final:
        m['partition_role'] = 'final_once'
    all_groups = [train, development, null_cal, null_eval, locked, final]
    identities = [m['mother_id'] for group in all_groups for m in group]
    assert len(identities)==len(set(identities)) and len(locked)==36 and len(final)==64
    scene_seeds = [m['seed'] for group in all_groups for m in group]
    assert len(scene_seeds)==len(set(scene_seeds))
    stress_mothers = [development[t*6+t] for t in range(6)]
    stress = []
    for spec in ({'name':'relative_gain_700_905', 'kind':'channel_gain',
                  'band_multipliers':[1.,1.,1.,1.,1.,.97,1.03]},
                 {'name':'emissivity_700_905', 'kind':'continuum_emissivity',
                  'band_multipliers':[1.,1.,1.,1.,1.,.95,1.05]}):
        stress.extend(family_plan(stress_mothers, coefficients=(0.,180.,-180.), role='calibration_stress', stress=spec))
    original_final=[]
    # Keep the original final-set definition: four positive conditions, two
    # noise draws, original seed streams and seed-derived amplitude ranges.
    # Do not draw final amplitudes or generate their fields before candidate freeze.
    for mother in final:
        for ci,(condition,bounds) in enumerate(zip(('blank','weak','normal','strong'),
                                                 ((0.,0.),(25.,55.),(140.,280.),(320.,520.)))):
            for repeat in range(2):
                case=f"{mother['mother_id']}_{condition}_noise{repeat}"
                original_final.append({'case_id':case,'mother_id':mother['mother_id'],'mother':copy.deepcopy(mother),
                    'partition_role':'final_once','condition':condition,'group':condition,
                    'coefficient_e':0. if condition=='blank' else None,
                    'coefficient_rule':'original_v4_final_levels','coefficient_seed':mother['template_seed']+301,
                    'original_condition_index':ci,'coefficient_range_e':list(bounds),
                    'input_kind':'noise','noise_index':repeat,
                    'noise_seed':mother['raw_noise_seed_base']+ci*17+repeat,
                    'geometry':mother['temperature_geometry'],'template_geometry':mother['template_geometry'],
                    'stress':None,'same_noise_realization_claimed':False,
                    'truth_role':'one-time original final evaluation after candidate freeze'})
    plans = {'development': family_plan(development, kinds=('continuous','quantization','noise')),
        'null_calibration': family_plan(null_cal, coefficients=(0.,), noise_repeats=8),
        'null_evaluation': family_plan(null_eval, coefficients=(0.,), noise_repeats=8),
        'training': family_plan(train, noise_repeats=2),
        'calibration_stress': stress,
        'signed_confirmation': family_plan(locked, coefficients=(45.,-45.,180.,-180.), noise_repeats=2),
        'final_once': original_final}
    cases = [r['case_id'] for group in plans.values() for r in group]
    assert len(cases)==len(set(cases))
    seeds = [r['noise_seed'] for group in plans.values() for r in group if r['noise_seed'] is not None]
    assert len(seeds)==len(set(seeds))
    return {'mothers':dict(zip(('training','development','null_calibration','null_evaluation','signed_confirmation','final_once'),all_groups)),
            'plans':plans, 'plan_hashes':{name:canonical_hash(rows) for name,rows in plans.items()},
            'confirmation_and_final_arrays_generated':False,
            'locked_metadata_only':True, 'no_arrays_opened':True}
