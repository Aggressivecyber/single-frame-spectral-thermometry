"""Metadata-only complete preseal and final factors; no arrays are generated."""
from . import storage as s
from . import experiment_registry as registry
import copy
import hashlib

GEOMETRIES=('hotspot_small','hotspot_large','double_hotspot','edge','filament')
CONDITIONS=(('blank',0.),('positive_weak',45.),('negative_weak',-45.),('positive_normal',180.),('negative_normal',-180.))
PERTURBATIONS={
    'gain':lambda a:{'relative_gain':[1+a,1-a,1+a,1-a,1+a,1-a,1+a]},
    'response':lambda a:{'response_tilt':a},
    'PSF':lambda a:{'psf_fraction':a},
    'registration':lambda a:{'registration_xy_px':[a,-a]},
    'emissivity':lambda a:{'relative_emissivity':[1.]*5+[1+a,1.]}}

def seed(namespace,*parts):
    return int.from_bytes(hashlib.sha256(('/'.join(map(str,('closeout001',namespace,*parts)))).encode()).digest()[:8],'big')&((1<<63)-1)

def descriptor(mid,base,index):
    types=('front','multipeak','ring','stripe','flat','smooth_random')
    templates=('blob','multipeak','ring','stripe','filament','smooth_random')
    return {'mother_id':mid,'seed':base,'temperature_seed':base+1,'amplitude_seed':base+29,
        'template_seed':base+53,'background_seed':base+71,
        'temperature_geometry':types[index%6],'template_geometry':templates[(index//6)%6]}

def build():
    mothers=[];cases=[]
    def mother(family,index,base,engine='core'):
        mid=f'closeout_{family}_mother_{index:04d}'
        d=descriptor(mid,base+index*101,index);mothers.append({'family':family,'engine':engine,**d});return d
    def case(d,family,condition,coefficient,**extra):
        cid=d['mother_id']+'_'+family+'_'+condition
        row={'case_id':cid,'mother_id':d['mother_id'],'mother':d,'family':family,'condition':condition,
            'coefficient_e':float(coefficient),'photon_ratio':1.,'normalization':'fixed_peak',
            'geometry':GEOMETRIES[d['seed']%5],'engine':'core','perturbation':{},**extra}
        row['noise_seed']=seed('raw-noise',cid);cases.append(row);return row
    for i in range(400):
        d=mother('null',i,8100000)
        case(d,'null_calibration' if i<200 else 'null_evaluation','blank',0.)
    for i in range(24):
        d=mother('independent',i,8200000,'independent');coefficient=CONDITIONS[i%5][1]
        effects={'nominal':{},'gain':PERTURBATIONS['gain'](.02),'response':PERTURBATIONS['response'](.25),
            'PSF':PERTURBATIONS['PSF'](.1),'registration':PERTURBATIONS['registration'](.25),
            'emissivity':PERTURBATIONS['emissivity'](.02)}
        if i<2:
            effects['combined_declared']={**PERTURBATIONS['gain'](.02),**PERTURBATIONS['response'](.25),
                **PERTURBATIONS['PSF'](.1),**PERTURBATIONS['registration'](.25),**PERTURBATIONS['emissivity'](.02)}
        for effect,perturb in effects.items():
            case(d,'independent',effect,coefficient,engine='independent',perturbation=perturb,
                geometry=GEOMETRIES[i%5])
    d=mother('structure',0,8500000)
    for geometry in GEOMETRIES:
        for normalization in ('fixed_peak','fixed_total_signal'):
            for level in (0.,15.,-15.,30.,-30.,45.,-45.,90.,-90.,180.,-180.):
                case(d,'structure',f'{geometry}_{normalization}_{level:+g}',level,
                    geometry=geometry,normalization=normalization)
    for i in range(6):
        d=mother('photons',i,8510000)
        for ratio in (.25,1.,2.):
            for condition,level in CONDITIONS:
                case(d,'photons',condition+f'_budget{ratio:g}',level,photon_ratio=ratio)
    for i in range(3):
        d=mother('calibration',i,8520000,'independent')
        for kind,levels in {'gain':(.005,.02,.05),'response':(.1,.25,.5),'PSF':(.03,.1,.2),
            'registration':(.1,.25,.5),'emissivity':(.005,.02,.05)}.items():
            for level in levels:
                case(d,'calibration',kind+f'_{level:g}',180. if i%2==0 else -180.,
                    engine='independent',perturbation=PERTURBATIONS[kind](level),geometry=GEOMETRIES[i])
    for i in range(2):
        d=mother('dynamic',i,8300000,'independent')
        for frame in range(6):
            case(d,'dynamic',f'frame{frame:02d}',180. if i==0 else -180.,engine='independent',
                geometry='hotspot_small' if i==0 else 'double_hotspot',mid_fraction=frame/5,width_fraction=.2)
    for i in range(12):
        d=mother('uncertainty',i,8400000);condition,level=CONDITIONS[i%6 if i%6<5 else 3]
        row=case(d,'uncertainty',condition,level,role='interval_calibration' if i<6 else 'interval_evaluation',
            MC=128 if i in (0,8) else 64,initial_MC=64,linearization_probes=8)
        row['MC_noise_seeds']=[seed('full-inverse-MC',row['case_id'],j) for j in range(row['MC'])]
        row['linearization_seed']=seed('whole-inverse-directions',row['case_id'])
    old=s.read(s.ROOT/'research_v5/family_plans.json');final_main=copy.deepcopy(old['plans']['final_once'])
    signed_mothers={r['mother_id']:r['mother'] for r in old['plans']['signed_confirmation']}
    final_signed=[]
    for mid,d in sorted(signed_mothers.items()):
        for condition,level in CONDITIONS:
            final_signed.append({'case_id':mid+'_closeout_'+condition+'_noise0','mother_id':mid,'mother':d,
                'family':'sealed_signed','condition':condition,'coefficient_e':level,'input_kind':'noise',
                'noise_seed':seed('signed-final-raw',mid,condition),'noise_index':0})
    prior_mothers={d['mother_id']:d for family in old['mothers'].values() for d in family} if isinstance(old['mothers'],dict) else {d['mother_id']:d for d in old['mothers']}
    # Exact original metadata formula, without importing/drawing mother fields.
    # Its source is included below; train/validation/final ranges are preserved.
    original_noise=set()
    for split,n,base in (('train',256,100000),('validation',64,200000),('final_unopened',64,300000)):
        for index in range(n):
            b=base+index*113;mid=f'v4_{split}_mother_{index:04d}'
            prior_mothers.setdefault(mid,descriptor(mid,b,index))
            original_noise.update(b+10000000+ci*17+j for ci in range(4) for j in range(2))
    # Parse only metadata; no sealed raw/label NPZ is touched by planning.
    new_ids={d['mother_id'] for d in mothers};old_ids=set(prior_mothers)
    field_keys=('temperature_seed','amplitude_seed','template_seed','background_seed')
    old_fields={d[k] for d in prior_mothers.values() for k in field_keys if k in d}
    new_fields=[d[k] for d in mothers for k in field_keys]
    new_noise=[r['noise_seed'] for r in cases+final_signed]+[v for r in cases for v in r.get('MC_noise_seeds',[])]
    old_noise={r['noise_seed'] for group in old['plans'].values() for r in group if r.get('noise_seed') is not None}
    old_noise|=original_noise
    old_noise|={r['noise_seed'] for r in s.read(s.ROOT/'research_v6/input_inventory.json')['rows'] if isinstance(r.get('noise_seed'),int)}
    assert not new_ids&old_ids and len(new_fields)==len(set(new_fields)) and not set(new_fields)&old_fields
    assert len(new_noise)==len(set(new_noise)) and not set(new_noise)&old_noise
    assert len(final_main)==512 and len(final_signed)==180 and len(signed_mothers)==36
    assert len({r['case_id'] for r in cases})==len(cases)
    counts={f:sum(r['family']==f for r in cases) for f in sorted({r['family'] for r in cases})}
    return {'schema':'closeout-science-plan-001','mothers':mothers,'cases':cases,'case_counts':counts,
        'final_main_metadata':final_main,'final_signed_metadata':final_signed,
        'sealed_payloads_opened':False,'signed_supplement':{'factors':dict(CONDITIONS),
            'one_noise_replica_per_condition':True,'negative_coefficients_are_not_concentrations':True,
            'all6_by6_temperature_template_geometry_pairs_retained':True,
            'negative_mean_unavailable_not_clipped':True,'main512_original_levels_and_two_noise_seeds_retained':True},
        'disjointness':{'new_mother_ids_vs_old':True,'new_field_streams_vs_old':True,
            'new_raw_streams_vs_old':True,'new_field_streams_mutually_unique':True,
            'new_raw_streams_mutually_unique':True,'old_metadata_sha256':s.sha(s.ROOT/'research_v5/family_plans.json'),
            'original_v4_descriptor_source_sha256':s.sha(s.ROOT/'training_v4/data.py'),
            'original_v4_train256_validation64_final64_included':True,
            'new_mothers':len(mothers),'new_field_seeds':len(new_fields),'new_raw_seeds':len(new_noise)},
        'prospective_new_call_ceiling':{'null':1600,'independent':438,
            'noise_structure_strength_calibration_dynamic':771,'uncertainty':1292,'sealed':2768},
        'methods':['v3_physics','old_B4','B1_lambda100'],
        'null_and_final_methods':['v3_physics','old_B4','B1_lambda100','B1_lambda0.1'],
        'selected_and_v3_identical_route_reused':True,'reference_electrons_exposure_s':.000020,
        'finite_optical_targets':{'core':'frozen original object OTF','independent':'declared finite Airy PSF plus pixel integration'},
        'calibration_distribution':'assumed simulation, no external measured calibration',
        'null_rule':'27-position/scale/sign dictionary max; method-specific order191 of200; strict greater;200 new evaluation mothers',
        'interval_rule':{'calibration_mothers':6,'evaluation_mothers':6,'MC_initial':64,'stability':[32,64,128],
            'draw_generation':'independently specified nominal synthetic Poisson/read/ADC measurement model',
            'whole_inverse_repeated':True,'truth_only_external_scoring':True,
            'operational_width_model':'per-statistic calibration constant times sqrt(fitted raw mean + read/ADC variance)',
            'calibration':'95th empirical absolute standardized error on first64 draws of6 calibration mothers',
            'evaluation':'different6 mothers, no inflation/refit using their labels',
            'unit':'marginal pixel or ROI statistic; mother-clustered interpretation; not simultaneous full image',
            'distribution_free95_coverage_guaranteed':False},
        'statistics':{'primary':'fixed full truth domain; signed template; failures and unavailable inputs retained',
            'pooling_unit':'mother; noise copies paired within mother, never independent mother counts',
            'paired_bootstrap_mother_repeats':2000,'bootstrap_seed':8750001,
            'calibration_stress_not_pooled_with_nominal':True,'final_driven_reselection':False}}

def register():
    assert not (s.REPORTS/'SCIENCE_REGISTRATION.json').exists(),'Never overwrite an executed scientific plan'
    checks=s.read(s.REPORTS/'SCIENCE_CONTRACTS.json');assert checks['all_passed']
    value=build();s.write(s.REPORTS/'SCIENTIFIC_PLAN.json',value)
    reg={'time':s.now(),'source_sha256':registry.source_hashes(),'plan_sha256':s.sha(s.REPORTS/'SCIENTIFIC_PLAN.json'),
        'selected_method_sha256':s.sha(s.REPORTS/'selected_method.json'),
        'numeric_contract_sha256':s.sha(s.REPORTS/'SCIENCE_CONTRACTS.json'),'final_payloads_opened':False}
    s.write(s.REPORTS/'SCIENCE_REGISTRATION.json',reg);return reg
