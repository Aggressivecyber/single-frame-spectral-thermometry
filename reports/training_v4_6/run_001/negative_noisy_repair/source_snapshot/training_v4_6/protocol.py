"""One bounded signed-parent/noise contrast; registration precedes generation."""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/training_v4_6/run_001'
DATA = ROOT/'outputs/training_v4_6/run_001'
WEIGHTS = ROOT/'training_v4_6/checkpoints'
INITIAL = ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
CALIBRATION = ROOT/'reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json'
PREVIOUS = ROOT/'reports/training_v4_5/run_001'
OLD_MANIFEST = ROOT/'reports/training_v4_1/run_001/data_manifest.json'
REGISTRY = ROOT/'reports/training_v4_4/run_001/data_audit/repair_parent_registry.json'
DESIGN = ROOT/'reports/training_v4_4/run_001/data_audit/frozen_data_repair_plan.json'
SOURCE_DRAWS = ROOT/'reports/training_v4_3/run_001/negative_balanced/sampling_sequence.json'
TEMP_TYPES = ['front','multipeak','ring','stripe','flat','smooth_random']
SPECIES_TYPES = ['blob','multipeak','ring','stripe','filament','smooth_random']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT)).replace('\\','/')


def write_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False)
        f.write('\n')


def planned_sampling():
    """Only metadata determine the draw sequence; no arrays or torch imported."""
    old=read(OLD_MANIFEST)
    original=read(SOURCE_DRAWS)['draws']
    assert len(original)==500 and all(len(x)==4 for x in original)
    from training_v4_6.data import new_case_specs
    new=new_case_specs()
    assert len(new)==432
    metadata=old['rows']+new
    by={r['case_id']:r for r in metadata}
    assert len(by)==len(metadata)==3784
    registry=read(REGISTRY)
    mothers={m['mother_id']:m for m in old['signed_mothers'] if m['split']=='train'}
    mothers.update({m['mother_id']:m for m in registry['new_parents'] if m['split']=='train'})
    assert len(mothers)==36
    def order(row):
        m=mothers[row['mother_id']]
        t=TEMP_TYPES.index(m['temperature_geometry'])
        s=SPECIES_TYPES.index(m['template_geometry'])
        noise=row.get('noise_index',0) if row['input_kind']=='noise' else 0
        if noise is None: noise=0
        return (noise,(s-t)%6,t,row['case_id'])
    pools=defaultdict(list)
    for r in metadata:
        if r['split']=='train' and r['signed_aux']:
            pools[(r['condition'],r['input_kind'])].append(r)
    for key,pool in pools.items():
        pool.sort(key=order)
        assert len(pool)==(72 if key[1]=='noise' else 36)
    counts=Counter(); draws=[]; replacements=[]
    for u,batch in enumerate(original,1):
        current=[]
        for j,case in enumerate(batch):
            oldrow=by[case]
            if oldrow['signed_aux'] and oldrow['condition'].startswith('negative_') and oldrow['input_kind']=='noise':
                key=(oldrow['condition'],oldrow['input_kind'])
                chosen=pools[key][counts[key]%len(pools[key])]
                counts[key]+=1
                newcase=chosen['case_id']
                replacements.append({'update':u,'microbatch':j,'old_case_id':case,'new_case_id':newcase,
                    'condition':key[0],'input_kind':key[1]})
            else:
                newcase=case
            current.append(newcase)
        draws.append(current)
    for before,after in zip(original,draws):
        for a,b in zip(before,after):
            ra,rb=by[a],by[b]
            assert rb['split']=='train'
            assert (ra['signed_aux'],ra['condition'],ra['input_kind'])==(rb['signed_aux'],rb['condition'],rb['input_kind'])
            changed_stratum=ra['signed_aux'] and ra['condition'].startswith('negative_') and ra['input_kind']=='noise'
            assert changed_stratum or a==b
    exposure=[]
    for key in sorted(pools):
        used=[by[c] for batch in draws for c in batch if by[c]['signed_aux'] and (by[c]['condition'],by[c]['input_kind'])==key]
        unique={r['case_id'] for r in used}
        mother_ids={r['mother_id'] for r in used}
        pairs={(mothers[m]['temperature_geometry'],mothers[m]['template_geometry']) for m in mother_ids}
        exposure.append({'condition':key[0],'input_kind':key[1],'draws':len(used),'unique_cases':len(unique),
            'unique_mothers':len(mother_ids),'geometry_pairs':len(pairs),'available_cases':len(pools[key]),
            'minimum_draws_per_available_case':min(Counter(r['case_id'] for r in used).get(r['case_id'],0) for r in pools[key])})
    negative=[x for x in exposure if x['condition'].startswith('negative') and x['input_kind']=='noise']
    assert len(negative)==2 and all(x['unique_cases']==72 and x['unique_mothers']==36 and x['minimum_draws_per_available_case']>=3 for x in negative)
    assert len(replacements)==513
    new_ids={r['case_id'] for r in new}
    actual_new=sorted({c for b in draws for c in b}&new_ids)
    return {'schema':'v4.6-paired-signed-parent-noise-draws','metadata_only':True,'draws':draws,
        'draw_sha256':canonical(draws),'source_draws_sha256':canonical(original),
        'source_sampling_file_sha256':sha(SOURCE_DRAWS),'unsigned_positions_unchanged':True,
        'condition_input_kind_positions_unchanged':True,
        'signed_microbatches':sum(by[c]['signed_aux'] for b in original for c in b),
        'negative_noisy_positions_replaced':len(replacements),
        'negative_noisy_resampled_positions':len(replacements),
        'resampled_positions_returning_original_case_id':sum(x['old_case_id']==x['new_case_id'] for x in replacements),
        'positions_with_actual_case_id_change':sum(x['old_case_id']!=x['new_case_id'] for x in replacements),
        'all_other_1487_case_ids_unchanged':True,
        'replacements':replacements,'exposure':exposure,'new_unique_cases_to_cache':actual_new,
        'ordering':'Only negative noisy positions: per condition, noise_index, cyclic geometry offset (S-T mod 6), T. Each stratum starts at index0. Repeats cycle without performance inspection. All other IDs retain their exact old positions.',
        'claim_limit':'Negative noisy 36-mother / two-noise exposure repair only. All positive and clean input IDs are unchanged. Newly generated positive/clean cases are not trained or cached by this contrast; no full-factorial-trained claim.',
        'final_test_used':False}


def register():
    path=OUT/'execution_protocol.json'
    if path.exists():
        value=read(path)
        assert value['frozen'] and read(OUT/'protocol.json')==value
        return value
    assert read(PREVIOUS/'balanced_fixed_continuum/status.json')['stage']=='COMPLETE'
    assert read(PREVIOUS/'balanced_fixed_continuum/training_receipt.json')['updates']==500
    candidate=read(PREVIOUS/'candidate_selection.json')
    assert candidate['candidate_count']==0 and candidate['selected'] is None
    audit=read(PREVIOUS/'science_review/input_integrity_receipt.json')
    assert audit['status']=='PASS' and not audit['failures']
    sampling=planned_sampling()
    assert not (OUT/'data_manifest.json').exists(), 'Registration must precede generation'
    write_new(OUT/'sampling_sequence.json',sampling)
    value=copy.deepcopy(read(PREVIOUS/'protocol.json'))
    value.update(schema='training_v4_6_execution_v1',name='v4.6 bounded negative noisy parent and repeat repair',
        registered_at=datetime.now(timezone.utc).isoformat(),frozen=True,authorized_by='root',split='train',
        allowed_actions=['generate_train','cache_drawn_new_train','train_one_paired_arm'],
        counts={'new_train_rows':432,'merged_rows':3784,'merged_train_rows':3008,'old_validation_rows':776,'new_train_mothers':24},
        cpu_threads=2,cache_workers=1,maximum_generation_seconds=600,maximum_cache_seconds=3600,
        maximum_gpu_seconds_per_training_arm=1800,train_updates=500,checkpoint_interval=100,
        locked_development_generated=False,final_test_used=False,public_push_authorized_in_this_stage=False,
        sampling_sequence_path=rel(OUT/'sampling_sequence.json'),sampling_sequence_sha256=sha(OUT/'sampling_sequence.json'),
        sampling_source=rel(OUT/'sampling_sequence.json'),sampling_file_sha256=sha(OUT/'sampling_sequence.json'),
        draws_sha256=sampling['draw_sha256'],source_draws_sha256=sampling['source_draws_sha256'],
        anchor_manifest_sha256=sha(OLD_MANIFEST),new_training_cases=432,
        planned_new_unique_cache_count=len(sampling['new_unique_cases_to_cache']),
        expected_new_cache_cases=len(sampling['new_unique_cases_to_cache']),
        one_changed_variable='Only the 513 negative noisy parent/noise choices change within exact condition positions; every other 1487 case ID stays paired to v4.5.',
        paired_controls=['v4.5 balanced_fixed_continuum'],restart_from_old_initial_not_continuation=True,
        hard_stop='One seed, one 500-update data contrast. No extra seeds/steps/losses/samplers after failure. No candidate closes this finite branch; preserve negative evidence and deliver. Locked DEV is generated only after a single candidate and endpoint are frozen, under its own protocol.',
        interpretation=sampling['claim_limit'],
        predecessor_protocol_sha256=sha(PREVIOUS/'protocol.json'),
        predecessor_candidate_selection_sha256=sha(PREVIOUS/'candidate_selection.json'),
        predecessor_training_receipt_sha256=sha(PREVIOUS/'balanced_fixed_continuum/training_receipt.json'),
        independent_review_sha256=sha(PREVIOUS/'science_review/DATA_REPAIR_REVIEW.md'),
        frozen_generation_design_sha256=sha(DESIGN),frozen_parent_registry_sha256=sha(REGISTRY),
        storage_budget_new_bytes=4*1024**3,minimum_free_E_bytes=8*1024**3)
    files=[ROOT/'reports/training_v4/run_001/data_manifest.json',OLD_MANIFEST,DESIGN,REGISTRY,
        ROOT/'training_v4/data.py',ROOT/'training_v4_1/data.py',ROOT/'physics.py',ROOT/'infer.py',ROOT/'network.py',
        ROOT/'checkpoints/selected_checkpoint.pt',ROOT/'reports/training_v4/run_001/backbone_selection.json',CALIBRATION,
        ROOT/'training_v4_4/model.py',ROOT/'training_v4_4/evaluate.py',ROOT/'training_v4_1/model.py',
        ROOT/'training_v4/scalar_engine.py',ROOT/'training_v4/model.py',ROOT/'training_v4_1/evaluate.py',
        ROOT/'training_v4_3/residual_scaling.py',ROOT/'training_v4/common.py',ROOT/'training_v4/train.py',
        ROOT/'training_v4_1/common.py',ROOT/'training_v4_1/cache.py',ROOT/'training_v4/cache.py',
        ROOT/'evaluation_v2/common.py',ROOT/'runtime.py',
        ROOT/'repair_v3/solver.py',
        ROOT/'solver.py',ROOT/'config.py',
        PREVIOUS/'balanced_fixed_continuum/status.json',PREVIOUS/'candidate_selection.json',
        PREVIOUS/'science_review/input_integrity_receipt.json',PREVIOUS/'science_review/DATA_REPAIR_REVIEW.md',INITIAL]
    files+=sorted((ROOT/'training_v4_6').glob('*.py'))
    assert (ROOT/'training_v4_6/data.py') in files and (ROOT/'training_v4_6/cache.py') in files and (ROOT/'training_v4_6/train.py') in files
    value['frozen_inputs_sha256']={rel(p):sha(p) for p in files}
    value['sources']={rel(p):sha(p) for p in (ROOT/'training_v4_6').glob('*.py')}
    write_new(OUT/'protocol.json',value)
    write_new(path,value)
    return value


if __name__=='__main__':
    value=register()
    print({'protocol':str(OUT/'execution_protocol.json'),'sha256':sha(OUT/'execution_protocol.json'),
           'new_unique_cache_count':value['planned_new_unique_cache_count'],'updates':value['train_updates']})
