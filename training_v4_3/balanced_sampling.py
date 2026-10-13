"""A frozen, single-slot sampling intervention on existing training cases."""
import random
from collections import Counter
import runtime
from training_v4.common import read, write
from evaluation_v2.common import canonical_hash
from training_v4_1.common import OUT as OLD

OUT = runtime.ROOT/'reports/training_v4_3/run_001/negative_balanced'


def prepare():
    rows = read(OLD/'data_manifest.json')['rows']
    by_id = {r['case_id']:r for r in rows}
    reference = read(OLD/'sampling_sequence.json')['draws'][:500]
    rng = random.Random(4101)
    pools = {c:sorted(r['case_id'] for r in rows if r['split']=='train'
                     and r['input_kind']=='noise' and r['condition']==c)
             for c in ('negative_weak','negative_normal')}
    assert all(len(pool)==12 for pool in pools.values())
    pending = {c:[] for c in pools}
    draws, substitutions = [], []
    for index, batch in enumerate(reference):
        assert [by_id[i]['input_kind'] for i in batch]==['noise','noise','continuous','quantization']
        condition = 'negative_weak' if index%2==0 else 'negative_normal'
        if not pending[condition]:
            pending[condition] = pools[condition].copy()
            rng.shuffle(pending[condition])
        new_case = pending[condition].pop()
        new_batch = [batch[0],new_case,batch[2],batch[3]]
        assert all(by_id[i]['split']=='train' for i in new_batch)
        draws.append(new_batch)
        substitutions.append({'update':index+1,'slot':1,'before':batch[1],'after':new_case,
                              'condition':condition})
    counts=Counter((by_id[i]['condition'],by_id[i]['input_kind']) for b in draws for i in b)
    value={'draws':draws,'substitutions':substitutions,'seed':4101,
        'draw_sha256':canonical_hash(draws),'original_draw_sha256':canonical_hash(reference),
        'unchanged_slot_comparisons':1500,'all_unchanged_slots_match':all(a[j]==b[j] for a,b in zip(draws,reference) for j in (0,2,3)),
        'training_manifest_sha256':runtime.sha256(OLD/'data_manifest.json'),
        'pool_case_ids':pools,
        'counts':[{'condition':k[0],'input_kind':k[1],'microbatches':v} for k,v in sorted(counts.items())],
        'in_batch_duplicate_count':sum(len(set(b))<4 for b in draws),
        'validation_used_for_sampling':False,'new_training_cases':0}
    assert value['all_unchanged_slots_match'] and len(draws)==500
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'sampling_sequence.json'
    if path.exists():
        assert read(path)==value,'Never change a registered sequence'
    else:
        write(path,value)
    return value


if __name__=='__main__':
    s=prepare()
    print({'draw_sha256':s['draw_sha256'],'unchanged_slots':s['unchanged_slot_comparisons'],
           'in_batch_duplicates':s['in_batch_duplicate_count'],'counts':s['counts']})
