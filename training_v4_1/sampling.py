import numpy as np
from training_v4_1.common import *

def build_draws(rows):
    if (OUT/'sampling_sequence.json').exists(): return read(OUT/'sampling_sequence.json')['draws']
    rng = np.random.default_rng(PROTOCOL['seed'])
    train = [r for r in rows if r['split'] == 'train']
    draws = []
    cycles = [[] for _ in range(4)]
    signed_cycles = [[] for _ in range(4)]
    for update in range(1500):
        batch = []
        for slot, kind in enumerate(('noise','noise','continuous','quantization')):
            signed = update % 5 == 1 and slot == (update//5) % 4
            sequence = signed_cycles[slot] if signed else cycles[slot]
            conditions = ['negative_weak','negative_normal','positive_weak','positive_normal'] if signed else ['blank','weak','normal','strong']
            if not sequence: sequence.extend(rng.permutation(conditions).tolist())
            condition = sequence.pop()
            pool = [r for r in train if r['input_kind'] == kind and r['signed_aux'] == signed and r['condition'] == condition]
            assert pool
            batch.append(pool[int(rng.integers(len(pool)))]['case_id'])
        draws.append(batch)
    write(OUT/'sampling_sequence.json', {'seed': PROTOCOL['seed'], 'draws': draws,
        'roles': ['noise','noise','continuous','quantization'], 'reused_exactly_for_all_controls': True})
    return draws
