import os
from training_v4_1.common import *
from training_v4_1.cache import CachePool
from training_v4_1.sampling import build_draws

def main():
    setup(); manifest = read(OUT/'data_manifest.json')
    rows = manifest['rows']; draws = build_draws(rows); by_id = {r['case_id']: r for r in rows}
    ordered = [by_id[i] for batch in draws[:100] for i in batch]
    ordered += [r for r in rows if r['split'] == 'validation']
    ordered += [by_id[i] for batch in draws[100:] for i in batch]+rows
    seen = set(); priority = []
    for row in ordered:
        if row['case_id'] not in seen: priority.append(row); seen.add(row['case_id'])
    write(OUT/'cache_producer_status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    try:
        pool = CachePool(priority, read(OLD_OUT/'backbone_selection.json')); pool.finish()
        write(OUT/'cache_producer_status.json', {'status': 'COMPLETE', 'pid': os.getpid(), 'cases': len(rows)})
    except BaseException as exc:
        write(OUT/'cache_producer_status.json', {'status': 'FAILED', 'pid': os.getpid(),
            'exception': type(exc).__name__, 'message': str(exc)})
        raise

if __name__ == '__main__': main()
