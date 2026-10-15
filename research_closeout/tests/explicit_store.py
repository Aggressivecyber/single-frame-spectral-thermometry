"""CPU contracts for the opt-in D store; never starts GPU or an inverse.

--no-write checks arithmetic, path rejection and a manufactured float64
forward reference without creating files. Full mode honors both live floors
and creates only one new, uniquely named current-task directory on D.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import uuid
from unittest import mock

import numpy as np
import torch

from .. import storage as s
from .. import compact_joint_cache as c
from .. import explicit_store as e
from .compact_joint_cache import FixturePhysics, original_fixture


def run(*, no_write=False):
    from ..runtime_entry import install_platform_fallback
    runtime_entry = install_platform_fallback()
    torch.set_num_threads(2)
    token = uuid.uuid4().hex
    store = e.ExplicitDStore('explicit_store_cpu_'+token, owner_token=token)
    checks = []

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    def rejects(action, expected=(e.StoreIntegrityError, c.CacheIntegrityError, c.CacheNotScoreable,
                                  FileNotFoundError, FileExistsError, ValueError)):
        try:
            action()
        except expected:
            return True
        return False

    check('fixed_D_prefixes_have_no_duplicate_closeout_segment_and_construction_is_read_only',
          str(store.data_root).replace('\\', '/') == 'D:/CodexWork/thermometry/closeout_001/outputs/repair_001/'+store.task_name
          and str(store.report_root).replace('\\', '/') == 'D:/CodexWork/thermometry/closeout_001/reports/repair_001/'+store.task_name
          and not store.data_root.exists() and not store.report_root.exists())
    check('absolute_resolved_root_escape_and_other_task_writes_are_rejected',
          all(rejects(lambda path=path: e._checked_path(path, store.roots)) for path in
              (Path('C:/forbidden'), e.DATA/'other_task/file', store.data_root/'../../escape',
               Path('relative/file'), e.PROJECT/'outputs/repair_001_elsewhere/file',
               store.data_root/'cache.npz:unbilled_stream', store.data_root/'CON.txt',
               store.data_root/'EXPLICIT_STORE_OWNER.json.'))
          and rejects(lambda: store.write_path(store.report_root/'explicit_store_owner.json'))
          and rejects(lambda: e.ExplicitDStore('../bad'))
          and rejects(lambda: e.ExplicitDStore('NUL'))
          and rejects(lambda: e.ExplicitDStore('safe', owner_token='short')))
    check('unchanged_E_whitelist_still_rejects_the_new_D_roots',
          rejects(lambda: s.write_path(store.data_root))
          and not c.format_contract()['new_D_root_writes_supported']
          and c.format_contract(storage=store)['new_D_root_writes_supported'])
    budget = e.ExplicitBudget(store.roots,
        free_bytes={'E': e.E_FREE_FLOOR_BYTES, 'D': e.D_FREE_FLOOR_BYTES+100})
    exact = {'instance': {'bytes': budget.instance_limit-100},
             'D_project': {'bytes': e.D_PROJECT_LIMIT_BYTES-100},
             'global_new_study': {'bytes': e.GLOBAL_LIMIT_BYTES-100}}
    with mock.patch.object(budget, 'ledger', return_value=exact):
        equality = budget.check(100)
        over = rejects(lambda: budget.check(101))
    check('D_256MiB_global_4GiB_D_128MiB_and_E_2GiB_exact_boundaries',
          equality['free_after_reserve_bytes'] == e.D_FREE_FLOOR_BYTES
          and equality['E_free_after_D_reserve_bytes'] == e.E_FREE_FLOOR_BYTES and over)
    empty = {key: {'bytes': 0} for key in ('instance', 'D_project', 'global_new_study')}
    low_e = e.ExplicitBudget(store.roots,
        free_bytes={'E': e.E_FREE_FLOOR_BYTES-1, 'D': 100*2**30})
    low_d = e.ExplicitBudget(store.roots,
        free_bytes={'E': 100*2**30, 'D': e.D_FREE_FLOOR_BYTES+99})
    with mock.patch.object(low_e, 'ledger', return_value=empty), mock.patch.object(low_d, 'ledger', return_value=empty):
        check('abundant_D_cannot_bypass_E_floor_and_E_cannot_offset_D_reserve',
              rejects(lambda: low_e.check(0)) and rejects(lambda: low_d.check(100)))
    caps = []
    for name, limit in (('instance', budget.instance_limit), ('D_project', e.D_PROJECT_LIMIT_BYTES),
                        ('global_new_study', e.GLOBAL_LIMIT_BYTES)):
        values = {key: {'bytes': limit if key == name else 0} for key in empty}
        with mock.patch.object(budget, 'ledger', return_value=values):
            caps.append(rejects(lambda: budget.check(1)))
    check('each_independent_instance_project_and_global_cap_is_enforced', all(caps))
    check('global_ledger_includes_all_E_and_D_data_and_reports',
          budget.global_roots == tuple(p.resolve() for p in (s.DATA, s.REPORTS, e.DATA, e.REPORTS))
          and budget.d_project_roots == (e.DATA.resolve(), e.REPORTS.resolve()))
    raw_dn = np.array([[100, 101], [102, 103]], dtype=np.uint16)
    raw = torch.as_tensor(raw_dn.astype(np.float64))[None]
    p = FixturePhysics()
    original = original_fixture(p, raw)
    rebuilt = c.forward_reconstruct(p, raw, {key: c.array(original[key]) for key in c.PARAMETER_KEYS},
                                    original['info'], grid=2)
    check('float64_manufactured_forward_is_byte_exact_without_truth_or_an_inverse',
          all(c.descriptor(rebuilt[key]) == c.descriptor(original[key])
              for key in c.CANONICAL_KEYS+('continuum_latent',))
          and p.inverse_calls == 0 and p.M.device.type == 'cpu')
    invoked = set()
    for name in ('research_closeout/compact_joint_cache.py', 'research_closeout/explicit_store.py'):
        tree = ast.parse((s.ROOT/name).read_text(encoding='utf8'))
        invoked.update(node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                       for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, (ast.Name, ast.Attribute)))
    forbidden = {'minimize', 'lsmr', 'repaired_infer', 'refined_infer', 'initialize', 'load_diagnostic_truth'}
    check('cache_and_store_never_invoke_inverse_optimizer_solver_or_truth_loader', not (invoked & forbidden))
    e_free, d_free = shutil.disk_usage(s.ROOT).free, shutil.disk_usage(e.PROJECT.anchor).free
    if e_free < e.E_FREE_FLOOR_BYTES or d_free < e.D_FREE_FLOOR_BYTES:
        actual_refusal = rejects(lambda: store.write_path(store.report_root/'should_not_exist.json'))
        check('live_low_space_refuses_enrollment_without_creating_D_evidence',
              actual_refusal and not store.data_root.exists() and not store.report_root.exists(),
              E_free_bytes=e_free, D_free_bytes=d_free, actual_refusal_executed=True)
    else:
        check('live_low_space_refuses_enrollment_without_creating_D_evidence', True,
              E_free_bytes=e_free, D_free_bytes=d_free, actual_refusal_executed=False,
              reason='Live floors pass; synthetic one-byte failures were checked without filesystem writes')

    if not no_write:
        # No mocks apply to the store or its physical disk gates. Fail before
        # enrollment and before any fixture writes if either live floor fails.
        store.budget().check(2*2**20)
        old = [s.REPORTS/'repair_001/quick_v001/QUICK_REGISTRATION.json',
               s.REPORTS/'repair_001/quick_v001/QUICK_RESULT.json',
               s.REPORTS/'repair_001/quick_v002/QUICK_REGISTRATION.json',
               s.REPORTS/'repair_001/quick_v002/QUICK_RESULT.json']
        old += list((s.REPORTS/'repair_001/compact_cache_001').glob('COMPACT_CACHE_CONTRACTS*.json'))
        old_hashes = {str(path): s.sha(path) for path in old}
        raw_ref = store.save(store.data_root/'raw_fixture.npz', raw_dn=raw_dn)
        raw_ref.update(array_sha256=c.array_sha(raw_dn), dtype='uint16', shape=[2, 2], fixture_only=True)
        cases = []

        def new_case(name):
            physics = FixturePhysics()
            cases.append(physics)
            cache = c.CompactJointCache(p=physics, storage=store,
                identity={'registration_sha256': 'fixture-only-'+token,
                          'method': 'CPU-fixture-explicit-D-C1v3', 'case_id': name},
                input_reference=raw_ref,
                calibration={'fingerprint': 'a'*64, 'physical_objective_version': 'CPU-fixture-C1v3'},
                config={'grid': 2, 'fixture_only': True})
            return cache, physics, original_fixture(physics, raw)

        cache, pc, out = new_case('ready')
        ticket = cache.begin(raw)
        reserve = Path(cache._read('RESERVATION.json')['reservation']['path'])
        check('D_dense_fallback_is_reserved_and_only_a_new_owner_gets_one_ticket',
              ticket['run_new_inverse'] and reserve.stat().st_size == cache.plan['fallback_zip_bound_bytes']
              and not cache.begin(raw)['run_new_inverse'])
        check('foreign_task_owner_and_cache_owner_cannot_release_the_reserve',
              rejects(lambda: e.ExplicitDStore(store.task_name, owner_token='f'*32))
              and rejects(lambda: store.release_reservation(reserve, owner={'pid': 0, 'token': 'f'*32}))
              and rejects(lambda: store.release_reservation(reserve, owner=ticket['owner']))
              and reserve.exists())
        result = cache.persist(ticket, out, raw)
        check('D_persist_keeps_float64_bytes_verifies_forward_and_releases_only_own_reserve',
              result['status'] == 'LOSSLESS_READY' and not reserve.exists()
              and all(c.descriptor(cache.load_for_score(raw)[key]) == c.descriptor(out[key])
                      for key in c.CANONICAL_KEYS+('continuum_latent',))
              and cache.context['storage_contract'] == store.contract()
              and e.SOURCE in cache.sources)
        reopened = e.ExplicitDStore(store.task_name, owner_token=token)
        restored = c.CompactJointCache(p=pc, storage=reopened, identity=cache.context['identity'],
            input_reference=raw_ref, calibration=cache.context['calibration'], config=cache.context['config'])
        before_sha = s.sha(cache.data/'PARAMETERS.npz')
        check('same_owner_resume_and_external_score_retry_never_repeat_an_inverse',
              restored.resume(raw)['status'] == 'LOSSLESS_READY'
              and not restored.begin(raw)['run_new_inverse'] and pc.inverse_calls == 0
              and restored.record_score('cpu', {'status': 'finite'}, raw)
                  == restored.record_score('cpu', {'ignored': True}, raw)
              and s.sha(cache.data/'PARAMETERS.npz') == before_sha)
        failed, pf, bad = new_case('initial_mismatch')
        bad['predicted_raw_e'] = bad['predicted_raw_e'].clone()
        bad['predicted_raw_e'][0, 0, 0] = torch.nextafter(bad['predicted_raw_e'][0, 0, 0], torch.tensor(float('inf')))
        failure = failed.persist(failed.begin(raw), bad, raw)
        calls = pf.op.forward_calls
        check('D_initial_byte_mismatch_retains_full_original_fallback_and_stays_failed',
              failure['status'] == 'FAIL_CLOSED_INITIAL_BYTE_MISMATCH'
              and failure['original_fallback_available']
              and all(c.descriptor(failed.load_fallback()[key]) == c.descriptor(bad[key]) for key in c.CANONICAL_KEYS)
              and not failed.resume(raw)['score_allowed'] and not failed.begin(raw)['run_new_inverse']
              and rejects(lambda: failed.load_for_score(raw)) and pf.op.forward_calls == calls)
        interrupted, pi, oi = new_case('recover_parameters')
        with mock.patch.object(interrupted, '_verify_forward', side_effect=KeyboardInterrupt('fixture exit')):
            try:
                interrupted.persist(interrupted.begin(raw), oi, raw)
            except KeyboardInterrupt:
                pass
        check('D_parameter_first_crash_recovers_with_forward_only_and_no_new_ticket',
              (interrupted.data/'PARAMETERS.npz').exists()
              and interrupted.resume(raw)['status'] == 'LOSSLESS_READY'
              and not interrupted.begin(raw)['run_new_inverse'] and pi.inverse_calls == 0)
        unavailable, pu, ou = new_case('unexpected_floor_drop')
        unavailable_ticket = unavailable.begin(raw)
        with mock.patch.object(store, 'write_path', side_effect=e.StoreIntegrityError('simulated live floor drop')):
            stopped = unavailable.persist(unavailable_ticket, ou, raw)
        check('unexpected_floor_drop_cannot_claim_fallback_or_durable_receipt_and_never_reissues_a_ticket',
              stopped['status'] == 'FAIL_CLOSED_ORIGINAL_PERSISTENCE_UNAVAILABLE'
              and not stopped['original_fallback_available'] and not stopped['durable_failure_receipt_written']
              and not stopped['score_allowed'] and stopped['caller_must_stop_and_log_this_receipt']
              and (unavailable.report/'STARTED.json').exists()
              and not unavailable.begin(raw)['run_new_inverse'] and pu.inverse_calls == 0)
        later, pl, ol = new_case('later_mismatch')
        later.persist(later.begin(raw), ol, raw)
        pl.op.offset = 1.
        check('D_later_failure_keeps_parameters_hashes_and_explicitly_has_no_original_fallback',
              rejects(lambda: later.load_for_score(raw))
              and not later.inspect()['original_fallback_available']
              and (later.data/'PARAMETERS.npz').exists() and not (later.data/'FALLBACK.npz').exists()
              and not later.begin(raw)['run_new_inverse'])
        lengths = {'PARAMETERS.npz': 11, 'FALLBACK.npz': 13, 'FALLBACK_RESERVED_fixture.bin': 17,
                   'archive.pending.fixture': 19, 'metadata.json': 23, 'run.log': 29, 'cache.bin': 31}
        folder = store.data_root/'ledger_fixture'
        for name, length in lengths.items():
            path = store.write_path(folder/name)
            store.budget().check(length)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(bytes(length))
        ledger = c.tree_ledger((folder, folder))
        actual = store.budget().ledger()
        check('D_meta_reserve_fallback_pending_cache_logs_and_owner_markers_are_all_billed',
              ledger['bytes'] == sum(lengths.values()) and ledger['files'] == len(lengths)
              and actual['D_project']['bytes'] >= actual['instance']['bytes']
              and actual['global_new_study']['bytes'] >= actual['D_project']['bytes']
              and all((root/e.OWNER_FILE).exists() for root in store.roots))
        check('D_adapter_did_not_modify_or_convert_prior_E_evidence',
              all(s.sha(path) == digest for path, digest in old_hashes.items())
              and all(physics.inverse_calls == 0 for physics in cases))

    names = ('research_closeout/compact_joint_cache.py', 'research_closeout/explicit_store.py',
             'research_closeout/tests/compact_joint_cache.py', 'research_closeout/tests/explicit_store.py',
             'research_closeout/runtime_entry.py')
    result = {'time': s.now(), 'schema': e.SCHEMA, 'cache_schema': c.SCHEMA,
              'all_passed': all(item['passed'] for item in checks), 'checks': checks,
              'source_sha256': {name: s.sha(s.ROOT/name) for name in names},
              'storage_contract': store.contract(), 'format_contract': c.format_contract(storage=store),
              'runtime_entry': runtime_entry,
              'no_write_mode': no_write, 'D_persistence_contracts_executed': not no_write,
              'GPU_used': False, 'formal_complete_inversions': 0, 'truth_loaded': False,
              'native_GPU_byte_exact_qualification': False,
              'native_frame_conservative_plan': c.frame_plan(shape=(256, 256), bands=7, template_dimension=1153, grid=256)}
    if not no_write:
        name = 'EXPLICIT_STORE_CONTRACTS.json' if result['all_passed'] else 'EXPLICIT_STORE_CONTRACTS_INITIAL.json'
        destination = store.report_root/name
        store.write(destination, result)
        result['receipt'] = str(destination)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(no_write=args.no_write)['all_passed'] else 2)
