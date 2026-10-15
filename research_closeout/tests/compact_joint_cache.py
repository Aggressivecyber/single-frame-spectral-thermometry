"""CPU-only cache contracts; 2x2 forward fixtures, no formal inverse or GPU."""
from .. import storage as s
from .. import compact_joint_cache as c
import ast
import copy
import json
from pathlib import Path
import uuid
from unittest import mock
import numpy as np
import torch
from torch import nn


class FixtureOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('object_otf', torch.ones((2, 2), dtype=torch.float64))
        self.offset = 0.
        self.forward_calls = 0

    def object_blur(self, value):
        return value

    def encode_images(self, value):
        self.forward_calls += 1
        return value.sum(1)+self.offset


class FixturePhysics(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = {'shape': [2, 2], 'black_level_DN': 0., 'fixture_only': True}
        self.register_buffer('M', torch.tensor([[.1, .2, 1.], [.2, .3, .5], [.3, .4, -.25]], dtype=torch.float64))
        self.register_buffer('q', torch.zeros(3, dtype=torch.float64))
        self.register_buffer('background_basis', torch.zeros((3, 2, 2), dtype=torch.float64))
        self.op = FixtureOperator()
        self.T0, self.Tscale, self.Bscale, self.adc_step, self.extra_variance = 3000., 1000., 1., 1., 2.
        self.fail_continuum = False
        self.inverse_calls = 0

    def continuum(self, temperature):
        if self.fail_continuum:
            raise ArithmeticError('Injected forward-only failure')
        return torch.stack((temperature, 2*temperature, 4*temperature), 1), None

    def raw_to_electrons(self, raw):
        return raw

    def color_temperature(self, bands):
        return bands[:, 0], bands[:, -1] > 0

    def initialize(self, *args, **kwargs):
        self.inverse_calls += 1
        raise AssertionError('A cache must never call an initializer or inverse')


def original_fixture(p, raw):
    """Independent known-DC forward reference, without cache reconstruction."""
    z = torch.tensor([[[[0., -0.], [.1, .2]],
                       [[np.log(.5), np.log(.75)], [0., -0.]]]], dtype=torch.float64)
    xi = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    beta = torch.zeros((1, 3), dtype=torch.float64)
    T = p.T0+p.Tscale*z[:, 0]
    C = torch.stack((T, 2*T, 4*T), 1)*z[:, 1:2].exp()
    u = torch.full((1, 2, 2), .5, dtype=torch.float64)
    bands = C+p.M[:, 2][None, :, None, None]*u[:, None]
    prediction = bands.sum(1)
    info = {'method': 'CPU-fixture-joint-C1v3', 'physical_objective_version': 'fixture-C1v3',
            'numerical_certified': False, 'stage': 3, 'final_gtol': 1e-6,
            'nonfinite_diagnostic': np.uint64(0x7ff8000000000042).view(np.float64).item(), 'truth_used': False}
    return {'temperature_color_K': C[:, 0], 'template_postblur_e_ref': u,
            'continuum_postblur_bands': C, 'predicted_raw_e': prediction,
            'predicted_raw_DN': prediction, 'standardized_residual': (prediction-raw)*(raw.clamp_min(0)+2).reciprocal().sqrt(),
            'valid': C[:, -1] > 0, 'low_signal': C[:, -1] <= 1,
            'z_grid': z, 'xi': xi, 'beta': beta, 'continuum_latent': C,
            'numerical_valid': False, 'info': info}


def run():
    torch.set_num_threads(2)
    token = uuid.uuid4().hex
    root = c.REPORT_DIR/'_cpu_fixtures'/token
    root.mkdir(parents=True, exist_ok=False)
    raw_dn = np.array([[100, 101], [102, 103]], dtype=np.uint16)
    raw_path = root/'raw_fixture.npz'
    s.save(raw_path, raw_dn=raw_dn)
    raw = torch.as_tensor(raw_dn.astype(np.float64))[None]
    raw_ref = {'path': str(raw_path), 'sha256': s.sha(raw_path), 'array_sha256': c.array_sha(raw_dn),
               'dtype': 'uint16', 'shape': [2, 2], 'fixture_only': True}
    baseline_files = [s.REPORTS/'repair_001/quick_v001/QUICK_REGISTRATION.json',
                      s.REPORTS/'repair_001/quick_v001/QUICK_RESULT.json',
                      s.REPORTS/'repair_001/quick_v002/QUICK_REGISTRATION.json',
                      s.REPORTS/'repair_001/quick_v002/QUICK_RESULT.json']
    baseline_hashes = {str(path): s.sha(path) for path in baseline_files}
    checks, all_physics = [], []

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    def rejects(action, expected=(c.CacheIntegrityError, c.CacheNotScoreable, FileNotFoundError)):
        try:
            action()
        except expected:
            return True
        return False

    def new_case(name, extra_sources=None):
        p = FixturePhysics()
        all_physics.append(p)
        report, data = root/name/'reports', root/name/'data'
        budget = c.Budget((report, data), global_roots=(root,),
                          free_bytes=c.FREE_FLOOR_BYTES+16*2**20)
        calibration = {'fingerprint': 'a'*64, 'physical_objective_version': 'fixture-C1v3',
                       'calibration_version': 'CPU_fixture_only'}
        cache = c.CompactJointCache(report, data, p=p,
            identity={'registration_sha256': 'fixture-registration', 'method': 'fixture-joint-C1v3', 'case_id': name},
            input_reference=raw_ref, calibration=calibration, config={'grid': 2, 'fixture_only': True},
            source_sha256=extra_sources, budget=budget)
        return cache, p, original_fixture(p, raw)

    cache, p, out = new_case('success')
    rebuilt = c.forward_reconstruct(p, raw, {k: c.array(out[k]) for k in c.PARAMETER_KEYS}, out['info'], grid=2)
    check('pure_forward_reconstruction_matches_independent_known_DC_reference',
          all(c.array_sha(rebuilt[k]) == c.array_sha(out[k]) for k in c.CANONICAL_KEYS+('continuum_latent',)),
          canonical_keys=list(c.CANONICAL_KEYS), inverse_calls=p.inverse_calls)
    ticket = cache.begin(raw)
    reserved = cache._read('RESERVATION.json')['reservation']
    before = c.tree_ledger((cache.report_root, cache.data_root))
    check('begin_densely_reserves_full_fallback_before_issuing_single_inverse_ticket',
          ticket['run_new_inverse'] and Path(reserved['path']).stat().st_size == cache.plan['fallback_zip_bound_bytes']
          and before['by_kind_bytes']['reserve'] == cache.plan['fallback_zip_bound_bytes']
          and not cache.begin(raw)['run_new_inverse'])
    check('reserve_cannot_be_released_before_byte_verification_or_by_foreign_owner',
          rejects(lambda: cache.release_reservation(ticket['owner']))
          and rejects(lambda: cache._own_reservation({'pid': ticket['owner']['pid'], 'token': 'foreign'}))
          and Path(reserved['path']).exists())
    result = cache.persist(ticket, out, raw)
    check('persist_reads_parameters_back_on_same_physics_and_device_before_lossless_acceptance',
          result['status'] == 'LOSSLESS_READY' and result['score_allowed']
          and result['lossless_representation_verified'] and result['write_read_same_physics_instance']
          and result['verification']['all_byte_identical'] and not result['numerical_certification']
          and not result['native_GPU_qualification_claimed'] and not Path(reserved['path']).exists())
    params, metadata, _ = cache._load_parameters()
    check('float64_parameter_bytes_including_signed_zero_and_original_info_are_exact',
          all(c.descriptor(params[k]) == c.descriptor(out[k]) for k in c.PARAMETER_KEYS)
          and c.array_sha(params['z_grid']) != c.array_sha(np.where(params['z_grid'] == 0, 0., params['z_grid']))
          and np.float64(metadata['original_info']['nonfinite_diagnostic']).view(np.uint64).item() == 0x7ff8000000000042
          and metadata['original_info']['final_gtol'] == 1e-6)
    check('metadata_binds_method_calibration_sources_original_input_device_and_dtype',
          metadata['context']['identity']['method'] == 'fixture-joint-C1v3'
          and metadata['context']['calibration']['fingerprint'] == 'a'*64
          and metadata['context']['input_reference'] == raw_ref
          and metadata['context']['runtime']['device'] == 'cpu'
          and metadata['context']['runtime']['dtype'] == 'torch.float64'
          and 'research_closeout/tests/compact_joint_cache.py' in metadata['context']['source_sha256'])
    loaded = cache.load_for_score(raw)
    check('score_load_has_all_original_canonical_and_latent_bytes_without_inverse',
          all(c.array_sha(loaded[k]) == c.array_sha(out[k]) for k in c.CANONICAL_KEYS+('continuum_latent',))
          and p.inverse_calls == 0)
    original_parameter_sha = s.sha(cache.data/'PARAMETERS.npz')
    try:
        raise RuntimeError('Injected external evaluator failure after loading cache')
    except RuntimeError:
        pass
    cache.load_for_score(raw)
    score = cache.record_score('fixed_domain', {'status': 'finite_uncertified', 'template_SSE': 4.}, raw)
    score_sha = s.sha(cache.report/'SCORE_fixed_domain.json')
    repeated = cache.record_score('fixed_domain', {'ignored_duplicate': True}, raw)
    check('external_score_retry_and_already_completed_persist_never_refit_or_overwrite',
          score == repeated and s.sha(cache.report/'SCORE_fixed_domain.json') == score_sha
          and s.sha(cache.data/'PARAMETERS.npz') == original_parameter_sha
          and cache.persist(ticket, out, raw)['status'] == 'LOSSLESS_READY'
          and not cache.begin(raw)['run_new_inverse'] and p.inverse_calls == 0)

    failed, pf, bad_out = new_case('initial_mismatch')
    bad_out['predicted_raw_e'] = bad_out['predicted_raw_e'].clone()
    bad_out['predicted_raw_e'][0, 0, 0] = torch.nextafter(bad_out['predicted_raw_e'][0, 0, 0], torch.tensor(float('inf')))
    failed_ticket = failed.begin(raw)
    failure = failed.persist(failed_ticket, bad_out, raw)
    fallback = failed.load_fallback()
    check('initial_single_bit_mismatch_preserves_every_original_canonical_byte_and_fails_closed',
          failure['status'] == 'FAIL_CLOSED_INITIAL_BYTE_MISMATCH'
          and failure['original_fallback_available'] and not failure['score_allowed']
          and not failure['lossless_representation_verified']
          and set(fallback) == set(c.CANONICAL_KEYS)
          and all(c.array_sha(fallback[k]) == c.array_sha(bad_out[k]) for k in c.CANONICAL_KEYS)
          and not Path(failed._read('RESERVATION.json')['reservation']['path']).exists())
    calls = pf.op.forward_calls
    failed.resume(raw)
    failed.persist(failed_ticket, bad_out, raw)
    check('failed_cache_cannot_issue_another_ticket_reconstruct_again_or_be_scored',
          not failed.begin(raw)['run_new_inverse'] and rejects(lambda: failed.load_for_score(raw))
          and pf.op.forward_calls == calls and pf.inverse_calls == 0)

    exception_cache, pe, exception_out = new_case('initial_forward_exception')
    exception_ticket = exception_cache.begin(raw)
    pe.fail_continuum = True
    exception_result = exception_cache.persist(exception_ticket, exception_out, raw)
    check('initial_forward_exception_still_retains_full_original_fallback_and_parameters',
          exception_result['status'] == 'FAIL_CLOSED_INITIAL_PERSISTENCE_OR_FORWARD'
          and exception_result['original_fallback_available']
          and (exception_cache.data/'PARAMETERS.npz').exists()
          and all(c.array_sha(exception_cache.load_fallback()[k]) == c.array_sha(exception_out[k]) for k in c.CANONICAL_KEYS))
    bad_parameter, pb, bad_parameter_out = new_case('invalid_parameter_precision')
    bad_parameter_out['z_grid'] = bad_parameter_out['z_grid'].float()
    bad_parameter_result = bad_parameter.persist(bad_parameter.begin(raw), bad_parameter_out, raw)
    preserved_bad_parameters = s.arrays(bad_parameter.data/'PARAMETERS.npz')
    check('invalid_parameter_contract_still_preserves_original_precision_and_full_canonical_fallback',
          bad_parameter_result['status'] == 'FAIL_CLOSED_INITIAL_PERSISTENCE_OR_FORWARD'
          and bad_parameter_result['original_fallback_available']
          and preserved_bad_parameters['z_grid'].dtype == np.float32
          and c.array_sha(preserved_bad_parameters['z_grid']) == c.array_sha(bad_parameter_out['z_grid'])
          and all(c.array_sha(bad_parameter.load_fallback()[k]) == c.array_sha(bad_parameter_out[k]) for k in c.CANONICAL_KEYS))

    later, pl, later_out = new_case('later_mismatch')
    later.persist(later.begin(raw), later_out, raw)
    saved_sha = s.sha(later.data/'PARAMETERS.npz')
    pl.op.offset = 1.
    check('later_byte_mismatch_cannot_fabricate_an_original_fallback_or_allow_scoring',
          rejects(lambda: later.load_for_score(raw))
          and later.inspect()['status'] == 'FAIL_CLOSED_LATER_BYTE_MISMATCH_NO_ORIGINAL_FALLBACK'
          and not later.inspect()['original_fallback_available']
          and later.inspect()['cannot_fabricate_or_approximate_fallback']
          and not (later.data/'FALLBACK.npz').exists() and s.sha(later.data/'PARAMETERS.npz') == saved_sha)
    calls = pl.op.forward_calls
    check('already_failed_later_cache_never_reconstructs_again_or_restarts_an_inverse',
          not later.begin(raw)['run_new_inverse'] and not later.resume(raw)['score_allowed']
          and pl.op.forward_calls == calls and pl.inverse_calls == 0)
    late_error, ple, late_error_out = new_case('later_scoring_forward_exception')
    late_error.persist(late_error.begin(raw), late_error_out, raw)
    one_good_forward = late_error._verify_forward(raw)
    with mock.patch.object(late_error, '_verify_forward', side_effect=[one_good_forward, ArithmeticError('late forward error')]):
        denied = rejects(lambda: late_error.load_for_score(raw))
    check('a_second_forward_error_during_score_loading_is_permanently_fail_closed',
          denied and late_error.inspect()['status'] == 'FAIL_CLOSED_LATER_FORWARD_ERROR_NO_ORIGINAL_FALLBACK'
          and not late_error.inspect()['original_fallback_available'] and ple.inverse_calls == 0)

    recovered, pr, recovered_out = new_case('parameters_then_crash')
    recovered_ticket = recovered.begin(raw)
    with mock.patch.object(recovered, '_verify_forward', side_effect=KeyboardInterrupt('simulated process exit')):
        try:
            recovered.persist(recovered_ticket, recovered_out, raw)
        except KeyboardInterrupt:
            pass
    recovered_result = recovered.resume(raw)
    check('interruption_after_parameter_persistence_recovers_forward_only_without_new_ticket',
          recovered_result['status'] == 'LOSSLESS_READY' and recovered_result['recovered_after_interruption']
          and all(c.array_sha(recovered.load_for_score(raw)[k]) == c.array_sha(recovered_out[k]) for k in c.CANONICAL_KEYS)
          and not recovered.begin(raw)['run_new_inverse'] and pr.inverse_calls == 0)

    no_receipt, pn, no_receipt_out = new_case('npz_before_receipt_crash')
    no_receipt_ticket = no_receipt.begin(raw)
    write_once = c._write_once
    def interrupted_receipt(path, value):
        if Path(path).name == 'PARAMETERS.json':
            raise KeyboardInterrupt('simulated exit before parameter receipt')
        return write_once(path, value)
    with mock.patch.object(c, '_write_once', side_effect=interrupted_receipt):
        try:
            no_receipt.persist(no_receipt_ticket, no_receipt_out, raw)
        except KeyboardInterrupt:
            pass
    check('sealed_original_hashes_recover_complete_parameter_npz_before_its_receipt',
          no_receipt.resume(raw)['status'] == 'LOSSLESS_READY'
          and no_receipt._read('PARAMETERS.json')['recovered_receipt'] and pn.inverse_calls == 0)

    pending, pp, pending_out = new_case('complete_pending_archive_crash')
    pending_ticket = pending.begin(raw)
    original_replace = Path.replace
    def interrupted_replace(path, target):
        if path.name.startswith('PARAMETERS.npz.pending.'):
            raise KeyboardInterrupt('simulated exit after ZIP write before atomic rename')
        return original_replace(path, target)
    with mock.patch.object(Path, 'replace', interrupted_replace):
        try:
            pending.persist(pending_ticket, pending_out, raw)
        except KeyboardInterrupt:
            pass
    check('complete_pending_parameter_archive_recovers_only_after_original_byte_hash_checks',
          pending.resume(raw)['status'] == 'LOSSLESS_READY'
          and (pending.data/'PARAMETERS.npz').exists() and not list(pending.data.glob('PARAMETERS.npz.pending.*'))
          and pp.inverse_calls == 0)

    empty, pi, empty_out = new_case('interrupted_without_parameters')
    empty.begin(raw)
    empty_result = empty.resume(raw)
    check('started_without_parameters_fails_closed_and_is_never_recomputed',
          empty_result['status'] == 'FAIL_CLOSED_RECOVERY_WITHOUT_ORIGINAL_CANONICAL'
          and not empty_result['original_fallback_available'] and not empty.begin(raw)['run_new_inverse']
          and empty.resume(raw) == empty_result and pi.op.forward_calls == 0 and pi.inverse_calls == 0)

    corrupt, pc, corrupt_out = new_case('corrupt_parameters')
    corrupt.persist(corrupt.begin(raw), corrupt_out, raw)
    with (corrupt.data/'PARAMETERS.npz').open('ab') as handle:
        handle.write(b'fixture corruption')
    check('corrupt_parameter_file_is_rejected_without_forward_or_inverse_retry',
          rejects(lambda: corrupt.load_for_score(raw)) and not corrupt.begin(raw)['run_new_inverse']
          and pc.inverse_calls == 0)
    changed, pd, changed_out = new_case('changed_physics')
    changed.persist(changed.begin(raw), changed_out, raw)
    pd.M[0, 0] += 1.
    check('different_physics_buffers_and_raw_bytes_cannot_reuse_a_cached_endpoint',
          rejects(lambda: changed.load_for_score(raw))
          and rejects(lambda: cache.load_for_score(raw+.5)))
    dependency = root/'fixture_dependency.txt'
    dependency.write_text('original', encoding='utf8')
    dependency_name = dependency.relative_to(s.ROOT).as_posix()
    source_case, ps, source_out = new_case('changed_source', {dependency_name: s.sha(dependency)})
    source_case.persist(source_case.begin(raw), source_out, raw)
    dependency.write_text('changed', encoding='utf8')
    check('changed_dependency_hash_prevents_loading_scoring_or_new_inverse_ticket',
          rejects(lambda: source_case.load_for_score(raw)) and rejects(lambda: source_case.begin(raw))
          and ps.inverse_calls == 0)
    initial_dependency = root/'initial_dependency.txt'
    initial_dependency.write_text('before inverse', encoding='utf8')
    initial_dependency_name = initial_dependency.relative_to(s.ROOT).as_posix()
    initial_source, pis, initial_source_out = new_case('source_changed_before_persist',
        {initial_dependency_name: s.sha(initial_dependency)})
    initial_source_ticket = initial_source.begin(raw)
    initial_dependency.write_text('changed during external inverse', encoding='utf8')
    initial_source_result = initial_source.persist(initial_source_ticket, initial_source_out, raw)
    check('initial_dependency_failure_preserves_original_parameters_info_and_full_output',
          initial_source_result['status'] == 'FAIL_CLOSED_INITIAL_PERSISTENCE_OR_FORWARD'
          and initial_source_result['original_fallback_available']
          and initial_source._read('PREPARED.json')['original_info']['method'] == initial_source_out['info']['method']
          and (initial_source.data/'PARAMETERS.npz').exists()
          and all(c.array_sha(initial_source.load_fallback()[k]) == c.array_sha(initial_source_out[k]) for k in c.CANONICAL_KEYS))

    ledger_root = root/'ledger_fixture'
    ledger_root.mkdir()
    lengths = {'PARAMETERS.npz': 11, 'FALLBACK.npz': 13, 'FALLBACK_RESERVED_owner.bin': 17,
               'PARAMETERS.npz.pending.owner': 19, 'metadata.json': 23, 'run.log': 29, 'cache.bin': 31}
    for name, count in lengths.items():
        (ledger_root/name).write_bytes(bytes(count))
    ledger = c.tree_ledger((ledger_root, ledger_root))
    check('ledger_counts_all_data_meta_fallback_pending_reserve_cache_and_logs_once',
          ledger['bytes'] == sum(lengths.values()) and ledger['files'] == len(lengths)
          and ledger['by_kind_bytes']['parameters'] == 11 and ledger['by_kind_bytes']['fallback'] == 13
          and ledger['by_kind_bytes']['reserve'] == 17 and ledger['by_kind_bytes']['pending'] == 19)
    budget = c.Budget((ledger_root,), global_roots=(ledger_root,),
                      free_bytes=c.FREE_FLOOR_BYTES+100)
    synthetic = {'instance': {'bytes': 0}, 'global_new_study': {'bytes': c.GLOBAL_LIMIT_BYTES-100}}
    with mock.patch.object(budget, 'ledger', return_value=synthetic):
        equality = budget.check(100)
        over_global = rejects(lambda: budget.check(101))
    floor_budget = c.Budget((ledger_root,), global_roots=(ledger_root,), free_bytes=c.FREE_FLOOR_BYTES+99)
    instance_budget = c.Budget((ledger_root,), global_roots=(ledger_root,),
        instance_limit_bytes=sum(lengths.values())+99, free_bytes=c.FREE_FLOOR_BYTES+1000)
    check('four_GiB_and_two_GiB_equalities_pass_but_one_byte_overrun_fails_before_begin',
          equality['free_after_reserve_bytes'] == c.FREE_FLOOR_BYTES and over_global
          and rejects(lambda: floor_budget.check(100)) and rejects(lambda: instance_budget.check(100)),
          global_limit_bytes=c.GLOBAL_LIMIT_BYTES, free_floor_bytes=c.FREE_FLOOR_BYTES)
    formal_budget = c.Budget((root,))
    check('formal_global_scope_is_all_new_study_DATA_and_REPORTS',
          formal_budget.global_roots == (s.DATA.resolve(), s.REPORTS.resolve())
          and formal_budget.global_limit == 4*2**30 and formal_budget.floor == 2*2**30)
    permission_p = FixturePhysics()
    check('new_D_root_is_explicitly_unsupported_by_unchanged_storage_whitelist',
          c.format_contract()['new_D_root_writes_supported'] is False
          and rejects(lambda: c.CompactJointCache(root/'D_permission_report',
              Path('D:/CodexWork/thermometry/closeout_001/outputs/repair_001'), p=permission_p,
              identity={'registration_sha256': 'fixture', 'method': 'fixture', 'case_id': 'D-root-denied'},
              input_reference=raw_ref, calibration={'fingerprint': 'a'*64, 'physical_objective_version': 'fixture'},
              config={'grid': 2}), ValueError))

    tree = ast.parse((s.ROOT/'research_closeout/compact_joint_cache.py').read_text(encoding='utf8'))
    prohibited = {'minimize', 'lsmr', 'repaired_infer', 'refined_infer', 'load_diagnostic_truth', 'invert_single', 'initialize'}
    invoked = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
               for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    check('cache_has_no_inverse_optimizer_linear_solver_or_truth_loader_calls',
          not (invoked & prohibited) and all(physics.inverse_calls == 0 for physics in all_physics),
          complete_inversions=0, prohibited_calls_found=sorted(invoked & prohibited))
    check('v001_v002_registrations_and_results_remain_byte_identical',
          all(s.sha(path) == digest for path, digest in baseline_hashes.items()))
    from ..smooth_calibration import physics as native_physics
    from ..observable_basis import RealFourierBasis
    native = native_physics('cpu')
    native_plan = c.frame_plan(shape=native.cfg['shape'], bands=native.M.shape[0],
                              template_dimension=RealFourierBasis(native.op.object_otf).dimension, grid=256)
    check('native_conservative_bounds_include_dense_full_fallback_and_all_float64_parameters',
          native_plan['parameter_payload_bytes'] == 1057824
          and native_plan['canonical_payload_bytes'] == 6422528
          and native_plan['initial_safe_reserve_bytes'] > native_plan['fallback_zip_bound_bytes']
          and native_plan['successful_compact_bound_bytes'] < 2**21)
    result = {'time': s.now(), 'all_passed': all(item['passed'] for item in checks), 'checks': checks,
              'source_sha256': {name: s.sha(s.ROOT/name) for name in
                               ('research_closeout/compact_joint_cache.py', 'research_closeout/tests/compact_joint_cache.py')},
              'schema': c.SCHEMA, 'native_frame_conservative_plan': native_plan,
              'format_contract': c.format_contract(),
              'fixture_shape': [2, 2], 'formal_complete_inversions': 0, 'GPU_used': False,
              'native_frame_byte_exact_validation_performed': False,
              'native_GPU_byte_exact_qualification': False, 'truth_loaded': False,
              'v001_v002_data_converted_or_deleted': False, 'fixture_directory': str(root),
              'initial_mismatch_full_fallback_required': True,
              'later_mismatch_without_original_fallback_is_explicit_fail_closed': True}
    destination = c.REPORT_DIR/'COMPACT_CACHE_CONTRACTS.json'
    if destination.exists() or not result['all_passed']:
        state = 'RECHECK' if result['all_passed'] else 'INITIAL'
        destination = c.REPORT_DIR/f'COMPACT_CACHE_CONTRACTS_{state}_{token}.json'
    s.write(destination, result)
    print(json.dumps({**result, 'receipt': str(destination)}, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(0 if run()['all_passed'] else 2)
