"""Small CPU contracts for the independent four-inverse C1v3 controller."""
from .. import storage as s
from .. import quick_validation as q
from .. import calibration_quick as c
import copy
import json
from pathlib import Path
import uuid
import numpy as np


def run():
    # The adapter's own numerical checks must already have fixed its API and
    # scientific interpolation-error contract. No inverse is called here.
    from ..smooth_calibration import physics, contract
    import torch
    torch.set_num_threads(2)
    p = physics('cpu')
    adapter = s.plain(contract(p))
    token = uuid.uuid4().hex
    report, data = c.REPORT_DIR/'_cpu_fixtures'/token, c.DATA_DIR/'_cpu_fixtures'/token
    report.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    def rejects(action, exception=(q.QuickIntegrityError, ValueError)):
        try:
            action()
        except exception:
            return True
        return False

    old_registration, old_result, old_files = c._verify_v001()
    old_hashes = {receipt['path']: receipt['sha256'] for receipt in old_files}
    refs = copy.deepcopy(old_registration['references'][:2])
    planned = c.planned_cells(refs)
    check('new_matrix_is_exactly_two_old_raws_times_two_initializers',
          len(planned) == 4 and {cell['initializer'] for cell in planned} == {'v3', 'smooth_demod'}
          and all('C1v3' in cell['method'] and cell['family'] == 'joint_C1v3' for cell in planned)
          and c.CONFIG['planned_new_inverse_calls'] == 4 and c.CONFIG['inverse_wall_budget_s'] == 720
          and c.CONFIG['numerical_options']['cost_limit_s'] == 180
          and c.CONFIG['physical_options'] == {'grid': 256, 'continuum_multiplier': 1., 'template_multiplier': 1.}
          and not c.CONFIG['v3_refined_rerun'])
    check('no_additional_raw_or_grid_can_enter',
          rejects(lambda: c.planned_cells(old_registration['references']))
          and rejects(lambda: c.planned_cells(list(reversed(refs)))))
    check('v001_all_six_finite_and_uncertified_control_cells_are_preserved',
          old_result['summary']['planned_cells'] == 6
          and len(old_result['summary']['cells']) == 6
          and old_result['summary']['finite_output_cells'] == 6
          and old_result['summary']['numerically_certified_cells'] == 0,
          result_reference=q._file_ref(c.V001_RESULT))
    identity = c.adapter_identity(adapter)
    check('actual_cpu_adapter_has_explicit_new_C1v3_identity_and_original_buffers',
          identity['calibration_version'] == c.ADAPTER_VERSION
          and identity['physical_objective_version'] == c.OBJECTIVE_VERSION
          and identity['original_buffers_byte_identical'] and identity['common_color_lookup_unchanged'],
          calibration_version=identity['calibration_version'], fingerprint=identity['fingerprint'])
    check('adapter_mismatch_is_rejected_before_registration',
          rejects(lambda: c.adapter_identity({**adapter, 'calibration_version': 'unregistered-calibration'}))
          and rejects(lambda: c.adapter_identity({**adapter, 'original_buffers_byte_identical': False})))
    source_fixture = {'fixture_source.py': '1'*64}
    fingerprint = c.objective_fingerprint(adapter, source_fixture)
    check('objective_fingerprint_binds_C1v3_adapter_and_source_bytes',
          fingerprint != c.objective_fingerprint({**adapter, 'fingerprint': '2'*64}, source_fixture)
          and fingerprint != c.objective_fingerprint(adapter, {'fixture_source.py': '3'*64}))

    lock_a, lock_b = report/'global_lock_fixture.json', report/'local_lock_fixture.json'
    with c.phase_locks('fixture-digest', paths=(lock_a, lock_b)):
        value = s.read(lock_a)
        check('new_phase_lock_identifies_C1v3_real_pid_and_is_exclusive',
              value['phase'] == c.VERSION and value['pid'] == c.os.getpid()
              and rejects(lambda: _enter_lock(lock_a), FileExistsError))
    check('own_global_and_local_fixture_locks_are_released', not lock_a.exists() and not lock_b.exists())
    foreign = {'pid': -1, 'owner_token': 'foreign'}
    with c.phase_locks('fixture-digest', paths=(lock_a,)):
        s.write(lock_a, foreign, replace=True)
    check('finally_preserves_replaced_foreign_lock', s.read(lock_a) == foreign)

    fixture_registration = {'calibration_contract': adapter, 'source_sha256': source_fixture,
                            'config_fingerprint': 'fixture-config', 'objective_fingerprint': fingerprint}
    field = np.arange(4, dtype=np.float64).reshape(1, 2, 2)+.12345678912345678
    core = {'method': 'joint_profile_repaired_v2', 'status': 'NUMERICAL_UNCERTIFIED_V2',
            'physical_objective_version': 'original_v2_label_preserved', 'truth_used': False,
            'physical_objective': {'version': 'original_v2_label_preserved',
                                  'continuum_regscale': [1.25, 2.5], 'regularizer_fingerprint': 'original-regularizer'},
            'returned_gradient': {'scaled_projected_linf': 5e-5}, 'output_comparisons': []}
    original = {'temperature_color_K': field+3000., 'template_postblur_e_ref': field.copy(),
                'continuum_postblur_bands': np.repeat(field[:, None], 3, axis=1),
                'predicted_raw_e': field*20., 'valid': np.ones((1, 2, 2), dtype=bool),
                'low_signal': np.zeros((1, 2, 2), dtype=bool), 'numerical_valid': False,
                'z_grid': np.stack((field, field*.1), axis=1), 'xi': np.arange(5, dtype=np.float64)+.25,
                'beta': np.array([[.1, .2, .3]], dtype=np.float64), 'continuum_latent': np.repeat(field[:, None], 3, axis=1),
                'info': core}
    cell, digest = planned[0], 'C1v3-controller-fixture-digest'
    wrapped = c.wrap_output(original, cell, fixture_registration)
    check('outer_C1v3_version_and_anchor_regscale_preserve_core_v2_declaration',
          wrapped['info']['physical_objective_version'] == c.OBJECTIVE_VERSION
          and wrapped['info']['method'] == cell['method']
          and wrapped['info']['core_reported_physical_objective_version'] == 'original_v2_label_preserved'
          and wrapped['info']['numerical_engine'] == core
          and wrapped['info']['anchor_regscale_contract']['continuum_regscale'] == [1.25, 2.5]
          and core['physical_objective_version'] == 'original_v2_label_preserved'
          and wrapped['info']['C1v3_adapter_fingerprint'] == adapter['fingerprint'])
    parameter_identity = c.identity_for(cell, digest)
    receipt = c.save_parameters(wrapped, parameter_identity, fixture_registration,
                                report_dir=report/'roundtrip', data_dir=data/'roundtrip', shape=(2, 2))
    arrays, metadata = c.load_parameters(receipt, parameter_identity, fixture_registration)
    check('exact_float64_z_xi_beta_saved_without_repeated_seven_band_continuum',
          set(arrays) == {'z_grid', 'xi', 'beta'}
          and all(arrays[name].dtype == np.float64 and np.array_equal(arrays[name], original[name]) for name in arrays)
          and metadata['continuum_latent_stored'] is False
          and 'no inverse' in metadata['continuum_latent_reconstruction']
          and metadata['source_sha256'] == source_fixture
          and metadata['calibration_contract']['fingerprint'] == adapter['fingerprint'])
    check('parameter_source_config_adapter_mismatches_are_rejected',
          rejects(lambda: c.load_parameters(receipt, parameter_identity, {**fixture_registration, 'config_fingerprint': 'different'}))
          and rejects(lambda: c.load_parameters(receipt, {**parameter_identity, 'method': 'wrong-method'}, fixture_registration)))
    ledger = c.storage_ledger(data/'roundtrip', report/'roundtrip')
    check('storage_ledger_includes_parameter_npz_and_all_metadata',
          ledger['data_npz_bytes'] == Path(receipt['parameter_cache']['path']).stat().st_size
          and ledger['total_new_bytes'] > ledger['data_npz_bytes']
          and ledger['parameter_and_canonical_npz_both_counted'])

    callbacks, scoring, guards = [], [], []
    reserve = c.reserve_bytes(shape=(2, 2), bands=3, template_dimension=5)
    fake_free = c.CONFIG['free_space_floor_bytes']+2**24
    def guard():
        guards.append(c.ensure_headroom(reserve, data_dir=data/'cache', report_dir=report/'cache',
            free_bytes=fake_free, floor_guard=lambda: None))
    def action():
        callbacks.append('inverse_callback')
        out = c.wrap_output(copy.deepcopy(original), cell, fixture_registration)
        saved = c.save_parameters(out, parameter_identity, fixture_registration,
                                  report_dir=report/'cache', data_dir=data/'cache', shape=(2, 2))
        out['info']['parameter_cache_receipt'] = saved
        return out
    def score_action(out):
        scoring.append(True)
        parameter = out['info']['parameter_cache_receipt']
        c.load_parameters(parameter, parameter_identity, fixture_registration)
        assert (data/'cache'/(parameter_identity['cache_key']+'.npz')).exists()
        assert (report/'cache/cells'/parameter_identity['cache_key']/'INFERENCE.json').exists()
        if len(scoring) == 1:
            raise RuntimeError('Injected external score failure')
        return {'status': 'FINITE_NUMERICALLY_UNCERTIFIED', 'production_numerical_certified': False,
                'metrics': {'domain_pixels': 4, 'template_SSE': 4., 'temperature_SSE_K2': 16.,
                            'frame_detection_v2': {'threshold': None, 'detected': None}}}
    evaluate = {'registration_sha256': digest, 'initializer_provenance': {'kind': 'CPU_raw_only_fixture'},
                'report_dir': report/'cache', 'data_dir': data/'cache', 'action': action, 'score_action': score_action,
                'count_call': lambda *args: callbacks.append(args), 'budget_guard': guard,
                'compact_info': lambda out: s.plain(out['info']), 'output_budget_bytes': 2**20, 'shape': (2, 2)}
    first = q.evaluate_cell(cell, **evaluate)
    second = q.evaluate_cell(cell, **evaluate)
    cache_ledger = c.storage_ledger(data/'cache', report/'cache')
    check('parameters_and_canonical_persist_before_truth_callback_and_score_retry_does_not_refit',
          first['finite_output'] and first['score_status'] == 'SCORING_FAILED'
          and second['score_status'] == 'SCORED' and second['reused_inference']
          and callbacks == [('repair_quick', cell['method'], cell['case_id']), 'inverse_callback']
          and len(guards) == 1 and len(scoring) == 2 and not second['production_numerical_certified'])
    parameter_bytes = second['info']['parameter_cache_receipt']['parameter_bytes']
    canonical_bytes = second['output_bytes']
    check('combined_output_budget_charges_canonical_plus_parameter_npz_exactly',
          cache_ledger['data_npz_bytes'] == parameter_bytes+canonical_bytes,
          parameter_ledger_bytes=cache_ledger['data_parameter_npz_bytes'],
          canonical_ledger_bytes=cache_ledger['data_canonical_npz_bytes'],
          parameter_bytes=parameter_bytes, canonical_bytes=canonical_bytes,
          total_npz_bytes=cache_ledger['data_npz_bytes'])
    check('headroom_checks_output_cap_and_two_GiB_floor_before_inverse',
          rejects(lambda: c.ensure_headroom(reserve, data_dir=data/'cache', report_dir=report/'cache',
              free_bytes=fake_free, floor_guard=lambda: None,
              budget_bytes=cache_ledger['data_npz_bytes']+reserve['one_inverse_reserve_bytes']-1))
          and rejects(lambda: c.ensure_headroom(reserve, data_dir=data/'cache', report_dir=report/'cache',
              free_bytes=c.CONFIG['free_space_floor_bytes']+reserve['one_inverse_reserve_bytes']-1,
              floor_guard=lambda: None)))
    check('new_failed_or_unattempted_cells_keep_four_cell_denominator',
          q.summarize_cells(planned, [second])['planned_cells'] == 4
          and len(q.summarize_cells(planned, [second])['cells']) == 4
          and q.summarize_cells(planned, [second])['numerically_certified_cells'] == 0)
    check('new_cache_identity_cannot_reuse_v001_method_or_registration',
          q._cache_key(cell, digest) != q._cache_key({**cell, 'method': q.ROUTES[0]['method']}, digest)
          and q._cache_key(cell, digest) != q._cache_key(cell, old_result['registration_sha256']))
    recover_digest = digest+'-parameter-recovery'
    recover_identity = c.identity_for(cell, recover_digest)
    recover_path, recover_metadata_path, recover_receipt_path = c._parameter_paths(
        recover_identity, report/'recovery', data/'recovery')
    recover_arrays, recover_metadata = c.load_parameters(second['info']['parameter_cache_receipt'],
                                                         parameter_identity, fixture_registration)
    s.save(recover_path, **recover_arrays)
    s.write(recover_metadata_path, {**recover_metadata, **recover_identity})
    recovered_parameter = c.parameter_receipt(cell, recover_digest, fixture_registration,
        report_dir=report/'recovery', data_dir=data/'recovery')
    check('parameter_npz_and_metadata_recover_an_interrupted_receipt_without_refit',
          recovered_parameter['recovered_parameter_receipt'] and recover_receipt_path.exists())
    c.restore_canonical(cell, recover_digest, fixture_registration, None, None,
        report_dir=report/'recovery', data_dir=data/'recovery', shape=(2, 2),
        reconstruct=lambda arrays, metadata: copy.deepcopy(original))
    before = len(callbacks)
    recovered = q.evaluate_cell(cell, **{**evaluate, 'registration_sha256': recover_digest,
        'report_dir': report/'recovery', 'data_dir': data/'recovery',
        'score_action': lambda out: {'status': 'FINITE_NUMERICALLY_UNCERTIFIED',
                                     'production_numerical_certified': False, 'metrics': None}})
    check('exact_parameters_reconstruct_canonical_output_with_zero_new_inverse_calls',
          recovered['recovered_after_inverse_receipt_interruption']
          and recovered['finite_output'] and recovered['new_inverse_calls_this_run'] == 0
          and len(callbacks) == before)
    broken_path = Path(receipt['parameter_cache']['path'])
    with broken_path.open('ab') as handle:
        handle.write(b'fixture-corruption')
    check('modified_parameter_cache_is_quarantined_without_inverse_callback',
          rejects(lambda: c.load_parameters(receipt, parameter_identity, fixture_registration))
          and len(callbacks) == 2)
    check('all_v001_registered_control_files_unchanged_after_controller_fixture',
          all(s.sha(path) == digest for path, digest in old_hashes.items()))
    result = {'time': s.now(), 'all_passed': all(row['passed'] for row in checks), 'checks': checks,
              'source_sha256': {name: s.sha(s.ROOT/name) for name in c.EXTRA_SOURCES[:2]},
              'calibration_contract': adapter, 'fixture_report_directory': str(report),
              'fixture_data_directory': str(data), 'formal_complete_inversions': 0,
              'GPU_used': False, 'new_raw_payloads_loaded': False, 'diagnostic_truth_loaded': False,
              'existing_v001_canonical_outputs_verified': 6, 'v001_denominator_preserved': 6,
              'final_payloads_opened': False, 'fixture_inferences_are_numpy_callbacks': True,
              'scientific_performance_or_numerical_qualification_claimed': False}
    destination = c.REPORT_DIR/(c.CONTROLLER_PREFIX+'.json')
    if destination.exists() or not result['all_passed']:
        suffix = 'RECHECK' if result['all_passed'] else 'INITIAL'
        destination = c.REPORT_DIR/f'{c.CONTROLLER_PREFIX}_{suffix}_{token}.json'
    s.write(destination, result)
    print(json.dumps({**result, 'receipt': str(destination)}, ensure_ascii=True, indent=2), flush=True)
    return result


def _enter_lock(path):
    with c.phase_locks('fixture-other-owner', paths=(path,)):
        pass


if __name__ == '__main__':
    raise SystemExit(0 if run()['all_passed'] else 2)
