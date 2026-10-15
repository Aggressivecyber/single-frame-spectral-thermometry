"""Metadata-only repairs to the next independent/uncertainty protocol.

This does not generate arrays, select a method, or open final payloads. A fresh
source/threshold registration is required before these scientific runs.
"""
from . import storage as s
from .scientific_plan import descriptor, seed, CONDITIONS
from .independent_renderer import MATCHED_RELAY, LEGACY_RELAY_STRESS
from .uncertainty import STATISTIC_NAMES
import collections
import json

FIELD_KEYS = ('temperature_seed', 'amplitude_seed', 'template_seed', 'background_seed')
TEMPERATURE_GEOMETRIES = ('front', 'multipeak', 'ring', 'stripe', 'flat', 'smooth_random')
TEMPLATE_GEOMETRIES = ('blob', 'multipeak', 'ring', 'stripe', 'filament', 'smooth_random')


def build():
    old = s.read(s.REPORTS/'SCIENTIFIC_PLAN.json')
    inventory = s.read(s.ROOT/'research_v6/input_inventory.json')['rows']
    prior = s.read(s.ROOT/'research_v5/family_plans.json')
    prior_mothers = ([d for group in prior['mothers'].values() for d in group]
                     if isinstance(prior['mothers'], dict) else prior['mothers'])
    all_old_mothers = old['mothers']+prior_mothers
    old_ids = {d['mother_id'] for d in all_old_mothers}
    old_ids |= {r['mother_id'] for r in inventory}
    old_ids |= {r['mother_id'] for r in old['final_main_metadata']+old['final_signed_metadata']}
    old_fields = {d[k] for d in all_old_mothers for k in FIELD_KEYS if k in d}
    # Exact old v4 descriptor ranges, including unopened final metadata only.
    for count, base in ((256, 100000), (64, 200000), (64, 300000)):
        old_fields.update(base+i*113+offset for i in range(count) for offset in (1, 29, 53, 71))
    old_raw = {r['noise_seed'] for r in inventory if isinstance(r.get('noise_seed'), int)}
    for r in old['cases']+old['final_main_metadata']+old['final_signed_metadata']:
        if isinstance(r.get('noise_seed'), int):
            old_raw.add(r['noise_seed'])
        old_raw.update(r.get('MC_noise_seeds', []))
    for group in prior['plans'].values():
        old_raw.update(r['noise_seed'] for r in group if isinstance(r.get('noise_seed'), int))
    for count, base in ((256, 100000), (64, 200000), (64, 300000)):
        old_raw.update(base+i*113+10000000+ci*17+j for i in range(count) for ci in range(4) for j in range(2))
    rows = []
    # Every temperature/template/strength stratum has one calibration and one
    # independent evaluation mother. Role no longer determines template type.
    for role_index, role in enumerate(('interval_calibration', 'interval_evaluation')):
        for i in range(6):
            mid = f'closeout_repaired_uq_001_{role}_{i:04d}'
            d = descriptor(mid, 9100000+role_index*10000+i*101, i)
            d['temperature_geometry'] = TEMPERATURE_GEOMETRIES[i]
            d['template_geometry'] = TEMPLATE_GEOMETRIES[i]
            condition, coefficient = CONDITIONS[i % len(CONDITIONS)]
            count = 128 if i == 0 else 64
            rows.append({
                'case_id': mid+'_'+condition, 'mother_id': mid, 'mother': d,
                'family': 'uncertainty', 'engine': 'core', 'role': role,
                'condition': condition, 'coefficient_e': coefficient,
                'photon_ratio': 1., 'normalization': 'fixed_peak',
                'geometry': 'display_only_not_used_for_core_fields', 'perturbation': {},
                'actual_temperature_geometry': d['temperature_geometry'],
                'actual_template_geometry': d['template_geometry'],
                'noise_seed': seed('repair-uq-base-v2', mid), 'MC': count, 'initial_MC': 64,
                'MC_noise_seeds': [seed('repair-uq-mc-v2', mid, j) for j in range(count)],
                'bootstrap_noise_seeds': [seed('repair-uq-bootstrap-v2', mid, j) for j in range(64)],
                'linearization_seed': seed('repair-uq-linearization-v2', mid),
                'linearization_probes': 2, 'linearization_epsilon_grid': [.1, .03, .01],
            })
    ids = [r['mother_id'] for r in rows]
    fields = [r['mother'][k] for r in rows for k in FIELD_KEYS]
    raw_seeds = [r['noise_seed'] for r in rows]+[v for r in rows for v in r['MC_noise_seeds']+r['bootstrap_noise_seeds']]
    checks = {
        'twelve_disjoint_new_mothers': len(ids) == len(set(ids)) == 12 and not set(ids)&old_ids,
        'new_field_streams_unique_and_disjoint': len(fields) == len(set(fields)) and not set(fields)&old_fields,
        'new_measurement_streams_unique_and_disjoint': len(raw_seeds) == len(set(raw_seeds)) and not set(raw_seeds)&old_raw,
        'six_actual_geometries_in_each_role': all(len({r['actual_template_geometry'] for r in rows if r['role'] == role}) == 6 for role in ('interval_calibration', 'interval_evaluation')),
        'role_strata_matched': collections.Counter((r['actual_temperature_geometry'], r['actual_template_geometry'], r['condition']) for r in rows[:6]) == collections.Counter((r['actual_temperature_geometry'], r['actual_template_geometry'], r['condition']) for r in rows[6:]),
        'full_MC_denominator_896': sum(r['MC'] for r in rows) == 896,
        'raw_only_bootstrap_scale_denominator_768': sum(len(r['bootstrap_noise_seeds']) for r in rows) == 768,
        'no_final_payload_opened': not old['sealed_payloads_opened'],
    }
    assert all(checks.values()), checks
    return {
        'schema': 'corrected-scientific-protocol-metadata-v2', 'time': s.now(),
        'requires_fresh_execution_registration': True, 'execution_started': False,
        'enabled_after': 'bounded quick numerical repair validation and explicit frozen method registration',
        'old_science_plan_sha256': s.sha(s.REPORTS/'SCIENTIFIC_PLAN.json'),
        'old_science_results_preserved': True, 'old_relay_batch_interpretation': {
            'old_renderer': LEGACY_RELAY_STRESS, 'role': 'registered relay/calibration-mismatch stress',
            'new_nominal_renderer': MATCHED_RELAY,
            'matched_scope': 'relay transfer only; finite object Airy, object continuation and pixel integration remain independent models',
            'old_nominal_rows_relabelled_in_place': False},
        'uncertainty_cases': rows, 'statistics': list(STATISTIC_NAMES),
        'scale_rule': {
            'schema': 'marginal-statistic-scale-v2',
            'source': '12 output-statistic SDs from64 whole-inverse bootstrap refits of each raw-only fitted mean',
            'fitted_mean_uses_truth': False, 'MC_coverage_generation': 'independently specified scene mean, outside inference',
            'bootstrap_conditional_fit_is_not_independent_coverage': True,
            'calibration_uses': 'first64 independent MC draws of6 calibration mothers only',
            'evaluation_uses': 'first64 independent MC draws of6 disjoint evaluation mothers; no refit or width inflation',
            'base_frame_conditional_scale_shared_within_scene': True,
            'scope': 'case-conditional empirical marginal widths; not a universal deployed scale predictor',
            'failure_policy': 'unavailable scale or failed inverse retains every planned replicate in its denominator',
            'distribution_free_95_guaranteed': False, 'simultaneous_image_coverage_claimed': False,
        },
        'local_linearization': {
            'calls': 144, 'probes': 2, 'epsilon_grid': [.1, .03, .01],
            'same_directions_all_steps': True, 'role': 'bounded diagnostic only',
            'trusted_covariance_if_unstable': False, 'low_probe_interval_scale_eligible': False},
        'prospective_full_inverse_calls': {'base': 12, 'independent_MC': 896, 'conditional_bootstrap': 768, 'local_diagnostic': 144, 'total': 1820},
        'new_method_detection': 'method-specific independent200-null calibration plus200-null evaluation required before freezing a new method threshold; historical v3 thresholds are not transferred',
        'calibration_distribution': 'assumed simulation; measured camera calibration unavailable',
        'final_payloads_opened': False, 'metadata_contracts': checks,
    }


def main():
    value = build()
    s.write(s.REPORTS/'repair_001/REPAIRED_SCIENTIFIC_PROTOCOL.json', value)
    checks = {'time': s.now(), 'all_passed': all(value['metadata_contracts'].values()),
              'checks': value['metadata_contracts'], 'formal_raw_generated': False,
              'new_complete_inversions': 0,
              'source_sha256': {'research_closeout/repair_protocol.py': s.sha(__file__)}}
    s.write(s.REPORTS/'repair_001/REPAIRED_PROTOCOL_CONTRACTS.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
