"""Bounded CPU controls for matched fresh families and sealed partitions."""
from __future__ import annotations

import copy
import json
import time
import unittest
from uuid import uuid4
from . import common

common.configure_runtime()


class FamilyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from config import default_config
        from physics import SpectralPhysics
        from .family_data import generate_family
        from .planning import family_plan, new_mothers
        torch.set_num_threads(2)
        cls.np, cls.torch = np, torch
        cls.physics = SpectralPhysics(default_config(24), dtype=torch.float64)
        cls.folder = common.DATA / '_implementation_checks' / ('family_' + uuid4().hex)
        cls.mothers = new_mothers('development_selection', 1, 8100000)
        cls.plan = family_plan(cls.mothers, kinds=('continuous', 'quantization', 'noise'))
        cls.manifest = generate_family(cls.physics, cls.plan, cls.folder / 'matched', {'fixture_only': True})

    def test_01_fields_protected_and_truth_separate(self):
        manifest = self.manifest
        self.assertEqual(manifest['planned_rows'], 15)
        self.assertEqual(manifest['generated_raw_rows'], 15)
        self.assertTrue(manifest['matched_T_loga_beta_shape_verified'])
        self.assertFalse(manifest['mother_fields_modified_for_legality'])
        self.assertEqual(len({json.dumps(x['protected_field_hashes'], sort_keys=True)
                              for x in manifest['labels']}), 1)
        for row in manifest['rows']:
            raw_path = self.folder / 'matched' / row['raw_path']
            self.assertEqual(set(common.arrays(raw_path)), {'raw_dn'})
            self.assertEqual(common.sha(raw_path), row['raw_file_sha256'])
            label_path = self.folder / 'matched' / row['label_path']
            self.assertEqual(common.sha(label_path), row['label_sha256'])
            label = common.arrays(label_path)
            self.assertTrue(self.np.array_equal(label['template_e'], label['template_postblur_e']))
            self.assertTrue(self.np.array_equal(label['temperature_color_K'], label['temperature_color_K_nominal']))
            self.assertTrue(self.np.array_equal(label['continuum_latent'], label['continuum_latent_nominal']))

    def test_02_same_mean_seed_is_reproducible_not_same_noise_across_means(self):
        from .family_data import generate_family
        other = generate_family(self.physics, self.plan, self.folder / 'replay', {'fixture_only': True})
        self.assertEqual([r['noise_seed'] for r in other['rows']], [r['noise_seed'] for r in self.manifest['rows']])
        self.assertEqual([r['raw_hash'] for r in other['rows']], [r['raw_hash'] for r in self.manifest['rows']])
        seeds = [r['noise_seed'] for r in other['rows'] if r['input_kind'] == 'noise']
        self.assertEqual(len(set(seeds)), 5)
        self.assertFalse(other['same_noise_realization_claimed'])
        with self.assertRaises(FileExistsError):
            generate_family(self.physics, self.plan, self.folder / 'replay', {'fixture_only': True})

    def test_03_gain_and_emissivity_stress_keep_nominal_and_reference_separate(self):
        from .family_data import generate_family
        from .planning import family_plan
        for kind in ('channel_gain', 'continuum_emissivity'):
            spec = {'name': 'control_' + kind, 'kind': kind,
                    'band_multipliers': [1., 1., 1., 1., 1., .95, 1.05]}
            plan = family_plan(self.mothers, coefficients=(0.,), kinds=('continuous',),
                               role='calibration_stress', stress=spec)
            folder = self.folder / kind
            manifest = generate_family(self.physics, plan, folder, {'fixture_only': True})
            label = common.arrays(folder / manifest['labels'][0]['path'])
            self.assertGreater(float(self.np.max(self.np.abs(label['temperature_color_K_reference']
                                                           - label['temperature_color_K_nominal']))), 1.)
            self.assertTrue(self.np.array_equal(label['temperature_color_K'], label['temperature_color_K_reference']))
            nominal_matches_physical = self.np.array_equal(label['continuum_latent'], label['continuum_latent_nominal'])
            self.assertEqual(nominal_matches_physical, kind == 'channel_gain')
            self.assertFalse(self.np.array_equal(label['continuum_latent_reference'], label['continuum_latent_nominal']))

    def test_04_sealed_partitions_refused_before_output_creation(self):
        from .family_data import generate_family
        for role in ('signed_confirmation', 'final_once'):
            descriptor = copy.deepcopy(self.plan[0])
            descriptor['partition_role'] = role
            folder = self.folder / ('forbidden_' + role)
            with self.assertRaisesRegex(ValueError, 'sealed'):
                generate_family(self.physics, [descriptor], folder, {'fixture_only': True})
            self.assertFalse(folder.exists())

    def test_05_original_final_rules_metadata_preserved_without_generation(self):
        from .planning import build_family_plans
        families = build_family_plans()
        final = families['plans']['final_once']
        self.assertEqual(len(final), 512)
        self.assertEqual(len(families['plans']['signed_confirmation']), 288)
        self.assertTrue(families['no_arrays_opened'])
        self.assertFalse(families['confirmation_and_final_arrays_generated'])
        for row in final:
            self.assertEqual(row['coefficient_rule'], 'original_v4_final_levels')
            self.assertEqual(row['noise_seed'], row['mother']['raw_noise_seed_base']
                             + row['original_condition_index'] * 17 + row['noise_index'])
            if row['condition'] != 'blank':
                self.assertIsNone(row['coefficient_e'])

    def test_06_original_rng_rule_with_unregistered_synthetic_unit_mother(self):
        from .family_data import _actual_coefficient
        # This fake unit descriptor is not one of the 64 sealed final mothers.
        seed = 9100001
        rng = self.np.random.default_rng(seed + 301)
        expected = [0., float(rng.uniform(25, 55)), float(rng.uniform(140, 280)), float(rng.uniform(320, 520))]
        cache = {}
        for i, bounds in enumerate(((0., 0.), (25., 55.), (140., 280.), (320., 520.))):
            row = {'mother_id': 'unit_only_original_rule', 'mother': {'template_seed': seed},
                   'coefficient_rule': 'original_v4_final_levels', 'original_condition_index': i,
                   'coefficient_range_e': list(bounds)}
            self.assertEqual(_actual_coefficient(row, cache), expected[i])

    def test_07_conflicting_mother_identity_rejected(self):
        from .family_data import generate_family
        plan = copy.deepcopy(self.plan[:2])
        plan[1]['mother']['temperature_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            generate_family(self.physics, plan, self.folder / 'bad_fields', {'fixture_only': True})


def main():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FamilyContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'schema': 'v5-family-cpu-controls-v1', 'all_passed': result.wasSuccessful(),
               'checks': result.testsRun, 'failures': [[str(case), trace] for case, trace in result.failures],
               'errors': [[str(case), trace] for case, trace in result.errors],
               'fixture_folder': common.relative(FamilyContracts.folder),
               'formal_scientific_acceptance': False, 'sealed_arrays_generated': False, 'device': 'cpu',
               'wall_s': time.perf_counter() - started,
               'source_sha256': {common.relative(common.CODE / name): common.sha(common.CODE / name)
                           for name in ('family_data.py', 'planning.py', 'tests_family.py')}}
    target = common.REPORTS.parent / 'implementation_checks' / 'family' / ('cpu_' + uuid4().hex + '.json')
    common.write_json(target, receipt)
    print(str(target), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
