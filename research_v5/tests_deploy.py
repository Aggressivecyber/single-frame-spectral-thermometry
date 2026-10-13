"""CPU numerical and refusal controls for the uncertainty deployment entry."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import uuid4
import time
from . import common

common.configure_runtime()


class ReliableDeploymentControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from config import default_config
        from physics import SpectralPhysics
        from .deploy import reliable_reconstruct
        torch.set_num_threads(2)
        cls.torch = torch
        cls.p = SpectralPhysics(default_config(24), dtype=torch.float64)
        state = torch.zeros((1, 5, 24, 24), dtype=torch.float64)
        state[:, 0] = (3052.5-cls.p.T0)/cls.p.Tscale
        state[:, 1] = -.5
        state[:, 4] = 45/cls.p.Sscale
        beta = torch.tensor([[.5, .01, -.01]], dtype=torch.float64)
        cls.raw = cls.p(state, beta)/cls.p.adc_step+cls.p.cfg['black_level_DN']
        cls.selection = {'device': 'cpu', 'dtype': 'float64',
                         'continuum_options': {'warm_iterations': 2},
                         'linear_options': {'alpha_g': .01}}
        cls.output = reliable_reconstruct(cls.raw, cls.p.cfg, cls.selection,
                        uncertainty_options={'seed': 510135},
                        detection={'z_threshold': 8., 'threshold_source_sha256': '0'*64})

    def test_01_eight_actual_repeats_rerun_core_and_export_uncertainty(self):
        out = self.output
        summary = out['info']['uncertainty']
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(len(summary['repeats']), 8)
        self.assertEqual(len(set(summary['seed_plan'])), 8)
        self.assertEqual(summary['completed_repeats'], 8)
        for field in ('temperature_uncertainty', 'sigma_S', 'signed_ci_lower_e', 'signed_ci_upper_e',
                      'temperature_ci_lower_K', 'temperature_ci_upper_K', 'undetectable',
                      'low_confidence', 'template_abstain', 'temperature_abstain', 'detection_sigma_proxy_e'):
            self.assertEqual(tuple(out[field].shape), (1, 24, 24))
        self.assertFalse(out['info']['detection']['proxy_is_MC_sigma'])
        self.assertTrue(self.torch.equal(out['abstain'], ~out['valid']))

    def test_02_nan_intervals_and_boolean_masks_reopen_exactly(self):
        tensors = {key: value.cpu().numpy() for key, value in self.output.items() if isinstance(value, self.torch.Tensor)}
        folder = common.DATA / '_implementation_checks' / ('deploy_' + uuid4().hex)
        digest = common.save_npz(folder / 'prediction.npz', **tensors)
        self.assertEqual(digest, common.sha(folder / 'prediction.npz'))
        common.save_npz(folder / 'explicit_unavailable.npz', sigma=self.torch.full((1, 2, 2), float('nan')).numpy(),
                        valid=self.torch.zeros((1, 2, 2), dtype=self.torch.bool).numpy())

    def test_03_truth_and_identity_arguments_are_rejected(self):
        from .deploy import reliable_reconstruct
        for forbidden in ('truth', 'condition', 'case_id', 'clean_identity'):
            with self.assertRaises((ValueError, TypeError)):
                reliable_reconstruct(self.raw, self.p.cfg, {**self.selection, forbidden: 'not_allowed'})
        with self.assertRaises(ValueError):
            reliable_reconstruct(self.raw, {'physics_config': self.p.cfg, 'truth': {}}, self.selection)

    def test_04_mc_failure_abstains_and_preserves_physical_fields(self):
        from .deploy import reliable_reconstruct
        failure = {'status': 'abstained', 'summary': {'status': 'abstained', 'reason': 'unit failed interval control'}}
        with patch('research_v5.deploy.monte_carlo_uncertainty', return_value=failure):
            out = reliable_reconstruct(self.raw, self.p.cfg, self.selection)
        self.assertTrue(bool(out['abstain'].all()))
        self.assertTrue(bool(out['template_abstain'].all()))
        self.assertTrue(bool(self.torch.isnan(out['sigma_S']).all()))
        self.assertTrue(bool(self.torch.isfinite(out['template_postblur_e']).all()))
        self.assertTrue(out['info']['continuum_frozen'])

    def test_05_null_threshold_source_and_mc_budget_enforced(self):
        from .deploy import reliable_reconstruct
        for detection in ({'z_threshold': 8.}, {'z_threshold': -1., 'threshold_source_sha256': '0'*64},
                          {'z_threshold': 8., 'threshold_source_sha256': '0'*64, 'truth': {}}):
            with self.assertRaises(ValueError):
                reliable_reconstruct(self.raw, self.p.cfg, self.selection, detection=detection)
        with self.assertRaises(ValueError):
            reliable_reconstruct(self.raw, self.p.cfg, self.selection, uncertainty_options={'repeats': 16})


def main():
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReliableDeploymentControls))
    receipt = {'schema': 'v5-reliable-deployment-CPU-controls-v1', 'device': 'cpu',
               'fixture_only_not_scientific_acceptance': True, 'all_passed': result.wasSuccessful(),
               'checks': result.testsRun, 'wall_s': time.perf_counter()-started,
               'failures': [[str(case), trace] for case, trace in result.failures],
               'errors': [[str(case), trace] for case, trace in result.errors],
               'source_sha256': {common.relative(common.CODE/name): common.sha(common.CODE/name)
                                for name in ('common.py', 'deploy.py', 'tests_deploy.py', 'reconstruct.py', 'uncertainty.py')}}
    target = common.REPORTS.parent / 'implementation_checks' / 'deploy' / ('cpu_' + uuid4().hex + '.json')
    common.write_json(target, receipt)
    print(str(target), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
