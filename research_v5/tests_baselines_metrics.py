"""CPU measurement/coverage controls and four-baseline adapter wiring."""
from __future__ import annotations

import copy
import time
import unittest
from uuid import uuid4
from . import common

common.configure_runtime()


class BaselineMetricControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from config import default_config
        from physics import SpectralPhysics
        from training_v4.model import physical_output
        from .baselines import BaselineSet
        torch.set_num_threads(2)
        cls.torch, cls.physical_output = torch, staticmethod(physical_output)
        cls.p = SpectralPhysics(default_config(16), dtype=torch.float64)
        cls.state = torch.zeros((1,5,16,16), dtype=torch.float64)
        cls.state[:,0] = (3052.5-cls.p.T0)/cls.p.Tscale
        cls.state[:,1] = -.5
        cls.state[:,4] = 180/cls.p.Sscale
        cls.beta = torch.tensor([[.5,.01,-.01]], dtype=torch.float64)
        cls.raw_e = cls.p(cls.state, cls.beta)
        cls.raw_dn = cls.raw_e/cls.p.adc_step+cls.p.cfg['black_level_DN']
        cls.calls = []
        class OldHead:
            def infer_electrons(self, raw_e, use_prior=True):
                cls.calls.append(('old_B4', raw_e.clone(), use_prior))
                return physical_output(cls.p, cls.state.clone(), cls.beta.clone(), raw_e)
        class FixedHead:
            physics = cls.p
            def forward_electrons(self, raw_e, state, beta):
                cls.calls.append(('v44', raw_e.clone(), True))
                return physical_output(cls.p, state.clone(), beta.clone(), raw_e)
        def demod(p, raw_e, alpha):
            cls.calls.append(('B1', raw_e.clone(), alpha))
            return physical_output(p, cls.state.clone(), cls.beta.clone(), raw_e)
        def inverse(p, raw_e, method, options):
            cls.calls.append(('v3', raw_e.clone(), method))
            return cls.state.clone(), cls.beta.clone(), [], .01
        cls.adapter = BaselineSet._from_components_for_contract_test('cpu',
            {'method':'old_init__joint_update_v3','solver_config':{}}, OldHead(), FixedHead(), demod, inverse, physical_output)
        cls.outputs = cls.adapter.evaluate(cls.raw_dn)
        physical = physical_output(cls.p, cls.state, cls.beta, cls.raw_e)
        cls.truth = {'evaluation_domain': physical['valid'].clone(),
                     'template_postblur_e': physical['template_e'].clone(),
                     'temperature_color_K': physical['temperature_color_K'].clone()}

    def test_01_all_four_paths_share_one_observed_frame_and_one_v3_inverse(self):
        from .baselines import BASELINE_NAMES
        self.assertEqual(set(self.outputs), set(BASELINE_NAMES))
        self.assertTrue(all(out['info']['status']=='SUCCESS' for out in self.outputs.values()))
        self.assertEqual(len([call for call in self.calls if call[0]=='v3']), 1)
        self.assertEqual(next(call[2] for call in self.calls if call[0]=='B1'), .1)
        for _, raw, _ in self.calls:
            self.assertTrue(self.torch.allclose(raw, self.raw_e, atol=1e-10, rtol=1e-12))
        info = self.outputs['v4.4_final_continuum_fixed']['info']
        self.assertTrue(info['final_continuum_fixed'])
        self.assertTrue(info['internal_continuum_proposal_still_active'])

    def test_02_saturation_refuses_all_heads_before_calls(self):
        before = len(self.calls)
        saturated = self.torch.full_like(self.raw_dn, 2**self.p.cfg['adc_bits']-1)
        rejected = self.adapter.evaluate(saturated)
        self.assertEqual(len(self.calls), before)
        self.assertTrue(all(x['info']['status']=='ABSTAINED' for x in rejected.values()))
        self.assertTrue(all(not bool(x['valid'].any()) for x in rejected.values()))

    def test_03_abstention_does_not_improve_full_truth_domain_error(self):
        from .metrics import score_prediction
        out = copy.deepcopy(self.outputs['v3_physics'])
        out['template_postblur_e'] += 10
        out['temperature_color_K'] += 15
        out['valid'].zero_()
        measured = score_prediction(out, self.truth, self.p)['metrics']
        self.assertAlmostEqual(measured['template_RMSE_e'], 10., places=8)
        self.assertAlmostEqual(measured['temperature_RMSE_K'], 15., places=8)
        self.assertEqual(measured['valid_fraction'], 0.)
        self.assertIsNone(measured['conditional_template_RMSE_e'])
        self.assertEqual(measured['signed_recall'], 0.)
        self.assertEqual(measured['signed_recall_unconditional'], 1.)

    def test_04_raw_residual_uses_observed_raw_and_not_clean_truth(self):
        from .metrics import score_prediction
        out = copy.deepcopy(self.outputs['v3_physics'])
        out['raw_e'] += 30
        truth = copy.deepcopy(self.truth)
        truth['expected_e'] = out['expected_e'].clone()
        measured = score_prediction(out, truth, self.p)['metrics']
        self.assertAlmostEqual(measured['raw_RMSE_e'], 30., places=8)
        self.assertGreater(measured['raw_weighted_RMSE'], 0.)

    def test_05_negative_signal_scoring_is_signed(self):
        from .metrics import score_prediction
        out, truth = copy.deepcopy(self.outputs['v3_physics']), copy.deepcopy(self.truth)
        out['template_postblur_e'] *= -1
        truth['template_postblur_e'] *= -1
        scored = score_prediction(out, truth, self.p)['metrics']
        self.assertEqual(scored['negative_recall'], 1.)
        self.assertIsNone(scored['positive_recall'])
        self.assertEqual(scored['signed_projection'], 1.)
        out['template_postblur_e'] *= -1
        wrong = score_prediction(out, truth, self.p)['metrics']
        self.assertEqual(wrong['sign_recall'], 0.)
        self.assertEqual(wrong['signed_projection'], -1.)

    def test_06_nonfinite_and_missing_fields_do_not_count_success(self):
        from .metrics import score_prediction
        out = copy.deepcopy(self.outputs['v3_physics'])
        out['template_postblur_e'][0,0,0] = float('nan')
        self.assertFalse(score_prediction(out, self.truth, self.p)['numerical_execution_passed'])
        self.assertEqual(score_prediction({}, self.truth, self.p)['status'], 'MISSING_PREDICTION')


def main():
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BaselineMetricControls))
    receipt = {'schema':'v5-baseline-metric-CPU-controls-v1','device':'cpu','all_passed':result.wasSuccessful(),
               'checks':result.testsRun,'wall_s':time.perf_counter()-started,
               'fixture_only_not_scientific_acceptance':True,'formal_old_weights_loaded':False,
               'failures':[[str(case),trace] for case,trace in result.failures],
               'errors':[[str(case),trace] for case,trace in result.errors],
               'source_sha256':{common.relative(common.CODE/name):common.sha(common.CODE/name)
                               for name in ('baselines.py','metrics.py','tests_baselines_metrics.py')}}
    target = common.REPORTS.parent/'implementation_checks'/'baselines'/('cpu_'+uuid4().hex+'.json')
    common.write_json(target,receipt)
    print(str(target),flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__':
    main()
