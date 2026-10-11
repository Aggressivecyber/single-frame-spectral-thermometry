import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('post_metrics', Path(__file__).resolve().parents[1]/'reports/audit_v2/run_001/post_metrics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_perfect_postblur_spillover_is_not_ghost_error():
    target = np.array([[100., 25.], [100., 25.]])
    source = np.array([[True, False], [True, False]])
    result = module.source_halo_metrics(target, target.copy(), source, np.ones((2, 2), bool), 20.)
    assert result['outside_truth_optical_spillover_RMS_e'] == 25.
    assert result['outside_predicted_signal_RMS_e'] == 25.
    assert result['outside_error_RMSE_e'] == 0
    assert result['outside_abs_error_gt60_fraction'] == 0


def test_unexplained_exterior_structure_and_nonfinite_recorded():
    target = np.zeros((2, 2))
    prediction = np.array([[0., 100.], [0., 100.]])
    source = np.array([[True, False], [True, False]])
    result = module.source_halo_metrics(target, prediction, source, np.ones((2, 2), bool), 60.)
    assert result['outside_error_RMSE_e'] == 100.
    assert result['outside_nonsupport_detected_fraction'] == 1.
    prediction[0, 1] = np.nan
    assert module.source_halo_metrics(target, prediction, source, np.ones((2, 2), bool), 60.)['outside_error_RMSE_e'] is None
