import numpy as np
import torch
import pytest
from config import default_config
from network import ParameterUnfoldedNet
from evaluation_v2.common import MeasurementInput
from evaluation_v2.methods import Reconstructor, demod_fit
from evaluation_v2.metrics import score_prediction


def test_zero_estimator_is_exact_unit_L2_temperature_NA():
    model = ParameterUnfoldedNet(default_config(32)).eval()
    raw = np.full((32, 32), 200., np.float32)
    out = Reconstructor(model, "Z0_zero_template").reconstruct(MeasurementInput(raw))
    truth = {"template_e": np.ones((32, 32)), "temperature_observed_color_K": np.full((32, 32), 3000.),
             "evaluation_domain": np.ones((32, 32), bool)}
    s = score_prediction(truth, out["prediction"], {"pixel_threshold_e": 60., "frame_threshold_e": 60.})
    assert s["status"] == "SUCCESS"
    assert s["metrics"]["template_relative_L2"] == 1.
    assert s["metrics"]["temperature_observed_RMSE_K"] is None
    assert out["raw_reprojection_RMSE_e"] is None


def test_actual_reconstructor_rejects_saturated_and_truth_interface():
    model = ParameterUnfoldedNet(default_config(32)).eval()
    recon = Reconstructor(model, "B4_published")
    assert recon.reconstruct(MeasurementInput(np.full((32, 32), 4095.)))["status"] == "SATURATION_REJECTED"
    with pytest.raises(TypeError):
        recon.reconstruct({"raw_dn": np.zeros((32, 32)), "truth": {}})


def test_B1_fits_postblur_target_without_second_object_blur():
    p = ParameterUnfoldedNet(default_config(64), torch.float64).physics
    yy, xx = p.background_basis[2], p.background_basis[1]
    z = torch.zeros(1, 5, 64, 64, dtype=torch.float64)
    z[:, 1] = np.log(.6)
    z[:, 4] = 4*torch.exp(-((xx-.2)**2+(yy+.1)**2)/.03)
    beta = torch.tensor([[.45, .06, -.04]], dtype=torch.float64)
    out = demod_fit(p, p(z, beta))
    target = p.op.object_blur(z[:, 4]*p.Sscale)
    err = torch.linalg.vector_norm(out["species_signed_e"][:, 2]-target)/torch.linalg.vector_norm(target)
    assert err < 1e-3  # scaled-fit ridge is the only nominal residual
