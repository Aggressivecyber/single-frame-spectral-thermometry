import numpy as np
import pytest
import torch
from config import default_config
from physics import SpectralPhysics
from evaluation_v2.physics_v2 import ExposurePhysicsV2, adc_quantize
from evaluation_v2.metrics import score_prediction

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
torch.set_num_threads(2)


@pytest.fixture(params=[torch.float32, torch.float64])
def physics(request):
    return SpectralPhysics(default_config(64), request.param)


@pytest.mark.parametrize("device", DEVICES)
def test_forward_adjoint_inner_product(physics, device):
    p = physics.to(device)
    gen = torch.Generator(device=device).manual_seed(451)
    x = torch.randn((1, 7, 64, 64), generator=gen, device=device, dtype=p.M.dtype)
    r = torch.randn((1, 64, 64), generator=gen, device=device, dtype=p.M.dtype)
    lhs, rhs = (p.op(x)*r).sum(), (x*p.op.adjoint(r)).sum()
    tol = 2e-5 if p.M.dtype == torch.float32 else 2e-12
    assert float(abs(lhs-rhs)/(abs(lhs)+abs(rhs)).clamp_min(1e-20)) < tol


@pytest.mark.parametrize("device", DEVICES)
def test_jvp_vjp_and_finite_difference_off_knots(physics, device):
    p = physics.to(device)
    gen = torch.Generator(device=device).manual_seed(452)
    state = torch.zeros(1, 5, 64, 64, dtype=p.M.dtype, device=device)
    state[:, 0] = -.1977  # 2802.3 K, away from 5 K LUT knots
    state[:, 1] = np.log(.6)
    state[:, 4] = 1.
    beta = torch.tensor([[.45, .06, -.04]], dtype=p.M.dtype, device=device)
    dz = torch.randn(state.shape, generator=gen, device=device, dtype=p.M.dtype)*.02
    db = torch.randn(beta.shape, generator=gen, device=device, dtype=p.M.dtype)*.02
    r = torch.randn((1, 64, 64), generator=gen, device=device, dtype=p.M.dtype)
    j = p.jvp(state, beta, dz, db)
    gz, gb = p.vjp(state, beta, r)
    a, b = (j*r).sum(), (gz*dz).sum()+(gb*db).sum()
    tol = 3e-5 if p.M.dtype == torch.float32 else 2e-12
    assert float(abs(a-b)/(abs(a)+abs(b))) < tol
    # FP32 cancellation in differences of ~10^4 e intensities is appreciable.
    # Increase the perturbation while proving it remains in the same LUT segment;
    # keep the error tolerance unchanged. Initial smaller-epsilon failures retained.
    eps = .02 if p.M.dtype == torch.float32 else 1e-5
    for sign in (-1, 1):
        moved_T = p.T0+p.Tscale*(state[:, 0]+sign*eps*dz[:, 0])
        assert torch.all(((moved_T-p.temperatures[0])/5).floor() ==
                         (((p.T0+p.Tscale*state[:, 0])-p.temperatures[0])/5).floor())
    fd = (p(state+eps*dz, beta+eps*db)-p(state-eps*dz, beta-eps*db))/(2*eps)
    rel = float(torch.linalg.vector_norm(fd-j)/torch.linalg.vector_norm(j))
    assert rel < (3e-3 if p.M.dtype == torch.float32 else 2e-7)


def test_noiseless_decoder_target_is_post_object_blur(physics):
    yy, xx = physics.background_basis[2], physics.background_basis[1]
    bands = torch.stack([(k+1)*(1+.15*torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy)) for k in range(7)])[None]
    blurred = physics.op.object_blur(bands)
    decoded = physics.op.decode(physics.op(bands))
    # The sampled periodic field has energy outside the optical pass band, removed by A.
    rel = float(torch.linalg.vector_norm(decoded-blurred)/torch.linalg.vector_norm(blurred))
    assert rel < (2e-5 if physics.M.dtype == torch.float32 else 2e-12)


def test_G_700nm_multiplier_changes_observed_not_latent(physics):
    latent = torch.full((1, 64, 64), 3000., dtype=physics.M.dtype)
    continuum, _ = physics.continuum(latent)
    reference, valid = physics.color_temperature(physics.op.object_blur(continuum))
    altered = continuum.clone()
    altered[:, 5] *= .97
    observed, _ = physics.color_temperature(physics.op.object_blur(altered))
    assert float((observed-reference).abs().mean()) > 10
    assert torch.all(latent == 3000)
    truth = {"template_e": np.zeros((64, 64)), "temperature_latent_K": latent[0].numpy(),
             "temperature_reference_color_K": reference[0].numpy(), "temperature_observed_color_K": observed[0].numpy(),
             "uniform_temperature_domain": np.ones((64, 64), bool), "evaluation_domain": valid[0].numpy()}
    pred = {"template_e": np.zeros((64, 64)), "temperature_K": observed[0].numpy(), "valid": valid[0].numpy()}
    m = score_prediction(truth, pred, {"pixel_threshold_e": 60, "frame_threshold_e": 60})["metrics"]
    assert m["temperature_observed_RMSE_K"] == 0
    assert m["temperature_latent_uniform_RMSE_K"] > 10


def test_exposure_half_counts_and_noise_variances(physics):
    p = ExposurePhysicsV2(physics.cfg, 20e-6, physics.M.dtype)
    half = ExposurePhysicsV2(physics.cfg, 10e-6, physics.M.dtype)
    z = torch.zeros(1, 5, 64, 64, dtype=physics.M.dtype)
    z[:, 1] = np.log(.6)
    z[:, 4] = 1.2
    beta = torch.tensor([[.45, .06, -.04]], dtype=physics.M.dtype)
    e1, e2 = p(z, beta), half(z, beta)
    assert torch.allclose(e2, .5*e1, rtol=2e-6, atol=1e-8)
    assert half.cfg["read_noise_e"]**2 == p.cfg["read_noise_e"]**2 == 9
    assert torch.allclose(half.noise_variance_e2(e2, False)-9, .5*(p.noise_variance_e2(e1, False)-9))
    assert half.template_threshold_current_e(60) == 30
    assert half.coefficient_unit == "reference_20us_equivalent_electrons"


def test_empirical_poisson_and_read_noise_exposure():
    gen = torch.Generator().manual_seed(453)
    a = torch.poisson(torch.full((80000,), 1000.), generator=gen)
    b = torch.poisson(torch.full((80000,), 500.), generator=gen)
    read = torch.randn(80000, generator=gen)*3
    assert float(a.mean()) == pytest.approx(1000, rel=.002)
    assert float(b.mean()) == pytest.approx(500, rel=.003)
    assert float(b.var()/a.var()) == pytest.approx(.5, rel=.04)
    assert float(read.var()) == pytest.approx(9, rel=.03)


def test_ADC_and_full_well_units(physics):
    e = torch.tensor([0., physics.cfg["full_well_e"]], dtype=physics.M.dtype)
    dn = adc_quantize(e, physics)
    assert dn.tolist() == [128, 4095]
    raw = torch.full((1, 64, 64), physics.cfg["black_level_DN"]+0.4, dtype=physics.M.dtype)
    continuous = physics.raw_to_electrons(raw)
    quantized = physics.raw_to_electrons(raw.round())
    assert torch.any(continuous != quantized)
