"""Independent reference-exposure path; the published LUT is never re-anchored."""
import copy
import torch
from physics import SpectralPhysics

REFERENCE_EXPOSURE_S = 20e-6


class ExposurePhysicsV2(SpectralPhysics):
    def __init__(self, nominal_cfg, exposure_s, dtype=torch.float32):
        # Build at the fixed reference exposure, then scale physical electron counts.
        cfg = copy.deepcopy(nominal_cfg)
        cfg["exposure_s"] = float(exposure_s)
        super().__init__(cfg, dtype)
        self.exposure_ratio = exposure_s/REFERENCE_EXPOSURE_S
        self.continuum_lut.mul_(self.exposure_ratio)
        self.Sscale *= self.exposure_ratio
        self.Bscale *= self.exposure_ratio
        self.coefficient_unit = "reference_20us_equivalent_electrons"
        # Read noise and ADC conversion stay in current-exposure electron units.

    def noise_variance_e2(self, expected_e, include_adc=True):
        return expected_e.clamp_min(0) + self.cfg["read_noise_e"]**2 + (self.adc_step**2/12 if include_adc else 0.)

    def template_threshold_current_e(self, reference_e):
        return reference_e*self.exposure_ratio


def adc_quantize(expected_e, physics):
    return (expected_e/physics.adc_step + physics.cfg["black_level_DN"]).round().clamp(0, 2**physics.cfg["adc_bits"]-1)
