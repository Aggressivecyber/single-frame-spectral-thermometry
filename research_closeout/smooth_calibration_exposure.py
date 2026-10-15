"""Known-exposure support for the explicit C1 calibration, for future runs.

The frozen calibration core and historical exposure helper are unchanged.
This wrapper is not part of the photon_ratio=1 quick_v002 registration.
"""
from . import storage as s
from .deployment import known_exposure as _legacy_known_exposure
from .smooth_calibration import SmoothCalibrationPhysics,CALIBRATION_VERSION
from contextlib import contextmanager
import math


EXPOSURE_ADAPTER_VERSION='c1-calibration-known-exposure-v001'


@contextmanager
def known_exposure(physics,ratio):
    """Scale all electron-domain components and C1 slopes; restore identities.

    Ratios compose multiplicatively in nested contexts.  ADC/read units,
    optics, common color lookup and calibration temperatures stay fixed.
    Every exit, including exceptions, restores the exact previous slope
    buffer object; the historical helper restores its own saved objects.
    """
    try: ratio=float(ratio)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('A finite positive known exposure ratio is required') from error
    if not math.isfinite(ratio) or ratio<=0:
        raise ValueError('A finite positive known exposure ratio is required')
    if not isinstance(physics,SmoothCalibrationPhysics):
        raise TypeError('The C1 exposure wrapper requires SmoothCalibrationPhysics')
    previous_slopes=physics.c1_shared_slopes
    with _legacy_known_exposure(physics,ratio):
        try:
            physics.c1_shared_slopes=previous_slopes*ratio
            yield physics
        finally:
            physics.c1_shared_slopes=previous_slopes


def contract():
    """Registration metadata only; no physics construction or inverse call."""
    value={
        'exposure_adapter_version':EXPOSURE_ADAPTER_VERSION,
        'calibration_version':CALIBRATION_VERSION,
        'known_ratio_rule':'finite positive scalar; nested contexts multiply',
        'electron_domain_scaled':['continuum_lut','c1_shared_slopes','Sscale','Bscale'],
        'fixed_units':['adc_step','extra_variance','read_noise_e','full_well_e','adc_bits','black_level_DN'],
        'restoration':'same buffer/config object identities at each nesting level, including exceptions',
        'v002_integration':False,'formal_inverse_calls':0,'truth_used':False,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/smooth_calibration_exposure.py',
            'research_closeout/smooth_calibration.py','research_closeout/deployment.py')},
    }
    value['fingerprint']=s.canonical(value)
    return value
