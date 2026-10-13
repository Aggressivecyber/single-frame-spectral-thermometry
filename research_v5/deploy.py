"""Raw-only physical estimation with actual MC uncertainty and frozen detection."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from . import common

common.configure_runtime()
import torch
from .reconstruct import reconstruct, _calibration_config
from .uncertainty import monte_carlo_uncertainty


@torch.no_grad()
def reliable_reconstruct(raw_dn, calibration, selection, *, uncertainty_options=None, detection=None):
    """Return postblur physics, noise intervals and explicit rejection fields.

    Truth, condition and identity are not accepted. Eight noise repeats rerun
    continuum and signed inversion. MC sigma is distinct from the stationary
    noise proxy used for independent-null detection calibration. Nominal noise
    intervals do not include unknown gain/emissivity/calibration systematics.
    """
    config = _calibration_config(calibration)
    if not isinstance(selection, dict):
        raise TypeError('A fixed external core selection mapping is required')
    core = copy.deepcopy(selection)
    central = reconstruct(raw_dn, config, core)
    dtype = central['expected_e'].dtype
    device = str(central['expected_e'].device)
    physics = common.make_physics(device, dtype, config=config)
    if detection is not None:
        if not isinstance(detection, dict) or set(detection)-{'z_threshold', 'threshold_source_sha256'}:
            raise ValueError('Detection accepts only a frozen z_threshold and its external source hash')
        z = detection.get('z_threshold')
        digest = detection.get('threshold_source_sha256')
        if (not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0
                or not isinstance(digest, str) or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('A positive finite threshold and SHA256 calibration receipt are required')
    options = dict(uncertainty_options or {})
    if int(options.get('repeats', 8)) != 8:
        raise ValueError('The reliable v5 physical route uses the frozen eight MC repeats')
    options['repeats'] = 8
    mc = monte_carlo_uncertainty(physics, raw_dn, reconstruct, config, core, options, baseline=central)
    result = {name: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
              for name, value in central.items()}
    s = result['template_postblur_e']
    result['physical_valid'] = result['valid'].clone()
    if mc['status'] == 'completed':
        for key, value in mc.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().clone()
        result['temperature_uncertainty'] = result['temperature_mc_sigma_K'].clone()
        result['template_mc_sigma_e'] = result['sigma_S'].clone()
        result['SNR_mc'] = result['SNR'].clone()
        result['template_abstain'] = result['low_confidence'].clone()
        result['temperature_abstain'] = ~result['interval_valid']
        result['abstain'] = ~result['valid']
    else:
        invalid = torch.zeros_like(result['valid'])
        nan = torch.full_like(s, float('nan'), dtype=torch.float64)
        for key in ('sigma_S', 'template_mc_sigma_e', 'temperature_mc_sigma_K', 'temperature_uncertainty',
                    'signed_ci_lower_e', 'signed_ci_upper_e', 'temperature_ci_lower_K',
                    'temperature_ci_upper_K', 'SNR', 'SNR_mc'):
            result[key] = nan.clone()
        result.update(valid=invalid, interval_valid=invalid.clone(), undetectable=invalid.clone(),
                      low_confidence=~invalid, abstain=~invalid, template_abstain=~invalid,
                      temperature_abstain=~invalid)
    detection_info = {'status': 'UNAVAILABLE_NO_INDEPENDENT_NULL_THRESHOLD',
                      'MC_interval_zero_exclusion_is_FOV_FPR_control': False}
    if detection is not None:
        from .development import raw_noise_proxy, detection_diagnostics
        proxy = raw_noise_proxy(physics, raw_dn, core)
        sigma_proxy = proxy['sigma_S']
        diagnostics = detection_diagnostics(result, sigma_proxy, float(detection['z_threshold']))
        snr = s.abs()/sigma_proxy
        detected = result['valid'] & result['interval_valid'] & (snr > detection['z_threshold'])
        result.update(detection_sigma_proxy_e=sigma_proxy.detach().clone(), detection_SNR_proxy=snr,
                      detection_limit_e=sigma_proxy*float(detection['z_threshold']),
                      template_detected=detected, undetectable=result['valid'] & ~detected,
                      low_confidence=~detected, template_abstain=~detected)
        detection_info = {'status': 'FROZEN_NULL_CALIBRATED_PROXY_RULE',
                          'threshold_source_sha256': detection['threshold_source_sha256'],
                          'z_threshold': detection['z_threshold'], 'diagnostics': diagnostics,
                          'noise_proxy': proxy['summary'], 'proxy_is_MC_sigma': False,
                          'threshold_recalibrated': False}
    result['info']['uncertainty'] = mc['summary']
    result['info']['temperature_uncertainty_status'] = mc['status']
    result['info']['detection'] = detection_info
    result['info']['reliable_output_limits'] = [
        'approximate finite-MC intervals require external empirical nominal coverage',
        'unknown gain or emissivity is not detectable universally from one frame',
        'no calibrated gas-temperature or physical-species concentration claim',
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--uncertainty-options', type=Path)
    parser.add_argument('--detection', type=Path, help='z_threshold + threshold_source_sha256 only')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args(argv)
    import numpy as np
    raw = common.arrays(args.raw, raw_only=True)['raw_dn'] if args.raw.suffix.lower()=='.npz' else np.load(args.raw, allow_pickle=False)
    if args.raw.suffix.lower() not in ('.npy', '.npz'):
        raise ValueError('Raw input must be NPY or raw_dn-only NPZ')
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Use fresh output and report paths')
    result = reliable_reconstruct(raw, common.read_json(args.calibration), common.read_json(args.selection),
        uncertainty_options=common.read_json(args.uncertainty_options) if args.uncertainty_options else None,
        detection=common.read_json(args.detection) if args.detection else None)
    tensors = {key: value.detach().cpu().numpy() for key, value in result.items() if isinstance(value, torch.Tensor)}
    digest = common.save_npz(args.output, **tensors)
    receipt = {'schema': 'v5-reliable-raw-only-deployment-v1', 'output': common.relative(args.output),
               'output_sha256': digest, 'raw_sha256': common.sha(args.raw),
               'calibration_sha256': common.sha(args.calibration), 'selection_sha256': common.sha(args.selection),
               'info': result['info'], 'valid_fraction': float(result['valid'].double().mean()),
               'template_abstain_fraction': float(result['template_abstain'].double().mean()),
               'shapes': {key: list(value.shape) for key, value in tensors.items()}}
    common.write_json(args.report, receipt)
    print({'output': str(args.output), 'valid_fraction': receipt['valid_fraction'],
           'uncertainty_status': result['info']['temperature_uncertainty_status']}, flush=True)


if __name__ == '__main__':
    main()
