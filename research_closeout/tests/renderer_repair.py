"""Bounded CPU forward contracts for the calibrated independent relay repair.

No inverse, training, saved scientific raw or final payload is used. Native256
flat fields and harmonic eigenfunctions test the optical contract. Small fresh
32x32 fixtures test signed-template and exposure semantics. Expected waveforms
are computed directly as constant plus attenuated cosines, not by another copy
of the production convolution or by importing the core physical forward.
"""
from .. import storage as s
from .. import independent_renderer as renderer
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid


def _calibration_table(cfg):
    """Independent float64 integration of the declared 81-point/5K calibration."""
    temperature = np.arange(cfg['temperature_range_K'][0],
                            cfg['temperature_range_K'][1] + .5,
                            cfg['temperature_lut_step_K'], dtype=np.float64)
    if cfg.get('sampled_instrument_response'):
        raise ValueError('This bounded fixture is registered for the current top-hat calibration')
    flux = []
    for center, width, qe in zip(cfg['centers_nm'], cfg['widths_nm'],
                                 cfg['quantum_efficiencies']):
        wave = np.linspace(center-width/2, center+width/2, 81) * 1e-9
        photons = 2 * 299792458. / wave[None]**4 / np.expm1(
            .01438776877 / (temperature[:, None] * wave[None]))
        flux.append(trapezoid(photons * qe, wave, axis=1))
    flux = np.stack(flux, axis=1)
    anchor = np.interp(cfg['reference_temperature_K'], temperature, flux[:, -1])
    flux *= cfg.get('absolute_radiometric_scale', cfg['reference_905_e']/anchor)
    return temperature, flux


def run(output=None):
    started = time.perf_counter()
    cfg = copy.deepcopy(s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config'])
    if cfg['shape'] != [256, 256]:
        raise ValueError('Native optical contract expects the declared 256x256 calibration')
    h, w = cfg['shape']
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    checks = []

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    matched = renderer.relay_contract(cfg)
    stress = renderer.relay_contract(cfg, relay_model=renderer.LEGACY_RELAY_STRESS)
    check('default_matches_only_relay_calibration',
          matched['relay_model'] == renderer.MATCHED_RELAY
          and matched['renderer_schema'] == renderer.RENDERER_SCHEMA
          and matched['relay_contract_version'] == 2
          and stress['historical_v1_relay_replay']
          and matched['nominal_calibration_matched']
          and matched['nominal_calibration_matched_scope'] == 'relay_transfer_only'
          and not matched['full_forward_core_equivalent']
          and not stress['nominal_calibration_matched'])

    dc = np.full((h, w), 12000.)
    dc_error = float(np.max(np.abs(renderer.apply_relay(dc, cfg)-dc)))
    check('native256_DC_preserved', dc_error < 1e-8, maximum_error_e=dc_error)

    harmonic_rows = []
    for k, ((bx, by), phase) in enumerate(zip(cfg['carrier_bins_xy'], cfg['phase_rad'])):
        wave = np.cos(2*np.pi*(bx*xx/w + by*yy/h) + phase)
        nominal = np.exp(-2*np.pi**2*cfg['relay_psf_sigma_px']**2*((bx/w)**2+(by/h)**2))
        actual = renderer.apply_relay(4000.+1000.*wave, cfg)
        expected = 4000.+1000.*nominal*wave
        error = float(np.max(np.abs(actual-expected)))
        contract = matched['relay_carrier_responses'][k]
        relative_transfer_error = abs(contract['actual_over_nominal']-1.)
        harmonic_rows.append({'band_nm': cfg['centers_nm'][k],
            'maximum_error_e': error, 'relative_transfer_error': relative_transfer_error})
    check('native256_seven_carrier_amplitude_and_phase',
          max(r['maximum_error_e'] for r in harmonic_rows) < 1e-7
          and max(r['relative_transfer_error'] for r in harmonic_rows) < 1e-12,
          carriers=harmonic_rows)

    # Different axis lengths check centering/origin and axis ordering separately.
    rectangular_errors = []
    for shape in ([31, 32], [32, 31]):
        probe_cfg = {**cfg, 'shape': shape}
        sy, sx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        fy, fx = -5/shape[0], 7/shape[1]
        wave = np.cos(2*np.pi*(fx*sx+fy*sy)+.31)
        attenuation = np.exp(-2*np.pi**2*cfg['relay_psf_sigma_px']**2*(fx*fx+fy*fy))
        error = float(np.max(np.abs(renderer.apply_relay(2000.+300.*wave, probe_cfg)
                                    -(2000.+300.*attenuation*wave))))
        rectangular_errors.append(error)
    check('odd_even_rectangular_kernel_centering', max(rectangular_errors) < 1e-8,
          maximum_errors_e=rectangular_errors)

    temperature, dense, _, _ = renderer._tables(json.dumps(cfg, sort_keys=True), 0.)
    cal_temperature, cal_flux = _calibration_table(cfg)
    cal_ratio = cal_flux[:, -2]/cal_flux[:, -1]
    flat_rows = []
    for target_temperature in (2900., 3100., 3300.):
        band = .55*np.array([np.interp(target_temperature, temperature, dense[:, k])
                             for k in range(7)])
        coded = np.zeros((h, w)); expected = np.zeros((h, w))
        for b, (bx, by), gain, depth, phase in zip(band, cfg['carrier_bins_xy'],
                cfg['channel_gain'], cfg['modulation_depth'], cfg['phase_rad']):
            wave = np.cos(2*np.pi*(bx*xx/w+by*yy/h)+phase)
            attenuation = np.exp(-2*np.pi**2*cfg['relay_psf_sigma_px']**2*((bx/w)**2+(by/h)**2))
            coded += b*gain*.5*(1+depth*wave)
            expected += b*gain*.5*(1+depth*attenuation*wave)
        actual = renderer.apply_relay(coded, cfg)
        recovered = []
        for (bx, by), gain, depth, phase in zip(cfg['carrier_bins_xy'],
                cfg['channel_gain'], cfg['modulation_depth'], cfg['phase_rad']):
            projection = np.mean(actual*np.exp(-2j*np.pi*(bx*xx/w+by*yy/h)))
            attenuation = np.exp(-2*np.pi**2*cfg['relay_psf_sigma_px']**2*((bx/w)**2+(by/h)**2))
            recovered.append(float((projection/(gain*depth/4*np.exp(1j*phase)*attenuation)).real))
        measured_color = float(np.interp(recovered[-2]/recovered[-1], cal_ratio, cal_temperature))
        flat_rows.append({'temperature_K': target_temperature,
            'raw_maximum_error_e': float(np.max(np.abs(actual-expected))),
            'band_maximum_error_e': float(np.max(np.abs(np.array(recovered)-band))),
            'recovered_color_K': measured_color, 'color_error_K': measured_color-target_temperature})
    check('three_flat_temperature_forward_contracts',
          max(r['raw_maximum_error_e'] for r in flat_rows) < .17
          and max(r['band_maximum_error_e'] for r in flat_rows) < .17
          and max(abs(r['color_error_K']) for r in flat_rows) < .2,
          temperatures=flat_rows, template_e=0., background_e=0., noise_added=False)

    # Preserve the previously measured stress transfer; do not silently label it nominal.
    historical = np.array([1.0760526567488837, 1.0760526567488837,
        1.1578893200963312, 1.1578893200963312, 1.3815123135958225,
        1.3815123135958225, 1.4865799953760819])
    stress_ratios = np.array([r['actual_over_nominal'] for r in stress['relay_carrier_responses']])
    stress_error = float(np.max(np.abs(stress_ratios-historical)))
    check('historical_sampled_kernel_remains_explicit_stress',
          stress_error < 1e-12 and stress['relay_boundary'] == 'edge continuation of coded field',
          maximum_transfer_ratio_change=stress_error, actual_over_nominal=stress_ratios.tolist())
    stress_wave = np.cos(2*np.pi*(102*xx/w+51*yy/h)+.35)
    stress_actual = renderer.apply_relay(4000.+1000.*stress_wave, cfg,
                                        relay_model=renderer.LEGACY_RELAY_STRESS)
    stress_expected = 4000.+1000.*stress['relay_carrier_responses'][-1]['actual_transfer_real']*stress_wave
    stress_interior_error = float(np.max(np.abs((stress_actual-stress_expected)[4:-4, 4:-4])))
    check('stress_interior_response_matches_its_declared_kernel', stress_interior_error < 1e-7,
          interior_maximum_error_e=stress_interior_error, boundary_excluded_px=4)

    small = {**cfg, 'shape': [32, 32],
             'carrier_bins_xy': [[round(x/8), round(y/8)] for x, y in cfg['carrier_bins_xy']]}
    fixture = dict(seed=913901, geometry='double_hotspot', coefficient_e=-45.)
    base = renderer.render(small, **fixture)
    twice = renderer.render(small, photon_ratio=2., **fixture)
    old = renderer.render(small, relay_model=renderer.LEGACY_RELAY_STRESS, **fixture)
    budget_error = float(np.max(np.abs(twice['expected_e']-2*base['expected_e'])))
    check('signed_template_and_reference20us_scaling_preserved',
          base['physically_available'] and np.min(base['template_postblur_e_ref']) < 0
          and budget_error < 1e-8
          and np.array_equal(base['template_postblur_e_ref'], twice['template_postblur_e_ref'])
          and np.array_equal(base['continuum_postblur_bands'], twice['continuum_postblur_bands']),
          maximum_all_component_scaling_error_e=budget_error, exposure_s=cfg['exposure_s'])
    target_keys = ('template_postblur_e_ref', 'continuum_postblur_bands', 'temperature_color_K')
    check('relay_selection_does_not_change_independent_target_H',
          all(np.array_equal(base[k], old[k]) for k in target_keys)
          and not base['metadata']['infinite_Airy_or_nominal_periodic_equivalence_claimed']
          and 'before native-grid mask' in base['metadata']['pixel_integration'])
    dynamic = renderer.render_exposure(small, mid_fraction=.5, width_fraction=.2, samples=3, **fixture)
    dynamic_twice = renderer.render_exposure(small, mid_fraction=.5, width_fraction=.2,
                                             samples=3, photon_ratio=2., **fixture)
    dynamic_error = float(np.max(np.abs(dynamic_twice['expected_e']-2*dynamic['expected_e'])))
    check('exposure_average_keeps_reference_electron_units',
          dynamic['physically_available'] and dynamic_error < 1e-8
          and dynamic['metadata']['relay_model'] == renderer.MATCHED_RELAY
          and dynamic['metadata']['exposure_s'] == 2e-5
          and abs(dynamic['metadata']['sequence_reference_span_s']-1e-4) < 1e-16
          and np.array_equal(dynamic['template_postblur_e_ref'], dynamic_twice['template_postblur_e_ref']),
          maximum_exposure_scaling_error_e=dynamic_error)
    rejected = False
    try:
        renderer.apply_relay(dc, cfg, relay_model='unregistered_typo')
    except ValueError:
        rejected = True
    check('unknown_relay_cannot_silently_select_another_model', rejected)

    result = {'time': s.now(), 'all_passed': all(c['passed'] for c in checks),
        'tests': checks, 'matched_relay_contract': matched, 'legacy_stress_contract': stress,
        'native_shape': [256, 256], 'small_signed_fixture_shape': [32, 32],
        'full_inverse_calls': 0, 'GPU_used': False, 'formal_raws_saved': False,
        'final_payload_accessed': False, 'full_forward_equivalence_claimed': False,
        'wall_s': time.perf_counter()-started,
        'source_sha256': {p: s.sha(s.ROOT/p) for p in (
            'research_closeout/independent_renderer.py',
            'research_closeout/tests/renderer_repair.py', 'physics.py',
            'reports/training_v4/run_001/backbone_selection.json')}}
    if output is not None:
        output = Path(output)
        if not output.is_absolute():
            output = s.ROOT/output
        s.write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional new JSON receipt on E/D; existing receipts are preserved')
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result['all_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
