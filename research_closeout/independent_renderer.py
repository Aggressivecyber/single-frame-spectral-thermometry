"""Independent dense-band quadrature and explicitly calibrated spatial rendering.

No reconstruction/legacy physical forward or FFT-OTF core is imported here.
The default relay implements the declared native-grid Gaussian transfer with
real-space periodic convolution. Its discrete kernel is constructed here from
calibration, not truth or reconstructed data. The historical sampled Gaussian
with edge continuation remains a named model-mismatch stress. Object Airy
truncation, object continuation and pixel integration remain independent assumed
models; matching the relay does not equate the complete physical forwards.
The supplied template/background spectral matrices are synthetic integrated
electron responses. Within-band flat photon spectra are an explicit assumed
extension for response-shape stress, not a measured molecular line list.
"""
from . import storage as s
from functools import lru_cache
import json
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d
from scipy.special import j1
from scipy.integrate import trapezoid

MATCHED_RELAY = 'matched_bandlimited_periodic'
LEGACY_RELAY_STRESS = 'sampled_finite_gaussian_edge_stress'
RELAY_MODELS = (MATCHED_RELAY, LEGACY_RELAY_STRESS)
RENDERER_SCHEMA = 'thermometry-independent-renderer-002-relay-calibrated'

@lru_cache(maxsize=16)
def _tables(config_text,tilt,wavelength_samples=321):
    cfg=json.loads(config_text);temperatures=np.arange(cfg['temperature_range_K'][0],cfg['temperature_range_K'][1]+1.,1.)
    reference=[];actual=[];response_factors=[]
    for center,width,qe in zip(cfg['centers_nm'],cfg['widths_nm'],cfg['quantum_efficiencies']):
        lam=np.linspace(center-width/2,center+width/2,wavelength_samples)*1e-9
        exponent=.01438776877/(temperatures[:,None]*lam[None])
        photons=2*299792458./lam[None]**4/np.expm1(exponent)
        nominal=np.full(wavelength_samples,qe)
        if cfg.get('sampled_instrument_response'):
            k=len(reference);sr=cfg['sampled_instrument_response']
            nominal=np.interp(lam*1e9,sr['wavelength_nm'],sr['response'][k],left=0,right=0)
        distortion=1+tilt*(lam*1e9-center)/width
        changed=nominal*distortion
        if np.any(changed<0): raise ValueError('Negative spectral response')
        reference.append(trapezoid(photons*nominal,lam,axis=1))
        actual.append(trapezoid(photons*changed,lam,axis=1))
        response_factors.append(float(trapezoid(changed,lam)/trapezoid(nominal,lam)))
    reference=np.stack(reference,1);actual=np.stack(actual,1)
    at_ref=np.interp(cfg['reference_temperature_K'],temperatures,reference[:,-1])
    scale=cfg.get('absolute_radiometric_scale',cfg['reference_905_e']/at_ref)
    return temperatures,reference*scale,actual*scale,np.asarray(response_factors)

def airy_psf(cutoff,oversampling,radius_px):
    coordinate=np.arange(-radius_px*oversampling,radius_px*oversampling+1)/oversampling
    y,x=np.meshgrid(coordinate,coordinate,indexing='ij');argument=np.pi*cutoff*np.hypot(x,y)
    amplitude=np.ones_like(argument);mask=argument!=0
    amplitude[mask]=2*j1(argument[mask])/argument[mask]
    psf=amplitude**2
    return psf/psf.sum()

def _gaussian_psf(sigma):
    """Historical sampled kernel, which differs from the calibrated Gaussian OTF."""
    if sigma==0: return np.ones((1,1))
    coordinate=np.arange(-max(1,int(np.ceil(5*sigma))),max(1,int(np.ceil(5*sigma)))+1)
    y,x=np.meshgrid(coordinate,coordinate,indexing='ij');kernel=np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return kernel/kernel.sum()

@lru_cache(maxsize=32)
def _matched_relay_axis_kernel(length, sigma):
    """Centered circular kernel of the calibrated native-pixel Gaussian relay.

    The inverse DFT of the separable transfer samples is real but can contain
    signed sidelobes. It is a calibrated discrete operator, not a claimed
    nonnegative continuous intensity PSF. Forward application uses spatial
    convolution; no core forward or core calibration buffer is imported.
    """
    length = int(length)
    if length < 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Invalid relay grid or Gaussian width')
    frequency = np.fft.fftfreq(length)
    transfer = np.exp(-2 * np.pi**2 * float(sigma)**2 * frequency**2)
    kernel = np.fft.fftshift(np.fft.ifft(transfer).real)
    kernel.setflags(write=False)
    return kernel


def _relay_setup(cfg, relay_model):
    if relay_model not in RELAY_MODELS:
        raise ValueError('Undeclared relay model: ' + str(relay_model))
    h, w = (int(v) for v in cfg['shape'])
    sigma = float(cfg['relay_psf_sigma_px'])
    if min(h, w) < 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Invalid relay shape or Gaussian width')
    return h, w, sigma


def apply_relay(coded, cfg, *, relay_model=MATCHED_RELAY):
    """Apply a named relay to the already sampled coded image, without labels.

    The matched path uses periodic native-grid boundaries, as specified by the
    calibration. The stress path retains the historical finite sampled Gaussian
    and edge continuation exactly. Neither path adds pixel integration.
    """
    h, w, sigma = _relay_setup(cfg, relay_model)
    coded = np.asarray(coded, dtype=np.float64)
    if coded.shape != (h, w) or not np.isfinite(coded).all():
        raise ValueError('Relay input must be a finite native-grid image')
    if relay_model == MATCHED_RELAY:
        ky = _matched_relay_axis_kernel(h, sigma)
        kx = _matched_relay_axis_kernel(w, sigma)
        result = convolve1d(coded, ky, axis=0, mode='wrap', origin=0)
        return convolve1d(result, kx, axis=1, mode='wrap', origin=0)
    relay = _gaussian_psf(sigma)
    pad = relay.shape[0] // 2
    return fftconvolve(np.pad(coded, pad, mode='edge'), relay, mode='valid')


def relay_contract(cfg, *, relay_model=MATCHED_RELAY):
    """Record actual carrier transfer; the matched flag has relay-only scope.

    For edge stress, carrier responses characterize its interior kernel; edge
    continuation itself is not Fourier diagonal. This flag does not certify
    object H, spectral perturbations or an experimentally calibrated instrument.
    """
    h, w, sigma = _relay_setup(cfg, relay_model)
    matched = relay_model == MATCHED_RELAY
    if matched:
        ky = _matched_relay_axis_kernel(h, sigma)
        kx = _matched_relay_axis_kernel(w, sigma)
        ly = np.arange(h) - h // 2
        lx = np.arange(w) - w // 2
    else:
        kernel = _gaussian_psf(sigma)
        ky, kx = kernel.sum(1), kernel.sum(0)
        ly = np.arange(len(ky)) - len(ky) // 2
        lx = np.arange(len(kx)) - len(kx) // 2
    carriers = []
    for band, (bx, by) in zip(cfg['centers_nm'], cfg['carrier_bins_xy']):
        fx, fy = float(bx) / w, float(by) / h
        actual = (np.sum(kx * np.exp(-2j * np.pi * fx * lx))
                  * np.sum(ky * np.exp(-2j * np.pi * fy * ly)))
        nominal = float(np.exp(-2 * np.pi**2 * sigma**2 * (fx*fx + fy*fy)))
        carriers.append({'band_nm': band, 'carrier_bins_xy': [bx, by],
            'frequency_cpp': [fx, fy], 'actual_transfer_real': float(actual.real),
            'actual_transfer_imag': float(actual.imag),
            'nominal_transfer': nominal,
            'actual_over_nominal': float(actual.real / nominal)})
    return {'renderer_schema': RENDERER_SCHEMA,
        'relay_contract_version': 2,
        'relay_model': relay_model,
        'historical_v1_default_relay_model': LEGACY_RELAY_STRESS,
        'historical_v1_relay_replay': not matched,
        'relay_boundary': ('periodic native pixel grid' if matched
                           else 'edge continuation of coded field'),
        'nominal_calibration_matched': matched,
        'nominal_calibration_matched_scope': 'relay_transfer_only',
        'full_forward_core_equivalent': False,
        'relay_sigma_px': sigma,
        'relay_kernel_construction': ('independent inverse DFT of analytic Gaussian calibration; separable real-space convolution'
            if matched else 'historical finite Gaussian sampled on integer pixel centers'),
        'relay_kernel_has_signed_coefficients': bool(np.any(kx < 0) or np.any(ky < 0)),
        'relay_is_nonnegative_continuous_PSF_claimed': False,
        'relay_DC_transfer': float(kx.sum() * ky.sum()),
        'relay_carrier_responses': carriers,
        'relay_carrier_response_scope': ('exact periodic operator' if matched
                                        else 'interior kernel only; edge boundary is not Fourier diagonal')}


def render(cfg,*,seed,geometry='hotspot_small',coefficient_e=0.,photon_ratio=1.,
           normalization='fixed_peak',perturbation=None,oversampling=2,frame_fraction=0.,wavelength_samples=321,
           relay_model=MATCHED_RELAY):
    """Return external measurement truth and expected electrons without saving data.

    Formal callers must preregister mother IDs/seeds and the factor matrix.
    Target object response remains the independent finite nominal Airy H;
    calibration perturbations change the measurement, not that target.
    Template output is reference20us. The matched relay is the default; use
    LEGACY_RELAY_STRESS explicitly to reproduce historical relay mismatch.
    Historical plans with no relay field used that stress path (schema v1);
    they must pass it explicitly instead of replaying against the new default.
    """
    perturbation=dict(perturbation or {});rng=np.random.default_rng(seed)
    h,w=cfg['shape'];radius=max(8,min(128,min(h,w)//2));pad=radius;os=int(oversampling)
    if os<1 or photon_ratio<=0: raise ValueError('Invalid integration or photon budget')
    yy,xx=np.meshgrid((np.arange(-pad*os,(h+pad)*os)+.5)/os-.5,
                     (np.arange(-pad*os,(w+pad)*os)+.5)/os-.5,indexing='ij')
    cx=rng.uniform(.32,.68)*(w-1)+frame_fraction*.1*w
    cy=rng.uniform(.32,.68)*(h-1)
    small=min(h,w)/64;large=min(h,w)/12
    small=np.maximum(small,.5)
    if geometry=='hotspot_small': shape=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*small**2))
    elif geometry=='hotspot_large': shape=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*large**2))
    elif geometry=='double_hotspot':
        separation=min(h,w)/12
        shape=np.exp(-((xx-cx-separation)**2+(yy-cy)**2)/(2*small**2))+.8*np.exp(-((xx-cx+separation)**2+(yy-cy)**2)/(2*small**2))
    elif geometry=='edge': shape=.5*(1+np.tanh((xx-cx)/max(small/2,.25)))
    elif geometry=='filament': shape=np.exp(-(yy-cy-.06*h*np.sin(xx/w*2*np.pi))**2/(2*small**2))
    else: raise ValueError('Undeclared independent structure: '+geometry)
    core=(slice(pad*os,(pad+h)*os),slice(pad*os,(pad+w)*os))
    shape=shape/shape[core].max()
    reference_area=2*np.pi*(min(h,w)/64)**2
    if normalization=='fixed_total_signal': shape*=reference_area/(shape[core].sum()/os**2)
    elif normalization!='fixed_peak': raise ValueError('Undeclared signal normalization')
    smooth=np.cos((xx-cx)/w*np.pi)*np.cos((yy-cy)/h*np.pi)
    hot=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*large**2))
    T=3000+rng.uniform(-80,80)+rng.uniform(60,160)*(smooth+.3*hot)
    amplitude=rng.uniform(.4,.7)*(1+.08*np.sin(yy/h*2*np.pi+rng.uniform(0,6)))
    template=shape*float(coefficient_e)*(1+.1*frame_fraction)
    beta=np.array([rng.uniform(.3,.9),rng.uniform(-.06,.06),rng.uniform(-.06,.06)])
    background=cfg['background_scale_e']*(beta[0]+beta[1]*(2*xx/(w-1)-1)+beta[2]*(2*yy/(h-1)-1))
    temperature_grid,reference_lut,actual_lut,response_factors=_tables(json.dumps(cfg,sort_keys=True),float(perturbation.get('response_tilt',0.)),wavelength_samples)
    nominal_kernel=airy_psf(cfg['object_cutoff_cpp'],os,radius)
    changed_kernel=airy_psf(cfg['object_cutoff_cpp']/(1+perturbation.get('psf_fraction',0.)),os,radius)
    def pixel_integrate(field):
        a=field[core];return a.reshape(h,os,w,os).mean((1,3))
    def nominal_blur(field): return pixel_integrate(fftconvolve(field,nominal_kernel,mode='same'))
    def actual_blur(field): return pixel_integrate(fftconvolve(field,changed_kernel,mode='same'))
    template_target=nominal_blur(template);background_actual=actual_blur(background)
    template_actual=actual_blur(template)
    Ctarget=[];bands=[];min_latent=np.inf
    gains=np.asarray(perturbation.get('relative_gain',[1.]*7),dtype=float)
    emissivity=np.asarray(perturbation.get('relative_emissivity',[1.]*7),dtype=float)
    if gains.shape!=(7,) or emissivity.shape!=(7,) or np.any(gains<=0) or np.any(emissivity<=0):
        raise ValueError('Invalid band gain/emissivity factor')
    for k in range(7):
        reference_C=amplitude*np.interp(T,temperature_grid,reference_lut[:,k])
        actual_C=amplitude*np.interp(T,temperature_grid,actual_lut[:,k])*emissivity[k]
        Ctarget.append(nominal_blur(reference_C))
        M=cfg['template_response'][k][2];q=cfg['background_response'][k]
        min_latent=min(min_latent,float((actual_C+response_factors[k]*(M*template+q*background)).min()))
        bands.append(gains[k]*(actual_blur(actual_C)+response_factors[k]*(M*template_actual+q*background_actual)))
    Ctarget=np.stack(Ctarget);bands=np.stack(bands)*photon_ratio
    yn,xn=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');coded=np.zeros((h,w))
    registration=perturbation.get('registration_xy_px',[0.,0.])
    for k,((kx,ky),gain,depth,phase) in enumerate(zip(cfg['carrier_bins_xy'],cfg['channel_gain'],cfg['modulation_depth'],cfg['phase_rad'])):
        mask=gain*.5*(1+depth*np.cos(2*np.pi*(kx*(xn+registration[0])/w+ky*(yn+registration[1])/h)+phase))
        coded+=mask*bands[k]
    expected=apply_relay(coded,cfg,relay_model=relay_model)
    ratio=Ctarget[-2]/np.maximum(Ctarget[-1],1e-12)
    ratio_lut=reference_lut[:,-2]/reference_lut[:,-1]
    color=np.interp(ratio,ratio_lut,temperature_grid)
    legal=bool(min_latent>=0 and np.isfinite(expected).all() and expected.min()>=0)
    return {'expected_e':expected,'template_postblur_e_ref':template_target,
        'temperature_color_K':color,'continuum_postblur_bands':Ctarget,
        'temperature_latent_K':pixel_integrate(T),'evaluation_domain':np.ones((h,w),bool),
        'physically_available':legal,'minimum_latent_band_e_ref':min_latent,'beta':beta,
        'metadata':{'seed':int(seed),'geometry':geometry,'coefficient_e':coefficient_e,
            'normalization':normalization,'reference_signal_area_px':reference_area,'photon_ratio':photon_ratio,
            'perturbation':perturbation,'oversampling':os,'wavelength_samples':wavelength_samples,
            'reference_temperature_table_step_K':1.,'finite_Airy_radius_px':radius,'boundary':'nonperiodic extended object and crop',
            **relay_contract(cfg,relay_model=relay_model),'target_optics_fixed':True,
            'pixel_integration':'object postblur subpixels averaged before native-grid mask; coded product not area integrated',
            'finite_Airy_radius_is_declared_validation_model':True,
            'infinite_Airy_or_nominal_periodic_equivalence_claimed':False,
            'template_spectrum':'assumed uniform within band, not a measured molecular line list',
            'radiometry':'synthetic reference-electron anchor; absolute incident flux unverified',
            'core_forward_imported':False,'target_constructed_from_reconstruction':False}}

def render_exposure(cfg,*,mid_fraction,width_fraction=.2,samples=5,**kwargs):
    if samples<3 or width_fraction<=0: raise ValueError('Exposure integration requires at least three time samples')
    frames=[render(cfg,frame_fraction=t,**kwargs) for t in np.linspace(mid_fraction-width_fraction/2,mid_fraction+width_fraction/2,samples)]
    def temporal_mean(key): return trapezoid(np.stack([f[key] for f in frames]),axis=0)/(samples-1)
    result=dict(frames[samples//2])
    for key in ('expected_e','template_postblur_e_ref','continuum_postblur_bands','temperature_latent_K'):
        result[key]=temporal_mean(key)
    T,lut,_,_=_tables(json.dumps(cfg,sort_keys=True),0.,kwargs.get('wavelength_samples',321))
    C=result['continuum_postblur_bands']
    result['temperature_color_K']=np.interp(C[-2]/np.maximum(C[-1],1e-12),lut[:,-2]/lut[:,-1],T)
    result['physically_available']=all(f['physically_available'] for f in frames)
    result['metadata']={**result['metadata'],'exposure_time_samples':samples,
        'mid_fraction':mid_fraction,'width_fraction':width_fraction,'exposure_s':cfg['exposure_s'],
        'sequence_reference_span_s':cfg['exposure_s']/width_fraction,
        'color_target':'ratio of exposure-integrated continuum bands, not averaged temperature',
        'inverse_temporal_prior_used':False}
    return result

def sample_raw(expected_e,cfg,seed):
    mean=np.asarray(expected_e,dtype=float)
    if not np.isfinite(mean).all() or mean.min()<0: raise ValueError('Illegal Poisson expectation')
    rng=np.random.default_rng(seed)
    measured=rng.poisson(mean)+rng.normal(0,cfg['read_noise_e'],mean.shape)
    step=cfg['full_well_e']/(2**cfg['adc_bits']-1-cfg['black_level_DN'])
    dn=np.rint(measured/step+cfg['black_level_DN']);top=2**cfg['adc_bits']-1
    return np.clip(dn,0,top).astype(np.uint16),{'saturated_pixels':int((dn>=top).sum()),
        'lower_clipped_pixels':int((dn<=0).sum()),'raw_pixels':int(dn.size),'noise_seed':int(seed),
        'Poisson_read_ADC_once':True,'ADC_step_e':step}
