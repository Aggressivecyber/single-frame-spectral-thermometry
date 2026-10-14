"""Controlled core experiments and an explicitly separate independent renderer."""
from . import storage as s
from . import measurement as m
from . import independent_renderer as independent
import numpy as np
import torch

def array_hash(array):
    from research_v6.inputs import array_hash as original_hash
    return original_hash(array)

def structure(p,geometry,normalization):
    h,w=p.cfg['shape'];y,x=torch.meshgrid(torch.arange(h,device=p.M.device,dtype=p.M.dtype),
        torch.arange(w,device=p.M.device,dtype=p.M.dtype),indexing='ij')
    cx=.47*(w-1);cy=.53*(h-1);small=min(h,w)/64;large=min(h,w)/12
    if geometry=='hotspot_small': field=torch.exp(-((x-cx).square()+(y-cy).square())/(2*small**2))
    elif geometry=='hotspot_large': field=torch.exp(-((x-cx).square()+(y-cy).square())/(2*large**2))
    elif geometry=='double_hotspot':
        sep=min(h,w)/12
        field=torch.exp(-((x-cx-sep).square()+(y-cy).square())/(2*small**2))+.8*torch.exp(-((x-cx+sep).square()+(y-cy).square())/(2*small**2))
    elif geometry=='edge': field=.5*(1+torch.tanh((x-cx)/(small/2)))
    elif geometry=='filament': field=torch.exp(-(y-cy-.06*h*torch.sin(x/w*2*torch.pi)).square()/(2*small**2))
    else: raise ValueError('Unknown registered structure')
    field=field/field.max()
    if normalization=='fixed_total_signal': field=field*(2*np.pi*small**2)/field.sum()
    elif normalization!='fixed_peak': raise ValueError('Unknown normalization')
    return field

@torch.no_grad()
def render_core(row,p):
    from training_v4.data import mother_state
    state,beta,original_shape,provenance=mother_state(p,row['mother'])
    shape=structure(p,row['geometry'],row['normalization']) if row['family']=='structure' else original_shape
    state[:,4]=row['coefficient_e']*shape/p.Sscale
    C,S,B,_=p.components(state,beta);bands=C+S+B
    Cpost=p.op.object_blur(C);T,_=p.color_temperature(Cpost)
    expected=p.op(bands)[0]*row['photon_ratio'];u=p.op.object_blur(state[:,4]*p.Sscale)[0]
    return {'expected_e':expected.cpu().numpy(),'template_postblur_e_ref':u.cpu().numpy(),
        'temperature_color_K':T[0].cpu().numpy(),'continuum_postblur_bands':Cpost[0].cpu().numpy(),
        'temperature_latent_K':(p.T0+p.Tscale*state[0,0]).cpu().numpy(),
        'evaluation_domain':np.ones(p.cfg['shape'],bool),'physically_available':bool(bands.min()>=0),
        'raw_available':bool(torch.isfinite(expected).all() and expected.min()>=0),
        'minimum_latent_band_e_ref':float(bands.min()),
        'metadata':{'engine':'original_nominal_forward','target':'original fixed object OTF',
            'photon_ratio':row['photon_ratio'],'ADC_read_units_fixed':True,
            'negative_spectra_retained_as_numerical_stress_only':bool(bands.min()<0),**provenance}}

def render(row,p):
    if row['engine']=='core': return render_core(row,p)
    kwargs={'seed':row['mother']['seed'],'geometry':row['geometry'],
        'coefficient_e':row['coefficient_e'],'photon_ratio':row['photon_ratio'],
        'normalization':row['normalization'],'perturbation':row['perturbation']}
    if row['family']=='dynamic':
        value=independent.render_exposure(p.cfg,mid_fraction=row['mid_fraction'],
            width_fraction=row['width_fraction'],**kwargs)
    else: value=independent.render(p.cfg,**kwargs)
    mean=value['expected_e'];value['raw_available']=bool(np.isfinite(mean).all() and mean.min()>=0)
    return value

def truth_for_score(frame,p):
    values={k:torch.as_tensor(frame[key],device=p.M.device,dtype=torch.bool if k=='evaluation_domain' else p.M.dtype)[None]
        for k,key in {'template_postblur_e':'template_postblur_e_ref','temperature_color_K':'temperature_color_K',
            'continuum_postblur_bands':'continuum_postblur_bands','temperature_latent_K':'temperature_latent_K',
            'evaluation_domain':'evaluation_domain'}.items()}
    return values
