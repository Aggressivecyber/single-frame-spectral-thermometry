"""The complete frozen physical forward operator and deployment output semantics."""
from . import storage as s
import copy
import torch
from physics import SpectralPhysics

def selection(): return s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')
def physics(device='cpu', *, dtype=torch.float64):
    p=SpectralPhysics(selection()['physics_config']).to(device)
    if dtype==torch.float32: return p
    # Promote exact original coefficients, preserving complex decode transfers.
    for module in p.modules():
        for key,value in module._buffers.items():
            if value is not None:
                module._buffers[key]=value.to(torch.complex128 if value.is_complex() else dtype)
    return p
def raw_tensor(raw_dn,p):
    raw=torch.as_tensor(raw_dn,device=p.M.device,dtype=p.M.dtype)
    if raw.ndim==2: raw=raw[None]
    if raw.shape!=(1,*p.cfg['shape']) or not torch.isfinite(raw).all(): raise ValueError('Invalid raw DN frame')
    return raw
def fixed_weights(p,raw_e): return (raw_e.clamp_min(0)+p.extra_variance).reciprocal()
def extra_smooth(image,sigma):
    if sigma==0: return image
    h,w=image.shape[-2:]
    fy,fx=torch.meshgrid(torch.fft.fftfreq(h,device=image.device,dtype=image.dtype),
        torch.fft.fftfreq(w,device=image.device,dtype=image.dtype),indexing='ij')
    transfer=torch.exp(-2*torch.pi**2*sigma**2*(fx.square()+fy.square()))
    return torch.fft.ifft2(torch.fft.fft2(image)*transfer).real
def continuum(p,z,grid=None):
    if grid is not None:
        z=torch.nn.functional.interpolate(z,size=p.cfg['shape'],mode='bilinear',align_corners=False)
    T=p.T0+p.Tscale*z[:,0]; C,_=p.continuum(T)
    return C*z[:,1:2].exp()
def output(p,raw_e,C,u,beta,info):
    Cpost=p.op.object_blur(C)
    bg=p.Bscale*torch.einsum('bl,lhw->bhw',beta,p.background_basis)
    bgpost=p.op.object_blur(bg)
    bands=Cpost+p.M[:,2][None,:,None,None]*u[:,None]+p.q[None,:,None,None]*bgpost[:,None]
    predicted=p.op.encode_images(bands)
    temperature,color_valid=p.color_temperature(Cpost)
    numeric=bool(info.get('numerical_certified',False))
    return {'temperature_color_K':temperature,'template_postblur_e_ref':u,
        'continuum_postblur_bands':Cpost,'predicted_raw_e':predicted,
        'predicted_raw_DN':predicted/p.adc_step+p.cfg['black_level_DN'],
        'standardized_residual':(predicted-raw_e)*fixed_weights(p,raw_e).sqrt(),
        'valid':color_valid,'numerical_valid':numeric,
        'low_signal':Cpost[:,-1]<=1,'calibration_status':'calibration_assumed_not_verified',
        'detection_status':'NOT_CALIBRATED','beta':beta,'info':info}
