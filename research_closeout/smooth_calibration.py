"""An explicit C1 interpolation change to the frozen promoted calibration.

The original LUT knots, optics and common color reporting remain untouched.
Only +/-0.25 K neighborhoods of internal knots replace the old straight line
with monotone cubic Hermite segments.  This is a new physical objective, not
a higher precision implementation of the historical v3 estimator.
"""
from . import storage as s
from . import measurement as m
from physics import SpectralPhysics
import copy
import hashlib
import torch


CALIBRATION_VERSION='continuum-c1-local-hermite-v003'
PHYSICAL_OBJECTIVE_VERSION='joint-profile-v003-shared-anchor-c1-calibration'
HALF_WINDOW_K=.25
ALGORITHM_REFERENCE='https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html'


def tensor_sha(value):
    """Hash dtype, shape and exact CPU tensor payload, including complex data."""
    array=value.detach().cpu().contiguous().numpy()
    header=str(array.dtype).encode()+b'|'+str(array.shape).encode()+b'|'
    return hashlib.sha256(header+array.tobytes()).hexdigest()


def _shared_slopes(knots,values):
    h=knots[1:]-knots[:-1]
    delta=(values[1:]-values[:-1])/h[:,None]
    if not torch.isfinite(values).all() or not bool((h>2*HALF_WINDOW_K).all()):
        raise ValueError('Calibration must be finite with disjoint fixed Hermite windows')
    if not bool((delta>=0).all()):
        raise ValueError('The declared C1 adapter requires nondecreasing band calibration')
    w1=(2*h[1:]+h[:-1])[:,None];w2=(h[1:]+2*h[:-1])[:,None]
    left=delta[:-1];right=delta[1:];positive=(left>0)&(right>0)
    safe_left=torch.where(positive,left,torch.ones_like(left))
    safe_right=torch.where(positive,right,torch.ones_like(right))
    middle=torch.where(positive,(w1+w2)/(w1/safe_left+w2/safe_right),torch.zeros_like(left))
    # The calibrated range is closed: its endpoints keep the original line
    # slopes.  Shared internal slopes use the PCHIP harmonic-mean rule.
    return torch.cat([delta[:1],middle,delta[-1:]],0)


def _hermite(fraction,width,left,right,left_slope,right_slope):
    u=fraction[...,None];v=1-u
    values=(1+2*u)*v.square()*left+u*v.square()*width*left_slope
    values+=u.square()*(3-2*u)*right+u.square()*(u-1)*width*right_slope
    secant=(right-left)/width
    derivative=left_slope*(1-4*u+3*u.square())
    derivative+=right_slope*(-2*u+3*u.square())+6*secant*u*v
    return values,derivative


class SmoothCalibrationPhysics(SpectralPhysics):
    calibration_version=CALIBRATION_VERSION
    physical_objective_version=PHYSICAL_OBJECTIVE_VERSION

    def __init__(self,reference):
        if reference.M.dtype!=torch.float64:
            raise ValueError('The C1 adapter accepts the original float32-promoted float64 physics')
        # Build a normal float64 module, then replace every original buffer
        # with the audited promoted calibration.  No optical re-computation is
        # retained merely because this constructor uses a different dtype.
        super().__init__(copy.deepcopy(reference.cfg),dtype=torch.float64)
        self.to(reference.M.device)
        self.load_state_dict(reference.state_dict(),strict=True)
        self._original_buffer_hashes={name:tensor_sha(value) for name,value in reference.named_buffers()}
        actual=dict(self.named_buffers())
        if any(tensor_sha(actual[name])!=digest for name,digest in self._original_buffer_hashes.items()):
            raise ArithmeticError('Original calibrated buffer bytes changed in the C1 adapter')
        if self.cfg!=reference.cfg:
            raise ArithmeticError('Original calibration configuration changed')
        self._original_config_sha256=s.canonical(reference.cfg)
        # Compute derived calibration bytes once on CPU.  Target-device
        # division must not silently create another physical calibration.
        slopes=_shared_slopes(self.temperatures.detach().cpu(),self.continuum_lut.detach().cpu())
        self.register_buffer('c1_shared_slopes',slopes.to(reference.M.device))

    def continuum(self,temperature):
        temperature=torch.as_tensor(temperature,device=self.temperatures.device,dtype=self.temperatures.dtype)
        if not torch.isfinite(temperature).all():
            raise ValueError('Nonfinite calibration temperature')
        low=self.temperatures[0];high=self.temperatures[-1]
        bounded=temperature.clamp(low,high)
        index=(torch.searchsorted(self.temperatures,bounded.contiguous(),right=True)-1).clamp(0,len(self.temperatures)-2)
        x0=self.temperatures[index];x1=self.temperatures[index+1]
        y0=self.continuum_lut[index];y1=self.continuum_lut[index+1]
        delta=(y1-y0)/(x1-x0)[...,None]
        values=y0+(bounded-x0)[...,None]*delta
        derivative=delta
        width=HALF_WINDOW_K
        at_left=(index>0)&((bounded-x0)<width)
        u_left=((bounded-x0)/width).clamp(0,1)
        left_values,left_derivative=_hermite(u_left,width,y0,y0+width*delta,
            self.c1_shared_slopes[index],delta)
        at_right=(index<len(self.temperatures)-2)&((x1-bounded)<width)
        u_right=((bounded-(x1-width))/width).clamp(0,1)
        right_values,right_derivative=_hermite(u_right,width,y1-width*delta,y1,
            delta,self.c1_shared_slopes[index+1])
        values=torch.where(at_left[...,None],left_values,values)
        values=torch.where(at_right[...,None],right_values,values)
        derivative=torch.where(at_left[...,None],left_derivative,derivative)
        derivative=torch.where(at_right[...,None],right_derivative,derivative)
        # Constant extension is only a guard outside the calibrated range.
        # C1 continuity is claimed within that closed range, at every internal
        # knot and window seam.  The returned derivative follows this clamp.
        inside=(temperature>=low)&(temperature<=high)
        derivative=torch.where(inside[...,None],derivative,torch.zeros_like(derivative))
        return values.movedim(-1,-3),derivative.movedim(-1,-3)


def physics(device='cpu'):
    """Raw/calibration-only factory; no labels, cached state or inverse calls."""
    return SmoothCalibrationPhysics(m.physics(device,dtype=torch.float64))


def contract(p):
    if not isinstance(p,SmoothCalibrationPhysics):
        raise TypeError('C1 contract requires the explicit calibration adapter')
    buffers=dict(p.named_buffers())
    hashes={name:tensor_sha(buffers[name]) for name in p._original_buffer_hashes}
    value={
        'calibration_version':CALIBRATION_VERSION,
        'physical_objective_version':PHYSICAL_OBJECTIVE_VERSION,
        'physical_objective_changed':True,
        'historical_v3_precision_upgrade':False,
        'original_buffers_byte_identical':hashes==p._original_buffer_hashes,
        'original_buffer_sha256':hashes,
        'derived_slope_sha256':tensor_sha(p.c1_shared_slopes),
        'derived_slope_compute_device':'cpu float64, then exact device transfer',
        'config_sha256':s.canonical(p.cfg),
        'original_config_unchanged':s.canonical(p.cfg)==p._original_config_sha256,
        'common_color_lookup_unchanged':hashes['color_ratio_lut']==p._original_buffer_hashes['color_ratio_lut'],
        'interpolation':'fixed-window monotone cubic Hermite; original straight line elsewhere',
        'internal_shared_derivative':'weighted harmonic mean of adjacent secants (PCHIP rule)',
        'half_window_K':HALF_WINDOW_K,
        'window_selection':'fixed before any new real-raw inverse; calibration-only bound',
        'extrapolation':'constant clamp outside calibration; derivative zero outside',
        'C1_scope':'all internal knots and window seams within the closed calibrated temperature range',
        'error_bound_per_band':'(4/27)*half_window_K*abs(shared_slope-segment_slope)',
        'calibration_provenance':p.cfg.get('response_provenance',{}),
        'algorithm_reference':ALGORITHM_REFERENCE,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/smooth_calibration.py','research_closeout/measurement.py','physics.py')},
        'truth_used':False,
    }
    value['fingerprint']=s.canonical(value)
    return value
