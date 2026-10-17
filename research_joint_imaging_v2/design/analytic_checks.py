"""Analytic checks using formulas transcribed from the pinned six-band branch.
Does not load trained weights or replay any saved dataset.
"""
import json, numpy as np
from pathlib import Path
root=Path(__file__).parent
centers=np.array([431.5,486.,516.5,700.,800.,905.])
wl=(centers[:,None]+10*np.linspace(-.5,.5,9))/1000
q=np.array([.5,1,1,1,1,1,1,1,.5])[None,:]*10/8000

def curve(T):
    u=14387.76877/(wl*T); em=np.expm1(u); b=wl**-4/em
    c=(b*q).sum(1); d=(b*u/T*(1+1/em)*q).sum(1)
    ref=(wl[4]**-4/np.expm1(14387.76877/(wl[4]*3000))*q[0]).sum()
    return c/ref,d/ref

c,d=curve(3000.); a=4000/c[4]; I=a*c
ct,dt=c[3:],d[3:]; h=(ct@dt)/(ct@ct)
j=a*(dt-ct*h); JT=j/(j@j); logJT=JT*I[3:]
Jcv=np.outer(c[:3],ct)/(ct@ct)+np.outer(a*(d[:3]-c[:3]*h),JT)
f=np.fft.fftfreq(128); fy,fx=np.meshgrid(f,f,indexing='ij'); support=(fx**2+fy**2)<.12**2
bins=np.array([[10,49],[20,-30],[30,20],[39,-59],[49,-10],[59,39]])
relay=np.exp(-2*np.pi**2*.35**2*(fx**2+fy**2))
ac=np.array([(.85/4)**2*(np.roll(relay**2,(ky,kx),(0,1))+np.roll(relay**2,(-ky,-kx),(0,1))) for kx,ky in bins])
mu=.5*ac[:,support].mean(1)
# The constant-weight projected normal is diagonal(sidebands)+rank1(DC).
u=.5*relay[:, :][...,None]
N=np.diag(ac[:,0,0])+.25*np.ones((6,6))
retention=np.linalg.solve(N+np.diag(mu),np.diag(mu))
results={
 'scope':'independent analytic local checks only; not checkpoint evaluation',
 'pinned_commit':'9719bdc19c2b800b7685084709b5c6c403a9b3bb',
 'temperature_K':3000.,'mean_800_e_uniform':4000.,
 'band_e_uniform':I.tolist(),'shared_thermal_scale_e':float(I[3:].mean()),
 'old_single_CNN_proposal_max_e_per_channel':float(.2*I[3:].mean()),
 'old_net_parameters':18*24*9+24+24*24*9+24+24*6*9+6,
 'thermal_dT_d_logI_K':logJT.tolist(),
 'temperature_error_first_order_700_plus1pct_905_minus1pct_K':float(logJT@np.array([.01,0,-.01])),
 'temperature_error_first_order_common1pct_K':float(logJT@np.full(3,.01)),
 'visible_continuum_J_e_per_thermal_e':Jcv.tolist(),
 'supported_fourier_bins':int(support.sum()),
 'six_band_real_bandlimited_coordinate_count_approx':int(6*support.sum()),
 'DC_prior_proposal_retention_eigenvalues':np.linalg.eigvals(retention).real.tolist(),
 'DC_max_offdiagonal_abs_in_proposal_retention':float(np.max(np.abs(retention-np.diag(np.diag(retention)))))
}
(root/'analytic_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
