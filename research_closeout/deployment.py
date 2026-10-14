"""Raw plus known calibration interface for the frozen research prototype."""
from . import storage as s
from . import measurement as m
from .baselines import FrozenBaselines
from contextlib import contextmanager
import argparse
import copy
import json
import numpy as np
import torch

METHODS=('v3_physics','old_B4','B1_lambda100')

@contextmanager
def known_exposure(physics,ratio):
    """Change incident electrons, preserving ADC/read units and spectral ratios."""
    if ratio<=0: raise ValueError('Positive known exposure ratio required')
    old_lut=physics.continuum_lut;old_s=physics.Sscale;old_b=physics.Bscale;old_cfg=physics.cfg
    physics.continuum_lut=old_lut*ratio;physics.Sscale=old_s*ratio;physics.Bscale=old_b*ratio
    physics.cfg={**old_cfg,'exposure_s':old_cfg['exposure_s']*ratio,
        'reference_905_e':old_cfg['reference_905_e']*ratio,
        'species_scale_e':old_cfg['species_scale_e']*ratio,'background_scale_e':old_cfg['background_scale_e']*ratio}
    try: yield physics
    finally:
        physics.continuum_lut=old_lut;physics.Sscale=old_s;physics.Bscale=old_b;physics.cfg=old_cfg

class FrozenPrototype:
    def __init__(self,device='cuda'):
        selected=s.read(s.REPORTS/'selected_method.json')
        if selected['method']!='v3_physics': raise ValueError('This registered controller requires the selected frozen v3 route')
        self.suite=FrozenBaselines(device);self.physics=self.suite.p
        self.qualified=bool(selected['deployment_qualified'])
    @torch.no_grad()
    def infer(self,raw_dn,method='v3_physics',*,photon_ratio=1.):
        # Only the observed raw and known calibration cross this boundary.
        # No case ID, condition, geometry, seed, label or truth argument exists.
        if method=='old_B4' and self.suite.old_model is None:
            from infer import load_model,CHECKPOINT_SHA256
            path=s.LEGACY/'checkpoints/selected_checkpoint.pt';assert s.sha(path)==CHECKPOINT_SHA256
            self.suite.old_model=load_model(path,device=self.physics.M.device)
        p=self.suite.old_model.physics if method=='old_B4' else self.physics
        with known_exposure(p,float(photon_ratio)):
            if method=='v3_physics': out=self.suite.v3(raw_dn)
            elif method=='old_B4': out=self.suite.old_B4(raw_dn)
            elif method.startswith('B1_lambda'):
                lam=float(method.removeprefix('B1_lambda'))
                if lam not in (.1,100.): raise ValueError('Reporting traditional configuration is frozen')
                out=self.suite.traditional(raw_dn,lam)
            else: raise ValueError('Unregistered frozen reporting method')
        out['template_postblur_e_ref']=out['template_postblur_e_ref']/photon_ratio
        out['continuum_postblur_bands']=out['continuum_postblur_bands']/photon_ratio
        out['numerical_valid']=False  # Retain the failed production qualification.
        out['calibration_status']='calibration_assumed_not_verified'
        out['detection_status']='THRESHOLD_NOT_ATTACHED'
        out['uncertainty_status']='NOT_COMPUTED'
        out['info'].update(known_photon_ratio=float(photon_ratio),ADC_read_units_rescaled=False,
            template_unit='electron coefficient at reference20us',research_baseline_only=True,
            production_qualified=self.qualified if method=='v3_physics' else False,
            checkpoint_or_estimator_reselected=False)
        return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True);parser.add_argument('--calibration',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();cal=s.read(args.calibration)
    nominal=m.selection()['physics_config']
    if cal['physics_config']!=nominal: raise ValueError('Unsupported unverified instrument calibration')
    raw=np.load(args.raw,allow_pickle=False);engine=FrozenPrototype(args.device)
    out=engine.infer(raw,photon_ratio=float(cal.get('known_photon_ratio',1.)))
    path=s.write_path(args.output)
    receipt=s.save(path,**{k:out[k] for k in ('temperature_color_K','template_postblur_e_ref',
        'continuum_postblur_bands','predicted_raw_e','valid','low_signal')})
    s.write(path.with_suffix('.json'),{'result':receipt,'status':out['info'],
        'calibration_status':out['calibration_status'],'numerical_valid':out['numerical_valid'],
        'uncertainty_status':out['uncertainty_status'],'real_experimental_validation':False})
    print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
