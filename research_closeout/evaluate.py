"""Root-owned CLI for the registered closeout; no global stop on candidate failure."""
from . import storage as s
from . import measurement as m
from . import experiment_registry as registry
import argparse
import json
import os
import time
import traceback
import torch

def require_foundation():
    reg=s.read(s.REPORTS/'FOUNDATION_REGISTRATION.json')
    for name,digest in reg['source_sha256'].items():
        assert s.sha(s.ROOT/name)==digest,'Registered foundation source changed: '+name
    assert s.sha(s.ROOT/'research_closeout/execution_spec.yaml')==reg['spec_sha256']
    assert s.sha(s.ROOT/'research_v6/input_inventory.json')==reg['raw_manifest_sha256']
    return reg

@torch.no_grad()
def representation_checks(device):
    from research_v6 import inputs
    from .source_study import sources_for
    s.frozen_bridge()
    spec=s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    refs=s.read(s.ROOT/'research_v6/input_inventory.json')['rows']
    selected=[r for r in refs if r['cohort']=='development' and r['input_kind']=='noise'
        and r['mother_id'] in spec['development']['selection_mothers']]
    p32=m.physics(device,dtype=torch.float32);p=m.physics(device)
    rows=[]
    for ref in selected:
        raw=inputs.load_raw(ref,p32);source=sources_for(ref,raw,p32)['C_v3'];state=source['state'].double()
        native=m.continuum(p,state[:,:2]);native_post=p.op.object_blur(native)
        native_color,_=p.color_temperature(native_post);native_raw=p.op(native)
        for g in spec['development']['representation_grids']:
            coarse=torch.nn.functional.interpolate(state[:,:2],size=(g,g),mode='bilinear',align_corners=False)
            C=m.continuum(p,coarse,grid=g);Cpost=p.op.object_blur(C);color,_=p.color_temperature(Cpost)
            color_error=float((color-native_color).square().mean().sqrt())
            raw_error=float((p.op(C)-native_raw).square().mean().sqrt())
            rows.append({'case_id':ref['case_id'],'mother_id':ref['mother_id'],'grid':g,
                'raw_sha256':ref['raw_reference']['sha256'],'color_RMS_difference_K':color_error,
                'raw_RMS_difference_e':raw_error,'reference':'same frozen-v3 state, native256 nonlinear continuum',
                'truth_used':False,'passed':color_error<=spec['development']['representation_color_budget_K']
                     and raw_error<=spec['development']['representation_raw_budget_e']})
    acceptable=[g for g in spec['development']['representation_grids'] if all(r['passed'] for r in rows if r['grid']==g)]
    if not acceptable: raise ArithmeticError('No globally valid continuum representation')
    result={'time':s.now(),'cases':len(selected),'mothers':12,'selected_global_grid':min(acceptable),
        'rows':rows,'test_truth_used_for_selection':False,'scientific_improvement_claimed':False}
    s.write(s.REPORTS/'CONTINUUM_REPRESENTATION.json',result)
    return result

def cost_pilot(device,grid):
    from .joint_estimator import infer
    from .source_study import sources_for
    from research_v6 import inputs
    s.frozen_bridge()
    case_id='v5_development_selection_mother_0000_positive_normal_noise'
    ref=next(r for r in s.read(s.ROOT/'research_v6/input_inventory.json')['rows'] if r['case_id']==case_id)
    p32=m.physics(device,dtype=torch.float32);p=m.physics(device)
    raw=inputs.load_raw(ref,p32);v3=sources_for(ref,raw,p32)['C_v3'];truth=inputs.load_diagnostic_truth(ref,p32)
    rows=[]
    for initial in ('v3','smooth_demod'):
        registry.count_call('development','joint_cost_pilot_'+initial,case_id)
        started=time.perf_counter()
        try:
            out=infer(raw,p,{'grid':grid},initializer=initial,v3_state=v3['state'] if initial=='v3' else None)
            domain=truth['evaluation_domain'].reshape(p.cfg['shape']).bool()
            template_target=truth['template_postblur_e'].reshape(p.cfg['shape']).double()
            temp_key='temperature_reference_color_K' if 'temperature_reference_color_K' in truth else 'temperature_color_K'
            temp_target=truth[temp_key].reshape(p.cfg['shape']).double()
            row={'initializer':initial,'case_id':case_id,'raw_sha256':ref['raw_reference']['sha256'],
                'status':out['info']['status'],'info':s.plain(out['info']),
                'template_RMSE_e':float((out['template_postblur_e_ref'][0,domain]-template_target[domain]).square().mean().sqrt()),
                'color_RMSE_K':float((out['temperature_color_K'][0,domain]-temp_target[domain]).square().mean().sqrt()),
                'scoring_pixels':int(domain.sum()),'truth_passed_to_inference':False,
                'wall_s':time.perf_counter()-started,'source_grid':grid}
        except Exception as error:
            row={'initializer':initial,'case_id':case_id,'raw_sha256':ref['raw_reference']['sha256'],
                'status':'NUMERICAL_OR_COMPLEXITY_UNCERTIFIED','error_type':type(error).__name__,'error':str(error),
                'wall_s':time.perf_counter()-started,'source_grid':grid,'failure_kept_in_denominator':True}
            s.append(s.REPORTS/'failures.jsonl',row)
        rows.append(row);print('closeout pilot '+initial+' '+row['status'],flush=True)
    result={'time':s.now(),'rows':rows,'finite_candidate_grid_not_yet_executed':True,
        'method_selection_not_yet_frozen':True,'final_payloads_opened':False,
        'new_candidate_numeric_eligibility':all(r['status']=='SUCCESS' for r in rows),
        'failure_does_not_stop_independent_baseline_or_paper_work':True}
    s.write(s.REPORTS/'JOINT_COST_PILOT.json',result)
    return result

def foundation(device):
    require_foundation()
    from .source_study import run
    result={'time':s.now(),'phase':'foundation','status':'RUNNING','final_payloads_opened':False}
    try:
        result['source_study']=run(device)
    except Exception as error:
        result['source_study']={'status':'MODULE_REQUIRES_DIAGNOSIS','error_type':type(error).__name__,'error':str(error)}
        s.append(s.REPORTS/'failures.jsonl',{'phase':'source_study',**result['source_study']})
    try:
        rep=representation_checks(device);result['representation_grid']=rep['selected_global_grid']
        result['joint_cost_pilot']=cost_pilot(device,rep['selected_global_grid'])
    except Exception as error:
        result['joint_cost_pilot']={'status':'MODULE_REQUIRES_DIAGNOSIS','error_type':type(error).__name__,'error':str(error)}
        s.append(s.REPORTS/'failures.jsonl',{'phase':'joint_cost_pilot',**result['joint_cost_pilot']})
    result.update(status='FOUNDATION_TERMINAL_CONTINUE_CLOSEOUT',finished_at=s.now(),
        still_required=['finite development selection and strong B1','production numerical comparisons',
            'independent renderer and boundaries','full-inverse uncertainty and null calibration',
            'freeze selected method and then final100 mothers','seven figure groups, Chinese manuscript and release'])
    s.write(s.REPORTS/'FOUNDATION_RESULT.json',s.plain(result))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('register','foundation'),required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.phase=='register': result=registry.register()
    else:
        active=s.write_path(s.REPORTS/'PHASE_ACTIVE.json');active.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(active,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w',encoding='utf8') as f: json.dump({'pid':os.getpid(),'phase':args.phase,'time':s.now()},f)
        s.append(s.REPORTS/'progress.jsonl',{'time':s.now(),'phase':args.phase,'status':'STARTED','pid':os.getpid(),'device':args.device})
        try: result=foundation(args.device)
        finally: active.unlink()
    print(json.dumps(s.plain(result),ensure_ascii=True),flush=True)

if __name__=='__main__': main()
