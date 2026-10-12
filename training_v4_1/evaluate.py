import time
import numpy as np
import torch
from collections import defaultdict
from training_v4.model import physical_output
from training_v4.train import protect_development_gates
from training_v4_1.cache import load
from training_v4_1.model import protective_loss
from training_v4_1.common import *

def score_arrays(S, T, truth):
    y = np.asarray(truth['template_e'],np.float64); S=np.asarray(S,np.float64)
    error=S-y; energy=float(np.square(y).sum()); pixels=y.size
    support=np.abs(y)>=.2*float(np.abs(y).max()) if energy else np.zeros_like(y,dtype=bool)
    pos=support & (y>0); neg=support & (y<0)
    te=None if T is None else np.asarray(T,np.float64)-truth['temperature_color_K']
    return {'template_SSE':float(np.square(error).sum()),'truth_energy':energy,'pixels':pixels,
        'template_MSE_e2':float(np.square(error).mean()), 'template_RMSE_e':float(np.sqrt(np.square(error).mean())),
        'template_L2':float(np.sqrt(np.square(error).sum()/energy)) if energy else None,
        'temperature_MSE_K2':float(np.square(te).mean()) if te is not None else None,
        'color_RMSE_K':float(np.sqrt(np.square(te).mean())) if te is not None else None,
        'positive_recall60':float((S[pos]>60).mean()) if pos.any() else None,
        'negative_recall60':float((S[neg]<-60).mean()) if neg.any() else None,
        'positive_blank_FP60':float((S>60).mean()) if not energy else None,
        'negative_blank_FP60':float((S< -60).mean()) if not energy else None,
        'signed_projection':float((S*y).sum()/energy) if energy else None,
        'gradient_energy':float(np.square(S-np.roll(S,1,0)).mean()+np.square(S-np.roll(S,1,1)).mean()),
        'truth_gradient_energy':float(np.square(y-np.roll(y,1,0)).mean()+np.square(y-np.roll(y,1,1)).mean())}

def wait_ready(row, timeout=14400):
    start=time.monotonic(); path=DATA/'physical_scores'/(row['case_id']+'.json')
    while not path.exists():
        status=read(OUT/'cache_producer_status.json')
        if status['status']=='FAILED': raise RuntimeError('Fixed cache producer failed: '+status['message'])
        if status['status']=='COMPLETE' or time.monotonic()-start>timeout: raise RuntimeError('Missing matched cache '+row['case_id'])
        time.sleep(.5)

def summarize(records):
    groups=[]
    for kind, signed in [(k,s) for k in ('noise','continuous','quantization') for s in (False,True)]:
        for group in ('blank','weak','normal','strong','negative'):
            rs=[r for r in records if r['input_kind']==kind and r['group']==group and r['signed_aux']==signed]
            if not rs: continue
            good=[r for r in rs if r['status']=='SUCCESS']; n=sum(r['metrics']['pixels'] for r in good)
            energy=sum(r['metrics']['truth_energy'] for r in good); sse=sum(r['metrics']['template_SSE'] for r in good)
            groups.append({'input_kind':kind,'signed_aux':signed,'group':group,'frames':len(rs),'failures':len(rs)-len(good),
                'template_RMSE_e':float(np.sqrt(sse/n)) if n else None,
                'template_L2':float(np.sqrt(sse/energy)) if energy else None,
                'color_RMSE_K':float(np.sqrt(np.mean([r['metrics']['temperature_MSE_K2'] for r in good]))) if good else None,
                'physical_color_RMSE_K':float(np.sqrt(np.mean([r['physical']['temperature_MSE_K2'] for r in good]))) if good else None,
                'mean_template_excess_e2':float(np.mean([r['metrics']['template_MSE_e2']-r['physical']['template_MSE_e2'] for r in good])) if good else None,
                'mean_temperature_excess_K2':float(np.mean([r['metrics']['temperature_MSE_K2']-r['physical']['temperature_MSE_K2'] for r in good])) if good else None})
    return groups

def evaluate(model, rows, selection, folder, update, safe_weight):
    model.eval(); journal=Journal(folder/f'validation_update_{update:04d}.jsonl')
    assert not journal.path.exists(), 'Validation artifacts are immutable'
    records=[]; total=[]
    with torch.no_grad():
        for row in rows:
            wait_ready(row); inputs, truth, baseline=load(row,selection)
            out=model(*inputs)
            loss,terms=protective_loss(model,inputs[0],out,truth,baseline,safe_weight)
            S=out['template_e'][0].cpu().numpy(); T=out['temperature_color_K'][0].cpu().numpy()
            if not (np.isfinite(S).all() and np.isfinite(T).all() and torch.isfinite(loss)):
                journal.append({'case_id':row['case_id'],'raw_hash':row['raw_hash'],'status':'NONFINITE',
                    'optimizer_update':update,'failed_frame_retained':True})
                raise FloatingPointError('Nonfinite validation frame '+row['case_id'])
            metrics=score_arrays(S,T,{k:v[0].cpu().numpy() for k,v in truth.items()})
            phys=physical_output(model.physics,inputs[1],inputs[2],model.physics.raw_to_electrons(inputs[0]))
            record={k:row[k] for k in ('case_id','mother_id','condition','group','geometry','input_kind','signed_aux','raw_hash','label_sha256')}
            record.update(status='SUCCESS',metrics=metrics,physical=baseline,optimizer_update=update,
                main_loss=float(terms['main']),safe_template=float(terms['safe_template']),safe_temperature=float(terms['safe_temperature']),
                template_correction_RMS_e=float((out['template_e']-phys['template_e']).square().mean().sqrt()),
                temperature_correction_RMS_K=float((out['temperature_color_K']-phys['temperature_color_K']).square().mean().sqrt()),
                invalid_output_fraction=float((~out['valid']).float().mean()))
            journal.append(record); records.append(record); total.append(float(terms['main']))
    gates_pass,gates=protect_development_gates(model,selection)
    write(folder/f'gates_update_{update:04d}.json',{'passed':gates_pass,'decisions':gates})
    protection=defaultdict(list)
    for r in records:
        if r['input_kind']!='noise': protection[(r['geometry'],r['condition'],r['input_kind'])].append(r)
    decisions=[]
    for key,rs in protection.items():
        ds=float(np.mean([r['metrics']['template_MSE_e2']-r['physical']['template_MSE_e2'] for r in rs]))
        dt=float(np.mean([r['metrics']['temperature_MSE_K2']-r['physical']['temperature_MSE_K2'] for r in rs]))
        decisions.append({'geometry':key[0],'condition':key[1],'input_kind':key[2],'frames':len(rs),
            'mean_template_MSE_delta':ds,'mean_temperature_MSE_delta':dt,
            'passed':ds<=PROTOCOL['safe_template_tolerance_e2'] and dt<=PROTOCOL['safe_temperature_tolerance_K2']})
    groups=summarize(records)
    weak=next(g for g in groups if g['input_kind']=='noise' and g['group']=='weak' and not g['signed_aux'])
    normal=next(g for g in groups if g['input_kind']=='noise' and g['group']=='normal' and not g['signed_aux'])
    temp_ok=all(g['color_RMSE_K']**2 <= 1.05*g['physical_color_RMSE_K']**2+.04 for g in groups if g['input_kind']=='noise')
    eligible=gates_pass and all(d['passed'] for d in decisions) and weak['template_L2']<1 and temp_ok
    small_large=[r for r in records if r['input_kind']!='noise' and r['physical']['weighted_raw_RMS']<.2
                 and (r['template_correction_RMS_e']>2 or r['temperature_correction_RMS_K']>.5)]
    result={'optimizer_update':update,'validation_cases':len(rows),'main_validation_loss':float(np.mean(total)),
        'original_gates_pass':gates_pass,'original_gates_pass_count':sum(g['passed'] for g in gates),
        'protection_groups':decisions,'protection_pass':all(d['passed'] for d in decisions),
        'noisy_temperature_pass':temp_ok,'weak_L2':weak['template_L2'],'normal_RMSE_e':normal['template_RMSE_e'],
        'eligible':eligible,'small_residual_large_correction_fraction':len(small_large)/sum(r['input_kind']!='noise' for r in records),
        'groups':groups,'final_test_used':False}
    write(folder/f'validation_summary_{update:04d}.json',result)
    return result
