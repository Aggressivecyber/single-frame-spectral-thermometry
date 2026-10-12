"""Independent validation-only decoder smoothing; never changes the backbone."""
import numpy as np
import torch
from infer import load_model
from evaluation_v2.methods import demod_fit
from training_v4_1.common import *
from training_v4_1.evaluate import score_arrays

def select():
    setup()
    if (OUT/'tikhonov_selection.json').exists(): return read(OUT/'tikhonov_selection.json')
    rows=[r for r in read(OUT/'data_manifest.json')['rows'] if r['split']=='validation' and r['input_kind']=='noise' and not r['signed_aux']]
    p=load_model(device='cuda').physics; journal=Journal(OUT/'tikhonov_validation_grid.jsonl')
    previous={r['identity']:r for r in journal.rows()}; summaries=[]
    with torch.no_grad():
        for lam in PROTOCOL['tikhonov_grid']:
            results=[]
            for row in rows:
                identity=canonical_hash({'raw':row['raw_hash'],'label':row['label_sha256'],'lambda':lam,
                    'camera':row['physics_config_sha256'],'method_source':runtime.sha256(runtime.ROOT/'evaluation_v2/methods.py')})
                if identity in previous: record=previous[identity]
                else:
                    raw=torch.as_tensor(npz(file_path(row,'raw_path'))['raw_dn'].astype(np.float32),device='cuda')[None]
                    pred=demod_fit(p,p.raw_to_electrons(raw),regularization=lam)
                    metrics=score_arrays(pred['species_signed_e'][0,2].cpu().numpy(),pred['temperature_color_K'][0].cpu().numpy(),npz(file_path(row,'label_path')))
                    record={'identity':identity,'case_id':row['case_id'],'raw_hash':row['raw_hash'],'lambda_decoder':lam,
                        'condition':row['condition'],'mother_id':row['mother_id'],'metrics':metrics,'status':'SUCCESS'}
                    journal.append(record)
                results.append(record)
            loss=float(np.mean([r['metrics']['template_MSE_e2']/3600+.2*r['metrics']['temperature_MSE_K2']/5625 for r in results]))
            groups=[]
            for condition in ('blank','weak','normal','strong'):
                rs=[r for r in results if r['condition']==condition]
                energy=sum(r['metrics']['truth_energy'] for r in rs); sse=sum(r['metrics']['template_SSE'] for r in rs)
                groups.append({'condition':condition,'frames':len(rs),'template_RMSE_e':float(np.sqrt(sse/sum(r['metrics']['pixels'] for r in rs))),
                    'template_L2':float(np.sqrt(sse/energy)) if energy else None,
                    'mean_gradient_energy':float(np.mean([r['metrics']['gradient_energy'] for r in rs])),
                    'mean_truth_gradient_energy':float(np.mean([r['metrics']['truth_gradient_energy'] for r in rs])),
                    'mean_signed_projection':float(np.mean([r['metrics']['signed_projection'] for r in rs])) if energy else None})
            fy,fx=torch.meshgrid(torch.fft.fftfreq(256,device='cuda'),torch.fft.fftfreq(256,device='cuda'),indexing='ij')
            passband=p.op.decode_window.bool(); transfer=(1+lam*(4*torch.sin(torch.pi*fx).square()+4*torch.sin(torch.pi*fy).square())).reciprocal()
            summaries.append({'lambda_decoder':lam,'validation_objective':loss,'groups':groups,
                'retained_passband_transfer_min':float(transfer[passband].min()),'retained_passband_transfer_median':float(transfer[passband].median()),
                'reference_note':'gradient energy and projection jointly describe smoothing/detail loss; noisy gradient energy alone is not recovered detail'})
            print({'decoder_lambda':lam,'validation_score':loss},flush=True)
    best=min(summaries,key=lambda r:r['validation_objective'])
    result={'selected_lambda_decoder':best['lambda_decoder'],'selection_rows':len(rows),'validation_only':True,
        'criterion':'absolute template MSE/60^2 +0.2 color MSE/75^2, main noisy validation only',
        'grid':summaries,'physical_solver_template_coefficient_unchanged':.1,'same_coefficient_as_physical_solver':False,'final_test_used':False}
    write(OUT/'tikhonov_selection.json',result); return result

if __name__=='__main__': select()
