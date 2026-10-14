"""Same-raw source attribution using original K, including propagated solver residuals."""
from . import storage as s
from . import measurement as m
from . import experiment_registry as registry
from .error_accounting import summarize
import importlib
import sys
import types
import time
import gc
import torch

def precision_reference():
    c=s.frozen_bridge()
    pkg=types.ModuleType('research_v6.repair_001');pkg.__path__=[str(s.LEGACY/'research_v6/repair_001')]
    shim=types.ModuleType('research_v6.repair_001.storage')
    shim.ROOT=s.LEGACY;shim.DATA=s.DATA;shim.REPORTS=s.REPORTS
    sys.modules['research_v6.repair_001']=pkg;sys.modules['research_v6.repair_001.storage']=shim
    old=s.read(s.LEGACY/'research_v6/repair_001/protocol.json')
    for name,digest in old['source_sha256'].items():
        assert s.sha(s.LEGACY/name)==digest,'External registered precision source changed'
    solver=importlib.import_module('research_v6.repair_001.solver')
    return c,solver

def sources_for(ref,raw,p):
    c=s.frozen_bridge()
    from research_v6 import continuum_sources as f
    journal=s.LEGACY/'reports/research_v6/repair_001/source_inference_receipts.jsonl'
    if not hasattr(sources_for,'prior'):
        sources_for.prior={(r['case_id'],r['source']):r for r in s.rows(journal)}
    prior=sources_for.prior
    result={}
    for name in ('C_v3','C_v5'):
        entry=prior.get((ref['case_id'],name))
        if entry and 'state_cache' in entry:
            assert entry['raw_sha256']==ref['raw_reference']['sha256']
            receipt=entry['state_cache'];path=s.LEGACY/receipt['path'];assert s.sha(path)==receipt['sha256']
            values=s.arrays(path)
            state=torch.as_tensor(values['state'],device=p.M.device,dtype=p.M.dtype)
            beta=torch.as_tensor(values.get('beta',[[0,0,0]]),device=p.M.device,dtype=p.M.dtype)
            latent=p.components(state,beta)[0]
            result[name]={'continuum_latent':latent,'state':state,'beta':beta,'cache_receipt':receipt}
        elif name=='C_v3':
            selected=s.read(s.LEGACY/'research_v6/protocol.json')['v3_selection']
            result[name]=f.cached_v3(raw,p,selected,ref['C_v3_cache']) if ref.get('C_v3_cache') else f.infer_v3(raw,p,selected)
        else:
            result[name]=f.cached_v5(raw,p,ref['C_v5_cache']) if ref.get('C_v5_cache') else f.infer_v5(raw,p)
        if result[name].get('status')=='SOURCE_FAILED': raise RuntimeError(result[name]['error'])
    return result

@torch.no_grad()
def run(device='cuda'):
    c,old=precision_reference()
    from research_v6 import inputs
    spec=s.read(s.ROOT/'research_closeout/execution_spec.yaml')
    refs=s.read(s.ROOT/'research_v6/input_inventory.json')['rows']
    refs=[r for r in refs if r['case_id'] in spec['source_cases']]
    order={name:i for i,name in enumerate(spec['source_order'])}
    refs.sort(key=lambda r:(order[r['input_kind']],r['mother_id'],r['condition']))
    path=s.REPORTS/'source_accounting.jsonl'
    done={(r['case_id'],r['source'],r['beta_mode']) for r in s.rows(path)} if path.exists() else set()
    p=m.physics(device,dtype=torch.float32);options=s.read(s.LEGACY/'research_v6/protocol.json')['linear_options']
    original_rtol,original_maxiter=old.RTOL,old.MAXITER
    def propagate(system,rhs):
        old.RTOL=spec['source_residual_reference_rtol'];old.MAXITER=spec['source_residual_reference_maxiter']
        try: return old.solve_rhs(system,rhs,'propagated_numerical_residual_reference')
        finally: old.RTOL=original_rtol;old.MAXITER=original_maxiter
    for index,ref in enumerate(refs):
        s.budget_guard();started=time.perf_counter()
        raw=inputs.load_raw(ref,p);raw_e=p.raw_to_electrons(raw)
        truth=inputs.load_diagnostic_truth(ref,p);src=sources_for(ref,raw,p)
        src['C_true']={'continuum_latent':truth['continuum_latent']}
        W=c.frozen_v5('signed_linear_solver').fixed_raw_weights(p,raw_e)
        for mode in ('true_beta','estimated_beta'):
            system=old.build_system(p,raw_e,torch.zeros((1,7,*p.cfg['shape']),device=p.M.device),{**options,'weights':W},
                fixed_beta=truth['beta'] if mode=='true_beta' else None)
            pp=system['physics'];G=system['linear_operator'];weight=system['weights']
            st=truth['template_latent_e'].reshape(p.cfg['shape']).double();bt=truth['beta'].reshape(3).double()
            ct=truth['continuum_latent'].reshape(1,7,*p.cfg['shape']).double();xt=system['pack'](st,None if mode=='true_beta' else bt)
            noise=system['raw_e']-pp.op(ct)[0]-G.apply_template(st)-G.apply_beta(bt)
            nrhs=system['measurement_adjoint'](weight*noise);brhs=-system['regularizer'](xt)
            xn,ninfo=old.solve_rhs(system,nrhs,'noise');xb,binfo=old.solve_rhs(system,brhs,'bias')
            cn,cninfo=propagate(system,nrhs-system['matvec'](xn))
            cb,cbinfo=propagate(system,brhs-system['matvec'](xb))
            for name in ('C_true','C_v5','C_v3'):
                if (ref['case_id'],name,mode) in done: continue
                registry.count_call('source',name+'__'+mode,ref['case_id'])
                out=old.solve(system,src[name]['continuum_latent'])
                lrhs=system['measurement_adjoint'](weight*pp.op(ct-src[name]['continuum_latent'].double())[0])
                xl,linfo=old.solve_rhs(system,lrhs,'continuum_leakage')
                cl,clinfo=propagate(system,lrhs-system['matvec'](xl))
                ra=out['rhs']-system['matvec'](out['vector']);ca,cainfo=propagate(system,ra)
                algebra=out['rhs']-system['matvec'](xt)-lrhs-nrhs-brhs
                cr,crinfo=propagate(system,algebra)
                def observable(vector): return pp.op.object_blur(system['unpack'](vector)[0])
                terms={'continuum_leakage':observable(xl+cl),'shared_measurement_noise':observable(xn+cn),
                    'regularization_bias':observable(xb+cb),'actual_numerical_error':-observable(ca)}
                label_round=pp.op.object_blur(st)-truth['template_postblur_e'].reshape(p.cfg['shape']).double()
                err=out['template_postblur_e'][0]-truth['template_postblur_e'].reshape(p.cfg['shape']).double()
                closure=err-sum(terms.values())-observable(cr)-label_round
                infos=[ninfo,binfo,linfo,cninfo,cbinfo,clinfo,cainfo,crinfo,out['info']['frames'][0]['CG']]
                certified=all(v['converged'] for v in infos) and float(closure.square().mean().sqrt())<.001
                row={'case_id':ref['case_id'],'mother_id':ref['mother_id'],'source':name,'beta_mode':mode,
                    'condition':ref['condition'],'input_kind':ref['input_kind'],'raw_sha256':ref['raw_reference']['sha256'],
                    'status':'SUCCESS' if certified else 'NUMERICAL_UNCERTIFIED','numerical_certified':certified,
                    'source_is_oracle':name=='C_true','beta_is_oracle':mode=='true_beta',
                    'actual_output_error_RMS_e':float(err.square().mean().sqrt()),
                    'four_terms':summarize(terms),'corrected_four_term_closure_RMS_e':float(closure.square().mean().sqrt()),
                    'label_roundoff_RMS_e':float(label_round.square().mean().sqrt()),
                    'algebra_roundoff_RMS_e':float(observable(cr).square().mean().sqrt()),
                    'component_residual_correction_RMS_e':{k:float(observable(v).square().mean().sqrt()) for k,v in [('leakage',cl),('noise',cn),('bias',cb)]},
                    'solver_diagnostics':infos,'original_K_W_lambda_preserved':True,
                    'objective':system['precision_contract'],'truth_use':'external labeled mechanism experiment only'}
                s.append(path,s.plain(row));done.add((ref['case_id'],name,mode))
                if not certified: s.append(s.REPORTS/'failures.jsonl',s.plain(row))
        s.append(s.REPORTS/'progress.jsonl',{'time':s.now(),'phase':'source_accounting','completed_inputs':index+1,'planned_inputs':90,'wall_s':time.perf_counter()-started})
        print(f'closeout source {index+1}/90 {ref["case_id"]}',flush=True)
        gc.collect()
    rows=list(s.rows(path));return {'status':'SOURCE_STUDY_COMPLETE','planned_rows':540,'actual_rows':len(rows),
        'certified_rows':sum(r['numerical_certified'] for r in rows),'numerical_failures_do_not_stop_baselines':True}
