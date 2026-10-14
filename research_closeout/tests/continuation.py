"""Regression checks for the observed reference-budget and input-identity faults."""
from .. import storage as s
from .. import measurement as m
from ..source_study import precision_reference
from ..error_accounting import propagate_residual
from ..evaluate import pilot_reference
import torch

def run():
    torch.set_num_threads(2);c,old=precision_reference();p=m.physics('cpu',dtype=torch.float32)
    yy,xx=p.background_basis[2],p.background_basis[1]
    raw=5000+500*torch.cos(xx*2)[None]
    options=s.read(s.LEGACY/'research_v6/protocol.json')['linear_options']
    system=old.build_system(p,raw,torch.zeros((1,7,256,256)),options)
    field=(torch.sin(xx.double()*5)+torch.cos(yy.double()*8))*3
    beta=field.new_tensor([.3,.02,-.03]);known=system['pack'](field,beta)
    rhs=system['matvec'](known)
    solved,info=propagate_residual(system,rhs,rtol=1e-11,max_iterations=1000)
    error=system['physics'].op.object_blur(system['unpack'](solved-known)[0])
    output_error=float(error.square().mean().sqrt())
    ref=pilot_reference()
    checks=[{'name':'native256_reference_known_observable_solution','error_RMS_e':output_error,
        'limit_e':1e-6,'passed':output_error<=1e-6 and info['converged']},
        {'name':'frozen_API_200_cap_with_global_1000_budget','passed':bool(info['blocks']) and
            all(b['requested_max_iterations']<=200 for b in info['blocks']) and info['maximum_iterations']==1000},
        {'name':'pilot_registered_raw_identity','case_id':ref['case_id'],'raw_sha256':ref['raw_reference']['sha256'],
            'passed':ref['case_id'].endswith('_noise0') and ref['cohort']=='development'}]
    result={'time':s.now(),'all_passed':all(x['passed'] for x in checks),'checks':checks,
        'reference':info,'formal_or_sealed_raw_payloads_opened':False,
        'original_estimator_rtol':old.RTOL,'original_estimator_max_iterations':old.MAXITER,
        'source_sha256':{v.relative_to(s.ROOT).as_posix():s.sha(v) for v in (s.ROOT/'research_closeout').rglob('*.py')}}
    s.write(s.REPORTS/'NUMERICAL_CONTINUATION_CHECKS.json',s.plain(result))
    print(__import__('json').dumps(s.plain(result),ensure_ascii=True),flush=True)
    return result

if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
