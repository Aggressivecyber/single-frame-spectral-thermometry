"""Reuse the same frozen raw-only source algorithms, with audited cache replay."""
from . import storage as s
from .. import common as c,continuum_sources as f,scoring
import torch


def original_receipts():
    return {(r['case_id'],r['source']):r for r in c.jsonl(s.OLD_REPORTS/'source_inference_receipts.jsonl')}


def obtain(ref,name,raw,p,old_protocol):
    prior=original_receipts().get((ref['case_id'],name))
    if prior and 'state_cache' in prior:
        assert prior['raw_sha256']==ref['raw_reference']['sha256']
        saved=prior['state_cache'];assert c.sha(c.ROOT/saved['path'])==saved['sha256']
        values=c.arrays(c.ROOT/saved['path'])
        state=torch.as_tensor(values['state'],device=p.M.device,dtype=p.M.dtype)
        beta=torch.as_tensor(values['beta'],device=p.M.device,dtype=p.M.dtype) if 'beta' in values else p.M.new_zeros((1,3))
        if name=='C_v3':
            _,physical=f._v3_functions()
            meta={**prior['source_metadata'],'mode':'prior_same_raw_v6_state_replay',
                 'prior_state_cache':saved,'current_cold_inference_executed':False}
            result=f._v3_result(raw,p,state,beta,[],physical,meta)
        else:
            latent=p.components(state,beta)[0]
            initial_T,initial_valid=p.color_temperature(p.op.decode(p.raw_to_electrons(raw)))
            result=scoring.continuum_result(p,latent,valid=initial_valid)
            result.update(status='SUCCESS',state=state,beta=beta,source_metadata={
                **prior['source_metadata'],'mode':'prior_same_raw_v6_state_replay','prior_state_cache':saved,
                'current_cold_inference_executed':False})
    elif name=='C_v3':
        receipt=ref.get('C_v3_cache')
        result=f.cached_v3(raw,p,old_protocol['v3_selection'],receipt) if receipt else f.infer_v3(raw,p,old_protocol['v3_selection'])
    else:
        receipt=ref.get('C_v5_cache')
        result=f.cached_v5(raw,p,receipt) if receipt else f.infer_v5(raw,p,old_protocol['continuum_options'])
    if result.get('status')=='SOURCE_FAILED':raise RuntimeError(result.get('error','Source failed'))
    assert bool(torch.isfinite(result['continuum_latent']).all())
    return result
