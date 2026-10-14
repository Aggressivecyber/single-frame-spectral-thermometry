"""External fixed-domain diagnostics; no truth is passed to source inference."""
from __future__ import annotations

import math
from collections import defaultdict
from . import common
common.configure_runtime()
import numpy as np
import torch


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def metadata(ref, source, beta_mode):
    coefficient = float(ref['coefficient_e'])
    return {**{key: ref.get(key) for key in ('case_id', 'mother_id', 'cohort',
             'input_kind', 'group', 'condition', 'partition_role', 'coefficient_e')},
            'sign': 'positive' if coefficient > 0 else 'negative' if coefficient < 0 else 'blank',
            'method': source + '__' + beta_mode, 'continuum_source': source,
            'beta_mode': beta_mode, 'raw_sha256': ref['raw_reference']['sha256'],
            'raw_array_sha256': ref['raw_reference'].get('array_sha256'),
            'diagnostic_oracle': source == 'C_true' or beta_mode == 'true_beta'}


def continuum_result(physics, latent, *, valid=None, info=None):
    post = physics.op.object_blur(latent)
    temperature, model_valid = physics.color_temperature(post)
    return {'continuum_latent': latent, 'continuum_postblur': post,
            'temperature_color_K': temperature,
            'valid': model_valid if valid is None else model_valid & valid,
            'info': info or {}}


def source_diagnostics(physics, source, continuum_true, weights, domain):
    target = physics.op.object_blur(continuum_true)
    error = source['continuum_postblur'] - target
    mask = torch.as_tensor(domain, device=error.device, dtype=torch.bool).reshape(error.shape[-2:])
    raw_error = physics.op(continuum_true-source['continuum_latent'])
    results = {'postblur_semantics': True, 'object_blur_count': 1,
               'evaluation_domain_pixels': int(mask.sum()),
               'continuum_raw_error_RMSE_e': float(raw_error.square().mean().sqrt()),
               'continuum_raw_error_fixed_W_RMS': float((weights*raw_error.square()).mean().sqrt()),
               'visible_continuum_RMSE_e': float(error[:, :5, mask].square().mean().sqrt()),
               'visible_continuum_SSE_e2': float(error[:, :5, mask].square().sum()),
               'visible_continuum_pixels': 5*int(mask.sum()),
               'bands': {}}
    for k, label in enumerate(('visible_0','visible_1','visible_2','visible_3','visible_4','700','905')):
        selected = error[:, k, mask]
        results['bands'][label] = {'pixels': int(selected.numel()),
              'SSE_e2': float(selected.square().sum()), 'bias_e': float(selected.mean()),
              'RMSE_e': float(selected.square().mean().sqrt())}
    return results


def score(ref, name, beta_mode, source, linear, raw_e, truth, physics, *, seconds, memory=None):
    row = metadata(ref, name, beta_mode)
    frame = linear['info']['frames'][0]
    solved = bool(frame['accepted'] and frame['CG']['converged'] and frame['expected_e_nonnegative'])
    result = {**source, **linear, 'raw_e': raw_e, 'valid': source['valid'] & solved}
    scored = common.frozen_v5('metrics').score_prediction(result, truth, physics)
    row.update(plain(scored))
    row['diagnostics'] = {'seconds': float(seconds), 'CG': plain(frame['CG']),
           'accepted': frame['accepted'], 'expected_e_nonnegative': frame['expected_e_nonnegative'],
           'objective_before': frame['objective_before'], 'objective_after': frame['objective_after'],
           'objective_terms_after': frame['objective_terms_after'],
           'lambda_g': frame['lambda_g'], 'lambda0': frame['lambda0'], 'lambda_beta': frame['lambda_beta'],
           'memory': memory, 'raw_likelihood_count': 1,
           'cold_inference_latency_claimed': False}
    if not solved:
        row.update(status='LINEAR_SOLVE_FAILED', numerical_execution_passed=False)
    return plain(row)


def score_v3(ref, output, raw_e, truth, physics, *, seconds, memory=None):
    result = {**output, 'raw_e': raw_e}
    value = common.frozen_v5('metrics').score_prediction(result, truth, physics)
    row = {**metadata(ref, 'original_v3', 'joint_original'), **plain(value),
           'diagnostics': {'seconds': float(seconds), 'memory': memory, 'CG': None,
                'CG_unavailable_reason': 'Frozen v3 nonlinear solver reports original history separately',
                'original_v3_info': plain(output.get('info', {}))},
           'diagnostic_oracle': False}
    return row


def failure(ref, source, beta_mode, error, *, kind):
    return {**metadata(ref, source, beta_mode), 'status': kind,
            'numerical_execution_passed': False, 'metrics': None,
            'error_type': type(error).__name__, 'error': str(error), 'diagnostics': {}}


def aggregate(rows):
    return common.frozen_v5('statistics').aggregate_metrics(rows)


def paired(rows, candidate, reference, *, group='normal', sign=None, field='template'):
    """Full fixed case pairing and mother resampling for either primary target."""
    selected = [r for r in rows if r['input_kind']=='noise' and r['cohort']=='development'
                and (group is None or r['group']==group) and (sign is None or r['sign']==sign)]
    left = {r['case_id']:r for r in selected if r['method']==candidate}
    right = {r['case_id']:r for r in selected if r['method']==reference}
    key = 'template_SSE' if field=='template' else 'temperature_SSE_K2'
    complete = set(left)==set(right) and bool(left)
    by_mother = defaultdict(list)
    paired_mse = defaultdict(list)
    for cid in sorted(set(left)|set(right)):
        l,r=left.get(cid),right.get(cid)
        if l and r:
            assert l['raw_sha256']==r['raw_sha256'] and l['mother_id']==r['mother_id']
        lm,rm=(l or {}).get('metrics') or {},(r or {}).get('metrics') or {}
        a,b=lm.get(key),rm.get(key)
        n,m=lm.get('domain_pixels',0),rm.get('domain_pixels',0)
        delta=a/n-b/m if a is not None and b is not None and n>0 and m>0 else None
        if delta is None: complete=False
        by_mother[(l or r)['mother_id']].append(delta)
        paired_mse[(l or r)['mother_id']].append((a/n,b/m) if delta is not None else None)
    mothers={k:sum(v)/len(v) if all(x is not None for x in v) else None for k,v in by_mother.items()}
    boot=common.frozen_v5('statistics').mother_bootstrap(mothers)
    ratio = {'estimate': None, 'CI95': None}
    if complete:
        values=np.asarray([np.mean(paired_mse[mid],axis=0) for mid in sorted(paired_mse)])
        rng=np.random.default_rng(53130)
        means=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
        if (means[:,1]>0).all() and values[:,1].mean()>0:
            ratio={'estimate':float(np.sqrt(values[:,0].mean()/values[:,1].mean())),
                   'CI95':np.quantile(np.sqrt(means[:,0]/means[:,1]),[.025,.975]).tolist()}
    return {'candidate':candidate,'reference':reference,'field':field,'group':group,'sign':sign,
            'case_pairs':len(set(left)|set(right)), 'complete':bool(complete and boot['complete']),
            'mother_MSE_difference':boot, 'mother_RMSE_ratio':ratio,
            'same_raw_hashes_verified':True,
            'exclusion_of_failed_cases':False}


def grouped(rows):
    groups=defaultdict(list)
    keys=('cohort','input_kind','group','sign','method')
    for row in rows:groups[tuple(row.get(k) for k in keys)].append(row)
    return [{**dict(zip(keys,key)),**aggregate(value)} for key,value in sorted(groups.items())]
