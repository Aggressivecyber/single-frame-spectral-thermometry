"""Bounded endpoint diagnostics on predeclared training strata; zero updates."""
import time
from collections import defaultdict
import runtime
import numpy as np
import torch
from training_v4.common import read,write,parameters_hash
from training_v4_1.common import setup,OUT as OLD,OLD_OUT,rel
from training_v4_1.cache import load
from training_v4_1.model import protective_loss
from training_v4_1.evaluate import score_arrays
from training_v4_2.analyze import strata
from training_v4_3.residual_scaling import ResidualScaledCNN
from evaluation_v2.common import Journal,canonical_hash

OUT=runtime.ROOT/'reports/training_v4_3/run_001/gradient_diagnostic'
ARMS=('residual_scaled','negative_balanced')


def cosine(a,b):
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom>0 else None


def run():
    setup(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'case_gradients.jsonl').exists(), 'Never overwrite completed or partial diagnostics'
    plan=read(runtime.ROOT/'reports/local_session_20261013/postrun_diagnostic_plan.json')
    assert plan['optimizer_steps']==0 and plan['maximum_cases']==96
    cells=defaultdict(list)
    for r in read(OLD/'data_manifest.json')['rows']:
        if r['split']=='train' and r['input_kind']=='noise':
            cells[(r['condition'],r['geometry'])].append(r)
    probes=[min(rs,key=lambda r:r['case_id']) for _,rs in sorted(cells.items())]
    assert len(probes)==48 and len({r['case_id'] for r in probes})==48
    conditions=sorted({r['condition'] for r in probes})
    assert len(conditions)==8 and all(sum(r['condition']==c for r in probes)==6 for c in conditions)
    for arm in ARMS:
        receipt=read(runtime.ROOT/f'reports/training_v4_3/run_001/{arm}/training_receipt.json')
        assert receipt['updates']==500
        assert read(runtime.ROOT/f'reports/training_v4_3/run_001/{arm}/status.json')['stage']=='COMPLETE'
    protocol={**plan,'case_ids':[r['case_id'] for r in probes],
        'case_ids_sha256':canonical_hash([r['case_id'] for r in probes]),
        'plan_sha256':runtime.sha256(runtime.ROOT/'reports/local_session_20261013/postrun_diagnostic_plan.json'),
        'source_sha256':runtime.sha256(__file__),'started':runtime.now(),
        'interpretation':'Raw loss gradients before clipping or Adam preconditioning; cosine is not the actual optimizer-step effect.'}
    write(OUT/'protocol.json',protocol)
    selection=read(OLD_OUT/'backbone_selection.json')
    journal=Journal(OUT/'case_gradients.jsonl'); matrices=[]; endpoint_receipts=[]; all_scores=[]
    started=time.monotonic()
    try:
        for arm in ARMS:
            path=runtime.ROOT/f'training_v4_3/checkpoints/{arm}/update_0500.pt'
            ck=torch.load(path,map_location='cpu',weights_only=False)
            model=ResidualScaledCNN(ck['physics_config'],ck['coefficient_calibration']).cuda().train()
            model.load_cnn(ck['cnn_state_dict'])
            before=parameters_hash(model); assert before==ck['parameters_sha256']
            buffers={k:v.clone() for k,v in model.named_buffers()}
            named=list(model.named_parameters()); params=tuple(p for _,p in named)
            names=[]
            for n,p in named: names.extend([n.split('.')[0]]*p.numel())
            names=np.array(names); assert len(names)==5797
            masks={k:(np.ones(len(names),bool) if k=='all' else names==k)
                   for k in ('all','trunk','continuum_head','emission_head')}
            sums={c:[np.zeros(5797,np.float64),np.zeros(5797,np.float64)] for c in conditions}
            scores=[]
            for index,row in enumerate(probes):
                if time.monotonic()-started>plan['maximum_seconds']: raise TimeoutError('Diagnostic time budget reached')
                inputs,truth,base=load(row,selection)
                pred=model(*inputs)
                _,terms=protective_loss(model,inputs[0],pred,truth,base,1.)
                ga=torch.autograd.grad(terms['main'],params,retain_graph=True,allow_unused=True)
                gb=torch.autograd.grad(terms['safe_total'],params,allow_unused=True)
                vectors=[]
                for gradients in (ga,gb):
                    v=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gradients,params)])
                    assert torch.isfinite(v).all()
                    vectors.append(v.detach().cpu().numpy().astype(np.float64))
                a,b=vectors; total=a+b
                sums[row['condition']][0]+=a/6
                sums[row['condition']][1]+=b/6
                metrics=score_arrays(pred['template_e'][0].detach().cpu().numpy(),
                    pred['temperature_color_K'][0].detach().cpu().numpy(),
                    {k:v[0].cpu().numpy() for k,v in truth.items()})
                score={k:row[k] for k in ('case_id','condition','geometry','input_kind','signed_aux','raw_hash','label_sha256')}
                score.update(arm=arm,split='train',metrics=metrics,physical=base,
                    main_loss=float(terms['main'].detach()),safe_loss=float(terms['safe_total'].detach()),
                    main_gradient_norm=float(np.linalg.norm(a)),safe_gradient_norm=float(np.linalg.norm(b)),
                    total_gradient_norm=float(np.linalg.norm(total)),main_safe_cosine=cosine(a,b),
                    valid_fraction=float(pred['valid'].float().mean()),optimizer_steps=0)
                journal.append(score); scores.append(score)
                if (index+1)%12==0:
                    write(OUT/'status.json',{'stage':'RUNNING','arm':arm,'completed_in_arm':index+1,'time':runtime.now()})
                    print({'arm':arm,'training_probes':index+1},flush=True)
            array=np.stack([np.stack(sums[c]) for c in conditions])
            np.savez_compressed(OUT/f'{arm}_mean_gradients.npz',gradients=array,
                conditions=np.asarray(conditions),components=np.asarray(['main','safe']),parameter_group=names)
            for component in ('main','safe','total'):
                v=array[:,0] if component=='main' else array[:,1] if component=='safe' else array.sum(1)
                for group,mask in masks.items():
                    matrices.append({'arm':arm,'component':component,'parameter_group':group,'conditions':conditions,
                        'gradient_norms':[float(np.linalg.norm(g[mask])) for g in v],
                        'cosine_matrix':[[cosine(a[mask],b[mask]) for b in v] for a in v]})
            assert before==parameters_hash(model)
            assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
            endpoint_receipts.append({'arm':arm,'checkpoint':rel(path),'checkpoint_sha256':runtime.sha256(path),
                'parameters_sha256_before_and_after':before,'physics_buffers_unchanged':True,'optimizer_steps':0,
                'training_probes':len(scores)})
            all_scores+=scores
            del model
        write(OUT/'group_cosines.json',matrices)
        pooled=[]
        for arm in ARMS:
            for s in strata([r for r in all_scores if r['arm']==arm]):
                rs=[r for r in all_scores if r['arm']==arm and r['condition']==s['condition']]
                energy=sum(r['metrics']['truth_energy'] for r in rs)
                projection=sum((r['metrics']['signed_projection'] or 0)*r['metrics']['truth_energy'] for r in rs)/energy if energy else None
                pooled.append({'arm':arm,**s,'energy_weighted_signed_projection':projection,
                    'safe_active_cases':sum(r['safe_loss']>0 for r in rs),
                    'mean_main_loss':float(np.mean([r['main_loss'] for r in rs])),
                    'mean_safe_loss':float(np.mean([r['safe_loss'] for r in rs]))})
        write(OUT/'training_probe_summary.json',pooled)
        write(OUT/'computation_receipt.json',{'stage':'COMPUTE_COMPLETE','cases':len(all_scores),'optimizer_steps':0,
            'endpoints':endpoint_receipts,'elapsed_s':time.monotonic()-started,'local_only':True,'final_test_used':False,
            'source_sha256':runtime.sha256(__file__)})
        write(OUT/'status.json',{'stage':'COMPUTE_COMPLETE','cases':len(all_scores),'time':runtime.now()})
    except BaseException as exc:
        write(OUT/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),
            'completed_cases':len(journal.rows()),'optimizer_steps':0,'failed_records_retained':True})
        write(OUT/'status.json',{'stage':'FAILED','time':runtime.now()})
        raise


def report(matrices,pooled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[m for m in matrices if m['component']=='total' and m['parameter_group']=='all']
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    labels={'blank':'blank','normal':'normal','weak':'weak','strong':'strong',
        'negative_normal':'neg normal','negative_weak':'neg weak','positive_normal':'pos normal','positive_weak':'pos weak'}
    for ax,m in zip(axes,chosen):
        a=np.array([[np.nan if x is None else x for x in row] for row in m['cosine_matrix']])
        im=ax.imshow(a,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set(xticks=range(8),xticklabels=[labels[c] for c in m['conditions']],
            yticks=range(8),yticklabels=[labels[c] for c in m['conditions']],title=m['arm']+' / update 500')
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
        for i in range(8):
            for j in range(8): ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes,shrink=.7,label='Cosine of mean raw loss gradients')
    fig.savefig(OUT/'group_cosines.png',dpi=170); plt.close(fig)
    lines=['# 500次末点训练梯度诊断','',
        '两个权重各读取48个已有含噪训练输入：八个条件×六种几何，每格按case_id选首个。共96次前向及对应main/safe梯度计算，optimizer更新0次。输入选择未使用误差、验证结果或最终测试。','',
        '这些是裁剪与Adam预条件之前的原始损失梯度。余弦描述当前权重附近的局部关系，不能等同实际optimizer更新方向、证明唯一原因或替代独立验证。','',
        '![按训练条件平均梯度的余弦](group_cosines.png)','',
        '## 主弱信号与负模板的局部关系','',
        '|权重|损失分量|参数组|weak vs negative_weak|weak vs negative_normal|',
        '|---|---|---|---:|---:|']
    for m in matrices:
        cs=m['conditions']; i=cs.index('weak')
        vals=[m['cosine_matrix'][i][cs.index(c)] for c in ('negative_weak','negative_normal')]
        text=['N/A' if v is None else f'{v:.6f}' for v in vals]
        lines.append(f"|{m['arm']}|{m['component']}|{m['parameter_group']}|{text[0]}|{text[1]}|")
    lines+=['','## 同一训练探针的输出','',
        '|权重|条件|模板RMSE e|L2|色温RMSE K|投影|保护项活跃/6|',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in pooled:
        l2='N/A' if s['template_L2'] is None else f"{s['template_L2']:.5f}"
        proj='N/A' if s['energy_weighted_signed_projection'] is None else f"{s['energy_weighted_signed_projection']:.5f}"
        lines.append(f"|{s['arm']}|{s['condition']}|{s['template_RMSE_e']:.4f}|{l2}|{s['color_RMSE_K']:.4f}|{proj}|{s['safe_active_cases']}|")
    lines+=['','每格六个探针均匀平均，不代表原采样分布的精确期望。每格完整main/safe平均梯度向量、参数分组、原始逐例结果均已保存；两模型参数哈希和物理缓冲区在诊断前后保持一致。','']
    (OUT/'GRADIENT_DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def finish_report():
    """CPU report; recover serialized completed computations without a GPU rerun."""
    rows=Journal(OUT/'case_gradients.jsonl').rows()
    matrices=read(OUT/'group_cosines.json'); pooled=read(OUT/'training_probe_summary.json')
    assert len(rows)==96 and len(matrices)==24 and len(pooled)==16
    recovered=False
    if (OUT/'computation_receipt.json').exists():
        computation=read(OUT/'computation_receipt.json')
    else:
        failure=read(OUT/'failure_record.json')
        protocol=read(OUT/'protocol.json')
        assert failure['type']=='ModuleNotFoundError' and failure['completed_cases']==96
        assert runtime.sha256(OUT/'source_at_compute.py')==protocol['source_sha256']
        endpoints=[]
        for arm in ARMS:
            assert len([r for r in rows if r['arm']==arm])==48
            vectors=np.load(OUT/f'{arm}_mean_gradients.npz',allow_pickle=False)
            assert vectors['gradients'].shape==(8,2,5797) and np.isfinite(vectors['gradients']).all()
            cp=runtime.ROOT/f'training_v4_3/checkpoints/{arm}/update_0500.pt'
            original=read(runtime.ROOT/f'reports/training_v4_3/run_001/{arm}/training_receipt.json')
            native=next(c for c in original['checkpoints'] if c['update']==500)
            assert runtime.sha256(cp)==native['sha256']
            endpoints.append({'arm':arm,'checkpoint':rel(cp),'checkpoint_sha256':native['sha256'],
                'parameters_sha256_before_and_after':original['final_parameters_sha256'],
                'invariant_assertions_completed_before_saved_matrices':True,'optimizer_steps':0,'training_probes':48})
        computation={'stage':'COMPUTE_COMPLETE','cases':96,'optimizer_steps':0,'endpoints':endpoints,
            'local_only':True,'final_test_used':False,'recovered_from_serialized_completed_computation':True,
            'gpu_rerun':False,'source_at_compute_sha256':protocol['source_sha256'],
            'reason':'All gradients, endpoint invariant assertions, matrices and pooled scores completed; subsequent matplotlib import failed.'}
        write(OUT/'computation_recovery_receipt.json',computation)
        recovered=True
    report(matrices,pooled)
    write(OUT/'receipt.json',{**computation,'stage':'COMPLETE','reporting_runtime':'existing CPU Python with matplotlib',
        'recovered_plot_failure':recovered,'report_sha256':runtime.sha256(OUT/'GRADIENT_DIAGNOSTIC_REPORT.md')})
    write(OUT/'status.json',{'stage':'COMPLETE','cases':96,'time':runtime.now(),'optimizer_steps':0})
    print({'diagnostic':'COMPLETE','cases':96,'optimizer_steps':0,'reused_computation':True},flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    finish_report() if args.report else run()
