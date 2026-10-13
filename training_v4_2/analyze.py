"""Derived, checkpoint-separated comparisons; never select using final-test data."""
import json
from collections import defaultdict
import runtime
import numpy as np
from training_v4.common import read, write
from evaluation_v2.common import Journal
from training_v4_2.experiment import OUT, OLD, WEIGHTS, LIMIT


def pooled(rows):
    metrics=[r['metrics'] for r in rows]
    n=sum(m['pixels'] for m in metrics); energy=sum(m['truth_energy'] for m in metrics)
    sse=sum(m['template_SSE'] for m in metrics)
    return {'frames':len(rows),'template_RMSE_e':float(np.sqrt(sse/n)),
            'template_L2':float(np.sqrt(sse/energy)) if energy else None,
            'color_RMSE_K':float(np.sqrt(np.mean([m['temperature_MSE_K2'] for m in metrics])))
                if all(m['temperature_MSE_K2'] is not None for m in metrics) else None,
            'negative_recall60':float(np.mean([m['negative_recall60'] for m in metrics if m['negative_recall60'] is not None]))
                if any(m['negative_recall60'] is not None for m in metrics) else None}


def strata(rows):
    cells=defaultdict(list)
    for r in rows: cells[(r['signed_aux'],r['condition'],r['input_kind'])].append(r)
    return [{'signed_aux':k[0],'condition':k[1],'input_kind':k[2],**pooled(rs)} for k,rs in sorted(cells.items())]


def paired_results():
    bases=Journal(OLD/'baseline_per_case.jsonl').rows()
    lookup={(r['case_id'],r['method']):r for r in bases}
    methods=sorted({r['method'] for r in bases})
    summaries=[]; checks=[]; count=0
    destination=OUT/'paired_all_checkpoints.jsonl'; temp=destination.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        for update in range(0,LIMIT+1,100):
            current=Journal(OUT/f'validation_update_{update:04d}.jsonl').rows()
            assert len(current)==776 and all(r['status']=='SUCCESS' for r in current)
            warm_file=OLD/f'safe1/validation_update_{update:04d}.jsonl'
            warm={r['case_id']:r for r in Journal(warm_file).rows()} if update else {}
            summary=read(OUT/f'validation_summary_{update:04d}.json')
            summaries.append({'method':'identity_restart','update':update,'strata':strata(current),'decision':summary})
            if update:
                summaries.append({'method':'warm_safe1','update':update,'strata':strata(list(warm.values())),
                                  'decision':read(OLD/f'safe1/validation_summary_{update:04d}.json')})
            for r in current:
                comparators=[lookup[(r['case_id'],m)] for m in methods]
                if update: comparators.append({**warm[r['case_id']],'method':'warm_safe1_same_update'})
                for b in comparators:
                    cache=b['physical']['cache_key'] if 'physical' in b else b['cache_key']
                    assert r['raw_hash']==b['raw_hash'] and r['label_sha256']==b['label_sha256']
                    assert r['physical']['cache_key']==cache and b['status']=='SUCCESS'
                    tbase=b['metrics']['temperature_MSE_K2']
                    pair={k:r[k] for k in ('case_id','mother_id','condition','input_kind','signed_aux','raw_hash','label_sha256')}
                    pair.update(optimizer_update=update,baseline=b['method'],cache_key=cache,
                        template_MSE_delta=r['metrics']['template_MSE_e2']-b['metrics']['template_MSE_e2'],
                        color_MSE_delta=r['metrics']['temperature_MSE_K2']-tbase if tbase is not None else None,
                        paired_success=True)
                    stream.write(json.dumps(pair,ensure_ascii=False,allow_nan=False)+'\n'); count+=1
            checks.append({'update':update,'cases':len(current),'warm_control_reused':bool(update),
                'new_results_sha256':runtime.sha256(OUT/f'validation_update_{update:04d}.jsonl'),
                'warm_results_sha256':runtime.sha256(warm_file) if update else None})
    temp.replace(destination)
    for method in methods:
        summaries.append({'method':method,'update':None,'strata':strata([r for r in bases if r['method']==method])})
    write(OUT/'comparison_summary.json',summaries)
    write(OUT/'paired_receipt.json',{'pairs':count,'hash':runtime.sha256(destination),'updates':checks,
        'baseline_source':str(OLD/'baseline_per_case.jsonl'),'baseline_source_sha256':runtime.sha256(OLD/'baseline_per_case.jsonl'),
        'all_raw_label_cache_hashes_matched':True,'baseline_inferences_rerun':0})
    return summaries


def diagnostic_summaries():
    stages=[]
    for path in sorted((OUT/'stages').glob('*.jsonl')):
        records=Journal(path).rows(); assert len(records)==144*7
        for step in range(7):
            for group in strata([r for r in records if r['stage']==step]):
                stages.append({'checkpoint':path.stem,'stage':step,**group})
    write(OUT/'stage_summary.json',stages)
    gradient=[]
    for path in sorted((OUT/'gradients').glob('*.jsonl')):
        rs=Journal(path).rows(); assert len(rs)==12
        cosine=[r['cosine'] for r in rs if r['cosine'] is not None]
        ratio=[r['safe_gradient_norm']/r['main_gradient_norm'] for r in rs if r['main_gradient_norm']>0]
        active_ratio=[r['safe_gradient_norm']/r['main_gradient_norm'] for r in rs if r['main_gradient_norm']>0 and r['safe_gradient_norm']>0]
        gradient.append({'file':path.name,'checkpoint':rs[0]['checkpoint'],'update':rs[0]['update'],
            'cases':len(rs),'active_protection_cases':len(cosine),
            'opposed_gradient_cases':sum(c<0 for c in cosine),
            'mean_cosine_when_active':float(np.mean(cosine)) if cosine else None,
            'median_safe_to_main_gradient_norm':float(np.median(ratio)),
            'median_active_safe_to_main_gradient_norm':float(np.median(active_ratio)) if active_ratio else None,
            'diagnostic_only_no_optimizer_steps':True})
    clipping=[]
    for name,path in [('warm_safe1',OLD/'safe1/training_curve.jsonl'),('identity_restart',OUT/'training_curve.jsonl')]:
        rs=Journal(path).rows()[:LIMIT]; assert len(rs)==LIMIT
        norms=[r['gradient_norm_before_clip'] for r in rs]
        clipping.append({'method':name,'updates':LIMIT,'clipped_updates':sum(g>1 for g in norms),
                         'median_gradient_norm':float(np.median(norms)),
                         'mean_main_loss':float(np.mean([r['loss']['main'] for r in rs])),
                         'mean_safe_loss':float(np.mean([r['loss']['safe_total'] for r in rs]))})
    write(OUT/'gradient_summary.json',{'probes':gradient,'actual_training_clipping':clipping,
        'interpretation_limit':'gradient opposition and clipping are diagnostics, not proof of a sole cause'})
    return stages,gradient,clipping


def figures(stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures'; folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,path in [('identity restart',OUT),('warm safe1',OLD/'safe1')]:
        train=Journal(path/'training_curve.jsonl').rows()[:LIMIT]
        axes[0,0].plot([r['optimizer_update'] for r in train],[r['loss']['total'] for r in train],alpha=.65,label=name)
        vals=[read(path/f'validation_summary_{i:04d}.json') for i in range(0 if path==OUT else 100,501,100)]
        for ax,key in [(axes[0,1],'weak_L2'),(axes[1,0],'original_gates_pass_count')]:
            ax.plot([r['optimizer_update'] for r in vals],[r[key] for r in vals],marker='o',label=name)
        axes[1,1].plot([r['optimizer_update'] for r in vals],
            [sum(d['passed'] for d in r['protection_groups']) for r in vals],marker='o',label=name)
    axes[0,0].set(yscale='log',title='Actual four-sample training loss')
    axes[0,1].set(title='Noisy main weak: pooled relative L2'); axes[0,1].axhline(1,color='black',ls='--',label='zero estimate')
    axes[1,0].set(title='Original deterministic gates',ylim=(0,16.5)); axes[1,0].axhline(16,color='black',ls='--')
    axes[1,1].set(title='Protection subgroups passed',ylim=(0,99)); axes[1,1].axhline(96,color='black',ls='--')
    for ax in axes.flat: ax.set_xlabel('Optimizer updates'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(folder/'training_comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name in ('v4_update2000','safe1_update1500','safe1_update0500','identity_update0500'):
        for ax,condition,kind,signed,key,title in [
            (axes[0],'negative_normal','continuous',True,'template_RMSE_e','Negative normal, continuous (6 frames)\nTemplate RMSE (e)'),
            (axes[1],'negative_normal','continuous',True,'color_RMSE_K','Negative normal, continuous (6 frames)\nColor temperature RMSE (K)'),
            (axes[2],'weak','noise',False,'template_L2','Main weak, noisy (6 diagnostic frames)\nRelative L2; full validation reported separately')]:
            rs=sorted([r for r in stages if r['checkpoint']==name and r['condition']==condition and r['input_kind']==kind and r['signed_aux']==signed],key=lambda r:r['stage'])
            assert len(rs)==7
            ax.plot([r['stage'] for r in rs],[r[key] for r in rs],marker='o',label=name); ax.set_title(title,fontsize=10)
    axes[2].axhline(1,color='black',ls='--',label='zero estimate')
    for ax in axes: ax.set_xlabel('Completed shared corrections (0 = physical)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(folder/'stage_diagnostics.png',dpi=160); plt.close(fig)


def report(summary,stages,gradient,clipping):
    new=[r for r in summary if r['method']=='identity_restart']
    failures=[]
    for r in new:
        d=r['decision']
        if not d['eligible']:
            failures.append({'update':r['update'],'original_gate_failures':16-d['original_gates_pass_count'],
                'protection_subgroup_failures':sum(not p['passed'] for p in d['protection_groups']),
                'noisy_temperature_pass':d['noisy_temperature_pass'],'weak_L2':d['weak_L2'],
                'eligible':False,'checkpoint_excluded_or_deleted':False})
    write(OUT/'failure_records.json',{'rejected_candidates':failures,'execution_failure':read(OUT/'failure_record.json') if (OUT/'failure_record.json').exists() else None})
    eligible=[r for r in new if r['update']>0 and r['decision']['eligible']]
    decision={'qualified_checkpoint_updates':[r['update'] for r in eligible],
        'automatic_extension':False,'completed_update_limit':500,
        'conclusion':'qualified within bounded pilot' if eligible else 'no qualified replacement within bounded pilot',
        'initialization_causal_scope':'only residual-head initialization changed; trunk/data/loss/optimizer settings/order/physics fixed',
        'limitations':['one inherited trunk, one seed, 500 updates; does not establish impossibility for longer training or other initializations',
                       'stage and gradient diagnostics do not independently prove a sole failure mechanism']}
    write(OUT/'decision.json',decision)
    final=next(r for r in new if r['update']==500)
    warm=next(r for r in summary if r['method']=='warm_safe1' and r['update']==500)
    def stage_value(name,step,key):
        return next(r[key] for r in stages if r['checkpoint']==name and r['stage']==step and r['condition']=='negative_normal' and r['input_kind']=='continuous')
    interpretation=[
        f"同为500次更新，清零残差头将原门禁从{warm['decision']['original_gates_pass_count']}/16提高到{final['decision']['original_gates_pass_count']}/16，新保护通过组从{sum(p['passed'] for p in warm['decision']['protection_groups'])}/96提高到{sum(p['passed'] for p in final['decision']['protection_groups'])}/96；主含噪弱L2则从{warm['decision']['weak_L2']:.4f}变为{final['decision']['weak_L2']:.4f}。初始化影响了保护/降噪取舍，尚未消除取舍。",
        f"六个干净负正常模板诊断帧上，旧v4 update2000的模板RMSE从物理起点{stage_value('v4_update2000',0,'template_RMSE_e'):.3f} e，第一步升至{stage_value('v4_update2000',1,'template_RMSE_e'):.3f} e，六步后为{stage_value('v4_update2000',6,'template_RMSE_e'):.3f} e。错误从第一步出现，随后继续累积。",
        f"新模型初始严格恒等，但训练500次后，同组第一步RMSE为{stage_value('identity_update0500',1,'template_RMSE_e'):.3f} e，六步后仍为{stage_value('identity_update0500',6,'template_RMSE_e'):.3f} e。错误校正在训练后重新出现；本次仅清零残差头不足以修复。",
        '本轮保留旧共享主干，因而没有排除主干继承偏置，也未检验整网重新初始化或更长训练。按预先约定停止在500次，不把初始16/16与后期的降噪结果合并为合格模型。']
    write(OUT/'interpretation.json',{'statements':interpretation,'new_optimizer_updates':500,'one_changed_training_variable':'residual-head initialization'})
    lines=['# TRAIN_V4_2_REPORT：恒等起点短程对照','',
        '基于 `281a81b`，实际完成500次Adam更新，保存初始化与每100次checkpoint。'+
        ('有候选通过既定联合要求。' if eligible else '**本次500次内没有满足全部既定要求的替换模型。**'),'',
        '唯一训练变量：继承v4 update2000共享主干，将两个残差头的权重和偏置清零。物理骨干、5797参数、19特征、原残差归一化、六次共享校正、保护权重1、lr=3e-5、累积4、clip=1、seed=4101和前500次抽样均不变。Adam从空状态开始。所有推理仅接收raw、cached_state、cached_beta。','',
        '旧safe1前500次按采样ID、raw hash、label hash和cache key匹配复用。额外复现其首批前后向损失与梯度，未执行旧控制的optimizer step。新初始化在144个按元数据分层选择的验证帧上逐项与物理输出严格相等；零头时首步主干梯度为零、活跃输出头有梯度，符合预期。','',
        '## 本轮能确定的结论','',*sum(([s,''] for s in interpretation),[]),
        '## 同更新次数对照','',
        '|方法|更新|原门禁/16|保护组/96|主含噪弱L2|正常模板RMSE e|含噪温度通过|合格|',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in summary:
        if r.get('decision'):
            d=r['decision']; lines.append(f"|{r['method']}|{r['update']}|{d['original_gates_pass_count']}|{sum(p['passed'] for p in d['protection_groups'])}|{d['weak_L2']:.4f}|{d['normal_RMSE_e']:.3f}|{d['noisy_temperature_pass']}|{d['eligible']}|")
    lines+=['','每一行对应独立checkpoint，不拼接不同权重的优势。原16项和96个新保护子组保持原标准；弱信号零估计L2=1，正负检测阈值仍为±60 e。所有776帧保留，最终测试未生成或打开，旧954项未重跑。','',
        '![实际训练与验证](figures/training_comparison.png)','',
        '## 末次500权重的分层结果','',
        '|方法|辅助组|条件|输入|帧数|模板RMSE e|模板L2|色温RMSE K|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['method'] in ('fixed_physics','v4_update2000') or (r['method'] in ('identity_restart','warm_safe1') and r['update']==500):
            for s in r['strata']:
                l2='N/A' if s['template_L2'] is None else f"{s['template_L2']:.4f}"
                lines.append(f"|{r['method']}|{s['signed_aux']}|{s['condition']}|{s['input_kind']}|{s['frames']}|{s['template_RMSE_e']:.3f}|{l2}|{s['color_RMSE_K']:.3f}|")
    lines+=['','正负辅助样本与主样本分开；支持域使用abs(truth)。six-stage诊断选144验证帧：六种几何×主四强度×三种输入，以及六种几何×正负弱/正常×三种输入，每格按case_id选首帧，不按误差挑帧。','',
        '## 六次校正的诊断','',
        '阶段0为固定物理输出，阶段1—6严格保留原stage索引和原correction调用；第6次输出已逐帧核对等于原forward。truth仅用于评分。图中每个切片有6帧；右图是固定诊断子集，不是主含噪弱组128帧总体指标，不能混用两者的L2。','',
        '![阶段误差](figures/stage_diagnostics.png)','',
        '|checkpoint|阶段|负正常连续模板RMSE e|负正常连续色温RMSE K|',
        '|---|---:|---:|---:|']
    for r in stages:
        if r['condition']=='negative_normal' and r['input_kind']=='continuous' and r['checkpoint']!='identity_initial':
            lines.append(f"|{r['checkpoint']}|{r['stage']}|{r['template_RMSE_e']:.3f}|{r['color_RMSE_K']:.3f}|")
    lines+=['','## 梯度与裁剪','',
        '固定12个训练探针：主空白/弱、辅助正/负正常，各连续/量化/含噪。分别用autograd计算主损失和保护损失梯度；不执行optimizer step，也不改变采样。诊断中的单样本裁剪因子不等同于正式四样本累积梯度。','',
        '|方法|实际更新数|发生clip的更新数|裁剪前梯度范数中位数|平均主loss|平均保护loss|',
        '|---|---:|---:|---:|---:|---:|']
    for r in clipping:
        lines.append(f"|{r['method']}|{r['updates']}|{r['clipped_updates']}|{r['median_gradient_norm']:.4g}|{r['mean_main_loss']:.5g}|{r['mean_safe_loss']:.5g}|")
    lines+=['','|方法|更新|保护梯度活跃探针数|主/保护梯度夹角为钝角的探针数|活跃探针的保护/主梯度范数比中位数|',
        '|---|---:|---:|---:|---:|']
    for r in gradient:
        ratio='N/A' if r['median_active_safe_to_main_gradient_norm'] is None else f"{r['median_active_safe_to_main_gradient_norm']:.3g}"
        lines.append(f"|{r['checkpoint']}|{r['update']}|{r['active_protection_cases']}|{r['opposed_gradient_cases']}|{ratio}|")
    if (OUT/'actual_batch_gradients.json').exists():
        lines+=['','进一步从初始化、100、300次的权重重构实际第1、101、301次更新所用四个样本的累积梯度。合成总梯度范数与真实训练日志核对；不执行额外optimizer更新。','',
            '|方法|实际更新|主梯度范数|保护梯度范数|余弦|实际clip系数|',
            '|---|---:|---:|---:|---:|---:|']
        for r in read(OUT/'actual_batch_gradients.json')['records']:
            cosine='N/A' if r['cosine'] is None else f"{r['cosine']:.4f}"
            lines.append(f"|{r['method']}|{r['actual_gradient_update']}|{r['main_gradient_norm']:.4g}|{r['safe_gradient_norm']:.4g}|{cosine}|{r['clip_factor']:.4g}|")
    lines+=['','裁剪频率、梯度冲突或阶段误差只能用于定位现象，不能单独证明唯一原因。恒等起点效果的因果解释仅限于本次固定主干、固定样本序列、一个种子和500次更新；未排除更长训练或其他初始化的可能。','',
        '## 复现与保留','',
        '原代码、物理模块、缓存、旧权重与旧报告按preserved_files.json保留原字节。新代码在training_v4_2，新checkpoint在training_v4_2/checkpoints，新报告在本目录，全部位于E盘。大数组仍使用已有E盘开发缓存，不上传重复副本。','',
        '需要原v4.1开发数据及物理缓存。将新代码复制到有这些冻结输入的新E/D工作副本，且该副本的reports/training_v4_2/run_001中没有训练曲线，然后执行：','',
        '```powershell','powershell -NoProfile -ExecutionPolicy Bypass -File training_v4_2/run.ps1',
        '& $GpuPython -B -m training_v4_2.verify',
        '& $GpuPython -B -m training_v4_2.batch_gradients',
        '& $ReportPython -B -m training_v4_2.analyze','```','',
        '解释器路径与测试命令见training_v4_2/README.md。已有结果不覆盖；输入/缓存哈希、完整采样前缀身份、每帧验证、配对、阶段记录、梯度、真实曲线和失败记录均已保存。物理推理与最终验收分开，不宣称本试验达到实测性能。','']
    (OUT/'TRAIN_V4_2_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'analysis_sources.json',{'derived_report_code_sha256':runtime.sha256(__file__),
        'source_affects_training':False,'training_source':'training_v4_2/experiment.py'})


if __name__=='__main__':
    assert read(OUT/'training_receipt.json')['updates']==LIMIT
    summary=paired_results(); stages,gradient,clipping=diagnostic_summaries()
    figures(stages); report(summary,stages,gradient,clipping)
    print({'report':str(OUT/'TRAIN_V4_2_REPORT.md'),'derived_analysis_complete':True},flush=True)
