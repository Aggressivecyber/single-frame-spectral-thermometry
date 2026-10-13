"""Summarize immutable v4.4 records without rerunning models or opening final tests."""
import math
import runtime
from training_v4.common import read, write
from training_v4_2.analyze import strata
from training_v4_3.scaled_analyze import distribution
from training_v4_4.protocol import OUT, ARMS
from evaluation_v2.common import Journal


def describe(folder, name, update, checkpoint):
    d=read(folder/f'validation_summary_{update:04d}.json')
    s=read(folder/f'science_summary_{update:04d}.json')
    records=Journal(folder/f'validation_update_{update:04d}.jsonl').rows()
    audits=Journal(folder/f'forward_audit_{update:04d}.jsonl').rows()
    assert len(records)==len(audits)==776
    assert [r['case_id'] for r in records]==[r['case_id'] for r in audits]
    raw=[]
    for group in strata(records):
        rs=[r for r in audits if r['condition']==group['condition'] and r['input_kind']==group['input_kind']]
        assert rs
        raw.append({'condition':group['condition'],'input_kind':group['input_kind'],'frames':len(rs),
            'pooled_weighted_raw_RMS':math.sqrt(sum(r['weighted_raw_RMS']**2 for r in rs)/len(rs)),
            'physical_pooled_weighted_raw_RMS':math.sqrt(sum(r['physical_weighted_raw_RMS']**2 for r in rs)/len(rs))})
    return {'name':name,'update':update,'checkpoint':checkpoint,'decision':d,'science':s,
        'strata':strata(records),'raw_strata':raw,'alpha':distribution([r['alpha'] for r in audits]),
        'all_forward_C_and_T_exact':all(r['final_C_equals_physics'] and r['temperature_equals_physics'] for r in audits),
        'validation_sha256':runtime.sha256(folder/f'validation_update_{update:04d}.jsonl')}


def run():
    assert read(OUT/'zero_audit/status.json')['stage']=='COMPLETE'
    assert read(OUT/'fixed_continuum/status.json')['stage']=='COMPLETE'
    protocol=read(OUT/'protocol.json'); entries=[]
    for arm in ARMS:
        entries.append(describe(OUT/'zero_audit'/arm,arm+'_final_C_fixed',0,
                                protocol['zero_update_endpoints'][arm]['path']))
    for update in range(0,501,100):
        name='initial.pt' if update==0 else f'update_{update:04d}.pt'
        entries.append(describe(OUT/'fixed_continuum','fixed_continuum_train',update,
                               f'training_v4_4/checkpoints/fixed_continuum/{name}'))
    candidates=[r for r in entries if r['science']['science_candidate']]
    candidates.sort(key=lambda r:(r['decision']['weak_L2'],r['decision']['normal_RMSE_e'],r['update']))
    selected=candidates[0] if candidates else None
    selection={'rule':protocol['selection_rule'],'candidate_count':len(candidates),
        'selected':None if selected is None else {'name':selected['name'],'update':selected['update'],
            'checkpoint':selected['checkpoint'],'checkpoint_sha256':runtime.sha256(runtime.ROOT/selected['checkpoint']),
            'forward_model':'training_v4_4.model.FixedContinuumCNN',
            'model_source_sha256':runtime.sha256(runtime.ROOT/'training_v4_4/model.py'),
            'validation_sha256':selected['validation_sha256']},
        'final_test_used':False,'independent_validation_established':False}
    write(OUT/'comparison_summary.json',entries); write(OUT/'candidate_selection.json',selection)
    pairs=[]
    for update in range(0,501,100):
        current=Journal(OUT/f'fixed_continuum/validation_update_{update:04d}.jsonl').rows()
        control_path=runtime.ROOT/f'reports/training_v4_3/run_001/residual_scaled/validation_update_{update:04d}.jsonl'
        control={r['case_id']:r for r in Journal(control_path).rows()}
        for r in current:
            b=control[r['case_id']]
            assert r['raw_hash']==b['raw_hash'] and r['label_sha256']==b['label_sha256']
            assert r['physical']['cache_key']==b['physical']['cache_key']
            pairs.append({'update':update,'case_id':r['case_id'],'raw_hash':r['raw_hash'],
                'label_sha256':r['label_sha256'],'cache_key':r['physical']['cache_key'],
                'condition':r['condition'],'input_kind':r['input_kind'],
                'template_MSE_delta_e2':r['metrics']['template_MSE_e2']-b['metrics']['template_MSE_e2'],
                'temperature_MSE_delta_K2':r['metrics']['temperature_MSE_K2']-b['metrics']['temperature_MSE_K2']})
    write(OUT/'paired_same_updates.json',{'pairs':pairs,'pair_count':len(pairs),
        'baseline':'v4.3 residual_scaled at identical update and sample prefix','all_raw_label_cache_hashes_match':True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.94,wspace=.27,hspace=.32)
    train=[r for r in entries if r['name']=='fixed_continuum_train']; updates=[r['update'] for r in train]
    controls=[read(runtime.ROOT/f'reports/training_v4_3/run_001/residual_scaled/validation_summary_{u:04d}.json') for u in updates]
    for name,ds in [('Final continuum fixed',[r['decision'] for r in train]),('v4.3 residual scaled',controls)]:
        axes[0,0].plot(updates,[d['weak_L2'] for d in ds],'o-',label=name)
        axes[0,1].plot(updates,[d['normal_RMSE_e'] for d in ds],'o-',label=name)
    for condition in ('negative_weak','positive_weak'):
        ss=[next(s for s in r['science']['signed_noisy_conditions'] if s['condition']==condition) for r in train]
        axes[1,0].plot(updates,[s['template_L2'] for s in ss],'o-',label=condition)
    for condition in ('negative_normal','positive_normal'):
        ss=[next(s for s in r['science']['signed_noisy_conditions'] if s['condition']==condition) for r in train]
        axes[1,1].plot(updates,[s['template_RMSE_e'] for s in ss],'o-',label=condition)
        axes[1,1].axhline(math.sqrt(ss[0]['MSE_limit_e2']),ls='--',label=condition+' dev limit')
    axes[0,0].axhline(1,color='black',ls='--'); axes[1,0].axhline(1,color='black',ls='--')
    titles=['Main noisy weak L2','Main noisy normal RMSE (e)','Signed weak L2, separate groups','Signed normal RMSE (e)']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Optimizer updates'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=170); plt.close(fig)
    lines=['# v4.4 固定最终连续谱对照','',
        '先对两预登记500次端点执行零更新全量审查，再完成一个同初始化、同500组抽样的训练对照。最终测试未打开；下面全部为已暴露分布上的开发结果。','',
        '|模型/阶段|更新|原16项|保护96组|主弱L2|主正常RMSE e|旧eligible|新科学筛查|',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in entries:
        d,s=r['decision'],r['science']
        lines.append(f"|{r['name']}|{r['update']}|{d['original_gates_pass_count']}|{sum(g['passed'] for g in d['protection_groups'])}|{d['weak_L2']:.6f}|{d['normal_RMSE_e']:.4f}|{d['eligible']}|{s['science_candidate']}|")
    lines+=['','最终连续谱字段与物理缓存逐帧完全一致；776帧色温校正严格为0。内部六阶段连续谱头仍可影响signed修正，并有梯度；不属于完整展开网络。raw和有效性由最终组合重新计算，完整分层在comparison_summary.json。','',
        '新增科学筛查在新推理前冻结：四类signed含噪组各自MSE≤1.05×物理MSE+0.25 e²，两类signed weak各自L2<1，全部776帧有限且有效。这是开发非劣性筛查，既不是独立确认也不是幅值恢复证明；弱模板峰值45e低于60e检测阈值，弱组召回仅作诊断。','',
        '|同一500次末点signed组|模板L2|模板RMSE e|开发RMSE上限e|投影|固定60e召回|通过|',
        '|---|---:|---:|---:|---:|---:|---|']
    for s in train[-1]['science']['signed_noisy_conditions']:
        recall='N/A' if s['macro_recall_at_fixed_60e'] is None else f"{s['macro_recall_at_fixed_60e']:.6f}"
        lines.append(f"|{s['condition']}|{s['template_L2']:.6f}|{s['template_RMSE_e']:.4f}|{math.sqrt(s['MSE_limit_e2']):.4f}|{s['energy_weighted_signed_projection']:.6f}|{recall}|{s['passed']}|")
    lines+=['',f'按预登记顺序筛选后的开发候选数量：{len(candidates)}。选择记录见candidate_selection.json。',
        '4656组与旧residual_scaled相同步数的raw/label/cache配对通过；所有旧权重和失败记录保留。训练过程、Adam步数、源代码快照与原生重放见fixed_continuum/training_receipt.json。',
        '数据审计与科学审查分别见data_audit/AUDIT.md与science_review/REVIEW.md。独立最终评估必须先冻结同一候选及输入接口；旧final登记不覆盖专门signed条件。','']
    report=OUT/'V4_4_REPORT.md'; report.write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'analysis_receipt.json',{'entries':len(entries),'same_update_pairs':len(pairs),
        'candidate_count':len(candidates),'report_sha256':runtime.sha256(report),
        'figure_sha256':runtime.sha256(OUT/'comparison.png'),'source_sha256':runtime.sha256(__file__),
        'final_test_used':False})
    print({'analysis_complete':True,'entries':len(entries),'candidates':len(candidates),'report':str(report)},flush=True)


if __name__=='__main__':
    run()
