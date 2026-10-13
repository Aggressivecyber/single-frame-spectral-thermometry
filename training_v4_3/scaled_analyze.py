"""Read-only inference records -> local report; preserves checkpoint separation."""
import json
from collections import defaultdict
import runtime
import numpy as np
from training_v4.common import read, write
from evaluation_v2.common import Journal
from training_v4_2.analyze import strata, pooled

OUT = runtime.ROOT/'reports/training_v4_3/run_001/residual_scaled'
OLD = runtime.ROOT/'reports/training_v4_1/run_001'


def distribution(values):
    x = np.asarray(values, float)
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                    map(float, np.quantile(x, [0, .05, .5, .95, 1]))))


def temperature_failures(decision):
    return [{**g, 'limit_RMSE_K': float(np.sqrt(1.05*g['physical_color_RMSE_K']**2+.04))}
            for g in decision['groups'] if g['input_kind']=='noise'
            and g['color_RMSE_K']**2 > 1.05*g['physical_color_RMSE_K']**2+.04]


def analyze(name='residual_scaled'):
    global OUT
    assert name in ('residual_scaled','negative_balanced')
    balanced=name=='negative_balanced'
    OUT=runtime.ROOT/'reports/training_v4_3/run_001'/name
    control=runtime.ROOT/'reports/training_v4_3/run_001/residual_scaled' if balanced else OLD/'safe1'
    control_name='residual_scaled' if balanced else 'warm_safe1'
    receipt = read(OUT/'training_receipt.json')
    assert receipt['updates']==500 and len(receipt['checkpoints'])==6
    protocol = read(OUT/'protocol.json')
    calibration = read(OUT/'coefficient_calibration.json')
    assert runtime.sha256(OUT/'coefficient_calibration.json')==protocol['coefficient_calibration_sha256']
    baseline_rows = Journal(OLD/'baseline_per_case.jsonl').rows()
    baseline = {(r['case_id'], r['method']):r for r in baseline_rows}
    methods = sorted({r['method'] for r in baseline_rows})
    summary, failed, checks, signed_fidelity, pairs = [], [], [], [], 0
    destination = OUT/'paired_all_checkpoints.jsonl'
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for update in range(0, 501, 100):
            path = OUT/f'validation_update_{update:04d}.jsonl'
            rows = Journal(path).rows()
            assert len(rows)==776 and len({r['case_id'] for r in rows})==776
            assert all(r['status']=='SUCCESS' for r in rows)
            decision = read(OUT/f'validation_summary_{update:04d}.json')
            warm_path = control/f'validation_update_{update:04d}.jsonl'
            warm = {r['case_id']:r for r in Journal(warm_path).rows()} if update else {}
            record = {'update':update, 'decision':decision, 'strata':strata(rows),
                      'warm_same_update': read(control/f'validation_summary_{update:04d}.json') if update else None,
                      'maximum_invalid_output_fraction':max(r['invalid_output_fraction'] for r in rows)}
            summary.append(record)
            for condition in ('negative_weak','negative_normal','positive_weak','positive_normal'):
                rs=[r for r in rows if r['condition']==condition and r['input_kind']=='noise']
                energy=sum(r['metrics']['truth_energy'] for r in rs)
                projection=sum(r['metrics']['signed_projection']*r['metrics']['truth_energy'] for r in rs)/energy
                recall_key='negative_recall60' if condition.startswith('negative') else 'positive_recall60'
                recalls=[r['metrics'][recall_key] for r in rs if r['metrics'][recall_key] is not None]
                signed_fidelity.append({'update':update,'condition':condition,'frames':len(rs),
                    'energy_weighted_signed_projection':projection,
                    'macro_recall_at_fixed_signed_60e':float(np.mean(recalls)) if recalls else None,
                    'diagnostic_only_no_acceptance_rule_change':True})
            if not decision['eligible']:
                failed.append({'update':update, 'original_failed':16-decision['original_gates_pass_count'],
                    'protection_failed':sum(not g['passed'] for g in decision['protection_groups']),
                    'weak_L2':decision['weak_L2'], 'noisy_temperature_failures':temperature_failures(decision)})
            for r in rows:
                controls = [baseline[(r['case_id'], m)] for m in methods]
                if update:
                    controls.append({**warm[r['case_id']], 'method':control_name+'_same_update'})
                for b in controls:
                    key = b['physical']['cache_key'] if 'physical' in b else b['cache_key']
                    assert r['raw_hash']==b['raw_hash'] and r['label_sha256']==b['label_sha256']
                    assert r['physical']['cache_key']==key and b['status']=='SUCCESS'
                    tb = b['metrics']['temperature_MSE_K2']
                    p = {k:r[k] for k in ('case_id','mother_id','condition','input_kind','signed_aux','raw_hash','label_sha256')}
                    p.update(optimizer_update=update, baseline=b['method'], cache_key=key,
                        template_MSE_delta=r['metrics']['template_MSE_e2']-b['metrics']['template_MSE_e2'],
                        color_MSE_delta=r['metrics']['temperature_MSE_K2']-tb if tb is not None else None,
                        paired_success=True)
                    stream.write(json.dumps(p, ensure_ascii=False, allow_nan=False)+'\n')
                    pairs += 1
            checks.append({'update':update, 'validation_sha256':runtime.sha256(path),
                           'warm_control_sha256':runtime.sha256(warm_path) if update else None,
                           'maximum_invalid_output_fraction':record['maximum_invalid_output_fraction']})
    temporary.replace(destination)
    write(OUT/'comparison_summary.json', summary)
    write(OUT/'signed_fidelity.json', signed_fidelity)
    eligible = [s['update'] for s in summary if s['decision']['eligible']]
    write(OUT/'failure_records.json', {'rejected_checkpoints':failed, 'qualified_checkpoints':eligible,
                                      'deleted_frames':0, 'final_test_used':False})
    write(OUT/'paired_receipt.json', {'pairs':pairs, 'methods':methods, 'updates':checks,
        'all_raw_label_cache_hashes_matched':True, 'baseline_inferences_rerun':0,
        'baseline_source_sha256':runtime.sha256(OLD/'baseline_per_case.jsonl'),
        'paired_file_sha256':runtime.sha256(destination)})

    coeffs = Journal(OUT/'validation_coefficients.jsonl').rows()
    assert len(coeffs)==776 and all(r['truth_used'] is False for r in coeffs)
    coeff_summary = []
    for kind in ('continuous','quantization','noise'):
        rs = [r for r in coeffs if r['input_kind_for_reporting_only']==kind]
        coeff_summary.append({'input_kind':kind, 'frames':len(rs),
                              'rho':distribution([r['rho'] for r in rs]),
                              'alpha':distribution([r['alpha'] for r in rs])})
    write(OUT/'coefficient_distributions.json', coeff_summary)

    train = Journal(OUT/'training_curve.jsonl').rows()
    old_train = Journal(control/'training_curve.jsonl').rows()[:500]
    assert len(train)==len(old_train)==500
    if balanced:
        assert all(a['samples'][j]==b['samples'][j] for a,b in zip(train, old_train) for j in (0,2,3))
        sampling=read(OUT/'sampling_sequence.json')
        assert all([s['case_id'] for s in a['samples']]==b for a,b in zip(train,sampling['draws']))
    else:
        assert all(a['samples']==b['samples'] for a,b in zip(train, old_train))
    probes = Journal(OUT/f'gradients/{name}_0500.jsonl').rows()
    assert len(probes)==12 and all(r['split']=='train' for r in probes)
    diagnostic = {'probes':probes, 'clipping':[]}
    for arm, rs in [(name,train),(control_name,old_train)]:
        norms = [r['gradient_norm_before_clip'] for r in rs]
        diagnostic['clipping'].append({'method':arm, 'updates':len(rs),
            'clipped_updates':sum(n>1 for n in norms), 'gradient_norm':distribution(norms),
            'mean_main_loss':float(np.mean([r['loss']['main'] for r in rs])),
            'mean_safe_loss':float(np.mean([r['loss']['safe_total'] for r in rs]))})
    write(OUT/'gradient_summary.json', diagnostic)

    proposal = Journal(OUT/'proposal_vs_scaled.jsonl').rows()
    assert len(proposal)==288 and len({r['case_id'] for r in proposal})==144
    cells = defaultdict(list)
    for r in proposal:
        cells[(r['output'],r['signed_aux'],r['condition'],r['input_kind'])].append(r)
    proposal_summary = []
    for key, rs in sorted(cells.items()):
        proposal_summary.append({'output':key[0], 'signed_aux':key[1], 'condition':key[2],
            'input_kind':key[3], **pooled(rs), 'minimum_valid_fraction':min(r['valid_fraction'] for r in rs),
            'mean_signed_template_correction_e':float(np.mean([r['signed_template_correction_mean_e'] for r in rs])),
            'template_correction_RMS_e':float(np.sqrt(np.mean([r['template_correction_RMS_e']**2 for r in rs]))),
            'temperature_correction_RMS_K':float(np.sqrt(np.mean([r['temperature_correction_RMS_K']**2 for r in rs])))})
    write(OUT/'proposal_summary.json', proposal_summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16,9.2))
    fig.subplots_adjust(left=.065,right=.985,bottom=.075,top=.955,wspace=.24,hspace=.30)
    updates = [s['update'] for s in summary]
    for arm, key in [(name,'decision'),(control_name,'warm_same_update')]:
        rs = [s for s in summary if s[key] is not None]
        x = [r['update'] for r in rs]; d = [r[key] for r in rs]
        axes[0,0].plot(x,[r['weak_L2'] for r in d], 'o-', label=arm)
        axes[0,1].plot(x,[r['original_gates_pass_count'] for r in d], 'o-', label=arm)
        axes[0,2].plot(x,[sum(g['passed'] for g in r['protection_groups']) for r in d], 'o-', label=arm)
        negative = [next(g for g in r['groups'] if g['input_kind']=='noise' and g['signed_aux'] and g['group']=='negative') for r in d]
        axes[1,0].plot(x,[g['color_RMSE_K'] for g in negative], 'o-', label=arm)
    ng = next(g for g in summary[0]['decision']['groups'] if g['input_kind']=='noise' and g['signed_aux'] and g['group']=='negative')
    axes[1,0].axhline(np.sqrt(1.05*ng['physical_color_RMSE_K']**2+.04), color='black', ls='--', label='frozen temperature limit')
    for kind in ('continuous','quantization','noise'):
        rs=[r for r in coeffs if r['input_kind_for_reporting_only']==kind]
        axes[1,1].scatter([r['rho'] for r in rs],[max(r['alpha'],1e-20) for r in rs], s=9, alpha=.5, label=kind)
    for condition in ('negative_weak','negative_normal'):
        rs=[next(g for g in s['strata'] if g['condition']==condition and g['input_kind']=='noise') for s in summary]
        axes[1,2].plot(updates,[r['template_L2'] for r in rs], 'o-', label=condition)
    axes[0,0].axhline(1,color='black',ls='--'); axes[0,0].set_title('Main noisy weak: relative L2')
    axes[0,1].axhline(16,color='black',ls='--'); axes[0,1].set(title='Original gates',ylim=(0,16.8))
    axes[0,2].axhline(96,color='black',ls='--'); axes[0,2].set(title='Protection groups',ylim=(0,100))
    axes[1,0].set(title='Noisy negative: color RMSE',ylabel='K')
    axes[1,1].set(title='Frozen input coefficient, 776 dev frames',xscale='log',yscale='log',xlabel='Initial weighted raw RMS',ylabel='alpha')
    axes[1,2].axhline(1,color='black',ls='--'); axes[1,2].set_title('Signed noisy template: relative L2')
    for i, ax in enumerate(axes.flat):
        if i!=4: ax.set_xlabel('Optimizer updates')
        ax.title.set_fontsize(12)
        ax.tick_params(labelsize=10,labelleft=True,labelcolor='black')
        ax.xaxis.label.set_fontsize(10); ax.yaxis.label.set_fontsize(10)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=170); plt.close(fig)

    change_description=('只改变每次训练的第二个含噪抽样位置，交替使用已有negative_weak/negative_normal训练池；每池12帧，种子4101打乱轮循。其余三个抽样位置与residual_scaled逐例相同；同样从旧v4/update2000初始化，重置Adam，冻结系数、物理、safe1损失、lr3e-5、累积4、clip1和500次预算。'
        if balanced else '只修改输出校正系数；物理、旧v4/update2000初始化、前500次训练抽样、safe1损失、lr3e-5、累积4、clip1与seed4101均与旧safe1相同。')
    lines=['# v4.3 本地有限对照：'+('负模板抽样平衡' if balanced else '冻结原始残差系数'),'',
        '本轮是单视场重叠输入的物理反演后CNN校正；5797参数、19通道、六次共享校正。'+change_description,'',
        '新结果仅在E盘本地保存，未推送；最终测试未打开。每个checkpoint独立验收，不合并不同权重的优势。','',
        f'|更新|原检查/16|保护/96|主弱L2|主正常RMSE e|含噪温度通过|联合合格|同更新{control_name}弱L2|',
        '|---:|---:|---:|---:|---:|---|---|---:|']
    for s in summary:
        d=s['decision']; w=s['warm_same_update']
        wl=f"{w['weak_L2']:.6f}" if w is not None else '初始权重为旧v4/update2000'
        lines.append(f"|{s['update']}|{d['original_gates_pass_count']}|{sum(g['passed'] for g in d['protection_groups'])}|{d['weak_L2']:.6f}|{d['normal_RMSE_e']:.4f}|{d['noisy_temperature_pass']}|{d['eligible']}|{wl}|")
    lines+=['','联合合格checkpoint：'+(', '.join(map(str,eligible)) if eligible else '无')+'。','',
        '![各checkpoint与固定系数](comparison.png)','',
        '## 系数的证据边界','',
        f"rho由固定物理预测与raw的加权残差计算，alpha=sigmoid({calibration['power']}×log(rho/{calibration['tau']:.10f}))。所有2576个训练输入的raw残差及训练输入类型确定tau与指数；没有用truth误差数值定标。系数在CNN运行前固定，不可由CNN修改自身残差来改变。",
        '', '方案设计动机受此前已查看的开发验证残差间隔启发，这是开发诊断，不能称独立验证。近乎分开的合成噪声模式不能推及真实相机；raw残差较小不保证truth误差小。负模板仍可输出负值，未使用truth、case_id、condition进行推理或屏蔽。','',
        '|输入类型|帧数|rho最小|rho中位|rho最大|alpha最小|alpha最大|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in coeff_summary:
        lines.append(f"|{c['input_kind']}|{c['frames']}|{c['rho']['min']:.6g}|{c['rho']['median']:.6g}|{c['rho']['max']:.6g}|{c['alpha']['min']:.6g}|{c['alpha']['max']:.6g}|")
    lines+=['','## 温度失败组','',
        '原规则保持：每个含噪group的温度MSE不超过1.05×固定物理MSE+0.04 K²。negative_weak与negative_normal另列诊断，但没有替换原先合并的negative验收组。','',
        '|更新|辅助组|组别|帧数|色温RMSE K|物理RMSE K|原阈值 K|',
        '|---:|---|---|---:|---:|---:|---:|']
    for s in summary:
        for g in temperature_failures(s['decision']):
            lines.append(f"|{s['update']}|{g['signed_aux']}|{g['group']}|{g['frames']}|{g['color_RMSE_K']:.4f}|{g['physical_color_RMSE_K']:.4f}|{g['limit_RMSE_K']:.4f}|")
    lines+=['','## 末次完整分层','',
        '|辅助组|条件|输入|帧数|模板RMSE e|相对L2|色温RMSE K|',
        '|---|---|---|---:|---:|---:|---:|']
    for s in summary[-1]['strata']:
        l2='N/A' if s['template_L2'] is None else f"{s['template_L2']:.6f}"
        lines.append(f"|{s['signed_aux']}|{s['condition']}|{s['input_kind']}|{s['frames']}|{s['template_RMSE_e']:.4f}|{l2}|{s['color_RMSE_K']:.4f}|")
    lines+=['','## 含噪有符号模板保真诊断','',
        '投影为sum(pred×truth)/sum(truth²)，按真值能量合并六帧；接近0提示与真值同向的信号几乎消失，理想为1。召回率沿用各帧abs(truth)支持域与固定±60 e阈值，表中为帧平均。此处不修改验收规则；联合eligible也不能替代这些分层结果。','',
        '|更新|条件|有符号投影|固定阈值召回率|','|---:|---|---:|---:|']
    for f in signed_fidelity:
        if f['condition'].startswith('negative'):
            recall='N/A' if f['macro_recall_at_fixed_signed_60e'] is None else f"{f['macro_recall_at_fixed_signed_60e']:.6f}"
            lines.append(f"|{f['update']}|{f['condition']}|{f['energy_weighted_signed_projection']:.6f}|{recall}|")
    lines+=['','## 回执与诊断','',
        '500次实际更新、2000个微批次，'+('1500个抽样位置与控制逐例相同，500个第二含噪位置使用预先冻结的新序列；' if balanced else '逐例抽样与旧safe1完全相同；')+f'6个checkpoint的Adam步数与模型参数哈希重开已核对，3个代表帧原生预测复现通过。6×776条新验证记录与冻结旧结果生成{pairs}条配对，raw、标签和物理缓存键全匹配；旧推理重跑0次。',
        '',f"所有checkpoint最大无效像素比例为{max(s['maximum_invalid_output_fraction'] for s in summary):.6g}。144帧六阶段未衰减proposal与实际输出共288条记录保留负模板、色温及有效比例；12个固定训练帧梯度探针没有更新权重。",'',
        '|训练臂|500次中裁剪次数|梯度范数中位数|平均main loss|平均safe loss|',
        '|---|---:|---:|---:|---:|']
    for g in diagnostic['clipping']:
        lines.append(f"|{g['method']}|{g['clipped_updates']}|{g['gradient_norm']['median']:.6f}|{g['mean_main_loss']:.6f}|{g['mean_safe_loss']:.6f}|")
    lines+=['','完整逐例、失败、梯度、proposal输出和系数记录位于本报告同目录。个别梯度冲突或改善不证明唯一原因；该结果限于一个种子、500次预算与当前合成开发分布。','',
        '复现入口：在含冻结训练数据且没有本臂已有结果的新E/D工作副本中，设置E/D缓存和临时目录，执行CUDA解释器的 `python -B -m training_v4_3.'+('balanced_train' if balanced else 'scaled_train')+'`；只读报告生成使用 `python -B -m training_v4_3.scaled_analyze --name '+name+'`。跨日期重跑需登记新的有界截止，不能静默沿用本次时限。','']
    if balanced:
        lines+=['本臂只有12个训练母体的24个含噪负模板，增加的是重复暴露次数，不是独立数据量。现有开发验证母体独立于训练母体，但本次研究决策受开发结果影响；不能用这次对照替代新的独立采集或最终测试。','']
    report=OUT/('TRAIN_V4_3_NEGATIVE_BALANCED_REPORT.md' if balanced else 'TRAIN_V4_3_RESIDUAL_SCALED_REPORT.md')
    report.write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'analysis_receipt.json', {'paired_rows':pairs,'new_validation_rows':4656,
        'coefficient_rows':len(coeffs),'proposal_rows':len(proposal),'gradient_probes':len(probes),
        'report_sha256':runtime.sha256(report),'qualified_checkpoints':eligible,
        'local_only':True,'final_test_used':False})
    print({'analysis':'complete','pairs':pairs,'qualified_checkpoints':eligible},flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',choices=('residual_scaled','negative_balanced'),default='residual_scaled')
    analyze(parser.parse_args().name)
