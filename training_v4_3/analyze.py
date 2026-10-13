"""Local-only, same-input comparison with the frozen v4.2 control."""
import json
from collections import defaultdict
import runtime
import numpy as np
from training_v4.common import read,write
from evaluation_v2.common import Journal
from training_v4_2.analyze import strata

ROOT=runtime.ROOT/'reports/training_v4_3/run_001'
REFERENCE=runtime.ROOT/'reports/training_v4_2/run_001'
OLD=runtime.ROOT/'reports/training_v4_1/run_001'


def analyze(name='fresh_trunk'):
    folder=ROOT/name; receipt=read(folder/'training_receipt.json'); assert receipt['updates']==500
    summary=[]; failures=[]; counts=0
    temp=folder/'paired_to_inherited_trunk.tmp'
    with temp.open('w',encoding='utf-8') as stream:
        for update in range(0,501,100):
            now=Journal(folder/f'validation_update_{update:04d}.jsonl').rows()
            prior={r['case_id']:r for r in Journal(REFERENCE/f'validation_update_{update:04d}.jsonl').rows()}
            assert len(now)==len(prior)==776
            decision=read(folder/f'validation_summary_{update:04d}.json')
            for r in now:
                p=prior[r['case_id']]
                assert r['status']==p['status']=='SUCCESS'
                assert r['raw_hash']==p['raw_hash'] and r['label_sha256']==p['label_sha256']
                assert r['physical']['cache_key']==p['physical']['cache_key']
                record={k:r[k] for k in ('case_id','raw_hash','label_sha256','condition','input_kind','signed_aux')}
                record.update(update=update,template_MSE_delta=r['metrics']['template_MSE_e2']-p['metrics']['template_MSE_e2'],
                    color_MSE_delta=r['metrics']['temperature_MSE_K2']-p['metrics']['temperature_MSE_K2'],paired_success=True,
                    baseline='v4_2_inherited_trunk_zero_heads_same_update',cache_key=r['physical']['cache_key'])
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); counts+=1
            summary.append({'update':update,'decision':decision,'strata':strata(now),
                            'reference':read(REFERENCE/f'validation_summary_{update:04d}.json')})
            if not decision['eligible']:
                failures.append({'update':update,'original_gate_failures':16-decision['original_gates_pass_count'],
                    'protection_failures':sum(not d['passed'] for d in decision['protection_groups']),
                    'weak_L2':decision['weak_L2'],'noisy_temperature_pass':decision['noisy_temperature_pass']})
    temp.replace(folder/'paired_to_inherited_trunk.jsonl')
    write(folder/'comparison_summary.json',summary)
    write(folder/'failure_records.json',{'rejected_checkpoints':failures,'deleted_frames':0,'qualified_checkpoints':[r['update'] for r in summary if r['decision']['eligible']]})
    stage_rows=Journal(folder/'stages'/f'{name}_update0500.jsonl').rows()
    stage_summary=[{'stage':stage,**s} for stage in range(7) for s in strata([r for r in stage_rows if r['stage']==stage])]
    write(folder/'stage_summary.json',stage_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label,key in [(name,'decision'),('inherited trunk','reference')]:
        values=[r[key] for r in summary]
        axes[0].plot([r['optimizer_update'] for r in values],[r['weak_L2'] for r in values],marker='o',label=label)
        axes[1].plot([r['optimizer_update'] for r in values],[r['original_gates_pass_count'] for r in values],marker='o',label=label)
        axes[2].plot([r['optimizer_update'] for r in values],[sum(g['passed'] for g in r['protection_groups']) for r in values],marker='o',label=label)
    axes[0].axhline(1,ls='--',color='black'); axes[0].set_title('Noisy weak: pooled relative L2')
    axes[1].axhline(16,ls='--',color='black'); axes[1].set(title='Original gates',ylim=(0,16.5))
    axes[2].axhline(96,ls='--',color='black'); axes[2].set(title='Protection subgroups',ylim=(0,99))
    for ax in axes: ax.set_xlabel('Optimizer updates'); ax.legend(); ax.grid(alpha=.2)
    figure.savefig(folder/'comparison.png',dpi=160); plt.close(figure)
    lines=['# v4.3本地有限对照：'+name,'',
        '新结果仅本地保存；未推送公开仓库。固定物理后的5797参数、19通道、六次共享CNN校正器，属于物理反演后CNN校正，未反传LM/PCG。','',
        '本臂仅将v4.2继承的共享主干换成新的随机初始化，两个残差头仍为零。保持seed4101、lr3e-5、safe1、累积4、clip1、相同原始数据与前500次抽样。物理与特征不变；不打开最终测试，不重跑旧954项。','',
        '|更新|新臂原门禁/16|v4.2原门禁/16|新臂保护/96|v4.2保护/96|新臂弱L2|v4.2弱L2|新臂正常RMSE e|新臂合格|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in summary:
        d,b=r['decision'],r['reference']
        lines.append(f"|{r['update']}|{d['original_gates_pass_count']}|{b['original_gates_pass_count']}|{sum(x['passed'] for x in d['protection_groups'])}|{sum(x['passed'] for x in b['protection_groups'])}|{d['weak_L2']:.4f}|{b['weak_L2']:.4f}|{d['normal_RMSE_e']:.3f}|{d['eligible']}|")
    lines+=['','![同更新次数](comparison.png)','',
        '每行是独立checkpoint；恒等初始化的保护性不与后期降噪合并报告。负模板使用abs(truth)支持域及负阈值检测，未改原门禁。','',
        '## 末次分层结果','',
        '|辅助组|条件|输入|帧数|模板RMSE e|L2|色温RMSE K|','|---|---|---|---:|---:|---:|---:|']
    for s in summary[-1]['strata']:
        l2='N/A' if s['template_L2'] is None else f"{s['template_L2']:.4f}"
        lines.append(f"|{s['signed_aux']}|{s['condition']}|{s['input_kind']}|{s['frames']}|{s['template_RMSE_e']:.3f}|{l2}|{s['color_RMSE_K']:.3f}|")
    lines+=['','完成500次实际更新与原生checkpoint/Adam步数重开，3个代表帧原生预测已复现。数据、标签和缓存键逐例匹配v4.2；生成'+str(counts)+'条同输入配对。初始144帧物理恒等和活跃头梯度检查已执行。','',
        '本比较限于一个种子和500次预算；较保守的输出不等同于有效降噪，较低loss也不替代联合门禁。完整新结果、阶段误差、梯度与失败均在同目录保留。','',
        '复现：在含冻结E/D开发数据、尚无本臂训练曲线的新工作副本中，使用既有CUDA解释器执行：','',
        '```powershell','& $GpuPython -B -m training_v4_3.fresh_trunk',
        '& $ReportPython -B -m training_v4_3.analyze','```','',
        '需设置PYTHONDONTWRITEBYTECODE=1、TMP/TEMP/CUDA_CACHE_PATH至E/D工作副本；解释器路径见training_v4_2/README.md。时限按本地时段注册，跨日期复现时必须明确改为新的有界截止。','']
    (folder/'TRAIN_V4_3_LOCAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(folder/'analysis_receipt.json',{'paired_rows':counts,'validation_rows':6*776,'stage_rows':len(stage_rows),
        'report_sha256':runtime.sha256(folder/'TRAIN_V4_3_LOCAL_REPORT.md'),'local_only':True,
        'qualified_checkpoints':[r['update'] for r in summary if r['decision']['eligible']]})
    print({'name':name,'analysis':'complete','qualified':[r['update'] for r in summary if r['decision']['eligible']]},flush=True)


if __name__=='__main__': analyze()
