"""CPU-only paired analysis of saved v4.6 and v4.5 development records."""
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/training_v4_6/run_001'
NEW=OUT/'negative_noisy_repair'
OLD=ROOT/'reports/training_v4_5/run_001/balanced_fixed_continuum'
UPDATES=range(0,501,100)
SIGNED=('negative_weak','negative_normal','positive_weak','positive_normal')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).relative_to(ROOT)).replace('\\','/')


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def key_matches(a,b):
    for k in ('case_id','mother_id','condition','group','geometry','input_kind','signed_aux','raw_hash','label_sha256'):
        assert a[k]==b[k],(k,a['case_id'])
    assert a['physical']==b['physical'],a['case_id']
    for k in ('pixels','truth_energy'):
        assert a['metrics'][k]==b['metrics'][k],(k,a['case_id'])


def run():
    assert read(NEW/'status.json')['stage']=='COMPLETE'
    receipt=read(NEW/'training_receipt.json');assert receipt['updates']==500
    assert read(OLD/'status.json')['stage']=='COMPLETE'
    assert not (OUT/'analysis_receipt.json').exists(),'Never overwrite completed analysis'
    started=datetime.now(timezone.utc)
    attempt=OUT/'analysis_runs'/started.strftime('%Y%m%dT%H%M%S_%fZ')
    attempt.mkdir(parents=True,exist_ok=False)
    inputs={}
    def use(path):
        inputs[rel(path)]=sha(path)
        return read(path)
    protocol=use(OUT/'protocol.json')
    sampling=use(OUT/'sampling_sequence.json')
    use(OUT/'generation_receipt.json')
    use(OUT/'cache_receipt.json')
    receipts={'v4.6':use(NEW/'training_receipt.json'),'v4.5':use(OLD/'training_receipt.json')}
    protocol_paths={'v4.6':OUT/'protocol.json','v4.5':OLD.parent/'protocol.json'}
    use(protocol_paths['v4.5'])
    checkpoints={}
    for name,training in receipts.items():
        assert training['updates']==500 and len(training['checkpoints'])==6
        table={r['update']:r for r in training['checkpoints']}
        assert set(table)==set(UPDATES),'Checkpoint update inventory changed'
        for u,r in table.items():
            assert r['Adam_steps']==([] if u==0 else [u]),'Checkpoint Adam inventory changed'
            assert sha(ROOT/r['path'])==r['sha256'],'Checkpoint bytes changed after training receipt'
            inputs[r['path']]=r['sha256']
        checkpoints[name]=table
    entries=[];paired=[];parent_groups=defaultdict(list)
    for u in UPDATES:
        observed={}
        for name,folder in [('v4.5',OLD),('v4.6',NEW)]:
            summary=use(folder/f'validation_summary_{u:04d}.json')
            screen=use(folder/f'science_summary_{u:04d}.json')
            path=folder/f'validation_update_{u:04d}.jsonl'
            inputs[rel(path)]=sha(path)
            rows=lines(path)
            assert len(rows)==776 and len({r['case_id'] for r in rows})==776
            assert summary['optimizer_update']==screen['optimizer_update']==u
            assert all(r['optimizer_update']==u for r in rows)
            assert screen['protocol_sha256']==sha(protocol_paths[name]),'Science screen protocol mismatch'
            assert screen['validation_sha256']==sha(path)
            assert screen['science_candidate']==(summary['eligible'] and screen['new_development_screen_pass'])
            assert screen['all_frames_finite'] and screen['maximum_invalid_output_fraction']==0
            assert all(r['temperature_correction_RMS_K']==0 for r in rows)
            for condition in SIGNED:
                group=[r for r in rows if r['signed_aux'] and r['input_kind']=='noise' and r['condition']==condition]
                assert len(group)==6
                s=next(x for x in screen['signed_noisy_conditions'] if x['condition']==condition)
                mse=sum(r['metrics']['template_MSE_e2'] for r in group)/6
                l2=math.sqrt(sum(r['metrics']['template_SSE'] for r in group)/sum(r['metrics']['truth_energy'] for r in group))
                assert math.isclose(mse,s['template_MSE_e2'],rel_tol=1e-10,abs_tol=1e-10)
                assert math.isclose(l2,s['template_L2'],rel_tol=1e-10,abs_tol=1e-10)
            entry={'arm':name,'update':u,'summary':summary,'science':screen}
            if name=='v4.6':
                entry['checkpoint_path']=rel(ROOT/'training_v4_6/checkpoints/negative_noisy_repair'/('initial.pt' if u==0 else f'update_{u:04d}.pt'))
                entry['checkpoint_sha256']=sha(ROOT/entry['checkpoint_path'])
                assert entry['checkpoint_path']==checkpoints[name][u]['path']
                assert entry['checkpoint_sha256']==checkpoints[name][u]['sha256']
            entries.append(entry)
            observed[name]={r['case_id']:r for r in rows}
        assert set(observed['v4.5'])==set(observed['v4.6'])
        for case,a in observed['v4.5'].items():
            b=observed['v4.6'][case];key_matches(a,b)
            if u==0:assert a==b,'Zero-update development predictions differ'
            delta=b['metrics']['template_MSE_e2']-a['metrics']['template_MSE_e2']
            row={'update':u,'case_id':case,'mother_id':a['mother_id'],'geometry':a['geometry'],
                'condition':a['condition'],'input_kind':a['input_kind'],'signed_aux':a['signed_aux'],
                'v45_MSE_e2':a['metrics']['template_MSE_e2'],'v46_MSE_e2':b['metrics']['template_MSE_e2'],
                'paired_delta_MSE_e2':delta,'same_raw_label_cache':True}
            paired.append(row)
            parent_groups[(u,a['mother_id'],a['condition'],a['input_kind'])].append(row)
    assert len(paired)==4656
    parents=[{'update':k[0],'mother_id':k[1],'condition':k[2],'input_kind':k[3],
        'frames':len(v),'paired_mean_delta_MSE_e2':sum(r['paired_delta_MSE_e2'] for r in v)/len(v)}
        for k,v in sorted(parent_groups.items())]
    candidates=[r for r in entries if r['arm']=='v4.6' and r['science']['science_candidate']]
    candidates.sort(key=lambda r:(r['summary']['weak_L2'],r['summary']['normal_RMSE_e'],r['update']))
    selection={'candidate_count':len(candidates),'selected':candidates[0] if candidates else None,
        'eligible_candidates':candidates,'selection_rule':protocol['selection_rule'],
        'locked_development_used':False,'final_test_used':False,
        'next_action':'freeze_one_candidate_then_register_locked_confirmation' if candidates else 'close_bounded_data_contrast_no_additional_optimization'}
    write(attempt/'candidate_selection.json',selection)
    write(attempt/'comparison_summary.json',{'entries':entries,'actual_training_exposure':sampling['exposure'],
        'paired_case_count':4656,'parent_unit_statistics':'Descriptive paired summaries; no pixel-independent significance claim',
        'final_test_used':False})
    write(attempt/'paired_parent_summary.json',parents)
    with (attempt/'paired_cases.jsonl').open('x',encoding='utf-8') as f:
        for row in paired:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.environ['MPLCONFIGDIR']=str(ROOT/'_runtime/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('Main noisy weak','weak_L2',None),('Main noisy normal','normal_RMSE_e',None),
        ('Negative weak','template_L2','negative_weak'),('Positive weak','template_L2','positive_weak'),
        ('Negative normal','template_RMSE_e','negative_normal'),('Positive normal','template_RMSE_e','positive_normal')]
    fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
    for ax,(title,key,condition) in zip(axes.flat,panels):
        for arm,color,marker in [('v4.5','#555555','o'),('v4.6','#0066AA','s')]:
            es=[e for e in entries if e['arm']==arm]
            values=[e['summary'][key] if condition is None else next(r[key] for r in e['science']['signed_noisy_conditions'] if r['condition']==condition) for e in es]
            ax.plot([e['update'] for e in es],values,color=color,marker=marker,label=arm)
        if 'L2' in key:ax.axhline(1,color='#333333',ls=':',lw=1.2)
        elif condition:
            row=next(r for r in entries[-1]['science']['signed_noisy_conditions'] if r['condition']==condition)
            ax.axhline(math.sqrt(row['MSE_limit_e2']),color='#333333',ls=':',lw=1.2)
        ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Pooled L2' if 'L2' in key else 'Template RMSE (e)');ax.grid(alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncols=2)
    fig.savefig(attempt/'paired_comparison.png',dpi=160)
    fig.savefig(attempt/'paired_comparison.svg')
    plt.close(fig)
    from PIL import Image
    with Image.open(attempt/'paired_comparison.png') as im:
        im.load();assert im.width>=1000 and im.height>=1000
        image_size=list(im.size)
    report=['# v4.6 负含噪母场景和噪声覆盖对照','',
        '形成开发候选，须冻结单一权重后再登记独立确认。' if candidates else '六个检查点均未形成开发候选。本次有限数据对照关闭，不追加种子、步数、损失或采样试验。','',
        '全部结果来自已反复查看的776帧开发集。新确认集和最终测试均未打开；本报告不能提供独立泛化或真实测温结论。','',
        '相对v4.5，仅对513个负含噪位置执行预定母体/噪声重采样，其中11个位置恰好返回旧case，因此实际ID变化502个；其余1487个位置逐项原样保留。两负含噪条件各覆盖36母体、每母体两次独立噪声，每个raw被抽3或4次。生成432条TRAIN输入，实际新增反演与训练输入120条；新正组及无噪/量化组未用于本次训练。','',
        '母体覆盖与噪声重复同时改变，不能分别归因。固定最终连续谱、旧初始化与空Adam、优化器、预算、条件时序和全部筛查相同。温度逐帧等于物理基线，属于结构恒等。','',
        '|更新|旧eligible|新增筛查|联合候选|主weak L2|主normal RMSE e|失败signed组|',
        '|---:|---|---|---|---:|---:|---|']
    for e in entries:
        if e['arm']!='v4.6':continue
        s,c=e['summary'],e['science']
        failures=', '.join(x['condition'] for x in c['signed_noisy_conditions'] if not x['passed']) or '无'
        report.append(f"|{e['update']}|{s['eligible']}|{c['new_development_screen_pass']}|{c['science_candidate']}|{s['weak_L2']:.6f}|{s['normal_RMSE_e']:.4f}|{failures}|")
    report+=['','![同更新配对曲线](paired_comparison.png)','','|500更新|条件|模板RMSE e|冻结RMSE上限e|pooled L2|signed投影|通过|','|---|---|---:|---:|---:|---:|---|']
    for e in entries:
        if e['update']!=500:continue
        for r in e['science']['signed_noisy_conditions']:
            report.append(f"|{e['arm']}|{r['condition']}|{r['template_RMSE_e']:.4f}|{math.sqrt(r['MSE_limit_e2']):.4f}|{r['template_L2']:.6f}|{r['energy_weighted_signed_projection']:.6f}|{r['passed']}|")
    report+=['','每个更新点逐case核对raw、标签、物理缓存及其元数据身份，共4656组配对；0更新的全部776条记录与v4.5完全相同。每母体/条件/输入类型的描述性误差差值保留在paired_parent_summary.json；它们不是独立像素统计显著性。','',
        '完整旧16门禁、96保护组、温度筛查和signed分层见comparison_summary.json及原始训练报告。本文仅读取保存的结果，模型推理、物理反演和GPU调用均为0。']
    (attempt/'V4_6_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    outputs={p.name:sha(p) for p in attempt.iterdir() if p.is_file()}
    receipt={'status':'COMPLETE','time':datetime.now(timezone.utc).isoformat(),'paired_cases':4656,
        'candidate_count':len(candidates),'input_files_sha256':inputs,'outputs_sha256':outputs,
        'source_sha256':sha(Path(__file__)),'png_reopened_size':image_size,
        'model_inferences':0,'GPU_calls':0,'final_test_used':False,'locked_development_used':False}
    write(attempt/'analysis_receipt.json',receipt)
    for p in attempt.iterdir():
        destination=OUT/p.name
        with destination.open('xb') as f:f.write(p.read_bytes())
        assert sha(destination)==sha(p)
    print(json.dumps({'analysis_complete':True,'candidate_count':len(candidates),'paired_cases':4656,'report':str(OUT/'V4_6_REPORT.md')}))
    return receipt


if __name__=='__main__':run()
