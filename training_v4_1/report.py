"""Frozen-input audit, actual curves and individual candidate conclusions."""
import json
import hashlib
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from training_v4_1.common import *

def report():
    setup()
    plot_cache=runtime.RUNTIME/'matplotlib'
    plot_cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(plot_cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    old=read(OUT/'preserved_baseline.json')
    assert all(runtime.sha256(runtime.ROOT/p)==h for p,h in old['files'].items())
    manifest=read(OUT/'data_manifest.json'); rows=manifest['rows']
    assert len(rows)==3352 and not manifest['final_test_opened']
    assert not list((OLD_DATA/'raw').glob('v4_final_unopened_*'))
    matrix=read(OUT/'coverage_matrix.json'); assert len(matrix)==96 and all(r['new_count'] for r in matrix)
    inventory=[]; labels=set(); errors=[]
    for row in rows:
        raw_path=file_path(row,'raw_path'); assert runtime.sha256(raw_path)==row['raw_file_sha256']
        assert array_hash(npz(raw_path)['raw_dn'])==row['raw_hash']
        score=read(DATA/'physical_scores'/(row['case_id']+'.json'))
        assert score['raw_hash']==row['raw_hash'] and score['label_sha256']==row['label_sha256']
        cache=runtime.ROOT/score['cache_path']; meta=read(cache.with_suffix('.json'))
        assert not meta['labels_used'] and runtime.sha256(cache)==score['cache_sha256']
        assert runtime.sha256(cache.with_suffix('.json'))==score['cache_metadata_sha256']
        values=npz(cache); assert np.isfinite(values['state']).all() and np.isfinite(values['beta']).all()
        labels.add((row['label_path'],row['label_sha256']))
        inventory.append({'case_id':row['case_id'],'raw_hash':row['raw_hash'],'cache_key':score['cache_key'],
                          'cache_sha256':score['cache_sha256'],'cache_path':score['cache_path'],'label_sha256':row['label_sha256']})
    for path,h in labels: assert runtime.sha256(runtime.ROOT/path)==h
    write(OUT/'input_cache_hash_inventory.json',inventory)
    native=read(OUT/'native_checkpoint_receipt.json'); selection=read(OUT/'candidate_selection.json')
    from training_v4_1.diagnostics import make_diagnostics
    detection,protection_diagnostics=make_diagnostics()
    tikh=read(OUT/'tikhonov_selection.json'); assert not tikh['final_test_used']
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axes,('train','validation')):
        array=np.array([[next(r['new_count'] for r in matrix if r['split']==split and r['geometry']==g and r['condition']==c and r['input_kind']=='continuous')
                         for c in ('blank','weak','normal','strong')] for g in ('front','multipeak','ring','stripe','flat','smooth_random')])
        ax.imshow(array,vmin=0,vmax=8,cmap='Blues'); ax.set_title(split+' complete continuous coverage; quantization identical')
        ax.set_xticks(range(4),('blank','weak','normal','strong')); ax.set_yticks(range(6),('front','multipeak','ring','stripe','flat','smooth_random'))
        for y in range(6):
            for x in range(4): ax.text(x,y,str(array[y,x]),ha='center',va='center',color='white' if array[y,x]>5 else 'black')
    fig.tight_layout(); fig.savefig(figures/'coverage.png',dpi=140); plt.close(fig)
    baselines=Journal(OUT/'baseline_per_case.jsonl').rows(); assert len(baselines)==776*6
    assert len(Journal(OUT/'original_16_baseline_per_case.jsonl').rows())==16*5
    summaries=[]
    lines=['# TRAIN_V4_1_REPORT：保护性校正训练','',f'基于 `{BASE}`。物理骨干、旧权重、旧报告均保持原字节；不重跑旧954项，不打开最终测试。', '',
        '5797 参数、19 通道、六次共享有界校正。从 v4 update2000 的 CNN 参数初始化并重置 Adam。主标签均为真实合成 postblur 输出。每次更新严格为2个完整噪声、1个连续保护、1个仅量化保护；强度在各采样流平衡，5%有符号辅助样本轮流混入四种位置。', '',
        '## 覆盖与物理缓存', '',
        '训练选48个、验证选24个原母场景：每种温度几何分别8个/4个，同时平衡模板形态。每母场景补齐四强度的连续和仅量化输入。96个 split×温度几何×强度×输入类型单元全非空；原v4每split的24种干净组合缺12种，现已补齐，包括干净弱信号温度前沿。', '',
        '另有独立训练12个/验证6个有符号母场景，包含正负弱/正常系数与三种输入，共216帧。生成阶段为正负对使用相同强度缩放，仅为确保七波段非负、测量期望合法；实际强度、范围与缩放逐例记录在 signed_expectation_legality.json。原16检查帧从未用于梯度训练。', '',
        '共复用2560个原含噪raw，补充576个保护raw与216个有符号raw。旧物理缓存按actual raw、物理配置、标定及原求解器hash匹配复用；新缓存独立保存。标签独立，物理基线MSE仅在单独评分文件保存，绝不进入物理求解或推理。', '',
        '![完整覆盖](figures/coverage.png)', '', '## 保护损失与有限对照', '',
        '保留原真实合成监督，新增两项输出域hinge：模板MSE超过固定物理MSE+0.25 e²，温度MSE超过固定物理MSE+0.04 K²时分别惩罚。各项除以固定物理MSE+1（对应单位平方），避免近零分母；不除真值能量，不约束latent高频状态。所有四个microbatch均计算保护损失。', '',
        '推理只接收raw、cached_state、cached_beta。clean_identity、condition、case_id与truth均不传入网络。Adam lr=3e-5、累积4、clip=1，各正式试验1500次真实更新，每100次保存checkpoint与776帧分层验证。safe1与safe5若均执行，初始参数与完整抽样序列完全一致。', '',
        '初始两轮保留原残差特征。只有两个safe对照都无合格候选，且最保护性候选仍在>5%保护帧出现小残差（物理加权raw RMS<0.2）与大输出校正（模板>2e或色温>0.5K），才执行一次固定训练残差标尺的幅度保留特征消融，不加模块。', '',
        '合格候选必须原16/16、新保护各几何/强度/输入组均无超容差平均MSE退化，主含噪弱L2<1，含噪温度无系统性退化（各组温度MSE<=1.05×物理+.04）。<=0.85弱L2和<=12e正常RMSE是争取目标，不能当作既有性能。', '',
        '## 各checkpoint独立结果', '']
    for run in selection['completed_runs']:
        folder=OUT/run; receipt=read(folder/'training_receipt.json')
        training=Journal(folder/'training_curve.jsonl').rows(); events=Journal(folder/'validation_curve.jsonl').rows()
        assert len(training)==1500 and len(events)==15
        lines += [f'### {run}', '',
            '|试验|更新|原门禁/16|新保护组通过/96|含噪温度通过|主弱L2|正常RMSE e|可接受|',
            '|---|---:|---:|---|---|---:|---:|---|']
        for event in events:
            protected_pass=sum(g['passed'] for g in event['protection_groups'])
            lines.append(f'|{run}|{event["optimizer_update"]}|{event["original_gates_pass_count"]}|{protected_pass}/96|{event["noisy_temperature_pass"]}|{event["weak_L2"]:.4f}|{event["normal_RMSE_e"]:.4f}|{event["eligible"]}|')
            if not event['eligible']: errors.append({'kind':'REJECTED_CANDIDATE','run':run,**event})
            summaries.append({'run':run,**event})
        fig,axes=plt.subplots(1,3,figsize=(15,4.2))
        x=[r['optimizer_update'] for r in training]
        for key in ('total','main','safe_template','safe_temperature'): axes[0].plot(x,[r['loss'][key] for r in training],label=key,alpha=.55)
        axes[0].set_yscale('symlog',linthresh=.01); axes[0].legend(fontsize=8)
        axes[1].plot([e['optimizer_update'] for e in events],[e['weak_L2'] for e in events],'o-'); axes[1].axhline(1,color='black',linestyle='--')
        axes[2].plot([e['optimizer_update'] for e in events],[e['original_gates_pass_count'] for e in events],'o-',label='original gates')
        axes[2].axhline(16,color='black',linestyle='--'); axes[2].set_ylim(0,16.5)
        for ax,title in zip(axes,('real losses, separate protective terms','main noisy weak L2; zero=1','original regression count /16')):
            ax.set_title(title); ax.set_xlabel('optimizer update'); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figures/(run+'_training.png'),dpi=140); plt.close(fig)
        lines+=['',f'![{run}真实曲线](figures/{run}_training.png)','']
        if receipt.get('interruption_recovery'):
            recovery=read(runtime.ROOT/receipt['interruption_recovery'])
            lines += [f'{run}曾受外部进程中断影响，从更新{recovery["checkpoint_update"]}的已验证checkpoint恢复完整Adam状态。重算{recovery["replayed_updates"]}次更新，损失相对差最大为{recovery["max_replay_loss_relative_difference"]:.3g}；原曲线与部分验证原字节保存在 interruption_01。最终权重Adam step仍为1500，额外重算不计为新更新，不拼接权重。','']
            errors.append({'kind':'RECOVERED_EXTERNAL_INTERRUPTION','run':run,**recovery})
    lines+=['','## 同输入对照与空间细节','',
        'zero、old_B4、固定物理、v4 update100、v4 update2000和每个新候选在相同raw上逐例配对；另加独立验证选参的传统Tikhonov。旧预测只有raw、标签、缓存hash全匹配时才复用，逐例标记。主含噪组与独立有符号组分开，连续与仅量化保护逐帧保存。', '',
        '有符号支持域为abs(truth)>=0.2*max(abs(truth))，正目标用预测>60e、负目标用预测<−60e，分别报告召回与正负空白误报。未提高阈值、未删除失败帧。配对差值在各试验 paired_all_checkpoints.jsonl，所有更新单独保留，不能拼接不同权重的好处。', '',
        '|同一组输入|方法/独立checkpoint|帧数|模板RMSE e|模板L2|色温RMSE K|',
        '|---|---|---:|---:|---:|---:|']
    def table_row(label,method,rs):
        good=[r for r in rs if r['status']=='SUCCESS']
        n=sum(r['metrics']['pixels'] for r in good); sse=sum(r['metrics']['template_SSE'] for r in good)
        energy=sum(r['metrics']['truth_energy'] for r in good)
        ts=[r['metrics']['temperature_MSE_K2'] for r in good if r['metrics']['temperature_MSE_K2'] is not None]
        rmse=np.sqrt(sse/n) if n else None; l2=np.sqrt(sse/energy) if energy else None; temp=np.sqrt(np.mean(ts)) if ts else None
        fmt=lambda v:'N/A' if v is None else f'{v:.4f}'
        return f'|{label}|{method}|{len(rs)}（失败{len(rs)-len(good)}）|{fmt(rmse)}|{fmt(l2)}|{fmt(temp)}|'
    for kind,signed,group in [('noise',False,g) for g in ('blank','weak','normal','strong')]+[('noise',True,'negative')]:
        label=kind+'/'+('signed_aux/' if signed else 'main/')+group
        for method in ('zero','old_B4','fixed_physics','v4_update100','v4_update2000','tikhonov_validation_selected'):
            rs=[r for r in baselines if r['method']==method and r['input_kind']==kind and r['signed_aux']==signed and r['group']==group]
            lines.append(table_row(label,method,rs))
        for run in selection['completed_runs']:
            terminal=Journal(OUT/run/'validation_update_1500.jsonl').rows()
            rs=[r for r in terminal if r['input_kind']==kind and r['signed_aux']==signed and r['group']==group]
            lines.append(table_row(label,run+'/1500（末次独立权重）',rs))
            receipt=read(OUT/run/'training_receipt.json')
            if receipt['selected_checkpoint'] and not receipt['selected_checkpoint'].endswith('update_1500.pt'):
                update=int(Path(receipt['selected_checkpoint']).stem.split('_')[1])
                selected_rows=Journal(OUT/run/f'validation_update_{update:04d}.jsonl').rows()
                rs=[r for r in selected_rows if r['input_kind']==kind and r['signed_aux']==signed and r['group']==group]
                lines.append(table_row(label,run+f'/{update}（验证选中）',rs))
    lines+=['',
        f'传统解调平滑lambda独立从0.1、1、10、100、1000验证选择，选中 `{tikh["selected_lambda_decoder"]}`；它与物理求解器latent模板系数0.1不是同一个量。', '',
        '|解调lambda|验证目标|保留频带最小传递率|中位传递率|', '|---:|---:|---:|---:|']
    for result in tikh['grid']: lines.append(f'|{result["lambda_decoder"]}|{result["validation_objective"]:.6f}|{result["retained_passband_transfer_min"]:.4f}|{result["retained_passband_transfer_median"]:.4f}|')
    lines+=['','|解调lambda|正常模板RMSE e|正常投影系数|预测/真值空间梯度能量|','|---:|---:|---:|---:|']
    for result in tikh['grid']:
        group=next(g for g in result['groups'] if g['condition']=='normal')
        ratio=group['mean_gradient_energy']/group['mean_truth_gradient_energy']
        lines.append(f'|{result["lambda_decoder"]}|{group["template_RMSE_e"]:.4f}|{group["mean_signed_projection"]:.4f}|{ratio:.4f}|')
    lines+=['','## 新保护与有符号诊断','',
        '|试验/末次独立权重|模板退化帧/保护帧|温度退化帧/保护帧|小残差大校正比例|',
        '|---|---:|---:|---:|']
    for item in protection_diagnostics:
        if item['optimizer_update']==1500:
            lines.append(f'|{item["run"]}/1500|{item["template_degraded_frames"]}/{item["frames"]}|{item["temperature_degraded_frames"]}/{item["frames"]}|{item["small_residual_large_correction_fraction"]:.4f}|')
    lines+=['','退化以逐帧输出MSE超过同输入固定物理基线MSE加既定小容差定义。完整776帧×15次验证、96个保护子组和原16项均保存；该表是诊断，不据此删除帧或放宽筛选。', '',
        '|负系数含噪输入|独立方法/权重|帧数|模板L2（zero=1）|负目标召回 <−60e|',
        '|---|---|---:|---:|---:|']
    terminal_methods={'zero','old_B4','fixed_physics','v4_update100','v4_update2000','tikhonov_validation_selected'}|{r+'/1500' for r in selection['completed_runs']}
    for item in detection:
        if item['method'] in terminal_methods and item['input_kind']=='noise' and item['condition'] in ('negative_weak','negative_normal'):
            lines.append(f'|{item["condition"]}|{item["method"]}|{item["frames"]}|{item["template_L2"]:.4f}|{item["mean_negative_recall60"]:.4f}|')
    lines+=['','正负召回、正负空白误报、每个输入类型、每个独立checkpoint和空间细节统计见 signed_detection_and_detail_summary.json。支持域均按abs(truth)定义；原16项同输入逐权重配对另见 original_16_paired_checkpoints.jsonl。','']
    lines+=['','tikhonov_selection.json与tikhonov_validation_grid.jsonl同时记录真值和预测的空间梯度能量、模板投影系数。降低噪声梯度不等于保留细节；大lambda的真实结构幅度损失须与RMSE一起判断。', '', '## 本轮结论与交付','']
    best=selection['selected']
    if best: lines.append(f'合格候选为 `{best["run"]}` 更新 `{best["optimizer_update"]}`，主弱L2={best["weak_L2"]:.4f}，正常模板RMSE={best["normal_RMSE_e"]:.4f} e。所有保护与收益均属于该同一checkpoint。')
    else: lines.append('本轮已完成实际训练与checkpoint保存，但没有一个同一checkpoint同时通过全部保护和含噪收益条件。所有末次权重按诊断用途交付，旧模型保持不变；不合并不同checkpoint的优点。')
    ablation=read(OUT/'feature_ablation_decision.json')
    if ablation.get('triggered'):
        reference=read(OUT/'amplitude_feature_reference.json')
        terminal=Journal(OUT/'amplitude_safe5/validation_curve.jsonl').rows()[-1]
        lines += ['',f'幅度消融按预登记规则触发：safe5 更新{ablation["diagnostic_checkpoint_update"]}仍有{ablation["small_residual_large_correction_fraction"]:.1%}保护帧满足小残差大校正。固定标尺由64个训练raw计算，weighted-adjoint RMS中位数={reference["median_weighted_adjoint_RMS"]:.10f}。末次该现象比例为{terminal["small_residual_large_correction_fraction"]:.2%}，原16项过{terminal["original_gates_pass_count"]}项、弱L2={terminal["weak_L2"]:.4f}、正常RMSE={terminal["normal_RMSE_e"]:.4f}e；该单一特征变化未产生合格候选。', '']
        for name in selection['completed_runs']:
            endpoint=Journal(OUT/name/'validation_curve.jsonl').rows()[-1]
            negative=next(r for r in detection if r['method']==name+'/1500' and r['input_kind']=='noise' and r['condition']=='negative_normal')
            lines.append(f'{name}末次原门禁{endpoint["original_gates_pass_count"]}/16，新保护通过{sum(g["passed"] for g in endpoint["protection_groups"])}/96，含噪负正常目标召回={negative["mean_negative_recall60"]:.4f}。')
        physical_negative=next(r for r in detection if r['method']=='fixed_physics' and r['input_kind']=='noise' and r['condition']=='negative_normal')
        lines += ['',f'相同含噪负正常输入下，固定物理的负目标召回为{physical_negative["mean_negative_recall60"]:.4f}。这些结果分别归属各checkpoint，构成本轮不选替换权重的证据。']
    lines+=['','新权重在 training_v4_1/checkpoints/：每组全部15个每100次检查点与末次导出均保留。native_checkpoint_receipt.json核查重置Adam、真实1500 step、5797参数、原物理缓冲区一致和重新加载重复推理。6项测试通过。raw/cache/标签库存见 input_cache_hash_inventory.json，覆盖、保护逐例、原16检查、配对与失败记录均保留。', '',
        '复现按 training_v4_1/README.md：data→tests→prepare_cache→safe1→controls→tikhonov→compare→verify→report。原目录的训练试验不可覆盖，另建E/D工作目录；CLI环境必须是同一本机GPU PyTorch。', '',
        '这些是开发与回归结果，最终测试未打开，未提供实测相机泛化承诺。光谱响应/QE仍沿用原工程合成配置。']
    errors += [r for r in Journal(OUT/'cache_per_case.jsonl').rows() if r['status']!='SUCCESS']
    errors += [r for r in baselines if r['status']!='SUCCESS']
    for p in OUT.rglob('failure_record.json'): errors.append(read(p))
    if (OUT/'data_initial_uint16_failure.log').exists(): errors.append({'kind':'DATA_GENERATION_PORTABILITY_FIX','log':'data_initial_uint16_failure.log','fixed_before_training':True})
    write(OUT/'failure_records.json',errors); write(OUT/'all_candidates_summary.json',summaries)
    (OUT/'TRAIN_V4_1_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    images=[]
    for p in figures.glob('*.png'):
        with Image.open(p) as im: im.load(); size=list(im.size)
        images.append({'path':rel(p),'sha256':runtime.sha256(p),'size':size})
    write(OUT/'delivery_receipt.json',{'raw_cache_pairs_verified':len(inventory),'labels_verified':len(labels),
        'original_files_preserved':len(old['files']),'runs':selection['completed_runs'],'updates_each_run':1500,
        'qualified_candidate_available':selection['qualified_model_available'],'required_coverage_cells':96,
        'native_checkpoints_reopened':True,'final_test_generated_or_evaluated':0,'old_954_rerun':False,
        'images':images,'report_sha256':runtime.sha256(OUT/'TRAIN_V4_1_REPORT.md')})
    print({'report_complete':True,'runs':selection['completed_runs'],'qualified':selection['qualified_model_available']})

if __name__=='__main__': report()
