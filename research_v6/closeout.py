"""Close a registered numerical-stop branch using existing evidence only.

Historical output differences are observational accounting, not a new inverse
or a replacement for the failed three-physical-component identity.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import math
from pathlib import Path
from . import common
common.configure_runtime()
import numpy as np


def _file(path,expected,parent):
    p=(common.ROOT/path).resolve();q=(common.ROOT/parent).resolve()
    assert q in p.parents,'Read outside already opened historical cohort'
    assert common.sha(p)==expected,'Historical immutable payload changed'
    return p


def _one(path,key):
    with np.load(path,allow_pickle=False) as z:
        a=z[key].astype(np.float64,copy=True)
    while a.ndim>2 and a.shape[0]==1:a=a[0]
    assert a.shape==(256,256) and np.isfinite(a).all()
    return a


def historical_accounting():
    output=common.REPORTS/'HISTORICAL_O0_O1_OUTPUT_ACCOUNTING.json'
    if output.exists():return common.read_json(output)
    p=common.V5_REPORTS/'ORACLE_LADDER.jsonl'
    records=list(common.jsonl(p));groups=defaultdict(dict)
    for r in records:
        if r['level'] in ('O0','O1'):groups[r['case_id']][r['level']]=r
    assert len(groups)==180 and all(set(v)=={'O0','O1'} for v in groups.values())
    refs={r['case_id']:r for r in common.read_json(common.CODE/'input_inventory.json')['rows'] if r['cohort']=='oracle'}
    rows=[]
    for cid,pair in sorted(groups.items()):
        a,b=pair['O0'],pair['O1'];ref=refs[cid]
        assert a['raw_file_sha256']==b['raw_file_sha256']==ref['raw_reference']['sha256']
        assert a['label_sha256']==b['label_sha256']==ref['label_reference']['sha256']
        assert a['metrics']['raw_weight_sha256']==b['metrics']['raw_weight_sha256']
        assert a['linear_options']==b['linear_options'] and a['linear_options']['alpha_g']==0.0
        x0=_one(_file(a['prediction_path'],a['prediction_sha256'],'outputs/research_v5/run_001/oracle_predictions'),'template_postblur_e')
        x1=_one(_file(b['prediction_path'],b['prediction_sha256'],'outputs/research_v5/run_001/oracle_predictions'),'template_postblur_e')
        target=_one(_file(ref['label_reference']['path'],ref['label_reference']['sha256'],'outputs/research_v5/run_001/oracle_data/diagnostic_labels'),'template_postblur_e')
        e0=x0-target;delta=x1-x0;e1=x1-target
        closure=e1-e0-delta
        m0=float(np.mean(e0**2));md=float(np.mean(delta**2));cross=float(2*np.mean(e0*delta));m1=float(np.mean(e1**2))
        c=float(np.corrcoef(e1.ravel(),delta.ravel())[0,1]) if e1.std()>0 and delta.std()>0 else None
        row={'case_id':cid,'mother_id':a['mother_id'],'condition':a['condition'],'input_kind':a['input_kind'],
          'group':a['group'],'sign':'positive' if a['coefficient_e']>0 else 'negative' if a['coefficient_e']<0 else 'blank',
          'raw_sha256':a['raw_file_sha256'],'label_sha256':a['label_sha256'],'pixels':target.size,
          'O0_MSE_e2':m0,'source_change_MSE_e2':md,'twice_cross_O0_source_change_e2':cross,'O1_MSE_e2':m1,
          'accounted_O1_MSE_e2':m0+md+cross,'field_closure_RMS_e':float(np.sqrt(np.mean(closure**2))),
          'MSE_closure_abs_e2':abs(m1-m0-md-cross),'O1_error_source_change_Pearson':c,
          'source_change_scope':'O1-O0 returned-output change, including finite solve differences; not certified K-inverse leakage',
          'alpha_g':0.0,'new_inverse_executed':False,
          'prediction_references':{'O0':{'path':a['prediction_path'],'sha256':a['prediction_sha256']},
                                   'O1':{'path':b['prediction_path'],'sha256':b['prediction_sha256']}}}
        rows.append(row);common.append_jsonl(common.REPORTS/'historical_o0_o1_output_accounting.jsonl',row)
    summaries=[]
    for kind in ('continuous','quantization','noise'):
        for sign in ('positive','negative','blank'):
            for group in ('normal','weak','blank'):
                chosen=[r for r in rows if r['input_kind']==kind and r['sign']==sign and r['group']==group]
                if not chosen:continue
                fields=('O0_MSE_e2','source_change_MSE_e2','twice_cross_O0_source_change_e2','O1_MSE_e2')
                means={k:sum(r[k] for r in chosen)/len(chosen) for k in fields}
                mothers={r['mother_id']:r['O1_MSE_e2']-r['O0_MSE_e2'] for r in chosen}
                interval=common.frozen_v5('statistics').mother_bootstrap(mothers)
                summaries.append({'input_kind':kind,'sign':sign,'group':group,'cases':len(chosen),
                  'mothers':len(mothers),**means,'O0_RMSE_e':math.sqrt(means['O0_MSE_e2']),
                  'source_change_RMS_e':math.sqrt(means['source_change_MSE_e2']),
                  'O1_RMSE_e':math.sqrt(means['O1_MSE_e2']),
                  'source_change_term_divided_by_excess_MSE':means['source_change_MSE_e2']/(means['O1_MSE_e2']-means['O0_MSE_e2'])
                      if means['O1_MSE_e2']!=means['O0_MSE_e2'] else None,
                  'paired_mother_excess_MSE':interval,
                  'max_field_closure_RMS_e':max(r['field_closure_RMS_e'] for r in chosen),
                  'mean_error_source_change_Pearson':sum(r['O1_error_source_change_Pearson'] for r in chosen)/len(chosen)
                      if all(r['O1_error_source_change_Pearson'] is not None for r in chosen) else None})
    result={'schema':'v6-historical-output-difference-accounting-001','time':common.now(),'all_passed':True,
      'historical_record_sha256':common.sha(p),'pairs':180,'mothers':12,'alpha_g':0.0,
      'identity':'E1=E0+D; MSE1=MSE0+MSE_D+2 mean(E0 D), D=stored S1-stored S0',
      'new_reconstruction_or_optimization_executed':False,'pure_physical_leakage_certified':False,
      'scope':'read-only historical frozen outputs; no substitute for failed v6 alpha1 numerical identity',
      'max_field_closure_RMS_e':max(r['field_closure_RMS_e'] for r in rows),
      'max_MSE_closure_abs_e2':max(r['MSE_closure_abs_e2'] for r in rows),'groups':summaries,
      'per_case_evidence_sha256':common.sha(common.REPORTS/'historical_o0_o1_output_accounting.jsonl')}
    common.write_json(output,result)
    return result


def plots(decompositions,historical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=common.ed_path(common.REPORTS/'figures');figures.mkdir(parents=True,exist_ok=True)
    failure=next(r for r in decompositions if not r['numerical_gate_passed'])
    data=common.arrays(common.ROOT/failure['component_arrays']['per_source']['path'])
    shared=common.arrays(common.ROOT/failure['component_arrays']['shared_noise_and_bias']['path'])
    panels=[('Actual template error',data['template_postblur_error_e']),('Leakage',data['leakage_postblur_e']),
       ('Exact n from stored input',shared['noise_postblur_e']),('Regularization bias',shared['regularization_bias_postblur_e']),
       ('Three-component sum',data['leakage_postblur_e']+shared['noise_postblur_e']+shared['regularization_bias_postblur_e']),
       ('Recomposition residual',data['recomposition_residual_postblur_e'])]
    fig,axes=plt.subplots(2,3,figsize=(12,7.2),constrained_layout=True)
    for ax,(label,a) in zip(axes.ravel(),panels):
        limit=max(float(np.max(np.abs(a))),1e-8)
        im=ax.imshow(a,cmap='RdBu_r',vmin=-limit,vmax=limit,origin='lower')
        ax.set_title(label+'\nRMS '+format(float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2))),'.5g')+' e',fontsize=12)
        ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,shrink=.84,label='e')
    fig.suptitle('First registered synthetic input: C_true + estimated beta (float32, alpha=1)\nNumerical identity failed; each panel uses its own scale',fontsize=14)
    f=figures/'stage_A_first_failure_components.png';fig.savefig(f,dpi=160);plt.close(fig)
    normal=[r for r in historical['groups'] if r['input_kind']=='noise' and r['group']=='normal']
    normal.sort(key=lambda r:r['sign'],reverse=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    x=np.arange(len(normal));width=.25
    for offset,key,label,color in ((-1,'O0_RMSE_e','O0 true continuum','#2c6ca2'),
                                  (0,'source_change_RMS_e','Stored O1 - O0 change','#75509a'),
                                  (1,'O1_RMSE_e','O1 estimated continuum','#d07635')):
        axes[0].bar(x+offset*width,[r[key] for r in normal],width,label=label,color=color)
    axes[0].set_xticks(x,[r['sign'] for r in normal]);axes[0].set_ylabel('Template RMS / RMSE (e)')
    axes[0].set_ylim(0,100)
    axes[0].legend(fontsize=9);axes[0].set_title('Historical noisy normal templates')
    for offset,key,label,color in ((-1,'O0_MSE_e2','O0 MSE','#2c6ca2'),
                                  (0,'source_change_MSE_e2','Source-change term','#75509a'),
                                  (1,'twice_cross_O0_source_change_e2','Cross term','#d07635')):
        axes[1].bar(x+offset*width,[r[key] for r in normal],width,label=label,color=color)
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_xticks(x,[r['sign'] for r in normal])
    axes[1].set_ylim(0,6500)
    axes[1].set_ylabel('Exact output-error accounting (e squared)');axes[1].legend(fontsize=9)
    axes[1].set_title('Cross terms retained')
    fig.suptitle('Frozen v5 outputs: alpha=0, 12 mothers\nStored output change includes finite-solver effects',fontsize=12)
    f2=figures/'historical_O0_O1_output_accounting.png';fig.savefig(f2,dpi=160);plt.close(fig)
    return [{'path':common.relative(p),'sha256':common.sha(p),'bytes':p.stat().st_size} for p in (f,f2)]


def audit_stop(decision,decompositions,rows):
    protocol=common.require_registered()
    assert decision['status']=='STOP_NUMERICAL_OR_ORACLE_AUDIT' and not decision['B_eligible']
    assert not (common.REPORTS/'PHASE_ACTIVE.json').exists()
    assert (common.REPORTS/'A_ONCE_STARTED.json').is_file()
    assert len(rows)==5040 and len({(r['case_id'],r['method']) for r in rows})==5040
    refs=common.read_json(common.CODE/'input_inventory.json')['rows']
    before=common.read_json(common.REPORTS/'INPUT_INTEGRITY_BEFORE.json')
    after=common.read_json(common.REPORTS/'INPUT_INTEGRITY_AFTER_A.json')
    assert before['file_hashes_sha256']==after['file_hashes_sha256']
    assert before['rows']==after['rows']==720 and before['unique_files_hashed']==after['unique_files_hashed']==960
    assert before['reference_sha256']==common.canonical_hash(refs)
    executed_order=sorted(refs,key=lambda r:(0 if r['cohort']=='oracle' else 1,
        0 if r['input_kind']=='continuous' else 1 if r['input_kind']=='quantized' else 2,
        0 if r['group']=='normal' else 1,r['case_id']))
    assert after['reference_sha256']==common.canonical_hash(executed_order)
    historical=common.read_json(common.REPORTS/'HISTORICAL_O0_O1_OUTPUT_ACCOUNTING.json')
    assert historical['pairs']==180 and historical['max_field_closure_RMS_e']<1e-9
    assert historical['max_MSE_closure_abs_e2']<1e-8
    old={(r['case_id'],r['level']):r for r in common.jsonl(common.V5_REPORTS/'ORACLE_LADDER.jsonl')}
    for row in common.jsonl(common.REPORTS/'historical_o0_o1_output_accounting.jsonl'):
        for level in ('O0','O1'):
            record=old[row['case_id'],level]
            assert record['metrics']['pixels']==row['pixels']==65536
            assert math.isclose(row[level+'_MSE_e2'],record['metrics']['template_MSE_e2'],rel_tol=1e-5,abs_tol=1e-7)
    original=common.read_json(common.REPORTS/'BASELINE_INVENTORY_BEFORE.json')
    changed=[]
    for item in original['files']:
        if common.sha(common.ROOT/item['path'])!=item['sha256']:changed.append(item['path'])
    assert not changed,changed
    failed=[d for d in decompositions if not d['numerical_gate_passed']]
    assert len(failed)==1 and failed[0]['case_id']==decision['stop_reason']['case_id']
    counts=Counter(r['status'] for r in rows)
    # Source inference, CG convergence, identity closure and information claims
    # are distinct audit outcomes; accepted fields do not rescue closure.
    result={'schema':'v6-stop-branch-deliverable-audit-001','time':common.now(),'all_passed':True,
      'completion_scope':'requested mandatory stop and numerical diagnosis, not a completed720-case scientific comparison',
      'registered_raw':len(refs),'processed_raw':decision['processed_raw'],'matrix_rows_with_fixed_denominator':len(rows),
      'row_status_counts':dict(counts),'decomposition_rows':len(decompositions),'failed_identities':len(failed),
      'source_inference_failures':sum(r['status']=='CONTINUUM_SOURCE_FAILED' for r in rows),
      'linear_estimate_failures':sum(r['status']=='LINEAR_SOLVE_FAILED' for r in rows),
      'information_insufficiency_demonstrated':False,'baseline_files_byte_checked':len(original['files']),
      'baseline_changed_files':changed,'old_weight_hashes_unchanged':protocol['old_weight_sha256'],
      'input_before_after_identical':True,'signed36_final64_payloads_opened':False,'new_CNN_training_steps':0,
      'reference_list_order_changed_without_membership_or_byte_changes':True,
      'historical_180_pair_accounting_matches_stored_metrics_and_closes':True,
      'B_not_executed_reason':'Mandatory Stage A numerical stop blocks joint reconstruction',
      'design_not_executed_reason':'A did not reach an interpretable algorithm-gain decision; do not bypass numerical stop with optimization',
      'task_requirements':{
        'fixed_base_and_new_branch':'registered protocol + first v6 commit',
        'same_raw_source_alpha_W_budget':'input audit + four executed decomposition records',
        'three_component_identity_and_strict_stop':'failed identity + bounded residual/fixed-RHS64 diagnosis',
        'all_case_records_and_invalid_denominators':'5040 unique rows including5035 NOT_EXECUTED',
        'C_v3_no_double_blur_raw_only':'13 CPU contracts + exact selected inverse/source receipts',
        'input_source_weight_audits':'before/after960 input files; baseline inventory; registered hashes',
        'B_solver_grid_and16_regressions':'conditional not eligible; V6_B_REPORT documents no execution',
        'fixed_photon_budget_design':'conditional not eligible after numerical stop',
        'independent_confirmation_final':'kept sealed; no qualifying candidate',
        'scientific_visual_QA':'figures generated; manual visual receipt still required',
        'GitHub_delivery':'pending external success receipt'} }
    common.write_json(common.REPORTS/'CLOSEOUT_AUDIT.json',result)
    return result


def write_reports(decision,decompositions,historical,audit,figures):
    failed=next(r for r in decompositions if not r['numerical_gate_passed'])
    diagnostic=common.read_json(common.REPORTS/'FAILED_CASE_FLOAT64_DIAGNOSTIC.json')
    residual=diagnostic['finite_solve_residual_diagnostic']
    h=[r for r in historical['groups'] if r['input_kind']=='noise' and r['group']=='normal']
    lines=['# v6 阶段 A：数值门槛停止报告','',
      '**结论：阶段 A 按预登记数值门槛提前停止，本轮关闭。** 720 个 raw 已全部完成哈希／形状审计；正式推理处理1个已查看的 oracle raw，得到原 v3 输出1行及线性置换4行。阶段 B、设计优化、训练与封存评估均未执行。',
      '', '## 首个实际输入的分解','',f"输入：`{failed['case_id']}`，negative_normal、continuous；固定 alpha=1、W、正则、float32 和200次最大PCG预算。",'',
      '|连续谱|背景|实际模板误差 RMS (e)|泄漏 RMS (e)|噪声 RMS (e)|正则偏差 RMS (e)|重组残差 RMS (e)|门槛|',
      '|---|---|---:|---:|---:|---:|---:|---|']
    for r in decompositions:
        c=r['postblur_components']
        lines.append(f"|{r['continuum_source']}|{r['beta_mode']}|{r['postblur_error']['rms']:.6g}|{c['leakage']['rms']:.6g}|{c['noise']['rms']:.6g}|{c['regularization_bias']['rms']:.6g}|{r['postblur_recomposition_residual']['rms']:.6g}|{'通过' if r['numerical_gate_passed'] else '失败'}|")
    lines+=['',f"C_true＋estimated_beta 的重组残差 {failed['postblur_recomposition_residual']['rms']:.6g} e，超过0.05 e绝对门槛，同时未满足0.005相对门槛。实际解与三项分量CG均报告收敛；法方程代数相对残差 {failed['normal_equation_algebra_relative_equilibrated']:.3g}，未发现尺度、beta单位、重复blur或W改变。",
      '',f"按相同系统和预算计算实际法方程残差的贡献，其 RMS 为{residual['actual_residual_contribution_postblur']['rms']:.6g} e；解释后的剩余为{residual['unexplained_after_actual_residual']['rms']:.6g} e。固定原始RHS和标定缓冲的float64参考仍有{diagnostic['float64_reference']['diagnostic']['postblur_recomposition_residual']['rms']:.6g} e重组差。主要原因是相对大RHS停止的有限PCG逆随RHS而变，不能视为精确可加的K逆。",
      '', '这是数值停止结论，不能作为信息耗尽、真实器件极限或算法总体不可能改善的证据。原门槛、权重与输出均未替换。',
      '', '## 历史 O0／O1 的返回输出差异核对','',
      '只读复用12个已查看oracle母场景的180对历史输出，alpha=0。令D为保存的O1输出减O0输出，则E1=E0+D及MSE1=MSE0+MSE_D+2mean(E0D)可直接核验。D包含连续谱来源改变导致的返回输出差异和有限求解差异；它不能冒充本轮尚未通过恒等检查的纯物理K逆泄漏。','',
      '|正常含噪组|O0 RMSE (e)|D RMS (e)|O1 RMSE (e)|O0 MSE|D平方项|交叉项|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in h:lines.append(f"|{r['sign']}|{r['O0_RMSE_e']:.5g}|{r['source_change_RMS_e']:.5g}|{r['O1_RMSE_e']:.5g}|{r['O0_MSE_e2']:.6g}|{r['source_change_MSE_e2']:.6g}|{r['twice_cross_O0_source_change_e2']:.6g}|")
    lines+=['',f"180对输出的最大场恒等残差为{historical['max_field_closure_RMS_e']:.3g} e。全部normal正负、weak正负、blank及三个输入版本的历史核对均单列；本轮alpha=1统计不会与它混合。",
      '', '## 未执行项与决策范围','',
      'continuum_source_comparison.jsonl保留5040个固定case×method记录：5个实际返回，5035个明确NOT_EXECUTED。development36母场景的正式置换、正常含噪正负、弱信号和blank的本轮总体比较均未执行。它们未被删除，也未被记为通过；全组聚合为不完整。',
      '', '唯一下一方向：**终止此轮算法研究，保留冻结v3为比较基准。** 当前没有合格无网络候选；不启动B、调参或封存确认。',
      '', '## 复核与交付','',
      '注册协议：research_v6/protocol.json；逐例物理分解：continuum_leakage_decomposition.jsonl；数值归因：FAILED_CASE_FLOAT64_DIAGNOSTIC.json；历史核对：historical_o0_o1_output_accounting.jsonl；不变性和范围审计：CLOSEOUT_AUDIT.json。',
      '', '![首次数值失败的分解](figures/stage_A_first_failure_components.png)',
      '', '![历史O0和O1返回输出核对](figures/historical_O0_O1_output_accounting.png)',
      '', '所有量仍属于冻结合成标定中的postblur色温和有符号数值模板。外部定标、非灰体、真实光谱、真实物种浓度、实际设备与10kHz端到端吞吐量，以及36 signed confirmation和64 final母场景均未验证。']
    snapshot=common.REPORTS/'V6_A_REPORT_STAGE_SNAPSHOT.md'
    if not snapshot.exists():snapshot.write_text((common.REPORTS/'V6_A_REPORT.md').read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    (common.REPORTS/'V6_A_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    decision_lines=['# DECISION_V6','', '**本轮按阶段 A 数值门槛关闭；科学全矩阵未完成。**','',
      '1. **O0≈8 e至O1≈70 e由多少连续谱投影造成？** 历史返回输出的来源变化D是主要平方项，正负明细如下；精确的三物理分量归因未通过本轮数值门槛，不能把D直接称为纯物理泄漏。']
    for r in h:decision_lines.append(f"   {r['sign']}：MSE1={r['O0_MSE_e2']:.6g}+{r['source_change_MSE_e2']:.6g}+({r['twice_cross_O0_source_change_e2']:.6g})={r['O1_MSE_e2']:.6g} e²；O0/O1={r['O0_RMSE_e']:.4g}/{r['O1_RMSE_e']:.4g} e。")
    decision_lines+=['',
      '2. **raw-only v3连续谱能否显著改善正常含噪正负模板？** 未建立。预登记停止发生在首个negative_normal连续无噪输入；该输入的true-beta模板RMSE为Cv5 0.11984 e、Cv3 0.93004 e，原v3整体为1.05679 e。只有这一例，正组和含噪组未执行，不能给出总体或配对显著改善结论。',
      '',f"3. **可观察频带联合优化是否优于v3？** 未通过进入资格，B未执行。实际CG收敛仍留下{failed['postblur_recomposition_residual']['rms']:.6g} e恒等残差；数值残差归因将其解释到{residual['unexplained_after_actual_residual']['rms']:.6g} e。没有可宣称优于v3的候选。",'',
      '4. **唯一下一方向：终止此轮算法研究。** 保留冻结v3作为比较基准。本轮没有证据支持把数值门槛失败改写成光谱／硬件设计优先或信息已经耗尽。', '',
      '5. **未知项：** 尚缺可精确归因的完整连续谱泄漏、Cv3置换的正常含噪正负／弱信号／blank总体效果，以及B算法性能；外部标定、非灰体、真实光谱、实际气体温度／物种浓度、实际设备、10kHz端到端吞吐量及封存确认／最终测试均未验证。', '',
      '完整证据、未执行分母和图示见V6_A_REPORT.md。神经网络训练0步；旧文件、权重与输入哈希不变；36 signed confirmation及64 final母场景保持封存。']
    (common.REPORTS/'DECISION_V6.md').write_text('\n'.join(decision_lines)+'\n',encoding='utf-8',newline='\n')
    (common.REPORTS/'V6_B_REPORT.md').write_text('# v6 阶段 B：未执行\n\n阶段A的预登记数值恒等门槛失败，B_eligible=false。联合可观察带限算子／求解器、小正则网格、16回归与36开发母场景B比较均未执行；没有候选、不劣性或10%改善结论。\n\n未创建占位bandlimited_operator.py或variable_projection_solver.py，未进行CNN训练、独立确认或最终集读取。详细停止原因见V6_A_REPORT.md及FAILED_CASE_FLOAT64_DIAGNOSTIC.json。\n',encoding='utf-8',newline='\n')
    common.write_json(common.REPORTS/'FINAL_STOP_DECISION.json',{'time':common.now(),'status':'CLOSED_BY_REQUIRED_STAGE_A_NUMERICAL_STOP',
      'next_research_direction':'terminate_this_algorithm_round','scientific_A_matrix_complete':False,
      'B_executed':False,'design_optimization_executed':False,'new_training_steps':0,
      'requested_stop_and_diagnosis_complete':True,'sealed36_and64_opened':False,
      'historical_accounting_scope':'read-only output changes, no new inverse and no certified pureK leakage',
      'figures':figures,'source_sha256':{common.relative(common.CODE/'closeout.py'):common.sha(common.CODE/'closeout.py')}})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    if (common.REPORTS/'FINAL_STOP_DECISION.json').is_file() and (common.REPORTS/'CLOSEOUT_AUDIT.json').is_file():
        common.require_registered()
        visual=common.REPORTS/'VISUAL_QA_ADDENDUM.json'
        record=common.read_json(visual if visual.exists() else common.REPORTS/'FINAL_STOP_DECISION.json')
        for item in record['figures']:assert common.sha(common.ROOT/item['path'])==item['sha256']
        print({'status':'EXISTING_REQUIRED_STOP_RECEIPT','reconstruction_or_document_overwrite':False},flush=True)
        return
    decision=common.read_json(common.REPORTS/'A_DECISION.json')
    decomp=list(common.jsonl(common.REPORTS/'continuum_leakage_decomposition.jsonl'))
    rows=list(common.jsonl(common.REPORTS/'continuum_source_comparison.jsonl'))
    historical=historical_accounting();figures=plots(decomp,historical)
    audit=audit_stop(decision,decomp,rows)
    write_reports(decision,decomp,historical,audit,figures)
    print({'status':'CLOSED_BY_REQUIRED_NUMERICAL_STOP','historical_pairs':historical['pairs'],
           'fixed_matrix_records':len(rows),'old_files_byte_checked':audit['baseline_files_byte_checked'],
           'figures':[r['path'] for r in figures]},flush=True)


if __name__=='__main__':main()
