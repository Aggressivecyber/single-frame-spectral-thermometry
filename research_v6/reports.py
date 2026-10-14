"""Evidence-bound Stage A decisions and explicit conditional phase status."""
from __future__ import annotations

from . import common
from . import scoring


def _ratio_pass(pair, limit, *, require_CI=False):
    ratio=pair.get('mother_RMSE_ratio',{})
    point=ratio.get('estimate');ci=ratio.get('CI95')
    return bool(pair.get('complete') and point is not None and point<=limit
                and (not require_CI or ci is not None and ci[1]<=limit))


def _improved(pair, minimum):
    ratio=pair.get('mother_RMSE_ratio',{}).get('estimate')
    ci=pair.get('mother_MSE_difference',{}).get('CI95')
    return bool(pair.get('complete') and ratio is not None and ratio<=1-minimum and ci is not None and ci[1]<0)


def decide_A(rows, decompositions, *, planned_raw, processed_raw, stop_reason, criteria):
    expected=planned_raw*6
    matrix=[r for r in rows if r['continuum_source']!='original_v3']
    baselines=[r for r in rows if r['continuum_source']=='original_v3']
    returned_matrix=[r for r in matrix if r['status']!='NOT_EXECUTED']
    returned_baselines=[r for r in baselines if r['status']!='NOT_EXECUTED']
    complete=processed_raw==planned_raw and len(returned_matrix)==expected and len(returned_baselines)==planned_raw
    identity_all=bool(decompositions) and len(decompositions)==expected and all(d.get('numerical_gate_passed') is True for d in decompositions)
    source_and_solve=bool(matrix) and all(r.get('numerical_execution_passed') is True for r in matrix)
    comparisons=[]
    for sign in ('positive','negative'):
        for source in ('C_true','C_v5','C_v3'):
            comparisons.append(scoring.paired(rows,source+'__estimated_beta','original_v3__joint_original',sign=sign))
        comparisons.append(scoring.paired(rows,'C_true__true_beta','original_v3__joint_original',sign=sign))
        comparisons.append(scoring.paired(rows,'C_v3__estimated_beta','C_v5__estimated_beta',sign=sign))
    temperature=scoring.paired(rows,'C_v3__estimated_beta','C_v5__estimated_beta',group=None,field='temperature')
    temperature_v3=scoring.paired(rows,'C_v3__estimated_beta','original_v3__joint_original',group=None,field='temperature')
    def find(candidate,reference,sign):
        return next(r for r in comparisons if r['candidate']==candidate and r['reference']==reference and r['sign']==sign)
    oracle_checks={}
    for sign in ('positive','negative'):
        pair=find('C_true__true_beta','original_v3__joint_original',sign)
        selection=[r for r in rows if r['cohort']=='development' and r['input_kind']=='noise' and r['group']=='normal'
                   and r['sign']==sign and r['method']=='C_true__true_beta']
        a=scoring.aggregate(selection) if selection else {}
        oracle_checks[sign]={'point_RMSE_vs_v3':_ratio_pass(pair,criteria['O0_normal_noise_vs_v3_RMSE_ratio_max']),
                            'projection':a.get('signed_projection'),
                            'noncollapse':a.get('signed_projection') is not None and a['signed_projection']>=criteria['oracle_noncollapse_projection_min']}
    oracle_sane=all(v['point_RMSE_vs_v3'] and v['noncollapse'] for v in oracle_checks.values())
    sufficient=all(_ratio_pass(find('C_v3__estimated_beta','original_v3__joint_original',s),criteria['v3_ratio_max'],require_CI=True)
                   and _improved(find('C_v3__estimated_beta','C_v5__estimated_beta',s),criteria['Cv5_improvement_RMSE_fraction_min'])
                   for s in ('positive','negative')) and _ratio_pass(temperature_v3,criteria['v3_ratio_max'],require_CI=True)
    temp_better=_improved(temperature,criteria['temperature_improvement_fraction_for_B'])
    ready=complete and identity_all and source_and_solve and oracle_sane and not stop_reason
    if stop_reason or not complete or not identity_all or not source_and_solve or not oracle_sane:
        status='STOP_NUMERICAL_OR_ORACLE_AUDIT';direction='numerical_data_chain_diagnosis';B=False
    elif sufficient:
        status='CONTINUUM_ROUTE_SUPPORTED';direction='continuum_inversion_from_frozen_v3';B=False
    elif temp_better:
        status='B_REQUIRED';direction='observable_joint_physical_reconstruction';B=True
    else:
        status='ALGORITHMIC_GAIN_NOT_SUPPORTED';direction='fixed_budget_imaging_design';B=False
    return {'schema':'v6-stage-A-decision-001','time':common.now(),'status':status,
            'A_complete':complete,'expected_linear_rows':expected,'returned_linear_rows':len(returned_matrix),
            'recorded_linear_rows_including_unexecuted':len(matrix),
            'expected_v3_rows':planned_raw,'returned_v3_rows':len(returned_baselines),'processed_raw':processed_raw,
            'numerical_gate_passed':bool(complete and identity_all),'source_and_linear_success':source_and_solve,
            'oracle_sane':oracle_sane,'oracle_checks':oracle_checks,'stop_reason':stop_reason,
            'continuum_route_sufficient':bool(ready and sufficient),'v3_continuum_temperature_improved':temp_better,
            'B_eligible':bool(ready and B),'design_diagnostic_eligible':bool(ready and status=='ALGORITHMIC_GAIN_NOT_SUPPORTED'),
            'next_research_direction':direction,'template_pairings':comparisons,
            'temperature_Cv3_vs_Cv5':temperature,'temperature_Cv3_vs_original_v3':temperature_v3,
            'all_group_metrics':scoring.grouped(rows),'new_CNN_training_steps':0,
            'sealed_confirmation_and_final_payloads_opened':False,
            'selected_oracle_used_as_deployable_input':False}


def _number(value,unit=''):
    return '未得到完整证据' if value is None else f'{value:.4g}{unit}'


def write_A_report(decision, decompositions, source_receipts):
    path=common.REPORTS/'V6_A_REPORT.md'
    failed=[r for r in decompositions if r.get('numerical_gate_passed') is not True]
    lines=['# v6 阶段 A：固定 raw 的连续谱来源置换','',
       f"状态：**{decision['status']}**。实际处理 {decision['processed_raw']}/720 个冻结 raw；返回线性比较 {decision['returned_linear_rows']}/4320 行、原 v3 {decision['returned_v3_rows']}/720 行。",
       '', '原输入、标定、v3 方法、W、alpha=1、正则与迭代预算均固定。C_true 和 true_beta 属于合成 oracle 诊断；C_v3 与 C_v5 的推理只接收同一 raw 和标定。',
       '', '## 同 raw 的正常含噪模板比较','',
       '|正负|连续谱／背景|模板 RMSE (e)|色温 RMSE (K)|幅度投影|完整 raw 数|',
       '|---|---|---:|---:|---:|---:|']
    for r in decision['all_group_metrics']:
        if r['cohort']=='development' and r['input_kind']=='noise' and r['group']=='normal':
            lines.append(f"|{r['sign']}|{r['method']}|{_number(r.get('template_RMSE_e'))}|{_number(r.get('temperature_RMSE_K'))}|{_number(r.get('signed_projection'))}|{r['cases']}|")
    lines+=['','完整分组统计保留 continuous、quantized、noise；normal 正负、weak 正负和 blank；development 与 oracle 单独列出。见 A_DECISION.json 的 all_group_metrics。任何失败／无效输出均保留，主误差不能通过剔除像素或帧改善。',
       '', '## 分量恒等检查','',
       '`实际误差 = 连续谱泄漏 + 同 raw 噪声 + 正则化偏差 + 数值重组残差`。所有量在可观察 postblur 模板域比较，交叉项必须保留，RMS 不能直接相加。',
       '', f"分解行数：{len(decompositions)}；未通过数值门槛：{len(failed)}；全矩阵数值门槛：{decision['numerical_gate_passed']}。",
       '', '每例分解见 continuum_leakage_decomposition.jsonl；实际／泄漏／噪声／偏差求解均记录物理与平衡真残差。浮点 64 参考只诊断固定目标的数值问题，不能替代失败的正式浮点 32 输出。',
       '', '历史 O0/O1/O2/O3 使用 alpha=0，当前置换使用固定 alpha=1。历史值仅作背景。当前真连续谱无噪误差应由相同正则化的偏差解释；不能移植 alpha=0 的无噪绝对门槛。',
       '', '## 决策','',
       f"唯一下一步方向：`{decision['next_research_direction']}`。阶段 B 执行资格：{decision['B_eligible']}。固定预算成像设计诊断资格：{decision['design_diagnostic_eligible']}。",
       f"停止原因：{decision['stop_reason'] or '无提前数值停止'}。",'',
       '所有结果仍是冻结合成标定中的 postblur 色温和有符号数值模板；未验证真实气体温度、物种浓度、实物硬件吞吐量或封存确认／最终集。']
    common.ed_path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return path


def write_current_decision(decision):
    """This file explicitly remains interim until all conditional work is done."""
    lines=['# DECISION_V6：阶段状态','',f"当前证据状态：{decision['status']}。",'',
      '1. 连续谱投影造成的模板误差：逐例泄漏、噪声、正则化偏差及交叉项见阶段 A 分解；只有全数值门槛通过后才能给出全组归因。',
      '2. raw-only v3 连续谱的正负模板效果：同 raw 母场景配对和完整误差见 A_DECISION.json；阶段 A 未完成时不作总体改善结论。',
      f"3. 可观察频带联合优化：{'阶段 B 尚待执行' if decision['B_eligible'] else '当前阶段 A 决策没有授权进入阶段 B'}；不能声称优于 v3。",
      f"4. 当前唯一方向：`{decision['next_research_direction']}`。",
      '5. 外部标定、非灰体、真实光谱、实际设备、36 signed confirmation 与 64 final 母场景仍未验证。', '',
      '**本文件为阶段状态，条件适用工作与 GitHub 交付完成后才形成最终决策。**']
    p=common.ed_path(common.REPORTS/'DECISION_V6.md')
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return p
