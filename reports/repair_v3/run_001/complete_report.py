"""Offline evidence supplement; no solver, network or training is executed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
DATA = ROOT/'outputs/repair_v3/run_001'
OLD_DATA = ROOT/'outputs/audit_v2/run_001'
SELECTED = 'consistent_init__joint_update_v3'
COMBINATIONS = ['B2_six_stage_no_CNN', 'old_init__joint_update_v3',
                'consistent_init__old_update_v3', SELECTED]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def journal(name):
    path = OUT/name
    if not path.exists():
        return []
    records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    return list({r['identity_sha256']: r for r in records}.values())


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name].copy() for name in f.files}


def gate_evidence():
    rows, old = journal('per_case.jsonl'), journal('reused_audit_v2.jsonl')
    blocks, ablation = [], []
    for case in ('front_continuous_e', 'front_quantization_only', 'front_old_noisy_quantized'):
        original = read(OUT/(case+'_old_first_step.json'))
        for step in original['subblocks']:
            blocks.append({'case_id': case, 'method': COMBINATIONS[0],
                           'input_hash': original['input_hash'], **step})
        for r in rows:
            if r['case_id'] != case:
                continue
            for step in r.get('trace', [])[:4]:
                blocks.append({'case_id': case, 'method': r['method'],
                               'input_hash': r['input_hash'], **step})
        selected = [next(r for r in old+rows if r['case_id'] == case and r['method'] == m)
                    for m in COMBINATIONS]
        assert len({r['input_hash'] for r in selected}) == 1
        base = selected[0]['metrics']
        ablation.append({'case_id': case, 'input_hash': selected[0]['input_hash'],
            'rows': [{'method': r['method'], 'T_RMSE_K': r['metrics']['temperature_observed_RMSE_K'],
                      'S_RMSE_e': r['metrics']['template_RMSE_e'], 'seconds': r.get('seconds'),
                      'T_RMSE_change_from_old_K': r['metrics']['temperature_observed_RMSE_K']-base['temperature_observed_RMSE_K'],
                      'S_RMSE_change_from_old_e': r['metrics']['template_RMSE_e']-base['template_RMSE_e'],
                      'reused': r.get('reused_from_audit_v2', False)} for r in selected],
            'scope': 'one frozen input per variant; no population-level superiority or necessity claim'})
    runtime.write_json(OUT/'first_step_diagnostics.json', blocks)
    runtime.write_json(OUT/'initializer_update_comparison.json', ablation)
    render_gate_maps(rows, old)
    return blocks, ablation


def render_gate_maps(rows, old):
    destination = OUT/'figures'
    destination.mkdir(exist_ok=True)
    manifest = read(OUT/'frozen_manifest.json')['cases']
    for case in ('front_continuous_e', 'front_quantization_only'):
        fixture = next(f for f in manifest if f['case_id'] == case)
        truth = arrays(DATA/fixture['path'])
        records = [next(r for r in old+rows if r['case_id'] == case and r['method'] == m)
                   for m in COMBINATIONS+['B4_published']]
        predictions = [arrays((OLD_DATA if r.get('reused_from_audit_v2') else DATA)/r['prediction_path'])
                       for r in records]
        smax = max(float(np.abs(p['template_e']).max()) for p in predictions)
        tmax = max(float(np.abs(p['temperature_K']-truth['temperature_observed_color_K']).max()) for p in predictions)
        fig, axes = plt.subplots(len(records), 2, figsize=(9, 2.5*len(records)))
        for i, (r, p) in enumerate(zip(records, predictions)):
            for j, (field, limit, title) in enumerate([
                    (p['temperature_K']-truth['temperature_observed_color_K'], tmax, 'color error / K'),
                    (p['template_e'], smax, 'signed template / e')]):
                image = axes[i, j].imshow(field, cmap='RdBu_r', vmin=-limit, vmax=limit)
                axes[i, j].set_title(r['method']+'\n'+title, fontsize=8)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                fig.colorbar(image, ax=axes[i, j], fraction=.04)
        fig.suptitle(case+'; shared color scales include every displayed pixel', fontsize=11)
        fig.tight_layout()
        fig.savefig(destination/(case+'_shared_full_range.png'), dpi=135)
        plt.close(fig)
    cases = ['same_front_signal_continuous', 'same_front_weak_quantization_only',
             'same_front_negative_quantization_only', 'rotated70_null_quantization_only',
             'rotated25_signal_quantization_only', 'flat_signal_quantization_only']
    truth_maps = {case: arrays(DATA/next(f for f in manifest if f['case_id'] == case)['path']) for case in cases}
    temperature_min = min(float(t['temperature_observed_color_K'].min()) for t in truth_maps.values())
    temperature_max = max(float(t['temperature_observed_color_K'].max()) for t in truth_maps.values())
    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 2.55*len(cases)))
    for i, case in enumerate(cases):
        f = next(f for f in manifest if f['case_id'] == case)
        r = next(r for r in rows if r['case_id'] == case and r['method'] == SELECTED)
        truth, pred = truth_maps[case], arrays(DATA/r['prediction_path'])
        limit = max(float(np.abs(truth['template_e']).max()), float(np.abs(pred['template_e']).max()), 1.)
        fields = [(truth['temperature_observed_color_K'], 'observed truth / K', 'inferno', None),
                  (truth['template_e'], 'signed truth / e', 'RdBu_r', limit),
                  (pred['template_e'], 'signed reconstruction / e', 'RdBu_r', limit),
                  (pred['template_e']-truth['template_e'], 'template error / e', 'RdBu_r',
                   max(float(np.abs(pred['template_e']-truth['template_e']).max()), .01))]
        for j, (field, title, cmap, bound) in enumerate(fields):
            image = axes[i, j].imshow(field, cmap=cmap,
                vmin=temperature_min if j == 0 else (-bound if bound else None),
                vmax=temperature_max if j == 0 else bound)
            axes[i, j].set_title((case+'\n' if j == 0 else '')+title, fontsize=8)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            fig.colorbar(image, ax=axes[i, j], fraction=.04)
    fig.tight_layout()
    fig.savefig(destination/'nonzero_negative_rotated_flat_regressions.png', dpi=135)
    plt.close(fig)


def supplement_report(blocks, ablation, comparisons, warnings):
    path = OUT/'REPAIR_V3_REPORT.md'
    text = path.read_text(encoding='utf-8')
    marker = '\n<!-- OFFLINE_V3_SUPPLEMENT -->\n'
    text = text.split(marker)[0]
    begin, end = '<!-- V3_OUTCOME_BEGIN -->', '<!-- V3_OUTCOME_END -->'
    if begin in text:
        head, tail = text.split(begin, 1)
        text = head+tail.split(end, 1)[1].lstrip('\n')
    lead = [begin,
        '本轮在本机 RTX 4060 完成，未训练。16/16 物理开发门禁通过：原空白连续前沿的色温/模板 RMSE 为 **0.302 K / 0.465 e**，仅量化为 **1.106 K / 1.896 e**，两者历史 >60 e 误报均为零。正模板、弱模板、负模板、不同前沿方向和平坦场一起回归。', '',
        '四组合显示**初始化和更新包各自能修复前沿问题，联合更新包对最终模板误差的改善更大**。两者组合通过全部门禁；单帧消融没有证明 warm start 对最终结果必不可少，量化前沿上仅改更新还略好且更快。带噪声 smoke 的结论单独报告，不能据无噪声门禁声称取代 CNN。', end, '']
    title, body = text.split('\n', 1)
    text = title+'\n\n'+'\n'.join(lead)+'\n'+body.lstrip('\n')
    text = text.replace('正式新增测试通过 14 项。', '正式物理测试通过 14 项，旧结果复用隔离测试另通过 1 项，共 15 项。')
    new_rows, old_rows = journal('per_case.jsonl'), journal('reused_audit_v2.jsonl')
    def category(r):
        if r.get('known_calibration_oracle'):
            return 'known_calibration_oracle'
        return r['scene'] if r['scene'] in ('channel_gain_3pct', 'non_gray_continuum') else 'nominal'
    overview = []
    for cat in ('nominal', 'known_calibration_oracle', 'channel_gain_3pct', 'non_gray_continuum'):
        for method, pool in ((SELECTED, new_rows), ('B4_published', old_rows)):
            rs = [r for r in pool if r['suite'] == 'smoke' and r['method'] == method and category(r) == cat and r.get('metrics')]
            if not rs:
                continue
            n = sum(r['metrics']['domain_pixels'] for r in rs)
            nulls = [r for r in rs if r['metrics']['template_truth_energy'] == 0]
            nn = sum(r['metrics']['domain_pixels'] for r in nulls)
            overview.append({'category': cat, 'method': method, 'frames': len(rs),
                'pooled_T_observed_RMSE_K': float(np.sqrt(sum(r['metrics']['temperature_SSE_K2'] for r in rs)/n)),
                'pooled_S_RMSE_e': float(np.sqrt(sum(r['metrics']['template_SSE'] for r in rs)/n)),
                'historical60_null_pixel_FP': sum(r['metrics']['historical60_null_FP_fixed_domain']*r['metrics']['domain_pixels'] for r in nulls)/nn if nn else None,
                'validation_selected_null_frame_FP': float(np.mean([r['metrics']['blank_frame_false_positive'] for r in nulls])) if nulls else None,
                'median_inference_seconds': float(np.median([r['seconds'] for r in rs]))})
    runtime.write_json(OUT/'overall_strata_summary.json', overview)
    lines = [marker, '## 四类 smoke 总览', '',
        '下表在各类别的全部强度/空白输入上合并像素，逐强度结果保留在上面的分层表和 stratified_summary.json。已知标定类别没有原注册队列中的匹配 B4 结果，因此不补造旧已知标定基线。', '',
        '|类别|方法|帧数|pooled 色温 RMSE K|pooled 模板 RMSE e|历史 >60e 空白像素误报率|新/旧各自验证阈值的空白帧误报率|',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in overview:
        lines.append(f'|{s["category"]}|{s["method"]}|{s["frames"]}|{s["pooled_T_observed_RMSE_K"]:.3f}|{s["pooled_S_RMSE_e"]:.3f}|{s["historical60_null_pixel_FP"]:.4f}|{s["validation_selected_null_frame_FP"]:.4f}|')
    lines += ['', '新方法的检测阈值仅从 24 个独立 validation-null 得到：pixel=94.7813 e、frame score=140.8350 e。固定阈值 >60 e 也完整保留，避免高阈值掩盖背景误差。验证集很小，不声称统计保证的 1% 帧误报控制。', '',
        '## 两项修复分别解决什么', '',
        '旧解调的 700/905 色温已经处于 postblur 波段域；B1 的约 0.009 K 不等于把该值放进 latent T 再前向模糊也有 0.009 K。旧 latent 初值前向得到的色温误差为 39.010 K。空间一致 warm start 将其降到 0.00912 K，保持模板初值 2.2605 e 不变；它校正了初值的空间域，而非调用真实温度或真实模板。', '',
        '|同一输入|只改初始化：色温/模板 RMSE|只改更新：色温/模板 RMSE|两者组合：色温/模板 RMSE|',
        '|---|---:|---:|---:|']
    for item in ablation:
        by_method = {r['method']: r for r in item['rows']}
        values = [by_method[m] for m in (COMBINATIONS[2], COMBINATIONS[1], COMBINATIONS[3])]
        lines.append('|'+item['case_id']+'|'+'|'.join(f'{v["T_RMSE_K"]:.4f} K / {v["S_RMSE_e"]:.4f} e' for v in values)+'|')
    lines += ['', '连续前沿：旧纯物理模板 63.2828 e，初始化单改为 3.0626 e，更新单改为 0.4990 e，两者为 0.4647 e。仅量化：更新单改为 1.0365 K / 1.8697 e，组合为 1.1062 K / 1.8959 e。组合在连续输入上的小幅模板优势不构成统计优越性证据。', '',
        '这里“更新包”同时改变了耦合求解、预条件、CG 预算和 LM 接受规则；本轮没有进一步拆解这些因素，不能将改善全部归因于“联合”二字。旧 CG 是光谱 4 次加温度 1 次，新联合 CG 每个 LM 尝试最多 120 次；两者最多 12 次外层更新，计算量并不相等。三个新组合在连续前沿上的实测推理耗时分别约 8.07、15.07、20.81 s；包含 warm start 和 trace，未包含加载模型。', '',
        '原带噪前沿：保留的旧 CNN 模板 RMSE 为 34.6643 e，组合纯物理为 43.9538 e；组合色温为 15.5646 K，旧 CNN 为 34.4274 K。温度与模板存在不同方向的变化，这也是继续保留 CNN、把迁移/重训留到下一提交的证据。', '',
        '## 第一次 data_step 的证据', '',
        '下表统一用观测导出的固定权重评分 raw 目标。旧算法实际接受用的是旧预测导出权重；评分目标与旧接受目标不混称。raw 目标下降同时伴随模板恶化，说明只检查 raw 残差不足以辨别状态串扰。', '',
        '|初始化/更新|阶段|色温 RMSE K|模板 RMSE e|固定权重 raw 目标|',
        '|---|---|---:|---:|---:|']
    for b in blocks:
        if b['case_id'] != 'front_continuous_e' or b['label'] == 'old_initializer_seed':
            continue
        lines.append(f'|{b["method"]}|{b["label"]}|{b["metrics"]["temperature_observed_RMSE_K"]:.5f}|{b["metrics"]["template_RMSE_e"]:.5f}|{b["raw_objective"]:.6g}|')
    lines += ['', '旧光谱子块的 latent 活跃模板增量 RMS 为 266.45 e，输出域模板 RMSE 跳到 185.57 e；温度子块保持模板不变。修复后的联合步没有独立的光谱/温度接受顺序，分量预览只是从已接受的同一增量中暂时隐藏温度分量，标记 `feeds_solver=false`，不反馈求解。完整 T/loga/S/beta 增量、边界比例和 CG/LM 明细见 [first_step_diagnostics.json](first_step_diagnostics.json) 与 [per_case.jsonl](per_case.jsonl)。', '',
        '## 目标、约束与数值边界', '',
        '`F(z,beta)=encode(H[a C(T)+M S+B(beta)])`，H 仍为原 object_blur。归一化 latent 温度为 `(T−3000)/1000`，模板为 `S/60`；沿用原 T、loga、signed S 和 beta 约束。活跃模板仍由原配置指定（当前一项），没有新增 S=0 特判或模板正值掩码。', '',
        '`Jraw = 0.5 sum_i (F_i−y_i)^2 / (max(y_i,0)+sigma_extra^2)`；`R = 0.5 sum_c lambda_c (||Dx z_c||^2+||Dy z_c||^2)`。T/loga/S 正则系数分别为 1e−5、1e−5、1e−3；inactive 模板系数为零。正则在目标与 Hessian 中，LM 的 `mu D` 只进入求步系统。预期下降基于约束投影后增量、raw+R 的 Gauss–Newton 模型，不把 LM 惩罚当数据收益。', '',
        'warm start 单独拟合 `H[a C700(T)]` 和 `H[a C905(T)]`，最多 30 次，不访问 OTF 逆。最终 12 次联合更新只拟合 raw，解调波段不构成另一份独立观测。warm start 与 raw 目标不能跨阶段拼接成一条似然下降曲线。', '',
        f'{len(warnings)} 个新结果含至少一个未达到 1e−4 相对真实残差的 CG 求解，明细保存在 [numerical_limitations.json](numerical_limitations.json)。线性迭代到上限与非线性步被接受是两个不同事件，全部已接受步的实际/预期下降另经离线核查；没有宣称全局最优或 KKT 收敛。', '',
        '## smoke 的配对结果与负面记录', '',
        f'同原 48 帧、同输入哈希逐帧与保留的 B4 比较，纯物理 v3 有 {sum(r["template_worse_than_preserved_CNN"] for r in comparisons)}/48 帧模板 RMSE 更大，{sum(r["temperature_worse_than_preserved_CNN"] for r in comparisons)}/48 帧色温 RMSE 更大。明细见 [paired_smoke_CNN_comparison.json](paired_smoke_CNN_comparison.json)，包括改善和退化的全部帧。数值执行成功不等于研究指标达标；负面结果见 [negative_findings.json](negative_findings.json)，执行异常/门禁失败见 [failure_records.json](failure_records.json)。', '',
        '四类汇总只覆盖原冻结工程仿真 smoke：名义、已知真实标定 oracle、未知通道增益失配、非灰体。已知 oracle 直接给定 true_config，不使用真实温度/模板，也不代表自动标定能力。既有响应矩阵、QE 和滤波器参数仍是未实测的工程假设；本轮没有新增真实仪器数据或修改它们。', '',
        '## 复用、封存与交付核查', '',
        '原 audit_v2 的 journal 含两条额外的已知标定 resume-probe，它们不在原注册的八方法队列中。进入 smoke 前仅修正旧结果复用白名单；保留它们和旧 journal 原字节，不作为 v3 失败或重复测量。原 registration 没有改写；修正前 run.py 原件、兼容封条及逐函数 AST 比较证明求解/评分/执行函数完全未改。见 [orchestration_compatibility_v3_001.json](orchestration_compatibility_v3_001.json)。', '',
        '所有 295 个原已跟踪文件（含网络、physics.forward、solver、旧权重和原报告）逐字节 SHA256 不变。旧完整 audit_v2 推理重跑 0 次；只为补齐子块日志重算 3 个旧前沿的第一次 data_step。新计算为 23 个门禁/消融组合、48 帧 smoke、8 个同输入已知标定 oracle、24 个 validation-null。旧结果复用 390 条，其中 smoke 为 48×8=384 条。', '',
        '数值输入、预测数组及缓存位于 E 盘被忽略的 outputs/_runtime，GitHub 交付代码、配置、哈希、逐例 JSONL、图、测试 XML、失败记录和报告。已有 registration 的复制复现需要同时复制并校验冻结 NPZ；公开仓库不包含这些大数组，缺少时必须明确失败，不能重新抽样后称作同一冻结输入。精确路径/哈希见 [input_hash_inventory.json](input_hash_inventory.json)。', '',
        '本次 GPU 解释器为 `E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe`，Torch 2.12.0+cu130，TF32 关闭；报告解释器为本机 Python 3.13（读取程序可在 C，全部写入 E）。运行命令与主机/GPU预算见两份 `local_*_launcher_receipt.json`。', '',
        '门禁阶段尚未完成时用 `python -B repair_next.py --phase gates --resume`；本次门禁已封存，不必再调用 gates。已有且已核验的 run_001 可直接 `--phase smoke --resume`；随后报告环境执行 `--phase summarize --device cpu` 和 `python -B reports/repair_v3/run_001/complete_report.py --mode final`。首次创建其他登记应从 5f2ae63 checkout 配合本轮修复源码、测试 XML 开始，不能在任意新 HEAD 冒充原基线重新 prepare。', '',
        '下一提交再做修复后纯物理、旧 CNN 迁移、同规模 CNN 重训三者对照；训练场景需覆盖多方向/尺度前沿、平坦、多峰与非平稳背景，先小规模验证，再扩展数据与种子。本轮没有启动该训练。', '',
        '## 图像与记录', '',
        '![连续前沿全范围共同色标](figures/front_continuous_e_shared_full_range.png)', '',
        '![仅量化前沿全范围共同色标](figures/front_quantization_only_shared_full_range.png)', '',
        '![正负模板、旋转前沿和平坦场](figures/nonzero_negative_rotated_flat_regressions.png)', '',
        '更多收据：[gate_receipt.json](gate_receipt.json)、[smoke_receipt.json](smoke_receipt.json)、[stratified_summary.json](stratified_summary.json)、[delivery_QA.json](delivery_QA.json)。', '']
    path.write_text(text.rstrip()+'\n'+'\n'.join(lines), encoding='utf-8')


def complete_final(blocks, ablation):
    smoke, reg = read(OUT/'smoke_receipt.json'), read(OUT/'registration.json')
    expected = {'new_smoke_nominal_and_unknown_configurations': 48,
                'new_known_calibration_oracles': 8, 'validation_null_frames': 24,
                'old_smoke_method_inputs_reused': 384}
    assert all(smoke[k] == v for k, v in expected.items())
    rows, validation, reused = journal('per_case.jsonl'), journal('validation_per_case.jsonl'), journal('reused_audit_v2.jsonl')
    assert len(rows) == 79 and len(validation) == 24 and len(reused) == 390
    warnings, accepted_checks = [], []
    for r in rows+validation:
        solves = [(('warm', s['iteration']), s) for s in r.get('warm_start', [])]
        solves += [(('raw_joint', s['iteration']), s) for s in r.get('curve', []) if 'CG' in s]
        nonconverged = []
        for (phase, iteration), s in solves:
            if not s['CG']['converged']:
                nonconverged.append({'phase': phase, 'iteration': iteration,
                    'CG_stop': s['CG']['stop'], 'iterations': s['CG']['iterations'],
                    'relative_true_residual': s['CG']['relative_residual'],
                    'accepted_nonlinear_step': s['accepted'], 'damping': s['damping_used']})
            if s['accepted']:
                assert s['objective_after'] < s['objective_before']
                assert s['actual_reduction'] > 0 and s['predicted_reduction'] > 0
                assert s['actual_predicted_ratio'] >= .05
                accepted_checks.append({'identity_sha256': r['identity_sha256'], 'phase': phase,
                                        'iteration': iteration, 'objective_drop': s['actual_reduction']})
        if nonconverged:
            warnings.append({'case_id': r['case_id'], 'method': r['method'],
                'known_calibration_oracle': r.get('known_calibration_oracle', False),
                'nonconverged_CG_solves': nonconverged,
                'meaning': 'Bounded inexact linear solve; nonlinear acceptance is checked separately. No stationarity certificate.'})
    runtime.write_json(OUT/'numerical_limitations.json', warnings)
    runtime.write_json(OUT/'accepted_objective_checks.json', accepted_checks)
    comparisons = []
    for r in rows:
        if r['suite'] != 'smoke' or r['known_calibration_oracle'] or not r.get('metrics'):
            continue
        b = next(s for s in reused if s['case_id'] == r['case_id'] and s['method'] == 'B4_published')
        assert r['input_hash'] == b['input_hash'] and r['fixture_sha256'] == b['fixture_sha256']
        comparisons.append({'case_id': r['case_id'], 'scene': r['scene'], 'condition': r['condition'],
            'new_T_RMSE_K': r['metrics']['temperature_observed_RMSE_K'],
            'old_CNN_T_RMSE_K': b['metrics']['temperature_observed_RMSE_K'],
            'new_S_RMSE_e': r['metrics']['template_RMSE_e'],
            'old_CNN_S_RMSE_e': b['metrics']['template_RMSE_e'],
            'template_worse_than_preserved_CNN': r['metrics']['template_RMSE_e'] > b['metrics']['template_RMSE_e'],
            'temperature_worse_than_preserved_CNN': r['metrics']['temperature_observed_RMSE_K'] > b['metrics']['temperature_observed_RMSE_K'],
            'scope': 'physical repair versus old published CNN; no retraining or CNN migration'})
    runtime.write_json(OUT/'paired_smoke_CNN_comparison.json', comparisons)
    negative = [dict(kind='PERFORMANCE_REGRESSION_VS_PRESERVED_CNN', **r) for r in comparisons
                if r['template_worse_than_preserved_CNN'] or r['temperature_worse_than_preserved_CNN']]
    negative += [dict(kind='NUMERICAL_NONCONVERGENCE_RETAINED', **w) for w in warnings]
    runtime.write_json(OUT/'negative_findings.json', negative)
    input_inventory = []
    manifests = [(DATA, read(OUT/'frozen_manifest.json')['cases']),
                 (OLD_DATA, [f for f in read(ROOT/'reports/audit_v2/run_001/frozen_manifest.json')['cases']
                             if f['suite'] == 'smoke' or (f['suite'] == 'validation' and f['metadata']['condition'].startswith('null'))])]
    for root, fixtures in manifests:
        for fixture in fixtures:
            assert runtime.sha256(root/fixture['path']) == fixture['file_sha256']
            records = [r for r in rows+validation if r['case_id'] == fixture['case_id']]
            input_inventory.append({'case_id': fixture['case_id'], 'suite': fixture['suite'],
                'fixture_path': str(root/fixture['path']), 'fixture_sha256': fixture['file_sha256'],
                'measurement_hashes': sorted({(r['input_kind'], r['input_hash']) for r in records}),
                'new_result_identities': [r['identity_sha256'] for r in records]})
    runtime.write_json(OUT/'input_hash_inventory.json', input_inventory)
    for r in reused:
        assert runtime.sha256(OLD_DATA/r['prediction_path']) == r['prediction_sha256']
    assert all(runtime.sha256(ROOT/p) == h for p, h in reg['original_tracked_sha256'].items())
    tests = [ET.parse(OUT/name).getroot() for name in ('tests.xml', 'cohort_tests.xml')]
    suites = [s for tree in tests for s in (tree.findall('testsuite') or [tree])]
    test_count = sum(int(s.attrib['tests']) for s in suites)
    assert test_count == 15 and all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    from PIL import Image
    image_records = []
    for path in sorted((OUT/'figures').glob('*.png')):
        with Image.open(path) as im:
            im.load()
            image_records.append({'path': str(path.relative_to(OUT)), 'size': list(im.size), 'sha256': runtime.sha256(path)})
    supplement_report(blocks, ablation, comparisons, warnings)
    receipt = {'baseline_commit': '5f2ae6307a644dbfbbab426b9c239397981d24b7',
        'tests_passed': test_count, 'physical_gates_passed': 16,
        'new_gate_and_ablation_inferences': 23, 'new_smoke_inferences': 48,
        'new_known_calibration_smoke_oracles': 8, 'new_validation_null_inferences': 24,
        'old_result_rows_reused': 390, 'old_smoke_rows_reused': 384,
        'original_tracked_files_preserved': 295, 'old_audit_journal_unchanged': True,
        'first_old_steps_recomputed_for_missing_subblock_diagnostics_only': 3,
        'old_complete_audit_inferences_rerun': 0, 'optimizer_updates': 0,
        'CNN_weight_updates': 0, 'network_modules_added': 0,
        'accepted_steps_with_actual_predicted_objective_checks': len(accepted_checks),
        'nonconverged_CG_case_records': len(warnings),
        'paired_smoke_template_worse_than_old_CNN': sum(r['template_worse_than_preserved_CNN'] for r in comparisons),
        'figures_reopened': image_records,
        'visual_QA_representatives': ['front_continuous_e_shared_full_range.png',
            'front_quantization_only_shared_full_range.png', 'nonzero_negative_rotated_flat_regressions.png'],
        'visual_QA_note': 'Representative pixels inspected separately; all PNGs fully decoded. Full-range comparison scales plus signed nonzero regression maps retained.',
        'all_data_on_E_or_D': True, 'training_deferred_to_next_submission': True,
        'report_sha256': runtime.sha256(OUT/'REPAIR_V3_REPORT.md')}
    runtime.write_json(OUT/'delivery_QA.json', receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gates', 'final'], default='gates')
    args = parser.parse_args()
    blocks, ablation = gate_evidence()
    if args.mode == 'final':
        complete_final(blocks, ablation)
    print({'offline_evidence_completed': args.mode, 'first_step_records': len(blocks)})


if __name__ == '__main__':
    main()
