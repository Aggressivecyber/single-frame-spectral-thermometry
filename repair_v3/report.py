"""Rebuild repair diagnostics from saved results, never repeat inference."""
from pathlib import Path
import numpy as np
import runtime
from evaluation_v2.common import Journal
from evaluation_v2.metrics import aggregate_metrics
from repair_v3.infer import SELECTED
from repair_v3.protocol import PROTOCOL
from repair_v3.run import read, guard, OLD, OLD_DATA


def fmt(x, digits=4):
    return 'N/A' if x is None else f'{x:.{digits}f}'


def summarize(args):
    reg = guard(args)
    rows = list({r['identity_sha256']: r for r in Journal(args.output/'per_case.jsonl').rows()}.values())
    reused = Journal(args.output/'reused_audit_v2.jsonl').rows()
    gate = read(args.output/'gate_receipt.json')
    groups = []
    for row in rows:
        if row['suite'] != 'smoke':
            continue
        category = 'known_calibration_oracle' if row['known_calibration_oracle'] else (
            'unknown_channel_gain_3pct' if row['scene'] == 'channel_gain_3pct' else (
                'non_gray_continuum' if row['scene'] == 'non_gray_continuum' else 'nominal'))
        for group in [groups]:
            if not any(g['category'] == category and g['condition'] == row['condition'] for g in group):
                group.append({'category': category, 'condition': row['condition']})
    summaries = []
    for group in groups:
        category, condition = group['category'], group['condition']
        def belongs(row):
            kind = 'known_calibration_oracle' if row.get('known_calibration_oracle') else (
                'unknown_channel_gain_3pct' if row['scene'] == 'channel_gain_3pct' else (
                    'non_gray_continuum' if row['scene'] == 'non_gray_continuum' else 'nominal'))
            return row['suite'] == 'smoke' and kind == category and row['condition'] == condition
        for method in (SELECTED, 'B4_published', 'B2_six_stage_no_CNN', 'B1_tikhonov'):
            selected = [r for r in (rows if method == SELECTED else reused) if r['method'] == method and belongs(r)]
            if selected:
                summaries.append({**group, 'method': method, **aggregate_metrics(selected),
                                  'temperature_reference_RMSE_K_mean': float(np.mean([r['metrics']['temperature_reference_RMSE_K'] for r in selected if r.get('metrics')])),
                                  'raw_RMSE_e_mean': float(np.mean([r['raw_RMSE_e'] for r in selected])) if all('raw_RMSE_e' in r for r in selected) else None})
    runtime.write_json(args.output/'stratified_summary.json', summaries)
    front_cases = ['front_continuous_e', 'front_quantization_only', 'front_old_noisy_quantized']
    ablation = []
    for case in front_cases:
        old = [r for r in reused if r['case_id'] == case]
        new = [r for r in rows if r['case_id'] == case]
        for row in old+new:
            ablation.append({'case_id': case, 'method': row['method'], 'metrics': row.get('metrics'),
                             'input_hash': row['input_hash'], 'prediction_sha256': row.get('prediction_sha256'),
                             'reused_from_audit_v2': row.get('reused_from_audit_v2', False)})
        assert len({r['input_hash'] for r in old+new}) == 1
    runtime.write_json(args.output/'four_combination_ablation.json', ablation)
    failed = [r for r in rows if r['status'] != 'SUCCESS']
    failed += [{'kind': 'DEVELOPMENT_GATE_FAILED', **d} for d in gate['decisions'] if not d['passed']]
    runtime.write_json(args.output/'failure_records.json', failed)
    checked_predictions = 0
    for row in rows+Journal(args.output/'validation_per_case.jsonl').rows():
        if row['status'] != 'SUCCESS':
            continue
        assert runtime.sha256(args.data_root/row['prediction_path']) == row['prediction_sha256']
        with np.load(args.data_root/row['prediction_path'], allow_pickle=False) as f:
            assert np.isfinite(f['template_e']).all() and np.isfinite(f['temperature_K']).all()
        if row.get('trace_path'):
            assert runtime.sha256(args.data_root/row['trace_path']) == row['trace_sha256']
        checked_predictions += 1
    figures = render_figures(args, rows, reused)
    curves = [h for r in rows for h in r.get('curve', []) if 'CG' in h]
    cgs = [h['CG'] for h in curves]
    receipt = {'schema': PROTOCOL['schema'], 'physical_gates_passed': gate['all_physical_development_gates_passed'],
               'gate_passed': gate['passed'], 'gate_required': gate['required'],
               'new_evaluation_records': len(rows), 'new_successes': sum(r['status'] == 'SUCCESS' for r in rows),
               'failures': len(failed), 'saved_predictions_reopened': checked_predictions,
               'old_result_records_reused': len(reused), 'original_files_unchanged': len(reg['original_tracked_sha256']),
               'old_full_audit_inferences_rerun': 0, 'optimizer_updates': 0, 'CNN_updates': 0,
               'new_modules': 0, 'figures': figures, 'CG_outer_solves': len(cgs),
               'CG_converged_fraction': sum(h['converged'] for h in cgs)/len(cgs) if cgs else None,
               'scientific_acceptance': 'not_claimed', 'saturated_frames_not_silently_removed': True}
    runtime.write_json(args.output/'final_receipt.json', receipt)
    lines = ['# 单帧七通道光谱测温：物理求解修复 v3', '',
             '基于 `5f2ae6307a644dbfbbab426b9c239397981d24b7`。保留原 physics.forward/object_blur、旧 solver、网络和权重；新增 repair_v3，不训练。', '',
             f'正式新增测试通过 {reg["tests_passed"]} 项。开发门禁 {gate["passed"]}/{gate["required"]} 通过，全部通过={gate["all_physical_development_gates_passed"]}。详细门禁及输入哈希见 `gate_receipt.json`。', '',
             '## 初始化与联合更新', '',
             '初始化只从 raw 解调获取 700/905 波段及旧初值，约束优化 latent T/loga，使 H[a C700(T)]、H[a C905(T)] 接近解调波段。前向的 object_blur 没有删除，没有直接除以 OTF，也没有访问真值。暖启动不作为第二份测量进入最终目标。', '',
             '联合步同时使用原 JVP/VJP 更新归一化 T、loga、活跃有符号模板和 beta。固定的 raw 观测导出方差权重使每步比较同一个目标；它是工程加权最小二乘，不声称严格 Poisson 似然。空间邻域二次正则进入总目标和梯度/Hessian；LM 对角阻尼只进入求步系统。接受要求实际下降为正、投影后预期下降为正且比值 >=0.05，并执行步长搜索、拒绝增大阻尼。PCG 使用相对真实残差停止，上限与未收敛标记保留。', '',
             '旧步的首次光谱/温度子块分别重算并记录增量、CG 残差、接受步长与触边比例；旧完整终点直接按哈希复用。联合步是原子的，首次“光谱分量预览”只用于评分展示，随后加入温度分量为真正接受的完整联合步；预览从不反馈求解或影响接受。truth 只进入评分。', '',
             '## 同一冻结前沿的四组合消融', '',
             '四组合的共同比较基准是纯物理 12 次更新。旧 B2 六阶段共 12 个旧 data-step，直接复用；B4 另作保留 CNN 的旧发布参照，不能把 B2 当 B4。三个新组合也最多 12 次更新，空间一致初始化另有最多 30 次 warm start，耗时应一起计入。', '',
             '|输入|方法|色温 RMSE K|模板 RMSE e|历史 >60e 空白误报|旧结果复用|',
             '|---|---|---:|---:|---:|---|']
    for row in ablation:
        m = row['metrics'] or {}
        lines.append(f'|{row["case_id"]}|{row["method"]}|{fmt(m.get("temperature_observed_RMSE_K"))}|{fmt(m.get("template_RMSE_e"))}|{fmt(m.get("historical60_null_FP_fixed_domain"))}|{row["reused_from_audit_v2"]}|')
    lines += ['', '## 回归门禁与范围', '',
              '16 个门禁帧：原冻结前沿空白的连续/仅量化两帧；同前沿 +210/+45/−120 e、70°空白前沿、25°非零前沿、3000 K 平坦空白/非零模板各两种输入。非零模板同时要求 L2<=0.25，零估计 L2=1，不能靠全零通过。负模板必须恢复符号。原有带噪声前沿另作诊断，不要求达到无噪声门禁。所有数字是开发目标；这些场景已被用于开发，并非最终锁定测试。', '',
              '## 原 48 帧 smoke 与标定分层', '']
    if (args.output/'smoke_receipt.json').exists():
        smoke = read(args.output/'smoke_receipt.json')
        lines += [f'新方法在原 48 帧上执行 {smoke["new_smoke_nominal_and_unknown_configurations"]} 次，另 8 个同输入已知真实标定 oracle；24 个独立 validation-null 仅用于新方法检测阈值。旧 smoke 的 {smoke["old_smoke_method_inputs_reused"]} 个结果按实际输入、冻结容器及预测文件哈希复用，不重跑原 954 项。', '',
                  '新方法阈值仅在旧独立验证母场景拟合，历史 >60 e 仍保留。已知标定行主动给定 A_true，不代表从单帧识别增益。名义组排除增益和非灰体，避免互相掩盖；reference/observed 色温误差分别保存。', '',
                  '|类别|强度|方法|pooled 模板 L2|observed 色温 RMSE K|reference 色温 RMSE 均值 K|', '|---|---|---|---:|---:|---:|']
        for s in summaries:
            lines.append(f'|{s["category"]}|{s["condition"]}|{s["method"]}|{fmt(s["pooled_template_L2"])}|{fmt(s["pooled_temperature_RMSE_K"])}|{fmt(s["temperature_reference_RMSE_K_mean"])}|')
    else:
        lines += ['未执行 smoke：门禁尚未通过或 smoke 未完成。不会借用 audit_v2 完成状态声称新方法完成。']
    lines += ['', '## 收敛证据与限制', '',
              f'外层联合步的 PCG 相对残差门限达到比例为 {fmt(receipt["CG_converged_fraction"])}；固定迭代上限或不再接受步均不证明全局最优/KKT。CG 历史、每次尝试的阻尼、alpha、实际/预期下降，以及 T/loga/S/beta 增量和触边比例保存在 per_case.jsonl。', '',
              f'{receipt["original_files_unchanged"]} 个原已跟踪文件的 SHA256 全部未变，checkpoint SHA256 `{reg["fingerprints"]["checkpoint_sha256"]}`。失败记录和开发单帧试跑均保留。本轮没有 CNN 迁移或训练；门禁通过也不代表消除了低光噪声、未知标定或非灰体不可辨性。', '',
              '复现：先在 E/D checkout 中设置 PYTHONDONTWRITEBYTECODE=1，使用本机 GPU PyTorch 解释器执行 `python -B repair_next.py --phase gates --resume`；只有 gate_receipt 全部通过，才执行 `--phase smoke --resume`。使用带 matplotlib 的报告解释器执行 `python -B repair_next.py --phase summarize --device cpu`。初次 prepare 先运行 `python -B -m pytest tests/test_repair_v3.py --junitxml=reports/repair_v3/run_001/tests.xml`，再执行 `--phase prepare`；已有 registration 时不可再次 prepare。冻结数组在被忽略的 outputs/repair_v3/run_001，旧冻结目录也必须存在；跨机器需要复制并核验，不能重新抽样冒充同输入。', '']
    (args.output/'REPAIR_V3_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(receipt, flush=True)


def render_figures(args, rows, reused):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = args.output/'figures'
    output.mkdir(exist_ok=True)
    figures = []
    for case in ('front_continuous_e', 'front_quantization_only'):
        selected = [r for r in rows if r['case_id'] == case and r['status'] == 'SUCCESS']
        initial = next(f for f in read(args.output/'frozen_manifest.json')['cases'] if f['case_id'] == case)
        with np.load(args.data_root/initial['path'], allow_pickle=False) as f:
            truth = {k: f[k].copy() for k in f.files}
        methods = [r for r in reused if r['case_id'] == case]+selected
        fig, axes = plt.subplots(len(methods)+1, 4, figsize=(14, 2.35*(len(methods)+1)))
        raw = truth['expected_e'] if case.endswith('continuous_e') else truth['raw_dn']
        panels = [(raw, 'input continuous e' if case.endswith('continuous_e') else 'input quantized DN', 'gray', None),
                  (truth['temperature_observed_color_K'], 'truth observed color K', 'inferno', None),
                  (truth['template_e'], 'truth signed template e', 'RdBu_r', 190),
                  (truth['saturated'], 'stored DN saturation', 'gray', None)]
        arrays = [panels]
        for row in methods:
            root = OLD_DATA if row.get('reused_from_audit_v2') else args.data_root
            with np.load(root/row['prediction_path'], allow_pickle=False) as f:
                pred = {k: f[k].copy() for k in f.files}
            arrays.append([(pred['temperature_K'], row['method']+' color K', 'inferno', None),
                           (pred['temperature_K']-truth['temperature_observed_color_K'], 'color error K', 'RdBu_r', 110),
                           (pred['template_e'], 'signed template e', 'RdBu_r', 190),
                           (pred['valid'], 'valid (fixed scoring domain separate)', 'gray', None)])
        for axrow, panels in zip(axes, arrays):
            for ax, (values, title, cmap, limit) in zip(axrow, panels):
                im = ax.imshow(values, cmap=cmap, vmin=-limit if limit else None, vmax=limit)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=.045)
        fig.tight_layout()
        path = output/(case+'_ablation.png')
        fig.savefig(path, dpi=120); plt.close(fig)
        figures.append(str(path.relative_to(args.output)).replace('\\', '/'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for row in [r for r in rows if r['case_id'] == 'front_continuous_e']:
        trace = row.get('trace', [])
        for ax, key in zip(axes, ('temperature_observed_RMSE_K', 'template_RMSE_e', 'historical60_null_FP_fixed_domain')):
            ax.plot(range(len(trace)), [s['metrics'][key] for s in trace], '.-', label=row['method'])
            ax.set_ylabel(key); ax.set_xlabel('saved trace index (includes scoring-only preview)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    path = output/'front_stage_curves.png'
    fig.savefig(path, dpi=140); plt.close(fig)
    figures.append(str(path.relative_to(args.output)).replace('\\', '/'))
    return figures
