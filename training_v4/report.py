"""Offline plots, checkpoint reopening, dataset/cache audit and final report."""
from pathlib import Path
import json
import numpy as np
import torch
import runtime
from evaluation_v2.common import Journal, canonical_hash, array_hash
from training_v4.common import OUT, DATA, setup, read, write, npz
from training_v4.cache import cache_identity
from training_v4.analysis import paired_analysis


def report():
    setup()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    manifest = read(OUT/'data_manifest.json')
    selection = read(OUT/'backbone_selection.json')
    preserved = read(OUT/'preserved_baseline.json')
    assert all(runtime.sha256(runtime.ROOT/p) == h for p, h in preserved['original_files'].items())
    final_registration = read(OUT/'final_test_registration.json')
    assert not final_registration['opened_for_selection']
    assert final_registration['raw_generated'] == final_registration['evaluations'] == 0
    assert not list((DATA/'raw').glob('v4_final_unopened_*'))
    assert not list((DATA/'labels').glob('v4_final_unopened_*'))
    rows = manifest['rows']
    assert len({r['raw_hash'] for r in rows}) == len(rows) == 2880
    train_parents = {r['mother_id'] for r in rows if r['split'] == 'train'}
    val_parents = {r['mother_id'] for r in rows if r['split'] == 'validation'}
    final_parents = {r['mother_id'] for r in final_registration['parents']}
    assert len(train_parents) == 256 and len(val_parents) == len(final_parents) == 64
    assert not train_parents & val_parents and not (train_parents | val_parents) & final_parents
    write(OUT/'dataset_integrity_receipt.json', {'train_mothers': 256, 'validation_mothers': 64,
        'final_registered_mothers': 64, 'mother_overlap': 0, 'duplicate_raw_hashes': 0,
        'all_rows': 2880, 'final_raw_directories_found': 0, 'final_label_directories_found': 0,
        'final_test_used': False})
    native_receipt = read(OUT/'native_identity_and_checkpoint_receipt.json')
    # Cache identity includes the producer's Torch version. The reporting
    # interpreter may use CPU Torch; it must not invent another cache key.
    producer_torch = native_receipt['torch']
    raw_checks, cache_checks, labels = [], [], set()
    solver_log = Journal(OUT/'cache_solver_diagnostics.jsonl')
    if solver_log.path.exists():
        solver_log.path.unlink()
    for row in rows:
        assert runtime.sha256(DATA/row['raw_path']) == row['raw_file_sha256']
        assert array_hash(npz(DATA/row['raw_path'])['raw_dn']) == row['raw_hash']
        raw_checks.append(row['raw_hash'])
        labels.add((row['label_path'], row['label_sha256']))
        key = canonical_hash({'raw_hash': row['raw_hash'], 'physics_config_sha256': canonical_hash(manifest['physics_config']),
            'calibration_sha256': row['calibration_sha256'], 'method': selection['method'],
            'solver_config': selection['solver_config'],
            'engine_sha256': runtime.sha256(runtime.ROOT/'training_v4/scalar_engine.py'),
            'engine': 'single_frame_cached_Jacobian_v3_math', 'dtype': 'float32', 'torch': producer_torch})
        path = DATA/'cache'/key[:2]/(key+'.npz')
        meta = read(path.with_suffix('.json'))
        assert meta['raw_hash'] == row['raw_hash'] and not meta['labels_used']
        assert meta['solver_config_sha256'] == canonical_hash(selection['solver_config'])
        assert meta['physics_config_sha256'] == row['physics_config_sha256']
        assert meta['calibration_sha256'] == row['calibration_sha256']
        assert runtime.sha256(path) == meta['cache_sha256']
        cached = npz(path)
        assert cached['state'].shape == (5, 256, 256) and cached['beta'].shape == (3,)
        assert np.isfinite(cached['state']).all() and np.isfinite(cached['beta']).all()
        steps = meta['history']['joint']
        solver_log.append({'case_id': row['case_id'], 'raw_hash': row['raw_hash'], 'cache_key': key,
            'outer_steps': len(steps), 'accepted_steps': sum(s['accepted'] for s in steps),
            'rejected_outer_steps': sum(not s['accepted'] for s in steps),
            'CG_tolerance_not_reached_steps': sum(not s['CG']['converged'] for s in steps),
            'CG_stop_reasons': [s['CG']['stop'] for s in steps],
            'CG_relative_residuals': [s['CG']['relative_residual'] for s in steps],
            'accepted_actual_predicted_ratios': [s['actual_predicted_ratio'] for s in steps if s['accepted']],
            'meaning': 'numerical stop diagnostics, not a removed evaluation frame'})
        cache_checks.append({'case_id': row['case_id'], 'raw_hash': row['raw_hash'],
            'cache_key': key, 'cache_sha256': meta['cache_sha256'],
            'metadata_sha256': runtime.sha256(path.with_suffix('.json')),
            'physics_config_sha256': meta['physics_config_sha256'], 'calibration_sha256': meta['calibration_sha256']})
    for path, digest in labels:
        assert runtime.sha256(DATA/path) == digest
    write(OUT/'cache_hash_inventory.json', cache_checks)
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    receipts, failed = [], []
    for folder in sorted(OUT.glob('seed_*')):
        if not (folder/'training_receipt.json').exists():
            continue
        receipt = read(folder/'training_receipt.json')
        paired_analysis(receipt['seed'])
        assert receipt['optimizer_updates'] == 2000
        training = Journal(folder/'training_curve.jsonl').rows()
        assert [r['optimizer_update'] for r in training] == list(range(1, 2001))
        validation = Journal(folder/'validation_curve.jsonl').rows()
        assert [r['optimizer_update'] for r in validation] == list(range(100, 2001, 100))
        checkpoint = receipt['selected_checkpoint'] or receipt['final_checkpoint']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['optimizer_updates'] > 0 and payload['formal_training']
        assert payload['parameters_sha256'] != payload['initial_parameters_sha256']
        assert sum(t.numel() for t in payload['cnn_state_dict'].values()) == 5797
        pairs = Journal(folder/'paired_validation.jsonl').rows()
        assert len(pairs) == 576*6
        for r in pairs:
            if r['status'] != 'SUCCESS':
                failed.append(r)
            else:
                assert runtime.sha256(DATA/r['prediction_path']) == r['prediction_sha256']
        for event in validation:
            if not event['development_gates_passed']:
                gate_path = folder/f'gates_update_{event["optimizer_update"]:04d}.json'
                decisions = read(gate_path)['decisions']
                failed.append({'kind': 'DEVELOPMENT_GATE_REJECTED_CHECKPOINT', 'seed': receipt['seed'],
                    **event, 'gate_path': str(gate_path.relative_to(OUT)),
                    'failed_gate_decisions': [g for g in decisions if not g['passed']]})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        xs = np.array([r['optimizer_update'] for r in training])
        for key in ('total', 'template', 'color', 'signed_blank'):
            axes[0].plot(xs, [r['loss'][key] for r in training], alpha=.45, linewidth=.7, label=key)
        axes[1].plot([r['optimizer_update'] for r in validation], [r['validation_loss'] for r in validation], 'o-')
        accepted = [r for r in validation if r['development_gates_passed']]
        axes[1].scatter([r['optimizer_update'] for r in accepted], [r['validation_loss'] for r in accepted],
                        color='green', s=65, marker='s', label='all 16 gates pass', zorder=4)
        axes[1].legend(fontsize=8)
        axes[2].plot(xs, [r['gradient_norm_before_clip'] for r in training], linewidth=.7)
        for ax, title in zip(axes, ('real minibatch losses', 'full validation every 100 updates', 'gradient norm before clip=1')):
            ax.set_title(title); ax.set_xlabel('optimizer update'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        path = figures/(folder.name+'_real_training_curves.png')
        fig.savefig(path, dpi=140); plt.close(fig)
        noisy = [r for r in pairs if not r['clean_identity'] and r['status'] == 'SUCCESS']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        for name, marker in (('zero_template', '_'), ('repair_physics_validation_selected', 'x'), ('trained_CNN', 'o')):
            group = sorted([r for r in noisy if r['condition'] == 'weak' and r['method'] == name],
                           key=lambda r: (r['mother_id'], r['case_id']))
            axes[0].scatter(np.arange(len(group)), [r['metrics']['template_relative_L2'] for r in group],
                            s=12, alpha=.6, marker=marker, label=name)
        axes[0].axhline(1, color='black', linewidth=.8, linestyle='--')
        axes[0].set(xlabel='weak validation frame (same parents, two noise replicates)', ylabel='template relative L2',
                    title='Weak signals versus the zero estimate L2=1')
        axes[0].legend(fontsize=7)
        deltas = Journal(folder/'paired_differences.jsonl').rows()
        for condition in ('blank', 'weak', 'normal', 'strong'):
            group = [r['template_RMSE_delta_e'] for r in deltas if r['baseline'] == 'repair_physics_validation_selected'
                     and not r['clean_identity'] and r['condition'] == condition and r['paired_success']]
            axes[1].hist(group, bins=20, alpha=.45, label=condition)
        axes[1].axvline(0, color='black', linewidth=.8)
        axes[1].set(xlabel='trained CNN RMSE - fixed physics RMSE (e)', ylabel='frame count',
                    title='Same-input paired error differences')
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figures/(folder.name+'_weak_and_paired.png'), dpi=150); plt.close(fig)
        receipts.append(receipt)
    assert receipts
    for path in OUT.rglob('failure_record.json'):
        failed.append(read(path))
    failed += [r for r in Journal(OUT/'cache_per_case.jsonl').rows() if r['status'] != 'SUCCESS']
    failed.append({'kind': 'REJECTED_BATCHED_PHYSICS_ACCELERATION', **read(OUT/'batched_engine_rejected.json')})
    write(OUT/'failure_records.json', failed)
    overfit = Journal(OUT/'overfit_eight_curve.jsonl').rows()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r['optimizer_update'] for r in overfit], [r['eight_sample_loss'] for r in overfit], 'o-')
    ax.set(xlabel='optimizer update', ylabel='true-target loss', title='Actual eight-sample overfit, separate initialization')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/'eight_sample_overfit.png', dpi=150); plt.close(fig)
    image_receipts = []
    for path in figures.glob('*.png'):
        with Image.open(path) as im:
            im.load()
            image_receipts.append({'path': str(path.relative_to(OUT)), 'size': list(im.size), 'sha256': runtime.sha256(path)})
    first = receipts[0]
    paired = read(OUT/f'seed_{first["seed"]}/paired_summary.json')
    benefit = read(OUT/f'seed_{first["seed"]}/benefit_decision.json')
    assert first['seed'] == 101
    assert len(receipts) == (3 if benefit['clear_validation_gain'] else 1)
    history = read(OUT/f'seed_{first["seed"]}/all_checkpoint_validation_summary.json')
    terminal = next(event for event in history if event['optimizer_update'] == 2000)
    terminal_weak = next(g for g in terminal['groups'] if not g['clean_identity'] and g['condition'] == 'weak')
    terminal_normal = next(g for g in terminal['groups'] if not g['clean_identity'] and g['condition'] == 'normal')
    outcome = ('验证收益达到扩种子条件。' if benefit['clear_validation_gain'] else
               '本次未达到扩种子条件，停止于一个正式种子。可训练接口与真实训练已完成，但不能把它宣称为全部性能目标通过。')
    lines = ['# TRAIN_V4_REPORT：物理反演后 CNN 校正器', '',
        '基于 `9a7ac7e86339ba9b9d3006765fef731f10941029`。本轮实际训练并保存新 checkpoint，模型为 5,797 参数的 19→16→16 共享 CNN、两个残差头，六次有界校正。原 physics/object_blur、repair_v3、旧 CNN/权重和旧报告均保留。', '',
        f'正式完成种子 {", ".join(str(r["seed"]) for r in receipts)}，每个种子 2,000 次 Adam optimizer updates、8,000 次 microbatch、每 100 次完整验证。首个训练 checkpoint 的选择更新为 {first["selected_update"]}；新参数已实际改变。明确验证收益={benefit["clear_validation_gain"]}，弱信号 pooled L2={benefit["weak_pooled_L2"]:.4f}，零估计 L2=1。{outcome}', '',
        f'末次第 2,000 次权重在含噪弱信号上 L2={terminal_weak["template_L2"]:.4f}，正常模板 RMSE={terminal_normal["template_RMSE_e"]:.4f} e，说明小 CNN 确实学到了恢复能力；但原门禁只有 {terminal["gates_passed_count"]}/16 通过，因此它是诊断权重。受保护的第 {first["selected_update"]} 次权重通过16/16，却仍未胜过弱信号零估计。这两种结果不能合并成一个“全部达标”结论。', '',
        '## 骨干的有限选择', '',
        f'不含 warm start 的原 joint_update 通过原 16 项门禁。32 个新验证 raw 上，组合 v3 的监督评分 {selection["warm_comparison_loss"]["consistent_init__joint_update_v3"]:.5f}，old_init + joint_update 为 {selection["warm_comparison_loss"]["old_init__joint_update_v3"]:.5f}。选中 {selection["method"]}，模板正则 {selection["selected_lambda"]}，并通过全部 16 项门禁。组合 v3 仍作为原始参考保留，未完整重跑旧 audit。', '',
        '|模板正则|32 帧验证评分|', '|---:|---:|']
    lines += [f'|{r["lambda"]}|{r["validation_loss"]:.6f}|' for r in selection['finite_template_regularization_grid']]
    lines += ['', '只有验证数据用于该三点参数选择，固定开发门禁只作可接受性约束。传统解调 Tikhonov 同样仅在这 32 个验证 raw 上作三点选择。没有扩大求解器搜索。', '',
        '跨帧批处理因 float64 CG 轨迹与逐帧结果不一致而被拒绝，其失败测试和日志保留。生产直接调用原 v3 scalar warm/LM/PCG，只缓存每个线性步不变的 Jacobian；并行任务之间独立，不做跨帧 CG。', '',
        '## 数据、缓存和梯度接口', '',
        '256 个训练母场景与 64 个验证母场景，四条件各两次独立 Poisson/read-noise/ADC 噪声，共 2,048/512 个主 raw；另加每母场景一个干净身份 raw，共 256/64。全部为原 256×256 编码器。最终测试的 64 个母场景只登记，生成与评估均为零。原 expanded/audit/repair 只作开发/回归，没有进入主训练。', '',
        '温度、幅度、模板形态使用独立随机流。多方向前沿、多峰、环/条、平坦和平滑随机场通过场景坐标旋转/平移/缩放后再编码，未直接旋转或裁剪 raw。父场景派生的条件、噪声与身份样本从不跨 split。', '',
        '标签独立保存在 labels/，仅由真实合成 latent 场经原物理模块得到 postblur 模板、色温和七波段。B4/v3 输出从未用作真值。所有 raw 的匹配缓存保存 state/beta；键包含 actual raw dtype/shape/bytes hash、物理配置、标定、固定求解器和引擎 hash。噪声变化使 raw hash 改变，旧缓存不能通过检查。缓存的逐例 CG 停止原因、未达到相对残差条件的步数和 LM 接受记录摘要保存在 cache_solver_diagnostics.jsonl，完整历史在独立的缓存元数据中；没有把数值未收敛帧删除。', '',
        '网络从缓存 tensor 开始，在每阶段重算 19 通道物理特征。逆过程与缓存 tensor detach，六次 CNN→物理输出保持可导；没有从 NumPy 预测反传，也没有把整个 warm/LM/PCG 纳入图。它明确是“物理反演后CNN校正器”，不是完整展开训练。', '',
        '## 身份与实际训练检查', '',
        '零头六阶段输出与物理缓存恒等；活跃 continuum/emission 头梯度非零，第一次零头时主干梯度为零符合链式求导，第二步主干梯度出现。Adam 确实改变参数。标准化残差的 RMS 在 sqrt 前 clamp，以保证恰好零残差的干净样本也有有限梯度。', '',
        f'8 样本独立诊断运行 200 次真实更新，损失由 {read(OUT/"overfit_eight_receipt.json")["before_loss"]:.6f} 降为 {read(OUT/"overfit_eight_receipt.json")["after_loss"]:.6f}，参数 hash 改变并保存 overfit_eight.pt。该诊断权重没有用于正式初始化。', '',
        '主损失为 template/60 的平方误差、0.2×color/75、0.2×bands/300、0.01×raw 加权一致性，以及空白的有符号平方误差。弱信号主损失不除真值能量。Adam lr=1e−4，batch=1，累积4，clip=1；不使用 U-Net、注意力、新网络模块、阈值提高或删除失败帧。', '',
        '## 同输入六方法对照', '',
        '下表主验证含噪 raw 的每组 128 帧。另 64 个身份样本在 paired_summary 中单独保留。所有方法使用同一 raw，物理配置与标定一致；旧 CNN 迁移指把旧两个头及主干接在新固定物理缓存之后，未经训练，与原完整 B4 区分。零模板的色温未定义，不填假结果。', '',
        '|条件|方法|帧数|模板 RMSE e|模板 pooled L2|色温 RMSE K|空白 >60e 像素误报|失败数|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'
    for r in paired:
        if r['clean_identity']:
            continue
        lines.append(f'|{r["condition"]}|{r["method"]}|{r["records"]}|{fmt(r["pooled_template_RMSE_e"])}|{fmt(r["pooled_template_L2"])}|{fmt(r["pooled_color_RMSE_K"])}|{fmt(r["mean_blank_FP60"])}|{r["failures"]}|')
    lines += ['', '逐帧配对的误差差值保存在 paired_differences.jsonl。paired_confidence_intervals.json 使用母场景为单位的 2,000 次 bootstrap，同一母场景的噪声副本一起重采样；这些区间仅用于结果分析，不用于选参或 checkpoint 选择。', '',
        '末次 2000 更新的诊断配对保存在 terminal_2000_paired_diagnostic.jsonl，与旧 B4、纯物理逐例按同一 raw/cache hash 对齐，复用已经完成的完整验证记录，没有新增推理，也没有参与模型选择。', '',
        '## 门禁与训练目标的取舍', '',
        '下表保留所有保存候选的验证成绩。完成 2,000 次训练与选中哪次更新是两个不同事实：最终权重始终保存，只有通过全部门禁的候选可以被选为受保护版本。低验证损失不能抵消确定性回归。', '',
        '|更新|完整验证 loss|门禁通过数/16|含噪弱模板 L2|含噪正常模板 RMSE e|',
        '|---:|---:|---:|---:|---:|']
    for event in history:
        weak = next(g for g in event['groups'] if not g['clean_identity'] and g['condition'] == 'weak')
        normal = next(g for g in event['groups'] if not g['clean_identity'] and g['condition'] == 'normal')
        lines.append(f'|{event["optimizer_update"]}|{event["validation_loss"]:.6f}|{event["gates_passed_count"]}|{weak["template_L2"]:.4f}|{normal["template_RMSE_e"]:.4f}|')
    lines += ['', '新增训练母场景的模板为正幅度，推理输出仍有符号；原门禁中的负模板照常评分。对这些回归不能使用 ReLU、固定零输出或更高阈值遮掩。门禁拒绝详情与旧批处理加速失败均在 failure_records.json，所有原始候选权重保留。', '']
    lines += ['', '## Checkpoint、曲线与限制', '']
    for r in receipts:
        lines += [f'- 种子 {r["seed"]}：最终训练 checkpoint `{r["final_checkpoint"]}`，SHA256 `{r["final_checkpoint_sha256"]}`；验证选中更新 `{r["selected_update"]}`。',
                  f'  可下载的保存副本：[最终 2000 更新](../../../training_v4/checkpoints/seed_{r["seed"]}_final_2000.pt)、[门禁约束下验证选中](../../../training_v4/checkpoints/seed_{r["seed"]}_validation_selected.pt)。两个文件均逐字节匹配原训练保存文件，native_identity_and_checkpoint_receipt.json 记录完整 hash 与 Adam 实际 step。',
                  f'![种子 {r["seed"]} 真实训练曲线](figures/seed_{r["seed"]}_real_training_curves.png)', '']
        lines += [f'![弱信号与同输入配对误差](figures/seed_{r["seed"]}_weak_and_paired.png)', '']
    lines += ['![8样本实际过拟合](figures/eight_sample_overfit.png)', '',
        '每次验证都保存逐例 JSONL；开发门禁失败的候选 checkpoint 保留而不进入选中集合。最终测试没有打开；当前数字是开发验证结果，非最终泛化或实测相机性能。原响应矩阵/QE 仍为工程合成假设。', '',
        '原目录复现使用同一本机 GPU PyTorch 解释器，E/D checkout 下执行 `python -B -m training_v4.run --phase data`、`--phase select`、`--phase overfit`、`--phase train --seed 101`、`--phase compare --seed 101`。然后运行 `python -B -m training_v4.verify_delivery`，报告环境执行 `--phase report`。全新目录用 `training_v4/reproduce.ps1 -Worktree E:\\CodexWork\\thermometry\\v4_reproduction_001`，避免覆盖本次日志；该入口创建 9a7ac7e 基线 worktree 并只恢复 16 项门禁输入，不重跑旧954项。冻结输入、标签、缓存与详细 hash 在 data_manifest.json / cache_hash_inventory.json，数组位于 E 盘 ignored outputs/training_v4/run_001。', '',
        '本轮止于可训练的小 CNN 校正器。接回物理更新以及完整六阶段展开训练属于后续独立研究，没有作为本轮开始训练前的阻塞项。', '']
    (OUT/'TRAIN_V4_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write(OUT/'delivery_receipt.json', {'raw_files_verified': len(raw_checks), 'matching_caches_reopened': len(cache_checks),
        'label_files_verified': len(labels), 'preserved_original_files': len(preserved['original_files']),
        'training_seeds_completed': [r['seed'] for r in receipts], 'updates_per_seed': 2000,
        'new_checkpoint_reopened': True, 'parameter_count': 5797, 'unfolded_training': False,
        'final_test_generated': 0, 'final_test_evaluated': 0, 'old_954_full_rerun': False,
        'figures': image_receipts, 'report_sha256': runtime.sha256(OUT/'TRAIN_V4_REPORT.md')})
    print({'report_completed': True, 'seeds': [r['seed'] for r in receipts]}, flush=True)
