"""Rebuild all summaries and scientific figures from durable per-case records."""
import collections
import json
from pathlib import Path
import numpy as np
import runtime
from .common import Journal, outcome_counts
from .metrics import aggregate_metrics, all_required_screens, distribution
from .methods import METHODS
from .protocol import PROTOCOL


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_effects(rows):
    results = []
    rng = np.random.default_rng(73117001)
    for condition in PROTOCOL["conditions"]:
        subset = [r for r in rows if r["suite"] == "smoke" and r["condition"] == condition and r["status"] == "SUCCESS"]
        by = {(r["case_id"], r["method"]): r for r in subset}
        for other in ("B0_constant_train", "B0_mean_field_train", "B1_demod_fit", "B2_six_stage_no_CNN", "B3_physics_iterated", "B1_tikhonov"):
            for key in ("temperature_observed_RMSE_K", "template_relative_L2", "template_RMSE_e"):
                parent_deltas = collections.defaultdict(list)
                for a in subset:
                    if a["method"] != "B4_published":
                        continue
                    b = by.get((a["case_id"], other))
                    if b is None or a["metrics"][key] is None or b["metrics"][key] is None:
                        continue
                    assert a["input_hash"] == b["input_hash"]
                    parent_deltas[a["parent_scene_id"]].append(a["metrics"][key]-b["metrics"][key])
                values = np.array([np.mean(v) for v in parent_deltas.values()])
                boots = rng.choice(values, (2000, len(values)), replace=True).mean(1) if values.size else []
                results.append({"condition": condition, "metric": key, "contrast": "B4_published minus "+other,
                                "independent_parents": len(values), "mean_effect": float(values.mean()) if values.size else None,
                                "group_bootstrap_95pct_interval": np.quantile(boots, [.025, .975]).tolist() if values.size else None,
                                "negative_means_lower_error_for_B4": True,
                                "limit": "12 synthetic parents pooled across six families; only 2 parents per family. Diagnostic evidence, not independent hardware acceptance"})
    return results


def method_screens(rows):
    out = {}
    for method in METHODS:
        subset = [r for r in rows if r["suite"] == "smoke" and r["method"] == method]
        grouped = {c: aggregate_metrics([r for r in subset if r["condition"] == c]) for c in PROTOCOL["conditions"]}
        def signal_pass(c):
            g = grouped[c]
            if g["cases"] != 12 or not g["complete"]:
                return None
            l2, temp = g["pooled_template_L2"], g["pooled_temperature_RMSE_K"]
            bias = g["per_case_absolute_integral_bias"]["mean"]
            return bool(l2 is not None and l2 <= .25 and temp is not None and temp <= 75 and bias is not None and bias <= .1)
        nulls = [r for r in subset if r["condition"].startswith("null_")]
        null_pass = bool(all(r.get("metrics") and r["metrics"]["null_FP_fixed_domain"] is not None and
                             r["metrics"]["null_FP_fixed_domain"] <= .01 for r in nulls)) if len(nulls) == 24 else None
        robust = [r for r in subset if r["scene"] in ("channel_gain_3pct", "non_gray_continuum") and r["condition"].startswith("signal")]
        robustness = bool(all(r.get("metrics") and r["metrics"]["temperature_reference_RMSE_K"] is not None and
                              r["metrics"]["temperature_reference_RMSE_K"] <= 75 and r["metrics"]["template_relative_L2"] <= .25
                              for r in robust)) if len(robust) == 8 else None
        screens = {"numerical_execution_passed": all(r["status"] == "SUCCESS" for r in subset) if len(subset) == 48 else None,
                   "nominal_screen_passed": signal_pass("signal_nominal"),
                   "weak_signal_screen_passed": signal_pass("signal_weak"),
                   "null_screen_passed": null_pass, "robustness_screen_passed": robustness}
        out[method] = {"screens": screens, **all_required_screens(screens),
                       "scope": "preregistered legacy internal engineering screens; no scientific/hardware acceptance",
                       "grouped": grouped}
    return out


def published_prediction_check(args, rows):
    checks = []
    root = runtime.ROOT/"reports/expanded_evaluation_001"
    for r in rows:
        meta = r["metadata"]
        if r["suite"] != "legacy" or r["method"] != "B4_published" or r["status"] != "SUCCESS" or "published_representative_sha256" not in meta:
            continue
        source_suite = "experimental_shape_test" if "partition" in meta else "synthetic"
        original = root/source_suite/"representatives"/f"{meta['family']}_{meta['condition']}.npz"
        assert runtime.sha256(original) == meta["published_representative_sha256"]
        with np.load(original, allow_pickle=False) as old, np.load(args.data_root/r["prediction_path"], allow_pickle=False) as new:
            s = float(np.max(np.abs(old["pred_template_e"].astype(float)-new["template_e"])))
            t = float(np.max(np.abs(old["pred_color_temperature_K"].astype(float)-new["temperature_K"])))
        checks.append({"case_id": r["case_id"], "max_template_difference_e": s, "max_temperature_difference_K": t,
                       "comparison": "current 4060 CUDA output vs archived 5070 CUDA output; not byte identity",
                       "diagnostic_tolerance_e": .5, "diagnostic_tolerance_K": .1,
                       "within_diagnostic_tolerance": s <= .5 and t <= .1})
    return {"checks": checks, "count": len(checks), "all_within_diagnostic_tolerance": all(c["within_diagnostic_tolerance"] for c in checks)}


def plots(args, rows, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = args.output/"figures"
    output.mkdir(exist_ok=True)
    chosen = ["front_continuous_e", "front_quantization_only", "front_old_noisy_quantized",
              "legacy_experimental_shape_test_024_signal_weak", "legacy_synthetic_15_00_null_clean",
              "morph_filament_fixed_peak"]
    actual_flat = next(f["case_id"] for f in manifest["cases"] if f["suite"] == "legacy" and
                       f["metadata"].get("partition") == "flat_flame_OH" and f["metadata"].get("replicate") == 0 and
                       f["metadata"]["condition"] == "signal_weak")
    chosen[3] = actual_flat
    paths = []
    for case_id in chosen:
        f = next(v for v in manifest["cases"] if v["case_id"] == case_id)
        case_rows = [r for r in rows if r["case_id"] == case_id and r["status"] == "SUCCESS" and "prediction_path" in r]
        with np.load(args.data_root/f["path"], allow_pickle=False) as v:
            a = {k: v[k].copy() for k in v.files}
        preds = {}
        for r in case_rows:
            with np.load(args.data_root/r["prediction_path"], allow_pickle=False) as v:
                preds[r["method"]] = {k: v[k].copy() for k in v.files}
        if not preds:
            continue
        smax = max(float(np.max(np.abs(a["template_e"]))), max(float(np.max(np.abs(p["template_e"]))) for p in preds.values()), 1.)
        terrmax = max((float(np.max(np.abs(p["temperature_K"]-a["temperature_observed_color_K"]))) for p in preds.values() if np.isfinite(p["temperature_K"]).all()), default=1.)
        terrmax = max(terrmax, 1.)
        fig, axes = plt.subplots(1+len(preds), 5, figsize=(18, 2.45*(1+len(preds))), squeeze=False)
        header = [(a["raw_dn"], "frozen raw DN", "gray", None),
                  (a["temperature_observed_color_K"], "truth observed color K", "inferno", None),
                  (a["template_e"], "truth postblur signed template e", "RdBu_r", smax),
                  (a["evaluation_domain"], "fixed evaluation domain", "gray", None),
                  (a["saturated"], "raw measurement saturation", "gray", None)]
        for ax, (arr, title, cm, lim) in zip(axes[0], header):
            im = ax.imshow(arr, cmap=cm, vmin=-lim if lim else None, vmax=lim)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=.045)
            ax.set_xticks([]); ax.set_yticks([])
        for index, (name, pred) in enumerate(preds.items(), 1):
            temp = pred["temperature_K"]
            signed = pred["template_e"]
            valid = pred["valid"]
            panels = [(temp, name+"\ncolor K", "inferno", None),
                      (temp-a["temperature_observed_color_K"], "temperature error K", "RdBu_r", terrmax),
                      (signed, "signed template e", "RdBu_r", smax),
                      (signed-a["template_e"], "template error e", "RdBu_r", smax),
                      (valid, "model valid (not scoring domain)", "gray", None)]
            for ax, (arr, title, cm, lim) in zip(axes[index], panels):
                im = ax.imshow(arr, cmap=cm, vmin=-lim if lim else None, vmax=lim)
                ax.set_title(title, fontsize=8)
                fig.colorbar(im, ax=ax, fraction=.045)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(case_id+"; all registered methods; symmetric error/template scales; Z0 temperature N/A", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .985))
        path = output/(case_id+".png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path.relative_to(args.output)).replace("\\", "/"))
    # Every data step/CNN correction is retained, not only selected endpoints.
    traced = [r for r in rows if r["case_id"].startswith("front_") and r.get("trace")]
    if traced:
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        for column, case_id in enumerate(chosen[:3]):
            for r in [v for v in traced if v["case_id"] == case_id]:
                ms = [s["metrics"] for s in r["trace"]]
                xs = np.arange(len(ms))
                for ax, key in zip(axes[:, column], ("template_RMSE_e", "template_signed_mean_e", "historical60_null_FP_fixed_domain")):
                    ax.plot(xs, [m[key] for m in ms], marker=".", ms=2, label=r["method"])
                    ax.set_ylabel(key)
                    ax.grid(alpha=.2)
                axes[0, column].set_title(case_id)
                axes[-1, column].set_xlabel("saved step index; see per_case.jsonl labels")
            axes[0, column].legend(fontsize=7)
        fig.tight_layout()
        path = output/"front_stage_diagnostics.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        paths.append(str(path.relative_to(args.output)).replace("\\", "/"))
    # Preserve saturation diagnostic as a separate view, outside accepted methods.
    sat = runtime.ROOT/"reports/expanded_evaluation_001/synthetic/representatives/bright_saturated_signal.npz"
    with np.load(sat, allow_pickle=False) as v:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, key in zip(axes, ("raw_DN", "saturated", "pred_template_e", "valid")):
            im = ax.imshow(v[key], cmap="RdBu_r" if key == "pred_template_e" else "gray")
            ax.set_title("historical diagnostic: "+key, fontsize=8); ax.axis("off"); fig.colorbar(im, ax=ax)
        fig.suptitle("SATURATION_REJECTED for every v2 deployment method; old output is failure diagnosis only")
        fig.tight_layout()
        path = output/"saturation_rejection_historical.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        paths.append(str(path.relative_to(args.output)).replace("\\", "/"))
    return paths


def summarize_run(args):
    manifest = read(args.output/"frozen_manifest.json")
    journal = Journal(args.output/"per_case.jsonl")
    attempts = journal.rows()
    latest = {r["identity_sha256"]: r for r in attempts}
    rows = list(latest.values())
    expected = (manifest["counts"]["smoke"]+manifest["counts"]["legacy"]+manifest["counts"]["diagnostics"])*len(METHODS)+2
    counts = outcome_counts(attempts, expected)
    groups = collections.defaultdict(list)
    strength_groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["scene"], r["condition"], r["method"])].append(r)
        strength_groups[(r["suite"], r["condition"], r["method"])].append(r)
    stratified = [{"suite": k[0], "scene": k[1], "condition": k[2], "method": k[3], **aggregate_metrics(v)} for k, v in sorted(groups.items())]
    strengths = [{"suite": k[0], "condition": k[1], "method": k[2], **aggregate_metrics(v)} for k, v in sorted(strength_groups.items())]
    runtime.write_json(args.output/"stratified_summary.json", stratified)
    runtime.write_json(args.output/"strength_summary.json", strengths)
    runtime.write_json(args.output/"paired_parent_effects.json", paired_effects(rows))
    runtime.write_json(args.output/"screen_semantics.json", method_screens(rows))
    numeric = [r for r in rows if r["input_kind"] == "oracle_continuous_e"]
    runtime.write_json(args.output/"oracle_numeric_diagnostics.json", numeric)
    calibration = [r for r in rows if r.get("known_calibration_oracle") or r["suite"] == "legacy" and r["scene"] == "channel_gain_3pct"]
    runtime.write_json(args.output/"known_calibration_diagnostics.json", calibration)
    fails = [{"case_id": r["case_id"], "method": r["method"], "status": r["status"], "metrics": r.get("metrics"),
              "prediction_path": r.get("prediction_path"), "error": r.get("error")} for r in rows if r["status"] != "SUCCESS" or
              (r.get("metrics") and r["metrics"]["template_relative_L2"] is not None and r["metrics"]["template_relative_L2"] > .25)]
    runtime.write_json(args.output/"failure_samples.json", fails)
    repro = published_prediction_check(args, rows)
    runtime.write_json(args.output/"published_prediction_recheck.json", repro)
    fig_paths = plots(args, rows, manifest)
    baseline = read(args.output/"baseline_tracked_files.json")
    preservation = {p: runtime.sha256(runtime.ROOT/p) == h for p, h in baseline["initial_tracked_sha256"].items()}
    runtime.write_json(args.output/"baseline_preservation.json", {"original_tracked_files": len(preservation),
                "all_unchanged": all(preservation.values()), "files": preservation,
                "checkpoint_sha256": runtime.sha256(runtime.ROOT/"checkpoints/selected_checkpoint.pt")})
    successes = [r for r in rows if r["status"] == "SUCCESS"]
    receipt = {"utc": runtime.now(), **counts,
               "registered_smoke_frames": 48, "completed_smoke_frames_all_methods": len({r["case_id"] for r in rows if r["suite"] == "smoke" and all(
                   any(s["case_id"] == r["case_id"] and s["method"] == m and s["status"] == "SUCCESS" for s in rows) for m in METHODS)}),
               "explicit_saturation_rejections": sum(r["status"] == "SATURATION_REJECTED" for r in rows),
               "exceptions": sum(r["status"] == "EXCEPTION" for r in rows), "timeouts": sum(r["status"] == "TIMEOUT" for r in rows),
               "optimizer_updates": 0, "all_original_tracked_files_unchanged": all(preservation.values()),
               "peak_method_GPU_allocated_MiB": max((r.get("peak_GPU_allocated_MiB", 0) for r in rows), default=0),
               "method_seconds": {m: distribution(r.get("seconds") for r in successes if r["method"] == m) for m in METHODS},
               "summary_from_jsonl_only": True, "figures": fig_paths,
               "first_round_delivered": counts["remaining"] == 0 and all(preservation.values()),
               "scientific_acceptance": "not_accepted", "real_temperature_species_hardware_validated": False,
               "no_automatic_training": True, "no_automatic_upload": True,
               "deferred": PROTOCOL["deferred"]}
    runtime.write_json(args.output/"final_receipt.json", receipt)
    # The report is a concise evidence index. All pooled/per-case metrics, source
    # support results, raw residuals, timings and convergence curves are in JSON.
    inv, training = read(args.output/"baseline_inventory.json"), read(args.output/"training_statistics.json")
    lines = ["# 单帧七通道光谱测温：首轮 v2 审计", "",
             f"本轮完成 {receipt['completed_smoke_frames_all_methods']}/48 个冻结 smoke 帧的全部 {len(METHODS)} 方法对照；另复核 50 个旧冻结帧及 21 个小型诊断输入。", "",
             f"注册 {counts['registered']} 个方法/输入组合，尝试 {counts['attempted']}，成功 {counts['completed']}，拒绝/失败/不完整 {counts['failed_or_incomplete']}，未运行 {counts['remaining']}。饱和拒绝 {receipt['explicit_saturation_rejections']}，异常 {receipt['exceptions']}，超时 {receipt['timeouts']}。数值完成不代表科学接收。", "",
             "## 基线、环境与保留证据", "",
             f"原 HEAD `{inv['original_HEAD']}`，最初工作区干净。{len(preservation)} 个原已跟踪文件均未改动：{all(preservation.values())}。", "",
             f"计算使用 {inv['device']}，PyTorch {inv['torch']}，CUDA {inv['CUDA_build']}；报告绘图在已有 CPU 环境运行。全程优化器更新 0。", "",
             f"Checkpoint SHA256 `{inv['fingerprints']['checkpoint_sha256']}`。CNN 参数、物理 buffers、全部 buffers、完整配置分别见 `baseline_inventory.json`。已知标定另用方法名及物理指纹，未改发布文件。", "",
             f"生产测试通过 {inv['tests']['tests']} 项，见 `tests_gate.log/xml`。`tests_initial.log/xml` 保留首次 FP32 有限差分失败；增大扰动并检查不跨 LUT 节点后通过，未放宽原容差。", "",
             "原报告正常完成 480+288=768 次推理。50 个研究归档 PT 均按旧 manifest 哈希校验，其中 30 个发布代表 NPZ 原始输入和目标逐数组完全一致。其余完整冻结帧未下载、未重跑；旧报告的 768 次结果不能称本轮全部复现。", "",
             f"当前 CUDA 重跑与旧 CUDA 输出比对 {repro['count']} 个无饱和代表；诊断容差 0.5 e / 0.1 K 全通过：{repro['all_within_diagnostic_tolerance']}。精确差异见 `published_prediction_recheck.json`，不同 GPU 不要求重新采样噪声一致。", "",
             "## 评估口径", "",
             "真值支持区主口径为自身模糊后正峰值的 20%，真值 >= 阈值；10%/30% 为敏感性分析。预测使用严格 > 独立 validation-null 拟合的阈值；帧分数为全视场有符号模板 P99，与测试真值 ROI 无关。历史严格 >60 e 同时保留。", "",
             "固定 evaluation_domain 仅依真值有效性/构造视场，valid=False 的有限预测仍评分。非有限、缺失、越界单列状态及 null 指标；任何饱和测量全帧 SATURATION_REJECTED。条件误差始终带 coverage。原模型只有布尔 valid，risk-coverage 仅报告接受点与全覆盖点，不捏造置信度排序。", "",
             "形态帧分别评分全构造视场、source_valid_fraction>0.5 源图支持区、>0 敏感性及源外伪信号；letterbox 不是独立实测空白。温度分 latent、reference color、observed color；非均匀辐射混合的色温不称温度算术平均。旧帧 latent/reference 为归档生成规则补算，原 observed truth 保留。", "",
             "pooled-L2、逐帧 L2 均值/中位/P90/最大、有符号积分偏差及逐帧绝对偏差同时保存；信号、弱信号、两类空白分别汇总，空白 L2 为不适用。all_required_screens 由已注册五项计算，缺项 incomplete。75 K/.25/1% 是内部 legacy 筛选值。", "",
             "## 方法与验证", "",
             f"B0 恒温 {training['constant_K']:.3f} K 和平均色温场来自选中 low_lr_control 的 160 个训练 ID（30000–30159）重放；训练 raw 哈希匹配 {training['training_raw_replay_hash_matches']}/160。没有使用审计测试真值拟合。重放差异若存在详见 `training_statistics.json`，不宣称原训练数组字节已留存。", "",
             "Z0 正式零模板对照对每个非零目标 L2=1，温度及复投影不适用。B1 为解调/色温/局部模板拟合，已模糊域不会再模糊一次。B2 是相同六阶段关闭 CNN；B3 是最多 60 次纯物理更新、谱 CG12，报告真实迭代数与停止原因；停滞不证明全局最优。B4 是原发布网络。B1_tikhonov 为验证集选择的二次空间正则传统基线。", "",
             "48 smoke = 六族×两独立母场景×四条件，同母场景共享 T/a/几何，各方法读取同一冻结 DN。另有独立 48 个 validation 帧，其中 24 null 拟合检测阈值；信号/弱信号/null 验证损失选择空间正则。相同 Poisson 种子不等于相同加性噪声。", "",
             "## 48 帧主对照（按强度分层）", "",
             "|方法|名义 pooled L2|弱 pooled L2|名义色温 RMSE K|弱色温 RMSE K|", "|---|---:|---:|---:|---:|"]
    screens = method_screens(rows)
    def fmt(v):
        return "N/A" if v is None else f"{v:.4f}"
    for m in METHODS:
        gs = screens[m]["grouped"]
        n, w = gs["signal_nominal"], gs["signal_weak"]
        lines.append(f"|{m}|{fmt(n['pooled_template_L2'])}|{fmt(w['pooled_template_L2'])}|{fmt(n['pooled_temperature_RMSE_K'])}|{fmt(w['pooled_temperature_RMSE_K'])}|")
    lines += ["", "每族两母场景的完整结果见 `stratified_summary.json`；按母场景配对和 2000 次分组 bootstrap 的差值/95% 区间见 `paired_parent_effects.json`。两母场景不足以保证每族稳定性，区间是本合成诊断集的不确定性，不代表独立实测泛化。", "",
              "## 优先失败诊断", "",
              "温度前沿：相同旧母场 T/a、真实 S=0 的连续电子、仅量化和旧冻结噪声 DN 分开记录；连续电子是 oracle 数值诊断，不能混入 raw-DN 部署主表。每次初始化、data-step、CNN 修正均存 trace，第一处出现伪模板和后续变化可由下表与 `front_stage_diagnostics.png` 查看。", "",
              "|输入|方法|绝对模板 RMSE e|有符号均值 e|历史 >60e 固定域误报|色温 RMSE K|", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["case_id"].startswith("front_") and r["method"] in ("B1_demod_fit", "B2_six_stage_no_CNN", "B3_physics_iterated", "B4_published") and r.get("metrics"):
            m = r["metrics"]
            lines.append(f"|{r['metadata']['variant']}|{r['method']}|{fmt(m['template_RMSE_e'])}|{fmt(m['template_signed_mean_e'])}|{fmt(m['historical60_null_FP_fixed_domain'])}|{fmt(m['temperature_observed_RMSE_K'])}|")
    lines += ["", "平焰弱信号：12 个旧冻结输入逐一重跑，零估计作为正式对照。旧 pooled 2.27223 对应 5.163 倍零估计平方误差；不能用二值召回空分母解释回归误差。其他五族旧弱信号 L2 <1 的结论仍保留。", "",
              "|平焰弱信号方法|pooled L2|逐帧绝对积分偏差均值|", "|---|---:|---:|"]
    for g in stratified:
        if g["suite"] == "legacy" and g["scene"] == "flat_flame_OH" and g["condition"] == "signal_weak":
            lines.append(f"|{g['method']}|{fmt(g['pooled_template_L2'])}|{fmt(g['per_case_absolute_integral_bias']['mean'])}|")
    lines += ["", "热连续谱/背景：统一 T=2700/3000/3300 K、a=.6、相同母几何，分别 0、+5% 486 连续谱、固定 +100 e 486 失配；实际电子数、温度、FP 与复投影逐帧记录，避免将更亮时更大的绝对失配全部归因温度泛化。", "",
              "700/905 标定：交替 ±3% 的 700/905 增益 0.97/1.03，色比改变 -5.825%。对同一旧 DN 比较名义 B4、已知正确 A_true 的 B4 和仅给定色比修正的 B1；后两种明确为已知标定 oracle。参照色温误差、observed 误差及复投影见 `known_calibration_diagnostics.json`，不证明单帧自主标定。", "",
              "形态预算：blob/multiple/filament × 固定峰值/固定积分/固定 L2 能量，均在生成端配平，预测未缩放；真实 postblur 峰值、积分、L2 能量及各波段计数在 frozen manifest。只比较规定模糊后域，未宣称恢复超出光学信息的纹理。", "",
              "## 下一步边界与复现", "",
              "本轮停止于上述诊断，没有训练、扩大结构、删失败帧或上传数据。是否优先改求解、先验/训练分布或标定，应以配对效应、已知标定改善和连续电子失败共同判断；详细证据可从下面文件直接复算，不通过筛选值变更美化结果。", "",
              "未完成且本轮按要求暂缓：全部 20 族重新评估、大型增益/相位/强度单因素及联合扫描、新温度几何锁定最终测试、三训练种子、重训与架构搜索、硬件测量。既有扩展测试属于开发/回归集。", "",
              "独立 exposure_v2 使用 t_ref=20 μs；当前曝光信号/LUT/模板/背景电子按曝光比缩放，参考曝光模板系数标签明确，read noise 不缩放。减半曝光期望/Poisson 方差减半、读噪声方差不变已测试；旧发布模型 exposure 字段仍保留原复现语义。", "",
              "```powershell", "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:PYTHONIOENCODING='utf-8'", "$env:PYTHONPATH=(Join-Path (Get-Location) '_runtime/audit_deps')",
              "# 在已安装 GPU PyTorch 的 E 盘解释器下运行；工作区位于 E/D。",
              "python -B -m pytest tests/test_metrics_v2.py tests/test_physics_v2.py tests/test_methods_v2.py --junitxml=reports/audit_v2/run_001/tests_gate.xml",
              "python -B audit_next.py --phase inventory --output reports/audit_v2/run_001",
              "# 首次运行已完成冻结；原 frozen_manifest 存在时 freeze-smoke 会拒绝覆盖。",
              "python -B audit_next.py --phase freeze-smoke --output reports/audit_v2/run_001",
              "python -B audit_next.py --phase compare --suite all --resume --output reports/audit_v2/run_001",
              "# 绘图解释器需有 matplotlib；只读取 JSONL/NPZ，不推理。",
              "python -B audit_next.py --phase summarize --output reports/audit_v2/run_001 --device cpu", "```", "",
              "冻结输入/预测保存在 checkout 的 `outputs/audit_v2/run_001`（被忽略），报告 `reports/audit_v2/run_001` 不被忽略。跨机器复用时需同步此冻结目录并核验 manifest，不能以 CPU/CUDA 重新抽噪声冒充同输入。", "",
              "逐例文件：`per_case.jsonl`（每条 fsync、失败保留、仅精确配置+输入成功记录可 resume），`validation_per_case.jsonl`、`frozen_manifest.json`、`detector_calibration.json`、`stratified_summary.json`、`strength_summary.json`、`screen_semantics.json`、`failure_samples.json`、`final_receipt.json`。", ""]
    (args.output/"AUDIT_V2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
