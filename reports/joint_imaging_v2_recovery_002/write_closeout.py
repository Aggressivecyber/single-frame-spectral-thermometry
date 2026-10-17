"""Render compact figures and a closeout report from verified small JSON reports."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import runtime
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RUN, AUDIT = OUT / "server_results", OUT / "closeout_audit_001"
summary = json.loads((RUN / "SUMMARY.json").read_text(encoding="utf-8"))
stats = json.loads((AUDIT / "SUPPLEMENTAL_STATISTICS.json").read_text(encoding="utf-8"))
audit = json.loads((AUDIT / "AUDIT_COMPLETION.json").read_text(encoding="utf-8"))
rows = json.loads((AUDIT / "REOPENED_METRICS.json").read_text(encoding="utf-8"))
timing = json.loads((RUN / "ONLINE_TIMING.json").read_text(encoding="utf-8"))
methods = ["tv", "tv_400", "ordinary_post", "grouped_post", "full", "no_pair"]
names = ["WTV", "WTV 400", "Ordinary post", "Grouped post", "Full TV-RRU", "No pair"]
colors = ["#777777", "#b4b4b4", "#8b6ab3", "#24876d", "#3375ac", "#d58b40"]
quantities = stats["methods"]
indices = np.random.default_rng(73191).integers(0, 16, (2000, 16))
comparison_samples = {}
for method in methods:
    group = [r for r in rows[method] if r["level_e"] == 4000 and r["condition"] == "all_weak"]
    values = np.asarray([[np.mean([r["temperature_MSE_K2"] for r in group if r["mother"] == mother]),
                           *np.mean([r["emission_actual_MSE_e2"] for r in group if r["mother"] == mother], axis=0)]
                          for mother in sorted({r["mother"] for r in group})])
    comparison_samples[method] = np.sqrt(values[indices].mean(axis=1))
comparisons = {}
for numerator, denominator in (("full", "grouped_post"), ("full", "no_pair"), ("grouped_post", "ordinary_post")):
    samples = comparison_samples[numerator] / comparison_samples[denominator]
    comparisons[f"{numerator}/{denominator}"] = {"quantity_order": ["T_RMSE", "431.5_actual_RMSE", "486_actual_RMSE", "516.5_actual_RMSE"],
                                                "paired_mother_ratio_CI95": np.quantile(samples, [.025, .975], axis=0).tolist()}
runtime.write_json(OUT / "PAIRED_METHOD_COMPARISONS.json", comparisons)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
y = np.arange(len(methods))
t = np.asarray([quantities[m]["4000/all_weak"]["temperature_MSE_K2"]["point"] for m in methods])
t_ci = np.asarray([quantities[m]["4000/all_weak"]["temperature_MSE_K2"]["CI95"] for m in methods])
axes[0].barh(y, t, color=colors)
axes[0].errorbar(t, y, xerr=np.stack((t - t_ci[:, 0], t_ci[:, 1] - t)), fmt="none", color="black", capsize=3)
axes[0].set_yticks(y, names); axes[0].invert_yaxis()
axes[0].axvline(50, color="#b44b40", linestyle="--", linewidth=1)
axes[0].set_xlabel("Effective color-temperature RMSE (K)")
for c, wl in enumerate((431.5, 486, 516.5)):
    points = np.asarray([quantities[m]["4000/all_weak"]["emission_actual_MSE_e2"]["point"][c] for m in methods])
    intervals = np.asarray([[quantities[m]["4000/all_weak"]["emission_actual_MSE_e2"]["CI95"][k][c] for k in (0, 1)] for m in methods])
    axes[1].errorbar(y + (c - 1) * .14, points, yerr=np.stack((points - intervals[:, 0], intervals[:, 1] - points)), fmt="o", capsize=3, label=f"{wl:g} nm")
axes[1].set_xticks(y, names, rotation=35, ha="right")
axes[1].set_ylabel("Excess RMSE vs actual injection (e)"); axes[1].legend()
band = 100 * np.asarray([quantities[m]["4000/all_weak"]["band_relative_L2"]["point"][0] for m in methods])
ci = 100 * np.asarray([[quantities[m]["4000/all_weak"]["band_relative_L2"]["CI95"][k][0] for k in (0, 1)] for m in methods])
axes[2].barh(y, band, color=colors)
axes[2].errorbar(band, y, xerr=np.stack((band - ci[:, 0], ci[:, 1] - band)), fmt="none", color="black", capsize=3)
axes[2].set_yticks(y, names); axes[2].invert_yaxis()
axes[2].axvline(5, color="#b44b40", linestyle="--", linewidth=1)
axes[2].set_xlabel("431.5 nm image relative L2 error (%)")
fig.suptitle("Seed 811 development | 16 held-out mothers | I800 = 4000 e, all weak | descriptive 95% mother intervals", fontsize=12)
fig.savefig(OUT / "verified_comparison.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), layout="constrained")
for row_index, method in enumerate(("grouped_post", "full")):
    for col, (key, title) in enumerate((("response_visible", "Visible image response\nH_V"), ("response_continuum", "Continuum response\nH_C"), ("injection_response_columns", "Excess response\nH_E = H_V - H_C"))):
        matrix = np.asarray(quantities[method]["4000/response_matrices"][key]["point"])
        limit = .006 if key == "response_continuum" else 1.02
        im = axes[row_index, col].imshow(matrix, cmap="RdBu_r", vmin=-limit if col == 1 else -.02, vmax=limit)
        axes[row_index, col].set_xticks(range(3), ["431.5", "486", "516.5"])
        axes[row_index, col].set_yticks(range(3), ["431.5", "486", "516.5"])
        axes[row_index, col].set_title(title, fontsize=10)
        axes[row_index, col].set_xlabel("Injected band (nm)")
        axes[row_index, col].set_ylabel(method + "\nReadout band (nm)")
        for i in range(3):
            for j in range(3):
                rgba = im.cmap(im.norm(matrix[i, j]))
                brightness = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                axes[row_index, col].text(j, i, f"{matrix[i,j]:+.4f}", ha="center", va="center", fontsize=9,
                                          color="white" if brightness < .5 else "black")
        fig.colorbar(im, ax=axes[row_index, col], shrink=.78)
fig.suptitle("Frozen-prediction response decomposition | I800 = 4000 e | continuum panel has its own color scale", fontsize=11)
fig.savefig(OUT / "response_decomposition.png", dpi=170); plt.close(fig)

text = """# TV-RRU 开发对照完成与复核

本批执行和交付已完成。**grouped_post（TV 起点＋分组先验后处理）是本批唯一通过全部原登记开发门槛的方法。完整 TV-RRU 明显优于 WTV，但在 431.5 nm 通道图的 5% 门槛上失败，也没有超过分组后处理。**

结果范围是一个种子 811、64/16/8/16 个新母场景的合成开发试验；假设滤片、QE、PSF、载频与噪声模型均按登记设置。它支持下一轮候选判断，尚未建立三种子稳定性、独立实测精度或硬件性能。

## 主条件与原门槛

主条件：800 nm 连续谱均值 4000 e、三路各 5% 弱注入；16 个新留出母场景，每个条件两次噪声重复，共 32 幅该条件 raw。每个方法完整评价 384 幅留出 raw。六路顺序为 431.5/486/516.5/700/800/905 nm。

|方法|有效色温 RMSE/K|431.5 nm 实际注入误差/e|486 nm/e|516.5 nm/e|原开发门槛|
|---|---:|---:|---:|---:|---|
"""
for method, name in zip(methods, names):
    value = summary["summaries"][method]["4000/all_weak"]
    e = value["emission_actual_RMSE_e"]
    gate = summary["development_gates"].get(method)
    text += f"|{name}|{value['temperature_RMSE_K']:.3f}|{e[0]:.3f}|{e[1]:.3f}|{e[2]:.3f}|{'通过' if gate and gate['passed'] else '未通过' if gate else '参考'}|\n"
text += "\n![主条件与母场景区间](verified_comparison.png)\n\n"
base = summary["summaries"]["tv"]["4000/all_weak"]
grouped = summary["summaries"]["grouped_post"]["4000/all_weak"]
reduction = 100 * (1 - np.asarray(grouped["emission_actual_RMSE_e"]) / base["emission_actual_RMSE_e"])
text += f"grouped_post 的三路实际注入 RMSE 分别比 WTV 降低 {reduction[0]:.2f}%、{reduction[1]:.2f}%、{reduction[2]:.2f}%；色温 RMSE 降低 {100*(1-grouped['temperature_RMSE_K']/base['temperature_RMSE_K']):.2f}%。\n\n"
text += "原门槛按点估计判断，沿用冻结值。ordinary_post 未过六图全部 5%、图像相对 WTV 不退化 5%、blank FPR≤2% 三项；full 和 no_pair 都只在六图全部 5% 项失败，其 431.5 nm 相对 L2 分别为 5.590% 和 6.316%。grouped_post 的 431.5 nm 点估计为 4.825%，其母场景区间应与点估计一起阅读。\n\n"
text += "这里的 5% 是六幅总滤光通道图的相对 L2 门槛。主条件每路弱注入的平均量设置为 200 e；grouped_post 的三路发光 RMSE 相当于该均值的约 14.88%/15.39%/17.28%。因此开发门槛通过不表示弱发光量已达到 5% 相对精度。\n\n"
text += "## 两种发光误差及母场景区间\n\n下表为 RMSE 点估计 [逐项 95% 母场景区间]。actual 对实际 post-PSF 注入；clean 对无噪六图经过同一 Radiometry 得到的超额读出。区间来自 2000 次母场景重采样，整组携带重复、注入和方法配对，未进行多重比较校正。\n\n|方法|温度/K|431.5 actual/e|486 actual/e|516.5 actual/e|431.5 clean/e|486 clean/e|516.5 clean/e|\n|---|---|---|---|---|---|---|---|\n"
for method, name in zip(methods, names):
    p = quantities[method]["4000/all_weak"]
    t = p["temperature_MSE_K2"]
    cells = [f"{t['point']:.2f} [{t['CI95'][0]:.2f}, {t['CI95'][1]:.2f}]"]
    for key in ("emission_actual_MSE_e2", "emission_clean_readout_MSE_e2"):
        v = p[key]
        cells.extend(f"{v['point'][c]:.2f} [{v['CI95'][0][c]:.2f}, {v['CI95'][1][c]:.2f}]" for c in range(3))
    text += "|" + name + "|" + "|".join(cells) + "|\n"
text += "\n每个光子水平、条件、六图、去均值细节、均值偏差、NIR 对数比例及响应矩阵的区间均在 [SUPPLEMENTAL_STATISTICS.json](closeout_audit_001/SUPPLEMENTAL_STATISTICS.json)。主条件与 TV 的配对 RMSE 比区间，以及不同神经方法之间的配对比区间，见该文件和 [PAIRED_METHOD_COMPARISONS.json](PAIRED_METHOD_COMPARISONS.json)。\n\n"
text += "## 六幅图、细节与均值\n\n每格为六路的百分数，顺序见上。图误差为逐帧相对 L2 后取均值；细节误差为去均值误差相对去均值真值，分母另含登记的 1% 辐射范数地板；均值列为有符号相对偏差。\n\n|方法|六路图相对 L2/%|六路去均值细节相对误差/%|六路均值偏差/%|\n|---|---|---|---|\n"
for method, name in zip(methods, names):
    p = summary["summaries"][method]["4000/all_weak"]
    cells = [" / ".join(f"{100*x:.3f}" for x in p[key]) for key in ("band_relative_L2_mean", "band_centered_relative_L2_mean", "band_mean_bias_fraction_mean")]
    text += "|" + name + "|" + "|".join(cells) + "|\n"
text += "\nfull 的 NIR 单图误差低于 grouped_post，但两项 NIR 对数比 RMSE 为 0.00822/0.00829，grouped_post 为 0.00413/0.00335。这与后者色温更准确一致，说明当前差异需要从比例误差分析；本结果尚未确定差异具体来自近端强度、有限迭代还是训练后的 proposal。\n\n"
text += "## 完整单波段响应与误报\n\n所有矩阵均为行=读出、列=注入，沿用实际 post-PSF 注入图作内积斜率分母；逐对计算，再按重复和母场景平均。\n\n"
for level in (1000, 4000):
    text += f"### {level} e 光强\n\n|方法|431.5 读出行|486 读出行|516.5 读出行|\n|---|---|---|---|\n"
    for method, name in zip(methods, names):
        matrix = summary["summaries"][method][f"{level}/single_injection_response_matrix"]
        text += "|" + name + "|" + "|".join(" / ".join(f"{x:+.4f}" for x in row) for row in matrix) + "|\n"
text += "\n空白误报与非目标注入误报分开列出。后者为三个单波段注入条件中，两个未注入读出通道的最大像素检出率。各方法均使用独立 calibration 母场景预先冻结的阈值；未在留出集重调阈值。\n\n|方法|1000 e blank 最大 FPR/%|4000 e blank 最大 FPR/%|1000 e 其他波段注入最大 FPR/%|4000 e 其他波段注入最大 FPR/%|\n|---|---:|---:|---:|---:|\n"
for method, name in zip(methods, names):
    p = summary["summaries"][method]
    blank = [100 * max(p[f"{level}/blank"]["detected_fraction_mean"]) for level in (1000, 4000)]
    other = [100 * max(p[f"{level}/{condition}"]["detected_fraction_mean"][k] for j, condition in enumerate(("CH_weak", "AlO_weak", "C2_weak")) for k in range(3) if k != j) for level in (1000, 4000)]
    text += "|" + name + "|" + "|".join(f"{v:.3f}" for v in (*blank, *other)) + "|\n"
text += "\n相应逐通道区间和相对 blank 的配对变化见 supplemental statistics；非目标注入误报保留为独立报告项，没有追认一个新的通过门槛。\n\n![响应的两条路径](response_decomposition.png)\n\nfull 中 486 nm 注入到 516.5 nm 读出的斜率为 +0.00882，可见图响应约 +0.00622，连续谱响应约 −0.00260，两者相减得到最终值。两条路径都存在；分组先验隔离没有移除测量与读出耦合。图中连续谱面板使用单独的颜色尺度，逐母场景区间均保留在 JSON 中。\n\n"
text += "配对损失在本批改善了实际注入误差：full 相对 no_pair 三路 RMSE 降低约 11.57%/12.49%/7.72%，但非对角响应并未全面降低。因此本批支持其幅值误差收益，尚不支持将串扰改善全部归因于该项。ordinary/grouped 比较还同时改变了坐标、输入特征和修正参数化，属于整套设计对照。\n\n"
text += "## 低光强与误差解释\n\n|方法|1000 e all-weak 色温 RMSE/K|三路实际注入 RMSE/e|\n|---|---:|---|\n"
for method, name in zip(methods, names):
    p = summary["summaries"][method]["1000/all_weak"]
    text += f"|{name}|{p['temperature_RMSE_K']:.3f}|" + " / ".join(f"{x:.3f}" for x in p["emission_actual_RMSE_e"]) + "|\n"
text += "\n低光强下四种神经方法的上述点估计均较 WTV 改善；50 K/5% 等绝对指标按登记的主条件验收，不能把这一开发通过扩展到全部条件。`algorithm_mixing_cross_e2` 保留了实际注入 MSE、无噪读出 MSE和连续谱混合偏差之间的交叉项，未假设它们独立。三路无噪混合 RMSE 在主条件约为 2.65/3.26/3.28 e，算法误差仍占主要量级。\n\n"
text += "## 复核、时延和交付\n\n原程序于 2026-10-18 03:58:58（UTC+8）以退出码 0 完成。原生复核于 04:12:44 完成：25 个源码文件、104 个缓存文件、原输出清单以及执行归档 SHA256 全部匹配；四组各 6000 步，日程均为 1500/1500/3000。验证权重先冻结，随后 calibration 阈值冻结，再进入新留出评价。ordinary_post 完整复用，旧 grouped_post 4182 步中断记录和全部数值测试失败均保留。\n\n"
text += f"六种方法各重新打开并原生重放 384 幅预测，最大相对差为 {max(v['maximum_native_replay_relative_L2'] for v in audit['native'].values()):.1f}；报告指标与标定阈值本次均完全复现。该检查针对冻结权重推理，未重放优化器训练轨迹。完整复核用时 {audit['wall_seconds']:.2f} s，没有新训练或目标/阈值修改。\n\n"
text += "|方法|含 TV 初始化的单 raw→读出软件时延/s|TV 轮数|\n|---|---:|---:|\n"
for method, name in zip(methods, names):
    text += f"|{name}|{timing[method]['seconds_one_raw_to_readout_including_TV']:.3f}|{timing[method]['TV_iterations']}|\n"
text += "\n计时是 RTX 5070 上登记母场景 81001、第 20 帧、batch=1 的单次软件测量，含电子转换、TV、网络和读出。一次计时不足以区分相近方法的速度差；它不包含真实设备采集、数据传输，也不表示相机采集帧率。缓存后额外推理计时另见原 METRICS；主运行峰值 PyTorch 分配显存约 3708 MiB。\n\n"
text += "- [原结果与图](server_results/RESULTS_zh.md)、[原门槛和全条件摘要](server_results/SUMMARY.json)、[原母场景 bootstrap](server_results/MOTHER_BOOTSTRAP.json)。\n- [原生复核收据](closeout_audit_001/AUDIT_COMPLETION.json)、[源码/缓存/训练日程复核](closeout_audit_001/PROVENANCE_AUDIT.json)、[本地交付哈希](closeout_audit_001/LOCAL_DELIVERY_VERIFIED.json)。\n- [附加统计](closeout_audit_001/SUPPLEMENTAL_STATISTICS.json)、[配对方法比较](PAIRED_METHOD_COMPARISONS.json)、[审计源码](closeout_audit_001/audit.py)。\n- 成功权重、末次权重及可恢复的 latest 状态位于 `server_results/`；实际执行源码保存在原 `source_snapshot/`。重放时将 audit.py 放到服务器执行根目录的 `reports/joint_imaging_v2_closeout_audit_001/` 中运行；不覆盖已有审计输出。\n\n本地仅下载源码、报告、PNG、日志与权重。10,715,430,944 字节 TV/raw 缓存和六份大预测数组保留在服务器；逐文件 SHA256 见交付清单。stderr 的 Matplotlib Axes3D 导入警告原样保留，所需二维 PNG 已生成并检查。\n\n"
text += "## 下一步判断\n\n当前将 grouped_post 保留为最有支持的开发候选，同时保留 full 的改善和失败。先在固定验证小集分析 proposal 到物理更新后的幅度与比例变化，核对加倍 ADMM/PCG 后输出是否稳定，再决定是否调整近端；该诊断尚未执行。若要声称分组后处理收益稳定，再预登记另外两个种子，并报告全部种子。当前 16 个留出母体已公开，后续模型改动后的科学结论须用新的未见母体。256/32/32/64 扩展、三种子正式验收和真实标定试验均需另登记；本次不自动开始扩展训练。\n"
(OUT / "FINAL_REVIEW_zh.md").write_bytes((text + "\n").encode("utf-8"))
print(json.dumps({"report": str(OUT / "FINAL_REVIEW_zh.md"), "paired_comparisons": comparisons}, ensure_ascii=False))
