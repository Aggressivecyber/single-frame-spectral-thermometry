# Single-frame spectral thermometry — v5 决策性研究实施任务书

**审阅基线：**GitHub `Aggressivecyber/single-frame-spectral-thermometry`，提交 `ddc4223fd1ecb9c5af82a8d1c489bf9cc54fd2d2`（2026-10-13）。

## 总决策

停止 v4.x 的损失权重、采样比例、500/1500/2000 次更新、网络模块重复微调。此前 v4.4/v4.5/v4.6 均无同一权重通过全部开发目标的候选：v4.4 主弱 L2 0.671、主正常 RMSE 9.57e、负正常 RMSE 52.55e；v4.6 主弱 L2 1.061、主正常 RMSE 13.47e、负正常 RMSE 35.36e；物理负正常参考为约 20.42e。v4.3 的字段分解已发现温度、模板同时退化，raw 重投影残差仍接近 1 的情况。下一步必须诊断信息与参数串扰，而不是依赖 raw 一致性掩盖结构错误。

**研究主目标：**已知标定、固定视场和规定光学带宽下，从一张七通道频域编码 DN 图得到可靠的“模糊后连续谱色温”，并输出某个**合成谱模板的可检测偏离量及其不确定度**。物理正发射模板与负系数的数值诊断分开报告；负系数不直接称负浓度，合成 AlO 不称实测物种分布。当前无真实标定、真实多波段同步谱和真实温度，不做真实气体温度/10kHz吞吐断言。

## 实施原则

- 代码另起 `research_v5/`；旧 checkpoint、v3/4.1-4.6 代码、冻结原始数组及原报告均保持不变。新报告 `reports/research_v5/`。
- 新程序推理只能接收 raw DN、经过外部确定的标定与选择参数；不能访问任何 truth、condition、clean_identity、输入 ID 或人工 oracle 标签。
- 不得将同一 raw 解码结果再次视为独立测量重复乘入目标；正式数据一致性只用 raw 域。
- 所有指标以模糊后域为主，另外报告潜在真实温度（若仿真可得）与 observed/reference 色温，不混名词。
- 已查看的 776 帧只用作开发回归；新验证/确认数据应登记新母场景，按母场景聚合/Bootstrap，不按像素假独立。此前登记但未开启的 36 个 signed 确认母场景和 64 个最终母场景保持封存，直到模型、阈值和统计计划预冻结。
- 每个新研究阶段以实际科学输出为交付；训练通过不等于科学验收通过。失败结果保留。

## 阶段 0：冻结选择和评价协议（不训练）

在 `research_v5/protocol.json` 保存 head SHA、旧权重哈希、physics/config/hash、FFT carrier、PSF、噪声/ADC、标定来源、场景种子、控制预算和新评价规则。先确认原平台所有文件只读，并保留 4 种基线：`B1_tikhonov`（当前验证选参）、`v3_physics`、`old_B4`、`v4.4_final_continuum_fixed`（后者必须明确内部 C 仍变化）。

主性能分开汇总：色温 RMSE/bias、模板正/负 normal RMSE/pooled L2/投影系数、±weak 回归与零估计 L2=1、null pixel/frame FP、测量域残差、有效覆盖、求解时延、CG 相对真残差。训练/验证/确认集明确不重叠，已查看分布不作为最终确认。

## 阶段 1：一次性物理信息判定（不训练，最优先）

**构造:** 从开发数据中取平坦、旋转前沿、多峰、细丝、环形等 12 个母场景，固定各自 T/a/背景/几何，分别注入 `S=0, +45e, -45e, +180e, -180e` 的符号模板，并生成三个输入形态：连续期望、仅量化、Poisson+读出噪声+量化。共约 180 个输入，先小集核验后固定；不可把同一 RNG seed 的不同 Poisson 均值说成“完全相同噪声”。核查预期电子数非负及 ADC 是否饱和。真值只进诊断分支。

**四级 oracle 梯子（同一个 raw、相同求解器）:**

- O0 `true continuum + true background`，只重建 signed 模板 S。
- O1 `estimated continuum + true background`，重建 S。
- O2 `true continuum + estimated structured background`，重建 S。
- O3 `estimated continuum + estimated structured background`，这是正式可部署路径，不能读真值。

O0/O1/O2 明确标记为仅诊断用 oracle，绝不放进实际推理模型或正式准确率；对每组保存 postblur 模板误差、投影、符号召回、raw 重投影、最佳迭代与收敛状态。

**数值与 Fisher 信息:** 对当前 forward 对 T、loga、S、beta 求 JVP/VJP，使用同样的 raw 噪声方差 W。核验内积 `<Jv,w>=<v,J^T w>`；对代表性空间低频基和 ROI 模式计算噪声加权的 S 导数在 T/loga/beta nuisance 子空间正交投影后的模长、最小奇异值、条件数与可检测信噪比。必要时用随机低维探针近似，不要伪称对全部 256×256 状态得到了精确 CRLB。

**关键判断与停机:**

- O0 在无噪/名义条件仍不能准确恢复模糊后正常强度的 ±S：先排查 forward/adjoint、频带与零空间；不要训练网络。若信息在设定编码中根本不存在，调整滤波/载频/光子预算的仿真设计而不是换网络。
- O0 可恢复、O1 明显变坏：温度/幅度估计串扰为主；先修 700/905 连续谱估计和域一致性。
- O0 可恢复、O2 明显变坏：背景 q 与 M 模板串扰为主；限制背景空间/光谱自由度与标定。
- O0/O1/O2 均可、O3 才变坏：估计误差联合传播，需要明确误差传播与不确定度，不盲目训练。
- O0 噪声下失败且 Fisher 可检测 SNR 很低：弱信号属于当前光子预算的不可可靠区域，转向 detection limit 与 abstention，不强迫每幅图 L2<1。

输出 `ORACLE_LADDER.csv`, `FISHER_IDENTIFIABILITY.md`, `OPTICAL_NULLSPACE_TESTS.md`, `DECISION_1.md`。

## 阶段 2：实现“两步物理+线性模板”的正式估计器（不训练）

**第一步：估计连续谱。** 保留真实编码算子、object_blur、band-integrated Planck LUT。用 700/905 两个名义上无合成模板/背景的通道初始化并稳健拟合连续谱 T、loga，采用空间域一致的约束优化，避免高频 OTF 直接除零；输出 `continuum_latent`, `continuum_postblur`, `temperature_color_K`, `temperature_uncertainty`, `valid`。未知发射率与未知增益是独立误差来源，绝不能把色温直接称真实气体温度。调试时可与现有 v3 物理连续谱直接对照。

**第二步：给定已经固定的连续谱，线性联合求 S 和 beta。** 不得再让 CNN 或模板子问题反向改 T/loga：

设 `A_S(S)=op(M_active*S)`，`A_b(beta)=op(q*B(beta))`，`y_c=op(a*C(T))`，统一都使用真实 `CodedOperator` 包含的 object blur 和编码（勿重复 blur），其中 op 表示已有从七波段辐射到 raw 的完整算子。

解以下凸二次问题（W 是固定、正的 raw 域噪声逆方差）：

`min_(S,beta) 0.5*||W^(1/2) [y-y_c-A_S S-A_b beta]||² + (lambda_g/2)||D S||² + (lambda_0/2)||S||² + (lambda_b/2)||beta||²`

- 对 signed 数值诊断：不加 S>=0，使用矩阵自由 PCG，输出 postblur `H S`。
- 对明确仅有物理发射的任务：可另做约束 S>=0 的 FISTA/投影方法，绝不能拿它评估负模板并声称“恢复正确”。
- 联合求 beta 的 3 个背景系数，不能把真实 beta 带进部署；另用 oracle 单独比较影响。
- 噪声权重来自 raw 及仪器参数，可迭代更新但每次求解的目标与接受条件必须固定、可追踪。
- 在开发验证母场景上只做一个小的归一化正则网格：`alpha_g` in `[0,0.01,0.1,1]`，使 `lambda_g=alpha_g*median(diag(A_S^T W A_S))`（实现时统一状态/空间尺度）；独立记录偏差—方差与梯度能量，不重复宽泛搜索。训练/确认不能参与选参。
- CG 相对**真实**残差目标 1e-4（可在数值诊断中用 float64 核验）、最大 200 次；上限/未收敛、接受与越界均留痕。线性子问题可以先使用 scipy 小尺寸稠密 `least_squares` 作数值参考，再用矩阵自由真实分辨率。

**不确定度与拒绝输出:** 用一组固定参数的 Monte Carlo 噪声重复 + 线性 Hessian 诊断估计 `sigma_S` 与色温置信范围；报告 nominal 条件下的经验覆盖。对于低 SNR 区域可输出 `undetectable/low-confidence` 与区间，而不是必须给确定正/负模板图。检测阈值由独立 null 验证集标定，并保留每帧 FP 与真值支持区的 recall。

输出 `continuum_solver.py`, `signed_linear_solver.py`, `reconstruct.py`, `uncertainty.py`, `tests_adjoint.py`, `PHYSICS_ONLY_V5_REPORT.md`。第一次交付不需要新 CNN。

## 阶段 3：一次**有条件**的小型神经对照（只有阶段 2 通过再启动）

不再用六个 CNN 阶段同时更改 T、loga、S。固定 T、loga、beta（或把 beta 只交给线性求解），CNN 只作用在第二步的 `S_linear`，不能改变连续谱或直接覆盖物理测量预测。

第一候选仅为**保持符号的有界增益去噪器**，不再学习任意绝对加法偏置：

`S_out = S_linear * (1 + 0.3*tanh(g_theta(z)))`

其中 `z` 只包含从测量与固定物理估计可计算的 `|S_linear|/sigma_S`、标准化残差能量/局部梯度幅值、噪声预算等特征；不包含 truth、sign 标签、条件元数据或 case_id。最后一层初始化为零，保证初始恒等、保证 S_linear=0 时不自发凭空生成模板、保证逐像素 S_out 与 S_linear 符号一致。可作为受控小网络，不承诺能生成测量中缺失的结构。若线性 S 的符号本身不可靠，先修物理与可靠性拒绝机制。

这与 v4.4 “只固定最后 C、内部 C 仍变化”不同：新模块整个过程中**没有**连续谱可训练状态或连续谱反馈。

训练数据需要同母场景 `+S/-S/0` 对照以及 6×6 几何覆盖，使用新的训练母场景（不要只反复使用原 12/36 个）；验证与训练按母场景分离。预登记 1 个模型、1 个学习率/预算、最多 1000 次真实 optimizer 更新、0/200/500/1000 权重评估。初始从恒等缩放开始，不复用 v4 update2000 的偏置头，不增加 U-Net/注意力，不训练整段 v3 LM/PCG。

只有同一个权重在有噪声正信号及负数值诊断均不逊于阶段 2，并在主要可检测弱信号中带来有意义改进，才保留此 CNN；否则最终方案直接以 **纯物理 + 不确定度** 收口，不再加 v5.1、v5.2 的连续微调。

输出 `signed_gain_denoiser.py`, `TRAIN_V5_REPORT.md`, `checkpoint.pt`、逐 checkpoint 主/零/正/负/弱对照和失败记录。

## 阶段 4：预登记、确认与终止条件

正式验收应先声明应用任务：

A. **主任务（颜色测温）**：已知标定下 postblur continuum color T；建议开发目标名义噪声 RMSE ≤20 K（当前物理基线在相近设置约 17~18 K），但不能将此值转述为真实气体绝对温度误差。未知增益、发射率失配分别报告失效和拒绝范围，不可混为名义 RMSE。

B. **次任务（模板偏离）**：正常 ±S 的 RMSE 和幅度投影必须分别优于/非劣于验证选好的最佳传统物理方法（开发容差可预登记为 RMSE 不高于基线 1.05 倍）；同时报告空白误报/有符号负召回。正模板是构造的发射增量；负模板是数值可逆性和模型偏置压力测试，不是负浓度。

C. **低于检出限的弱信号**：不要求所有 weak L2<1；先使用 Fisher 和 empirical null 给出 detection limit。若信噪比不足，必须输出低可信/不检出及覆盖率；对超过预注册检出限的子集再比较 L2、召回和误检。

D. **统计**：多母场景、按母场景配对；输出 pooled 指标、逐母体分布、母场景 bootstrap 95% 区间。旧16/96条成为工程回归记录而非“每一格零退化”的普适科学定理。训练与验证数据分布有重叠观测支持，任何单帧算法不可能普遍保证既对所有无噪声场景严格恒等、又在所有有噪声场景降低真值误差。

E. **最终集**：确定一个版本、配置、阈值及所有指标后再开启已有的 64 母体最终集；signed 需要额外预冻结并开启 signed 确认母场景。只评估一次，不对最终集调阈值或改模型，失败照实报告。

**严格停止：**

1. O0 noiseless 不成立 → **停止机器学习，转优化/修复光学算子与成像设计**。
2. O0 成立而 O1/O2 崩溃 → **停止机器学习，转温度标定或背景可辨识性**。
3. O3 的 signed 正常信号不能达到与旧物理参考同一量级 → **先把部署物理估计做到正确，不开始 CNN**。
4. 可辨识、部署物理可用、轻量网络在一次有限预算内仍不能兼顾正负 → **终止神经网络支线，发布负结果与纯物理基线**。
5. 全部通过 → 冻结候选、开启一次独立最终验证；不以代码运行成功替代物理科学验收。

## 交付和执行顺序（实现这些 CLI 后再运行）

```bash
python -m research_v5.run --phase register
python -m research_v5.run --phase oracle
python -m research_v5.run --phase identifiability
python -m research_v5.run --phase linear-solver
python -m research_v5.run --phase development-compare
# 上述通过门禁后，才允许可选训练：
python -m research_v5.run --phase train-denoiser
python -m research_v5.run --phase candidate-review
# 候选及协议真正冻结后才运行，绝不可提前：
python -m research_v5.run --phase final-once
```

这些是**拟新增 CLI 名称**，当前仓库尚不存在；不要将它们当作立即可执行的旧脚本。

最终 `V5_SCIENTIFIC_DECISION.md` 必须用一页回答：物理在无噪声是否可逆？在当前光子预算下 +/-S 是否可辨？误差主因是 T/a、beta、还是算法先验？是否有必要使用 CNN？哪一类输入必须拒绝？当前结论能否支持真实测温/论文？
