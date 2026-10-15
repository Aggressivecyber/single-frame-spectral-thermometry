**数值根因审阅：联合 profile 的 9/9 淘汰与 frozen v3 的 0/60 认证通过**

结论：联合方法的直接淘汰原因是基础外层停止精度未达到注册的输出稳定性预算；v3 的直接问题是有限 CG 配置对输出有可量化影响。两初值采用不同连续谱正则系数，是独立确认的目标函数一致性缺陷。现有证据支持这些数值结论；9 个组合只在同一已打开正模板正常强度 raw 上各执行两个初值，尚未进入完整科学性能筛选。

审阅日期：2026-10-15。证据逐项保存在 [numerical_evidence.json](E:/CodexWork/thermometry/closeout_001/reports/closeout_001/root_cause_team_001/numerical_evidence.json)。此次新增完整反演 0 次、GPU 作业 0 个、封存 payload 访问 0 次。独立复核只读已有 raw、v3 状态及 18 个已保存参数；没有读取诊断真值，也没有修改注册源码、原输入、权重或旧报告。核心 v3 源码及 backbone_selection 与原仓库逐文件 SHA256 相同。

**1. 已证实：base 的缩放梯度停止准则与输出预算不匹配，造成 17/18 次未认证。**

18 次 base 全部以 `NORM OF PROJECTED GRADIENT <= PGTOL` 成功停止；没有 base 达到 600 次迭代上限，也没有超时。总耗时 32.225–91.502 s，均低于 180 s。紧参考比较有 17/18 温度差超过 0.2 K、15/18 模板差超过 0.17 e。仅 `joint_c1_s0.1 / v3` 同时通过；其另一个初值失败，按预注册双初值规则淘汰整个组合。因而 0/9 来自这个明确规则，不能扩写成九个方法科学性能均失败。规则位于 [development.py:169](E:/CodexWork/thermometry/closeout_001/research_closeout/development.py:169)，资格条件位于 [numerical_strategy.py:82](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:82)。

独立重算 18 份保存的生产状态：缩放后的投影梯度无穷范数为 6.92902e-4–9.91059e-4，确实低于 base `gtol=1e-3`；所有变量均未触及边界。因此，这批停止不是边界截断或日志误记。重算的未缩放梯度 L2 范数与保存记录最大相差约 3.11e-10。v3 初值的两个通道坐标尺度为 0.00949683、0.01788691；`∇v f = scale × ∇z f`，该坐标变换保留目标函数，但 `gtol=1e-3` 本身没有转换成 K/e 输出误差保证。代码位于 [numerical_strategy.py:25](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:25) 和 [numerical_strategy.py:53](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:53)。SciPy 将 gtol 定义为投影梯度阈值，ftol 定义为目标函数相对变化阈值；两者都不是输出误差阈值。[SciPy L-BFGS-B 官方文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

紧参考的 6 个 `continuum_multiplier=0.1` 运行达到 300 次上限；其余 12 个按 ftol 成功停止，其中仍有 11 个输出比较失败。代码最终返回 base，reference 只用于认证，并不替换输出：[numerical_strategy.py:59](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:59)、[numerical_strategy.py:70](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:70)、[numerical_strategy.py:95](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:95)。

替代解释：6 个紧参考尚未成功，不能用其输出宣布非线性最优解。其余 12 个的 ftol 成功也不能单独保证全局最优。上述 17 个已观察到的输出移动，已经足以证伪“base 输出在注册预算内稳定”。最小可证伪实验：后续如注册独立数值诊断，在这一份已打开 raw 上固定同一目标，从保存的 base 状态续算，以相邻两次更紧求解的输出变化检验 0.2 K/0.17 e；不重跑九组合科学矩阵、不追改历史资格。

**2. 已证实：两初值改变了正则目标，不能据它们的终值差直接认定局部极小值或初值敏感性。**

[joint_estimator.py:43](E:/CodexWork/thermometry/closeout_001/research_closeout/joint_estimator.py:43) 用各自 `z0` 的连续谱 Jacobian 均方计算 `regscale=0.01×continuum_multiplier×scale`；[joint_estimator.py:50](E:/CodexWork/thermometry/closeout_001/research_closeout/joint_estimator.py:50) 将其直接乘入连续谱平滑惩罚。这是物理目标中的权重，不只是优化坐标。

在唯一预筛 raw `v5_development_selection_mother_0000_positive_normal_noise0` 上，使用哈希验证的原 v3 缓存和由同一 raw 计算的 smooth_demod 初值，独立得到：

| 连续谱通道 | v3 的 regscale | smooth_demod 的 regscale | 后者相对前者 |
|---|---:|---:|---:|
| 归一化温度 | 110.87734958 | 107.78891060 | -2.78546% |
| loga | 31.25572492 | 30.69270084 | -1.80135% |

固定 `joint_c1_s1 / v3` 已存生产状态与同一线性参数，W 和线性正则完全相同，数据项加线性先验为 31540.9858589643。两个连续谱正则分别为 153.6109989280、150.2430063672，故同一点目标函数相差 **3.3679925608**。这个无反演代数实验直接证明“两初值同目标”不成立。[numerical_strategy.py:25](E:/CodexWork/thermometry/closeout_001/research_closeout/numerical_strategy.py:25) 在坐标缩放中消掉 continuum multiplier，没有消掉 initializer 依赖。

替代解释：把它明确定义成 initializer 自适应正则估计器，算法仍可执行；但这时两条路线同时改变初值和目标，失去多初值同目标检验的意义。这一缺陷首先污染因果解释，尚无证据表明其 1.8%–2.8% 权重差是 17 次未认证的唯一或主要原因。最小可证伪实验：同一保存状态用共享的冻结 regscale 计算目标，应严格相同；再用一个已打开 raw、同一共享目标比较两初值的终值，才能讨论局部极小值。当前只完成前述反例复核。

**3. 已证实：v3 对有限 CG 配置敏感；float32 不是足够完整的解释。**

60 个原 float32 输出与两个 float64 路线比较，均保留 12 次外层迭代。注册定义在 [development.py:205](E:/CodexWork/thermometry/closeout_001/research_closeout/development.py:205)，原配置在 [backbone_selection.json](E:/CodexWork/thermometry/closeout_001/reports/training_v4/run_001/backbone_selection.json)。

| 比较路线 | 模板差中位 / 最大（e） | 颜色差中位 / 最大（K） | 同时达预算 | 最后一次 CG 达容限 |
|---|---:|---:|---:|---:|
| 同有限算法 float64，CG 120 / 1e-4 | 0.251498 / 1.343694 | 0.286409 / 1.691785 | 18/60 | 0/60 |
| float64，CG 600 / 1e-8 | 0.647292 / 2.316713 | 1.065979 / 3.911129 | 0/60 | 2/60 |

第一路线最后一次 CG 全部达到 120 次上限，真实相对残差中位 0.00661805、最大 0.0539703；第二路线 58/60 达 600 次上限，最后一次真实相对残差中位 0.000430835、最大 0.00195316。这里的计数只针对各反演最后一次 CG，不能扩写成所有内部 CG 的统计。完整逐帧证据在 [v3_production_numerics.jsonl](E:/CodexWork/thermometry/closeout_001/reports/closeout_001/v3_production_numerics.jsonl:1)。

进一步用反三角不等式作独立复核。令 A、B 为两个 float64 输出，F 为原 float32 输出，则 `RMS(A-B) ≥ |RMS(A-F)-RMS(B-F)|`。60/60 在至少一个输出上，仅这个下界就超过预算：模板 56/60 超过 0.17 e，颜色 58/60 超过 0.2 K；下界中位数分别 0.425268 e、0.716107 K。两个被比较路线都使用 float64 和相同 12 次外层算法，其唯一区别是 CG 配置，所以 CG 配置已经足以引发超预算输出变化。这个结论无需新增反演，也不依赖把某个参考宣称为真解。

代码机制清楚：[repair_v3/solver.py:161](E:/CodexWork/thermometry/closeout_001/repair_v3/solver.py:161) 使用有限 PCG 方向，[repair_v3/solver.py:172](E:/CodexWork/thermometry/closeout_001/repair_v3/solver.py:172) 按下降和 rho 接受方向，没有要求 `cg.converged`；[training_v4/scalar_engine.py:59](E:/CodexWork/thermometry/closeout_001/training_v4/scalar_engine.py:59) 在 12 步或拒绝时终止。这是有限迭代估计器，CG 预算参与定义最终输出。float64 改变初始化算术、PCG 及 LM 路径；提高 CG 精度又会改变 LM 路径。58 个紧参考末次 CG 仍不达容限，因而它也不是高精度非线性最优参考。

替代解释：这能证明有限算法敏感性，不能把全部输出差分配成“舍入误差”“内层截断误差”或“距离全局最优解误差”。最小可证伪实验：固定一份现有状态、权重、阻尼和线性化，分别改变 dtype 与 CG 预算，先比较同一线性系统的真实残差及输出方向；只有这一步可分离后，再考虑整条 12 步轨迹。当前分析没有启动这些求解。

**4. 已证实的排除证据：endpoint 内层精度不是本次大输出差的主要直接来源；完整梯度认证仍有明确覆盖范围。**

18 次生产 endpoint 的 scaled normal relative 为 1.74464e-12–7.08553e-12，进一步收紧内层后模板差为 1.36136e-10–7.49319e-10 e，远低于 0.17 e。与之相比，外层模板移动为 0.143368–1.106158 e，颜色移动为 0.182126–0.896084 K。内层结果与外层误差相差约九个数量级，日志支持优先诊断外层停止和坐标几何。

[joint_estimator.py:62](E:/CodexWork/thermometry/closeout_001/research_closeout/joint_estimator.py:62) 对线性解 detach，再对完整目标的连续谱部分求导；在线性驻点，关于线性解的链式项为零，符合 profile 导数公式。此次重算生产梯度与记录一致，验证了保存状态的梯度重建。它没有替代全维 profile 函数的方向有限差分。

已有 `native256_production_profile_gradient` 测试实际使用 256×256 **观测网格**、4×4 非线性网格、25 个线性 Fourier 模式；whole-recovered fixture 也是已知初值的常量场。精确范围见 [tests/contracts.py:79](E:/CodexWork/thermometry/closeout_001/research_closeout/tests/contracts.py:79)、[tests/development.py:17](E:/CodexWork/thermometry/closeout_001/research_closeout/tests/development.py:17)。它们检查了实现公式，不能被称为这 18 个 native256 全自由度运行的全局收敛证明。LSMR 的残差停止也不能单独转成参数误差界；这里更有力的是额外的模板输出比较。[SciPy LSMR 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)

替代解释：前期内层容限按调用次数由 1e-7、1e-9 变到 1e-11，并使用 warm start，可能影响早期准牛顿路径；这尚未被本次 endpoint 证据排除。最小可证伪实验：在同一保存外层状态、全 1153 模式上固定内层容限，检查少量预定方向的 profile 有限差分，再改变内层容限；若误差随内层容限缩小并远小于外层输出预算，进一步降低内层容限没有优先性。当前报告不把未测到的梯度问题写成既定根因。

**5. 强证据推断：native256 的大量潜在自由度和未处理的通道耦合增加外层求解难度；尚未测得全 profile Hessian 条件数。**

native256 有 131072 个连续谱参数、1153 个模板实 Fourier 系数和 3 个背景系数，而 raw 有 65536 像素。数据 Jacobian 因维数必然不能满列秩。进一步按当前算子精确计数，每个模糊后波段仅有 1153 个非零实 Fourier 自由度；全部 7 个波段的映射维数上界为 8071，故联合未正则数据 Jacobian 的代数秩上界为 8071，零空间维数下界为 124157。这是线性化结构上界，不是实测奇异值谱。算子与基定义见 [physics.py:19](E:/CodexWork/thermometry/closeout_001/physics.py:19)、[physics.py:54](E:/CodexWork/thermometry/closeout_001/physics.py:54)、[observable_basis.py:8](E:/CodexWork/thermometry/closeout_001/research_closeout/observable_basis.py:8)。正则可以约束这些方向，光学输出也可以比潜在场更稳定，故该维数事实不能推出目标颜色温度整体不可辨。

恢复策略仅使用两个固定全图通道尺度，未消去 T/loga 的非对角耦合或空间频率刚度。在 `joint_c1_s1 / v3` 生产状态，独立计算未模糊局部光谱 Gram 的两通道相关系数，中位 0.982364、范围 0.980860–0.987594。它说明存在显著局部耦合，不能当作完整 profiled Hessian 的条件数。

native256 的选择来自把已存 v3 潜在状态双线性降采样再上采样的保真测试：[evaluate.py:32](E:/CodexWork/thermometry/closeout_001/research_closeout/evaluate.py:32)。32/64/128 各 0/60 达预算；128 网格颜色差 0.666856–3.82119 K、raw 差 3.92465–16.4387 e。这个测试证实该直接重采样操作不能保持原 v3 输出；它没有检验“在较低维表示中重新优化后的最佳可达到输出”，因此也不能证明 131072 个潜在自由度都是数据所需。

替代解释：外层缓慢还可能受 5 K 分段线性 LUT 的导数拐点、profile 曲率及非线性路径影响，当前没有各项消融。最小可证伪实验：固定当前已保存状态和同一目标，在少量预定模式上测 Jacobian/正则曲率或做有界 Hessian-vector 诊断，区分通道耦合、空间高频先验和可观测输出方向；不据此直接启动新架构或新超参数矩阵。

**9 组预筛的完整数值去向**

下表每格均按“v3 / smooth_demod”顺序。base 均为 PGTOL 成功；紧参考 `L300` 表示 300 次上限失败，`F` 表示 ftol 成功。Δ 为 base 与紧参考输出的 RMS 差。原始行位于 [development_results.jsonl:301](E:/CodexWork/thermometry/closeout_001/reports/closeout_001/development_results.jsonl:301) 至 318；完整 inner 信息与哈希在随附 JSON。外层 row `status=SUCCESS` 表示有可评分有限输出，数值资格应读取 `production_numerical_certified`，二者由 [metrics.py:23](E:/CodexWork/thermometry/closeout_001/research_closeout/metrics.py:23) 分别保留。

| c / s | base 次数 | 紧参考停止与次数 | Δ模板 e | Δ颜色 K | 单初值认证 |
|---|---:|---|---:|---:|---|
| 0.1 / 0.1 | 156 / 99 | L300 / L300 | 0.432617 / 1.106158 | 0.337107 / 0.896084 | 否 / 否 |
| 0.1 / 1 | 156 / 100 | L300 / L300 | 0.408951 / 0.998923 | 0.341657 / 0.860466 | 否 / 否 |
| 0.1 / 10 | 161 / 97 | L300 / L300 | 0.220956 / 0.741303 | 0.259457 / 0.828644 | 否 / 否 |
| 1 / 0.1 | 109 / 82 | F160 / F147 | 0.161965 / 0.531029 | 0.182126 / 0.401133 | 是 / 否 |
| 1 / 1 | 103 / 87 | F161 / F149 | 0.209742 / 0.378120 | 0.240946 / 0.332771 | 否 / 否 |
| 1 / 10 | 103 / 85 | F163 / F151 | 0.161465 / 0.303642 | 0.214228 / 0.308312 | 否 / 否 |
| 10 / 0.1 | 102 / 103 | F155 / F156 | 0.179028 / 0.246152 | 0.533786 / 0.373169 | 否 / 否 |
| 10 / 1 | 101 / 111 | F158 / F153 | 0.184027 / 0.192085 | 0.541502 / 0.257631 | 否 / 否 |
| 10 / 10 | 103 / 99 | F151 / F158 | 0.143368 / 0.265822 | 0.496780 / 0.460519 | 否 / 否 |

当前根因分析已完成。若后续需要一个最小诊断次序，应先固定共同目标并检查保存状态的全模式 profile 导数，再在同一 raw 上检验输出导向的外层停止，最后在同一冻结线性化上拆分 v3 的 dtype 与 CG 影响。上述建议属于待注册诊断，未执行；现有 9 组淘汰、0/60 生产资格和所有旧负结果均保持原义。封存测试不参与这条诊断链。
