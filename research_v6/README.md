# v6：连续谱来源置换与条件联合反演

固定基线为 `431777df6c1d83fdae55fc54c30776e16ab74706`，工作分支为 `research/v6-continuum-profile`。原 v5 源码、权重、协议、报告和 raw 只读；本轮新文件限定在 E/D 的 v6 目录。

先核验已存在的 720 个 raw，登记源码与输入哈希，然后进行一次阶段 A：真连续谱、冻结 v5 连续谱、冻结 v3 的 raw-only latent 连续谱，分别配合 true-beta 和 estimated-beta 的同一线性目标。原 v3 联合输出作为同 raw 比较基准。C_true、true-beta 和标签只进入外部诊断。

正式求解保持 float32、alpha=1、固定观测 W、相同正则与 PCG 预算。连续谱泄漏、同 raw 噪声、正则化偏差均用相同系统求解；逐例核验 postblur 重组、真实残差与交叉项。失败时立即停止后续优化，保留全部未执行项的固定分母记录。一例 float64 固定系统参考可用于数值归因，不能替换正式失败结果。

在 E 盘仓库运行普通 Python `-B`，保持断言启用：

```powershell
python -B -m research_v6.tests_controls --output E:/.../reports/research_v6/implementation_checks/root/new_cpu.json
python -B -m research_v6.tests_native --device cuda --output E:/.../reports/research_v6/implementation_checks/native/new_native.json
python -B -m research_v6.run --phase register
python -B -m research_v6.run --phase stage-a --device cuda
```

输入／连续谱和分解各自的 CPU 检查入口随其 `tests_*.py` 提供。注册要求全部检查绑定当前源码；输出文件不可覆盖。实际环境解释器见 `common.PYTHON`。

报告写在 `reports/research_v6/run_001/`；数值组件写在 `outputs/research_v6/run_001/`。raw、旧标签和旧缓存不复制。所有失败、无效与未执行项都保留。缓存载入时间与实际推理／求解时间明确区分；当前阶段不构成实物吞吐量证据。

阶段 B 只有在完整阶段 A 证明必要且全部数值门槛通过时才登记和执行。36 个 signed confirmation 与 64 个 final 母场景继续封存。本轮不训练神经网络，也不进行重复 raw 似然或阈值调整。
