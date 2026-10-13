# v4.3 local-only bounded controls

本目录为2026-10-13本地研究时段，不推送公开仓库。原物理/权重/报告保持不变。

`fresh_trunk.py` 相对v4.2仅替换共享主干初始化；零残差头、物理、19通道、5797参数、六次共享校正、损失、采样、lr和500更新预算不变。实际执行设备为Windows RYLAN的RTX4060 Laptop，依据用户此前“那就用本机gpu完成”的直接要求。未占用N01的服务器GPU。

所有数据在E/D；缓存使用冻结raw对应的物理结果，不从NumPy预测反传，也不反传LM/PCG。推理不接收条件、身份标签或truth。属于物理反演后CNN校正器，不能称为完整展开训练。

本时段截止12:00，11:45前开始整理；代码在11:40后不再开始新的optimizer更新。每100次保存实际权重和完整776帧验证及原16项，最后重新加载核验步数与输出。报告保存全部失败；最终测试始终锁定。

`residual_scaling.py`/`scaled_train.py` 是第二个有限对照：保持v4.1 safe1的旧v4/update2000初始化、完整500次抽样前缀和优化设置，仅为六次共享CNN的总状态校正加一个冻结系数。系数由固定物理输出相对于输入raw的加权残差计算，数值定标只用训练输入，推理不读取truth或类别。设计动机已受开发验证结果影响；合成数据的噪声模式间隔不代表真实相机也有同样间隔。零残差恒等、负模板保留和含噪梯度三个CUDA检查已通过。

`scaled_analyze.py` 汇总每个checkpoint的全部门槛与分层指标，复用冻结旧结果逐例配对，记录系数分布、梯度及proposal/实际输出差异。最终测试保持锁定；无噪声保护通过本身不构成联合成功。

`balanced_sampling.py`/`balanced_train.py` 是第三个有限对照：在residual_scaled完成后，仅替换每次第二个含噪抽样，交替重复已有训练负弱/负正常各12帧；其余1500个位置与控制逐例一致。相同初始化与空Adam、系数、模型、物理、损失和500次预算。启动要求控制确已完成且剩余温度失败仅是含噪负组；零训练全量回执必须与控制精确一致。没有新增训练数据，也不把重复暴露次数作为独立样本数。报告入口 `python -B -m training_v4_3.scaled_analyze --name negative_balanced`。

三臂已完成，均没有联合合格checkpoint，当前不自动续跑。总报告见 `reports/local_session_20261013/LOCAL_STUDY_REPORT.md`。

`gradient_diagnostic.py` 在两500次末点各读取48个固定TRAIN分层探针，保存main/safe平均梯度与各参数块的余弦关系，更新0次。`field_decomposition.py` 对原144开发探针固定组合物理/CNN的连续谱及signed字段，完整组合必须复现原生输出；四组合全部报告，不能用于truth路由或选新模型。GPU只计算，分别用既有CPU绘图库环境运行 `python -B -m training_v4_3.gradient_diagnostic --report` 和 `python -B -m training_v4_3.field_decomposition --report` 生成报告。

`finalize_local.py` 在CPU上重新核对18个权重/Adam步数、数据缓存与原文件保留，生成本地总回执。所有逐例结果、失败和计算时源码快照均保留；报告图是从冻结记录生成的派生产物。
