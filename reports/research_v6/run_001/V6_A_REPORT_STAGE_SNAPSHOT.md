# v6 阶段 A：固定 raw 的连续谱来源置换

状态：**STOP_NUMERICAL_OR_ORACLE_AUDIT**。实际处理 1/720 个冻结 raw；返回线性比较 4/4320 行、原 v3 1/720 行。

原输入、标定、v3 方法、W、alpha=1、正则与迭代预算均固定。C_true 和 true_beta 属于合成 oracle 诊断；C_v3 与 C_v5 的推理只接收同一 raw 和标定。

## 同 raw 的正常含噪模板比较

|正负|连续谱／背景|模板 RMSE (e)|色温 RMSE (K)|幅度投影|完整 raw 数|
|---|---|---:|---:|---:|---:|
|negative|C_true__estimated_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|negative|C_true__true_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|negative|C_v3__estimated_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|negative|C_v3__true_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|negative|C_v5__estimated_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|negative|C_v5__true_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|negative|original_v3__joint_original|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|C_true__estimated_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|C_true__true_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|C_v3__estimated_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|C_v3__true_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|C_v5__estimated_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|C_v5__true_beta|未得到完整证据|未得到完整证据|未得到完整证据|36|
|positive|original_v3__joint_original|未得到完整证据|未得到完整证据|未得到完整证据|36|

完整分组统计保留 continuous、quantized、noise；normal 正负、weak 正负和 blank；development 与 oracle 单独列出。见 A_DECISION.json 的 all_group_metrics。任何失败／无效输出均保留，主误差不能通过剔除像素或帧改善。

## 分量恒等检查

`实际误差 = 连续谱泄漏 + 同 raw 噪声 + 正则化偏差 + 数值重组残差`。所有量在可观察 postblur 模板域比较，交叉项必须保留，RMS 不能直接相加。

分解行数：4；未通过数值门槛：1；全矩阵数值门槛：False。

每例分解见 continuum_leakage_decomposition.jsonl；实际／泄漏／噪声／偏差求解均记录物理与平衡真残差。浮点 64 参考只诊断固定目标的数值问题，不能替代失败的正式浮点 32 输出。

历史 O0/O1/O2/O3 使用 alpha=0，当前置换使用固定 alpha=1。历史值仅作背景。当前真连续谱无噪误差应由相同正则化的偏差解释；不能移植 alpha=0 的无噪绝对门槛。

## 决策

唯一下一步方向：`numerical_data_chain_diagnosis`。阶段 B 执行资格：False。固定预算成像设计诊断资格：False。
停止原因：{'case_id': 'v4_validation_mother_0000_v5_negative_normal_continuous', 'source': 'C_true', 'beta_mode': 'estimated_beta', 'reason': 'Primary fixed-objective numerical decomposition gate failed; no optimization allowed'}。

所有结果仍是冻结合成标定中的 postblur 色温和有符号数值模板；未验证真实气体温度、物种浓度、实物硬件吞吐量或封存确认／最终集。
