# v4.6 最终科学结果独立复核

**结论：六个检查点均未形成开发候选，应关闭已登记的这条有限数据修复分支。** 已完成的500步训练提供了可保留的负结果；本阶段科学验收目标尚未达到，不追加种子、步数、损失或采样变体。

本复核仅读取已保存的标量、JSON/JSONL及报告文字。独立重算了两臂共12个检查点、9312条开发记录上的汇总和门禁，并逐项核对4656组成对记录及3120条母体/条件/输入类型汇总。未打开模型、权重文件或raw/truth/state数组，未调用模型、物理反演或GPU。

## 第500步的同权重结果

|指标|v4.5|v4.6|冻结要求或本次判断|
|---|---:|---:|---|
|主weak pooled L2|1.063764|1.060628|必须小于1；失败|
|主normal RMSE / e|13.752353|13.474054|描述性下降，不构成联合通过|
|负weak pooled L2|1.197938|1.127744|必须小于1；失败|
|正weak pooled L2|1.250174|1.177063|必须小于1；失败|
|负normal RMSE / e|34.388603|35.361563|上限20.418884 e；失败|
|正normal RMSE / e|20.164745|19.464186|上限21.000538 e；通过|

v4.6 相对v4.5的负normal RMSE增加了 0.972960 e。其他列的局部改善没有形成同一权重的联合通过。负、正weak虽通过相对物理基线的MSE非劣界，pooled L2仍大于1；不能把这两种门禁混为一项通过。

500步原16门禁、96个无噪/量化保护分组和含噪温度筛查均通过，但主weak严格小于1的旧要求未通过，新增筛查另有负weak、负normal、正weak三组失败。负normal的MSE为1250.440146 e²，冻结上限416.930821 e²。

## 全部六点及配对复核

|更新|主weak L2|旧eligible|新增筛查|联合候选|失败signed组|
|---:|---:|---|---|---|---|
|0|0.686383|True|False|False|negative_weak, negative_normal|
|100|1.002103|False|False|False|negative_normal|
|200|1.029582|False|False|False|negative_normal, positive_weak|
|300|1.093808|False|False|False|negative_normal, positive_weak|
|400|1.109187|False|False|False|negative_weak, negative_normal, positive_weak|
|500|1.060628|False|False|False|negative_weak, negative_normal, positive_weak|

负normal在全部六点均失败。100步时两种signed weak已分别低于1，但主weak为1.002103且负normal仍超限，因此该点也不能选为候选。没有把不同检查点的局部最好数值拼接成一个模型。

两臂每个更新点的776个case ID、raw/label hash、物理缓存身份及元数据逐项相同；0更新的全部记录和JSONL字节也完全一致。4656个配对MSE差值与分析产物逐项相等；3120个母体分组均值重新计算一致。此处验证保存的身份和分数，没有重新读取相应数组字节。

保存的全部指标有限，invalid比例与连续谱色温修正RMS均为0。保存的forward审计在每点776帧和16原门禁上都记录最终连续谱、色温与物理基线恒等。温度表现属于结构约束，不能宣称学习改善。

## 训练暴露和报告边界

实际500×4个样本位置与冻结sampling完全一致：513个负含噪位置重采样，其中502个case ID实际改变、11个返回原case，其余1487个样本身份原样保留。实际971个案例、256个母体；新增432条TRAIN中用到120条，另外312条未进入本臂训练。两个负含噪条件各使用72个案例、36个母体，每个案例3或4次。旧首个noise记录没有noise_index字段，本复核没有向旧元数据补造该字段。

本干预同时扩充负含噪母体覆盖和噪声重复，不能分别归因。仍然使用反复查看的776帧开发集；这些配对是描述性比较，不是新的独立确认、独立像素统计显著性或真实测温/硬件验收。保存协议、训练和分析回执均未使用新确认集或final test，本复核也没有打开它们。

现有V4_6_REPORT.md的六点表、八行500步signed表、失败分组、4656配对及结论范围均与保存证据一致，没有发现需要阻断交付的科学表述问题。图像视觉QA、原始输入回放、权重字节和独立目录推理属于其他执行回执；本复核不替代这些验收。

## 身份和回执

- 冻结protocol：`4d49cd48853b20f62960754d07637a3cca3ae454e79e8e4ac67b8a2f7d78b807`。
- 完成训练回执：`a6fe06decc38c09133035146cf69c6fde3b597f985b6173c74c73c04acb9ce42`。
- update_0500.pt的原生训练回执声明hash：`65bd9f127b7cf89b4390d684ffcd81c743b0368b0c0b701bd8d1d43a6e252560`；本复核未重新打开该权重。
- 独立标量复核回执：[final_science_review_receipt.json](final_science_review_receipt.json)，SHA256 `b4397361a8e95efa2af46eefefc27b30e1fa7b722a8f49f6fac99d583e2b7734`。
- 审核脚本：[verify_final_science_readonly.py](verify_final_science_readonly.py)，SHA256 `56f4cbb4988fb9eab3e8c2cc5185f5e44a2fdd9b716645572be63184a2658086`。
