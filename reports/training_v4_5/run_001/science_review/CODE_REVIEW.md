# v4.5 独立源码、输入完整性与已保存结果审查

结论：本轮限定对照实现符合登记设计，未发现新的阻断级源码问题。v4.4 结束后修复的 P2 恢复检查点问题在静态控制流中已消除。v4.5 已完成 500 更新，但六个登记检查点均不是科学候选，12 母体小池分支应按既定规则关闭。

审查仅使用 CPU 标准库、源码、文件字节哈希及已保存的标量记录。独立审查没有执行模型、物理求解、训练、故障注入、GPU 或最终测试。训练回执中的原生重放和 Adam 状态来自主运行；本审查重新核对文件字节与保存记录，不把它们称为第二次独立原生执行。

## 配对与零更新锚点

- `training_v4_5/protocol.py` 在 v4.4 `COMPLETE`、500 更新和候选数 0 后另登记本臂；相对旧 v4.3 `negative_balanced`，只改最终连续谱 C 约束；相对 v4.4 `fixed_continuum`，只改为旧 `negative_balanced` 的完整抽样序列。
- 起点仍是旧 v4/update2000，Adam 状态为空，seed=4101、学习率 3e-5、累积 4、clip=1、safe_weight=1、500 更新上限和 1800 秒登记预算保持一致。没有从 v4.4 末点续训。
- `train.py` 在首个训练更新前计算全部 776 个开发输入和原 16 项检查，并直接断言完整 summary 等于 v4.4 的零更新 summary。独立检查进一步确认两份 summary、776 条逐案例 JSONL 和原 16 项门禁文件均字节一致。
- 500 次已保存训练更新的全部 `samples` 字段，逐更新等于旧 balanced 对照；实际共 2000 微批次。manifest、开发 ID 序列、抽样文件及抽样内容的 hash 也在运行入口检查。
- 五个已登记运行源文件与各自运行快照和开始 hash 一致。`source_integrity_receipt.json` 保存具体路径和哈希证据。

零更新三项文件 SHA256：

|文件|v4.4 / v4.5 共同 SHA256|
|---|---|
|validation_summary_0000.json|d62307374b2cf963a5f22de12c2a4170258b81af8562af1ad3433c024f35fca1|
|validation_update_0000.jsonl|c2de921f43722b1ad57330fcb3acfef922e490b62ec83a9d22bbb352d18d3f33|
|gates_update_0000.json|404a072985ea10e958c76d2fd6dbab662fa0ae2c54dcaaebdb3896cb5e974330|

## v4.4 结束后的 P2 修复

原风险是完成训练后重载早期 CNN 权重进行检查，而内存优化器仍在第 500 步；此阶段发生异常会把早期模型与末期优化器一起保存成 `interrupted.pt`。

修复在重载任何历史模型前设置 `phase='checkpoint_verification'`，异常分支仅当 `phase=='training'` 时保存中断检查点，并记录具体阶段与是否保存。v4.4 修复版和 v4.5 都遵循该控制流，因而检查阶段失败不会再制造该不一致恢复文件。此结论属于静态控制流审查，没有运行故障注入。

v4.4 原训练源码仍保存在 source snapshot，`postrun_code_hardening.json` 保存原 hash 和修复后 hash。对 `evaluate.py` 的变更仅增加可选 protocol 路径及对应回执来源；没有修改评分或阈值数学。固定最终 C 的模型源码未改变。

## 实际输入集合的独立字节审计

`verify_inputs_readonly.py` 对本臂实际使用的 851 个独立 TRAIN case 和全部 776 个 DEV case 重新读取文件字节并计算 SHA256，总计 1627 个 case、7338 个独立文件、3,137,822,839 字节，47.281 秒。1627 个 case 中有 1618 个与 v4.4 审计重叠、9 个是本次新增覆盖；所有文件均在本次重新 hash，没有复用前次审计的文件 hash 缓存。

核对包括 raw、label、cache、cache metadata、物理评分记录与 manifest；同时核对冻结申明、raw/label/cache 标识、实际缓存定位、physics/solver 配置及 `labels_used=false` 元数据。结果为 PASS，failure 列表为空。它证明审计时实际磁盘集合与冻结记录相符，不证明 runner 在启动前已逐字节执行同等规模审计，也不单靠元数据证明所有求解器内部实现。

## 六个检查点的独立保存记录复算

`verify_saved_screens_readonly.py` 从每个检查点的 776 条已保存标量记录重新合并 signed SSE、真值能量与像素数，复算四个 signed noisy 条件的 MSE/L2 门禁，并核对逐案例顺序、更新号、协议 hash、文件 hash、全量有限性及 invalid=0。六份 checkpoint 文件也逐个重新 hash，与训练回执一致。

|更新|主 weak pooled L2|旧 eligible|新科学筛查|联合候选|
|---:|---:|---|---|---|
|0|0.686383|通过|失败|否|
|100|0.937455|通过|失败|否|
|200|1.004925|失败|失败|否|
|300|1.112587|失败|失败|否|
|400|1.082229|失败|失败|否|
|500|1.063764|失败|失败|否|

500 更新端点：negative_normal RMSE=34.388603 e，高于冻结上限 20.418884 e；negative_weak L2=1.197938、positive_weak L2=1.250174，均未达到小于 1；主 weak L2=1.063764 也失败。正 normal 的 MSE 合格不能抵消这些失败。全 776 个输出有限且 invalid=0，最终温度修正 RMS=0；后者是固定最终 C 约束的结构结果，不能算作温度学习收益。

本独立复算从已保存 scores 出发，没有重新从模型输出数组运行 `score_arrays`。每个检查点使用自己的完整指标，不拼接检查点。候选数 0 与主任务 `candidate_selection.json` 一致；没有选出独立验证或部署端点。

科学解释仍限于六阶段、5797 参数的物理反演后 CNN。固定最终 C 不等于冻结内部所有连续谱头，也不构成全可微 LM/PCG 展开网络。四格对照可以描述该 seed 和该开发集合上的差异，不能给出跨种子显著性、泛化成功或因果机制证明。

## 证据文件

- `input_integrity_receipt.json`：独立实际输入字节审计，SHA256 `0c547834bbdecd1edab943c12c39efdeb62ba7aac8cfc98dacadb14629e475d4`。
- `source_integrity_receipt.json`：运行源码与快照、v4.4 结束后修复核对，SHA256 `e8d02c6791e4188b8371e9a966002abac5b68cd220a5648bafc7b6e3c0aae760`。
- `saved_screen_audit.json`：六个检查点保存记录复算，SHA256 `8f9cd3dbed6373ab9991027af83efabeb776ba3957709f5091bde017d79b9d88`。
- 登记 v4.5 协议 SHA256：`85d02a04783027c6e551ceee1168f237d1a8119104a0b3a1c38ac6eb2da5cfdd`。

下一步只能进入另登记的、有界负向含噪母体/噪声覆盖对照。不得因本轮失败追加小池种子、步数、损失或 C 变体。
