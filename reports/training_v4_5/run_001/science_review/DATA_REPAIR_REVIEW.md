# v4.6 负向含噪母体与噪声覆盖设计独立审查

结论：接受下述冻结的单个有界数据对照作为 v4.5 小池分支关闭后的下一步。它同时扩大负向含噪母体和噪声重复覆盖；不能把两种因素的效果分开，也不能声称所有新增正向/无噪声案例均已参与训练。本文确认科学设计与 metadata-only 抽样核对；最终 generator/cache/train 实现的静态审查应另存证据，本文没有执行或代替任何生成、缓存与训练。

v4.5 已完成 500 更新，六个登记检查点全部失败且候选数为 0。独立从保存的逐案例标量记录复算 signed 科学筛查得到同样结论。因此应停止在旧 12 母体 signed 小池上的优化，而不是追加小池分支。本次另登记方案不得改写旧失败结果。

## 冻结数据与真正接受训练的范围

拟生成的 432 个新增 TRAIN 描述符为：24 个新母体 × 4 个 signed 条件 × 4 种输入（continuous、quantization、noise0、noise1）=384，另为既有 12 个 TRAIN 母体的 4 条件各加一个 noise1，共 48。旧 signed 库 144 个 case 保留，合并得到 36 母体 × 4 条件 × 4 输入=576 个 signed TRAIN case。

原 TRAIN 的 12 个母体覆盖 6×6 几何网格中的 12 对，新 24 个母体覆盖其余 24 对；母体 ID 和预登记随机数命名空间与既有数据不冲突。这里只核对描述符，不把种子不冲突当作数组内容独立性证明。旧 3352 行 manifest 的行和文件保持原样，合并后 3784 行，其中 TRAIN=3008、旧 DEV=776。

新 registry 中另登记的 locked DEV 仅作为未生成的描述符保留。本阶段不生成、不缓存、不评估这些新 DEV，也不访问最终测试。旧 776 DEV 保持完全相同，只用于开发筛查；它们已经反复参与开发，不能改称独立验证。

## 逐位置配对与实际覆盖

独立 CPU 标准库核对最终 `planned_sampling()`，并自行按旧 manifest/新描述符重算抽样范围与计数，结果如下。

|项目|核对值|
|---|---:|
|总更新 / 微批次|500 / 2000|
|旧序列 unsigned / signed 微批次|1425 / 575|
|参与重新抽样的负向含噪位置|513|
|其他位置原 case ID 原位不变|1487|
|上述 513 中真正改变 case ID 的位置|502|
|抽回原 case ID 的位置|11|
|实际使用且需要新增缓存的独立 case|120|
|432 个新增描述符中本臂不使用的 case|312|
|新增正向或 continuous/quantization case 的训练次数|0|

513 个位置严格限于 `negative_weak/noise` 和 `negative_normal/noise`。其余 1487 个位置包含全部 1425 unsigned，以及 62 个原 signed 正向或无噪声位置；这些 case ID 和所在更新/微批次均保持旧 balanced 序列不变。条件、输入类型和 signed 标记在所有位置配对相同，因此不会暗改分层曝光次数。

每个负向 noisy pool 按 `noise_index`、`(S-T) mod 6`、T、case ID 排序，从第 0 项开始循环，不根据性能挑选顺序。实际完整曝光为：

|条件|draws|独立 case|母体|几何对|每母体噪声实现|每 case 次数|
|---|---:|---:|---:|---:|---:|---|
|negative_weak/noise|257|72|36|36|2|41 个为 4 次，31 个为 3 次|
|negative_normal/noise|256|72|36|36|2|40 个为 4 次，32 个为 3 次|

新增缓存的 120 个独立 case 由新 24 母体 × 2 负向条件 × 2 噪声=96，加旧 12 母体 × 2 负向条件的新 noise1=24 组成。旧 24 个负向 noise0 case 复用现有缓存。排序前六项覆盖六个 T 类和六个 S 类边际，但只占六个几何对；完整 36 对覆盖来自上述全序列，不能用前六项声称完整笛卡尔覆盖。

独立重算 draw SHA256 为 `85faa97c4acabfb8464868d098e9fe26a0c8c04a95f9d7d2d2d99c795300660b`，与主任务一致。旧源序列内容 SHA256 为 `dc3c70e5ead4893acf30446761aafc4971e6f5e8d6e4134e231523eac326d03c`。

## 唯一允许的运行与解释

本臂必须重新从旧 v4/update2000 初始化，并使用空 Adam，不能从 v4.5 失败末点继续。模型、校准、物理 baseline、保护损失、seed=4101、lr=3e-5、累积 4、clip=1、safe_weight=1、原 16 项门禁、96 组保护和冻结 signed 科学筛查保持不变。旧 balanced fixed-C 对照 v4.5 是直接配对参考；只有上述负向 noisy 父体/噪声选择可变化。

筛查仍在同一 checkpoint 上联合执行：旧 eligible 必须通过；四个 signed noisy 条件分别满足 MSE ≤ 1.05×physical MSE + 0.25 e²，两个 signed weak 条件分别满足 pooled L2<1，且全 776 个开发输出有限、invalid=0。候选按既定主 weak L2 最小、主 normal RMSE 次之、更早 checkpoint 再次之排序。不得取不同 checkpoint 的优势拼合候选。

登记上限为：432 TRAIN 生成 600 秒；只对实际新增 draw case 缓存，单 worker、CPU threads=2，3600 秒；一个 seed、单臂最多 500 更新，训练预算 1800 秒。中断和失败必须留存原始记录；预算到期或无候选后不得续加种子、步数、损失、采样或额外控制臂。若所有 checkpoint 仍失败，应关闭这一有限分支并交付负结果。

即使开发门禁通过，也只形成被冻结的开发候选。要声称独立验证成功，须先冻结唯一候选与评价端点，再按独立协议生成和使用 locked DEV；当前设计及审查没有开放该阶段。固定最终 C 得到的温度不变仍是结构约束，不能计作学习改善。本方案只能测试“负向 noisy 母体与重复覆盖联合变化”的效果，不能声称完整 36×4×4 库已受训、不能隔离父体数与噪声数效应、也不证明噪声外推或真实观测泛化。

## 证据与审查边界

本次 metadata 审计调用 `planned_sampling()` 和 `new_case_specs()`；已先检查模块顶层仅为标准库导入和函数/常量定义。审计没有导入 torch/numpy，没有读取数组，没有调用 register、generate、cache 或训练，没有创建新数据目录。只在独立 science_review 目录写了审查脚本和回执。

- `verify_repair_sampling_readonly.py`：独立核对脚本，SHA256 `2bb2475f90009a39ffc5df1d59bae5a425907759f69efbcadb40983f96f47793`。
- `data_repair_sampling_audit.json`：重开核对通过，SHA256 `f4df29e068fc79790ba6e15a4411b5b9641c9857675851f0c203de1fe42be12d`。
- 审计时 `training_v4_6/protocol.py` SHA256：`e3090b1821016da4fbb476c6dafc434817aad2e694ebb3210e22c764dcd63a0e`。
- 审计时 `training_v4_6/data.py` SHA256：`64709e98d939c92d87e299fdfae94db365690eb0eef420f79d7acdd7bcd3cdf6`。此为 metadata 实现版本，正式生成实现后续可能完成；不得把该 hash 当作正式执行源码已审查凭据。

执行注册还应冻结最终源文件与本审查的 hash，并在实际生成/缓存/训练入口校验。任何改变抽样 draw SHA、层内排序、case 选择、筛查或预算的修改，都不在本次设计审查范围内。
