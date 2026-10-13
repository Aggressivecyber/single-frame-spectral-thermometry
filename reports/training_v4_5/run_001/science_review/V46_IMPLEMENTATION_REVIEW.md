# v4.6 实现冻结前独立审查

结论：对本文件列出的五个源码 hash，静态审查与标准库真实元数据检查通过，当前没有未关闭的代码发现。本次发现的两项 P2 均已在冻结前修复。该结论不等于正式 CUDA 生成、反演、训练、计时或科学验收已经通过。

本审查未修改运行实现，未调用 register、generate、缓存物化、训练或正式分析器，没有导入 numpy/torch/runtime，没有读取数组或使用 GPU。仅在独立 science_review 目录写本报告与回执。科学设计仍以已定稿 `DATA_REPAIR_REVIEW.md` 为准；该设计文件未改动。

## 两项发现与关闭依据

|编号|级别|原问题与影响|冻结前修复|
|---|---|---|---|
|V46-REVIEW-001|P2|分析器最初只计算当前 checkpoint 的 hash，而未与训练回执绑定。训练结束后若权重文件被替换，保存评分仍可能被错误关联到替换权重，并产生不对应的候选。|分别读取 v4.5/v4.6 的六个 checkpoint 回执，核对更新清单、声明 Adam 步数和当前文件 hash；v4.6 候选 path/hash 再与同更新回执绑定。每份 science 的协议 hash、summary/science/全部逐案例的更新号也校验。静态复核通过。|
|V46-REVIEW-002|P2|已冻结 scalar_engine 与 repair_v3/solver，但后者实际调用顶层 solver.py；漏掉该依赖时，算法源变更不会触发本次冻结检查。模型构造还使用 config.py。|顶层 solver.py 和 config.py 已进入 protocol 的 frozen inputs；各正式入口检查全部冻结输入，训练结束也重新检查。静态复核通过。|

未运行上述异常路径的故障注入；“修复”仅指对应检查已在正确的控制流位置落实。

## 核心路径核对

`protocol.py` 只在 v4.5 完成 500 更新、候选数为 0、独立输入审计 PASS 后登记。抽样在任何新数组生成前由描述符确定，源文件与科学设计审查同时被冻结；生成、缓存、训练需提供 execution protocol 的精确字节 hash。独立调用 `planned_sampling()` 的 draw SHA 仍为 `85faa97c4acabfb8464868d098e9fe26a0c8c04a95f9d7d2d2d99c795300660b`，与已审查设计一致。

`data.py` 的正式入口先核对动作、split、计数、预算、冻结文件、v4.5 关闭状态和 TRAIN-only 抽样，然后才创建数据目录、导入 runtime 或启动 CUDA。新母体沿用既有 mother_state、+250 K 连续谱变换及 paired signed 合法性规则，登记种子不因失败替换。只生成 432 个 TRAIN raw、96 个新标签文件，并只读复用 48 个旧 TRAIN 标签；旧 3352 行和 776 DEV 保持原样。保存文件拒绝覆盖，npz 保存后重开比较实际数组。非法、饱和、非有限或重复数据保留诊断后失败；不生成 locked DEV/final 数组。

数据碰撞检查的证据边界写入 generation receipt：新 pretransform 字段对旧 manifest hash、实际 posttransform 字段在新母体之间、raw 数组/文件和标签文件 hash 均检查；旧 posttransform 字段/标签数组没有为此重新打开，近重复和统计独立性也没有估计。不能把该实现描述为已证明任意层面的独立性。

`cache.py` 按冻结 draw 的首次出现顺序去重，只物化实际用到的 120 个新 TRAIN case。物理反演接收 raw 与固定 camera/solver 参数，标签在反演结果落盘后才为评分侧文件打开。物理缓存和模型输入路由保留旧实现；新增路径在读取数组前校验 cache/score/raw/label 身份、metadata hash、physics/calibration/solver/method/engine、labels_used=false 与缓存字节。旧行走旧只读 loader；不产生其他 312 个新增 case 的缓存。

`train.py` 的 preflight 检查生成/缓存 COMPLETE、两份 manifest 与回执 hash、120 个实际新增缓存集合、旧 3352 行逐项相同、开发集行与顺序不变、500×4 抽样及真实旧训练日志配对。初始 CNN 仍为旧共享起点，空 Adam；零更新先完整复算 776 DEV 并要求 summary 与 v4.5 相同。训练中其余 1487 个位置的 raw/cache/sample 身份逐个比较。六个检查点与旧 DEV 原生回放、两个新增 TRAIN 输入/输出重载比较在训练结束执行；进入历史权重检查前切换 phase，避免把早期模型与末期优化器保存成中断恢复文件。只有主运行完成这些路径，相关执行证据才能成立。

`analyze.py` 仅读取保存分数，核对同更新 776 个 case 的 raw/label/physical cache 身份，0 更新逐条相同，保留全部 4656 对。只在 v4.6 六个检查点内按已冻结规则选联合候选；报告明确反复使用的开发集、联合母体/噪声改变、温度结构恒等和无候选后的硬停止。输出先进入新 analysis_runs 目录，发布文件拒绝覆盖。图像实际绘制和视觉验收尚未执行。

## 本审查实际执行的检查

- 五个 `.py` 文件均经 Python AST 解析通过，没有生成 pyc。
- `train.self_test()`：合法不变位置与负向含噪替换通过；五类禁止修改均被拒绝，包括普通 raw 变动、正向替换、无噪声替换、改变条件、DEV 进入训练。
- 使用真实旧 manifest、432 个新描述符、最终 planned_sampling 和 v4.5 的 500 条实际训练记录调用 `sampling_checks()`：2000 微批次、575 signed、513 负向含噪位置、1487 原样位置、502 个真正改变的 case ID；两负向组均 72 case / 36 母体 / 36 几何对 / 每母体两个噪声、每 case 至少 3 次。
- `selected_new_case_ids()` 得到的 120 个唯一 case 与冻结 sampling 的新增缓存集合完全相同。
- 核对检查期间源码 hash 未变化；检查结束确认 numpy、torch、runtime 均未导入。

回执为 `v46_implementation_review_receipt.json`，已重开核对，SHA256：`4faac627d34516b66b3b397894f7acc427ba3e7b33b2831efbea7f3d66ed2473`。

## 本次审查的精确源码

|文件|SHA256|
|---|---|
|training_v4_6/analyze.py|7a83190e53b8e9a6df4c89b2df1328f4c71506e505ed1c167d3fb605de736d1c|
|training_v4_6/cache.py|cadd854df150e03b8e53ce49961b82eafc1f988ea3dbec5c98a1527fcad5a34d|
|training_v4_6/data.py|9a2f31d383625fdc298d650bcb9ffe77f3e8fc64fc49b29b9810700ddb887462|
|training_v4_6/protocol.py|667e09f9e755c34b18485af45d556884472f2ea51ca6902df7629480b0cfdcbe|
|training_v4_6/train.py|383dda12572d5c333e1229f11d0803919af496c766e5d3aef99fcb782ad94c99|

本轮没有发现改变科学筛查或突破训练/开发/final 边界的剩余问题。实际 wall-time、内存、磁盘用量以及运行失败处理的完整性仍须由正式执行回执和验收给出；静态源码与 metadata 检查不能替代它们。
