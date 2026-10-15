# C1 第一 blank 分片只读成本诊断

已核验18格（16新＋2旧）终态；1080计划中44格有限且认证，1036格未执行。42个登记源和344个受保护旧文件均未变。诊断0 GPU、0 forward、0 inner solve、0完整逆、0 truth加载；仅3次CPU四根账本快照。

|同一终态文件mtime边界|预筛 positive_normal|本次 blank|
|---|---:|---:|
|启动至终态文件|18.94 min|52.51 min|
|16新数值核心 wall_s 合计|10.80 min|21.99 min|
|核心范围之外|8.14 min|30.52 min|

Blank核心约为预筛2.04倍，而外层评估次数5740/5650仅约1.016倍。两次raw条件不同，内部线性工作量和运行条件未经profiler测量，不能从次数接近推断每次工作的成本相同。52.5分钟并非全部来自账本或磁盘。

本次可以无重叠定位的回执创建→对应SHA sidecar落盘窗口合计20.09分钟；该路径包含预算全树遍历、源/身份检查、序列化/替换与主机调度，不等同于纯I/O计时。剩余未分配10.43分钟包括建模前检查、两旧端点只读复用、跨格间隙和全矩阵汇总等。CALL的sidecar落盘→PREPARED创建减去核心wall_s后，16格合计仅1.11s；先前CALL记录→PREPARED的约249s额外间隙，约247.9s发生在CALL回执本身的seal路径。RESULT sidecar落盘→cache score记录总共14.29s，覆盖外部评分加两次forward/readback等工作，不能将约30分钟外层开销归为评分。

SUMMARY JSON内部time为14:43:58.614Z，文件mtime为14:45:18.461Z，相差79.85s；字典的time在collect_rows和预算检查之前生成。完成分片记录为14:45:18.464Z，stdout最后写入14:45:35.736Z。内部时间不能直接当写完时刻。

源码成功路径模型：每个新成功格约32次Budget.check，每次分别扫描instance和四根global；旧预筛约27次。panel.seal先一次write_path，再两次store.write，共5次预算检查。模型由源码条件路径推导，并非动态profile。当前4547文件/424643484B，三次global tree_ledger分别0.909/1.951/3.150s，方差明显；32次global扫描仅以当前样本示意约29.1–100.8s/格，不能当历史归因或速度承诺。注册JSON6,316,764B，也存在每格重复读取/解析/哈希。

## 合成夹具边界

- `c1_development_001/cpu_contracts/fixtures`：229文件，571485B，4个run。
- `c1_selection_001/cpu_contracts/fixtures`：247文件，3893513B，2个run。

合计476文件/4464998B，占四根文件数10.47%、字节1.05%。六个run均匹配已结束的all_passed=true、GPU=false、formal_complete_inversions=0、truth_loaded=false合同。当前selection及原prescreen登记的直接read_dependencies均0条落在夹具树。NPZ仅读取头部，均为2×2、四系数xi等小型合成内容。固定数组/故障场景可由测试源码重建，UUID、owner、时间和历史文件hash不可由seed逐字节重现。

Root可依据用户已有清理授权，仅清理`CPU_FIXTURE_CLEANUP_MANIFEST.json`明确列出的六个run子目录；该文件保留逐文件SHA/大小、原合同引用和合成失败摘要。本诊断不删除。必须保留上级CONTRACTS*.json、冻结源码、本简报/manifest及全部真实科学raw、v3初值、inference、qualification、case_states和cell_receipts。夹具只占约一成文件，减少扫描工作是合理预期，但没有证据证明清理就能消除全部性能差。

## 执行判断

现在不改42冻源，不因性能删候选或分母；保持原登记，继续root明确选择的小分片是合理的。先做授权的小型fake清理并在后续独立CPU审计比较扫描耗时。若仍显著，再为新的账本/元数据事务去重方案单独做CPU故障合同和新登记；本轮不实现增量账本、不monkeypatch原store、不调整数值目标或认证规则。

下一输入`mother0000_positive_weak_noise0`是18格中0旧复用＋18新，不能按blank的16新估计。当前charged5131，16000授权余10869、原12000严格护栏余6869；development语义490/3400，余2910；本阶段16/1052，余1036新。全部1036新完成后charged6167、development1526，均有余量。下一18新后预计charged5149、已覆盖62/1080，仍不进行选型或24母确认。

证据：`READONLY_RUNTIME_ANALYSIS.json`（SHA ec2128225d13a06b049a472086de6d032600289d128694262706fdc12b5274c4）；逐项判断：`READONLY_RUNTIME_FINDINGS.json`；清理边界：`CPU_FIXTURE_CLEANUP_MANIFEST.json`。
