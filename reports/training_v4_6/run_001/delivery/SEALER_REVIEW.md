# v4.4–v4.6 封存脚本独立静态审核

审核对象是 `reports/parallel_session_20261013_v44_v46/seal_results.py`，154 行版本，SHA256 `d78240bb21e652be99ebbd25adc40ba05dfa22316af57ab1b054418f3ac7a7b9`。同字节审阅副本为 `seal_results_reviewed_source.py.txt`。root 统一修改原脚本；本审核没有执行封存、打包、模型或 GPU，也没有改写任何冻结训练文件。

结论：建议修复以下 6 项后再执行封存。当前 v4.6 尚未完成，本文不判定其训练结果、回放结果或搬移测试结果。

## 需要修复的项

1. **P2：原始输入回放回执路径错误，会确定性中止。** 第 53–54、82–83 行使用 `native_replay/receipt.json` 和 `native_replay/watchdog_receipt.json`。冻结的 `native_replay/replay.py` 定义 `EXECUTION = OUT / 'execution'`，两份回执写在 `native_replay/execution/`。应使用真实路径，并同时验证 watchdog 的 `worker_receipt_sha256` 等于当前 worker 回执 hash、`exit_code == 0`、`passed is True`、`watchdog_termination is False`；worker 要求 `passed is True`、`case_count == 3`，`endpoint` 绑定 v4.6 完成回执中唯一的 update 500。两份回执的成功状态字符串都是 `SUCCESS`。

2. **P2：推理包和搬移证据没有绑定，可能误报“运行包验证完成”。** 第 58 行的 `bundle` 此后没有被使用；任意能解析的 delivery 下 JSON 都会通过这个入口。应要求 builder 的 `status == 'COMPLETE_BUNDLE_BYTES_AND_ZIP_VERIFIED'`，校验 `checkpoint_update == 500`、`checkpoint_sha256` 与同一原生完成权重相等、`checkpoint_auto_selected is False`，并复核 `zip_sha256`、`zip_bytes` 和 `bundle_files_sha256_and_bytes`。第 60–61 行只检查 portability 的状态和一个布尔值，尚未绑定当前包或参考预测。root 新增 `verify_portability.py` 的生产 schema 已提供 `build_receipt_sha256`、`zip_sha256`、`inference_receipt_path`、`inference_receipt_sha256`、`reference_receipt_sha256`、`bundle_payload_unchanged`、`physical_state_and_beta_exact`、`original_checkout_module_imports`、`isolated_python`。封存器应逐项绑定这些字段，并从实际 inference 回执核对同一 raw、权重、参数和配置。该搬移设计只回放一个旧 DEV negative_normal/noise 案例，报告宜明确“一例搬移检查”，避免被理解为三个案例全部做了搬移检查。

3. **P2：候选数和完整检查点结论仅依赖未绑定的汇总值。** 第 37–47 行只看 `candidate_count == 0`、`selected is None` 和跨三臂总数 18；即使某臂有重复或缺失更新、Adam 步数错误，或者同臂 `science_summary_0500.json` 已标记候选，也不一定会拒绝。每臂应恰有 0/100/200/300/400/500 各一条，并验证 `Adam_steps` 为初始空列表或对应更新数。读取全部六组 summary/science 和各 776 条记录，核对更新号、每组 case ID 唯一性、`science.validation_sha256`、`science.protocol_sha256`、`science.old_eligible == summary.eligible`、同权重合取关系；由这六组重算候选数并与 selection 一致。不能用旧 selection 的 0 掩盖新产生的候选。已完成的分析回执可提供输入/输出 hash 链；注意不同版本 schema：v4.4 analysis receipt **没有 status**，v4.5 为 `SUCCESS`，v4.6 将为 `COMPLETE`。不要对三版施加同一状态字符串。

4. **P2：先写“完成”报告、后核验 ZIP，失败会留下误导性半成品并阻断原路径重试。** 第 69、88、117、125 行依次以 `x` 模式写永久汇总，直到第 130 行才发现已存在 ZIP 或其他压缩失败。若后段失败，`FINAL_REPORT.md` 仍写着全部完成，下一次运行又会先在已存在的 preservation receipt 处中止。建议在任何写入前预检所有最终输出路径；使用新 attempt 目录记录中间文件和失败原因，只有 ZIP 与全部证据通过后才发布完成标记和最终报告。失败产物保留，不自动删除或覆盖。

5. **P2：ZIP 逐项检查未使用冻结 inventory 的 hash。** 第 124 行先记录磁盘 hash，第 131 行再读取文件压缩，第 137 行却拿 ZIP 和当时磁盘的新 hash 比。如果文件在 inventory 后、ZIP 写入前变化，两个新副本可比较通过，而压缩包中的 `release_manifest.json` 仍是旧 hash。应逐 entry 对冻结的 inventory `sha256` 和 `bytes` 校验，并在封存前复核这些输入未变化。`release_manifest.json` 自身在外层回执单独绑定；最终 report/evidence hash 应与 ZIP 中相同 entry 一致。

6. **P2：历史文件在 ZIP 内的表述超过实际包范围。** 第 110 行“需要精确匹配历史字节哈希时应使用交付ZIP”没有范围限定。实际第 119 行只纳入新增 v4.4–v4.6 source/checkpoint/report 与本 session 文件夹，原 1046 个 starting files 并没有因此被复制进 ZIP；manifest 自己也写明原仓库和全量 raw 另存。建议改成“本次 release manifest 纳入文件的原始字节由结果 ZIP 保留；1046 个起始文件在原工作副本中逐项验证，未全部包含在此 ZIP”。这不要求扩展包体积。

## 已核实字段与可直接使用的证据

- 实际 generation receipt 的字段为 `generation_seconds`、`new_train_rows=432`、`merged_train_rows=3008`、`new_train_mothers=24`、`old_validation_rows=776`。cache 为 `selected_case_count=120`、`elapsed_s`；这些当前代码字段没有 schema 错误。
- 冻结 manifest 的 TRAIN 确有 3008 行、292 母体。sampling 描述计划 500×4 个位置，971 个不同案例、256 个母体。封存时的“实际训练”数量还应和完成回执的 `same_frozen_negative_noisy_parent_repair_draws`、`actual_unchanged_sample_identity_comparisons=1487`、`training_curve.jsonl` 的 2000 个样本身份相连。本文只验证已冻结计划，不预先宣称未完成的训练已执行这些位置。
- 生成数组审核回执 SHA256 为 `fa9848aa4a0d14ec102ed27dbbaca2d6da5cf0fe580918f752e8876cd93da1db`，状态 `PASS`、576 个文件、11580 项检查、0 失败。其 `array_audit_files.json` 逐文件成员元数据可导出数组总数；封存可由该表生成计数，而非写死。物理 buffer 不变是 generation receipt 的见证，数组审核未独立重跑物理数值；原报告的科学界限应保留。
- 六项 CPU mock 测试回执为 `training_v4_6/_test_runs/20261013T041311Z_f6bd0056/cpu_contract_test_receipt.json`，SHA256 `e27a0d603cd601fbcab5d13ecd0cc44c2eb53d0dee216447fb6c5e6aa9fc0528`；实际字段 `status=PASS`、`tests_run=6`、`failures=0`、`errors=0`。当前 ZIP 会排除整个 `_test_runs`，建议显式纳入这一份小回执并绑定其 source hashes，无需复制 mock 数据。
- v4.4 zero audit receipt SHA256 `71f3a93620eb2d8466b5ac80e020d635b5159f4069566430de34ae0e9780cb7a`，两条 arm 的 `new_optimizer_updates=0`、`paired_cases=776`、`science_candidate=False`。它们是对旧 v4.3 第 500 更新端点施加新筛查、额外训练 0 步；写成“两个额外零训练端点审计”比“两个零更新权重”准确。
- 程序当前只在“无候选”时收口是合理的门控。若最终出现候选，应停止这个负结果封存分支并遵守已登记确认规则，不能为了完成交付把候选结果改写为负结果。

## 未执行范围

只读取源码、JSON/JSONL 元数据和现有审核文字；没有打开 checkpoint payload、raw/truth/state 数组，没有调用训练/分析/封存主入口，没有启动模型或 GPU。审核对象正在由 root 修改，以上行号和结论严格绑定开头的源文件 SHA256；修复版本需要另行静态复核。
