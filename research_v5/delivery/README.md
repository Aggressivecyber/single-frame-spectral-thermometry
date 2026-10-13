# v5 决策性研究交付

本轮已执行全部预登记的适用科学诊断，并按任务书的停止规则得到负结果。v5 的部署物理估计未通过验收，CNN 未训练、候选未建立，36 个 signed 确认母场景和 64 个原始最终母场景继续封存。最终集性能未知。

从 [一页科学决策](../../reports/research_v5/run_001/V5_SCIENTIFIC_DECISION.md) 开始；完整指标、置信区间和限制见 [root 最终报告](../../reports/research_v5/run_001/ROOT_FINAL_REPORT.md)。[完成审计](../../reports/research_v5/run_001/COMPLETION_AUDIT.json) 将研究完成、数值失败、科学不通过和条件阶段不适用分别记录。[原始物理报告](../../reports/research_v5/run_001/development/PHYSICS_ONLY_V5_REPORT.md) 与补充执行 END 的哈希绑定保持不变。

代码、协议和可运行入口：

- 基线：`ddc4223fd1ecb9c5af82a8d1c489bf9cc54fd2d2`；冻结 v5 实现：`3b2de793e488e1a76b1f4f49241fb13d5c0cfd00`。
- [任务书](../TASK_SPEC.md)、[冻结协议](../protocol.json)、[完整阶段 CLI](../run.py)、[登记的场景计划](../family_plans.json)。推理接口仅接收 raw DN 与外部标定/选择参数。
- 实际 Python：`E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe`。在仓库根目录运行 `python -B -m research_v5.run --help` 可查看 CLI，不触发科学计算。运行环境记录在 `reports/research_v5/run_001/EXECUTION_ENVIRONMENT.json`。
- 历史执行顺序为 `register → oracle → identifiability → linear-solver → development-compare`。已完成输入不得在本 `run_001` 中重跑。原开发命令超时，唯一补充作业是 `python -B -m research_v5.execution_recovery.continue_development_001 --run`；其 prepare 和 run 均已完成，不得再次调用。
- `train-denoiser`、`candidate-review`、`final-once` 均有科学门禁。本次条件未满足，禁止开启它们。不存在 v5 训练 checkpoint 或通过验收的部署模型。
- 本目录的 `audit_saved_results_001.py` 只汇总与核对已经保存的结果，不能替代数值研究；其实际回执在 `reports/research_v5/run_001/root_review_001/`。它采用追加式输出，已经执行后不应覆盖运行。

Git 分支保存全部 v5 源码、协议、日志、逐例指标、图表与报告。GitHub Release 的分卷 ZIP 保存全部实际生成的 v5 科学数组和预测；各 ZIP 可独立解压到仓库根目录，内含完整相对路径。科学数据清单 `root_review_001/DATA_FILE_INVENTORY.json` 逐文件记录 SHA256；打包清单同时记录 ZIP 内容与 SHA256。运行时依赖、缓存和小型实现测试夹具不属于科学数据包。

全部新文件、临时文件和缓存位于 E/D。旧文件、权重与数组保持原状；旧封存数组仅核对元数据和只读属性，未为了交付打开内容。本轮结论适用于合成数据下的模糊后连续谱色温与 signed 模板偏离，不能据此声称真实气体温度、实测物种浓度或 10 kHz 性能。
