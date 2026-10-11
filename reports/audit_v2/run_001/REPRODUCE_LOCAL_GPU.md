# 本机 GPU 复核与报告重建

本轮全部模型推理使用本机 RTX 4060 Laptop GPU。原服务器仅用于取回旧冻结帧和读取 160 个训练温度 RNG 标量。本轮没有训练或更新 checkpoint。

已完成的 40 个正式测试分为：原评估/物理/方法 37 项、光学扩散后的源外误差 2 项、套件顺序不影响已知标定母帧选择 1 项。原 37 项门禁日志及注册文件保持不变，汇总见 `package_verification.json`。

默认复用已经冻结的 run_001。不要对这个目录再次运行 inventory、freeze-smoke 或 register-methods；前者记录运行起点，后两者用于新运行的首次冻结且明确拒绝覆盖。下面命令适用于当前本机，输出、缓存和临时文件均在 E 盘。

```powershell
Set-Location -LiteralPath 'E:\CodexWork\thermometry\public_repository_001'
$auditGpuPython = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
$auditReportPython = 'C:\Users\10629\AppData\Local\Programs\Python\Python313\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = Join-Path (Get-Location) '_runtime/audit_deps'
$env:TMP = Join-Path (Get-Location) '_runtime/tmp'
$env:TEMP = $env:TMP

# 可选：再次核查 40 项测试。新日志不覆盖原注册门禁。
& $auditGpuPython -B -m pytest tests --junitxml=reports/audit_v2/run_001/tests_recheck.xml

# 可选：核查续跑。成功记录在完整配置和实际输入哈希一致时复用；
# 8 项饱和拒绝会重新记账并保留，不会删除失败历史。
& $auditGpuPython -B audit_next.py --phase compare --suite all --resume --device cuda --output reports/audit_v2/run_001

# 下面只读冻结输入/预测/JSONL，重建分层结果和图，不执行神经网络推理。
& $auditReportPython -B audit_next.py --phase summarize --device cpu --output reports/audit_v2/run_001
& $auditReportPython -B reports/audit_v2/run_001/finalize_report.py
& $auditReportPython -B reports/audit_v2/run_001/render_continuous_input.py
& $auditReportPython -B reports/audit_v2/run_001/verify_package.py
```

图示核查补充：连续电子 oracle 图最初首面板显示容器中同时保存的量化 DN，现改为实际进入该诊断的 `expected_e`，明确以未量化电子为单位。原图保存在 `outputs/audit_v2/run_001/report_render_prior/`，数值推理和评分不变；`figure_input_correction.json` 记录旧图、新图和实际输入数组哈希。最终报告重建应包含上述 render 步骤。

`outputs/audit_v2/run_001/` 含 167 个冻结容器、训练统计、预测和 trace，约 787 MiB，被 Git 忽略；报告目录约 26 MiB，不被忽略。跨机器复现必须显式复制冻结目录、报告中的注册/标定文件和同版本源码，再核验哈希。公开仓库本身未包含全部研究归档，因此不能从重新抽样的数据宣称复现这 50 个旧帧，也不声称重算了原全部 768 帧。

数值队列全部处理：954 个预注册方法/输入组合，946 成功，8 个明确饱和拒绝，剩余 0。八个拒绝对应同一个旧饱和输入的八种方法；不是八个输入丢失。`final_receipt.json` 的 PARTIAL 表示有数值拒绝，`delivery_verification.json` 的执行状态表示队列已全部处理，两者含义不同。成功计算不等于工程筛选或科学接收。

两条额外续跑探针（一个缺 metadata 异常、一个额外比值诊断）仍留在 JSONL，明确排除在预注册 954 项之外。旧 152 条部分版本记录及预测也完整保留。源码兼容性修复见 `orchestration_compatibility_v003.json`；无模型、求解器或指标数值变更。
