# 声明—证据对应表

证据截止2026-10-15；本表随正式结果补齐。`manuscript_zh.md`是工作稿，封存最终评价尚未执行。

| 正文声明 | 输入与完整分母 | 直接证据 | 适用范围/未完成项 |
|---|---|---|---|
| 连续谱来源替换降低当前模板误差 | 36开发母场景，正常含噪正/负模板，同raw同线性配置 | [A复用比较](../reports/closeout_001/A_REUSED_COMPARISON.json) | 约64 e至18 e；未优于完整v3；各RMSE之差不是方差项 |
| 四项分解包含实际及分量求解残差 | 90输入、6母场景、540来源/背景组合 | [来源汇总](../reports/closeout_001/SOURCE_STUDY_SUMMARY.json)、[终态](../reports/closeout_001/FOUNDATION_CONTINUED_RESULT.json) | 闭合和数值项描述固定线性目标，不等于相机精度 |
| 强传统基线有限选参选择100 | 12母场景选参，24母场景/120含噪输入确认 | [开发科学汇总](../reports/closeout_001/DEVELOPMENT_SCIENTIFIC_SUMMARY.json) | 历史0.1与预选100分别命名；保留结构/投影取舍 |
| 原九组未完成12+24科学比较 | 9组合×2初值，同1个开发raw，18有限输出 | [开发终态](../reports/closeout_001/DEVELOPMENT_RESULT.json) | 两初值共同数值合格的组合为0；历史排除不等于科学无效 |
| v3生产精度尚未认证 | 60输入，固定12外层、float64/线性设置对照 | [开发科学汇总](../reports/closeout_001/DEVELOPMENT_SCIENTIFIC_SUMMARY.json) | 是有限估计器敏感性对照，非非线性最优参考 |
| 原生耦合预条件构建存在批量显存失败 | 同2raw、4登记格；2旧PCG收敛、2新构建OOM、0完整逆 | [原终态](../reports/closeout_001/repair_001/v3_tangent_probe_001/RESULT.json)、[逐格复核](../reports/closeout_001/repair_001/v3_tangent_probe_001/NATIVE_PROBE_ANALYSIS.json) | 33.14 GiB请求超过8 GiB；确切底层内核未定位；旧步未存，直接解差不可得；不作生产收益结论 |
| 新C1标定有明确增量预算 | 661×7节点；48769温度点及解析平场界 | [14项标定合同](../reports/closeout_001/repair_001/calibration_v003/CALIBRATION_CONTRACTS.json) | 改变插值及anchor导数，显式新目标；绝对LUT误差另报 |
| C1四格通过登记数值判据 | 2开发raw×2初值，4完整逆，4×65536评分像素 | [原终态](../reports/closeout_001/repair_001/quick_v002/QUICK_RESULT.json)、[独立全域重评分](../reports/closeout_001/repair_001/quick_v002/NATIVE_C1_ANALYSIS.json)、[独立端点复算](../reports/closeout_001/repair_001/quick_v002_analysis/FINAL_REVIEW_RECEIPT.json) | 三阶段/两平台；最终gtol1e-6，未执行1e-7；原stage1/2只作日志核验；不作最终选型 |
| C1两初值共享物理正则 | 同上述4格，逐raw比较共同anchor合同 | [全域重评分及初值比较](../reports/closeout_001/repair_001/quick_v002/NATIVE_C1_ANALYSIS.json) | 版本内目标相同；不把新C1与旧PLUT称同一物理目标 |
| relay响应差异造成平场色温偏差 | 平坦3100 K无噪反例与各载频响应 | [根因报告](../reports/closeout_001/root_cause_team_001/ROOT_CAUSE_REPORT.zh-CN.md) | 旧nominal按已登记失配压力解释；新匹配24母场景未执行 |
| 原MC记录保留，区间方案需更正 | 12母场景/896完整MC，原几何角色混杂 | [科学终态](../reports/closeout_001/SCIENCE_RESULT.json)、[更正协议](../reports/closeout_001/repair_001/REPAIRED_SCIENTIFIC_PROTOCOL.json) | 条件bootstrap不是独立覆盖，oracle回顾诊断不是部署区间 |
| 新方法检测效能尚待独立校准 | 计划200-null校准+200-null评价 | [研究协议](../reports/closeout_001/STUDY_PROTOCOL.yaml) | 四格未附新阈值；不采用legacy阈值得出检出结论 |
| 封存集用于最终评价 | 计划64main+36signed母场景，512+180输入 | [研究协议](../reports/closeout_001/STUDY_PROTOCOL.yaml) | 方法/阈值/计划未全部冻结，payload保持未开 |
| 光学复用、光谱VP与图像区间均有先例 | 原文与获取层级逐条登记 | [原始文献矩阵](../reports/closeout_001/LITERATURE_MATRIX.md) | 投稿前补齐温度/发射率论文受限全文；不主张通用结构首创 |
