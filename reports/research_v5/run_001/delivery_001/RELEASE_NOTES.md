# v5 决策性研究：物理负结果与完整科学证据

已执行任务书全部适用的预登记诊断；当前 v5 两步物理估计未通过，按严格停止规则关闭 CNN、候选和最终验证支线。

- Oracle：180输入、720四级结果、84×4基线；无噪O0正常±RMSE0.005612e。
- 开发：新36母场景、540输入、4320网格行；独立null96+96、MC12×8、Hessian8探针、压力36×5全部记录。
- 名义色温RMSE28.540K（目标20K）；正常正负模板RMSE64.629/63.761e，对照17.195/17.223e；两符号投影也失败。
- 色温MC覆盖89.729%未达90%；Hessian0/8收敛；增益/发射率失配造成大误差但当前零拒绝。weak阈值交叉不能作为正确弱结构检出。
- 原float32六项失败、高精度补充、原作业超时与单独预算补充、三例MC重复日志缺失均保留。
- CNN更新0，无v5 checkpoint；64原始最终与36signed确认母场景继续封存，最终性能未知。

[一页六问题决策](https://github.com/Aggressivecyber/single-frame-spectral-thermometry/blob/research/v5-decision-20261013/reports/research_v5/run_001/V5_SCIENTIFIC_DECISION.md) · [完整报告](https://github.com/Aggressivecyber/single-frame-spectral-thermometry/blob/research/v5-decision-20261013/reports/research_v5/run_001/ROOT_FINAL_REPORT.md) · [入口与完成审计](https://github.com/Aggressivecyber/single-frame-spectral-thermometry/tree/research/v5-decision-20261013/research_v5/delivery)

源码与研究报告提交：`598db69a70595c0e8aae0e2b935bdc75eb29cdaf`。原平台和冻结实现保持不变。数据分卷含全部2998个实际v5科学载荷文件，总原始文件大小8138672663bytes；各ZIP可独立解压至仓库根目录，路径原样保留。`V5_PACKAGE_MANIFEST.json`列明每卷内容和哈希，`SHA256SUMS.txt`提供本地生成包哈希。

这是合成平台上的模糊后连续谱色温与signed模板偏离研究；不证明真实气体温度、实测AlO浓度、可靠部署或10kHz性能。上传以成功命令回执为准，不进行重复下载核对。
