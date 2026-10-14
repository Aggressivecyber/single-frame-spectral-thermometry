# 原始文献与本课题证据边界

核验日期：2026-10-15（新加坡时间）。下表区分正文、作者摘要与检索索引。它用于约束论文贡献表述；算法性能与本课题实验结果分别引用。获取失败的全文保留为待完成项。

| 类别与原始来源 | 已核验内容及获取层级 | 与本研究的关系 | 论文采用的表述 |
|---|---|---|---|
| Andersson等，2024，[A light-efficient and versatile multiplexing method for snapshot spectral imaging](https://www.nature.com/articles/s41598-024-66386-2)，Scientific Reports14:16116，DOI10.1038/s41598-024-66386-2 | 已读出版社正文中的FRAME原理、光路和标定部分。被动分光、空间载频与单相机叠加解码已有实现；研究涵盖测温和高速成像。 | 与本系统的光学复用思想直接重合。其通道配置、实测标定和样品与本合成七通道任务不同。 | 空间频率复用及单色相机多光谱采集为已有技术。本研究评价固定七通道任务中的色温/模板误差传播。 |
| Andersson等，2026，[Multispectral coded light for high-speed temperature imaging of aluminum combustion](https://portal.research.lu.se/en/publications/multispectral-coded-light-for-high-speed-temperature-imaging-of-a/)，Optics Express34(7):13393–13404，DOI10.1364/OE.578636 | 已核验作者机构条目、摘要、作者及出版日期2026-04-06。原文研究使用被动FRAME与单台高速单色相机开展铝滴燃烧测温。全文方法及精度统计待核验。 | 直接排除以“单相机+光谱编码+高速测温”作为独立原创贡献。 | 与该实测研究分别列出对象、光谱响应、目标量、标定及误差统计；本项目数据为合成测量。 |
| O’Leary与Rust，[Variable Projection for Nonlinear Least Squares Problems](https://www.cs.umd.edu/users/oleary/software/varpro.pdf)；出版DOI10.1007/s10589-012-9492-9 | 已读作者稿的可分离加权最小二乘与线性参数消元定义。作者稿标题页未给最终卷期；引用出版版时核对正式书目。 | 非线性连续谱、线性模板/背景的条件消元属于既有变量投影方法。 | 本文采用变量投影结构，并对当前编码算子的可观测坐标及有限求解误差进行任务验证。实现该结构本身不作为方法原创证据。 |
| [A Generalized Variable Projection Algorithm for Least Squares Problems in Atmospheric Remote Sensing](https://elib.dlr.de/195878/1/mathematics-11-02839-v2.pdf)，Mathematics11(13):2839，2023 | 已定位原始机构PDF及原始摘要；后续全文提取超时。矩阵/导数、真实光谱应用与实验表尚待核验。 | 是变量投影在光谱遥感中的直接应用比较对象。 | 在完成全文公式对照前，仅列作明确相关先例；不写“首次用于光谱反演”。 |
| [Multiple Inversion Techniques with Multispectral Pyrometry…](https://www.tandfonline.com/doi/abs/10.1080/01457632.2023.2241172)，DOI10.1080/01457632.2023.2241172 | 出版社检索摘要讨论最小二乘/Bayesian温度与发射率反演；直接页面获取失败。全文待核验。 | 温度与光谱幅度/发射率同时估计已有研究，须区分灰体先验与任意未知光谱自由度。 | 名义模型采用已知相对光谱响应；标定/发射率偏离通过预登记实验报告。 |
| Prasad等，2012，[Statistical Performance Bounds for Coded-Aperture Compressive Spectral-Polarimetric Imaging](https://opg.optica.org/abstract.cfm?URI=COSI-2012-CTu3B.1)，DOI10.1364/COSI.2012.CTu3B.1 | 已核验出版社作者、会议与摘要。作者以统计信息及Bayesian估计理论研究编码光谱偏振图像的重建界。全文无开放获取。 | 编码测量的统计性能界是已有方向；条件Oracle和有限模式分析的语义需单独说明。 | 本研究的Oracle用于机制干预，有限模式数值参考用于求解验证；全图可部署性能由完整反演实验给出。 |
| Cai、Pereyra与McEwen，[Uncertainty quantification for radio interferometric imaging: II. MAP estimation](https://arxiv.org/abs/1711.04819)，MNRAS480:4170–4182，2018，DOI10.1093/mnras/sty2015 | 已读作者提交摘要及出版元数据。研究分别处理局部区间、可信区域和结构检验。全文公式比较待补。 | 为压缩/不适定成像中的区域、像素及结构不确定度提供原始研究先例；其Bayesian对象与本项目经验重复测量区间不同。 | 分别报告点位和ROI统计量；覆盖率由独立母场景评价，避免把局部曲率逆矩阵直接命名为频率学置信区间。 |
| JCGM101:2008，[Evaluation of measurement data—Supplement1](https://www.bipm.org/documents/20126/2071204/JCGM_101_2008_E.pdf) | 已核验官方文档中分布传播、相关输入与Monte Carlo部分，含9.4及附录C.5。 | 相关输入必须通过共同随机变换传播；输入分布的来源决定输出区间的适用条件。 | 同一raw驱动整条反演；模拟标定只描述为假定分布，重复噪声副本按母场景聚类解释。 |

本项目当前可陈述的研究内容是：在明确光学目标与电子标尺下，比较连续谱来源对模板误差的影响，显式量化实际求解残差，保留强传统基线的精度取舍，并评价分辨率、光子预算、检出、区间和标定条件。开发结果目前选择冻结v3研究基线；九组联合候选的数值排除记录属于结果部分。独立和最终实验完成后，以声明—证据表约束结论强度。

论文前仍须补读：2026测温文章全文、光谱变量投影应用的完整公式、温度/发射率联合反演全文以及上表不确定度研究的详细区间对象。正文暂不作算法首创或性能全面优于原工作的声明。
