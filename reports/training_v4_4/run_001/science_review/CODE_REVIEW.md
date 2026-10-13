# v4.4 静态代码与输入审查

审查对象为当前 `train.py`、`evaluate.py`、`protocol.py`，并沿调用链读取模型、缓存、旧评估和checkpoint保存函数。未执行模型、GPU测试、训练或最终测试；只额外执行了标准库磁盘哈希审计。训练源码SHA256为 `25391d58b8237fd67ff10e3d282568f8f9a7e5b8c5549034e3d0b196d2dd545c`，本审查不修改正在运行的源码。

**结论：当前成功执行路径符合已冻结的单变量对照和新旧验收规则；发现一项重载审计失败路径的P2回执错误。** 该错误不改变正常训练的预测或验收数值，但触发时不得使用生成的中断checkpoint恢复训练。

1. **[P2] 历史模型可能与500步优化器混存。** `train.py:103–108`依次向同一个model重载0/100/…/500参数，只检查checkpoint内的Adam步数，没有把对应优化器状态载入当前optimizer。当前optimizer仍在500步。如果重载审计在加载某个历史权重后失败，`train.py:129–132`无条件把该model、500步optimizer和`completed=500`写入`interrupted.pt`。例如已加载100步模型后哈希断言失败，就可产生标成500步的混合状态。应把重载审计与训练中断分开：使用独立审计model，或失败时仅保留错误记录和既有不可变checkpoint；若另存恢复文件，必须证明模型、优化器及update来自同一checkpoint。主任务已确认当前运行不热改代码；若发生此类异常，将该interrupted标为INVALID、禁止恢复，正常完成后再修未来失败分支。

|核对项|静态结论|
|---|---|
|模型|`FixedContinuumCNN`继承原ResidualScaledCNN，最终cat缓存C与原缩放signed字段，保持beta并重新计算完整物理输出；内部连续谱仍可训练，未新增truth/condition输入。|
|初始化与优化|旧v4/update2000参数、空Adam、seed4101、lr3e-5、累积4、clip1、safe1、原draws一致；最多500更新，100间隔全量评估。|
|抽样|draw前缀在运行入口核对哈希，逐update的case/raw/cache/input-kind/signed元信息与原safe1逐项相等，且全部训练split；没有使用negative_balanced替代抽样。|
|旧门禁|直接调用旧evaluate与原16项函数，完整保留原eligible及保护分组；没有把新screen写入旧eligible。|
|新门禁|四个signed含噪condition各断言6帧；MSE阈值为1.05×物理+0.25 e²，两weak分别pooled L2<1；776唯一case，有限MSE、无效比例0，温度校正逐帧为0。失败不从均值排除。|
|字段恒等|评估observer要求最终C、signed状态和T精确一致，共776+16次；零更新程序还与原端点签名指标逐例一致。这里只核对实现，没有重跑这些预测。|
|冻结来源|协议中6个源码哈希均匹配；本轮启动回执额外包含train.py，7个运行源码及其source_snapshot均与启动哈希一致。|

**选择规则以实际预注册协议为准。** 本轮登记的是“全部门禁通过后，最小主弱L2；并列比较正常RMSE，再比较更早checkpoint”，与此前REVIEW/decision中建议的main validation loss排序不同。该区别不是实现错误；我不修改预注册选择规则。这三个文件负责训练和评分，尚未执行最终候选选择；交付阶段须按实际协议生成单一候选和排序回执，不能默认取500末点或混合指标。

**输入回执已补齐。** `train.py:32–35`没有主动复验协议内manifest/validation-ID哈希，沿用的`cache.load`也没有逐次重验标签与缓存容器字节。因此新增独立审计脚本 [verify_inputs_readonly.py](verify_inputs_readonly.py)，按本轮500 draws+776 validation实际使用集合重新读取磁盘哈希，并同旧冻结输入回执交叉匹配，结果见 [input_integrity_receipt.json](input_integrity_receipt.json)。这解决了当前磁盘输入证据缺口，但不能追溯宣称runner启动前已经实施该检查。

核验结果：1183个唯一训练case、776个验证case，共1959个；6888个去重raw/label/cache/cache-metadata文件、1959个物理score文件及1个manifest，共8848个文件、3,798,057,806字节全部匹配，耗时46.094秒。manifest哈希、validation-ID哈希、500×4抽样哈希、物理score自身哈希、缓存物理/标定/求解器元信息及实际cache_location均通过，漂移0。回执已重新打开检查。

输入审计回执SHA256：`ab1f50159ce8c015e149ebc0ae2f8ee7373a47551b185f7b972892e73df6961d`。完整问题及处置状态见 [code_review_issues.json](code_review_issues.json)。
