# v4.3 本地有限对照：负模板抽样平衡

本轮是单视场重叠输入的物理反演后CNN校正；5797参数、19通道、六次共享校正。只改变每次训练的第二个含噪抽样位置，交替使用已有negative_weak/negative_normal训练池；每池12帧，种子4101打乱轮循。其余三个抽样位置与residual_scaled逐例相同；同样从旧v4/update2000初始化，重置Adam，冻结系数、物理、safe1损失、lr3e-5、累积4、clip1和500次预算。

新结果仅在E盘本地保存，未推送；最终测试未打开。每个checkpoint独立验收，不合并不同权重的优势。

|更新|原检查/16|保护/96|主弱L2|主正常RMSE e|含噪温度通过|联合合格|同更新residual_scaled弱L2|
|---:|---:|---:|---:|---:|---|---|---:|
|0|16|96|0.686383|9.5930|False|False|初始权重为旧v4/update2000|
|100|16|96|0.937880|11.3904|False|False|0.663912|
|200|16|96|0.997274|12.3609|False|False|0.691751|
|300|16|96|1.068555|13.1747|False|False|0.681369|
|400|16|96|0.981987|14.2968|False|False|0.683959|
|500|16|96|0.975068|13.6784|False|False|0.682292|

联合合格checkpoint：无。

![各checkpoint与固定系数](comparison.png)

## 系数的证据边界

rho由固定物理预测与raw的加权残差计算，alpha=sigmoid(9×log(rho/0.3124117966))。所有2576个训练输入的raw残差及训练输入类型确定tau与指数；没有用truth误差数值定标。系数在CNN运行前固定，不可由CNN修改自身残差来改变。

方案设计动机受此前已查看的开发验证残差间隔启发，这是开发诊断，不能称独立验证。近乎分开的合成噪声模式不能推及真实相机；raw残差较小不保证truth误差小。负模板仍可输出负值，未使用truth、case_id、condition进行推理或屏蔽。

|输入类型|帧数|rho最小|rho中位|rho最大|alpha最小|alpha最大|
|---|---:|---:|---:|---:|---:|---:|
|continuous|120|9.57152e-06|0.0018468|0.0206594|0|2.41829e-11|
|quantization|120|0.033783|0.0622061|0.0865014|2.02179e-09|9.56439e-06|
|noise|536|0.968451|0.9791|0.986498|0.999962|0.999968|

## 温度失败组

原规则保持：每个含噪group的温度MSE不超过1.05×固定物理MSE+0.04 K²。negative_weak与negative_normal另列诊断，但没有替换原先合并的negative验收组。

|更新|辅助组|组别|帧数|色温RMSE K|物理RMSE K|原阈值 K|
|---:|---|---|---:|---:|---:|---:|
|0|True|negative|12|26.0611|15.5624|15.9479|
|100|True|negative|12|18.9670|15.5624|15.9479|
|200|True|negative|12|17.3644|15.5624|15.9479|
|300|True|negative|12|16.7494|15.5624|15.9479|
|400|True|negative|12|16.7843|15.5624|15.9479|
|500|True|negative|12|16.6774|15.5624|15.9479|

## 末次完整分层

|辅助组|条件|输入|帧数|模板RMSE e|相对L2|色温RMSE K|
|---|---|---|---:|---:|---:|---:|
|False|blank|continuous|24|0.0559|N/A|0.0684|
|False|blank|noise|128|9.2826|N/A|16.8298|
|False|blank|quantization|24|1.0587|N/A|1.5671|
|False|normal|continuous|24|1.6218|0.025988|0.7446|
|False|normal|noise|128|13.6784|0.246819|17.2096|
|False|normal|quantization|24|1.9408|0.031099|1.6985|
|False|strong|continuous|24|3.2964|0.026402|1.4757|
|False|strong|noise|128|14.3679|0.126561|17.4392|
|False|strong|quantization|24|3.4780|0.027857|2.0724|
|False|weak|continuous|24|0.2993|0.026173|0.1446|
|False|weak|noise|128|9.9548|0.975068|16.8460|
|False|weak|quantization|24|1.0873|0.095072|1.5665|
|True|negative_normal|continuous|6|3.1626|0.058353|0.7424|
|True|negative_normal|noise|6|35.2937|0.651188|17.7237|
|True|negative_normal|quantization|6|3.3342|0.061518|1.2577|
|True|negative_weak|continuous|6|0.7943|0.058618|0.2008|
|True|negative_weak|noise|6|17.8337|1.316166|15.5608|
|True|negative_weak|quantization|6|1.2896|0.095173|1.0214|
|True|positive_normal|continuous|6|3.2169|0.059354|0.7357|
|True|positive_normal|noise|6|20.1727|0.372197|16.1090|
|True|positive_normal|quantization|6|3.3743|0.062257|1.2396|
|True|positive_weak|continuous|6|0.8003|0.059067|0.1991|
|True|positive_weak|noise|6|18.1527|1.339713|15.5873|
|True|positive_weak|quantization|6|1.2864|0.094940|1.0222|

## 含噪有符号模板保真诊断

投影为sum(pred×truth)/sum(truth²)，按真值能量合并六帧；接近0提示与真值同向的信号几乎消失，理想为1。召回率沿用各帧abs(truth)支持域与固定±60 e阈值，表中为帧平均。此处不修改验收规则；联合eligible也不能替代这些分层结果。

|更新|条件|有符号投影|固定阈值召回率|
|---:|---|---:|---:|
|0|negative_weak|-0.002983|0.000000|
|0|negative_normal|0.008290|0.000000|
|100|negative_weak|0.572813|0.000000|
|100|negative_normal|0.222082|0.000000|
|200|negative_weak|0.686821|0.000000|
|200|negative_normal|0.328457|0.000000|
|300|negative_weak|0.674108|0.000000|
|300|negative_normal|0.407028|0.067211|
|400|negative_weak|0.636880|0.000034|
|400|negative_normal|0.406950|0.155565|
|500|negative_weak|0.681766|0.004388|
|500|negative_normal|0.457063|0.237787|

## 回执与诊断

500次实际更新、2000个微批次，1500个抽样位置与控制逐例相同，500个第二含噪位置使用预先冻结的新序列；6个checkpoint的Adam步数与模型参数哈希重开已核对，3个代表帧原生预测复现通过。6×776条新验证记录与冻结旧结果生成31816条配对，raw、标签和物理缓存键全匹配；旧推理重跑0次。

所有checkpoint最大无效像素比例为0。144帧六阶段未衰减proposal与实际输出共288条记录保留负模板、色温及有效比例；12个固定训练帧梯度探针没有更新权重。

|训练臂|500次中裁剪次数|梯度范数中位数|平均main loss|平均safe loss|
|---|---:|---:|---:|---:|
|negative_balanced|338|5.627708|0.082717|0.517708|
|residual_scaled|73|0.326307|0.027892|0.069170|

完整逐例、失败、梯度、proposal输出和系数记录位于本报告同目录。个别梯度冲突或改善不证明唯一原因；该结果限于一个种子、500次预算与当前合成开发分布。

复现入口：在含冻结训练数据且没有本臂已有结果的新E/D工作副本中，设置E/D缓存和临时目录，执行CUDA解释器的 `python -B -m training_v4_3.balanced_train`；只读报告生成使用 `python -B -m training_v4_3.scaled_analyze --name negative_balanced`。跨日期重跑需登记新的有界截止，不能静默沿用本次时限。

本臂只有12个训练母体的24个含噪负模板，增加的是重复暴露次数，不是独立数据量。现有开发验证母体独立于训练母体，但本次研究决策受开发结果影响；不能用这次对照替代新的独立采集或最终测试。
