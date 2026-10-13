# v4.4 固定最终连续谱对照

先对两预登记500次端点执行零更新全量审查，再完成一个同初始化、同500组抽样的训练对照。最终测试未打开；下面全部为已暴露分布上的开发结果。

|模型/阶段|更新|原16项|保护96组|主弱L2|主正常RMSE e|旧eligible|新科学筛查|
|---|---:|---:|---:|---:|---:|---|---|
|residual_scaled_final_C_fixed|0|16|96|0.682292|9.5767|True|False|
|negative_balanced_final_C_fixed|0|16|96|0.975068|13.6784|True|False|
|fixed_continuum_train|0|16|96|0.686383|9.5930|True|False|
|fixed_continuum_train|100|16|96|0.670165|9.5707|True|False|
|fixed_continuum_train|200|16|96|0.688440|9.5711|True|False|
|fixed_continuum_train|300|16|96|0.680044|9.7211|True|False|
|fixed_continuum_train|400|16|96|0.682849|9.7628|True|False|
|fixed_continuum_train|500|16|96|0.671299|9.5692|True|False|

最终连续谱字段与物理缓存逐帧完全一致；776帧色温校正严格为0。内部六阶段连续谱头仍可影响signed修正，并有梯度；不属于完整展开网络。raw和有效性由最终组合重新计算，完整分层在comparison_summary.json。

新增科学筛查在新推理前冻结：四类signed含噪组各自MSE≤1.05×物理MSE+0.25 e²，两类signed weak各自L2<1，全部776帧有限且有效。这是开发非劣性筛查，既不是独立确认也不是幅值恢复证明；弱模板峰值45e低于60e检测阈值，弱组召回仅作诊断。

|同一500次末点signed组|模板L2|模板RMSE e|开发RMSE上限e|投影|固定60e召回|通过|
|---|---:|---:|---:|---:|---:|---|
|negative_weak|0.965521|13.0826|19.9195|0.049477|0.000000|True|
|negative_normal|0.969648|52.5539|20.4189|0.031985|0.000000|False|
|positive_weak|0.689739|9.3458|20.4570|0.638348|0.004263|True|
|positive_normal|0.274818|14.8949|21.0005|1.048529|0.551812|True|

按预登记顺序筛选后的开发候选数量：0。选择记录见candidate_selection.json。
4656组与旧residual_scaled相同步数的raw/label/cache配对通过；所有旧权重和失败记录保留。训练过程、Adam步数、源代码快照与原生重放见fixed_continuum/training_receipt.json。
数据审计与科学审查分别见data_audit/AUDIT.md与science_review/REVIEW.md。独立最终评估必须先冻结同一候选及输入接口；旧final登记不覆盖专门signed条件。
