# 固定字段分解：连续谱与有符号模板

对两组500次末点，严格使用既定144个开发诊断帧，每帧列出四种固定组合，共1152条记录；beta不变，optimizer更新0次。P=缓存固定物理，N=CNN实际最终输出；表中组合顺序为“连续谱字段/有符号模板字段”。truth只用于评分，没有用于选择组合。

这是开发子集上的机制诊断，不是新的模型选择，也不代表776帧全量验收。physics.constrain允许signed字段在[-20,20]范围，负值没有显式截成0；训练后的压零现象应单独解释。

代码中的色温输出只由连续谱字段计算，模板输出只由signed字段计算。本次所有288个端点帧验证了这两个输出层面的独立性（最大差0）；完整N/N与同一checkpoint原生forward及已保存逐例指标均复现。raw重投影与有效性仍依赖两类字段，因此不能仅凭一类truth误差挑选混合结果。

|权重|含噪条件|C/S组合|模板RMSE e|L2|色温RMSE K|模板投影|raw RMS|最大无效比例|
|---|---|---|---:|---:|---:|---:|---:|---:|
|residual_scaled|blank|P/P|16.4516|N/A|17.6565|N/A|0.97810|0.000000|
|residual_scaled|normal|P/P|16.5243|0.38877|18.0288|0.98884|0.97935|0.000000|
|residual_scaled|strong|P/P|16.4674|0.20417|17.9393|0.99911|0.97834|0.000000|
|residual_scaled|weak|P/P|16.5927|2.17676|17.8422|1.05726|0.97981|0.000000|
|residual_scaled|negative_normal|P/P|19.9208|0.36755|15.7248|0.97154|0.97933|0.000000|
|residual_scaled|negative_weak|P/P|19.4333|1.43422|15.3983|0.88118|0.98191|0.000000|
|residual_scaled|positive_normal|P/P|20.4886|0.37803|16.0173|1.00138|0.97872|0.000000|
|residual_scaled|positive_weak|P/P|19.9580|1.47295|15.3824|1.11566|0.98030|0.000000|
|residual_scaled|blank|P/N|2.4232|N/A|17.6565|N/A|1.01480|0.000000|
|residual_scaled|normal|P/N|8.2282|0.19358|18.0288|0.97413|1.00918|0.000000|
|residual_scaled|strong|P/N|9.0466|0.11216|17.9393|0.99549|1.00629|0.000000|
|residual_scaled|weak|P/N|5.8315|0.76503|17.8422|0.48284|1.01632|0.000000|
|residual_scaled|negative_normal|P/N|51.9962|0.95936|15.7248|0.04209|1.29575|0.000000|
|residual_scaled|negative_weak|P/N|13.0758|0.96502|15.3983|0.04918|1.02836|0.000000|
|residual_scaled|positive_normal|P/N|14.9062|0.27503|16.0173|1.04293|1.00041|0.000000|
|residual_scaled|positive_weak|P/N|9.3646|0.69113|15.3824|0.54410|1.01250|0.000000|
|residual_scaled|blank|N/P|16.4516|N/A|16.0711|N/A|1.04668|0.000000|
|residual_scaled|normal|N/P|16.5243|0.38877|16.4706|0.98884|1.06294|0.000000|
|residual_scaled|strong|N/P|16.4674|0.20417|16.6439|0.99911|1.06462|0.000000|
|residual_scaled|weak|N/P|16.5927|2.17676|16.3877|1.05726|1.05639|0.000000|
|residual_scaled|negative_normal|N/P|19.9208|0.36755|29.5715|0.97154|1.38896|0.000000|
|residual_scaled|negative_weak|N/P|19.4333|1.43422|15.1074|0.88118|1.06890|0.000000|
|residual_scaled|positive_normal|N/P|20.4886|0.37803|15.4235|1.00138|1.04081|0.000000|
|residual_scaled|positive_weak|N/P|19.9580|1.47295|14.6517|1.11566|1.05209|0.000000|
|residual_scaled|blank|N/N|2.4232|N/A|16.0711|N/A|0.99615|0.000000|
|residual_scaled|normal|N/N|8.2282|0.19358|16.4706|0.97413|1.01520|0.000000|
|residual_scaled|strong|N/N|9.0466|0.11216|16.6439|0.99549|1.01897|0.000000|
|residual_scaled|weak|N/N|5.8315|0.76503|16.3877|0.48284|1.00197|0.000000|
|residual_scaled|negative_normal|N/N|51.9962|0.95936|29.5715|0.04209|1.02214|0.000000|
|residual_scaled|negative_weak|N/N|13.0758|0.96502|15.1074|0.04918|1.00280|0.000000|
|residual_scaled|positive_normal|N/N|14.9062|0.27503|15.4235|1.04293|1.02619|0.000000|
|residual_scaled|positive_weak|N/N|9.3646|0.69113|14.6517|0.54410|1.00847|0.000000|
|negative_balanced|blank|P/P|16.4516|N/A|17.6565|N/A|0.97810|0.000000|
|negative_balanced|normal|P/P|16.5243|0.38877|18.0288|0.98884|0.97935|0.000000|
|negative_balanced|strong|P/P|16.4674|0.20417|17.9393|0.99911|0.97834|0.000000|
|negative_balanced|weak|P/P|16.5927|2.17676|17.8422|1.05726|0.97981|0.000000|
|negative_balanced|negative_normal|P/P|19.9208|0.36755|15.7248|0.97154|0.97933|0.000000|
|negative_balanced|negative_weak|P/P|19.4333|1.43422|15.3983|0.88118|0.98191|0.000000|
|negative_balanced|positive_normal|P/P|20.4886|0.37803|16.0173|1.00138|0.97872|0.000000|
|negative_balanced|positive_weak|P/P|19.9580|1.47295|15.3824|1.11566|0.98030|0.000000|
|negative_balanced|blank|P/N|8.9546|N/A|17.6565|N/A|0.99253|0.000000|
|negative_balanced|normal|P/N|11.9238|0.28053|18.0288|0.88410|0.99573|0.000000|
|negative_balanced|strong|P/N|11.9737|0.14845|17.9393|0.95741|0.99338|0.000000|
|negative_balanced|weak|P/N|9.5821|1.25706|17.8422|0.80430|0.99366|0.000000|
|negative_balanced|negative_normal|P/N|35.2937|0.65119|15.7248|0.45706|1.12728|0.000000|
|negative_balanced|negative_weak|P/N|17.8337|1.31617|15.3983|0.68177|1.00448|0.000000|
|negative_balanced|positive_normal|P/N|20.1727|0.37220|16.0173|0.89670|0.99566|0.000000|
|negative_balanced|positive_weak|P/N|18.1527|1.33971|15.3824|0.77787|0.99800|0.000000|
|negative_balanced|blank|N/P|16.4516|N/A|16.7566|N/A|1.01807|0.000000|
|negative_balanced|normal|N/P|16.5243|0.38877|17.1582|0.98884|1.03765|0.000000|
|negative_balanced|strong|N/P|16.4674|0.20417|17.1774|0.99911|1.03272|0.000000|
|negative_balanced|weak|N/P|16.5927|2.17676|17.0053|1.05726|1.02374|0.000000|
|negative_balanced|negative_normal|N/P|19.9208|0.36755|17.7237|0.97154|1.30076|0.000000|
|negative_balanced|negative_weak|N/P|19.4333|1.43422|15.5608|0.88118|1.25287|0.000000|
|negative_balanced|positive_normal|N/P|20.4886|0.37803|16.1090|1.00138|1.23633|0.000000|
|negative_balanced|positive_weak|N/P|19.9580|1.47295|15.5873|1.11566|1.26243|0.000000|
|negative_balanced|blank|N/N|8.9546|N/A|16.7566|N/A|1.00296|0.000000|
|negative_balanced|normal|N/N|11.9238|0.28053|17.1582|0.88410|1.00972|0.000000|
|negative_balanced|strong|N/N|11.9737|0.14845|17.1774|0.95741|1.00762|0.000000|
|negative_balanced|weak|N/N|9.5821|1.25706|17.0053|0.80430|1.00610|0.000000|
|negative_balanced|negative_normal|N/N|35.2937|0.65119|17.7237|0.45706|1.15137|0.000000|
|negative_balanced|negative_weak|N/N|17.8337|1.31617|15.5608|0.68177|1.17564|0.000000|
|negative_balanced|positive_normal|N/N|20.1727|0.37220|16.1090|0.89670|1.15422|0.000000|
|negative_balanced|positive_weak|N/N|18.1527|1.33971|15.5873|0.77787|1.17820|0.000000|

GPU计算耗时27.73秒；288个完整N/N重放、模型参数哈希和物理缓冲区检查通过。连续、量化和含噪全部192个分层组合汇总在summary.json，完整逐例raw、投影、固定召回及有效性在per_case.jsonl。

本分解定位的是当前输出字段贡献；校正过程中共享主干、物理残差特征和多步反馈的因果关系不由这项静态替换单独确定。没有创建新候选权重、按truth路由输入或打开最终测试。
