# 原始输入回放与单帧计时

三个既有开发集示例均从 raw 重新执行物理反演，再运行同一个 v4.6 第500次更新权重。在线得到的 state/beta 与历史缓存逐值一致；14个输出数组也与缓存路径逐值一致。另将32文件推理包解压至仓库外目录，在隔离的 Python 进程中复跑一个负normal含噪示例，同样逐值一致，原仓库模块导入数为0。

|既有DEV示例|物理反演/秒|CNN/秒|state/beta 与14个输出数组|
|---|---:|---:|---|
|negative_normal noise|8.8355|0.1951|全部精确相等|
|weak noise0|7.6893|0.0412|全部精确相等|
|blank quantization|7.4581|0.0526|全部精确相等|

计时来自本机 CUDA 单次同步调用。模型加载、主机/设备输入传输、诊断和文件保存均不计入表格；此处未做预热吞吐或并发吞吐测试。推理分为固定物理反演和后置CNN，训练时复用缓存并不能消除实际 raw 推理所需的反演。当前结果不构成10 kHz实时测温验收。

这些检查验证了软件通路及搬移后的数值一致性。三个示例全部来自此前反复使用的开发集，不提供独立精度、实验真实性或仪器校准证据。该500步权重未通过联合开发条件，打包角色仍为诊断端点。

完整回执：

- [三例回放](../training_v4_6/run_001/native_replay/execution/receipt.json)
- [回放时间预算核验](../training_v4_6/run_001/native_replay/execution/watchdog_receipt.json)
- [独立目录迁移](../../outputs/training_v4_6/run_001/delivery/portability_receipt.json)
- [推理包文件与ZIP核验](../../outputs/training_v4_6/run_001/delivery/inference_bundle_build_receipt.json)
