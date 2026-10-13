# v4.2 恒等起点短程对照

继承 v4 update2000 的共享主干，仅将 continuum_head 与 emission_head 的权重、偏置清零。
模型仍为 5797 参数、19 通道、六次共享校正，物理骨干、残差特征、损失、seed、lr=3e-5、累积4、clip=1均与 v4.1 safe1 相同。最多500次更新，不自动扩跑。

所需输入是已有 `reports/training_v4_1/run_001/data_manifest.json`、冻结抽样序列和对应 `outputs/training_v4*/run_001` 开发数据、标签、物理缓存与评分sidecar。最终测试不参与。大数组未重复提交。

在含这些输入的新E/D工作副本上运行；已有训练曲线会被拒绝覆盖。保留历史 E/D 数据目录并复制本代码到工作副本即可，无需重算物理缓存。

```powershell
$GpuPython='E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
$ReportPython='C:\Users\10629\AppData\Local\Programs\Python\Python313\python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:TMP=Join-Path (Get-Location) '_runtime\tmp'
$env:TEMP=$env:TMP
$env:CUDA_CACHE_PATH=Join-Path (Get-Location) '_runtime\cuda'
$env:PYTHONPATH=Join-Path (Get-Location) '_runtime\audit_deps'
& $GpuPython -B -m pytest training_v4_2/tests.py -q --junitxml=reports/training_v4_2/run_001/tests.xml
powershell -NoProfile -ExecutionPolicy Bypass -File training_v4_2/run.ps1 -GpuPython $GpuPython
& $GpuPython -B -m training_v4_2.verify
& $GpuPython -B -m training_v4_2.batch_gradients
& $ReportPython -B -m training_v4_2.analyze
```

报告目录为 `reports/training_v4_2/run_001/`，权重为 `training_v4_2/checkpoints/`。C盘路径仅用于读取既有解释器，生成物和临时文件全部写入工作副本的E/D盘。

- 阶段诊断：144个预先按元数据分层选择的验证帧，记录物理起点与原六次校正；末阶段逐帧核对原forward。
- 梯度诊断：固定12个训练探针，分别记录主/保护损失的梯度，不执行optimizer step。
- 正式训练：复用safe1前500次相同抽样，实际训练500次；每100次保存权重及776帧完整验证和原16项门禁。
- 对照：旧safe1前500次、zero、old_B4、固定物理、v4 update100、v4 update2000、验证选参传统基线，仅在raw/label/cache hash完全匹配时复用。
- 新初始化输出为固定物理，含噪能力仍需实测；不得把原16/16或保护通过单独当作模型合格。

该试验只检验初始化影响，不改变网络结构，不宣称是完整展开训练。
