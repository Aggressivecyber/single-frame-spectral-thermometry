# v4.1 保护性校正训练

固定62538a8的物理骨干、原5797参数/19通道/六共享校正模型。主试验safe1，
失败时最多追加同初始化/同抽样的safe5；满足明确诊断触发条件才加一次残差幅度特征消融。
训练主监督使用真实合成postblur标签，保护项约束可观测温度/模板MSE的增量，
不约束latent状态，不向推理传入场景类型或真值。所有数据、缓存、临时文件放E/D。

本机GPU解释器：
`E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe`。
需要CUDA PyTorch、NumPy和pytest；CPU报告环境另需matplotlib/Pillow。

在新的E/D checkout执行，保留v4忽略目录内的冻结raw和匹配物理缓存。
将TMP/TEMP/CUDA_CACHE_PATH/PYTHONDONTWRITEBYTECODE指向E/D并设置后依次执行：

已有本项目冻结数据时，可用一条命令创建基于62538a8的全新E/D复现目录，
复制匹配的v4 raw/标签/缓存与原16项fixture，再生成新保护数据并实际训练：

```powershell
& .\training_v4_1\reproduce.ps1 -Destination 'E:\CodexWork\thermometry\v4_1_reproduction_001' -Run
```

`-DataSource`可指定保存旧忽略目录的E/D仓库。最终测试raw不复制；已存在的目标目录
会拒绝覆盖。公开Git仓库包含报告、配置、hash与小checkpoint，大型冻结raw/缓存需
在已有归档中提供，脚本会检查它们。仅省略`-Run`可先准备目录。

```powershell
python -B -c "from training_v4_1.data import generate; generate()"
python -B -m pytest training_v4_1/tests.py
python -B -m training_v4_1.prepare_cache
python -B -m training_v4_1.train --name safe1 --safe-weight 1
python -B -m training_v4_1.controls
python -B -m training_v4_1.tikhonov
python -B -m training_v4_1.compare
python -B -m training_v4_1.verify
python -B -m training_v4_1.report
```

prepare_cache可在一个终端运行，另一个终端开始safe1；训练只读取已原子写入的匹配缓存。
先后运行也可。禁止覆盖已完成训练试验的目录。独立传统解调lambda与物理骨干正则系数
是不同量；最终测试与旧954项均不运行。最终结论见
`reports/training_v4_1/run_001/TRAIN_V4_1_REPORT.md`。

外部中断时，先重新启动prepare_cache，再运行
`python -B -m training_v4_1.resume --name safe1`。它恢复最近已验证checkpoint的完整Adam状态，
保留原曲线和部分验证，重算未持久化更新并核对损失；不会再次重置Adam。
`finish_pipeline.ps1`可等待safe1完成，依次执行有限对照、解调选参、配对、测试、重载和报告。
