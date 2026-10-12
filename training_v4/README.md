# 物理反演后 CNN 校正器 v4

本轮基于 `9a7ac7e`，使用 5,797 参数的 19→16→16 共享 CNN 和两个残差头，
对固定物理反演缓存做六次有界校正。阶段间重算原物理特征；CNN 输出通过原
`physics.py` / `object_blur` 生成 postblur 温度、模板、波段和 raw 预测。
这是物理反演后的校正训练，不是完整展开训练。

数值流程、数据/缓存 hash、真实曲线、checkpoint 和逐例对照见
[TRAIN_V4_REPORT](../reports/training_v4/run_001/TRAIN_V4_REPORT.md)。
原网络、权重、repair_v3 和旧报告保持不变。

全部数据与缓存放在 E/D checkout 的 ignored `outputs/training_v4/run_001`。
标签在独立的 `labels` 目录，缓存不接收真值；更换 raw 噪声必须更换匹配缓存。
最终测试母场景已登记，本轮不生成或评估。

使用本机 GPU PyTorch 环境，先将 TMP/TEMP/CUDA 缓存指向 E/D，再按顺序执行：

```powershell
python -B -m training_v4.run --phase data
python -B -m pytest training_v4/tests.py
python -B -m training_v4.run --phase select
python -B -m training_v4.run --phase overfit
python -B -m training_v4.run --phase train --seed 101
python -B -m training_v4.run --phase compare --seed 101
```

报告环境需要 matplotlib/Pillow：

```powershell
python -B -m training_v4.run --phase report
```

正式训练为 Adam lr=1e−4、batch=1、累积4、clip=1，2,000 次 optimizer updates，
每100次完整验证。零残差头的首次主干梯度为零符合链式求导；活跃头梯度和
实际参数更新由测试及 8 样本过拟合验证。诊断权重不作为正式训练初始化。

只有明确验证收益且弱信号胜过 L2=1 零估计、保护门禁通过后，才增加训练种子。
所有候选 checkpoint、失败记录和固定 >60 e 检测指标保留。

全新目录复现用以下入口，它建立 `9a7ac7e` 的独立 E/D worktree、复制 v4
代码和精确匹配的 16 项门禁输入，再执行全部训练流程。需要解释器具备
CUDA PyTorch、NumPy、pytest；报告解释器需要 PyTorch、matplotlib、Pillow。
此脚本会实际重新训练，请使用新的目录以保留当前结果。

```powershell
.\training_v4\reproduce.ps1 -Worktree E:\CodexWork\thermometry\v4_reproduction_001
```

原目录完成比较后，先 `python -B -m training_v4.verify_delivery`，再生成报告。
发布的 `checkpoints/seed_101_validation_selected.pt` 与
`checkpoints/seed_101_final_2000.pt` 区分验证选中版本和完成 2,000 次更新的末次版本。
门禁失败的末次版本用于诊断，不因此变成可接受部署版本。
