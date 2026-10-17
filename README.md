# Single-frame spectral thermometry

## 当前开发：TV 锚定分组辐射保真展开网络

新实现[TV-RRU](research_joint_imaging_v2/README.md)以数值收敛的TV为固定起点，采用热通道强度—比例坐标、按可见波段隔离的共享NAF式U形先验，并在三个共享阶段中保留原TV正则。分组模型823300参数，普通全六路对照826326参数。

已完成原生CPU/GPU梯度与固定点检查、四组方法的短程端到端集成，以及128步固定小批量拟合（损失6.33降至2.99）。这些是软件验收。[集成证据](reports/joint_imaging_v2_integration_001/EXIT.json)与[float32步长诊断](reports/joint_imaging_v2_fd_diagnostic_001/RESULTS.json)保留；首个开发批次按64/16/8/16个新母场景、2496幅raw、四组各6000次更新在登记服务器执行，未见场景结果待评价。

服务器重启后已[恢复剩余对照](reports/joint_imaging_v2_recovery_002/RECOVERY_REVIEW_zh.md)：全部2496幅raw/TV缓存及完成6000步的普通后处理组直接复用。旧分组组停在4182步，缺失优化器状态，原记录保留，仅该未完成组按冻结种子重新训练。新增每200步原子保存完整训练状态；原生GPU同一中途检查点恢复测试通过。剩余三组和新留出评价仍在执行，不代表新方法已超过TV。

## 前一轮六通道测温与特征发光联合成像

2026-10-17完成首轮六通道合成对照：从同一视场、同一传感器区域上的一幅重叠编码图像恢复六幅滤光通道图像。700、800、905 nm用于连续谱有效色温拟合，431.5、486、516.5 nm用于CH*、AlO、C₂*相关波段的超额发光读出。

共44个独立合成母场景、1056幅原始观测及6次网络训练。预登记主条件下，噪声加权TV的色温RMSE为 **54.59 K**，混合网络三个种子为 **81.79–85.59 K**，且存在弱发光幅值低估与跨通道误报。**本轮混合原型未通过继续线；TV保留为后续研究参考。** 所选TV也未达到50 K及所有通道5%相对L2的绝对筛查线。

- [当前方案与文献依据](research_joint_imaging/README.md) · [冻结配置](research_joint_imaging/protocol.json) · [源码登记](research_joint_imaging/REGISTRATION.json)
- [完整中文结果](reports/joint_imaging_001/RESULTS_zh.md) · [方法对比图](reports/joint_imaging_001/review_001/method_comparison.png) · [联合读出图](reports/joint_imaging_001/review_001/joint_products.png)
- [重放与指标核验](reports/joint_imaging_001/review_001/VERIFICATION.json) · [完成审计](reports/joint_imaging_001/COMPLETION_AUDIT.json)

本分支包含当前源码、原始执行源码快照、配置、统计报告、图像和六个网络检查点（`reports/joint_imaging_001/cnn_{post,dc}_{701,702,703}.pt`）。大体积原始ADC、数据集及预测数组保留在项目E盘目录和计算服务器，其SHA256与大小记录在[输入清单](reports/joint_imaging_001/review_001/INPUT_MANIFEST.json)。后续TV锚定神经修正的开发状态见本页顶部。

结果对应理想滤片、已知空间算子与相机噪声模型下的合成工程试验；物理读出为有效色温与特征波段发光。实际滤片响应、物种交叉响应、曝光性能及硬件测温仍需标定与实测。

## 早期七通道推理原型

以下`infer.py`入口和默认权重对应早期七通道版本；当前六通道实验入口与配置位于`research_joint_imaging/`。

七通道单帧测温原型：物理参数展开 + 共享 CNN。公开代码和当前已选权重，支持 CPU/CUDA 推理。

A small physics-unfolded neural reconstruction prototype for seven spatially overlapping spectral channels in one common-field sensor image. The published model has **5,797 trainable parameters** and **six shared stages**. Each stage applies a matrix-free damped Gauss–Newton update before and after a bounded CNN correction.

**该版本扩展测试：**同一份权重完成 768 次推理。由 72 帧真实火焰图像形态构建的测试中，正常信号空间相对 L2 为 **27.6%–67.1%**，弱信号为 **51.8%–227.2%**。六组正常信号都未达到 25% 空间误差目标。真实部分是图像形态，编码测量与温度仍是模拟设定，当前不报告实测测温精度。

完整结果：[中文报告](reports/expanded_evaluation_001/RESULTS_zh.md) / [English report](reports/expanded_evaluation_001/RESULTS.md)。Only this selected model was evaluated; weights stayed fixed and there were no contribution controls. Two of 20 synthetic stress families passed all internal screens.

## Run

Use Python 3.10 or newer with PyTorch, NumPy and Matplotlib. Install a PyTorch build appropriate for your CPU or CUDA device, then:

```bash
python -m pip install -r requirements.txt
python -B infer.py --synthetic-demo --output outputs/demo
python -B infer.py --raw outputs/demo/input_raw_DN.npy --output outputs/replay
# GPU:
python -B infer.py --synthetic-demo --device cuda --output outputs/demo_cuda
```

The default checkpoint is included in this repository. Output directories must be new. Caches and temporary files stay inside `_runtime/`; set `THERMOMETRY_RUNTIME` to choose another location. On Windows, clone onto the data drive you wish to use.

`prediction.npz` contains continuum color temperature, signed template coefficients, seven reconstructed band images, raw reprojection residuals, valid masks and saturation masks. Raw input is a 256×256 DN image (or a batch with shape B×1×256×256), using the exact acquisition configuration in the checkpoint.

## Architecture and configuration

- Nominal channels: **430, 450, 486, 513, 535, 700, 905 nm**.
- One coded raw frame; all channels share the same field of view and image area.
- Five physical fields: normalized temperature, log continuum amplitude and three template slots; only the third slot is active in this checkpoint. A three-coefficient background is estimated separately.
- CNN: 19→16→16 shared circular convolutions; continuum head 16→2, emission head 16→3. Corrections are bounded and shared across six stages.
- Poisson and read noise, 12-bit ADC and synthetic optical transfer functions are included in the simulator.
- Inference receives raw DN only. Synthetic truth enters evaluation or supervision, never reconstruction.

The source physics, solver, network, configuration and original scene generator are preserved from the selected prototype. The public runtime and inference CLI are portable wrappers. This release keeps the selected model before the later ECA/multiscale experiments.

## Weight provenance and results

`checkpoints/selected_checkpoint.pt` is the exact selected `low_lr_control` checkpoint, after 160 additional Adam updates at learning rate 3e-5 from the earlier local pilot. It includes the state dictionary, configuration and optimizer state. Selection used validation data, with the new test data held out.

SHA256:

```text
d777888c53b086118128a824da007e1cba471f185ad9f8a973eb68c5edb434f1
```

The original 24-case synthetic test (8 signal, 8 clean null, 8 null with a +5% continuum error at 486 nm) gave:

| Metric | Selected model | Internal prototype screen |
|---|---:|---:|
| Continuum color-temperature RMSE | 9.911 K | ≤75 K |
| Template coefficient relative L2 | 0.287525 | ≤0.25 |
| Template integral bias | −8.822% | absolute ≤10% |
| Clean-null positive-pixel rate | 0.0109% | ≤1% |
| Background-mismatch null positive-pixel rate | 0.9714% | ≤1% |
| Valid coverage | 100% | ≥98% |
| Saturated pixels | 0 | 0 |

The spatial L2 screen is still failed. These thresholds are internal engineering screens, not scientific or hardware acceptance criteria. Expanded scenario tests are recorded in `reports/` with unchanged weights and explicit acquisition perturbations.

## Expanded evaluation

Only the published selected model is evaluated. No optimizer steps or contribution/ablation controls are used. Twenty registered synthetic families cover morphology, signal strength, cold/hot continuum, temperature fronts, saturation, background errors and acquisition calibration/noise mismatches:

```bash
python -B scenario_suite.py --device cuda --output outputs/expanded_001
```

A second suite uses 72 experimental CH*/OH* image shapes from three public acquisition datasets, in four constructed conditions each: nominal signal, weak signal, clean null and a structured background. Download the five original files listed by `prepare_real_shapes.py` from their attributed sources, then:

```bash
python -m pip install -r requirements-data.txt
python -B prepare_real_shapes.py --data-root data --output outputs/real_shapes
python -B real_shape_test.py --shapes outputs/real_shapes --device cuda --output outputs/real_shape_test
```

The exact processed shapes used in the recorded run are included, so they can be tested directly without redownloading the original archives:

```bash
python -B real_shape_test.py --shapes reports/expanded_evaluation_001/experimental_shapes --device cuda --output outputs/real_shape_replay
```

These are **experimental-image-derived simulations**. Native image morphology is real; continuum temperature, spectrum, coefficient scale, encoding and photon noise are constructed. Error is against the constructed post-blur target, not measured gas temperature or AlO. Different channel images are not treated as synchronized truth. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for sources, transformations and image licenses.

All frames are frozen and hashed before inference. Reports retain per-case errors, minimum coverage, null positives, saturation, nominal and weak-signal results, and example signed maps. The default GPU budget is 900 seconds per evaluation phase; CPU runs require an explicitly appropriate budget.

## Scope

**This is a synthetic engineering prototype.** The response matrix, flat QE, filters, photon scale, encoding patterns and camera settings are software assumptions. The “AlO” slot name denotes a synthetic numerical template; its response has not been measured or derived from a real line list. CH and C2 slots are inactive. Results do not establish real flame/gas temperature, species concentration, multi-species separation, camera throughput or hardware performance.

The output temperature is effective **continuum color temperature after the object blur**. Negative template estimates are retained. All spatial filtering and CNN padding are periodic, so boundary tests need that interpretation. Calibrated response transfer and independent measured spectra remain future work.

See [MODEL_CARD.md](MODEL_CARD.md) and [LICENSE](LICENSE).
