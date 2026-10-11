# Single-frame spectral thermometry

七通道单帧测温原型：物理参数展开 + 共享 CNN。公开代码和当前已选权重，支持 CPU/CUDA 推理。

A small physics-unfolded neural reconstruction prototype for seven spatially overlapping spectral channels in one common-field sensor image. The published model has **5,797 trainable parameters** and **six shared stages**. Each stage applies a matrix-free damped Gauss–Newton update before and after a bounded CNN correction.

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

The spatial L2 screen is still failed. These thresholds are internal engineering screens, not scientific or hardware acceptance criteria. More extensive scenario tests will be recorded in `reports/` with unchanged weights and explicit acquisition perturbations.

## Scope

**This is a synthetic engineering prototype.** The response matrix, flat QE, filters, photon scale, encoding patterns and camera settings are software assumptions. The “AlO” slot name denotes a synthetic numerical template; its response has not been measured or derived from a real line list. CH and C2 slots are inactive. Results do not establish real flame/gas temperature, species concentration, multi-species separation, camera throughput or hardware performance.

The output temperature is effective **continuum color temperature after the object blur**. Negative template estimates are retained. All spatial filtering and CNN padding are periodic, so boundary tests need that interpretation. Calibrated response transfer and independent measured spectra remain future work.

See [MODEL_CARD.md](MODEL_CARD.md) and [LICENSE](LICENSE).
