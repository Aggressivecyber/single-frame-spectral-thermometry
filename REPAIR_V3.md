# Physical solver repair v3

This additive experiment starts from commit `5f2ae63`. It preserves all original
physics, solver, CNN, checkpoint and audit_v2 files. No training or neural network
modules are added. The existing CNN remains available through the original API.

The versioned implementation is in [repair_v3/solver.py](repair_v3/solver.py).
It retains object blur, initializes latent temperature and log-amplitude against
the decoded 700/905 nm postblur bands, and then jointly updates temperature,
log-amplitude, signed active templates and background using raw measurements.
Spatial regularization and LM damping are separate; projected steps require
actual and predicted objective decrease. CG convergence failures are recorded.

The 16 physical development gates passed, including positive, weak and negative
templates, rotated fronts and flat fields. On the frozen blank front, the
continuous-electron diagnostic achieved 0.302 K color RMSE and 0.465 e template
RMSE; the quantization-only diagnostic achieved 1.106 K and 1.896 e. These are
development results under the existing synthetic engineering response model,
not measured camera performance or evidence that the CNN can be removed.

Read the complete [REPAIR_V3_REPORT.md](reports/repair_v3/run_001/REPAIR_V3_REPORT.md)
for four-combination attribution, first-step diagnostics, the original 48-frame
smoke, calibration strata, limitations and preserved negative findings.

## Raw-only inference

Use an E/D checkout and configure temporary/cache directories on E/D before
starting Python. Import `runtime` before numerical libraries. For a finite
256 x 256 raw DN array from the configured 12-bit camera model:

```python
import runtime
import numpy as np
from infer import load_model
from evaluation_v2.common import MeasurementInput
from repair_v3.infer import PhysicalReconstructor

with np.load('E:/data/measurement.npz', allow_pickle=False) as source:
    raw_dn = source['raw_dn'].copy()
physics = load_model(device='cuda').physics
result = PhysicalReconstructor(physics).reconstruct(MeasurementInput(raw_dn))
if result['status'] != 'SUCCESS':
    raise RuntimeError(result['status'])
np.savez_compressed('E:/data/prediction_v3.npz', **result['prediction'])
```

Inference receives raw DN only. It does not receive temperature, template truth,
scene metadata or scoring masks. Whole-frame ADC saturation is rejected. Returned
`temperature_K` is observed-domain color temperature; `template_e` is the signed
post-object-blur active template. `latent_state` is saved separately. Continuous
electrons use an explicitly separate diagnostic API.

## Registered experiment

The numerical protocol, immutable source registration, fixture/input hashes,
JSONL records, tests and receipts are under `reports/repair_v3/run_001`. Array
files remain in ignored E-drive `outputs`; they are not included in Git. To
resume the exact registered experiment on another machine, copy the frozen
arrays and verify their hashes first. Re-sampling is not exact replay.

`repair_next.py` has bounded `prepare`, `gates`, `smoke` and `summarize` phases.
Smoke is allowed only after the physical gate receipt passes. Existing successful
records are reused by identity and prediction hash; the full audit_v2 is never
invoked. See the report for the baseline checkout requirement and the explicit
post-gate correction to exclude two unregistered old resume-probe records.

The next submission may compare repaired pure physics, old-CNN transfer and
same-size CNN retraining with diverse temperature fields. This submission stops
after physical repair, regression, reporting and publication.
