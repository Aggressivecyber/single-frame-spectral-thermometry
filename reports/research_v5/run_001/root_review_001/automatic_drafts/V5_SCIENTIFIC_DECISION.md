# V5 scientific decision

Audit updated 2026-10-13T16:40:15.459258+00:00. Final scientific acceptance: **False**. Final state: **sealed evaluation not opened**.

| Scientific question | Evidence-bounded answer |
|---|---|
| 1. Is continuum color temperature retained? | Nominal postblur color RMSE=28.5399 K; acceptance=False; full-domain valid coverage=1 |
| 2. Are ±S observable under this optical budget? | O0 normal continuous pooled RMSE 0.00561248 e, L2 9.94733e-05, support sign recall 1; registered gate passed=True. |
| 3. Is the bottleneck T/a, beta, or their combination? | O1: worse=False, collapse=False; O2: worse=False, collapse=False; O3: worse=False, collapse=False |
| 4. Is a CNN needed? | No completed bounded neural-control receipt; stage 1 and physical development gates are required. |
| 5. What can one frame estimate or reject? | Nominal parent-mean MC coverage=0.897292; disjoint-null frame FP=0.0208333; physical eligibility=False. |
| 6. Do current results support experimental temperature/species claims? | No. Fixed synthetic calibration, postblur continuum color and synthetic signed template deviation do not establish true gas temperature, concentration, or 10 kHz instrument performance. |

**Execution and scope.** Source protocol is frozen. Oracle case/level accounting=True (720/720); information=24/24. Fisher scope is local finite DC/Fourier/ROI modes, not an exact full-image CRLB. O0/O1/O2 remain diagnostic; deployment uses raw DN and external calibration only.

**Stopped, skipped, or incomplete stages.** Real physical development acceptance failed: signed_normal_positive, signed_normal_negative, selection_feasible, nominal_color, MC_nominal_coverage The candidate/statistics freeze alone and FINAL_ONCE_STARTED.json alone do not establish completed confirmation. Independent final acceptance requires the completed FINAL_ONCE_RECEIPT.json bound to summary and all 800 candidate case records.

**Detection boundary.** Below identifiable signal SNR, report undetectable/low-confidence and the independently evaluated simulated null false-positive rate. Do not force every weak frame below L2=1. Calibration/emissivity mismatch, finite-probe Hessian estimates, eight-repeat MC noise and distribution scope remain explicit limits. No retuning of a failed sealed evaluation is allowed.
