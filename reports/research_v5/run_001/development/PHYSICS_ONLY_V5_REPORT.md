# Physics-only v5 development decision

Physics candidate eligible: **False**. Stage-1 no-hard-stop: **True**.

Frozen alpha_g=1.0; traditional reference=v3_physics.
One continuum fit per development raw preceded four signed S/beta solves. All four old methods received that same raw. No S feedback changed the continuum.

Main field errors use postblur template and continuum color on the complete fixed truth domain; abstained pixels remain in errors. Synthetic negative template coefficients are deviations, not concentration.

| Gate | Result |
|---|---|
| signed_normal_positive | False |
| signed_normal_negative | False |
| selection_feasible | False |
| nominal_color | False |
| valid_coverage | True |
| all_selected_noisy_CG_converged | True |
| null_heldout_FP | True |
| MC_nominal_coverage | False |

Nominal color RMSE: 28.5399 K; valid coverage: 1.
Normal positive RMSE/reference factor: 3.75867.
Normal negative RMSE/reference factor: 3.70212.

Independent null z threshold: 48.162; heldout frame FP: 0.0208333; pixel FP: 1.09673e-05.
Threshold is the .95 higher quantile of the full-FOV maximum absolute standardized postblur template over calibration frames. Null evaluation uses disjoint mothers and never retunes it.

The standardized scale is an observed-raw stationary postblur noise proxy. It omits continuum and beta propagation and provides no confidence interval or instrument guarantee. MC uncertainty is reported separately.

MC: 12 predeclared normal +/- cases, 8 repeats per case, rerunning raw-only continuum+signed inversion every repeat. Parent mean full-domain signed interval coverage=0.903969; color interval coverage=0.897292.
Small-sample normal intervals have no guaranteed coverage. Truth enters only the external coverage calculation. Pixel independence is never used for uncertainty on reported scene performance.

Hessian: eight fixed Rademacher probes on one selected normal case. Diagonal estimates are in packed latent S/beta coordinates; they are not postblur sigma or a full-image exact CRLB.
Hessian all probes converged: False.

Calibration stress is a separate fixed gain/emissivity experiment. Nominal acceptance is not extended to those stresses. Reference/nominal color errors, validity and rejection are retained in calibration_stress_records.jsonl.
Stress records: 180; no stress result was used to choose alpha or threshold.

records.jsonl retains per-case bias, fixed-domain error, true CG histories, regularization scales, latent gradient energy and route latency. paired_statistics.json resamples whole mothers 2000 times with seed 53130, preserving signed conditions and noise repeats.

Confirmation and final arrays remain sealed. An unsuccessful physics gate forbids a learned candidate and preserves a negative result.

## Execution recovery and task compliance

The original21600-second invocation ended FAILED/TimeoutError after 21626.92801609999 seconds. A separately registered5400-second diagnostic completion reused the complete4320-row grid, fixed selection,96+96 null rows and540 selected scores, without rerunning them. Three saved MC interval arrays were externally scored; their per-repeat logs were not persisted. Only the remaining9 MC cases, original8 Hessian probes and36 stress inputs were executed. All preexisting evidence and the original failure are preserved. This is an explicit execution-budget deviation, not a successful run within the original budget.

The additional root Task4B projection review is in ../execution_recovery_001/scientific_stop.json. Both signed normal projection checks fail, as do the registered color and normal-template RMSE requirements. Alpha1 is an infeasible diagnostic fallback. CNN remains untested; sealed final accuracy is unknown.
