# Shared implementation interfaces (before formal source freeze)

Core functions accept a passed `SpectralPhysics` object with the frozen 256x256 synthetic configuration. The active template is index 2. S is in electrons; beta is the existing three normalized coefficients (Bscale is handled by the physics operator).

- `estimate_continuum(physics, raw_e, options=None)` returns `state`, `continuum_latent`, `continuum_postblur`, `temperature_color_K`, `valid`, `info`/`history`. It uses 700/905 initialization and spatially consistent warm fitting only; no full-spectrum joint-LM or template feedback.
- `solve_signed_linear(physics, raw_e, continuum_latent, options=None, *, fixed_beta=None)` returns `template_latent_e`, `template_postblur_e`, `beta`, `expected_e`, `info`. A provided fixed_beta is for explicit diagnostic oracles only. Deployment jointly estimates beta.
- `reconstruct(raw_dn, calibration, selection)` accepts raw DN and external fixed physics/calibration and selection dictionaries, no truth/condition/identity/cache. Returns all the above physical fields with the canonical names, plus deployment diagnostics.
- `fisher_identifiability(physics, state, beta, raw_e, options=None)` is a separately labelled diagnostic and may receive truth only in that branch.
- MC uncertainty accepts a reconstruction callback from raw DN to the canonical output dict; use `template_postblur_e`, `temperature_color_K`, `valid`, `expected_e`, and original ADC/Poisson/read noise calibration. It never accepts truth.
- The oracle ladder fixes alpha_g=0, lambda0_factor=1e-6 and lambda_beta_factor=1e-6 (diagonal scales explicitly reported), raw W fixed from the same observed frame, true CG tolerance 1e-4 and max200 for every ladder level. Development later permits only alpha_g in [0,.01,.1,1].
- Raw inputs have batch shape B,H,W after DN-to-electron conversion; formal single-frame runs use B=1. Output template/temperature/valid shapes are B,H,W; continuum is B,7,H,W; beta B,3.

Root owns common.py/protocol/run/development/baselines/metrics/statistics/gated training/final orchestration. Agents own only their explicitly assigned files. Any alternative API or numerical choice is communicated before freezing source and execution.
