# v5 execution contract

- Work is authorized by the full TASK_SPEC.md and the root's active goal. Use gpt-6.1-sol with max for subagents.
- All generated code, arrays, reports, logs, temporary files and caches remain on E/D. Existing C interpreters may be read; never write data to C.
- Base is ddc4223fd1ecb9c5af82a8d1c489bf9cc54fd2d2. All earlier files are immutable. Add work only under research_v5/, reports/research_v5/, outputs/research_v5/.
- Root owns protocol, stage gates, run orchestration, GPU execution, commits and publication. Agents may run bounded small CPU numerical checks only. One root GPU job at a time; no full data generation or training without the registered stage gate.
- Runtime setup must run before importing legacy runtime/physics/torch: E/D TEMP/TMP/TMPDIR, CUDA_CACHE_PATH, THERMOMETRY_RUNTIME, no bytecode. Use ordinary Python -B with assertions enabled.
- Implement every requested phase CLI before formal execution. O0/O1/O2 are diagnostic oracles; O3 and deployment use raw DN plus external calibration/selection only. Truth/condition/identity never enter deployment.
- Raw is the single measurement likelihood. Decoded 700/905 fields are initializer inputs; never counted again as independent measurements in the raw objective.
- Object blur occurs exactly once through CodedOperator for raw prediction. Main outputs/metrics use postblur continuum color and postblur signed template.
- Report true CG residual and nonconvergence, numerical scales, declared finite-mode Fisher scope, validity/coverage, MC/null detection uncertainty and every failed gate. Preserve negative results.
- Existing 36 signed confirmation and 64 final mother descriptors remain sealed until candidate, thresholds and all evaluation rules are frozen; final evaluation is one-time and never used for tuning.
