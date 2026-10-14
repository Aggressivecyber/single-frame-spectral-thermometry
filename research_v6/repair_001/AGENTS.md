# Numerical repair001

- The user's continuation authorizes one targeted numerical repair and controlled retest after the documented run001 stop. Preserve all prior v6 files, protocols, reports, archives and GitHub releases as well as all older baselines.
- All new implementation belongs here. Reports belong in reports/research_v6/repair_001/; data/cache/logs belong in outputs/research_v6/repair_001/ on E/D.
- Change arithmetic and convergence accuracy only: original float32 calibration/operator coefficients, W and regularizer scalars are promoted exactly. Raw bytes, continuum source algorithms, alpha1 and maximum200 PCG iterations remain fixed. Every new RHS uses coherent float64 actions, same rtol1e-8 for every source/component/beta mode.
- Keep the original0.05e /0.005 observable closure gates and all scientific acceptance gates. No threshold relaxation, source-dependent changes, model fitting or new training.
- First register bounded repair checks and the single failed-input retest. Resume the720-case A matrix only if that controlled retest passes every component/actual-solve/closure gate. Do not combine old and repaired outputs to make a passing candidate.
- B remains conditional on complete repaired A and its unchanged scientific criteria. Signed36 and final64 stay sealed.
- Root runs one GPU process. For a long run, verify startup, schedule one wake and stop monitoring.
