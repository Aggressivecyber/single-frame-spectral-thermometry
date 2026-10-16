# Joint temperature and characteristic-emission imaging

The user's 2026-10-17 clarification defines the complete objective: several filter channels support thermometry and the remaining channels support CH*, C2* and AlO characteristic-emission imaging. A filter defines its wavelength; reconstruction estimates its transmitted image. Image reconstruction is an intermediate product of the joint diagnostic.

- Preserve one common field of view, one exposure and overlapping channels on the same sensor area.
- Reconstruct total passband images first. Fit a declared continuum model only in the thermometry readout. Report characteristic-band excess separately from chemical identity or concentration.
- The first pilot uses hypothetical calibrated filters, gray emission, a known spatial operator and a camera noise model. Preserve the optical/temporal averaging distinction and report effective color temperature against a noiseless optical reference.
- Compare a converged traditional estimator, independently trained CNN postprocessing and a physical-data-consistency hybrid on the same observations. Keep mother scenes disjoint across training, validation, threshold calibration and held-out evaluation.
- Architecture components and frequency multiplexing have prior art. A method contribution requires matched ablations and retained negative results.
- Heavy computation runs only on the registered server. Local files, caches and short CPU checks stay on E/D. For long work, verify startup and arrange one wake rather than continuous monitoring.
- Preserve all historical results, sealed tests and source snapshots. Register new runs in new directories.
