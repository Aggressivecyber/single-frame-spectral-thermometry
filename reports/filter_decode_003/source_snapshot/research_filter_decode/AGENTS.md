# Filter-channel image decoding

The user's 2026-10-17 clarification replaces the active temperature/species-template objective: filters define wavelength channels; reconstruct each channel's total transmitted image. The user then explicitly requested a goal and execution of the three-visible-channel plan.

- Keep one common field of view and overlapping channels on the same sensor area.
- Targets include all light within each passband. Do not impose chemical purity, continuum subtraction, Planck spectra, or species coefficients.
- Record optical bandwidth and compare at an explicitly specified resolution. Also report detail lost before encoding.
- Filter transmission, wavelength-dependent optics, camera response and measured calibration remain unknown until supplied or measured. Synthetic electron counts are not measured exposure/frame-rate evidence.
- Preserve historical studies and their sealed data. Use new independent image fixtures and a separate decoding ledger.
- Local work is short CPU correctness checks only; simulations and training run on the registered server. Keep local files/cache on E/D.
- No cumulative call budget; retain exact run counts. For a long job, verify startup and set one wake. Do not schedule while idle.
