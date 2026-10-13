# v4.6 delivery scope audit

Snapshot status: **FINDING_REQUIRES_SEAL_FIX**

One concrete sealing gap was found: `seal_results.py:56-61` reads the supplied bundle receipt but never uses its contents. The seal therefore does not check bundle completion, ZIP presence/hash, or bind portability to that exact bundle. This is a gate gap; no corrupt delivery has been observed, and delivery output does not yet exist at this snapshot.

Storage and static relocation checks passed. The inference entry checks runtime paths on E/D before importing runtime/torch, then rejects an existing output directory. Sources/configuration resolve from the relocated bundle. The inherited E-drive GPU_PYTHON constant is unused on the inference call path. Absolute paths stored in receipts/checkpoint provenance are not used to locate runtime dependencies. Portability is intentionally limited to Windows E/D; native relocation remains pending.

Actual v4.6 output: 432 raw files, 96 labels, 120 state caches plus 120 cache metadata files, and 120 physical-score files. All resolved entries stay on E/D. No registered new locked-development or final parent directory exists. The 432 new manifest rows are all TRAIN; 776 inherited development rows are unchanged. The builder deliberately copies three old-development raw examples, with no truth/state caches or new locked/final data.

Old-evidence checks: all 1046 baseline files still exist with unchanged sizes; all 803 non-NPZ/non-PT file hashes match. Array/checkpoint payload bytes were not reopened for this task. The final 1046-file byte-preservation seal is still pending. Builder/inference/seal use new output locations and reject existing final artifacts; no reviewed delivery write targets old evidence.

The existing builder static-review receipts match the current builder and all 19 dependency hashes. This audit did not build, run or relocate a model. It imported no numpy/torch/runtime, opened no array/checkpoint payload and used no GPU. `delivery_scope_audit.json` records exact source hashes, directory counts, current training status, evidence locations and limitations.
