# v4.6 generated TRAIN data integrity

The completed independent array audit is **PASS**. Use
[`attempt_002/array_audit_receipt.json`](attempt_002/array_audit_receipt.json)
and [`attempt_002/ARRAY_AUDIT.md`](attempt_002/ARRAY_AUDIT.md) as the completed
result. The receipt SHA256 is
`fa9848aa4a0d14ec102ed27dbbaca2d6da5cf0fe580918f752e8876cd93da1db`.

The audit checked all 432 new raw files, 96 new label files and 48 reused old
TRAIN label files: 576 files containing 1152 arrays. All 11580 checks passed in
15.375 seconds. File hashes, array hashes, shape, dtype, finite values, raw
bounds and exact-duplicate checks passed. Raw values span 252 to 2862 DN;
96 continuous inputs are float32 and 336 quantized/noisy inputs are uint16.

The 3352 inherited rows remain the exact manifest prefix. The merged manifest
has 3008 TRAIN and 776 inherited development rows. The 36 signed TRAIN parents
cover the complete 6 by 6 geometry matrix. Each of their four signed conditions
has one continuous input, one quantized input and two noise inputs. All 288
noise seed records are distinct; 48 original noise-zero seeds are independently
derived from the frozen old generator and mother metadata. Old raw arrays were
not opened. All 96 quantized inputs exactly round their continuous partners.

Forty frozen source/input files were hashed before and after the array reads;
all 80 comparisons passed. Generation-receipt SHA256:
`9da024964ba3514d4ae843fc312f5a00bd70e396367ca305afdee6ea06b53d70`.
Execution-protocol SHA256:
`4d49cd48853b20f62960754d07637a3cca3ae454e79e8e4ac67b8a2f7d78b807`.

The first attempt, retained in this directory, stopped before opening any
arrays because the auditor used a duplicate Python argument name. Its 4961
completed metadata checks had no failures. Only the report-directory auditor
was corrected; its original snapshot is retained as
`audit_generated_train_attempt_001.py`. The corrected attempt is separate and
does not overwrite that earlier result.

No frozen source or data file was modified. No seed was regenerated and no
GPU or torch was used. Inherited development, new locked development, final
test and old raw arrays were not opened. This audit verifies data integrity;
it does not rerun the physical forward/inverse, prove statistical independence,
or establish experimental realism or training effectiveness.
