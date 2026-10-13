# v4.6 TRAIN data and physical cache contracts

`data.py` imports only the standard library until an explicitly authorized entry
is used. Metadata inspection is read-only:

```powershell
python -B -m training_v4_6.data --check-registry
```

The frozen design keeps all 3352 inherited manifest rows unchanged. It adds
24 signed TRAIN parents covering the missing temperature/template geometry
pairs, four conditions per parent, and continuous, quantized and two independent
noise inputs per condition (384 rows). A second noise input is added for each
of the 48 conditions from the existing 12 signed TRAIN parents. The resulting
432 new TRAIN rows give 3008 TRAIN and 776 inherited development rows.

The data core uses the original mother-field generator and signed transforms,
the frozen sensor configuration, and registered noise seeds. New raw and label
files are saved exclusively and reopened for exact array verification. Both
pretransform and actual posttransform field hashes are recorded. Generation
rejects duplicate new raw array/file hashes, new label file hashes against old
manifest hashes, repeated new parent fields, nonfinite fields and saturation.
Intentional label/continuum reuse within a mother is retained. Failures keep
the original seed, diagnostic arrays and partial journals; no seed replacement,
silent overwrite or automatic restart is supported.

The paired 500-update sampling protocol changes only the 513 negative noisy
positions. The remaining 1487 case IDs stay in their exact previous positions.
Only 120 unique newly introduced negative noisy inputs need new physical
caches. New positive and clean inputs are generated but not cached or drawn
by this contrast. This does not establish full-factorial training exposure.

`cache.py` reuses the original scalar inverse and physical cache key. Each raw
is inverted before its TRAIN label is opened for the scoring sidecar. Existing
inputs route to their previous read-only caches. New-cache loading verifies
the raw, cache and metadata hashes and physical configuration identities.

Formal execution requires root to register an immutable `execution_protocol.json`
and provide its exact file SHA256. The gate verifies v4.5 completion with no
candidate, the frozen source/input hashes and the 500 by 4 TRAIN-only draws.
No formal job is started by metadata checks or by the CPU tests.

```powershell
python -B -m training_v4_6.data --split train --execute --protocol-sha256 <HASH>
python -B -m training_v4_6.cache --execute --protocol-sha256 <HASH>
```

These commands are bounded by 600 seconds for generation and 3600 seconds for
one physical-cache worker with two CPU threads. Caching stops before the next
frame after its budget expires; a currently running frozen scalar inversion
can overrun by one frame. Completed/failed artifacts are preserved. Locked new
development descriptors are metadata only; their arrays, inherited development
arrays and final-test arrays/results are never opened by these two stages.

The CPU contract tests use tiny 8 by 8 mock physics, blocked CUDA visibility,
one CPU thread and a separate E-drive `training_v4_6/_test_runs/CPU_MOCK_TEST`
fixture subtree. They test the real generation/file/hash and cache/scoring
control flow with mock numerics and do not validate native sensor numerics.

```powershell
python -B -m training_v4_6.test_data_cache_cpu
```
