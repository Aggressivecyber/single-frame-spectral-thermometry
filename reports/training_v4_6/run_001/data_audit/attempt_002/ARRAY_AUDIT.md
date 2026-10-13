# v4.6 TRAIN generated-array audit

Status: **PASS**

Independent CPU audit: 576 array files; 11580 checks; 0 failed checks.

Scope: 432 new TRAIN raw files, 96 new TRAIN labels and 48 reused old TRAIN labels.
Inherited manifest rows, split counts, 36 signed TRAIN geometry pairs and two-noise records are checked.
All frozen source/input hashes are checked before and after the array reads.

No new locked development, inherited development, final-test or old raw arrays are opened.
No source/data file is modified, no seed is regenerated, and no GPU is used.

Physical forward/inverse numerics and buffer invariance are not independently rerun here.
Seed uniqueness and distinct raw hashes are not a statistical proof of independence.

