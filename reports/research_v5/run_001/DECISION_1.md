# v5 stage 1 decision

Stage 1 evidence complete: **True**. Physical stage 2 eligible: **True**. ML stop required: **False**.

O0 normal continuous pooled results: pooled_RMSE_e=0.00561248, pooled_L2=9.94733e-05, pooled_support_sign_recall=1, maximum_true_CG_residual=9.68749e-05.

O1: pooled normal continuous RMSE=0.478564 e; worse-than-O0 screen=False; strict collapse=False.
O2: pooled normal continuous RMSE=0.608687 e; worse-than-O0 screen=False; strict collapse=False.
O3: pooled normal continuous RMSE=0.751305 e; worse-than-O0 screen=False; strict collapse=False.

Stop/restriction reasons: none from stage 1; development eligibility still required for training

If O0 fails, stop ML and inspect numerical inversion, forward/adjoint and optical information. If O0 passes and O1/O2 collapse, quantify continuum calibration or structured background identifiability. If only noisy weak modes have low observable SNR, report a detection limit and abstention; weak L2<1 is not required below that limit. Actual modal results are in FISHER_IDENTIFIABILITY.json.

This decision does not open final/confirmation data, establish final scientific acceptance, or support true gas-temperature/species-concentration/10 kHz experimental claims.
