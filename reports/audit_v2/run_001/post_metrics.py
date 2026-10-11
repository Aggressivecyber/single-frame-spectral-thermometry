"""Production post-scoring: separate optical spillover from unexplained source-exterior signal."""
import numpy as np

SCHEMA = 'spectral-thermometry-source-domain-postmetric-v2.3'


def source_halo_metrics(target, prediction, source_domain, evaluation_domain, threshold_e):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    domain = np.asarray(evaluation_domain, bool)
    outside = domain & ~np.asarray(source_domain, bool)
    if any(a.shape != domain.shape for a in (target, prediction, source_domain)):
        raise ValueError('Mismatched source-domain shape')
    n = int(outside.sum())
    base = {'metric_schema': SCHEMA, 'outside_source_pixels': n,
            'source_mask_definition': 'source_valid_fraction > 0.5; model valid never changes this region'}
    if not n:
        return {**base, 'status': 'NOT_APPLICABLE', 'outside_error_RMSE_e': None}
    if not np.isfinite(target[outside]).all() or not np.isfinite(prediction[outside]).all():
        return {**base, 'status': 'NONFINITE_PREDICTION_OR_TRUTH', 'outside_error_RMSE_e': None}
    error = prediction[outside]-target[outside]
    peak = float(target[domain].max()) if domain.any() else 0.
    support = domain & (target >= .2*peak) if peak > 0 else np.zeros(domain.shape, bool)
    nonsupport = outside & ~support
    return {**base, 'status': 'SUCCESS', 'outside_error_RMSE_e': float(np.sqrt(np.mean(error**2))),
            'outside_signed_error_mean_e': float(error.mean()),
            'outside_predicted_signal_RMS_e': float(np.sqrt(np.mean(prediction[outside]**2))),
            'outside_truth_optical_spillover_RMS_e': float(np.sqrt(np.mean(target[outside]**2))),
            'outside_true_support20_pixels': int(np.sum(outside & support)),
            'outside_nonsupport_detected_fraction': float(np.mean(prediction[nonsupport] > threshold_e)) if nonsupport.any() else None,
            'outside_abs_error_gt60_fraction': float(np.mean(np.abs(error) > 60.)),
            'prediction_threshold_e': threshold_e,
            'note': 'Target is postblur. Nonzero optical spillover is expected signal, not automatically a ghost.'}
