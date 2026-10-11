import numpy as np
import pytest
from evaluation_v2.metrics import (score_prediction, aggregate_metrics, all_required_screens,
                                   REQUIRED_SCREENS, support_mask, detection_score)
from evaluation_v2.common import MeasurementInput, measurement_input, Journal, execute_record, outcome_counts, case_identity

DETECTOR = {"pixel_threshold_e": 40., "frame_threshold_e": 40.}


def fixture(template=None, temperature=None, valid=None):
    t = np.full((10, 10), 45.) if template is None else np.asarray(template)
    temp = np.full(t.shape, 3000.) if temperature is None else np.asarray(temperature)
    truth = {"template_e": t, "temperature_latent_K": temp.copy(),
             "temperature_reference_color_K": temp.copy(), "temperature_observed_color_K": temp.copy(),
             "evaluation_domain": np.ones(t.shape, bool), "uniform_temperature_domain": np.ones(t.shape, bool)}
    pred = {"template_e": t.copy(), "temperature_K": temp.copy(),
            "valid": np.ones(t.shape, bool) if valid is None else valid.copy()}
    return truth, pred


def test_A_perfect_45e_weak_signal():
    t, p = fixture()
    m = score_prediction(t, p, DETECTOR)["metrics"]
    assert m["template_relative_L2"] == 0
    assert m["support20_pixels"] == 100
    assert m["support20_recall"] == 1
    assert m["historical60_recall_on_truth_support20"] == 0
    assert m["historical60_target_region_pixels"] == 0


def test_B_weak_failure_and_missing_requirement_cannot_pass():
    s = dict.fromkeys(REQUIRED_SCREENS, True)
    s["weak_signal_screen_passed"] = False
    assert all_required_screens(s)["all_required_screens_passed"] is False
    del s["weak_signal_screen_passed"]
    assert all_required_screens(s)["status"] == "incomplete"


def test_C_prediction_valid_cannot_improve_fixed_errors():
    t, p = fixture()
    p["template_e"][0, 0] = 200
    p["temperature_K"][0, 0] = 4000
    a = score_prediction(t, p, DETECTOR)["metrics"]
    p["valid"][0, 0] = False
    b = score_prediction(t, p, DETECTOR)["metrics"]
    assert a["template_RMSE_e"] == b["template_RMSE_e"]
    assert a["temperature_observed_RMSE_K"] == b["temperature_observed_RMSE_K"]
    assert b["conditional_temperature_RMSE_K"] == 0
    assert b["coverage"] == .99
    assert len(b["risk_coverage"]) == 2


def test_D_one_percent_1000K_errors_remain_100K_RMSE():
    t, p = fixture()
    p["temperature_K"][0, 0] += 1000
    p["valid"][0, 0] = False
    assert score_prediction(t, p, DETECTOR)["metrics"]["temperature_observed_RMSE_K"] == 100


def test_E_blank_invalid_ghosts_still_count():
    t, p = fixture(np.zeros((10, 10)))
    p["template_e"][0, 0] = 100
    p["valid"][0, 0] = False
    m = score_prediction(t, p, DETECTOR)["metrics"]
    assert m["null_FP_fixed_domain"] == .01
    assert m["historical60_null_FP_fixed_domain"] == .01
    assert m["null_FP_accepted"] == 0
    assert m["template_RMSE_e"] == 10
    assert m["template_relative_L2"] is None


def test_F_signed_cancellation_is_separate_from_absolute_bias():
    rows = []
    for factor in (1.4, .6):
        t, p = fixture(np.ones((10, 10)))
        p["template_e"] *= factor
        rows.append(score_prediction(t, p, DETECTOR))
    m = aggregate_metrics(rows)
    assert m["pooled_signed_integral_bias"] == pytest.approx(0, abs=1e-12)
    assert m["per_case_absolute_integral_bias"]["mean"] == pytest.approx(.4)


@pytest.mark.parametrize("failure", ["missing_domain", "empty", "saturated", "nonfinite", "missing", "out_of_bounds"])
def test_H_explicit_incomplete_or_failed_frame(failure):
    t, p = fixture()
    sat = False
    if failure == "missing_domain":
        del t["evaluation_domain"]
    if failure == "empty":
        t["evaluation_domain"][:] = False
    if failure == "saturated":
        sat = True
    if failure == "nonfinite":
        p["template_e"][0, 0] = np.nan
    if failure == "missing":
        del p["temperature_K"]
    if failure == "out_of_bounds":
        p["temperature_K"][0, 0] = 9000
    result = score_prediction(t, p, DETECTOR, saturated=sat)
    assert result["numerical_execution_passed"] is False
    assert result["status"] != "SUCCESS"
    if failure == "nonfinite":
        assert result["metrics"]["template_RMSE_e"] is None
        assert result["metrics"]["nonfinite_pixels"] == 1


def test_H_exception_timeout_durable_resume(tmp_path):
    j = Journal(tmp_path/"per_case.jsonl")
    execute_record(j, {"identity_sha256": "a"}, lambda: {"status": "SUCCESS"})
    def broken():
        raise ValueError("frame error")
    def timeout():
        raise TimeoutError("registered budget")
    execute_record(j, {"identity_sha256": "b"}, broken)
    execute_record(j, {"identity_sha256": "c"}, timeout)
    assert [r["status"] for r in j.rows()] == ["SUCCESS", "EXCEPTION", "TIMEOUT"]
    assert j.succeeded("a") and not j.succeeded("b") and not j.succeeded("changed-config")
    assert outcome_counts(j.rows(), 4)["remaining"] == 1


def test_I_exact_threshold_equalities():
    t, p = fixture(np.array([[100., 20., 19.999]]))
    p["template_e"][:] = 60.
    m = score_prediction(t, p, {"pixel_threshold_e": 60., "frame_threshold_e": 60.})["metrics"]
    assert support_mask(t["template_e"], t["evaluation_domain"])[0].tolist() == [True, True, False]
    assert m["support20_recall"] == 0  # prediction > threshold, target >= relative threshold
    assert m["frame_detected"] is False


def test_J_raw_only_interface_and_truth_independent_detector():
    raw = np.zeros((10, 10))
    inputs = measurement_input({"raw_dn": raw})
    assert isinstance(inputs, MeasurementInput)
    assert not hasattr(inputs, "truth") and not hasattr(inputs, "scene_id") and not hasattr(inputs, "split")
    for key in ("truth", "split", "scene_id"):
        with pytest.raises(ValueError):
            measurement_input({"raw_dn": raw, key: "forbidden"})
    assert detection_score(np.full((10, 10), 5.)) == 5.


def test_source_domain_and_outside_signal_not_diluted():
    t, p = fixture(np.zeros((10, 10)))
    t["source_image_domain"] = np.zeros((10, 10), bool)
    t["source_image_domain"][0] = True
    p["template_e"][0] = 100
    m = score_prediction(t, p, DETECTOR)["metrics"]
    assert m["null_FP_fixed_domain"] == .1
    assert m["source_image_domain"]["null_FP_fixed_domain"] == 1.
    assert m["outside_source_template_RMSE_e"] == 0


def test_resume_identity_distinguishes_mode_case_and_configuration(tmp_path):
    kwargs = dict(config_hash="frozen-model-and-training-statistics", case_id="continuous", fixture_hash="same-npz",
                  method="B4", input_kind="oracle_continuous_e", input_hash="expected-array", run_id="run_001")
    initial = case_identity(**kwargs)
    j = Journal(tmp_path/"per_case.jsonl")
    j.append({"identity_sha256": initial, "status": "SUCCESS"})
    assert j.succeeded(case_identity(**kwargs))
    for key, changed in (("case_id", "quantized"), ("input_kind", "raw_DN"), ("input_hash", "raw-DN-array"),
                         ("config_hash", "different-training-mean"), ("method", "B2")):
        new = case_identity(**{**kwargs, key: changed})
        assert new != initial and not j.succeeded(new)
