from evaluation_v2.audit import known_calibration_fixture


def test_all_suite_calibration_fixture_is_same_registered_legacy_parent():
    smoke = {'suite': 'smoke', 'metadata': {'family': 'channel_gain_3pct', 'condition': 'signal_nominal'}}
    legacy = {'suite': 'legacy', 'metadata': {'family': 'channel_gain_3pct', 'condition': 'null_clean', 'replicate': 0}}
    assert known_calibration_fixture([smoke, legacy]) is legacy
    assert known_calibration_fixture([legacy]) is legacy
