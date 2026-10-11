"""Audit result reuse must match the original registered method cohort."""
import runtime
from repair_v3 import run
from evaluation_v2.methods import METHODS


def test_reuse_excludes_unregistered_resume_probes_and_other_configurations(monkeypatch):
    core = [dict(config_sha256='active', identity_sha256=f'core-{m}', method=m,
                 status='SUCCESS') for m in METHODS]
    probes = [dict(config_sha256='active', identity_sha256=m, method=m,
                   status=status) for m, status in
              [('B4_known_true_calibration', 'EXCEPTION'),
               ('B1_known_700_905_ratio', 'SUCCESS')]]
    alternate = dict(config_sha256='other', identity_sha256='alternate',
                     method='B4_published', status='SUCCESS')

    class Journal:
        def __init__(self, path):
            pass

        def rows(self):
            return core+probes+[alternate, core[0]]

    monkeypatch.setattr(run, 'Journal', Journal)
    monkeypatch.setattr(run, 'read', lambda path: {'config_sha256': 'active'})
    assert run.old_rows() == core
