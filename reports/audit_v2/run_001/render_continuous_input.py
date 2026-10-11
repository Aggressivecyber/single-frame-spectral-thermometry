"""Correct the oracle figure's input panel; read saved arrays only, no inference."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation_v2.common import array_hash, Journal

OUT = Path(__file__).resolve().parent
DATA = ROOT / 'outputs/audit_v2/run_001'


def main():
    manifest = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    fixture = next(f for f in manifest['cases'] if f['case_id'] == 'front_continuous_e')
    assert fixture['metadata']['input_kind'] == 'oracle_continuous_e'
    fixture_path = DATA / fixture['path']
    assert runtime.sha256(fixture_path) == fixture['file_sha256']
    with np.load(fixture_path, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    assert array_hash(arrays['expected_e']) == fixture['array_sha256']['expected_e']
    config = json.loads((OUT / 'active_configuration.json').read_text(encoding='utf-8'))['config_sha256']
    selected = list({r['identity_sha256']: r for r in Journal(OUT / 'per_case.jsonl').rows()
                     if r['config_sha256'] == config and r['case_id'] == fixture['case_id']}.values())
    assert len(selected) == 8 and all(r['status'] == 'SUCCESS' for r in selected)
    predictions = {}
    for row in selected:
        path = DATA / row['prediction_path']
        assert runtime.sha256(path) == row['prediction_sha256']
        with np.load(path, allow_pickle=False) as f:
            predictions[row['method']] = {k: f[k].copy() for k in f.files}
    destination = OUT / 'figures/front_continuous_e.png'
    prior = DATA / 'report_render_prior/front_continuous_e_stored_DN.png'
    prior.parent.mkdir(parents=True, exist_ok=True)
    if not prior.exists():
        shutil.copy2(destination, prior)
    smax = max(1., float(np.abs(arrays['template_e']).max()),
               max(float(np.abs(p['template_e']).max()) for p in predictions.values()))
    terrmax = max(1., max(float(np.abs(p['temperature_K'] - arrays['temperature_observed_color_K']).max())
                         for p in predictions.values() if np.isfinite(p['temperature_K']).all()))
    fig, axes = plt.subplots(1 + len(predictions), 5, figsize=(18, 2.45 * (1 + len(predictions))))
    header = [(arrays['expected_e'], 'actual oracle input e (unquantized)', 'gray', None),
              (arrays['temperature_observed_color_K'], 'truth observed color K', 'inferno', None),
              (arrays['template_e'], 'truth postblur signed template e', 'RdBu_r', smax),
              (arrays['evaluation_domain'], 'fixed evaluation domain', 'gray', None),
              (arrays['saturated'], 'stored DN saturation (oracle N/A)', 'gray', None)]
    panel_rows = [header]
    for name, pred in predictions.items():
        temp, signed = pred['temperature_K'], pred['template_e']
        panel_rows.append([(temp, name + '\ncolor K', 'inferno', None),
                           (temp - arrays['temperature_observed_color_K'], 'temperature error K', 'RdBu_r', terrmax),
                           (signed, 'signed template e', 'RdBu_r', smax),
                           (signed - arrays['template_e'], 'template error e', 'RdBu_r', smax),
                           (pred['valid'], 'model valid (not scoring domain)', 'gray', None)])
    for axis_row, panels in zip(axes, panel_rows):
        for ax, (arr, title, cmap, limit) in zip(axis_row, panels):
            im = ax.imshow(arr, cmap=cmap, vmin=-limit if limit else None, vmax=limit)
            ax.set_title(title, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=.045)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('front_continuous_e; actual continuous-electron input; all methods; Z0 temperature N/A', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    runtime.write_json(OUT / 'figure_input_correction.json', {
        'scope': 'Figure input panel only; no model, metric, prediction or registration change',
        'case_id': fixture['case_id'], 'input_kind': 'oracle_continuous_e',
        'actual_display_array': 'expected_e', 'actual_display_array_sha256': array_hash(arrays['expected_e']),
        'prior_stored_DN_figure_path': str(prior.relative_to(DATA)).replace('\\', '/'),
        'prior_stored_DN_figure_sha256': runtime.sha256(prior),
        'corrected_figure_sha256': runtime.sha256(destination),
    })
    print('Continuous-electron input figure corrected; saved predictions unchanged.')


if __name__ == '__main__':
    main()
