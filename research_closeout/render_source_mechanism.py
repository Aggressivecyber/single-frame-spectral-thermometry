"""Figure2: existing source interventions and signed finite-solver accounting.

Fixed analysis: all36 development mothers, noisy normal +/- cases, four
declared routes;10000 paired parent bootstrap draws, percentile pointwise
intervals. No raw generation/inversion and no candidate selection.
"""
from . import storage as s
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = ('C_true__estimated_beta', 'C_v5__estimated_beta',
           'C_v3__estimated_beta', 'original_v3__joint_original')
SEED_NAME = 'closeout-source-figure-parent-bootstrap-v001-20261015'
BOOTSTRAP_DRAWS = 10000


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison = s.REPORTS/'A_REUSED_COMPARISON.json'
    summary_path = s.REPORTS/'SOURCE_STUDY_SUMMARY.json'
    comparison_data, summary = s.read(comparison), s.read(summary_path)
    journal = Path(comparison_data['source_path'])
    assert s.sha(journal) == comparison_data['source_sha256']
    accounting = s.ROOT/summary['source_journal']['path']
    assert s.sha(accounting) == summary['source_journal']['sha256']
    selected = [r for r in s.rows(journal) if r['cohort'] == 'development'
        and r['input_kind'] == 'noise' and r['group'] == 'normal' and r['method'] in METHODS]
    mothers = sorted({r['mother_id'] for r in selected})
    assert len(mothers) == 36 and len(selected) == 36*2*4
    mse = np.zeros((4, 2, 36))
    for k, method in enumerate(METHODS):
        for j, sign in enumerate(('negative', 'positive')):
            for i, mother in enumerate(mothers):
                rows = [r for r in selected if r['method'] == method and r['sign'] == sign and r['mother_id'] == mother]
                assert len(rows) == 1
                r = rows[0]
                assert r['status'] == 'SUCCESS' and r['metrics']['domain_pixels'] == 65536
                mse[k, j, i] = r['metrics']['template_SSE']/65536
    seed = int.from_bytes(hashlib.sha256(SEED_NAME.encode()).digest()[:8], 'little')
    indices = np.random.default_rng(seed).integers(0, 36, size=(BOOTSTRAP_DRAWS, 36))
    estimates = np.sqrt(mse.mean(axis=-1))
    sampled = np.sqrt(mse[:, :, indices].mean(axis=-1))
    limits = np.quantile(sampled, (.025, .975), axis=-1)
    component_names = ('continuum_leakage', 'shared_measurement_noise',
                       'regularization_bias', 'actual_numerical_error')
    sources = ('C_v5', 'C_v3')
    component_rms, cross_contributions, accounting_rows = [], [], []
    cross_names = None
    for source in sources:
        rows = [r for r in summary['rows'] if r['source'] == source and r['beta_mode'] == 'estimated_beta'
            and r['input_kind'] == 'noise' and r['condition'] in ('negative_normal', 'positive_normal')]
        assert len(rows) == 2 and all(r['cases'] == r['mothers'] == r['certified_rows'] == 6 for r in rows)
        component_rms.append([np.sqrt(np.mean([r['component_MSE_e2'][key] for r in rows])) for key in component_names])
        if cross_names is None:
            cross_names = tuple(rows[0]['signed_cross_contribution_e2'])
        assert all(tuple(r['signed_cross_contribution_e2']) == cross_names for r in rows)
        cross_contributions.append([np.mean([r['signed_cross_contribution_e2'][key] for r in rows]) for key in cross_names])
        accounting_rows.extend(rows)
    component_rms, cross_contributions = np.asarray(component_rms), np.asarray(cross_contributions)
    folder = s.ROOT/'paper/figures/F02_source_mechanism'
    folder.mkdir(parents=True, exist_ok=True)
    png, pdf = folder/'SOURCE_MECHANISM.png', folder/'SOURCE_MECHANISM.pdf'
    receipt = s.REPORTS/'repair_001/SOURCE_MECHANISM_FIGURE.json'
    assert not any(p.exists() for p in (png, pdf, receipt))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig = plt.figure(figsize=(13.5, 10.8))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1), hspace=.64, wspace=.30)
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    x, width = np.arange(4), .31
    for j, (sign, color) in enumerate((('Negative normal', '#24668d'), ('Positive normal', '#df923f'))):
        centers = x+(j-.5)*width
        bars = a.bar(centers, estimates[:, j], width, color=color, label=sign)
        bars[0].set_hatch('///')
        a.errorbar(centers, estimates[:, j], yerr=np.vstack((estimates[:, j]-limits[0, :, j],
                    limits[1, :, j]-estimates[:, j])), fmt='none', capsize=4, color='#20252b')
    a.set_xticks(x, ('Oracle C\n(extra information)', 'C_v5 + fixed linear solve',
                    'C_v3 + same linear solve', 'Complete v3\n(finite research baseline)'))
    a.set_ylabel('Template RMSE (e)')
    a.set_title('(a) Same-raw source intervention:36 development mothers', loc='left', fontsize=15, pad=12)
    a.legend(frameon=False, ncol=2, loc='upper right')
    a.text(.01, .96, '95% paired-mother bootstrap intervals; normal +/- cases', transform=a.transAxes, va='top', fontsize=11)
    a.grid(axis='y', alpha=.18)
    a.set_axisbelow(True)
    colors = ('#9b5056', '#24668d')
    for k, source in enumerate(sources):
        b.bar(np.arange(4)+(k-.5)*.32, component_rms[k], .32, label=source, color=colors[k])
        c.bar(np.arange(6)+(k-.5)*.32, cross_contributions[k], .32, label=source, color=colors[k])
    b.set_yscale('log')
    b.set_ylim(1e-6, 1e2)
    b.set_xticks(range(4), ('Leakage L', 'Shared\nnoise N', 'Prior B', 'Numerical E'))
    b.set_ylabel('Component RMS (e); magnitudes are not additive')
    b.set_title('(b) Four finite propagated components\n6 reference mothers,12 normal noisy raws', loc='left', fontsize=13, pad=12)
    b.legend(frameon=False, ncol=2)
    c.set_yscale('symlog', linthresh=1e-4)
    c.axhline(0, color='#555555', linewidth=.8)
    c.set_xticks(range(6), ('L,N', 'L,B', 'L,E', 'N,B', 'N,E', 'B,E'))
    c.set_ylabel('Signed cross contribution (e squared)')
    c.set_title('(c) Shared-raw signed cross terms\nSame6 reference mothers as panel(b)', loc='left', fontsize=13, pad=12)
    c.legend(frameon=False, ncol=2)
    for ax in (b, c):
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.5, .026, 'Panels(b,c): fixed original K/W/prior. Component/label roundoff is accounted separately.', ha='center', fontsize=12)
    fig.subplots_adjust(left=.08, right=.98, bottom=.13, top=.94)
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    result = {'schema': 'source-mechanism-figure-and-statistics-v001', 'time': s.now(),
        'renderer': {'path': str(Path(__file__)), 'sha256': s.sha(Path(__file__))},
        'sources': [{'path': str(p), 'sha256': s.sha(p)} for p in (comparison, summary_path, journal, accounting)],
        'panel_a': {'methods': list(METHODS), 'mothers': mothers, 'unique_raws': 72, 'cells': 288,
                    'template_RMSE_e': estimates.tolist(), 'pointwise_95_percentile_interval_e': limits.tolist(),
                    'bootstrap_namespace': SEED_NAME, 'bootstrap_seed': seed, 'bootstrap_draws': BOOTSTRAP_DRAWS,
                    'paired_parent_resampling_shared_across_methods_and_signs': True,
                    'interval_scope': 'conditional descriptive development parent bootstrap, not final generalization or coverage certification'},
        'panels_b_c': {'sources': list(sources), 'parent_mothers': 6, 'unique_raws': 12,
                    'component_names': component_names, 'component_RMS_e': component_rms.tolist(),
                    'cross_names': cross_names, 'signed_cross_contribution_e2': cross_contributions.tolist(),
                    'original_summary_rows': accounting_rows,
                    'finite_component_and_label_roundoff_retained_separately': True,
                    'corrected_closure_is_not_camera_accuracy': True},
        'outputs': [{'path': str(p), 'sha256': s.sha(p), 'bytes': p.stat().st_size} for p in (png, pdf)],
        'new_inverse_calls': 0, 'raw_regenerated': False, 'whole_closeout_complete': False,
        'PNG_visual_QA': 'pending', 'PDF_render_QA': 'pending'}
    s.write(receipt, s.plain(result))
    print(json.dumps({'figure': str(png), 'new_inverse_calls': 0, 'parent_bootstrap_draws': BOOTSTRAP_DRAWS}), flush=True)
    return result


if __name__ == '__main__':
    run()
