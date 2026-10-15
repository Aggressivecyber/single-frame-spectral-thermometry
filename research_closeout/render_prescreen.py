"""Render the fixed nine-pair, one-raw development diagnostic."""
from . import storage as s
from pathlib import Path
import json
import numpy as np


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = s.REPORTS/'repair_001/c1_development_001/NATIVE_PRESCREEN_ANALYSIS.json'
    value = s.read(source)
    assert value['native_reconstructed_cells'] == 18 and value['distinct_raw_inputs'] == 1
    assert value['all_canonical_latent_hashes_and_scoring_match']
    multipliers = (.1, 1., 10.)
    color, template = np.zeros((3, 3)), np.zeros((3, 3))
    for i, t in enumerate(multipliers):
        for j, c in enumerate(multipliers):
            rows = [r for r in value['rows'] if r['physical_options']['continuum_multiplier'] == c
                    and r['physical_options']['template_multiplier'] == t]
            assert len(rows) == 2 and {r['initializer'] for r in rows} == {'v3', 'smooth_demod'}
            pixels = sum(r['metrics']['domain_pixels'] for r in rows)
            assert pixels == 2*65536
            color[i, j] = np.sqrt(sum(r['metrics']['temperature_SSE_K2'] for r in rows)/pixels)
            template[i, j] = np.sqrt(sum(r['metrics']['template_SSE'] for r in rows)/pixels)
    baseline = next(r['metrics'] for r in value['baseline_cached_metrics'] if r['method'] == 'v3_physics')
    folder = s.ROOT/'paper/figures/supplement_c1_prescreen'
    folder.mkdir(parents=True, exist_ok=True)
    png, pdf = folder/'NINE_PAIR_DIAGNOSTIC.png', folder/'NINE_PAIR_DIAGNOSTIC.pdf'
    receipt = s.REPORTS/'repair_001/c1_development_001/PRESCREEN_FIGURE.json'
    assert not any(p.exists() for p in (png, pdf, receipt))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    for ax, matrix, label, units, control in zip(axes, (color, template),
            ('Color-temperature RMSE', 'Template RMSE'), ('K', 'e'),
            (baseline['temperature_RMSE_K'], baseline['template_RMSE_e'])):
        image = ax.imshow(matrix, origin='lower', cmap='YlGnBu', aspect='equal')
        ax.set_xticks(range(3), ('.1', '1', '10'))
        ax.set_yticks(range(3), ('.1', '1', '10'))
        ax.set_xlabel('Continuum regularizer multiplier')
        ax.set_ylabel('Template regularizer multiplier')
        ax.set_title(f'{label} ({units})\nSame-raw v3: {control:.2f} {units}', fontsize=14)
        for i in range(3):
            for j in range(3):
                text_color = 'white' if matrix[i, j] > (matrix.min()+matrix.max())/2 else '#142c41'
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color=text_color,
                        fontsize=15, fontweight='bold')
        bar = fig.colorbar(image, ax=ax, shrink=.85, pad=.04)
        bar.set_label(units)
    fig.suptitle('C1 development prescreen: all nine parameter pairs', fontsize=17, y=.98)
    fig.text(.5, .035, 'One positive-normal raw; two initializations pooled. No method selected.',
             ha='center', fontsize=12)
    fig.subplots_adjust(left=.07, right=.96, bottom=.16, top=.77, wspace=.42)
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    result = {'schema': 'one-raw-nine-pair-descriptive-figure-v001', 'time': s.now(),
        'source': {'path': str(source), 'sha256': s.sha(source)},
        'renderer': {'path': str(Path(__file__)), 'sha256': s.sha(Path(__file__))},
        'display': 'root mean squared error pooled over two initializations; not independent replicates',
        'distinct_raws': 1, 'distinct_mothers': 1, 'all_cells': 18, 'scientific_method_selection': False,
        'row_template_multipliers': list(multipliers), 'column_continuum_multipliers': list(multipliers),
        'color_RMSE_K': color.tolist(), 'template_RMSE_e': template.tolist(),
        'baseline_v3': {'color_RMSE_K': baseline['temperature_RMSE_K'], 'template_RMSE_e': baseline['template_RMSE_e']},
        'outputs': [{'path': str(p), 'sha256': s.sha(p), 'bytes': p.stat().st_size} for p in (png, pdf)],
        'PNG_visual_QA': 'pending', 'PDF_render_QA': 'pending', 'new_inverse_calls': 0}
    s.write(receipt, result)
    print(json.dumps({'PNG': str(png), 'PDF': str(pdf), 'source_sha256': s.sha(source)}), flush=True)
    return result


if __name__ == '__main__':
    run()
