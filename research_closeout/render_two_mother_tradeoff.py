"""Working two-parent development figure from recorded metrics; no inference."""
from . import storage as s
from pathlib import Path
import json
import os

source = s.REPORTS/'repair_001/c1_selection_001/TWO_COMPLETE_MOTHERS_DIAGNOSTIC.json'
data = s.read(source)
folder = s.REPORTS/'repair_001/c1_selection_001/figures'
folder.mkdir(parents=True, exist_ok=True)
cache = s.DATA/'_runtime/mpl'
cache.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(cache)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rows = [row for row in data['rows'] if row['scope'] == 'pooled_two']
colors = {0.1: '#D55E00', 1.0: '#009E73', 10.0: '#0072B2'}
markers = {0.1: 'o', 1.0: 's', 10.0: 'D'}
baselines = {'v3_physics': ('*', 'Frozen v3'), 'old_B4': ('v', 'Frozen B4'),
             'B1_lambda100': ('X', 'Tikhonov 100'), 'B1_lambda0.1': ('P', 'Historical 0.1')}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.dpi': 200})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))
for ax, title, xlim, ylim in zip(axes, ('a  All recorded methods', 'b  Same data, enlarged'),
                                ((0, 44), (5, 25)), ((0, 64), (9, 31))):
    for row in rows:
        m = row['all5']
        if row['family'] == 'C1':
            option = row['physical_options']
            color = colors[option['continuum_multiplier']]
            filled = row['initializer'] == 'v3'
            ax.scatter(m['color_RMSE_K'], m['template_RMSE_e'],
                       marker=markers[option['template_multiplier']], s=44 if filled else 104,
                       facecolors=color if filled else 'none', edgecolors=color, linewidths=1.1, zorder=3)
        else:
            ax.scatter(m['color_RMSE_K'], m['template_RMSE_e'], marker=baselines[row['method']][0],
                       s=110, c='black', zorder=4)
    ax.set(title=title, xlabel='Color-temperature RMSE / K', ylabel='Template RMSE / reference e',
           xlim=xlim, ylim=ylim)
    ax.grid(alpha=.2, zorder=0)
handles = [Line2D([], [], color=color, marker='o', ls='', label=f'Continuum c={value:g}')
           for value, color in colors.items()]
handles += [Line2D([], [], color='gray', marker=marker, ls='', label=f'Template s={value:g}')
            for value, marker in markers.items()]
handles += [Line2D([], [], color='gray', marker='o', ls='', label='v3 initialization'),
            Line2D([], [], color='gray', marker='o', mfc='none', ls='', label='Smooth initialization')]
handles += [Line2D([], [], color='black', marker=marker, ls='', label=label)
            for marker, label in baselines.values()]
fig.suptitle('Development snapshot: 2 complete parent scenes, 10 raw inputs', y=.98, fontsize=12)
fig.text(.07, .23, 'Pooled full-FOV SSE / pixels; original optical target. Both initializers are plotted.\n'
         'Frozen baselines use recorded outputs; numerical certification is reported separately.', fontsize=9)
fig.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(.5, .015), frameon=False, fontsize=9)
fig.tight_layout(rect=(0, .29, 1, .94))
outputs = []
for suffix in ('png', 'pdf'):
    path = folder/f'TWO_MOTHER_TRADEOFF_001.{suffix}'
    assert not path.exists(), 'Preserve the original rendered version'
    fig.savefig(path, bbox_inches='tight')
    outputs.append({'path': str(path), 'sha256': s.sha(path), 'bytes': path.stat().st_size})
plt.close(fig)
receipt = {'time': s.now(), 'source': {'path': str(source), 'sha256': s.sha(source)},
           'plot_source_sha256': s.sha(Path(__file__)), 'outputs': outputs, 'independent_mothers': 2,
           'C1_candidates': 9, 'initializers': 2, 'baseline_methods': 4, 'PNG_visual_QA': 'pending',
           'PDF_visual_render_QA_performed': False, 'working_figure': True,
           'seven_final_figure_groups_completed': False, 'complete_inverse_calls': 0, 'forward_calls': 0}
s.write(folder/'TWO_MOTHER_TRADEOFF_001.json', receipt)
print(json.dumps({'outputs': outputs, 'working_figure': True}), flush=True)
