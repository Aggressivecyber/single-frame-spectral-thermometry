"""CPU-only analysis of the completed six-checkpoint v4.5 control.

Only JSON sidecars and checkpoint bytes for SHA256 are read.  No model, raw or
label array is loaded.  A complete immutable analysis attempt is retained before
derived top-level files are published.  Historical controls are not candidates.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/training_v4_5/run_001'
UPDATES = (0, 100, 200, 300, 400, 500)
SIGNED = ('negative_weak', 'positive_weak', 'negative_normal', 'positive_normal')
ARMS = {
    'v43_unbalanced': {'folder': 'reports/training_v4_3/run_001/residual_scaled',
                       'label': 'v4.3 unbalanced / free C', 'color': '#31688e', 'marker': 'o'},
    'v43_balanced': {'folder': 'reports/training_v4_3/run_001/negative_balanced',
                     'label': 'v4.3 balanced / free C', 'color': '#e17c05', 'marker': 's'},
    'v44_fixed': {'folder': 'reports/training_v4_4/run_001/fixed_continuum',
                  'label': 'v4.4 unbalanced / fixed C', 'color': '#1b9e77', 'marker': '^'},
    'v45_fixed_balanced': {'folder': 'reports/training_v4_5/run_001/balanced_fixed_continuum',
                           'label': 'v4.5 balanced / fixed C', 'color': '#b33e87', 'marker': 'D'},
}
CURRENT = 'v45_fixed_balanced'
EXPECTED_RULE = ('After completing the registered comparisons, among candidates passing old eligible and all new screens, '
                 'minimize main weak L2; ties use normal RMSE then earlier checkpoint.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def mean(values):
    return math.fsum(values) / len(values)


def close(actual, expected, description):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), description)


def configure_output():
    require(OUT.resolve().drive.upper() in ('E:', 'D:'), 'Analysis output must resolve to E/D')
    cache = OUT / '_analysis_runtime'
    for child in ('tmp', 'mpl'):
        location = (cache / child).resolve()
        require(location.drive.upper() in ('E:', 'D:'), 'Analysis runtime must resolve to E/D')
        location.mkdir(parents=True, exist_ok=True)
    for variable in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[variable] = str(cache / 'tmp')
    os.environ['MPLCONFIGDIR'] = str(cache / 'mpl')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def pooled(rows):
    require(bool(rows), 'Empty reporting group')
    pixels = sum(row['metrics']['pixels'] for row in rows)
    energy = math.fsum(row['metrics']['truth_energy'] for row in rows)
    sse = math.fsum(row['metrics']['template_SSE'] for row in rows)
    result = {
        'frames': len(rows), 'mothers': len({row['mother_id'] for row in rows}), 'pixels': pixels,
        'truth_energy': energy, 'template_SSE': sse,
        'template_MSE_e2': mean([row['metrics']['template_MSE_e2'] for row in rows]),
        'template_RMSE_e': math.sqrt(sse / pixels),
        'template_L2': math.sqrt(sse / energy) if energy else None,
        'color_RMSE_K': math.sqrt(mean([row['metrics']['temperature_MSE_K2'] for row in rows])),
        'physical_template_MSE_e2': mean([row['physical']['template_MSE_e2'] for row in rows]),
        'physical_color_RMSE_K': math.sqrt(mean([row['physical']['temperature_MSE_K2'] for row in rows])),
        'maximum_invalid_output_fraction': max(row['invalid_output_fraction'] for row in rows),
    }
    for field in ('negative_recall60', 'positive_recall60', 'positive_blank_FP60', 'negative_blank_FP60'):
        values = [row['metrics'][field] for row in rows if row['metrics'][field] is not None]
        result['macro_' + field] = mean(values) if values else None
    projections = [row['metrics']['signed_projection'] for row in rows]
    result['energy_weighted_signed_projection'] = (
        math.fsum(row['metrics']['signed_projection'] * row['metrics']['truth_energy'] for row in rows) / energy
        if energy and all(value is not None for value in projections) else None)
    return result


def strata(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['signed_aux'], row['condition'], row['input_kind'])].append(row)
    return [{'signed_aux': key[0], 'condition': key[1], 'input_kind': key[2], **pooled(group)}
            for key, group in sorted(groups.items())]


def selected_group(rows, name):
    if name.startswith('main_'):
        condition = name.removeprefix('main_')
        return [row for row in rows if row['input_kind'] == 'noise'
                and not row['signed_aux'] and row['condition'] == condition]
    return [row for row in rows if row['input_kind'] == 'noise'
            and row['signed_aux'] and row['condition'] == name]


def signed_summaries(rows, screen):
    summaries = []
    for condition in SIGNED:
        group = selected_group(rows, condition)
        stats = pooled(group)
        limit = (screen['template_relative_MSE_factor'] * stats['physical_template_MSE_e2']
                 + screen['template_absolute_MSE_tolerance_e2'])
        weak = condition.endswith('_weak')
        weak_pass = (stats['template_L2'] < screen['separate_signed_weak_pooled_L2_strictly_less_than']) if weak else True
        mse_pass = stats['template_MSE_e2'] <= limit
        recall = 'macro_negative_recall60' if condition.startswith('negative') else 'macro_positive_recall60'
        summaries.append({'condition': condition, **stats, 'MSE_limit_e2': limit,
                          'macro_recall_at_fixed_60e': stats[recall],
                          'MSE_pass': mse_pass, 'weak_L2_pass': weak_pass,
                          'passed': mse_pass and weak_pass})
    return summaries


def legacy_checks(decision, gates):
    require(len(gates['decisions']) == 16, 'Original 16-gate ledger is incomplete')
    require(len(decision['protection_groups']) == 96, 'Protection 96-group ledger is incomplete')
    passed = sum(item['passed'] for item in gates['decisions'])
    require(passed == decision['original_gates_pass_count'], 'Original gate count differs from summary')
    require(gates['passed'] == (passed == 16) == decision['original_gates_pass'], 'Original gate decisions disagree')
    protection = all(item['passed'] for item in decision['protection_groups'])
    require(protection == decision['protection_pass'], 'Protection summary disagrees with its groups')
    temperature = []
    for group in decision['groups']:
        if group['input_kind'] != 'noise':
            continue
        limit = 1.05 * group['physical_color_RMSE_K'] ** 2 + .04
        temperature.append({**group, 'MSE_limit_K2': limit,
                            'passed': group['color_RMSE_K'] ** 2 <= limit})
    require(bool(temperature), 'No noisy-temperature groups')
    temperature_pass = all(item['passed'] for item in temperature)
    require(temperature_pass == decision['noisy_temperature_pass'], 'Noisy-temperature flags disagree')
    eligible = passed == 16 and protection and decision['weak_L2'] < 1 and temperature_pass
    require(eligible == decision['eligible'], 'Legacy eligible differs from its frozen component rules')
    return {'original_gate_passed_count': passed, 'original_gate_total': 16,
            'protection_passed_count': sum(item['passed'] for item in decision['protection_groups']),
            'protection_group_total': 96, 'main_weak_L2_pass': decision['weak_L2'] < 1,
            'noisy_temperature_pass': temperature_pass, 'legacy_eligible': eligible,
            'original_failed': [item for item in gates['decisions'] if not item['passed']],
            'protection_failed': [item for item in decision['protection_groups'] if not item['passed']],
            'noisy_temperature_groups': temperature,
            'noisy_temperature_failed': [item for item in temperature if not item['passed']]}


def check_science(rows, decision, published, signed, screen):
    by_name = {item['condition']: item for item in published['signed_noisy_conditions']}
    require(set(by_name) == set(SIGNED), 'Signed screen is incomplete')
    for item in signed:
        old = by_name[item['condition']]
        require(item['frames'] == old['frames'] == 6, 'Signed reporting group must contain six frames')
        for key in ('template_MSE_e2', 'physical_template_MSE_e2', 'MSE_limit_e2',
                    'template_RMSE_e', 'template_L2', 'energy_weighted_signed_projection',
                    'macro_recall_at_fixed_60e'):
            if item[key] is None or old[key] is None:
                require(item[key] is old[key], f'Signed null statistic differs: {key}')
            else:
                close(item[key], old[key], f'Signed statistic differs: {key}')
        require(all(item[key] == old[key] for key in ('MSE_pass', 'weak_L2_pass', 'passed')),
                'Signed pass flags differ; no threshold relaxation is allowed')
    finite = all(row['status'] == 'SUCCESS'
                 and math.isfinite(row['metrics']['template_MSE_e2'])
                 and math.isfinite(row['metrics']['temperature_MSE_K2']) for row in rows)
    invalid = max(row['invalid_output_fraction'] for row in rows)
    temperature_delta = max(row['temperature_correction_RMS_K'] for row in rows)
    added = finite and invalid <= screen['all_development_invalid_output_fraction'] and all(item['passed'] for item in signed)
    require(screen['finite_all_frames'] is True, 'Frozen finite-frame rule changed')
    require(temperature_delta == published['maximum_temperature_correction_RMS_K'] == 0., 'Fixed-C output changed temperature')
    require(invalid == published['maximum_invalid_output_fraction'], 'Invalid fraction differs from screen')
    require(finite == published['all_frames_finite'], 'Finite-frame screen disagrees')
    require(added == published['new_development_screen_pass'], 'New screen aggregate disagrees')
    require(decision['eligible'] == published['old_eligible'], 'Legacy eligible differs in science record')
    require((decision['eligible'] and added) == published['science_candidate'], 'Candidate flag disagrees')
    return {'all_frames_finite': finite, 'maximum_invalid_output_fraction': invalid,
            'maximum_temperature_correction_RMS_K': temperature_delta,
            'new_development_screen_pass': added, 'science_candidate': decision['eligible'] and added,
            'signed_failed_conditions': [item['condition'] for item in signed if not item['passed']]}


def raw_strata(audits):
    groups = defaultdict(list)
    for audit in audits:
        groups[(audit['condition'], audit['input_kind'])].append(audit)
    return [{'condition': key[0], 'input_kind': key[1], 'frames': len(group),
             'pooled_weighted_raw_RMS': math.sqrt(mean([item['weighted_raw_RMS'] ** 2 for item in group])),
             'physical_pooled_weighted_raw_RMS': math.sqrt(mean([item['physical_weighted_raw_RMS'] ** 2 for item in group]))}
            for key, group in sorted(groups.items())]


def match_four_cell(rows_by_arm, update):
    lookup = {arm: {row['case_id']: row for row in rows} for arm, rows in rows_by_arm.items()}
    reference = rows_by_arm[CURRENT]
    require(all(set(items) == set(lookup[CURRENT]) for items in lookup.values()), 'Four-cell case sets differ')
    pairs = []
    for row in reference:
        matches = {arm: items[row['case_id']] for arm, items in lookup.items()}
        for other in matches.values():
            for key in ('mother_id', 'condition', 'group', 'geometry', 'input_kind', 'signed_aux', 'raw_hash', 'label_sha256'):
                require(other[key] == row[key], f'Four-cell {key} mismatch: {row["case_id"]}')
            for key in ('cache_key', 'cache_sha256', 'cache_metadata_sha256'):
                require(other['physical'][key] == row['physical'][key], f'Four-cell {key} mismatch: {row["case_id"]}')
            for key in ('pixels', 'truth_energy'):
                require(other['metrics'][key] == row['metrics'][key], f'Paired label statistic differs: {key}')
        values = {arm: value['metrics']['template_MSE_e2'] for arm, value in matches.items()}
        free = values['v43_balanced'] - values['v43_unbalanced']
        fixed = values[CURRENT] - values['v44_fixed']
        pairs.append({key: row[key] for key in ('case_id', 'mother_id', 'condition', 'group', 'geometry',
                                               'input_kind', 'signed_aux', 'raw_hash', 'label_sha256')} | {
            'optimizer_update': update, 'cache_key': row['physical']['cache_key'],
            'cache_sha256': row['physical']['cache_sha256'],
            'cache_metadata_sha256': row['physical']['cache_metadata_sha256'],
            'template_MSE_e2': values, 'balanced_effect_with_free_C_e2': free,
            'balanced_effect_with_fixed_C_e2': fixed,
            'descriptive_interaction_MSE_e2': fixed - free,
        })
    return pairs


def interaction_groups(pairs, update):
    result = []
    for name in ('main_weak', 'main_normal', *SIGNED):
        rows = selected_group(pairs, name)
        require(len(rows) == (128 if name.startswith('main_') else 6), 'Interaction group incomplete')
        values = [row['descriptive_interaction_MSE_e2'] for row in rows]
        result.append({'update': update, 'condition': name, 'frames': len(rows),
                       'mothers': len({row['mother_id'] for row in rows}),
                       'mean_template_MSE_e2': {arm: mean([row['template_MSE_e2'][arm] for row in rows]) for arm in ARMS},
                       'mean_balanced_effect_free_C_e2': mean([row['balanced_effect_with_free_C_e2'] for row in rows]),
                       'mean_balanced_effect_fixed_C_e2': mean([row['balanced_effect_with_fixed_C_e2'] for row in rows]),
                       'mean_descriptive_interaction_MSE_e2': mean(values),
                       'median_descriptive_interaction_MSE_e2': statistics.median(values),
                       'minimum_descriptive_interaction_MSE_e2': min(values),
                       'maximum_descriptive_interaction_MSE_e2': max(values),
                       'fraction_negative_interaction': sum(value < 0 for value in values) / len(values)})
    return result


def choose_candidates(entries, rule):
    require(rule == EXPECTED_RULE, 'Unknown candidate rule; refuse an inferred replacement')
    candidates = [entry for entry in entries if entry['arm'] == CURRENT and entry['science_checks']['science_candidate']]
    candidates.sort(key=lambda entry: (entry['decision']['weak_L2'], entry['decision']['normal_RMSE_e'], entry['update']))
    return candidates


def render_figures(entries):
    """Return PNG/SVG bytes and reopen checks; every comparison panel has four arms."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    lookup = {(entry['arm'], entry['update']): entry for entry in entries}

    def series(arm, field, condition=None):
        values = []
        for update in UPDATES:
            entry = lookup[(arm, update)]
            values.append(entry['decision'][field] if condition is None else
                          next(group[field] for group in entry['signed_noisy_conditions'] if group['condition'] == condition))
        return values

    products, verification = {}, {}

    def finish(fig, name, panels):
        for suffix in ('png', 'svg'):
            buffer = io.BytesIO()
            fig.savefig(buffer, format=suffix, dpi=170, facecolor='white')
            data = buffer.getvalue()
            products[name + '.' + suffix] = data
            if suffix == 'png':
                with Image.open(io.BytesIO(data)) as picture:
                    picture.load()
                    width, height = picture.size
                    require(width >= 1800 and height >= 1000, 'Figure unexpectedly small')
                    require(any(low != high for low, high in picture.convert('RGB').getextrema()), 'Blank figure')
                verification[name + '.png'] = {'reopened': True, 'size': [width, height],
                                               'comparison_panels': panels, 'arm_curves_per_panel': 4}
            else:
                tree = ET.fromstring(data)
                require(tree.tag.endswith('svg'), 'Invalid SVG output')
                require(sum(element.tag.endswith('path') for element in tree.iter()) > 20, 'Incomplete SVG figure')
                verification[name + '.svg'] = {'reopened': True, 'comparison_panels': panels, 'arm_curves_per_panel': 4}
        plt.close(fig)

    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'legend.fontsize': 10, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(3, 2, figsize=(14.5, 12))
        fig.subplots_adjust(left=.075, right=.98, top=.90, bottom=.07, wspace=.23, hspace=.36)
        panels = [('Main noisy weak', 'weak_L2', None, 'Pooled L2'),
                  ('Main noisy normal', 'normal_RMSE_e', None, 'Template RMSE (e)'),
                  ('Negative signed weak', 'template_L2', 'negative_weak', 'Pooled L2'),
                  ('Positive signed weak', 'template_L2', 'positive_weak', 'Pooled L2'),
                  ('Negative signed normal', 'template_RMSE_e', 'negative_normal', 'Template RMSE (e)'),
                  ('Positive signed normal', 'template_RMSE_e', 'positive_normal', 'Template RMSE (e)')]
        handles = []
        for axis, (title, field, condition, ylabel) in zip(axes.flat, panels):
            curves = []
            for arm, spec in ARMS.items():
                line, = axis.plot(UPDATES, series(arm, field, condition), color=spec['color'], marker=spec['marker'],
                                  linewidth=1.9, markersize=5, label=spec['label'])
                curves.append(line)
            require(len(curves) == 4, 'Every main comparison panel must show all four arms')
            if not handles:
                handles = curves
            if 'L2' in ylabel:
                axis.axhline(1, color='#333333', linestyle=':', linewidth=1.2)
            elif condition:
                limits = [next(group['MSE_limit_e2'] for group in lookup[(CURRENT, update)]['signed_noisy_conditions']
                               if group['condition'] == condition) for update in UPDATES]
                require(max(limits) - min(limits) < 1e-9, 'Physical signed limit changed across checkpoints')
                axis.axhline(math.sqrt(limits[0]), color='#333333', linestyle=':', linewidth=1.2)
            axis.set(title=title, xlabel='Optimizer updates', ylabel=ylabel, xticks=UPDATES)
            axis.grid(alpha=.2)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .97), ncol=2, frameon=False)
        fig.text(.5, .015, 'Dotted lines: frozen development limits. Same-update, same-input comparisons; all four curves retained.',
                 ha='center', fontsize=10)
        finish(fig, 'four_cell_curves', 6)

        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
        fig.subplots_adjust(left=.055, right=.985, top=.84, bottom=.11, wspace=.27, hspace=.38)
        handles = []
        for column, condition in enumerate(SIGNED):
            for row, field in enumerate(('energy_weighted_signed_projection', 'macro_recall_at_fixed_60e')):
                axis = axes[row, column]
                curves = []
                for arm, spec in ARMS.items():
                    line, = axis.plot(UPDATES, series(arm, field, condition), color=spec['color'], marker=spec['marker'],
                                      linewidth=1.8, markersize=4.5, label=spec['label'])
                    curves.append(line)
                require(len(curves) == 4, 'Every fidelity panel must show all four arms')
                if not handles:
                    handles = curves
                axis.set(title=condition.replace('_', ' '), xlabel='Optimizer updates', xticks=UPDATES)
                if row == 0:
                    axis.axhline(1, color='#333333', linestyle=':', linewidth=1)
                    axis.set_ylabel('Energy-weighted signed projection')
                else:
                    axis.set_ylabel('Macro recall at fixed +/-60 e')
                    axis.set_ylim(-.02, 1.02)
                axis.grid(alpha=.2)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .97), ncol=2, frameon=False)
        fig.text(.5, .025, 'Projection and fixed-threshold recall are descriptive fidelity diagnostics; they do not replace the frozen gates.',
                 ha='center', fontsize=10)
        finish(fig, 'signed_fidelity', 8)
    return products, verification


def report_text(entries, interactions, selection, protocol):
    current = [entry for entry in entries if entry['arm'] == CURRENT]
    selected = selection['selected']
    outcome = (f"开发候选 {selection['candidate_count']} 个，按预登记顺序选择 update={selected['update']}；"
               '该选择只使用同一权重的完整门禁，仍需独立确认。' if selected else
               '六个检查点均未形成开发候选。按预登记硬停止规则，本次500更新收口，此小池不追加新种子、步数、损失或连续谱变体。')
    lines = ['# v4.5 固定最终连续谱与既有负模板平衡抽样', '', outcome, '',
             '本臂从旧v4/update2000重新初始化，沿用v4.3 negative_balanced的500组抽样；新增训练案例0。'
             '阈值与v4.4冻结规则一致。以下均为开发结果，未使用最终测试。', '',
             '|更新|原检查/16|保护/96|主weak L2|主normal RMSE e|含噪温度|旧eligible|新筛查|联合候选|失败signed条件|',
             '|---:|---:|---:|---:|---:|---|---|---|---|---|']
    for entry in current:
        legacy, science, decision = entry['legacy_checks'], entry['science_checks'], entry['decision']
        lines.append(f"|{entry['update']}|{legacy['original_gate_passed_count']}|{legacy['protection_passed_count']}|"
                     f"{decision['weak_L2']:.6f}|{decision['normal_RMSE_e']:.4f}|{legacy['noisy_temperature_pass']}|"
                     f"{legacy['legacy_eligible']}|{science['new_development_screen_pass']}|{science['science_candidate']}|"
                     f"{', '.join(science['signed_failed_conditions']) or '无'}|")
    lines += ['', '![四臂同更新完整曲线](four_cell_curves.png)', '',
              '固定C两臂的最终连续谱与色温逐帧保持物理结果；色温恒等来自输出约束，不能计为学习收益。'
              '内部六阶段连续谱头仍参与signed修正。完整16项、96组、含噪温度与所有signed分层保留在comparison_summary.json；'
              '每个v4.5检查点的全部失败见checkpoint_failures.json。', '',
              '|同一500更新末点|signed条件|L2|模板RMSE e|冻结RMSE上限e|投影|固定±60e召回|MSE通过|weak L2通过|',
              '|---|---|---:|---:|---:|---:|---:|---|---|']
    for condition in SIGNED:
        for arm, spec in ARMS.items():
            entry = next(item for item in entries if item['arm'] == arm and item['update'] == 500)
            value = next(group for group in entry['signed_noisy_conditions'] if group['condition'] == condition)
            recall = 'N/A' if value['macro_recall_at_fixed_60e'] is None else f"{value['macro_recall_at_fixed_60e']:.6f}"
            lines.append(f"|{spec['label']}|{condition}|{value['template_L2']:.6f}|{value['template_RMSE_e']:.4f}|"
                         f"{math.sqrt(value['MSE_limit_e2']):.4f}|{value['energy_weighted_signed_projection']:.6f}|"
                         f"{recall}|{value['MSE_pass']}|{value['weak_L2_pass']}|")
    lines += ['', '历史两条v4.3曲线的signed上限仅用于同一开发规则下的描述性对照，不把它们加入v4.5候选池。'
              '正负weak与normal全部列出，投影和固定阈值召回作为保真诊断保留。', '',
              '![四臂signed保真曲线](signed_fidelity.png)', '',
              '逐case描述性交互差为 `(v4.5−v4.4)−(v4.3 balanced−v4.3 unbalanced)`，单位e²。'
              '负值表示在这些固定开发输入上，平衡抽样的MSE变化在固定C条件下更低；它不构成独立验证或因果机制证明。', '',
              '|500更新组|v4.3非平衡 MSE|v4.3平衡 MSE|v4.4固定C MSE|v4.5平衡固定C MSE|交互差e²|',
              '|---|---:|---:|---:|---:|---:|']
    for group in interactions:
        if group['update'] != 500:
            continue
        values = group['mean_template_MSE_e2']
        lines.append(f"|{group['condition']}|{values['v43_unbalanced']:.5f}|{values['v43_balanced']:.5f}|"
                     f"{values['v44_fixed']:.5f}|{values[CURRENT]:.5f}|{group['mean_descriptive_interaction_MSE_e2']:.5f}|")
    lines += ['', '全部6个更新点共4656组四格配对，逐case记录中的raw/label SHA256、cache键、cache文件及其元数据SHA256均核对一致。'
              '实际数组字节审计属于独立数据审计范围；本分析只读记录。所有更新的逐case值与组均值分别见paired_four_cell.jsonl、interaction_summary.json。', '',
              '四条曲线使用同一更新数，选择不拼接不同检查点的优势。v4.5实际500次更新、2000微批次、6个检查点及Adam/原生重放的训练回执已核对。'
              '本分析模型推理、物理求解、GPU和原始数组读取均为0；独立最终评估须另有冻结候选与协议。', '',
              f'冻结选择规则：{protocol["selection_rule"]}', '',
              '分析输入哈希、图像重开及派生文件回执见analysis_receipt.json；历史分析尝试和失败记录保存在analysis_runs/。', '']
    return '\n'.join(lines)


def run():
    configure_output()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = OUT / 'analysis_runs' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    inputs, products = {}, {}

    def read(path, lines=False):
        data = path.read_bytes()
        inputs[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(data)
        return [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()] if lines else json.loads(data)

    def put_json(name, value):
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        require(json.loads(data) == value, f'JSON serialization changed {name}')
        products[name] = data

    try:
        protocol = read(OUT / 'protocol.json')
        previous = read(ROOT / 'reports/training_v4_4/run_001/protocol.json')
        require(protocol['predecessor_protocol_sha256'] == inputs['reports/training_v4_4/run_001/protocol.json'], 'Predecessor protocol changed')
        require(protocol['train_updates'] == 500 and protocol['checkpoint_interval'] == 100, 'Expected registered 500 updates and six checkpoints')
        require(protocol['selection_rule'] == EXPECTED_RULE, 'Selection rule changed')
        require(protocol['v4_4_additional_development_screen'] == previous['v4_4_additional_development_screen'], 'Inherited science thresholds changed')
        require(protocol['inherited_screen_unchanged'] and protocol['new_training_cases'] == 0 and not protocol['final_test_used'], 'Protocol scope differs')
        screen = protocol['v4_4_additional_development_screen']
        require(set(screen['signed_conditions']) == set(SIGNED), 'Signed screen groups changed')
        require(screen['template_relative_MSE_factor'] == 1.05 and screen['template_absolute_MSE_tolerance_e2'] == .25
                and screen['separate_signed_weak_pooled_L2_strictly_less_than'] == 1.
                and screen['all_development_invalid_output_fraction'] == 0., 'Frozen science limits differ')
        training_status = read(OUT / 'balanced_fixed_continuum/status.json')
        require(training_status['stage'] == 'COMPLETE' and training_status['optimizer_updates'] == 500, 'v4.5 training is not COMPLETE at 500 updates')
        receipts, checkpoint_by_arm = {}, {}
        for arm, spec in ARMS.items():
            folder = ROOT / spec['folder']
            receipt = read(folder / 'training_receipt.json')
            require(receipt['updates'] == 500 and receipt['microbatches'] == 2000, f'Unexpected completed training count: {arm}')
            checkpoint_by_arm[arm] = {item['update']: item for item in receipt['checkpoints']}
            require(set(checkpoint_by_arm[arm]) == set(UPDATES), f'Six-checkpoint ledger incomplete: {arm}')
            for update, checkpoint in checkpoint_by_arm[arm].items():
                require(checkpoint['Adam_steps'] == ([] if update == 0 else [update]), f'Adam step receipt mismatch: {arm}/{update}')
                location = (ROOT / checkpoint['path']).resolve()
                require(location.is_relative_to(ROOT.resolve() / 'training_v4_3')
                        or location.is_relative_to(ROOT.resolve() / 'training_v4_4')
                        or location.is_relative_to(ROOT.resolve() / 'training_v4_5'), 'Unexpected checkpoint location')
                actual = file_sha(location)
                require(actual == checkpoint['sha256'], f'Checkpoint bytes changed: {location}')
                inputs[str(location.relative_to(ROOT)).replace('\\', '/')] = actual
            receipts[arm] = receipt
        require(len({value['initial_parameters_sha256'] for value in receipts.values()}) == 1, 'Four arms did not share initial parameters')
        current_receipt = receipts[CURRENT]
        require(current_receipt['same_frozen_negative_balanced_500_draws'], 'Frozen balanced draw receipt missing')
        require(current_receipt['initial_full_development_summary_equals_v4_4'], 'Initial paired summary receipt missing')
        require(current_receipt['physical_buffers_unchanged'], 'Physical buffers changed')
        training_curve = read(OUT / 'balanced_fixed_continuum/training_curve.jsonl', True)
        require(len(training_curve) == 500, 'Actual training journal does not contain 500 updates')
        require(all(row['optimizer_update'] == index and row['status'] == 'SUCCESS' and len(row['samples']) == 4
                    for index, row in enumerate(training_curve, 1)), 'Training update/microbatch journal is inconsistent')
        validation_curve = read(OUT / 'balanced_fixed_continuum/validation_curve.jsonl', True)
        require([row['optimizer_update'] for row in validation_curve] == list(UPDATES), 'Validation journal does not contain exactly six checkpoints')
        for row in validation_curve:
            checkpoint = checkpoint_by_arm[CURRENT][row['optimizer_update']]
            require(row['checkpoint_sha256'] == checkpoint['sha256'] and row['checkpoint_path'] == checkpoint['path'], 'Validation/checkpoint reference mismatch')
        entries, pair_rows, interactions = [], [], []
        for update in UPDATES:
            four_rows = {}
            for arm, spec in ARMS.items():
                folder = ROOT / spec['folder']
                record_path = folder / f'validation_update_{update:04d}.jsonl'
                rows = read(record_path, True)
                require(len(rows) == len({row['case_id'] for row in rows}) == 776, f'Incomplete 776-frame record: {arm}/{update}')
                require(all(row['status'] == 'SUCCESS' for row in rows), 'Failed frames cannot be silently excluded')
                require(canonical([row['case_id'] for row in rows]) == protocol['validation_ids_sha256'], 'Validation population or order changed')
                require(all(row['raw_hash'] == row['physical']['raw_hash'] and row['label_sha256'] == row['physical']['label_sha256'] for row in rows), 'Row/physical identity mismatch')
                decision = read(folder / f'validation_summary_{update:04d}.json')
                gates = read(folder / f'gates_update_{update:04d}.json')
                require(decision['optimizer_update'] == update and decision['validation_cases'] == 776, 'Validation summary identity mismatch')
                legacy = legacy_checks(decision, gates)
                close(pooled(selected_group(rows, 'main_weak'))['template_L2'], decision['weak_L2'], 'Main weak pooled statistic mismatch')
                close(pooled(selected_group(rows, 'main_normal'))['template_RMSE_e'], decision['normal_RMSE_e'], 'Main normal statistic mismatch')
                signed = signed_summaries(rows, screen)
                require(all(value['frames'] == 6 for value in signed), 'Signed population changed')
                entry = {'arm': arm, 'label': spec['label'], 'update': update,
                         'checkpoint': checkpoint_by_arm[arm][update], 'decision': decision,
                         'original_gate_decisions': gates, 'legacy_checks': legacy,
                         'strata': strata(rows), 'signed_noisy_conditions': signed,
                         'validation_sha256': inputs[str(record_path.relative_to(ROOT)).replace('\\', '/')],
                         'science_checks': None, 'fixed_C_audit': None}
                if arm in ('v44_fixed', CURRENT):
                    published = read(folder / f'science_summary_{update:04d}.json')
                    require(published['validation_sha256'] == entry['validation_sha256'], 'Science record references different per-frame results')
                    expected_protocol = protocol if arm == CURRENT else previous
                    expected_path = 'reports/training_v4_5/run_001/protocol.json' if arm == CURRENT else 'reports/training_v4_4/run_001/protocol.json'
                    require(published['protocol_sha256'] == inputs[expected_path], 'Science record references a different frozen protocol')
                    entry['science_checks'] = check_science(rows, decision, published, signed, expected_protocol['v4_4_additional_development_screen'])
                    entry['published_science'] = published
                    audits = read(folder / f'forward_audit_{update:04d}.jsonl', True)
                    require(len(audits) == 776 and [row['case_id'] for row in audits] == [row['case_id'] for row in rows], 'Fixed-C forward audit population mismatch')
                    require(all(audit['raw_hash'] == row['raw_hash'] for audit, row in zip(audits, rows)), 'Forward audit raw identity differs')
                    require(all(audit['final_C_equals_physics'] and audit['temperature_equals_physics']
                                and audit['signed_state_equals_scaled_proposal'] for audit in audits), 'Fixed-C identity audit failed')
                    entry['fixed_C_audit'] = {'frames': len(audits), 'all_C_and_T_exact_physics': True,
                                              'temperature_identity_is_structural_not_learned': True,
                                              'raw_strata': raw_strata(audits)}
                else:
                    entry['signed_limits_used_for_descriptive_comparison_only'] = True
                entries.append(entry)
                four_rows[arm] = rows
            paired = match_four_cell(four_rows, update)
            pair_rows.extend(paired)
            interactions.extend(interaction_groups(paired, update))
        require(len(pair_rows) == 4656 and len(entries) == 24, 'Four-cell checkpoint coverage incomplete')
        initial_max = max(abs(row['descriptive_interaction_MSE_e2']) for row in pair_rows if row['optimizer_update'] == 0)
        require(initial_max == 0., 'Nonzero initial template interaction despite identical initial weights')
        candidates = choose_candidates(entries, protocol['selection_rule'])
        selected_entry = candidates[0] if candidates else None
        selected = None if selected_entry is None else {
            'name': 'balanced_fixed_continuum', 'update': selected_entry['update'],
            'checkpoint': selected_entry['checkpoint']['path'], 'checkpoint_sha256': selected_entry['checkpoint']['sha256'],
            'parameters_sha256': selected_entry['checkpoint']['parameters_sha256'],
            'forward_model': 'training_v4_4.model.FixedContinuumCNN',
            'model_source_sha256': file_sha(ROOT / 'training_v4_4/model.py'),
            'validation_sha256': selected_entry['validation_sha256'],
            'weak_L2': selected_entry['decision']['weak_L2'], 'normal_RMSE_e': selected_entry['decision']['normal_RMSE_e']}
        selection = {'rule': protocol['selection_rule'], 'scope': 'v4.5 six checkpoints only; same-weight complete gates',
                     'candidate_count': len(candidates), 'selected': selected,
                     'ranked_candidate_updates': [entry['update'] for entry in candidates],
                     'protocol_sha256': inputs['reports/training_v4_5/run_001/protocol.json'],
                     'final_test_used': False, 'independent_validation_established': False,
                     'no_cross_checkpoint_metric_splicing': True}
        failures = []
        for entry in entries:
            if entry['arm'] != CURRENT:
                continue
            failures.append({'update': entry['update'], 'checkpoint': entry['checkpoint'],
                             'legacy': entry['legacy_checks'], 'science': entry['science_checks'],
                             'signed_groups': entry['signed_noisy_conditions'],
                             'science_candidate': entry['science_checks']['science_candidate']})
        put_json('comparison_summary.json', {'arms': ARMS, 'entries': entries, 'final_test_used': False})
        put_json('candidate_selection.json', selection)
        put_json('checkpoint_failures.json', {'checkpoints': failures, 'all_six_retained': True,
                                             'failed_count': sum(not item['science_candidate'] for item in failures)})
        put_json('interaction_summary.json', {'formula': '(v45-v44)-(v43_balanced-v43_unbalanced)',
                                               'metric': 'template MSE in e2; macro mean of paired frame differences',
                                               'descriptive_only': True, 'independent_confirmation': False,
                                               'same_update_groups': interactions, 'four_cell_pairs': len(pair_rows),
                                               'all_recorded_raw_label_cache_hashes_match': True,
                                               'initial_maximum_absolute_interaction_e2': initial_max})
        products['paired_four_cell.jsonl'] = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in pair_rows).encode('utf-8')
        figures, figure_checks = render_figures(entries)
        products.update(figures)
        products['V4_5_REPORT.md'] = report_text(entries, interactions, selection, protocol).encode('utf-8')
        products['analyze_source.py'] = Path(__file__).read_bytes()
        for relative, expected in inputs.items():
            require(file_sha(ROOT / relative) == expected, f'An analysis input changed: {relative}')
        for name, data in products.items():
            path = attempt / name
            with path.open('xb') as handle:
                handle.write(data)
            require(path.read_bytes() == data, f'Output bytes did not reopen exactly: {name}')
        receipt = {'status': 'SUCCESS', 'created_at': datetime.now(timezone.utc).isoformat(),
                   'attempt_directory': str(attempt.relative_to(ROOT)).replace('\\', '/'),
                   'source_sha256': sha(products['analyze_source.py']), 'input_files_sha256': inputs,
                   'output_files_sha256': {name: sha(data) for name, data in products.items()},
                   'figure_reopen_checks': figure_checks, 'derived_files_reopened_exactly': True,
                   'completed_training_updates': 500, 'training_microbatches': 2000,
                   'checkpoints_per_arm': 6, 'comparison_arms': 4, 'entries': len(entries),
                   'same_update_four_cell_pairs': len(pair_rows), 'candidate_count': len(candidates),
                   'model_inference_calls': 0, 'physical_inverse_calls': 0, 'GPU_calls': 0,
                   'raw_or_label_array_files_opened': 0, 'final_test_used': False,
                   'old_analysis_and_data_audit_modified': False}
        receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        (attempt / 'analysis_receipt.json').write_bytes(receipt_bytes)
        # Publish only derived files after the entire attempt is verified.  The
        # attempt directory keeps previous versions even when a later analysis runs.
        for name, data in {**products, 'analysis_receipt.json': receipt_bytes}.items():
            if name == 'analyze_source.py':
                continue
            destination = OUT / name
            if destination.exists():
                prior = destination.read_bytes()
                if prior != data:
                    previous_copy = attempt / ('previous_' + name)
                    with previous_copy.open('xb') as handle:
                        handle.write(prior)
            temporary = OUT / (name + '.' + stamp + '.tmp')
            with temporary.open('xb') as handle:
                handle.write(data)
            temporary.replace(destination)
            require(destination.read_bytes() == data, f'Published bytes differ: {name}')
        print(json.dumps({'analysis_complete': True, 'checkpoints': 6, 'four_cell_pairs': len(pair_rows),
                          'candidate_count': len(candidates), 'report': str(OUT / 'V4_5_REPORT.md'),
                          'receipt_sha256': file_sha(OUT / 'analysis_receipt.json')}, ensure_ascii=False), flush=True)
        return receipt
    except BaseException as exc:
        failure = {'status': 'FAILED', 'time': datetime.now(timezone.utc).isoformat(),
                   'error_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                   'source_sha256': file_sha(Path(__file__)), 'input_files_sha256': inputs,
                   'completed_attempt_outputs': sorted(path.name for path in attempt.iterdir()),
                   'model_inference_calls': 0, 'GPU_calls': 0, 'final_test_used': False,
                   'failed_attempt_retained': True}
        (attempt / 'failure_record.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'analysis_failed': True, 'failure_record': str(attempt / 'failure_record.json'),
                          'message': str(exc)}, ensure_ascii=False), flush=True)
        raise


def self_test(with_plots=False):
    """Small in-memory fixtures only; never publish mock candidate/report files."""
    def candidate(update, weak, normal, passed=True, arm=CURRENT):
        return {'arm': arm, 'update': update, 'decision': {'weak_L2': weak, 'normal_RMSE_e': normal},
                'science_checks': {'science_candidate': passed}}
    tests = [candidate(300, .8, 9), candidate(200, .8, 9), candidate(100, .8, 10),
             candidate(500, .7, 11), candidate(0, .1, 1, False), candidate(100, .01, 1, True, 'v43_unbalanced')]
    require([entry['update'] for entry in choose_candidates(tests, EXPECTED_RULE)] == [500, 200, 300, 100], 'Candidate sort fixture failed')
    require(choose_candidates([candidate(0, .9, 10, False)], EXPECTED_RULE) == [], 'Null candidate fixture failed')
    base = {'case_id': 'fixture', 'mother_id': 'm0', 'condition': 'weak', 'group': 'weak', 'geometry': 'flat',
            'input_kind': 'noise', 'signed_aux': False, 'raw_hash': 'r', 'label_sha256': 'l',
            'physical': {'cache_key': 'k', 'cache_sha256': 'c', 'cache_metadata_sha256': 'm'},
            'metrics': {'pixels': 8, 'truth_energy': 80, 'template_MSE_e2': 10.}}
    four = {}
    for arm, mse in zip(ARMS, (10., 6., 8., 7.)):
        row = json.loads(json.dumps(base))
        row['metrics']['template_MSE_e2'] = mse
        four[arm] = [row]
    require(match_four_cell(four, 0)[0]['descriptive_interaction_MSE_e2'] == 3., 'Interaction sign fixture failed')
    four['v43_balanced'][0]['physical']['cache_sha256'] = 'changed'
    try:
        match_four_cell(four, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed cache bytes were not rejected')
    # The strict weak rule is evaluated without a numerical pass tolerance.
    require(not (math.sqrt(10. / 10.) < 1.), 'Weak boundary must remain strict')
    require(1.3 <= 1.05 * 1. + .25, 'MSE boundary must remain non-strict')
    checks = {'selection_sort_and_scope': True, 'null_candidate': True,
              'interaction_direction': True, 'cache_hash_mismatch_rejected': True,
              'strict_and_nonstrict_boundaries': True, 'fixture_only': True,
              'model_inference_calls': 0, 'real_final_results_created': False}
    if with_plots:
        configure_output()
        entries = []
        for index, arm in enumerate(ARMS):
            for update in UPDATES:
                entries.append({'arm': arm, 'update': update,
                    'decision': {'weak_L2': .7 + index * .06 + update / 10000,
                                 'normal_RMSE_e': 10 + index + update / 500},
                    'signed_noisy_conditions': [
                        {'condition': condition, 'template_L2': .8 + index * .1 + update / 10000,
                         'template_RMSE_e': 15 + index * 5 + update / 500, 'MSE_limit_e2': 420.,
                         'energy_weighted_signed_projection': .2 + index * .2 + update / 2000,
                         'macro_recall_at_fixed_60e': .1 + index * .1 + update / 2500}
                        for condition in SIGNED]})
        figures, verification = render_figures(entries)
        require(len(figures) == len(verification) == 4, 'In-memory figure product count differs')
        checks['in_memory_fixture_figure_reopen_checks'] = verification
    print(json.dumps(checks, ensure_ascii=False))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Run in-memory fixtures; do not analyze real results')
    parser.add_argument('--with-plots', action='store_true', help='Also render/reopen fixture plots in memory during self-test')
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test(arguments.with_plots)
    else:
        require(not arguments.with_plots, '--with-plots is a fixture self-test option')
        run()
