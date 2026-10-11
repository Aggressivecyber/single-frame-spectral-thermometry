"""Evidence tables and delivery QA from saved artifacts only, no model inference."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
from evaluation_v2.common import array_hash, Journal
from evaluation_v2.metrics import aggregate_metrics
from post_metrics import source_halo_metrics

OUT = Path(__file__).resolve().parent
DATA = ROOT/'outputs/audit_v2/run_001'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def f(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def main():
    manifest = read(OUT/'frozen_manifest.json')
    config = read(OUT/'active_configuration.json')['config_sha256']
    all_rows = Journal(OUT/'per_case.jsonl').rows()
    rows = list({r['identity_sha256']: r for r in all_rows if r['config_sha256'] == config and not (
        r['method'] in ('B4_known_true_calibration', 'B1_known_700_905_ratio') and r['suite'] != 'legacy')}.values())
    assert len(rows) == 954
    assert sum(r['status'] == 'SUCCESS' for r in rows) == 946
    assert sum(r['status'] == 'SATURATION_REJECTED' for r in rows) == 8
    assert not any(r['status'] in ('EXCEPTION', 'TIMEOUT') for r in rows)
    checked = 0
    for fixture in manifest['cases']:
        path = DATA/fixture['path']
        assert runtime.sha256(path) == fixture['file_sha256']
        with np.load(path, allow_pickle=False) as v:
            assert all(array_hash(v[k]) == h for k, h in fixture['array_sha256'].items())
        checked += 1
    by_case = collections.defaultdict(list)
    for row in rows:
        by_case[row['case_id']].append(row)
        if row['status'] == 'SUCCESS':
            path = DATA/row['prediction_path']
            assert runtime.sha256(path) == row['prediction_sha256']
            with np.load(path, allow_pickle=False) as v:
                assert {'template_e', 'temperature_K', 'valid'} == set(v.files)
                assert np.isfinite(v['template_e']).all()
                if row['metrics']['temperature_applicable']:
                    assert np.isfinite(v['temperature_K']).all()
            if row.get('trace_path'):
                assert runtime.sha256(DATA/row['trace_path']) == row['trace_sha256']
    assert all(len({r['input_hash'] for r in group}) == 1 for group in by_case.values())
    # Prior attempt files stay readable with their original journal hashes.
    prior = [r for r in all_rows if r['config_sha256'] != config]
    assert all(runtime.sha256(DATA/r['prediction_path']) == r['prediction_sha256'] for r in prior if r.get('prediction_path'))
    parents = collections.defaultdict(list)
    valparents = set()
    for fixture in manifest['cases']:
        if fixture['suite'] == 'smoke':
            parents[fixture['metadata']['parent_scene_id']].append(fixture)
        if fixture['suite'] == 'validation':
            valparents.add(fixture['metadata']['parent_scene_id'])
    assert len(parents) == 12 and not set(parents) & valparents
    for group in parents.values():
        assert len(group) == 4
        assert len({c['array_sha256']['temperature_latent_K'] for c in group}) == 1
        assert len({(c['metadata']['amplitude_min'], c['metadata']['amplitude_max'], c['metadata']['geometry_seed']) for c in group}) == 1
    receipts = [read(OUT/f'local_{suite}_launcher_receipt.json') for suite in ('diagnostics', 'legacy', 'smoke')]
    assert all(r['child_exit_code'] == 0 for r in receipts)
    preservation = read(OUT/'baseline_preservation.json')
    assert preservation['all_unchanged'] and preservation['original_tracked_files'] == 194
    verification = {'execution_status': 'ALL_REGISTERED_INPUTS_HANDLED_WITH_EXPLICIT_REJECTIONS',
                    'registered_method_input_combinations': 954, 'successful': 946,
                    'saturation_rejected': 8, 'remaining': 0, 'exceptions': 0, 'timeouts': 0,
                    'frozen_containers_and_all_inner_array_hashes_verified': checked,
                    'success_predictions_reopened_and_hash_verified': 946,
                    'prior_partial_revision_records_retained_and_artifacts_verified': len(prior),
                    'unregistered_resume_probe_outcomes_retained_excluded': 2,
                    'orchestration_fix_numerical_functions_AST_verified_unchanged': True,
                    'same_actual_input_hash_across_methods': True,
                    'smoke_parents': 12, 'validation_parents': len(valparents), 'parent_overlap': 0,
                    'peak_observed_audit_process_tree_host_RSS_MiB': max(r['peak_observed_host_working_set_MiB'] for r in receipts),
                    'scientific_acceptance': 'not_accepted', 'optimizer_updates': 0,
                    'original_checkpoint_and_194_tracked_files_unchanged': True,
                    'numerical_execution_passed_for_every_registered_input': False,
                    'reason_for_numerical_failures': 'Intentional full-frame rejection of one old saturated fixture by eight methods',
                    'journal_sha256': runtime.sha256(OUT/'per_case.jsonl'),
                    'frozen_manifest_sha256': runtime.sha256(OUT/'frozen_manifest.json'),
                    'evaluation_registration_sha256': runtime.sha256(OUT/'evaluation_registration_v002.json')}
    runtime.write_json(OUT/'delivery_verification.json', verification)
    results = {'front': [], 'hot_background': [], 'known_calibration': [], 'morphology_budget': [], 'legacy_temperature_baselines': []}
    source_halo_rows = []
    detector_calibration = read(OUT/'detector_calibration.json')
    fixtures = {f['case_id']: f for f in manifest['cases']}
    for r in rows:
        if r['status'] != 'SUCCESS':
            continue
        m, meta = r['metrics'], r['metadata']
        if meta.get('partition'):
            with np.load(DATA/fixtures[r['case_id']]['path'], allow_pickle=False) as truth, np.load(DATA/r['prediction_path'], allow_pickle=False) as prediction:
                halo = source_halo_metrics(truth['template_e'], prediction['template_e'], truth['source_image_domain'],
                            truth['evaluation_domain'], detector_calibration['detectors'][r['method']]['pixel_threshold_e'])
            source_halo_rows.append({'case_id': r['case_id'], 'method': r['method'], 'partition': meta['partition'], **halo})
        distilled = {'case_id': r['case_id'], 'method': r['method'], 'input_kind': r['input_kind'],
                     'template_RMSE_e': m['template_RMSE_e'], 'template_L2': m['template_relative_L2'],
                     'amplitude_projection': m['amplitude_projection'], 'orthogonal_relative_L2': m['orthogonal_relative_L2'],
                     'color_RMSE_K': m['temperature_observed_RMSE_K'], 'reference_RMSE_K': m['temperature_reference_RMSE_K'],
                     'historical60_null_FP': m['historical60_null_FP_fixed_domain'],
                     'calibrated_null_FP': m['null_FP_fixed_domain'],
                     'raw_reprojection_RMSE_e': r.get('raw_reprojection_RMSE_e'),
                     'raw_standardized_residual_RMS': r.get('raw_standardized_residual_RMS')}
        if r['case_id'].startswith('front_'):
            distilled['trace_initial_and_first_data'] = [{
                'stage': step['label'], 'template_RMSE_e': step['metrics']['template_RMSE_e'],
                'historical60_null_FP': step['metrics']['historical60_null_FP_fixed_domain']}
                for step in r.get('trace', [])[:3]]
            results['front'].append(distilled)
        if r['suite'] == 'diagnostics' and meta['family'] == 'hot_background_scan':
            results['hot_background'].append({**distilled, 'temperature_K': meta['uniform_temperature_K'],
                            'variant': meta['variant'], 'actual_486_extra_e': meta['actual_mean_486_extra_e']})
        if r['suite'] == 'legacy' and meta['family'] == 'channel_gain_3pct':
            results['known_calibration'].append(distilled)
        if r['suite'] == 'diagnostics' and meta['family'] == 'morphology_budget':
            results['morphology_budget'].append({**distilled, 'morphology': meta['morphology'],
                    'control': meta['normalization_control'], 'truth_peak_preblur_e': meta['actual_template_peak_preblur_e'],
                    'truth_peak_postblur_e': meta['actual_template_peak_postblur_e'],
                    'truth_integral_e': meta['actual_template_integral_e'], 'truth_L2_energy_e2': meta['actual_template_L2_energy_e2']})
    for method in ('B0_constant_train', 'B0_mean_field_train', 'B4_published'):
        subset = [r for r in rows if r['suite'] == 'legacy' and r['metadata'].get('partition') and r['method'] == method]
        results['legacy_temperature_baselines'].append({'method': method, **aggregate_metrics(subset)})
    runtime.write_json(OUT/'controlled_diagnostics_summary.json', results)
    runtime.write_json(OUT/'source_domain_halo_correction.json', source_halo_rows)
    paired = read(OUT/'paired_parent_effects.json')
    cnn = [p for p in paired if p['contrast'] == 'B4_published minus B2_six_stage_no_CNN']
    runtime.write_json(OUT/'CNN_paired_effects.json', cnn)
    def get(case, method):
        return next(r for r in rows if r['case_id'] == case and r['method'] == method)
    a = get('front_continuous_e', 'B1_demod_fit')['metrics']
    b = get('front_continuous_e', 'B4_published')['metrics']
    t0 = get('front_continuous_e', 'B4_published')['trace'][0]['metrics']
    t1 = get('front_continuous_e', 'B4_published')['trace'][1]['metrics']
    known = {r['method']: r for r in results['known_calibration']}
    flat = [r for r in rows if r['suite'] == 'legacy' and r['scene'] == 'flat_flame_OH' and r['condition'] == 'signal_weak']
    flat_model = aggregate_metrics([r for r in flat if r['method'] == 'B4_published'])
    flat_zero = aggregate_metrics([r for r in flat if r['method'] == 'Z0_zero_template'])
    lines = ['## 本轮证据支持的判断', '',
             f'1. **温度前沿空白不只是噪声问题。** 同一无噪声、无量化电子输入，B1 色温 RMSE 为 {f(a["temperature_observed_RMSE_K"])} K、模板 RMSE {f(a["template_RMSE_e"])} e，历史 >60 e 误报为 0；B4 为 {f(b["temperature_observed_RMSE_K"])} K、{f(b["template_RMSE_e"])} e，误报 {100*b["historical60_null_FP_fixed_domain"]:.3f}%。在这一已知名义构造场景中，测量含有有效色温信息，当前展开求解未保持 B1 的理想输入恢复结果。不能据此保证任意未知发射率下 latent 温度唯一可辨。', '',
             f'2. **伪模板在第一步物理更新便被放大。** 无量化电子路径中，初始化模板 RMSE {f(t0["template_RMSE_e"])} e、误报 {f(t0["historical60_null_FP_fixed_domain"])}，首次 data-step 变为 {f(t1["template_RMSE_e"])} e、误报 {100*t1["historical60_null_FP_fixed_domain"]:.2f}%。CNN 随后的修正会降低该误差；不能把首次异常全部归因于 CNN。优先检查初始化由解调后空间域映射到物理状态时的偏差、温度/幅度/模板耦合和物理更新精度；adjoint/JVP 测试通过不等于优化器已经充分利用信息。', '',
             f'3. **平焰弱信号确实劣于零模板。** 12 个旧冻结帧重算 B4 pooled L2={f(flat_model["pooled_template_L2"])}；Z0={f(flat_zero["pooled_template_L2"])}，平方误差比约 {flat_model["pooled_template_L2"]**2:.3f}。v2 的支持域/召回修复没有改变此回归事实。应查看幅度投影、正交残差及源外伪结构，而不是仅解释为分母小。', '',
             f'4. **已知标定的改善支持优先处理标定。** 同一交替 ±3% 增益输入，名义 B4 相对 reference 色温误差 {f(known["B4_published"]["reference_RMSE_K"])} K，给定正确 A_true 的 B4 为 {f(known["B4_known_true_calibration"]["reference_RMSE_K"])} K，已知比值修正 B1 为 {f(known["B1_known_700_905_ratio"]["reference_RMSE_K"])} K。三者输入完全相同、CNN 参数保持不变。后两者是独立给定标定的诊断，不是从单帧自主恢复增益。', '',
             '**CNN 增益按母场景配对给出，而不是由一次过线定义：** 下表为 B4−B2；负值代表 B4 错误较小。12 个母场景、2000 次按母场景 bootstrap，95% 区间是本合成 smoke 集上的描述性不确定性。', '',
             '|条件|指标|平均差 B4−B2|95% 区间|', '|---|---|---:|---|']
    for effect in cnn:
        if effect['mean_effect'] is not None:
            interval = effect['group_bootstrap_95pct_interval']
            lines.append(f'|{effect["condition"]}|{effect["metric"]}|{f(effect["mean_effect"])}|[{f(interval[0])}, {f(interval[1])}]|')
    lines += ['', '**固定相对和固定绝对失配分开看：** 以下为 B4，只有一个受控母场景，不能做广泛温度泛化结论。', '',
              '|均匀温度 K|486 失配方式|实际新增 e|模板 RMSE e|历史 >60e 误报|reference 色温误差 K|', '|---:|---|---:|---:|---:|---:|']
    for r in results['hot_background']:
        if r['method'] == 'B4_published':
            lines.append(f'|{r["temperature_K"]:.0f}|{r["variant"]}|{f(r["actual_486_extra_e"])}|{f(r["template_RMSE_e"])}|{100*r["historical60_null_FP"]:.3f}%|{f(r["reference_RMSE_K"])}|')
    lines += ['', '随温度增亮，+5% 的绝对误差电子数明显改变；固定 +100 e 时模板绝对 RMSE 更接近，误报趋势也不同。700/905 未受该 486 单波段扰动，因此这组中应分别判断色温和模板的可信度，不输出一个总准确率。原始残差也必须同时看，较小复投影不是正确模板或温度的证明。', '',
              '**形态/光子预算：** 归一化在生成端完成，以下真实目标来自冻结数组，不缩放预测。B4：', '',
              '|生成控制|形态|postblur峰值 e|积分 e|真值 L2 能量 e²|预测 L2|幅度投影|正交相对残差|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results['morphology_budget']:
        if r['method'] == 'B4_published':
            lines.append(f'|{r["control"]}|{r["morphology"]}|{f(r["truth_peak_postblur_e"])}|{f(r["truth_integral_e"],1)}|{f(r["truth_L2_energy_e2"],1)}|{f(r["template_L2"])}|{f(r["amplitude_projection"])}|{f(r["orthogonal_relative_L2"])}|')
    lines += ['', '**B0 和数据生成先验：** 旧真实形态衍生数据仍共享很窄的人工温度几何。平均训练温度场在这组上的误差应与网络并列，不能因 72 张形态图而声称有 72 个独立实测温度场：', '',
              '|旧形态衍生帧温度方法|pooled observed 色温 RMSE K|', '|---|---:|']
    for r in results['legacy_temperature_baselines']:
        lines.append(f'|{r["method"]}|{f(r["pooled_temperature_RMSE_K"])}|')
    lines += ['', '下一步应先研究已有小网络的物理更新、初始化空间域和训练温度几何，并优先建立可信标定；本轮不训练。保持独立锁定新母场景用于未来最终验证。弱信号、空白与不同背景需分层，CNN 在 B2 对照上有改善并不意味着超过充分迭代物理/经验证调参的传统基线或满足科学接收。', '',
              '源外信号另作光学扩散校正：v2.2 的 outside_source_template_RMSE_e 表示源外预测幅度，不能把合法的 postblur 光学扩散全部称作伪信号。`post_metrics.py` 正式后评分器在相同冻结目标上新增源外预测减真值误差、真值光学扩散 RMS 和真值支持区之外检出率，保存于 `source_domain_halo_correction.json`，不修改或重推理旧预测。两项新增生产测试覆盖完美扩散误差为零、源外伪结构和非有限结果，见 `tests_post_metrics.log/xml`。', '',
              '实际全套 resume 探针发现 known-calibration 的母帧选择依赖套件顺序：误选新 smoke 母帧产生一条 metadata 缺项异常和一条额外比值诊断，均在 JSONL 保留并排除预注册 cohort。已固定显式选择原 legacy gain-null 帧，并新增生产测试。`orchestration_compatibility_v003.json` 证明模型/求解/物理/评分源码及执行 work 的 AST 均未改动，原 954 个已注册组合的数值配置保留，可精确复用成功记录；两条未注册探针不算作首轮科学对照。源码一致性曾拦截过旧报告模块恢复成较早版本的启动，旧启动收据已保留并恢复到封存源码哈希，未改数据/模型。', '',
              '完整数值见 `controlled_diagnostics_summary.json`、`CNN_paired_effects.json`；最终交付核验见 `delivery_verification.json`。所有 167 个冻结容器及内层数组哈希、946 个成功预测、旧 152 条记录的原预测文件均重开校验。复现此补充只需 `python -B reports/audit_v2/run_001/finalize_report.py`，不推理。', '']
    supplemental = '\n'.join(lines)
    (OUT/'EVIDENCE_FINDINGS.md').write_text(supplemental, encoding='utf-8')
    report = OUT/'AUDIT_V2_REPORT.md'
    text = report.read_text(encoding='utf-8')
    marker = '## 本轮证据支持的判断'
    # Idempotent: a second delivery-QA call updates the supplement, not duplicate it.
    if marker in text:
        text = text[:text.index(marker)]
    report.write_text(text.rstrip()+'\n\n'+supplemental, encoding='utf-8')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
