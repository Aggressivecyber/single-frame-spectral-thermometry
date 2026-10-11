"""Reopen report artifacts, verify seals, and write a bounded delivery receipt."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
from PIL import Image

OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    tests = []
    for name in ('tests_gate.xml', 'tests_post_metrics.xml', 'tests_orchestration.xml'):
        root = ET.parse(OUT / name).getroot()
        suites = root.findall('testsuite') or [root]
        n = sum(int(s.attrib['tests']) for s in suites)
        assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
        tests.append({'receipt': name, 'passed': n, 'sha256': runtime.sha256(OUT / name)})
    assert sum(t['passed'] for t in tests) == 40
    initial = read('baseline_tracked_files.json')['initial_tracked_sha256']
    assert len(initial) == 194 and all(runtime.sha256(ROOT / name) == value for name, value in initial.items())
    compatibility = read('orchestration_compatibility_v003.json')
    assert all(runtime.sha256(ROOT / name) == value for name, value in compatibility['current_sources'].items())
    assert runtime.sha256(OUT / 'evaluation_registration_v002.json') == compatibility['original_registration_sha256']
    delivery = read('delivery_verification.json')
    assert delivery['registered_method_input_combinations'] == 954 and delivery['successful'] == 946
    assert delivery['saturation_rejected'] == 8 and delivery['remaining'] == 0
    assert delivery['exceptions'] == 0 and delivery['optimizer_updates'] == 0
    assert runtime.sha256(OUT / 'per_case.jsonl') == delivery['journal_sha256']
    assert runtime.sha256(OUT / 'frozen_manifest.json') == delivery['frozen_manifest_sha256']
    figures = []
    for path in sorted((OUT / 'figures').glob('*.png')):
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            assert im.width >= 1000 and im.height >= 400
            assert any(low != high for low, high in im.convert('RGB').getextrema())
            figures.append({'path': str(path.relative_to(OUT)), 'width': im.width,
                            'height': im.height, 'sha256': runtime.sha256(path)})
    assert len(figures) == 8
    correction = read('figure_input_correction.json')
    assert correction['corrected_figure_sha256'] == runtime.sha256(OUT / 'figures/front_continuous_e.png')
    report = OUT / 'AUDIT_V2_REPORT.md'
    text = report.read_text(encoding='utf-8')
    text = text.replace('生产测试通过 37 项，见 `tests_gate.log/xml`。',
                        '原注册门禁通过 37 项（`tests_gate.log/xml`），后续源外光学扩散评分 2 项和母帧选择 1 项也通过，共 40 项；见 `package_verification.json`。')
    if '```powershell' in text:
        start = text.index('```powershell')
        end = text.index('```', start + len('```powershell')) + 3
        text = text[:start] + '当前本机的可执行复现命令见 [REPRODUCE_LOCAL_GPU.md](REPRODUCE_LOCAL_GPU.md)。复用此 run 时只续跑/汇总，首次冻结和注册命令不应再次执行。' + text[end:]
    text = text.replace('source_valid_fraction>0.5 源图支持区、>0 敏感性及源外伪信号',
                        'source_valid_fraction>0.5 源图支持区、>0 敏感性及源外预测幅度；补充评分减去 postblur 真值，分开合法光学扩散和源外重建误差')
    report.write_text(text, encoding='utf-8')
    receipt = {
        'unit_tests_passed': sum(t['passed'] for t in tests), 'test_groups': tests,
        'original_tracked_files_unchanged_live': len(initial),
        'current_sources_match_compatible_seal': True,
        'immutable_numerical_registration_unchanged': True,
        'checkpoint_sha256': runtime.sha256(ROOT / 'checkpoints/selected_checkpoint.pt'),
        'registered_method_input_combinations': 954, 'successful': 946,
        'saturation_rejected': 8, 'remaining': 0, 'optimizer_updates': 0,
        'figures_reopened': figures,
        'representative_figures_for_visual_review': ['front_stage_diagnostics.png', 'front_continuous_e.png',
                              'legacy_experimental_shape_test_024_signal_weak.png',
                              'saturation_rejection_historical.png'],
        'automatic_image_check_scope': 'PIL integrity, reopen and dimensions; visual review performed separately',
        'report_sha256': runtime.sha256(report),
        'reproduction_instructions_sha256': runtime.sha256(OUT / 'REPRODUCE_LOCAL_GPU.md'),
        'data_and_report_root': str(ROOT), 'scientific_acceptance': 'not_accepted',
        'source_inputs_predictions_full_hash_check_receipt': 'delivery_verification.json',
    }
    runtime.write_json(OUT / 'package_verification.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('unit_tests_passed', 'original_tracked_files_unchanged_live',
          'current_sources_match_compatible_seal', 'registered_method_input_combinations', 'successful',
          'saturation_rejected', 'remaining', 'optimizer_updates')}))


if __name__ == '__main__':
    main()
