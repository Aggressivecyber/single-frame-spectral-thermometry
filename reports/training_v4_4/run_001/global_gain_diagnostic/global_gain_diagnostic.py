"""Stdlib-only necessary-condition diagnostic for a single global signed gain.

Read completed development JSON sidecars only.  Never load arrays, checkpoints,
Torch or a model.  Each invocation creates new immutable run files beside this
script, including a source snapshot; a later invocation can discover additional
completed checkpoints without changing an earlier result.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
D = Decimal
ZERO, ONE = D(0), D(1)
SIGNED = ('negative_weak', 'negative_normal', 'positive_weak', 'positive_normal')
U32 = D(2) ** -24
ROOT_PAD = D('1e-20')
ABS_STAT_BOUND = D('1e-9')


def dec(value):
    return D(str(value))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def store_new(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def poly(coefficients, value):
    a, b, c = coefficients
    return (a * value + b) * value + c


def intersect(intervals):
    if not intervals or any(item is None for item in intervals):
        return None
    lo = max(item[0] for item in intervals)
    hi = min(item[1] for item in intervals)
    lc = all(item[2] for item in intervals if item[0] == lo)
    hc = all(item[3] for item in intervals if item[1] == hi)
    if lo > hi or (lo == hi and not (lc and hc)):
        return None
    return lo, hi, lc, hc


def interval_json(interval):
    if interval is None:
        return {'empty': True}
    lo, hi, lc, hc = interval
    return {'empty': False, 'lower': float(lo), 'upper': float(hi),
            'lower_decimal': str(lo), 'upper_decimal': str(hi),
            'lower_closed': lc, 'upper_closed': hc}


def root_check(coefficients, root, repeated=False):
    a, b, c = coefficients
    scale = max(ONE, abs(a) + abs(b) + abs(c))
    residual = abs(poly(coefficients, root))
    assert residual <= D('1e-60') * scale
    check = {'root_decimal': str(root), 'root': float(root),
             'absolute_polynomial_residual_e2': str(residual), 'passed': True}
    if ZERO < root < ONE and a > 0 and not repeated:
        vertex = -b / (2 * a)
        lo, hi = ((ZERO, min(ONE, vertex)) if root < vertex
                  else (max(ZERO, vertex), ONE))
        flo, fhi = poly(coefficients, lo), poly(coefficients, hi)
        assert flo * fhi <= 0
        for _ in range(160):
            mid = (lo + hi) / 2
            fm = poly(coefficients, mid)
            if fm == 0:
                lo = hi = mid
                break
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        difference = abs((lo + hi) / 2 - root)
        assert difference < D('1e-45')
        check['independent_bisection_difference_lambda'] = str(difference)
    return check


def solve(coefficients, strict, *, outward=False):
    """Convex quadratic sublevel set, intersected with the exact domain [0,1]."""
    a, b, c = coefficients
    assert a >= 0
    roots = []
    if a == 0:
        if b == 0:
            interval = (ZERO, ONE, True, True) if (c < 0 if strict else c <= 0) else None
            return interval, roots
        root = -c / b
        roots.append(root_check(coefficients, root))
        if b > 0:
            interval = (ZERO, min(ONE, root), True, root > ONE or not strict)
        else:
            interval = (max(ZERO, root), ONE, root < ZERO or not strict, True)
    else:
        discriminant = b * b - 4 * a * c
        if discriminant < 0 or (discriminant == 0 and strict):
            return None, roots
        square_root = discriminant.sqrt()
        left, right = (-b - square_root) / (2 * a), (-b + square_root) / (2 * a)
        roots = [root_check(coefficients, value, discriminant == 0)
                 for value in sorted(set((left, right)))]
        interval = (max(ZERO, left), min(ONE, right),
                    left < ZERO or not strict, right > ONE or not strict)
    interval = intersect([interval, (ZERO, ONE, True, True)])
    if interval is not None:
        lo, hi, lc, hc = interval
        mid = (lo + hi) / 2
        value = poly(coefficients, mid)
        assert value <= D('1e-60') * max(ONE, abs(a) + abs(b) + abs(c))
        if strict and lo < hi:
            assert value < 0
        if outward:
            interval = (max(ZERO, lo - ROOT_PAD), min(ONE, hi + ROOT_PAD), True, True)
    return interval, roots


def minimum_value(coefficients):
    a, b, _ = coefficients
    points = [ZERO, ONE]
    if a:
        points.append(min(ONE, max(ZERO, -b / (2 * a))))
    return min(poly(coefficients, value) for value in points)


def statistic_bound(value, pixels):
    # Pessimistic sequential float32 reduction, plus a reserve for subtraction,
    # squaring, division and sqrt.  Actual tree reductions are usually tighter.
    count = D(pixels + 16)
    gamma = count * U32 / (1 - count * U32)
    assert gamma < D('.5')
    # Convert a relative-to-exact error bound into an observed-centered bound.
    return abs(value) * gamma / (1 - gamma) + ABS_STAT_BOUND


def frame_stats(row):
    pixels = row['metrics']['pixels']
    assert isinstance(pixels, int) and pixels > 0
    base = dec(row['physical']['template_MSE_e2'])
    model = dec(row['metrics']['template_MSE_e2'])
    correction = dec(row['template_correction_RMS_e'])
    delta2 = correction ** 2
    truth_mean_square = dec(row['metrics']['truth_energy']) / pixels
    assert all(value.is_finite() and value >= 0
               for value in (base, model, correction, truth_mean_square))
    eb, em, ed, et = [statistic_bound(value, pixels)
                      for value in (base, model, delta2, truth_mean_square)]
    # Sidecar-only consistency: endpoint-error norms and correction norm must
    # obey the triangle inequality after the same conservative uncertainty.
    blo, bhi = max(ZERO, base - eb).sqrt(), (base + eb).sqrt()
    dlo, dhi = max(ZERO, delta2 - ed).sqrt(), (delta2 + ed).sqrt()
    min_model = max(ZERO, blo - dhi, dlo - bhi) ** 2
    max_model = (bhi + dhi) ** 2
    assert model + em >= min_model and model - em <= max_model, row['case_id']
    return {'row': row, 'pixels': pixels, 'base': base, 'model': model,
            'delta2': delta2, 'truth_mean_square': truth_mean_square,
            'eb': eb, 'em': em, 'ed': ed, 'et': et}


def aggregate(frames, pooled):
    weights = [D(frame['pixels']) if pooled else ONE for frame in frames]
    total = sum(weights)
    return {name: sum(weight * frame[name] for weight, frame in zip(weights, frames)) / total
            for name in ('base', 'model', 'delta2', 'truth_mean_square', 'eb', 'em', 'ed', 'et')}


def make_constraint(name, frames, *, pooled, limit, strict, limit_error):
    stats = aggregate(frames, pooled)
    coefficients = (stats['delta2'], stats['model'] - stats['base'] - stats['delta2'],
                    stats['base'] - limit)
    assert abs(poly(coefficients, ZERO) - (stats['base'] - limit)) < D('1e-60')
    assert abs(poly(coefficients, ONE) - (stats['model'] - limit)) < D('1e-60')
    # M(lambda)=(1-lambda)B+lambda*M-lambda*(1-lambda)D.  On [0,1],
    # its uncertainty is <= eb+em+ed/4; adding the limit error expands every
    # potentially feasible value.  Strict inequalities are closed only here.
    tolerance = stats['eb'] + stats['em'] + stats['ed'] / 4 + limit_error
    nominal, roots = solve(coefficients, strict)
    relaxed_coefficients = (coefficients[0], coefficients[1], coefficients[2] - tolerance)
    relaxed, relaxed_roots = solve(relaxed_coefficients, False, outward=True)
    result = {
        'name': name, 'frames': len(frames), 'pixels': sum(frame['pixels'] for frame in frames),
        'aggregation': 'pixel-weighted mean for pooled SSE/truth energy' if pooled else 'macro mean MSE per frame',
        'case_ids': [frame['row']['case_id'] for frame in frames],
        'strict': strict, 'limit_e2': float(limit), 'limit_decimal': str(limit),
        'mse_lambda_0_e2': float(stats['base']), 'mse_lambda_1_e2': float(stats['model']),
        'correction_energy_e2': float(stats['delta2']),
        'q_definition': 'q(lambda)=MSE(lambda)-limit; require q<0 or q<=0 as strict specifies',
        'q_coefficients_decimal': [str(value) for value in coefficients],
        'nominal_interval': interval_json(nominal),
        'nominal_roots_checked': roots,
        'nominal_minimum_q_e2': float(minimum_value(coefficients)),
        'conservative_numerical_tolerance_e2': float(tolerance),
        'conservative_numerical_tolerance_decimal': str(tolerance),
        'uncertainty_components_e2': {key: float(value) for key, value in {
            'base': stats['eb'], 'model': stats['em'], 'delta2': stats['ed'], 'limit': limit_error}.items()},
        'conservative_outer_interval': interval_json(relaxed),
        'relaxed_roots_checked': relaxed_roots,
        'relaxed_minimum_q_e2': float(minimum_value(relaxed_coefficients)),
    }
    return result, nominal, relaxed


def intersection_certificate(named):
    interval = intersect([item[1] for item in named])
    result = {'interval': interval_json(interval)}
    if interval is not None:
        return result
    empty = [name for name, value in named if value is None]
    if empty:
        result.update(reason='one_or_more_individual_constraints_have_empty_intervals',
                      individually_impossible_constraints=empty)
        return result
    lower = max(value[0] for _, value in named)
    upper = min(value[1] for _, value in named)
    result.update(
        reason='incompatible_lower_and_upper_bounds',
        lower_bound_constraints=[name for name, value in named if value[0] == lower],
        upper_bound_constraints=[name for name, value in named if value[1] == upper],
        required_lower_lambda=float(lower), permitted_upper_lambda=float(upper),
        lower_minus_upper_lambda=float(lower - upper),
        required_lower_decimal=str(lower), permitted_upper_decimal=str(upper),
    )
    return result


def self_test():
    expected = (D('.25'), D('.75'), False, False)
    assert solve((ONE, -ONE, D('.1875')), True)[0] == expected
    assert solve((ONE, -ONE, D('.25')), True)[0] is None
    assert solve((ONE, -ONE, D('.25')), False)[0] == (D('.5'), D('.5'), True, True)
    assert solve((ZERO, ZERO, ONE), False)[0] is None
    assert solve((ZERO, ZERO, -ONE), True)[0] == (ZERO, ONE, True, True)
    assert solve((ZERO, ONE, D('-.4')), True)[0] == (ZERO, D('.4'), True, False)
    assert solve((ZERO, -ONE, D('.6')), True)[0] == (D('.6'), ONE, False, True)
    assert intersect([(ZERO, D('.2'), True, True), (D('.4'), ONE, True, True)]) is None
    # Two small scalar error/correction vectors, without any dataset or model.
    base_errors, corrections = (D(-2), D(1)), (D(1), D(3))
    base = sum(value ** 2 for value in base_errors) / 2
    model = sum((value + delta) ** 2 for value, delta in zip(base_errors, corrections)) / 2
    d2 = sum(value ** 2 for value in corrections) / 2
    for gain in (ZERO, D('.25'), D('.5'), ONE):
        actual = sum((value + gain * delta) ** 2 for value, delta in zip(base_errors, corrections)) / 2
        assert actual == base + gain * (model - base - d2) + gain * gain * d2
    return {'passed': True, 'interval_corner_cases': 8, 'synthetic_quadratic_identity_points': 4,
            'model_calls': 0, 'array_files_read': 0}


def run():
    script = Path(__file__).resolve()
    output, study, repository = script.parent, script.parents[1], script.parents[4]
    assert output.drive.upper() in ('E:', 'D:'), 'All diagnostic artifacts must be on E/D'
    input_hashes = {}

    def read(path, lines=False):
        data = path.read_bytes()
        input_hashes[str(path.relative_to(repository)).replace('\\', '/')] = digest(data)
        return [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()] if lines else json.loads(data)

    protocol = read(study / 'protocol.json')
    screen_protocol = protocol['v4_4_additional_development_screen']
    assert tuple(screen_protocol['signed_conditions']) == SIGNED
    assert dec(screen_protocol['separate_signed_weak_pooled_L2_strictly_less_than']) == 1
    factor = dec(screen_protocol['template_relative_MSE_factor'])
    absolute = dec(screen_protocol['template_absolute_MSE_tolerance_e2'])
    entries, pending = [], []
    folders = [(f'zero_audit/{name}', study / 'zero_audit' / name)
               for name in ('residual_scaled', 'negative_balanced')]
    folders.append(('fixed_continuum', study / 'fixed_continuum'))
    for name, folder in folders:
        for source in sorted(folder.glob('validation_update_*.jsonl')):
            update = int(source.stem.rsplit('_', 1)[1])
            marker = folder / f'science_summary_{update:04d}.json'
            summary_path = folder / f'validation_summary_{update:04d}.json'
            if not marker.exists() or not summary_path.exists():
                pending.append({'arm': name, 'update': update, 'reason': 'completed science/validation summary absent'})
                continue
            rows, screen, summary = read(source, True), read(marker), read(summary_path)
            assert len(rows) == len({row['case_id'] for row in rows}) == screen['validation_records'] == 776
            assert summary['validation_cases'] == 776
            assert canonical([row['case_id'] for row in rows]) == protocol['validation_ids_sha256']
            assert screen['validation_sha256'] == digest(source.read_bytes())
            assert screen['protocol_sha256'] == input_hashes['reports/training_v4_4/run_001/protocol.json']
            assert all(row['status'] == 'SUCCESS' and row['temperature_correction_RMS_K'] == 0 for row in rows)
            assert all(row['raw_hash'] == row['physical']['raw_hash']
                       and row['label_sha256'] == row['physical']['label_sha256'] for row in rows)
            groups = {'main_weak': [row for row in rows if row['input_kind'] == 'noise'
                                   and row['group'] == 'weak' and not row['signed_aux']]}
            groups.update({condition: [row for row in rows if row['input_kind'] == 'noise'
                                       and row['condition'] == condition] for condition in SIGNED})
            assert len(groups['main_weak']) == 128
            assert all(len(groups[condition]) == 6 and all(row['signed_aux'] for row in groups[condition])
                       for condition in SIGNED)
            frames = {key: [frame_stats(row) for row in group] for key, group in groups.items()}
            constraints, nominal, outer = [], [], []
            for condition in ('main_weak', 'negative_weak', 'positive_weak'):
                aggregate_stats = aggregate(frames[condition], True)
                item, exact_interval, outer_interval = make_constraint(
                    condition + '_pooled_L2_lt_1', frames[condition], pooled=True,
                    limit=aggregate_stats['truth_mean_square'], strict=True,
                    limit_error=aggregate_stats['et'])
                constraints.append(item)
                nominal.append((item['name'], exact_interval))
                outer.append((item['name'], outer_interval))
            science_groups = {value['condition']: value for value in screen['signed_noisy_conditions']}
            for condition in SIGNED:
                stats = aggregate(frames[condition], False)
                frozen_limit = dec(science_groups[condition]['MSE_limit_e2'])
                assert abs(frozen_limit - (factor * stats['base'] + absolute)) < D('1e-9')
                assert abs(stats['model'] - dec(science_groups[condition]['template_MSE_e2'])) < D('1e-9')
                item, exact_interval, outer_interval = make_constraint(
                    condition + '_science_MSE', frames[condition], pooled=False,
                    limit=frozen_limit, strict=False,
                    limit_error=factor * stats['eb'] + ABS_STAT_BOUND)
                constraints.append(item)
                nominal.append((item['name'], exact_interval))
                outer.append((item['name'], outer_interval))
            selected_rows = [frame['row'] for values in frames.values() for frame in values]
            identities = [[row['case_id'], row['raw_hash'], row['label_sha256'], row['physical']['cache_key']]
                          for row in selected_rows]
            exact_certificate = intersection_certificate(nominal)
            outer_certificate = intersection_certificate(outer)
            entries.append({
                'arm': name, 'optimizer_update_in_this_arm': update,
                'source_validation': str(source.relative_to(repository)).replace('\\', '/'),
                'source_validation_sha256': digest(source.read_bytes()),
                'development_frames': len(rows), 'frames_in_necessary_conditions': len(selected_rows),
                'input_identity_sha256': canonical(identities),
                'all_selected_sidecar_norm_triangle_checks_pass': True,
                'constraints': constraints,
                'nominal_joint_feasibility': exact_certificate,
                'conservative_joint_feasibility': outer_certificate,
                'global_gain_rescue_excluded_under_stated_numerical_bound': outer_certificate['interval']['empty'],
                'lambda_selected': None, 'candidate_created': False,
            })
    assert entries, 'No completed checkpoints found'
    assert len({entry['input_identity_sha256'] for entry in entries}) == 1
    for relative, expected in input_hashes.items():
        assert digest((repository / relative).read_bytes()) == expected, 'A source sidecar changed during reading'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    prefix = 'run_' + stamp
    result = {
        'schema': 'v4.4-fixed-output-global-gain-necessary-conditions-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(script.read_bytes()),
        'scope': 'Development mechanism diagnostic only; seven necessary conditions, not a complete acceptance audit',
        'formula': 'MSE(lambda)=B+lambda*(M-B-D)+lambda^2*D; D=template_correction_RMS_e^2; lambda in [0,1]',
        'interpretation': 'The quadratic identity is exact for fixed physical/model signed outputs and matching targets; serialized statistics are rounded.',
        'numerical_bound': {
            'decimal_precision': 80, 'float32_unit_roundoff': str(U32),
            'per_statistic_error': 'abs(observed)*gamma/(1-gamma)+1e-9 e2, gamma=(pixels+16)*u32/(1-(pixels+16)*u32)',
            'why_conservative': 'Pessimistic sequential float32 reduction; also applied to the more accurate float64 model/energy statistics.',
            'uniform_quadratic_minus_limit_error': 'epsilon_B+epsilon_M+epsilon_D/4+epsilon_limit over lambda in [0,1]',
            'science_limit_error': 'factor*epsilon_B+1e-9 e2; additionally allows numerical uncertainty in the frozen physical reference',
            'outer_interval': 'Solve q<=epsilon with strict boundaries closed, then expand roots by 1e-20 in lambda and intersect [0,1]',
            'limits': 'An arithmetic bound for matching sufficient statistics; it does not cover corrupted inputs, a changed model, or re-inversion differences.',
        },
        'arithmetic_self_tests': self_test(), 'input_files_sha256': input_hashes,
        'completed_outputs_examined': len(entries), 'pending_outputs_skipped': pending,
        'model_inference_calls': 0, 'physical_inverse_calls': 0, 'GPU_calls': 0,
        'training_updates': 0, 'raw_or_label_arrays_read': 0, 'final_test_used': False,
        'protocol_changed': False, 'lambda_selected': None, 'candidate_created': False,
        'all_examined_outputs_have_empty_conservative_intersections': all(
            entry['global_gain_rescue_excluded_under_stated_numerical_bound'] for entry in entries),
        'results': entries,
    }
    table = ['|固定输出|更新|名义交集|保守外扩后|冲突证据|', '|---|---:|---|---|---|']
    for entry in entries:
        cert = entry['conservative_joint_feasibility']
        if cert['interval']['empty'] and cert.get('reason') == 'incompatible_lower_and_upper_bounds':
            detail = (f"{', '.join(cert['lower_bound_constraints'])} 要求 λ≥{cert['required_lower_lambda']:.6f}；"
                      f"{', '.join(cert['upper_bound_constraints'])} 要求 λ≤{cert['permitted_upper_lambda']:.6f}")
        elif cert['interval']['empty']:
            detail = '单项已无解：' + ', '.join(cert['individually_impossible_constraints'])
        else:
            detail = '必要条件区间非空；未选择系数，未建立候选'
        table.append(f"|{entry['arm']}|{entry['optimizer_update_in_this_arm']}|"
                     f"{'空' if entry['nominal_joint_feasibility']['interval']['empty'] else '非空'}|"
                     f"{'空' if cert['interval']['empty'] else '非空'}|{detail}|")
    conclusion = ('所有已检查固定输出的保守外扩交集仍为空；在所列数值误差界内，'
                  '这批输出不能通过一个统一的 λ∈[0,1] 同时满足必要条件。'
                  if result['all_examined_outputs_have_empty_conservative_intersections'] else
                  '至少一个固定输出的保守区间非空；这仅表示所列必要条件尚未排除统一缩放，不能建立候选。')
    report = '\n\n'.join([
        '# 固定连续谱输出：全局 signed 缩放机制诊断',
        conclusion,
        '仅读取已完成的开发逐帧 JSON 与冻结规则。检查主 weak、正 signed weak、负 signed weak 的 pooled L2＜1，'
        '以及四个 signed-noise 组的冻结 MSE 上限。其他验收条件未替代检查；未选择 λ、未改模型或协议。',
        '\n'.join(table),
        '逐帧采用 B=物理 MSE、M=模型 MSE、D=校正 RMS²，构造 B+λ(M−B−D)+λ²D。'
        '系数按所存数值用 80 位 Decimal 计算；解析根检查多项式残差，并对域内简单根进行 160 次独立二分核查。',
        '数值外扩使用长度 N=pixels 的悲观串行 float32 归约界 γ=(N+16)u/[1−(N+16)u]，u=2⁻²⁴；'
        '每项允许 |统计值|γ/(1−γ)+10⁻⁹ e² 的误差，连 float64 模型与能量统计也采用同一宽松界。'
        '在 λ∈[0,1] 上，用 εB+εM+εD/4+εlimit 统一放宽，再把严格边界闭合、根额外外扩10⁻²⁰。'
        '这只处理配对充分统计量的算术误差，不能覆盖输入损坏或重新反演的差异。',
        f"本次纳入 {len(entries)} 个已完成输出；未完成条目 {len(pending)} 个。全部结果引用同一组输入身份。"
        '模型推理、物理求解、GPU、训练更新、原始 raw/标签数组读取及最终测试均为0。',
        f'完整系数、各约束区间、根核查及输入哈希见 `{prefix}.json`。脚本再次运行将产生新文件，保留本次冻结结果。',
    ]) + '\n'
    result_path, report_path, source_path = (output / (prefix + suffix) for suffix in ('.json', '.md', '_source.py'))
    store_new(result_path, result)
    with report_path.open('x', encoding='utf-8') as handle:
        handle.write(report)
    with source_path.open('xb') as handle:
        handle.write(script.read_bytes())
    receipt = {'created_at': result['created_at'], 'completed_outputs_examined': len(entries),
               'all_conservative_intersections_empty': result['all_examined_outputs_have_empty_conservative_intersections'],
               'files_sha256': {path.name: digest(path.read_bytes()) for path in (result_path, report_path, source_path)}}
    store_new(output / (prefix + '_receipt.json'), receipt)
    print(json.dumps({'result': str(result_path), 'report': str(report_path), **receipt}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Run only stdlib arithmetic checks, without reading sidecars')
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = 80
        if args.self_test:
            print(json.dumps(self_test()))
        else:
            run()
