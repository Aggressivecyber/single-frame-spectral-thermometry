"""The complete gated v5 phase CLI.

Run ordinary Python -B with assertions enabled. Formal arrays and GPU execution
are dispatched only here by the root after source registration. CPU fixture QA
can be requested separately with --implementation-check.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

from . import common

# This must precede all imports of legacy runtime/physics/torch.
common.configure_runtime()

from .reports import PHASES, contract_checks, write_scientific_decision, write_train_report


class GateRejected(RuntimeError):
    """A missing/failed real receipt, not an unimplemented phase."""


def _require(condition, reason):
    if not condition:
        raise GateRejected(reason)


def _receipt(name):
    path = common.REPORTS / name
    _require(path.is_file(), 'Required completed evidence is missing: ' + str(path))
    return common.read_json(path)


def _stage1(allow_ml_stop=False):
    value = _receipt('DECISION_1.json')
    for name, digest in value.get('evidence_sha256', {}).items():
        path = common.REPORTS / name
        _require(path.is_file() and common.sha(path) == digest, 'Stage 1 evidence hash changed: ' + name)
    _require(value.get('stage1_evidence_complete') is True, 'Full stage 1 evidence has not completed')
    _require(value.get('physical_stage2_eligible') is True, 'O0 or stage 1 numerical/optical gates block physical stage 2')
    if not allow_ml_stop:
        _require(value.get('ml_stop_required') is False,
                 'Stage 1 strict stop blocks ML: ' + '; '.join(value.get('reasons', [])))
    return value


def _physical_checks():
    value = _receipt('LINEAR_SOLVER_CHECKS.json')
    _require(value.get('all_passed') is True, 'Physical solver CPU controls failed or are unavailable')
    _require(value.get('stage1_decision_sha256') == common.sha(common.REPORTS / 'DECISION_1.json'),
             'Physical solver controls are not bound to the same stage 1 decision')
    return value


def _development():
    value = _receipt('development/development_gate.json')
    _require(value.get('physics_eligible') is True, 'Real physical development acceptance has not passed')
    stage1 = value.get('stage1', {})
    _require(stage1.get('sha256') == common.sha(common.REPORTS / 'DECISION_1.json'),
             'Development gate is not bound to the same completed stage 1 evidence')
    return value


def dispatch(phase, device='cuda'):
    """Connected APIs for all eight phases; gates are checked before generation."""
    _require(phase in PHASES, 'Unknown phase')
    if phase == 'register':
        from .protocol import register
        return register()
    _require((common.CODE / 'protocol.json').is_file(), 'Source protocol is not registered')
    protocol = common.require_registered()
    budget = common.Budget(protocol['phase_budgets_s'][phase])
    if phase == 'oracle':
        from .stage1 import run_oracle
        physics = common.make_physics(device=device, config=protocol['physics_config'])
        return run_oracle(protocol, physics, budget)
    if phase == 'identifiability':
        from .stage1 import run_identifiability
        prior = _receipt('ORACLE_SUMMARY.json')
        _require(prior.get('complete') is True, 'Information diagnostics require all 180 oracle input records')
        physics = common.make_physics(device=device, config=protocol['physics_config'])
        return run_identifiability(protocol, physics, budget)
    if phase == 'linear-solver':
        _stage1(allow_ml_stop=True)
        from .stage1 import run_linear_solver
        return run_linear_solver(protocol, budget)
    if phase == 'development-compare':
        _stage1(allow_ml_stop=True)
        _physical_checks()
        from .development import run_development
        return run_development(device=device)
    if phase == 'train-denoiser':
        _stage1()
        _physical_checks()
        development = _development()
        _require(development.get('no_hardstop') is True,
                 'Real development or stage 1 strict collapse blocks the bounded denoiser')
        from .train import train_denoiser
        return train_denoiser(device=device)
    if phase == 'candidate-review':
        _stage1(allow_ml_stop=True)
        _physical_checks()
        _development()
        from .candidate_review import review_candidate
        return review_candidate(device=device)
    # Final dispatch is reachable only after the canonical candidate receipt.
    _stage1(allow_ml_stop=True)
    candidate = _receipt('candidate_review/candidate_receipt.json')
    _require(candidate.get('eligible') is True,
             'A real eligible candidate receipt is required before sealed data is opened')
    _require((common.REPORTS / 'final_protocol.json').is_file(),
             'Candidate, thresholds and statistics are not frozen')
    _require(not (common.REPORTS / 'FINAL_ONCE_STARTED.json').exists(),
             'Sealed evaluation has already started once; no retry or retuning is permitted')
    from .final import run_final_once
    return run_final_once(device=device)


def implementation_checks(*, include_core_numerical=False, output=None):
    """Bounded CPU checks of this orchestration without registration/generation."""
    started = time.perf_counter()
    names = ('run.py', 'reports.py', 'protocol.py', 'stage1.py', 'common.py', 'planning.py', '__init__.py')
    covered_names = names + (('continuum_solver.py', 'signed_linear_solver.py', 'reconstruct.py', 'tests_adjoint.py')
                             if include_core_numerical else ())
    starting_sources = {common.relative(common.CODE / name): common.sha(common.CODE / name) for name in covered_names}
    checks = contract_checks()
    for name in names:
        tree = ast.parse((common.CODE / name).read_text(encoding='utf-8-sig'), filename=name)
        checks.append({'name': 'AST_parse_' + name, 'passed': bool(tree)})
    # Missing scientific evidence must be a gate rejection before any numerical
    # import or data-generation call. Temporary monkeypatches are restored here.
    real_receipt = globals()['_receipt']
    try:
        globals()['_receipt'] = lambda name: {}
        for action in (_stage1, _physical_checks, _development):
            try:
                action()
            except GateRejected:
                checks.append({'name': 'missing_evidence_rejected_' + action.__name__, 'passed': True})
            else:
                checks.append({'name': 'missing_evidence_rejected_' + action.__name__, 'passed': False})
        globals()['_receipt'] = lambda name: {'stage1_evidence_complete': True,
            'physical_stage2_eligible': True, 'ml_stop_required': True, 'reasons': ['synthetic collapse']}
        checks.append({'name': 'nontraining_diagnostic_allowed_after_O0',
                       'passed': _stage1(allow_ml_stop=True)['physical_stage2_eligible'] is True})
        try:
            _stage1()
        except GateRejected:
            checks.append({'name': 'strict_collapse_blocks_training', 'passed': True})
        else:
            checks.append({'name': 'strict_collapse_blocks_training', 'passed': False})
    finally:
        globals()['_receipt'] = real_receipt
    core = None
    if include_core_numerical:
        from .tests_adjoint import run_tests
        core = run_tests()
        checks.append({'name': 'bounded16_CPU_physics_linear_controls', 'passed': core['all_passed']})
    assert all(check['passed'] for check in checks), checks
    assert all(common.sha(common.ROOT / name) == digest for name, digest in starting_sources.items()), \
        'Checked source changed while implementation controls ran'
    sources = dict(starting_sources)
    if core is not None:
        sources.update(core['source_sha256'])
    value = {'schema': 'v5-orchestration-implementation-check-001', 'time': common.now(),
             'device': 'cpu', 'all_passed': all(check['passed'] for check in checks),
             'checks': checks, 'source_sha256': sources, 'core_numerical': core,
             'duration_s': time.perf_counter() - started, 'formal_generation_executed': False,
             'training_executed': False, 'final_or_confirmation_payloads_opened': False,
             'scope': 'small CPU numerical/gate implementation QA; no scientific acceptance',
             'assertions_enabled': not sys.flags.optimize, 'no_bytecode': sys.dont_write_bytecode}
    if output is None:
        stamp = common.now().replace(':', '').replace('.', '').replace('+', '_')
        output = common.REPORTS.parent / 'implementation_checks/orchestration' / ('cpu_' + stamp + '.json')
    common.write_json(output, value)
    return value, Path(output)


def _scientific_pass(phase, value):
    if phase == 'oracle':
        return value.get('o0_noiseless_gate', {}).get('passed') is True
    if phase == 'identifiability':
        return value.get('complete') is True and value.get('numerical_checks_passed') is True and value.get('optical_all_passed') is True
    if phase == 'development-compare':
        return value.get('physics_eligible') is True
    if phase == 'train-denoiser':
        # A failed/omitted control is a completed negative branch, not final acceptance.
        return value.get('accepted') is True
    if phase == 'candidate-review':
        return value.get('eligible') is True
    if phase == 'final-once':
        return value.get('scientific_acceptance_passed') is True
    return None


def _exclusive_phase_lock(phase):
    path = common.ed_path(common.REPORTS / 'PHASE_ACTIVE.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise GateRejected('Another phase is active or an interrupted phase requires root audit: ' + str(path)) from error
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        json.dump({'phase': phase, 'pid': os.getpid(), 'started_at': common.now(),
                   'one_root_GPU_job_at_a_time': True}, handle, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--phase', choices=PHASES)
    group.add_argument('--implementation-check', action='store_true')
    parser.add_argument('--include-core-numerical', action='store_true',
                        help='add bounded 16x16 CPU physics controls to implementation QA')
    parser.add_argument('--device', default='cuda', help='root-selected formal device, default cuda')
    parser.add_argument('--check-output', type=Path, help='new E/D reports/research_v5 CPU receipt')
    args = parser.parse_args(argv)
    if args.implementation_check:
        value, output = implementation_checks(include_core_numerical=args.include_core_numerical,
                                               output=args.check_output)
        print(json.dumps({'all_passed': value['all_passed'], 'receipt': str(output),
                          'duration_s': value['duration_s']}, allow_nan=False), flush=True)
        return 0
    if args.include_core_numerical or args.check_output:
        parser.error('Implementation-check options require --implementation-check')
    started = time.perf_counter()
    stamp = common.now().replace(':', '').replace('.', '').replace('+', '_')
    event = {'schema': 'v5-phase-execution-001', 'phase': args.phase, 'time': common.now(),
             'device': args.device, 'assertions_enabled': not sys.flags.optimize,
             'no_bytecode': sys.dont_write_bytecode, 'source_sha256': common.source_hashes(),
             'status': 'STARTED', 'numerical_execution_passed': False, 'scientific_gate_passed': None}
    lock = None
    code = 1
    try:
        lock = _exclusive_phase_lock(args.phase)
        common.append_jsonl(common.REPORTS / 'PHASE_EVENTS.jsonl', event)
        value = dispatch(args.phase, device=args.device)
        scientific = _scientific_pass(args.phase, value)
        numerical = (value.get('numerical_execution_passed') is True if args.phase == 'final-once' else
                     value.get('complete') is True and value.get('case_counts', {}).get('oracle_returned_rows') == 720
                     if args.phase == 'oracle' else
                     value.get('complete') is True and value.get('numerical_checks_passed') is True
                     if args.phase == 'identifiability' else
                     value.get('all_passed') is True if args.phase == 'linear-solver' else
                     not bool(value.get('failures')))
        event.update(status='COMPLETED' if scientific is not False else 'COMPLETED_NEGATIVE_RESULT',
                     numerical_execution_passed=numerical, scientific_gate_passed=scientific,
                     returned_status=value.get('status'), result_schema=value.get('schema'))
        code = 0 if scientific is not False else 2
    except GateRejected as error:
        event.update(status='NOT_ELIGIBLE', gate_rejection=str(error), numerical_execution_passed=False)
        code = 2
    except Exception as error:
        event.update(status='FAILED', error_type=type(error).__name__, error=str(error), numerical_execution_passed=False)
        code = 1
    finally:
        event.update(finished_at=common.now(), wall_s=time.perf_counter() - started,
                     protocol_sha256=common.sha(common.CODE / 'protocol.json') if (common.CODE / 'protocol.json').exists() else None)
        common.append_jsonl(common.REPORTS / 'PHASE_EVENTS.jsonl', event)
        path = common.REPORTS / 'phase_receipts' / (args.phase + '_' + stamp + '.json')
        common.write_json(path, event)
        try:
            if args.phase == 'train-denoiser' and (common.REPORTS / 'TRAIN_SUMMARY.json').is_file():
                write_train_report(common.read_json(common.REPORTS / 'TRAIN_SUMMARY.json'))
            write_scientific_decision()
        finally:
            if lock is not None:
                assert lock == common.ed_path(common.REPORTS / 'PHASE_ACTIVE.json')
                lock.unlink()
    print(json.dumps({'phase': args.phase, 'status': event['status'], 'scientific_gate_passed': event['scientific_gate_passed'],
                      'receipt': str(path), 'wall_s': event['wall_s'],
                      'reason': event.get('gate_rejection', event.get('error'))}, ensure_ascii=False, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
