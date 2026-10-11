"""Versioned repair phases. No audit_v2 inference or training is invoked."""
import runtime
import argparse
import os
for name in ('cuda', 'inductor', 'triton'):
    (runtime.RUNTIME/name).mkdir(parents=True, exist_ok=True)
os.environ['CUDA_CACHE_PATH'] = str(runtime.RUNTIME/'cuda')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(runtime.RUNTIME/'inductor')
os.environ['TRITON_CACHE_DIR'] = str(runtime.RUNTIME/'triton')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'gates', 'smoke', 'summarize'], required=True)
    parser.add_argument('--output', type=runtime.Path, default=runtime.ROOT/'reports/repair_v3/run_001')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.data_root = runtime.ROOT/'outputs/repair_v3'/args.output.name
    if args.output.drive.upper() not in ('E:', 'D:'):
        raise ValueError('All repair outputs must be on E/D')
    args.output.mkdir(parents=True, exist_ok=True)
    from repair_v3.run import prepare, gates, smoke
    if args.phase == 'summarize':
        from repair_v3.report import summarize
        summarize(args)
    else:
        {'prepare': prepare, 'gates': gates, 'smoke': smoke}[args.phase](args)


if __name__ == '__main__':
    main()
