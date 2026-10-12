"""Run native data, bounded selection, actual diagnostics, training and reporting."""
import runtime
import argparse
from training_v4.common import setup, OUT, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['data', 'select', 'overfit', 'train', 'compare', 'report'])
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    setup()
    if args.phase == 'data':
        from training_v4.data import generate
        generate()
    elif args.phase == 'select':
        from training_v4.select import run_selection
        run_selection()
    elif args.phase == 'overfit':
        from training_v4.train import overfit_eight
        overfit_eight(read(OUT/'backbone_selection.json'))
    elif args.phase == 'train':
        from training_v4.train import train_seed
        assert read(OUT/'overfit_eight_receipt.json')['passed']
        train_seed(args.seed)
    elif args.phase == 'compare':
        from training_v4.compare import compare
        compare(args.seed)
    else:
        from training_v4.report import report
        report()


if __name__ == '__main__':
    main()
