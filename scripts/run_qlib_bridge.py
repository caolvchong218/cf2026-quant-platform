"""Run the real, optional Qlib expression bridge in .venv-qlib."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=int, choices=[10, 20, 40], default=20)
    parser.add_argument('--start', default='2020-01-02')
    parser.add_argument('--end', default='2026-09-18')
    args = parser.parse_args()
    from cfquant.qlib_bridge import run_bridge
    run_bridge(ROOT, args.assets, args.start, args.end)


if __name__ == '__main__':
    main()
