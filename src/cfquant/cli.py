from pathlib import Path
import argparse
import json
from .config import Config
from .data import download_project
from .experiment import execute, study, json_safe


def main():
    p = argparse.ArgumentParser(description="CF2026 reproducible daily research")
    p.add_argument("--root", default=".", help="Project root containing configs and data")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="Download/cache real Tushare research data")
    d.add_argument("--token-file")
    d.add_argument("--assets", type=int, default=60)
    for name in ["run", "study"]:
        s = sub.add_parser(name)
        s.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.command == "download":
        download_project(root, args.token_file, args.assets)
    elif args.command == "run":
        _, metrics, folder = execute(root, Config.load(root/args.config))
        print(json.dumps(json_safe(metrics), indent=2))
        print(folder)
    else:
        print(study(root, Config.load(root/args.config)))


if __name__ == "__main__":
    main()
