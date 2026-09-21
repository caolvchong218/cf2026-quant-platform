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
    d.add_argument("--start", default="20221001")
    d.add_argument("--end", default="20251231")
    d.add_argument("--selection-date", default="20221230")
    d.add_argument("--study-start", default="20230103")
    d.add_argument("--delay", type=float, default=1.3, help="Seconds between API calls; match account quota")
    d.add_argument("--max-gib", type=float, default=3.0)
    d.add_argument("--workers", type=int, default=1, help="1-8 download threads sharing one rate limiter")
    sub.add_parser("research", help="Run the preregistered multi-factor/model comparison")
    sub.add_parser("features", help="Build dated and neutralized research features")
    merge=sub.add_parser("merge", help="Merge two snapshots into a new destination")
    merge.add_argument("--existing",required=True)
    merge.add_argument("--incoming",required=True)
    merge.add_argument("--destination",required=True)
    for name in ["run", "study"]:
        s = sub.add_parser(name)
        s.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.command == "download":
        download_project(root, args.token_file, args.assets, start=args.start, end=args.end,
                         selection_date=args.selection_date, study_start=args.study_start,
                         delay=args.delay, max_gib=args.max_gib, workers=args.workers)
    elif args.command == "features":
        from .features import build_features
        build_features(root)
    elif args.command == "research":
        from .research import run_research
        run_research(root)
    elif args.command == "merge":
        from .incremental import write_merged_snapshot
        print(write_merged_snapshot(root/args.existing,root/args.incoming,root/args.destination))
    elif args.command == "run":
        _, metrics, folder = execute(root, Config.load(root/args.config))
        print(json.dumps(json_safe(metrics), indent=2))
        print(folder)
    else:
        print(study(root, Config.load(root/args.config)))


if __name__ == "__main__":
    main()
