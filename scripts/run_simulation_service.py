"""Run the simulation worker once or keep it alive with a single-process lock."""
from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
import json
import os
from pathlib import Path
import time as clock
from zoneinfo import ZoneInfo

from cfquant.simulation_service import directory, run_cycle, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    folder = directory(root)
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "worker.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("An existing simulation worker holds the lock.")
            return 2
    else:
        import fcntl
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("An existing simulation worker holds the lock.")
            return 2
    refreshed = None
    refresh_on_start = args.refresh
    while True:
        now = datetime.now(timezone.utc)
        local = now.astimezone(ZoneInfo("Asia/Shanghai"))
        # One bounded daily refresh after published daily data. A failed fetch
        # is visible; do not hammer the data vendor indefinitely.
        due = (time(17, 5) <= local.time().replace(tzinfo=None) < time(17, 30)
               and refreshed != local.date())
        try:
            status = run_cycle(root, now=now, refresh=refresh_on_start or due)
        except Exception:
            status = {"asof": now.isoformat(), "pid": os.getpid(),
                      "enabled": False, "mode": "error",
                      "detail": "模拟服务遇到本地错误，未重试下单。", "strategies": []}
            write_json(folder / "service_status.json", status)
        if refresh_on_start or due:
            refreshed = local.date()
        refresh_on_start = False
        if not args.watch:
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        clock.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
