"""Run the simulation worker once or keep it alive with a single-process lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time as clock

# Scheduled tasks do not inherit the launcher's PYTHONPATH. Bind code to this
# script's checkout even when the interpreter has another editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfquant.simulation_service import directory, run_cycle, write_json
from cfquant.simulation_refresh import ContinuousRefresh


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
        # Reading the locked byte also raises PermissionError on Windows.
        # Inspect metadata without touching that byte before acquiring it.
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock.close()
            print("An existing simulation worker holds the lock.")
            return 2
    else:
        import fcntl
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("An existing simulation worker holds the lock.")
            return 2
    controller = ContinuousRefresh(root) if args.watch or args.refresh else None
    if args.refresh and not args.watch:
        # A bounded one-shot can refresh the calendar then one data batch.
        for _ in range(2):
            state = controller.tick()
            if not state["running"]:
                break
            controller.wait_for_current()
            controller.tick(start_jobs=False)
    while True:
        now = datetime.now(timezone.utc)
        try:
            state = controller.tick(now=now, start_jobs=args.watch) if controller else None
            status = run_cycle(root, now=now, refresh_status=state)
        except Exception:
            status = {"asof": now.isoformat(), "pid": os.getpid(),
                      "enabled": False, "mode": "error",
                      "detail": "模拟服务遇到本地错误，未重试下单。", "strategies": []}
            write_json(folder / "service_status.json", status)
        if not args.watch:
            print(json.dumps(status, ensure_ascii=False, indent=2))
            if controller:
                controller.close()
            return 0
        clock.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
