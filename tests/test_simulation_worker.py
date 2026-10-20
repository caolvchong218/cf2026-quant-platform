"""Actual Windows mandatory-lock regression: no provider or order calls."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows mandatory byte-range locks")
def test_second_worker_refuses_existing_lock_without_reading_locked_byte(tmp_path, monkeypatch, capsys):
    import msvcrt
    script = Path(__file__).resolve().parents[1] / "scripts/run_simulation_service.py"
    spec = importlib.util.spec_from_file_location("simulation_worker_test", script)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    folder = tmp_path / "runs/simulation_auto"
    folder.mkdir(parents=True)
    lock_path = folder / "worker.lock"
    lock_path.write_bytes(b"0")
    monkeypatch.setattr(sys, "argv", [str(script), "--root", str(tmp_path)])
    monkeypatch.setattr(worker, "run_cycle", lambda *a, **kw: pytest.fail("Duplicate worker entered cycle"))
    with lock_path.open("r+b") as held:
        msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            assert worker.main() == 2
            assert "existing simulation worker" in capsys.readouterr().out
        finally:
            held.seek(0)
            msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
