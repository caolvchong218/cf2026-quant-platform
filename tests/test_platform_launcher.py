"""Mocked supervisor handoff: never query or start a real scheduled task."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "qingxu_supervised_launcher", Path(__file__).resolve().parents[1] / "scripts/launch_platform.py",
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _enabled(root):
    folder = root / "runs/simulation_auto"
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text('{"enabled": true}', encoding="utf-8")
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_simulation_service.py").write_text("# fixture", encoding="utf-8")


def _windows(monkeypatch):
    # Replace the launcher's reference, not os.name globally (Path uses it).
    monkeypatch.setattr(launcher, "os", SimpleNamespace(
        name="nt", environ=launcher.os.environ,
    ))


def test_opt_in_is_checked_before_querying_any_registered_task(tmp_path, monkeypatch):
    task = Mock()
    monkeypatch.setattr(launcher, "_start_registered_simulation_task", task)
    monkeypatch.setattr(launcher.subprocess, "Popen", Mock())
    assert launcher.start_simulation_worker(tmp_path) is None
    _enabled(tmp_path)
    (tmp_path / "runs/simulation_auto/settings.json").write_text(
        '{"enabled": false}', encoding="utf-8",
    )
    assert launcher.start_simulation_worker(tmp_path) is None
    task.assert_not_called()
    launcher.subprocess.Popen.assert_not_called()


def test_matching_supervised_task_is_preferred_to_standalone_worker(tmp_path, monkeypatch):
    _enabled(tmp_path)
    _windows(monkeypatch)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="STARTED\n"))
    launch = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    assert launcher.start_simulation_worker(tmp_path) is None
    launch.assert_not_called()
    command = run.call_args.args[0]
    assert command[:3] == ["powershell.exe", "-NoProfile", "-NonInteractive"]
    assert "Get-ScheduledTask" in command[-1] and "Start-ScheduledTask -InputObject $task" in command[-1]
    assert run.call_args.kwargs["timeout"] == 8
    assert run.call_args.kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_absent_or_mismatched_task_preserves_only_current_checkout_fallback(tmp_path, monkeypatch):
    _enabled(tmp_path)
    _windows(monkeypatch)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="FALLBACK\n"))
    launch = Mock(return_value=SimpleNamespace(pid=789))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    assert launcher.start_simulation_worker(tmp_path) == 789
    args, kwargs = launch.call_args
    assert args[0] == [launcher.sys.executable,
                       str(tmp_path / "scripts/run_simulation_service.py"),
                       "--root", str(tmp_path), "--watch"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["PYTHONPATH"] == str(tmp_path / "src")
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


@pytest.mark.parametrize("stdout,returncode", [
    ("MATCHED_FAILED\n", 0), ("UNAVAILABLE\n", 0),
    ("STARTED\nPRIVATE_ERROR", 0), ("STARTED\n", 1),
])
def test_uncertain_or_failed_task_start_cannot_spawn_unsupervised_fallback(
        tmp_path, monkeypatch, capsys, stdout, returncode):
    _enabled(tmp_path)
    _windows(monkeypatch)
    monkeypatch.setattr(launcher.subprocess, "run",
                        Mock(return_value=SimpleNamespace(returncode=returncode, stdout=stdout)))
    launch = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    assert launcher.start_simulation_worker(tmp_path) is None
    launch.assert_not_called()
    output = capsys.readouterr().out
    assert "no standalone worker was launched" in output
    assert "PRIVATE_ERROR" not in output


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired("PRIVATE_COMMAND", 8),
    OSError("PRIVATE_PATH"),
])
def test_task_query_timeout_or_os_failure_is_sanitized_and_does_not_race_lock(
        tmp_path, monkeypatch, capsys, failure):
    _enabled(tmp_path)
    _windows(monkeypatch)
    monkeypatch.setattr(launcher.subprocess, "run", Mock(side_effect=failure))
    launch = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    assert launcher.start_simulation_worker(tmp_path) is None
    launch.assert_not_called()
    output = capsys.readouterr().out
    assert "no standalone worker was launched" in output and "PRIVATE_" not in output


def test_task_identity_checks_match_installer_paths_quoting_and_flags(tmp_path, monkeypatch):
    _windows(monkeypatch)
    root = tmp_path / "青序 project's copy"
    _enabled(root)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="FALLBACK"))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher._start_registered_simulation_task(root) == "fallback"
    source = run.call_args.args[0][-1]
    assert str(root.resolve()).replace("'", "''") in source
    assert str((root / "scripts/run_simulation_service.py").resolve()).replace("'", "''") in source
    assert "'^\"([^\"\\r\\n]+)\" --root \"([^\"\\r\\n]+)\" --watch$'" in source
    assert "Same-Path $action.WorkingDirectory $expectedRoot" in source
    assert "Same-Path $parts.Groups[1].Value $expectedWorker" in source
    assert "Same-Path $parts.Groups[2].Value $expectedRoot" in source
    assert "$actions.Count -ne 1" in source
    assert "OrdinalIgnoreCase" in source
    assert str(Path(launcher.sys.executable).resolve().with_name("pythonw.exe")) in source


@pytest.mark.skipif(launcher.os.name != "nt", reason="Execute only mocked Windows cmdlets")
@pytest.mark.parametrize("case,expected,count", [
    ("matching", "STARTED", 1),
    ("absent", "FALLBACK", 0),
    ("different_checkout", "FALLBACK", 0),
    ("different_python", "FALLBACK", 0),
    ("extra_arguments", "FALLBACK", 0),
    ("multiple_actions", "FALLBACK", 0),
    ("different_working_directory", "FALLBACK", 0),
    ("unquoted_arguments", "FALLBACK", 0),
])
def test_powershell_identity_gate_runs_only_the_matching_mock_task(
        tmp_path, monkeypatch, case, expected, count):
    """Execute the generated guard with local mock functions shadowing both cmdlets."""
    root = tmp_path / "青序 project's copy"
    _enabled(root)
    real_run = subprocess.run
    mocked_run = Mock(return_value=SimpleNamespace(returncode=0, stdout="FALLBACK"))
    monkeypatch.setattr(launcher.subprocess, "run", mocked_run)
    launcher._start_registered_simulation_task(root)
    command = mocked_run.call_args.args[0][-1]
    worker = root / "scripts/run_simulation_service.py"
    action = {
        "Execute": str(Path(launcher.sys.executable).resolve().with_name("pythonw.exe")),
        "Arguments": f'"{worker}" --root "{root}" --watch',
        "WorkingDirectory": str(root),
    }
    task = {"TaskName": "QingxuContinuousSimulation", "Actions": [action]}
    if case == "absent":
        task = None
    elif case == "different_checkout":
        action["Arguments"] = f'"{worker}" --root "{tmp_path / "other"}" --watch'
    elif case == "different_python":
        action["Execute"] = str(tmp_path / "other/pythonw.exe")
    elif case == "extra_arguments":
        action["Arguments"] += " --refresh"
    elif case == "multiple_actions":
        task["Actions"].append(dict(action))
    elif case == "different_working_directory":
        action["WorkingDirectory"] = str(tmp_path / "other")
    elif case == "unquoted_arguments":
        action["Arguments"] = f"{worker} --root {root} --watch"
    payload = base64.b64encode(json.dumps(task, ensure_ascii=False).encode()).decode()
    mock_functions = "\n".join([
        f"$script:fixtureTask=([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}')) | ConvertFrom-Json)",
        "$script:startCalls=0",
        "function Get-ScheduledTask {",
        "  [CmdletBinding()] param([string]$TaskPath)",
        "  return $script:fixtureTask",
        "}",
        "function Start-ScheduledTask {",
        "  [CmdletBinding()] param($InputObject)",
        "  $script:startCalls++",
        "}",
        "& {", command, "}",
        "Write-Output ('COUNT='+$script:startCalls)",
    ])
    result = real_run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", mock_functions],
        capture_output=True, text=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [expected, f"COUNT={count}"]


def test_non_windows_uses_legacy_worker_without_powershell_query(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="posix", environ=launcher.os.environ))
    run = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher._start_registered_simulation_task(tmp_path) == "fallback"
    run.assert_not_called()
