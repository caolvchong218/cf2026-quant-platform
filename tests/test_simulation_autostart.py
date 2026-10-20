"""Run the installer with every scheduled-task/CIM command replaced by mocks."""
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell installer")
INSTALLER = Path(__file__).resolve().parents[1] / "scripts/install_simulation_autostart.ps1"


MOCKS = r"""
$script:registerCount=0
$script:startCount=0
$script:registeredTask=$null
$script:clock=[datetime]'2026-10-20T19:00:00'
function Get-Date { return $script:clock }
function Get-ScheduledTask {
    [CmdletBinding()] param([string]$TaskName)
    if($null -ne $script:registeredTask){return $script:registeredTask}
    return $script:existingTask
}
function Register-ScheduledTask {
    [CmdletBinding()] param([string]$TaskName,$InputObject,[switch]$Force)
    $script:registerCount++
    $script:registeredTask=$InputObject
    $script:registeredTask | Add-Member TaskName $TaskName
    $script:registeredTask | Add-Member State 'Ready'
    return $script:registeredTask
}
function Start-ScheduledTask {
    [CmdletBinding()] param([string]$TaskName)
    $script:startCount++
    if($null -eq $script:registeredTask){throw 'Start before registration'}
    $script:registeredTask.State='Running'
}
function New-ScheduledTaskAction {
    [CmdletBinding()] param([string]$Execute,[string]$Argument,[string]$WorkingDirectory)
    return [pscustomobject]@{Execute=$Execute;Arguments=$Argument;WorkingDirectory=$WorkingDirectory}
}
function New-ScheduledTaskTrigger {
    [CmdletBinding()] param([switch]$AtLogOn,[string]$User,[switch]$Daily,
        [switch]$Once,[datetime]$At,[timespan]$RepetitionInterval,[timespan]$RepetitionDuration)
    $kind=if($AtLogOn){'logon'}elseif($Daily){'daily'}elseif($Once){'once'}else{'unknown'}
    $interval=if($PSBoundParameters.ContainsKey('RepetitionInterval')){
        [System.Xml.XmlConvert]::ToString($RepetitionInterval)}else{$null}
    $duration=if($PSBoundParameters.ContainsKey('RepetitionDuration')){
        [System.Xml.XmlConvert]::ToString($RepetitionDuration)}else{$null}
    $start=if($PSBoundParameters.ContainsKey('At')){$At.ToString('o')}else{$null}
    return [pscustomobject]@{
        Kind=$kind;User=$User;StartBoundary=$start;EndBoundary=$null
        Repetition=[pscustomobject]@{Interval=$interval;Duration=$duration;StopAtDurationEnd=$false}
        BoundParameters=@($PSBoundParameters.Keys)
    }
}
function New-ScheduledTaskSettingsSet {
    [CmdletBinding()] param([switch]$StartWhenAvailable,[string]$MultipleInstances,
        [int]$RestartCount,[timespan]$RestartInterval,[timespan]$ExecutionTimeLimit,
        [switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries)
    return [pscustomobject]@{
        StartWhenAvailable=$StartWhenAvailable.IsPresent;MultipleInstances=$MultipleInstances
        RestartCount=$RestartCount;RestartInterval=$RestartInterval.TotalSeconds
        ExecutionTimeLimit=$ExecutionTimeLimit.TotalSeconds
        AllowStartIfOnBatteries=$AllowStartIfOnBatteries.IsPresent
        DontStopIfGoingOnBatteries=$DontStopIfGoingOnBatteries.IsPresent
    }
}
function New-ScheduledTaskPrincipal {
    [CmdletBinding()] param([string]$UserId,[string]$LogonType,[string]$RunLevel)
    return [pscustomobject]@{UserId=$UserId;LogonType=$LogonType;RunLevel=$RunLevel}
}
function New-ScheduledTask {
    [CmdletBinding()] param($Action,$Trigger,$Settings,$Principal,[string]$Description)
    return [pscustomobject]@{
        Actions=@($Action);Triggers=@($Trigger);Settings=$Settings
        Principal=$Principal;Description=$Description
    }
}
"""


def _literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _run_installer(tmp_path, *, existing_case=None, configured=True):
    root = tmp_path / "青序 project copy"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_simulation_service.py").write_text("# fixture", encoding="utf-8")
    python = root / ".venv-clean/Scripts/pythonw.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture; never executed")
    settings = root / "runs/simulation_auto/settings.json"
    if configured:
        settings.parent.mkdir(parents=True)
        settings.write_text('{"enabled": false}', encoding="utf-8")
    expected_arguments = f'"{root / "scripts/run_simulation_service.py"}" --root "{root}" --watch'
    action = {"Execute": str(python), "Arguments": expected_arguments,
              "WorkingDirectory": str(root)}
    existing = None
    if existing_case:
        existing = {"TaskName": "QingxuContinuousSimulation", "Actions": [action]}
        if existing_case == "different_python":
            action["Execute"] = str(tmp_path / "other/pythonw.exe")
        elif existing_case == "different_root":
            action["Arguments"] = f'"{root / "scripts/run_simulation_service.py"}" --root "{tmp_path}" --watch'
        elif existing_case == "extra_arguments":
            action["Arguments"] += " --refresh"
        elif existing_case == "multiple_actions":
            existing["Actions"].append(dict(action))
    payload = base64.b64encode(json.dumps(existing, ensure_ascii=False).encode()).decode()
    installer_source = base64.b64encode(INSTALLER.read_bytes()).decode()
    command = "\n".join([
        "$ErrorActionPreference='Stop'",
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)",
        f"$script:existingTask=([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}')) | ConvertFrom-Json)",
        MOCKS,
        f"$installer=[ScriptBlock]::Create([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{installer_source}')))",
        "$failure=$null",
        "try {",
        f"  & $installer -Root {_literal(root)} -Python {_literal(python)} | Out-Null",
        "} catch {$failure=$_.Exception.Message}",
        "[pscustomobject]@{",
        "  Registered=$script:registerCount;Started=$script:startCount",
        "  Task=$script:registeredTask;Error=$failure",
        "} | ConvertTo-Json -Depth 12 -Compress",
    ])
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    if configured:
        assert json.loads(settings.read_text(encoding="utf-8")) == {"enabled": False}
    return report, root, python, expected_arguments


@pytest.mark.parametrize("existing_case", [None, "matching"])
def test_installer_adds_one_minute_indefinite_recovery_without_changing_action(tmp_path, existing_case):
    report, root, python, arguments = _run_installer(tmp_path, existing_case=existing_case)
    assert report["Error"] is None
    assert report["Registered"] == report["Started"] == 1
    task = report["Task"]
    assert task["Actions"] == [{
        "Execute": str(python), "Arguments": arguments, "WorkingDirectory": str(root),
    }]
    assert len(task["Triggers"]) == 4
    once = next(trigger for trigger in task["Triggers"] if trigger["Kind"] == "once")
    assert datetime.fromisoformat(once["StartBoundary"]) == datetime(2026, 10, 20, 19, 1)
    assert once["Repetition"]["Interval"] == "PT1M"
    assert once["Repetition"]["Duration"] is None
    assert once["EndBoundary"] is None
    assert not once["Repetition"]["StopAtDurationEnd"]
    assert "RepetitionDuration" not in once["BoundParameters"]
    assert sum(trigger["Kind"] == "logon" for trigger in task["Triggers"]) == 1
    daily = sorted(datetime.fromisoformat(trigger["StartBoundary"]).strftime("%H:%M")
                   for trigger in task["Triggers"] if trigger["Kind"] == "daily")
    assert daily == ["08:55", "17:10"]
    assert task["Settings"]["MultipleInstances"] == "IgnoreNew"
    assert task["Settings"]["StartWhenAvailable"]
    assert task["Settings"]["RestartCount"] == 3
    assert task["Settings"]["RestartInterval"] == 60
    assert task["Settings"]["ExecutionTimeLimit"] == 0
    assert task["Principal"]["LogonType"] == "Interactive"
    assert task["Principal"]["RunLevel"] == "Limited"


@pytest.mark.parametrize("case", [
    "different_python", "different_root", "extra_arguments", "multiple_actions",
])
def test_installer_refuses_unrelated_existing_task_before_registering_or_starting(tmp_path, case):
    report, *_ = _run_installer(tmp_path, existing_case=case)
    assert report["Registered"] == report["Started"] == 0
    assert report["Task"] is None
    assert "unrelated scheduled task" in report["Error"]


def test_installer_requires_existing_simulation_configuration(tmp_path):
    report, *_ = _run_installer(tmp_path, configured=False)
    assert report["Registered"] == report["Started"] == 0
    assert "existing configured" in report["Error"]
