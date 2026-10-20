<#
Install a per-user scheduled worker for explicitly configured simulated accounts.
Requires an existing private settings file; never creates an account or enables trading.
#>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "",
    [string]$TaskName = "QingxuContinuousSimulation"
)
$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$workerPath = Join-Path $resolvedRoot "scripts\run_simulation_service.py"
$settingsPath = Join-Path $resolvedRoot "runs\simulation_auto\settings.json"
if (-not (Test-Path -LiteralPath $workerPath) -or -not (Test-Path -LiteralPath $settingsPath)) {
    throw "Install only in an existing configured Qingxu simulation checkout."
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $resolvedRoot ".venv-clean\Scripts\pythonw.exe"
}
$resolvedPython = (Resolve-Path -LiteralPath $Python).Path
if ($resolvedRoot.Contains('"') -or $resolvedPython.Contains('"')) {
    throw "Unsupported quotation mark in installation path."
}
$arguments = '"' + $workerPath + '" --root "' + $resolvedRoot + '" --watch'
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.Actions.Count -ne 1 -or
        $existing.Actions[0].Execute -ne $resolvedPython -or
        $existing.Actions[0].Arguments -ne $arguments) {
        throw "An unrelated scheduled task uses this name; it has not been changed."
    }
}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $resolvedPython -Argument $arguments -WorkingDirectory $resolvedRoot
$logon = New-ScheduledTaskTrigger -AtLogOn -User $identity
$morning = New-ScheduledTaskTrigger -Daily -At "08:55"
$evening = New-ScheduledTaskTrigger -Daily -At "17:10"
# A repeating trigger also recovers failures for which RestartCount is not
# honored by Task Scheduler. Omit RepetitionDuration/EndBoundary deliberately:
# the pattern must not expire; IgnoreNew keeps an active worker undisturbed.
$recovery = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger @($logon, $morning, $evening, $recovery) `
    -Settings $settings -Principal $principal `
    -Description "Qingxu dedicated simulation only: durable data catch-up and guarded automatic paper orders."
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
