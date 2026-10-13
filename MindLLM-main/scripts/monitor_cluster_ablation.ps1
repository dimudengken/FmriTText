param(
    [string]$Server = 'myserver00',
    [string]$RunDir = '/mnt/data2/yjs2026/HLQ/MindLLM/MindLLM-main/outputs/cluster_ablation_s1_seed42_20261013_r2',
    [int]$IntervalSeconds = 120,
    [int]$StallMinutes = 20,
    [switch]$Once,
    [switch]$NoNotify
)

$ErrorActionPreference = 'Stop'
$workspaceDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$localDir = Join-Path $workspaceDir 'outputs\cluster_ablation_monitor'
[IO.Directory]::CreateDirectory($localDir) | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
$snapshotPath = Join-Path $localDir 'snapshot.json'
$monitorPath = Join-Path $localDir 'monitor.json'
$eventsPath = Join-Path $localDir 'events.jsonl'
$sshPath = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
$remoteInspector = '/mnt/data2/yjs2026/HLQ/MindLLM/MindLLM-main/scripts/inspect_cluster_ablation.py'
if ($Server -notmatch '^[A-Za-z0-9_.-]+$' -or
    $RunDir -notmatch '^/mnt/data2/yjs2026/HLQ/MindLLM/MindLLM-main/outputs/[A-Za-z0-9_-]+$') {
    throw 'Unexpected SSH host or run directory'
}

function Save-Json($Path, $Value) {
    $temporary = $Path + '.tmp'
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 16), $utf8)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Record-Event($Level, $Kind, $Message) {
    $record = @{ time_utc = [DateTime]::UtcNow.ToString('o'); level = $Level; kind = $Kind; message = $Message }
    [IO.File]::AppendAllText($eventsPath, (($record | ConvertTo-Json -Compress) + "`n"), $utf8)
    if (-not $NoNotify -and $script:tray) {
        try {
            $icon = [Windows.Forms.ToolTipIcon]::Info
            if ($Level -eq 'warning' -or $Level -eq 'error') { $icon = [Windows.Forms.ToolTipIcon]::Warning }
            $script:tray.ShowBalloonTip(10000, 'MindLLM experiment: ' + $Kind, $Message, $icon)
            [Windows.Forms.Application]::DoEvents()
        } catch {
            [IO.File]::AppendAllText($eventsPath, ('{"kind":"notification_error","message":"Desktop notification failed"}' + "`n"), $utf8)
        }
    }
}

function Read-RemoteSnapshot {
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $sshPath
    $info.Arguments = '-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 ' +
        $Server + ' "python3 ' + $remoteInspector + ' --run-dir ' + $RunDir + ' --write-progress --ensure-guardian"'
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = $utf8
    $info.StandardErrorEncoding = $utf8
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        $process.Start() | Out-Null
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(45000)) {
            $process.Kill()
            throw 'SSH health check timed out after 45 seconds'
        }
        if ($process.ExitCode -ne 0) { throw ('SSH check failed: ' + $stderr.Result.Trim()) }
        return ($stdout.Result | ConvertFrom-Json)
    } finally {
        $process.Dispose()
    }
}

$lockPath = Join-Path $localDir 'monitor.lock'
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$script:tray = $null
$connectionFailures = 0
$deadChecks = 0
$lastPhase = $null
$lastIssue = $null
$lastRecoveryCount = 0
$connectionAlerted = $false
$state = 'running'
try {
    if (-not $NoNotify) {
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $script:tray = New-Object System.Windows.Forms.NotifyIcon
        $script:tray.Icon = [Drawing.SystemIcons]::Information
        $script:tray.Text = 'MindLLM experiment monitor'
        $script:tray.Visible = $true
    }
    do {
        try {
            $snapshot = Read-RemoteSnapshot
            Save-Json $snapshotPath $snapshot
            if ($connectionAlerted) { Record-Event 'info' 'reconnected' 'SSH connection recovered.' }
            $connectionFailures = 0
            $connectionAlerted = $false
            $state = 'running'
            if ($lastPhase -and $lastPhase -ne $snapshot.phase) {
                Record-Event 'info' 'phase changed' ('Now running ' + $snapshot.phase)
            }
            $lastPhase = $snapshot.phase
            if ($snapshot.guardian -and $snapshot.guardian.recoveries_total -gt $lastRecoveryCount) {
                Record-Event 'warning' 'recovery attempt' ('Server guardian is recovering ' + $snapshot.phase + ' with the same configuration.')
                $lastRecoveryCount = $snapshot.guardian.recoveries_total
            }
            $issue = $null
            if ($snapshot.state -eq 'complete') {
                $message = 'All experiment stages finished. Check server summary.json.'
                if ($snapshot.summary -and $snapshot.summary.comparison) {
                    $delta = $snapshot.summary.comparison.delta_CIDEr_x100
                    $message += (' Functional-baseline CIDEr delta: ' + [Math]::Round($delta, 3))
                }
                Record-Event 'info' 'complete' $message
                $state = 'complete'
            } elseif ($snapshot.state -eq 'failed') {
                $issue = 'experiment failed'
            } elseif ($snapshot.nonfinite_loss) {
                $issue = 'nonfinite loss'
            } elseif ($snapshot.fatal_patterns.Count -gt 0) {
                $issue = 'error in current log'
            } elseif (-not $snapshot.supervisor_alive -or -not $snapshot.worker_alive) {
                $deadChecks++
                if ($deadChecks -ge 2) { $issue = 'experiment process disappeared' }
            } else {
                $deadChecks = 0
                if ($snapshot.progress_age_seconds -ge ($StallMinutes * 60)) {
                    $issue = 'progress stalled'
                }
            }
            if ($issue -and $issue -ne $lastIssue) {
                Record-Event 'error' $issue ('Phase: ' + $snapshot.phase + '. See ' + $snapshot.log)
            } elseif (-not $issue -and $lastIssue -and $snapshot.state -eq 'running') {
                Record-Event 'info' 'recovered' 'Experiment health checks recovered.'
            }
            $lastIssue = $issue
            if ($issue) { $state = 'attention' }
            Save-Json $monitorPath @{ pid = $PID; state = $state; checked_utc = [DateTime]::UtcNow.ToString('o');
                interval_seconds = $IntervalSeconds; run_dir = $RunDir; phase = $snapshot.phase;
                connection_failures = 0; issue = $issue; notifications = (-not $NoNotify) }
        } catch {
            $connectionFailures++
            $state = 'connection_error'
            [IO.File]::AppendAllText($eventsPath, ((@{time_utc = [DateTime]::UtcNow.ToString('o');
                kind = 'check_error'; message = $_.Exception.Message; failures = $connectionFailures} |
                ConvertTo-Json -Compress) + "`n"), $utf8)
            if ($connectionFailures -ge 3 -and -not $connectionAlerted) {
                Record-Event 'error' 'health checks failed' 'Three consecutive checks failed. Check SSH/network and the event log.'
                $connectionAlerted = $true
            }
            Save-Json $monitorPath @{pid = $PID; state = $state; checked_utc = [DateTime]::UtcNow.ToString('o');
                interval_seconds = $IntervalSeconds; run_dir = $RunDir; connection_failures = $connectionFailures;
                error = $_.Exception.Message; notifications = (-not $NoNotify)}
        }
        if ($Once -or $state -eq 'complete') { break }
        for ($i = 0; $i -lt $IntervalSeconds; $i++) {
            if ($script:tray) { [Windows.Forms.Application]::DoEvents() }
            Start-Sleep -Seconds 1
        }
    } while ($true)
    if ($state -eq 'complete' -and $script:tray) {
        for ($i = 0; $i -lt 10; $i++) { [Windows.Forms.Application]::DoEvents(); Start-Sleep -Seconds 1 }
    }
} finally {
    if ($script:tray) { $script:tray.Dispose() }
    $lock.Dispose()
}
