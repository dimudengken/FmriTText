param([Parameter(Mandatory=$true)][string]$QueueDir, [int]$IntervalSeconds=3600)
$ErrorActionPreference = 'Stop'
if ($QueueDir -notmatch '^/mnt/data2/yjs2026/HLQ/MindLLM/releases/readout-[0-9a-f]+/MindLLM-main/outputs/[A-Za-z0-9_-]+$') { throw 'Invalid queue directory' }
$project = Split-Path $PSScriptRoot -Parent
$local = Join-Path $project 'outputs/readout_queue_monitor'
[IO.Directory]::CreateDirectory($local) | Out-Null
$utf8 = New-Object Text.UTF8Encoding($false)
$lock = [IO.File]::Open((Join-Path $local 'monitor.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
Add-Type -AssemblyName System.Windows.Forms
$tray = New-Object Windows.Forms.NotifyIcon
$tray.Icon = [Drawing.SystemIcons]::Information
$tray.Text = 'MindLLM hourly progress'
$tray.Visible = $true
$ssh = Join-Path $env:SystemRoot 'System32/OpenSSH/ssh.exe'
$runner = ($QueueDir -replace '/outputs/[^/]+$', '') + '/scripts/run_readout_campaign.py'
try {
    do {
        try {
            $raw = & $ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 myserver00 "python3 $runner --mode inspect --output $QueueDir"
            if ($LASTEXITCODE -ne 0) { throw 'SSH progress check failed' }
            $value = ($raw -join "`n") | ConvertFrom-Json
            [IO.File]::WriteAllText((Join-Path $local 'snapshot.json'), ($value | ConvertTo-Json -Depth 20), $utf8)
            [IO.File]::AppendAllText((Join-Path $local 'hourly_progress.jsonl'), (($value | ConvertTo-Json -Depth 20 -Compress) + "`n"), $utf8)
            $message = 'Readout: ' + $value.state + ', ' + $value.phase
            if ($value.dependency) { $message = 'Current experiment: ' + $value.dependency.phase + '. Readout is queued.' }
            if ($value.error) { $message += '. ' + $value.error }
            $tray.ShowBalloonTip(15000, 'MindLLM hourly progress', $message, [Windows.Forms.ToolTipIcon]::Info)
            if ($value.state -in @('complete','needs_attention','budget_exhausted')) { break }
        } catch {
            [IO.File]::AppendAllText((Join-Path $local 'errors.log'), ([DateTime]::UtcNow.ToString('o') + ' ' + $_.Exception.Message + "`n"), $utf8)
            $tray.ShowBalloonTip(15000, 'MindLLM monitor', $_.Exception.Message, [Windows.Forms.ToolTipIcon]::Warning)
        }
        for ($tick=0; $tick -lt $IntervalSeconds; $tick++) { [Windows.Forms.Application]::DoEvents(); Start-Sleep -Seconds 1 }
    } while ($true)
} finally { $tray.Dispose(); $lock.Dispose() }
