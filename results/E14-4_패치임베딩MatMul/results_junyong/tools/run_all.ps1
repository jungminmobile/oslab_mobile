. $PSScriptRoot\e14env.ps1
$log = "$PSScriptRoot\run_all.log"
function Say($m) { "[$(Get-Date -Format s)] $m" | Tee-Object -FilePath $log -Append }
function WaitIdle {
    while ($true) {
        $la = (wsl.exe -d Ubuntu-22.04 -- cat /proc/loadavg) -split ' '
        $win = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples.CookedValue | Measure-Object -Average
        if ([double]$la[0] -lt 0.5 -and $win.Average -lt 15) { Say "idle check ok: wsl load1=$($la[0]) win cpu=$([int]$win.Average)%"; return }
        Say "busy: wsl load1=$($la[0]) win cpu=$([int]$win.Average)% -> wait 60s"; Start-Sleep 60
    }
}
Say "power: $(powercfg /getactivescheme)"
$steps = @(
    @('0'), @('rewrite'), @('g2'),
    @('g3','1'), 'REST', @('g3','2'),
    @('acc'), @('mem'),
    @('speed','1'), 'REST', @('speed','2'), 'REST', @('speed','3'),
    @('summary')
)
$i = 0; $n = ($steps | Where-Object { $_ -ne 'REST' }).Count
foreach ($s in $steps) {
    if ($s -eq 'REST') { Say "rest 120s"; Start-Sleep 120; continue }
    $i++
    WaitIdle
    $args_ = @('-u', 'e14\run_e14_4.py', '--step', $s[0])
    if ($s.Count -gt 1) { $args_ += @('--session', $s[1]) }
    Say "START [$i/$n] $($args_ -join ' ')"
    & C:\e14venv\Scripts\python.exe @args_ 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    $rc = $LASTEXITCODE
    Say "END [$i/$n] step=$($s -join ' ') rc=$rc"
    if ($rc -ne 0) { Say "FAILED, stopping"; exit $rc }
}
Say "ALL DONE"
