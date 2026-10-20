# run_when_idle.ps1 — PC가 한가할 때만 지연 측정을 시작한다 (속도는 다른 부하가 있으면 오염됨, CLAUDE.md 3절)
# 조건: ① 선행 작업 로그에 끝 표시(-WaitLog/-WaitPattern)가 나타남  ② 전체 CPU 사용률 1분 평균이 -Threshold(%) 미만인 상태가 -NeedMinutes분 연속
# 조건이 맞으면 -Command(Git Bash 명령)를 실행하고, 시작 시각·CPU 기록을 로그에 남긴다.
# 사용 예:
#   powershell -File scripts\run_when_idle.ps1 -WaitLog D:\...\g4_run2.log -WaitPattern "G4RUN DONE" `
#       -Command "cd /d/mokerton/oslab_mobile/siglip2 && bash scripts/g3_cumulative.sh speed" -Log D:\...\speed_idle.log
param(
  [string]$WaitLog = "",
  [string]$WaitPattern = "",
  [Parameter(Mandatory = $true)][string]$Command,
  [Parameter(Mandatory = $true)][string]$Log,
  [double]$Threshold = 15,
  [int]$NeedMinutes = 3
)
function Note($m) { "$(Get-Date -Format s) $m" | Out-File -FilePath $Log -Append -Encoding utf8 }
Note "watcher start: wait=$WaitPattern threshold=$Threshold% need=$NeedMinutes min"
if ($WaitLog -and $WaitPattern) {
  while (-not ((Test-Path $WaitLog) -and (Select-String -Path $WaitLog -Pattern $WaitPattern -Quiet))) { Start-Sleep -Seconds 60 }
  Note "prerequisite done"
}
$ok = 0
while ($ok -lt $NeedMinutes) {
  $s = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 12
  $avg = ($s.CounterSamples | Measure-Object CookedValue -Average).Average
  if ($avg -lt $Threshold) { $ok++ } else { $ok = 0 }
  Note ("cpu1min={0:N1}% idle_streak={1}" -f $avg, $ok)
}
Note "idle confirmed, start: $Command"
& "C:\Program Files\Git\bin\bash.exe" -lc "$Command" 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
Note "finished exit=$LASTEXITCODE"
