"""용 요청 지연 측정: 3세션 × 스레드 1·2·4, 한 실행에서 세 모델 번갈아 (연구실 g3rq_ort_speed.sh와 같은 순서).
각 실행 전후로 Windows CPU 부하·전원 상태를 gate.log에 기록. 실행 전 CPU 부하가 20% 넘으면 최대 3분 기다린 뒤 진행(기록 남김)."""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

W = Path(r"D:\siglip2_xcheck")
BASE = Path(r"C:\study\3\2\2026 모바일 학술대회")
WT = BASE / "oslab_xcheck" / "siglip2"
PY = BASE / "s2_venv" / "Scripts" / "python.exe"
OUT = W / "art" / "results" / "speed"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "gate.log"
M = ["fp32=siglip2_image_fp32.onnx",
     "R0_5eee=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx",
     "M2rot_ort=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx"]
env = dict(os.environ, SIGLIP2_ARTIFACTS=str(W / "art"), SIGLIP2_DATA=str(W / "data"), RN50_RESULTS=str(OUT),
           RN50_DATA=str(BASE / "oslab_mobile" / "rn50" / "data"), OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
           PYTHONIOENCODING="utf-8")
PS = ("$c=(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples.CookedValue;"
      "$b=(Get-CimInstance Win32_Battery);"
      "'cpu={0:N1}% plugged={1} batt={2}%' -f ($c|Measure-Object -Average).Average,($b.BatteryStatus -eq 2),$b.EstimatedChargeRemaining")


def state():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True)
    return r.stdout.strip()


def log(msg):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, encoding="mbcs").stdout.strip()
commit = subprocess.run(["git", "-C", str(WT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
log(f"start commit={commit} power_scheme={scheme}")
for s in (1, 2, 3):
    for t in (1, 2, 4):
        for _ in range(9):
            st = state()
            if float(st.split("cpu=")[1].split("%")[0]) < 20:
                break
            log(f"s={s} t={t} wait {st}")
            time.sleep(20)
        log(f"s={s} t={t} pre {st}")
        r = subprocess.run([str(PY), "-u", "s2_bench_speed.py", "--warmup", "20", "--n", "200", "--repeats", "3",
                            "--threads", str(t), "--models", *M], cwd=WT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        with open(OUT / f"bench_s{s}_t{t}.log", "w", encoding="utf-8") as f:
            f.write(r.stdout + "\n---stderr---\n" + r.stderr)
        log(f"s={s} t={t} done rc={r.returncode} post {state()}")
        if r.returncode:
            log("STOP: bench failed — " + r.stderr[-400:].replace("\n", " | "))
            sys.exit(1)
log("SPEED_DONE")
