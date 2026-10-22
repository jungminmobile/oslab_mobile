"""e12_4_throttle_check.py — E12-4 측정 환경 확인 (1차 3세션 중단 원인 확인용, 판정과 무관)
1T·4T A0를 전원 절약(EcoQoS) 끔 / 그대로 로 번갈아 2회씩 (워밍업 10 → 60장). 최소화 창에서 실행해 실제 측정 조건과 맞춘다.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx"
out = Path(sys.argv[1])
for rep in range(2):
    for th in (1, 4):
        for nt in (False, True):
            cfg = {"mode": "speed", "model": S, "tcombo": "A0", "threads": th, "warm": 10, "n": 60, "tail": 0.5, "no_throttle": nt}
            p = subprocess.run([sys.executable, str(HERE / "e12_4_speed.py"), json.dumps(cfg)], capture_output=True, text=True)
            r = json.loads(p.stdout.strip().splitlines()[-1])
            line = {"rep": rep, "threads": th, "no_throttle": nt, "p50_ms": r["p50_ms"], "p90_ms": r["p90_ms"], "throttle_off": r["throttle_off"]}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
with open(out, "a", encoding="utf-8") as f:
    f.write("DONE\n")
