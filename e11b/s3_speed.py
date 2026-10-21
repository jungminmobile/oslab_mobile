"""s3_speed.py — FP32·R0·S3 구성 속도(1·2·4T × 3세션, 세션 사이 2분; FP32는 1세션만)와 메모리(rn50 bench_mem, 4T)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S2 = Path(__file__).resolve().parents[2] / "oslab_s2" / "siglip2"
RN = Path(__file__).resolve().parents[1] / "rn50"
plan = json.load(open(sys.argv[1], encoding="utf-8"))
R0 = "siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx"
cfg = [f"{c['tag']}=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_{c['tag']}.onnx" for c in plan["configs"]]
for s in (1, 2, 3):
    models = (["fp32=siglip2_image_fp32.onnx"] if s == 1 else []) + [f"R0={R0}", *cfg]
    r = subprocess.run([sys.executable, "-u", str(S2 / "s2_bench_speed.py"), "--warmup", "20", "--n", "200", "--repeats", "3",
                        "--threads", "1", "2", "4", "--models", *models], cwd=S2)
    print(f"speed session {s} rc={r.returncode} {time.strftime('%H:%M:%S')}", flush=True)
    if s < 3:
        time.sleep(120)
env = {**os.environ, "RN50_ONNX": r"C:\s2art\onnx"}
rn_py = str(RN / ".venv" / "Scripts" / "python.exe")
r = subprocess.run([rn_py, "-u", "bench_mem.py", "--threads", "4", "--models", "fp32=siglip2_image_fp32.onnx", f"R0={R0}", *cfg],
                   cwd=RN, env=env)
print("mem rc", r.returncode, flush=True)
