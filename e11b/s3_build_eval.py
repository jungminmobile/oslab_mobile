"""s3_build_eval.py — s3_plan.json의 구성을 만들고(s2_quantize, R0와 같은 명령 + --exclude) dev 전체 정확도(s2_paired_ci)까지."""
import json
import subprocess
import sys
from pathlib import Path

S2 = Path(__file__).resolve().parents[2] / "oslab_s2" / "siglip2"
Q = ["--calib-source", "highres", "--calib-n", "256", "--op-types", "MatMul", "--method", "percentile",
     "--percentile", "99.999", "--hist-chunk", "16", "--act-type", "u8", "--reduce-range", "--dump-ops"]
plan = json.load(open(sys.argv[1], encoding="utf-8"))
files = []
for c in plan["configs"]:
    r = subprocess.run([sys.executable, "-u", str(S2 / "s2_quantize.py"), *Q, "--exclude", c["exclude"], "--tag", c["tag"], "--overwrite"],
                       cwd=S2, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-600:], flush=True)
    assert r.returncode == 0, r.stderr[-2000:]
    files.append(f"siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_{c['tag']}.onnx")
r = subprocess.run([sys.executable, "-u", str(S2 / "s2_paired_ci.py"), "--threads", "6", "--models", *files], cwd=S2)
assert r.returncode == 0
print("FILES", " ".join(files))
