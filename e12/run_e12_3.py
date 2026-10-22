"""run_e12_3.py — E12-3 2단계(프로파일 3세션, 사이 2분) + 2b(코어 고정). 계획서 8절 명령과 같은 순서·인자.
(.bat는 한글 경로가 cmd 코드페이지 949에서 깨져 실패 → 같은 명령을 Python으로 실행)
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "E12-3_ORT프로파일" / "results"
SM = r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx"
FM = r"C:\s2art\onnx\e12_3\b32\siglip2_b32_image_fp32.onnx"
P = str(REPO / "e12" / "e12_3_profile_ort.py")
C = ["--img", "256", "--n", "120", "--discard", "20", "--noprof-baseline"]
S, F = f"S={SM}:C3", f"F={FM}:C0"
ENV = str(OUT / "check_env.json")
STEPS = [
    ([S, F], ["1", "2", "4"], ["--session", "1", "--out", str(OUT / "profile.jsonl")], 120),
    ([F, S], ["1", "2", "4"], ["--session", "2", "--out", str(OUT / "profile.jsonl")], 120),
    ([S, F], ["4", "2", "1"], ["--session", "3", "--out", str(OUT / "profile.jsonl")], 60),
    ([S], ["4"], ["--session", "affinity_none", "--out", str(OUT / "affinity.jsonl")], 0),
    ([S], ["4"], ["--affinity-pcores", "--env", ENV, "--session", "affinity_p4", "--out", str(OUT / "affinity.jsonl")], 0),
    ([S], ["1"], ["--affinity-pcores", "--env", ENV, "--session", "affinity_p1", "--out", str(OUT / "affinity.jsonl")], 0),
]

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    print("START", datetime.now().isoformat(timespec="seconds"), flush=True)
    for models, th, extra, rest in STEPS:
        cmd = [sys.executable, "-u", P, "--models", *models, "--threads", *th, *C, *extra]
        print("CMD", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode:
            print("FAILED", datetime.now().isoformat(timespec="seconds"), flush=True)
            sys.exit(1)
        if rest:
            time.sleep(rest)
    print("ALL_DONE", datetime.now().isoformat(timespec="seconds"), flush=True)
