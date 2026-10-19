"""
bench_mem.py — 모델별 최대 메모리(peak RSS) 측정 (E0 ③)
=====================================================

[방법] 모델마다 새 프로세스를 띄워(앞 모델의 메모리가 섞이지 않게)
  ① ORT·numpy를 불러오고 입력 사진을 준비한 상태의 최대 메모리 = 기준(base)
  ② 세션 생성 + 사진 N장 추론(batch 1) 후의 최대 메모리 = peak
  보고값: peak(전체), peak − base(모델이 더한 메모리)
  리눅스는 getrusage(ru_maxrss), 윈도우는 psutil peak_wset(최대 작업 집합)을 쓴다.

[사용]  python bench_mem.py --models fp32=rn50_image_fp32_q1.onnx c3=rn50_image_int8_..._c3.onnx --threads 2
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime

import common as C

CHILD = r'''
import sys, json, platform, numpy as np
def peak_mib():
    if platform.system() == "Windows":
        import psutil; return psutil.Process().memory_info().peak_wset / 2**20
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
sys.path.insert(0, sys.argv[1]); import common as C
import onnxruntime
x = C.to_input(np.load(C.CACHE_DIR / "cifar_A_x.npy", mmap_mode="r")[: int(sys.argv[4])])
base = peak_mib()
s = C.session(C.onnx_path(sys.argv[2]), int(sys.argv[3]))
for i in range(len(x)):
    s.run(None, {"image": x[i:i + 1]})
print(json.dumps({"base_mib": round(base, 1), "peak_mib": round(peak_mib(), 1)}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=파일")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3, help="모델마다 새 프로세스로 몇 번 잴지(중앙값 보고)")
    a = ap.parse_args()
    meta = C.run_meta()
    for kv in a.models:
        k, f = kv.split("=", 1)
        runs = []
        for _ in range(a.repeats):
            out = subprocess.run([sys.executable, "-c", CHILD, str(C.HERE), f, str(a.threads), str(a.n)],
                                 capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
            runs.append(json.loads(out))
        peak = sorted(r["peak_mib"] for r in runs)[len(runs) // 2]
        base = sorted(r["base_mib"] for r in runs)[len(runs) // 2]
        row = {"date": datetime.now().isoformat(timespec="seconds"), "name": k, "file": C.onnx_path(f).name,
               "file_MiB": round(C.onnx_path(f).stat().st_size / 2**20, 1), "threads": a.threads, "n": a.n,
               "peak_rss_MiB": peak, "base_MiB": base, "model_added_MiB": round(peak - base, 1), "runs": runs, "meta": meta}
        print(json.dumps({x: row[x] for x in ("name", "file_MiB", "peak_rss_MiB", "base_MiB", "model_added_MiB")}, ensure_ascii=False))
        C.append_jsonl("memory", row)


if __name__ == "__main__":
    main()
