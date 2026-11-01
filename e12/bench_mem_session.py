"""bench_mem_session.py — E12-1 메모리 측정 (계획서 5-2)

(모델 × 조합)마다 새 자식 프로세스를 띄워 잰다 (앞 세션의 메모리가 섞이지 않게). 반복은 바깥, 조합은 안쪽 루프
(반복 r마다 C0~C4를 한 번씩 → 시간에 따른 흔들림이 조합 사이에 고르게 퍼진다). 워밍업 없음.
  base      : numpy·ORT import + 입력 준비 후, 세션 생성 전 working set (MiB)
  M-create  : 세션 생성 직후 peak working set
  M-infer   : 추론 구간 최대 working set (별도 스레드, interval ms 간격)
  M-private : 추론 구간 최대 private bytes
  M-peak    : 프로세스 전체 peak working set (psutil peak_wset; rn50/bench_mem.py와 같은 정의) ← 주 지표
입력 값은 메모리와 무관하므로 고화질 dev(pets_dev) 캐시 앞 N장을 그대로 쓴다 (224, batch 1).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402

CHILD = r'''
import sys, json, time, threading
import numpy as np, psutil
sys.path.insert(0, sys.argv[1]); import session_opts as SO
import onnxruntime
model, combo, threads, n, interval = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]) / 1000
P = psutil.Process(); MiB = 2 ** 20
x = np.ascontiguousarray(np.load(SO.CACHE / "pets_dev_x.npy", mmap_mode="r")[:n], np.float32)
m = P.memory_info(); base, base_peak = m.wset / MiB, m.peak_wset / MiB
t0 = time.perf_counter(); s = SO.make_session(model, combo, threads); t_create = time.perf_counter() - t0
m = P.memory_info(); create_peak, create_wset, create_priv = m.peak_wset / MiB, m.wset / MiB, m.private / MiB
mx = {"wset": 0.0, "priv": 0.0}; stop = threading.Event()
def sample():
    while True:
        mi = P.memory_info(); mx["wset"] = max(mx["wset"], mi.wset / MiB); mx["priv"] = max(mx["priv"], mi.private / MiB)
        if stop.wait(interval): break
th = threading.Thread(target=sample, daemon=True); th.start()
name = s.get_inputs()[0].name; t0 = time.perf_counter()
for i in range(n):
    s.run(None, {name: x[i:i + 1]})
t_infer = time.perf_counter() - t0
stop.set(); th.join()
m = P.memory_info()
mx["wset"] = max(mx["wset"], m.wset / MiB); mx["priv"] = max(mx["priv"], m.private / MiB)
r = {"base_MiB": base, "base_peak_MiB": base_peak, "create_peak_MiB": create_peak, "create_wset_MiB": create_wset,
     "create_private_MiB": create_priv, "infer_wset_max_MiB": mx["wset"], "infer_private_max_MiB": mx["priv"],
     "peak_MiB": m.peak_wset / MiB, "t_create_s": t_create, "t_infer_s": t_infer}
print(json.dumps({k: round(v, 3 if k.startswith("t_") else 1) for k, v in r.items()}))
'''


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=경로")
    ap.add_argument("--combos", nargs="+", default=list(SO.COMBOS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--interval-ms", type=float, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    git = SO.git_state()
    for name, path in SO.parse_models(a.models):
        runs = {c: [] for c in a.combos}
        for rep in range(a.repeats):
            for c in a.combos:
                p = subprocess.run([sys.executable, "-c", CHILD, str(SO.HERE), str(path), c, str(a.threads), str(a.n), str(a.interval_ms)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                assert p.returncode == 0, p.stderr[-2000:]
                r = json.loads(p.stdout.strip().splitlines()[-1])
                runs[c].append(r)
                row = {"type": "run", "date": datetime.now().isoformat(timespec="seconds"), "model": name, "file": path.name,
                       "combo": c, "rep": rep, "threads": a.threads, "n": a.n, "interval_ms": a.interval_ms, **r, "git": git}
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{name} rep{rep} {c} peak={r['peak_MiB']} infer={r['infer_wset_max_MiB']} create={r['create_peak_MiB']}"
                      f" priv={r['infer_private_max_MiB']} base={r['base_MiB']}", flush=True)
        keys = [k for k in runs[a.combos[0]][0] if not k.startswith("t_")]
        c0 = median([r["peak_MiB"] for r in runs["C0"]]) if "C0" in runs else None
        for c in a.combos:
            med = {k: round(median([r[k] for r in runs[c]]), 1) for k in keys}
            same_dir = None
            if c0 is not None and c != "C0":   # 반복 r마다 같은 반복의 C0과 비교 → 3회 모두 감소인지
                same_dir = all(runs[c][i]["peak_MiB"] < runs["C0"][i]["peak_MiB"] for i in range(a.repeats))
            row = {"type": "summary", "date": datetime.now().isoformat(timespec="seconds"), "model": name, "file": path.name,
                   "file_MiB": round(path.stat().st_size / 2 ** 20, 1), "combo": c, "settings": SO.describe(c),
                   "threads": a.threads, "n": a.n, "repeats": a.repeats, "median": med,
                   "peak_delta_vs_C0_pct": None if c0 is None else round(100 * (med["peak_MiB"] - c0) / c0, 2),
                   "peak_lower_all_reps": same_dir, "git": git}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("SUMMARY", name, c, json.dumps(med), "dPeak%", row["peak_delta_vs_C0_pct"], "all_lower", same_dir, flush=True)


if __name__ == "__main__":
    main()
