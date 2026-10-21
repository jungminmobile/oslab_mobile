"""bench_speed_session.py — E12-1 속도 측정 1세션분 (계획서 5-3)

모델 → 스레드(1·2·4) → 조합(--combos에 준 순서) 순으로, 조합마다 새 자식 프로세스에서
세션 생성 → 워밍업 W장 → N장 측정(batch 1, 장당 perf_counter). 장당 지연 중앙값이 판정 값.
판정은 같은 세션 안 C0 대비 비율의 3세션 중앙값 (REPORT 단계에서 계산). 실제 실행 순서는 order 필드로 기록.
입력 값은 속도와 무관하므로 고화질 dev(pets_dev) 캐시 앞 50장을 돌려 쓴다 (224, batch 1).
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402

CHILD = r'''
import sys, json, time
import numpy as np
sys.path.insert(0, sys.argv[1]); import session_opts as SO
model, combo, threads, warm, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
x = np.ascontiguousarray(np.load(SO.CACHE / "pets_dev_x.npy", mmap_mode="r")[:50], np.float32)
s = SO.make_session(model, combo, threads); name = s.get_inputs()[0].name
for i in range(warm):
    s.run(None, {name: x[i % 50:i % 50 + 1]})
t = []
for i in range(n):
    xi = x[i % 50:i % 50 + 1]; t0 = time.perf_counter(); s.run(None, {name: xi}); t.append((time.perf_counter() - t0) * 1000)
t = np.array(t)
print(json.dumps({"p50_ms": round(float(np.median(t)), 3), "mean_ms": round(float(t.mean()), 3), "std_ms": round(float(t.std()), 3),
                  "p95_ms": round(float(np.percentile(t, 95)), 3), "min_ms": round(float(t.min()), 3)}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="이름=경로")
    ap.add_argument("--combos", nargs="+", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SO.S2_SIGLIP))
    from scripts import env_info
    env = env_info.collect()
    env["git_oslab_mobile"] = SO.git_state()
    env_info.write_env(out.parent, env)
    order = 0
    for name, path in SO.parse_models(a.models):
        for th in a.threads:
            for c in a.combos:
                t0 = time.time()
                p = subprocess.run([sys.executable, "-c", CHILD, str(SO.HERE), str(path), c, str(th), str(a.warmup), str(a.n)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                assert p.returncode == 0, p.stderr[-2000:]
                r = json.loads(p.stdout.strip().splitlines()[-1])
                row = {"date": datetime.now().isoformat(timespec="seconds"), "session": a.session, "order": order,
                       "model": name, "file": path.name, "threads": th, "combo": c, "warmup": a.warmup, "n": a.n, **r,
                       "wall_s": round(time.time() - t0, 1), "power": env["power"], "git": env["git_oslab_mobile"],
                       **env_info.flat(env)}
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"S{a.session} #{order} {name} {th}T {c} p50={r['p50_ms']} ms p95={r['p95_ms']} ({row['wall_s']}s)", flush=True)
                order += 1
    print(f"SESSION {a.session} DONE {datetime.now():%H:%M:%S}", flush=True)


if __name__ == "__main__":
    main()
