"""run_e12_4.py — E12-4 단계별 Python 드라이버 (계획서 6·8절)

--step 0      : 다른 측정 확인, S sha, ORT v1.25.0 헤더 키 기록(gh api), A0 미지정 vs "1" (4T, 20장, 번갈아 2회씩), env.json
--step smoke  : A0·A1 × 4T × 2장
--step g1     : 스레드 1·2·3·4마다 A0 vs A0 대조군 → A1~A4 vs A0 (50장, 최대 절대 차이), 3T vs 4T(A0) 기록 → equal.json
--step speed --session k : 조합·스레드 순서 회전, (조합 × 스레드)마다 새 자식 프로세스, 워밍업 20 → 200장 → speed.jsonl
--step summary : 5절 규칙 판정 → summary.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_opts as SO   # noqa: E402

REPO = HERE.parent
OUT = REPO / "results" / "E12-4_스레드spinning" / "results"
S = Path(r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx")
S_SHA = "7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6"
CHILD = str(HERE / "e12_4_speed.py")
COMBOS = ["A0", "A1", "A2", "A3", "A4"]
ORDERS = {1: (["A0", "A1", "A2", "A3", "A4"], [1, 2, 3, 4]),
          2: (["A2", "A3", "A4", "A0", "A1"], [4, 3, 2, 1]),
          3: (["A4", "A1", "A3", "A2", "A0"], [2, 4, 1, 3])}


def child(cfg):
    p = subprocess.run([sys.executable, CHILD, json.dumps(cfg)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def speed_cfg(tc, th, warm=20, n=200):
    return {"mode": "speed", "model": str(S), "tcombo": tc, "threads": th, "warm": warm, "n": n, "tail": 0.5}


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def env_rows():
    sys.path.insert(0, str(SO.S2_SIGLIP))
    from scripts import env_info
    env = env_info.collect()
    env["git_oslab_mobile"] = SO.git_state()
    return env, env_info


def step0():
    others = subprocess.run(["powershell", "-NoProfile", "-c",
                             "Get-CimInstance Win32_Process -Filter \"name like 'python%'\" | % { $_.CommandLine }"],
                            capture_output=True, text=True, errors="replace").stdout
    busy = [l for l in others.splitlines() if ("e12" in l or "bench" in l or "eval" in l) and "run_e12_4" not in l and "e12_4_speed" not in l]
    print("다른 측정 프로세스:", busy or "없음", flush=True)
    assert not busy, busy
    h = hashlib.sha256()
    with open(S, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    sha_ok = h.hexdigest() == S_SHA
    print("S sha", h.hexdigest(), "OK" if sha_ok else "불일치", flush=True)
    assert sha_ok
    hdr = subprocess.run(["gh", "api", "repos/microsoft/onnxruntime/contents/include/onnxruntime/core/session/"
                          "onnxruntime_session_options_config_keys.h?ref=v1.25.0", "-H", "Accept: application/vnd.github.raw"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    keys = {k: (f'"{k}"' in hdr) for k in (SO.SPIN, SO.BLOCK, SO.FSTOP)}
    spin_lines = [l.strip() for l in hdr.splitlines() if "spin" in l.lower() or "dynamic_block" in l.lower()]
    print("헤더 키", keys, flush=True)
    env, env_info = env_rows()
    env_info.write_env(OUT, env)
    rows = []
    for rep in range(2):
        for tc in ("A0", "A0_1"):
            r = child(speed_cfg(tc, 4, warm=10, n=20))
            rows.append({"rep": rep, "tcombo": tc, **r})
            print(f"0단계 {tc} rep{rep}: p50 {r['p50_ms']} cpu/img {r['cpu_ms_per_img']} tail {r['tail_cpu_ms']}", flush=True)
            time.sleep(3)
    med = lambda tc, k: float(np.median([r[k] for r in rows if r["tcombo"] == tc]))
    d_p50 = med("A0", "p50_ms") / med("A0_1", "p50_ms") - 1
    d_cpu = med("A0", "cpu_ms_per_img") / med("A0_1", "cpu_ms_per_img") - 1
    same = abs(d_p50) <= 0.03 and abs(d_cpu) <= 0.10
    res = {"date": datetime.now().isoformat(timespec="seconds"), "sha_ok": sha_ok, "sha256": h.hexdigest(), "header_keys_v1.25.0": keys,
           "header_spin_lines": spin_lines, "a0_vs_1": rows, "d_p50": d_p50, "d_cpu": d_cpu, "A0_equals_spin1": same}
    (OUT / "step0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"A0(미지정) vs 1: p50 {d_p50:+.2%}, CPU {d_cpu:+.2%} → {'같음, A0=1 확정' if same else '다름 → 멈추고 보고'}", flush=True)
    assert same


def smoke():
    for tc in ("A0", "A1"):
        r = child(speed_cfg(tc, 4, warm=2, n=2))
        print("smoke", tc, r, flush=True)


def g1():
    tmp = Path(tempfile.mkdtemp(prefix="e12_4_g1_"))

    def run(tc, th, tag):
        f = tmp / f"{tag}.npz"
        r = child({"mode": "equal", "model": str(S), "tcombo": tc, "threads": th, "n": 50, "save": str(f)})
        z = np.load(f)
        return [z[k] for k in sorted(z.files)], r["keys"]

    def maxdiff(a, b):
        return max(float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) for x, y in zip(a, b))

    res = {"control": {}, "compare": {}, "keys": {}}
    base = {}
    for th in (1, 2, 3, 4):
        a, _ = run("A0", th, f"A0a_{th}")
        b, _ = run("A0", th, f"A0b_{th}")
        base[th] = a
        res["control"][th] = maxdiff(a, b)
        print(f"G1 대조군 {th}T A0 vs A0: {res['control'][th]}", flush=True)
    if any(v != 0 for v in res["control"].values()):
        (OUT / "equal.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print("대조군 불일치 → 멈추고 보고", flush=True)
        sys.exit(2)
    for th in (1, 2, 3, 4):
        for tc in ("A1", "A2", "A3", "A4"):
            o, keys = run(tc, th, f"{tc}_{th}")
            res["compare"][f"{tc}_{th}T"] = maxdiff(base[th], o)
            res["keys"][tc] = keys
            print(f"G1 {tc} {th}T vs A0: {res['compare'][f'{tc}_{th}T']}", flush=True)
    res["A0_3T_vs_4T"] = maxdiff(base[3], base[4])
    res["pass"] = {tc: all(res["compare"][f"{tc}_{th}T"] == 0 for th in (1, 2, 3, 4)) for tc in ("A1", "A2", "A3", "A4")}
    (OUT / "equal.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("G1 pass", res["pass"], "3T vs 4T", res["A0_3T_vs_4T"], flush=True)


def speed(session):
    env, env_info = env_rows()
    env_info.write_env(OUT, env)
    combos, threads = ORDERS[session % 10]   # 1T 대조군 규칙 재측정: 세션 1x = 세션 x 다시 (같은 순서)
    order = 0
    for tc in combos:
        for th in threads:
            t0 = time.time()
            r = child(speed_cfg(tc, th))
            row = {"date": datetime.now().isoformat(timespec="seconds"), "session": session, "order": order, "tcombo": tc,
                   "threads": th, "warm": 20, "n": 200, **r, "wall_s": round(time.time() - t0, 1), "file": S.name,
                   "power": env["power"], "git": env["git_oslab_mobile"], **env_info.flat(env)}
            append(OUT / "speed.jsonl", row)
            print(f"S{session} #{order} {tc} {th}T p50 {r['p50_ms']} p90 {r['p90_ms']} cpu/img {r['cpu_ms_per_img']} "
                  f"tail {r['tail_cpu_ms']} ({row['wall_s']}s)", flush=True)
            order += 1
    print(f"SESSION {session} DONE {datetime.now():%H:%M:%S}", flush=True)


def summary():
    rows = [json.loads(l) for l in (OUT / "speed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    eq = json.loads((OUT / "equal.json").read_text(encoding="utf-8"))
    allsess = sorted({r["session"] for r in rows})
    sessions = [x for x in allsess if x < 10 and (x + 10) not in allsess] + [x for x in allsess if x >= 10]   # 재측정 세션이 있으면 원래 세션 대신 사용 (원래 값은 speed.jsonl에 남음)
    get = lambda s, tc, th, k: next(r[k] for r in rows if r["session"] == s and r["tcombo"] == tc and r["threads"] == th)
    delta, abs_ = {}, {}
    drift = []
    for tc in COMBOS:
        for th in (1, 2, 3, 4):
            for k in ("p50_ms", "p90_ms", "p99_ms", "cpu_ms_per_img", "cpu_ms_per_img_loop", "tail_cpu_ms"):
                vals = [get(s, tc, th, k) / get(s, "A0", th, k) - 1 if get(s, "A0", th, k) else None for s in sessions]
                delta[f"{tc}_{th}T_{k}"] = {"per_session": vals, "median": float(np.median([v for v in vals if v is not None])) if any(v is not None for v in vals) else None}
                abs_[f"{tc}_{th}T_{k}"] = {"per_session": [get(s, tc, th, k) for s in sessions], "median": float(np.median([get(s, tc, th, k) for s in sessions]))}
            if th == 1 and tc != "A0":
                for s, v in zip(sessions, delta[f"{tc}_1T_p50_ms"]["per_session"]):
                    if abs(v) > 0.02:
                        drift.append({"session": s, "tcombo": tc, "d": v})
    dm = lambda tc, th, k: delta[f"{tc}_{th}T_{k}"]["median"]
    verdict = {}
    for tc in ("A1", "A2", "A3", "A4"):
        g = eq["pass"][tc]
        lowok = dm(tc, 1, "p50_ms") <= 0.02 and dm(tc, 2, "p50_ms") <= 0.02
        spd = g and dm(tc, 4, "p50_ms") <= -0.03 and lowok and dm(tc, 4, "p90_ms") <= 0.05
        cpu = g and abs(dm(tc, 4, "p50_ms")) <= 0.02 and lowok and dm(tc, 4, "cpu_ms_per_img") <= -0.10
        verdict[tc] = "채택(속도)" if spd else ("채택(CPU 효율)" if cpu else "효과 없음")
    res = {"sessions": sessions, "all_sessions": allsess, "delta": delta, "abs": abs_, "verdict": verdict, "drift_1T": drift, "g1": eq["pass"]}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("| 조합 | G1 | 1T p50 | 2T p50 | 3T p50 | 4T p50 | 4T p90 | 4T p99 | 4T CPU/장 | 꼬리 CPU | 판정 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda tc, th, k: f"{abs_[f'{tc}_{th}T_{k}']['median']:.2f}" + ("" if tc == "A0" else f" ({dm(tc, th, k):+.1%})")
    for tc in COMBOS:
        print(f"| {tc} | {'—' if tc == 'A0' else ('통과' if eq['pass'][tc] else '불통과')} | {f(tc,1,'p50_ms')} | {f(tc,2,'p50_ms')} | {f(tc,3,'p50_ms')} | "
              f"{f(tc,4,'p50_ms')} | {f(tc,4,'p90_ms')} | {abs_[f'{tc}_4T_p99_ms']['median']:.2f} | {f(tc,4,'cpu_ms_per_img')} | "
              f"{abs_[f'{tc}_4T_tail_cpu_ms']['median']:.1f} | {verdict.get(tc, '—')} |")
    print("1T drift:", drift or "없음")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["0", "smoke", "g1", "speed", "summary"])
    ap.add_argument("--session", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"0": step0, "smoke": smoke, "g1": g1, "summary": summary}.get(a.step, lambda: speed(a.session))()
