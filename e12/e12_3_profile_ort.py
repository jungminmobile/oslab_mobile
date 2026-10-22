"""e12_3_profile_ort.py — E12-3 ORT 프로파일 1세션분 (계획서 4절)

준혁 oslab_s2/siglip2/s2_profile_ort.py 기반 (group() 그대로 복사, 원본 수정 안 함). 바뀐 점:
  입력 256 (pets_dev_x[0] 224 캐시를 [0,255]로 되돌려 PIL bilinear 256 → x/127.5-1, 1장 반복),
  모델별 세션 설정 C0/C3 (session_opts), 120장 실행 후 앞 discard장 제외(model_run 시각 기준),
  장마다 perf_counter 벽시계, 무프로파일 기준(같은 자식 프로세스에서 프로파일 없는 세션 120장),
  --affinity-pcores: 프로세스 전체 P코어 제한(psutil, 0-based) + ORT 작업 스레드 고정(ORT용 번호 = 0-based+1).
(모델, 스레드)마다 새 자식 프로세스. 프로파일/무프로파일 순서는 세션 번호 홀짝으로 바꾼다 (order 필드 기록).
원본 프로파일 JSON은 C:\\s2art\\e12_3_prof\\ 에 보관 (커밋 안 함). 결과 행은 --out JSONL에 추가.
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

PROF_DIR = Path(r"C:\s2art\e12_3_prof")

CHILD = r'''
import sys, json, time, os, collections
import numpy as np
sys.path.insert(0, sys.argv[1]); import session_opts as SO
cfg = json.loads(sys.argv[2])
if cfg["pcores0"]:
    import psutil; psutil.Process().cpu_affinity(cfg["pcores0"])
from PIL import Image
x0 = np.load(SO.CACHE / "pets_dev_x.npy", mmap_mode="r")[0]
u8 = np.clip((np.transpose(x0, (1, 2, 0)) + 1) * 127.5, 0, 255).round().astype(np.uint8)
img = np.asarray(Image.fromarray(u8).resize((cfg["img"], cfg["img"]), Image.BILINEAR), np.float32)
x = np.ascontiguousarray(np.transpose(img / 127.5 - 1, (2, 0, 1))[None], np.float32)
import onnxruntime as ort

INT_MM = ("QLinearMatMul", "MatMulInteger", "DynamicQuantizeMatMul", "MatMulIntegerToFloat", "QGemm", "QLinearConv")
FLOAT_MM = ("MatMul", "FusedMatMul", "Gemm")
QDQ = ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear")
def group(op):
    if op in INT_MM: return "int_matmul"
    if op in FLOAT_MM: return "float_matmul"
    if op in QDQ: return "qdq"
    return op

def opts(prof):
    so = SO.make_options(cfg["combo"], cfg["threads"])
    if cfg["pcores0"] and cfg["threads"] > 1:
        so.add_session_config_entry("session.intra_op_thread_affinities", ";".join(str(c + 1) for c in cfg["pcores0"][1:cfg["threads"]]))
    if prof:
        so.enable_profiling = True; so.profile_file_prefix = cfg["prof_prefix"]
    return so

def run(prof):
    s = ort.InferenceSession(cfg["model"], opts(prof), providers=["CPUExecutionProvider"]); nm = s.get_inputs()[0].name
    t = []
    for _ in range(cfg["n"]):
        t0 = time.perf_counter(); s.run(None, {nm: x}); t.append((time.perf_counter() - t0) * 1000)
    w = np.array(t[cfg["discard"]:])
    r = {"wall_p50_ms": round(float(np.median(w)), 3), "wall_mean_ms": round(float(w.mean()), 3), "wall_p95_ms": round(float(np.percentile(w, 95)), 3)}
    if not prof:
        return r
    pf = s.end_profiling(); ev = json.load(open(pf, encoding="utf-8"))
    runs = sorted([e for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run"], key=lambda e: e["ts"])
    assert len(runs) == cfg["n"], (len(runs), cfg["n"])
    kept = runs[cfg["discard"]:]; k = len(kept)
    nodes = sorted([e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")], key=lambda e: e["ts"])
    by_grp, by_op, by_node, cnt = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    node_op = {}
    per_run_mr, per_run_k = [], []
    j = 0
    for rr in kept:
        a, b = rr["ts"], rr["ts"] + rr["dur"]
        while j < len(nodes) and nodes[j]["ts"] < a: j += 1
        ks = 0
        while j < len(nodes) and nodes[j]["ts"] < b:
            e = nodes[j]; op = e.get("args", {}).get("op_name", "?"); d = e["dur"]
            by_grp[group(op)] += d; by_op[op] += d; cnt[op] += 1
            nn = e["name"][:-len("_kernel_time")]; by_node[nn] += d; node_op[nn] = op; ks += d; j += 1
        per_run_mr.append(rr["dur"]); per_run_k.append(ks)
    mr, ks = np.array(per_run_mr) / 1000, np.array(per_run_k) / 1000
    ms = lambda c: {kk: round(v / k / 1000, 4) for kk, v in sorted(c.items(), key=lambda kv: -kv[1])}
    r.update({"model_run_p50_ms": round(float(np.median(mr)), 3), "model_run_mean_ms": round(float(mr.mean()), 3),
              "kernel_mean_ms": round(float(ks.mean()), 3), "outside_mean_ms": round(float((mr - ks).mean()), 3),
              "outside_p50_ms": round(float(np.median(mr - ks)), 3),
              "by_group_ms": ms(by_grp), "by_op_ms": ms(by_op), "node_calls_per_img": {kk: v // k for kk, v in cnt.items()},
              "top_nodes": [{"node": nn, "op": node_op[nn], "ms": round(v / k / 1000, 4)} for nn, v in by_node.most_common(20)],
              "profile_file": pf, "kept_runs": k})
    return r

out = {}
for step in cfg["order"]:
    out["prof" if step == "prof" else "noprof"] = run(step == "prof")
print(json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=경로:조합 (예: S=C:\\..\\a.onnx:C3)")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--discard", type=int, default=20)
    ap.add_argument("--noprof-baseline", action="store_true")
    ap.add_argument("--affinity-pcores", action="store_true")
    ap.add_argument("--env", help="check_env.json (P코어 번호)")
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SO.S2_SIGLIP))
    from scripts import env_info
    env = env_info.collect()
    git = SO.git_state()
    pcores0 = []
    if a.affinity_pcores:
        pcores0 = json.loads(Path(a.env).read_text(encoding="utf-8"))["cores"]["pcores_0based"]
        assert len(pcores0) >= 4, pcores0
    try:
        odd = int(a.session) % 2 == 1
    except ValueError:
        odd = True
    order_steps = (["prof", "noprof"] if odd else ["noprof", "prof"]) if a.noprof_baseline else ["prof"]
    order = 0
    for spec in a.models:
        name, rest = spec.split("=", 1)
        path, combo = rest.rsplit(":", 1)
        assert Path(path).exists(), path
        for th in a.threads:
            t0 = time.time()
            tag = f"{a.session}_{name}_{th}T{'_pin' if pcores0 else ''}"
            cfg = {"model": path, "combo": combo, "threads": th, "img": a.img, "n": a.n, "discard": a.discard,
                   "pcores0": pcores0, "order": order_steps, "prof_prefix": str(PROF_DIR / tag)}
            p = subprocess.run([sys.executable, "-c", CHILD, str(SO.HERE), json.dumps(cfg)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert p.returncode == 0, p.stderr[-3000:]
            r = json.loads(p.stdout.strip().splitlines()[-1])
            row = {"date": datetime.now().isoformat(timespec="seconds"), "session": a.session, "order": order, "model": name,
                   "file": Path(path).name, "combo": combo, "threads": th, "img": a.img, "n": a.n, "discard": a.discard,
                   "affinity_pcores0": pcores0, "step_order": order_steps, **r, "wall_s": round(time.time() - t0, 1),
                   "input": "pets_dev_x[0] (224 캐시) → [0,255] → PIL bilinear 256 → x/127.5-1, 1장 반복",
                   "power": env["power"], "git": git, **env_info.flat(env)}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pr = r["prof"]
            npw = r.get("noprof", {}).get("wall_p50_ms")
            print(f"S{a.session} #{order} {name} {th}T{' pin' if pcores0 else ''} model_run p50={pr['model_run_p50_ms']} "
                  f"kernel={pr['kernel_mean_ms']} outside={pr['outside_mean_ms']} | wall prof={pr['wall_p50_ms']} noprof={npw} "
                  f"({row['wall_s']}s)", flush=True)
            order += 1
    print(f"SESSION {a.session} DONE {datetime.now():%H:%M:%S}", flush=True)


if __name__ == "__main__":
    main()
