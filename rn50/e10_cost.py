"""
e10_cost.py — E10 0·1단계: 단위 정의, 기준 B2 동일성 확인, 단위별 비용표(s_u, t_u, m_u) 측정
==========================================================================================
사전 등록: results/E10_ilp/PREREG.md (커밋 0e99334)

[단위] 23개 (X.0은 INT8 / ds / full 3지선다) — PREREG 0-1
[모델] B0(전부 INT8, attention 계산만 FP32) + 단위 하나만 FP32로 되돌린 27개.
       quantize_onnx.py를 그대로 부른다: --only <B0 블록 전체> --only-exact <AttnPool 투영> --exclude-exact <단위 노드>
[s_u]  CIFAR A + Pets dev, 정수 커널 4T, cos 평균(두 세트 평균)의 B0 대비 증가분
[t_u]  bench_speed 4T n=100, 한 실행에 B0 + 단위 모델 7개 → t_u = 단위 − 같은 실행 B0
[m_u]  파일 크기 차이 (MiB)

사용:  python e10_cost.py units     → units_rn50.json
       python e10_cost.py check     → B2를 새 경로로 다시 만들어 sha256 비교 (다르면 exit 1)
       python e10_cost.py cost      → 27개 양자화 + 스크리닝 + 지연 → cost_rn50.jsonl
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import onnx

import common as C

OUT = C.RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
PRE = C.ONNX_DIR / f"{C.PREFIX}_image_fp32_q1_pre.onnx"
B2_SHA = "687097928a9d82146d486f440e9956ab51d0ebacfffdf34c37e2c432dd3f19cd"
STEM_PFX = "/conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool"
BLOCKS = [f"L1.{i}" for i in range(3)] + [f"L2.{i}" for i in range(4)] + [f"L3.{i}" for i in range(6)] + [f"L4.{i}" for i in range(3)]
pfx = lambda b: f"/layer{b[1]}/layer{b[1]}.{b[3]}/"
ALL = STEM_PFX + "," + ",".join(pfx(b) for b in BLOCKS)
AP = ["/attnpool/MatMul", "/attnpool/MatMul_1", "/attnpool/MatMul_2", "/attnpool/Gemm"]
FILE = lambda tag: f"{C.PREFIX}_image_int8_q1_percentile99999_highres256_s0_E10_{tag}.onnx"


def build_units():
    nodes = [n.name for n in onnx.load(str(PRE), load_external_data=False).graph.node]
    has = set(nodes)
    U = {}
    U["stem.c1"] = {"group": "stem.c1", "nodes": ["/conv1/Conv", "/act1/Relu"]}
    U["stem.c2"] = {"group": "stem.c2", "nodes": ["/conv2/Conv", "/act2/Relu"]}
    U["stem.c3"] = {"group": "stem.c3", "nodes": ["/conv3/Conv", "/act3/Relu", "/avgpool/AveragePool"]}
    for X in (1, 2, 3, 4):
        b = f"/layer{X}/layer{X}.0/"
        U[f"L{X}.0.ds"] = {"group": f"L{X}.0", "nodes": [n for n in nodes if n.startswith(b + "downsample/")] + [b + "Add"]}
        U[f"L{X}.0.full"] = {"group": f"L{X}.0", "nodes": [n for n in nodes if n.startswith(b)]}
    for blk in BLOCKS:
        if blk.endswith(".0"):
            continue
        U[blk] = {"group": blk, "nodes": [n for n in nodes if n.startswith(pfx(blk))]}
    for i, a in enumerate(AP):
        U[f"AP.{['q', 'k', 'v', 'c'][i]}"] = {"group": f"AP.{['q', 'k', 'v', 'c'][i]}", "nodes": [a]}
    for k, u in U.items():
        miss = set(u["nodes"]) - has
        assert u["nodes"] and not miss, (k, miss)
    assert len(U) == 27, len(U)
    (OUT / "units_rn50.json").write_text(json.dumps(U, ensure_ascii=False, indent=1), encoding="utf-8")
    return U


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def quantize(tag, excl_nodes):
    f = C.onnx_path(FILE(tag))
    if f.exists():
        return f
    cmd = [sys.executable, "quantize_onnx.py", "--enc", "image", "--variant", "_q1", "--calib-source", "highres",
           "--calib-n", "256", "--seed", "0", "--method", "percentile", "--only", ALL, "--only-exact", ",".join(AP),
           "--tag", f"E10_{tag}"]
    if excl_nodes:
        cmd += ["--exclude-exact", ",".join(sorted(set(excl_nodes)))]
    r = subprocess.run(cmd, cwd=C.HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(tag, [l for l in r.stdout.splitlines() if l.startswith(("quantize", "saved"))], flush=True)
    assert r.returncode == 0, r.stderr[-1500:]
    return f


def check_b2(U):
    excl = U["stem.c1"]["nodes"] + U["stem.c2"]["nodes"] + U["stem.c3"]["nodes"] + U["L1.0.full"]["nodes"]
    got = sha(quantize("B2chk", excl))
    ok = got == B2_SHA
    C.append_jsonl("e10_check", {"B2_expected": B2_SHA, "B2_rebuilt": got, "match": ok, "meta": C.run_meta()})
    print("B2 sha", got, "MATCH" if ok else "MISMATCH", flush=True)
    return ok


def screen(files):
    cmd = [sys.executable, "evaluate.py", "--exec", "int", "--threads", "4", "--data", "cifar_A", "pets_dev",
           "--models", *[f.name for f in files]]
    r = subprocess.run(cmd, cwd=C.HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-1500:]
    rows = {}
    for line in (OUT / "evaluate.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        rows[(d["model"], d["data"])] = d          # 같은 키는 마지막 행
    return rows


def bench(named):
    """named: [(이름, 파일)] — 한 실행에서 번갈아 4T n=100. 반환 {이름: ms}"""
    before = set(OUT.glob("speed_*_4T.json"))
    cmd = [sys.executable, "bench_speed.py", "--threads", "4", "--n", "100", "--repeats", "1", "--warmup", "20",
           "--models", *[f"{k}={f.name}" for k, f in named]]
    r = subprocess.run(cmd, cwd=C.HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-1500:]
    new = sorted(set(OUT.glob("speed_*_4T.json")) - before)
    d = json.loads(new[-1].read_text(encoding="utf-8"))
    return {k: v["ms_per_img_mean"] for k, v in d["results"].items()}, new[-1].name


def cost(U):
    files = {"B0": quantize("B0", [])}
    for k, u in U.items():
        files[k] = quantize(k, u["nodes"])
    rows = screen(list(files.values()))
    cosm = {k: sum(rows[(f.name, d)]["cos_mean"] for d in ("cifar_A", "pets_dev")) / 2 for k, f in files.items()}
    top1 = {k: {d: rows[(f.name, d)]["top1"] for d in ("cifar_A", "pets_dev")} for k, f in files.items()}
    keys = list(U)
    t, b0s, runs = {}, [], []
    for i in range(0, len(keys), 7):
        part = keys[i:i + 7]
        ms, fn = bench([("B0", files["B0"])] + [(k, files[k]) for k in part])
        b0s.append(ms["B0"])
        runs.append(fn)
        for k in part:
            t[k] = {"ms": ms[k], "b0_same_run": ms["B0"], "t_u": ms[k] - ms["B0"]}
    mib = {k: f.stat().st_size / 2**20 for k, f in files.items()}
    meta = C.run_meta()
    with open(OUT / "cost_rn50.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"unit": "B0", "cos": cosm["B0"], "top1": top1["B0"], "MiB": mib["B0"],
                             "b0_4T_runs_ms": b0s, "speed_files": runs, "meta": meta}, ensure_ascii=False) + "\n")
        for k, u in U.items():
            fh.write(json.dumps({"unit": k, "group": u["group"], "n_nodes": len(u["nodes"]), "s_u": cosm[k] - cosm["B0"],
                                 "cos": cosm[k], "top1": top1[k], **t[k], "m_u": mib[k] - mib["B0"], "MiB": mib[k],
                                 "file": files[k].name, "sha256": sha(files[k])}, ensure_ascii=False) + "\n")
    spread = max(b0s) - min(b0s)
    med_t = sorted(abs(v["t_u"]) for v in t.values())[len(t) // 2]
    print(f"B0 4T runs {['%.2f' % x for x in b0s]} spread {spread:.2f} ms vs median |t_u| {med_t:.2f} ms", flush=True)
    for k in U:
        print(f"{k:12s} s_u {cosm[k] - cosm['B0']:+.4f}  t_u {t[k]['t_u']:+.2f} ms  m_u {mib[k] - mib['B0']:+.1f} MiB", flush=True)


if __name__ == "__main__":
    step = sys.argv[1]
    U = build_units()
    if step == "units":
        print({k: len(v["nodes"]) for k, v in U.items()})
    elif step == "check":
        sys.exit(0 if check_b2(U) else 1)
    elif step == "cost":
        cost(U)


def retime(U, reps=3):
    """PREREG 1단계 규칙: B0 흔들림(최대−최소)이 |t_u| 중앙값보다 크면 t_u를 3회 중앙값으로 다시 잰다.
    1단계 첫 측정에서 흔들림 1.86 ms > 중앙값 1.12 ms 였으므로 27단위 모두 다시 잰다. 첫 값은 t_u_first로 남긴다."""
    from statistics import median
    rows = [json.loads(l) for l in (OUT / "cost_rn50.jsonl").read_text(encoding="utf-8").splitlines()]
    base, units = rows[0], rows[1:]
    files = {"B0": C.onnx_path(FILE("B0")), **{k: C.onnx_path(FILE(k)) for k in U}}
    keys = list(U)
    samples = {k: [] for k in keys}
    for rep in range(reps):
        for i in range(0, len(keys), 7):
            part = keys[i:i + 7]
            ms, fn = bench([("B0", files["B0"])] + [(k, files[k]) for k in part])
            base["b0_4T_runs_ms"].append(ms["B0"])
            base["speed_files"].append(fn)
            for k in part:
                samples[k].append(ms[k] - ms["B0"])
        print(f"retime rep {rep + 1}/{reps} done", flush=True)
    with open(OUT / "cost_rn50.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(base, ensure_ascii=False) + "\n")
        for r in units:
            r["t_u_first"] = r["t_u"]
            r["t_u_reps"] = samples[r["unit"]]
            r["t_u"] = median(samples[r["unit"]])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{r['unit']:12s} t_u first {r['t_u_first']:+.2f} -> median {r['t_u']:+.2f}  reps {['%+.2f' % x for x in samples[r['unit']]]}", flush=True)


if __name__ == "__main__" and sys.argv[1] == "retime":
    retime(build_units())
