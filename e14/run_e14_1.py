"""run_e14_1.py — E14-1 M2 마스크 식 정리 단계별 드라이버 (계획서 results/E14-1_M2마스크정리/E14-1_M2마스크정리.md 7·9절)

--step 0        다른 측정 확인, O sha, Sub 패턴 분류·마스크 0/1, env.json → step0.json
--step rewrite  W1·W2 생성 + G1 (Sub 수, Q/DQ 수·상수 sha, 출력 shape, ORT 최적화 후 융합 노드 수, 앞·뒤 노드, 새 상수 크기) → g1.json
--step smoke    O·W1·W2 × 2장 실행
--step g2       G2 (a) dev 6×200 O 대비 cos·top-1 (b) W1 vs W2 max|Δ| (c) 2장 중간 출력 → g2.json
--step acc      (G2b) O·W1·W2 dev 전체 + 층화 짝 부트스트랩 → acc.json
--step mem      O·W1·W2 × 3회 peak 메모리 (C3, 4T, 50장) → memory.jsonl
--step speed --session k   (모델 × 스레드)마다 새 자식 프로세스 (e12/e12_4_speed.py, A0 = C3 + 스레드 풀 기본값, EcoQoS 끔) → speed.jsonl
--step summary  6절 판정 → summary.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "e12"))
import rewrite_mask as RW     # noqa: E402
import session_opts as SO     # noqa: E402

OUT = REPO / "results" / "E14-1_M2마스크정리" / "results"
O = Path(r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx")
O_SHA = "7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6"
WORK = Path(r"C:\s2art_b32\e14")
ART = Path(r"C:\s2art_b32")
MODELS = {"O": O, "W1": WORK / "e14_1_W1_mulAdd.onnx", "W2": WORK / "e14_1_W2_where.onnx"}
DEV = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HRK = DEV[1:]
SPEED_CHILD = str(REPO / "e12" / "e12_4_speed.py")
ORDERS = {1: (["O", "W1", "W2"], [1, 2, 4]), 2: (["W2", "O", "W1"], [4, 2, 1]), 3: (["W1", "W2", "O"], [2, 4, 1])}
FUSED = ("QLinear", "MatMulIntegerToFloat", "QuantizeLinear", "DequantizeLinear", "DynamicQuantize", "QGemm", "QAttention")


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    tracked = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    untracked = run("status", "--porcelain", "--untracked-files=all", "--", "e14")
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(tracked or untracked)}


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def append(name, row):
    with open(OUT / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def env_rows():
    sys.path.insert(0, str(SO.S2_SIGLIP))
    from scripts import env_info
    env = env_info.collect()
    env["git_oslab_mobile"] = git_state()
    return env, env_info


def other_python():
    import psutil
    me = os.getpid()
    bad = []
    for p in psutil.process_iter(["name", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if p.pid != me and (p.info["name"] or "").lower().startswith("python") and "mcp" not in cmd.lower() and "run_e14_1" not in cmd:
            bad.append({"pid": p.pid, "cmd": cmd[:200]})
    return bad


def session(path, threads=4, opt_path=None, disable_opt=False):
    import onnxruntime as ort
    so = SO.make_options("C3", threads)
    if disable_opt:     # G2(c) 진단: ONNX 그대로의 값 (ORT 최적화가 중간 텐서 값을 바꾸는지 분리)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if opt_path:
        so.optimized_model_filepath = str(opt_path)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def load_dev(key, n=None):
    pix = np.load(ART / "dev256" / f"{key}_pix.npy", mmap_mode="r")
    y = np.load(ART / "dev256" / f"{key}_y.npy")
    W = np.load(ART / "clf" / f"{key}_clf.npy")
    return (pix, y, W) if n is None else (pix[:n], y[:n], W)


def to_x(pix):
    return np.ascontiguousarray((np.asarray(pix, np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2))


def embed(s, pix):
    nm = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {nm: to_x(pix[i:i + 1])})[0] for i in range(len(pix))]).astype(np.float32)


def l2n(e):
    return e / np.linalg.norm(e, axis=1, keepdims=True)


# ---------------------------------------------------------------- 0

def step0(a):
    import onnx
    others = other_python()
    print("다른 python (MCP 제외):", others or "없음", flush=True)
    assert not others, others
    sha = RW.file_sha(O)
    print("O sha", sha, "OK" if sha == O_SHA else "불일치", flush=True)
    assert sha == O_SHA
    m = onnx.load(str(O))
    sites = RW.find_sites(m)
    kinds = Counter(r["kind"] for r in sites)
    ops = Counter(n.op_type for n in m.graph.node)
    masks = sorted({r["m"] for r in sites if "m" in r})
    mask_info = {}
    init = {i.name: i for i in m.graph.initializer}
    from onnx import numpy_helper as nh
    for k in masks:
        v = nh.to_array(init[k])
        mask_info[k] = {"shape": list(v.shape), "unique": sorted(set(np.unique(v).tolist())), "ones_pos": [int(i) for i in np.flatnonzero(v.reshape(-1))]}
    env, env_info = env_rows()
    env_info.write_env(OUT, env)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "sha256": sha, "sha_ok": True, "nodes": len(m.graph.node),
           "op_counts": dict(ops), "n_sub": ops["Sub"], "kinds": dict(kinds), "masks": mask_info,
           "all_masks_binary": all(r.get("m_binary", False) for r in sites), "sites": sites, "other_python": others, "git": git_state()}
    dump("step0.json", res)
    for r in sites:
        print(f"{r['sub']:>14} {r['kind']:6} m={r.get('m')} pos={r.get('m_pos')} Y={'X' if r.get('Y') == r['X'] else r.get('Y')} {r.get('why', '')}"
              f"{' up=' + str(r.get('chain_up')) + ' overlap=' + str(r.get('overlap')) if 'chain_up' in r else ''}", flush=True)
    print("Sub", ops["Sub"], "종류", dict(kinds), "마스크 0/1", res["all_masks_binary"], flush=True)
    assert ops["Sub"] == 27 and res["all_masks_binary"]


# ---------------------------------------------------------------- rewrite + G1

def ort_ops(path, tag):
    import onnx
    opt = WORK / f"{tag}_ortopt.onnx"
    session(path, 4, opt)
    mo = onnx.load(str(opt))
    c = Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in mo.graph.node)
    fused = {k: v for k, v in c.items() if any(f in k for f in FUSED)}
    return {"n_nodes": len(mo.graph.node), "ops": dict(sorted(c.items())), "fused": dict(sorted(fused.items())), "file": str(opt)}


def shapes(model, names):
    import onnx
    inf = onnx.shape_inference.infer_shapes(model)
    vi = {v.name: v for v in list(inf.graph.value_info) + list(inf.graph.output)}
    f = lambda v: [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]
    return {n: (f(vi[n]) if n in vi else None) for n in names}


def step_rewrite(a):
    import onnx
    WORK.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(O))
    fp0 = RW.qdq_fingerprint(m)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "O": {"qdq": fp0, "n_nodes": len(m.graph.node),
           "n_sub": sum(n.op_type == "Sub" for n in m.graph.node), "size_MiB": round(O.stat().st_size / 2 ** 20, 3)}, "git": git_state()}
    left = sum(r["kind"] == "other" for r in RW.find_sites(m))
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n.op_type)
    prod = {o: n.op_type for n in m.graph.node for o in n.output}
    init = {i.name for i in m.graph.initializer}
    for v in ("W1", "W2"):
        mv, sites, extra = RW.rewrite(m, v)
        onnx.save(mv, str(MODELS[v]))
        fp = RW.qdq_fingerprint(mv)
        outs = [r["out"] for r in sites]
        sh0, sh1 = shapes(m, outs), shapes(mv, outs)
        ops = Counter(n.op_type for n in mv.graph.node)
        res[v] = {"file": str(MODELS[v]), "sha256": RW.file_sha(MODELS[v]), "size_MiB": round(MODELS[v].stat().st_size / 2 ** 20, 3),
                  "n_nodes": len(mv.graph.node), "n_sub": ops["Sub"], "n_where": ops["Where"], "n_rewritten": len(sites),
                  "kinds_rewritten": dict(Counter(r["kind"] for r in sites)), "qdq": fp, **extra,
                  "shape_same": all(sh0[o] == sh1[o] and sh0[o] is not None for o in outs), "shapes": sh1,
                  "neighbors": [{"site": r["sub"], "kind": r["kind"], "Y_from": "INIT" if r["Y"] in init else prod.get(r["Y"], "INPUT"),
                                 "out_to": cons.get(r["out"], [])} for r in sites]}
        res[v]["checks"] = {"sub_left_eq_other": ops["Sub"] == left,
                            "qdq_counts_same": fp["counts"] == fp0["counts"],
                            "qdq_scale_zp_same": fp["scale_zp_sha256"] == fp0["scale_zp_sha256"],
                            "qdq_weight_same": fp["dq_weight_sha256"] == fp0["dq_weight_sha256"],
                            "shape_same": res[v]["shape_same"]}
        print(v, "노드", len(m.graph.node), "→", len(mv.graph.node), "Sub", ops["Sub"], "Where", ops["Where"], "바꾼 곳", len(sites),
              "새 상수", extra["new_init_bytes"], "B", res[v]["checks"], flush=True)
    for tag in ("O", "W1", "W2"):
        res.setdefault(tag, {})["ort_opt"] = ort_ops(MODELS[tag], f"e14_1_{tag}")
        print(tag, "ORT 최적화 후 노드", res[tag]["ort_opt"]["n_nodes"], res[tag]["ort_opt"]["fused"], flush=True)
    for v in ("W1", "W2"):
        res[v]["checks"]["ort_fused_same"] = res[v]["ort_opt"]["fused"] == res["O"]["ort_opt"]["fused"]
    res["pass"] = all(all(res[v]["checks"].values()) for v in ("W1", "W2"))
    dump("g1.json", res)
    print("G1", "통과" if res["pass"] else "불통과", {v: res[v]["checks"] for v in ("W1", "W2")}, flush=True)
    if not res["pass"]:
        sys.exit(2)


def smoke(a):
    pix, _, _ = load_dev("pets_dev", 2)
    for k, p in MODELS.items():
        e = embed(session(p), pix)
        print("smoke", k, e.shape, float(np.linalg.norm(e[0])), flush=True)


# ---------------------------------------------------------------- G2

def diag_model(path, names):
    import onnx
    from onnx import helper, TensorProto
    m = onnx.load(str(path))
    have = {o.name for o in m.graph.output}
    for n in names:
        if n not in have:
            m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    p = WORK / f"diag_{Path(path).stem}.onnx"
    onnx.save(m, str(p))
    return p


def step_g2(a):
    import onnx
    from onnx import numpy_helper as nh
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state()}
    # (a) dev 6×200
    emb = {k: {} for k in MODELS}
    for k, p in MODELS.items():
        s = session(p)
        t = time.time()
        for key in DEV:
            pix, y, W = load_dev(key, 200)
            emb[k][key] = embed(s, pix)
        print(f"G2(a) {k} 임베딩 {time.time() - t:.0f}s", flush=True)
    a_res = {}
    for v in ("W1", "W2"):
        per = {}
        for key in DEV:
            _, y, W = load_dev(key, 200)
            e0, e1 = l2n(emb["O"][key]), l2n(emb[v][key])
            p0, p1 = (e0 @ W).argmax(1), (e1 @ W).argmax(1)
            per[key] = {"cos_min": float((e0 * e1).sum(1).min()), "cos_mean": float((e0 * e1).sum(1).mean()),
                        "agree": 100 * float((p0 == p1).mean()), "n_diff": int((p0 != p1).sum()),
                        "top1_O": 100 * float((p0 == y).mean()), "top1": 100 * float((p1 == y).mean()),
                        "max_abs_raw": float(np.abs(emb["O"][key] - emb[v][key]).max())}
        a_res[v] = {"per_set": per, "cos_min": min(x["cos_min"] for x in per.values()),
                    "n_diff": sum(x["n_diff"] for x in per.values()),
                    "hr_top1_O": float(np.mean([per[k]["top1_O"] for k in HRK])), "hr_top1": float(np.mean([per[k]["top1"] for k in HRK]))}
        print(f"G2(a) {v}: cos 최소 {a_res[v]['cos_min']:.8f}, top-1 다른 장 {a_res[v]['n_diff']}/1200", flush=True)
    res["a"] = a_res
    # (b) W1 vs W2
    b = {key: {"max_abs": float(np.abs(emb["W1"][key] - emb["W2"][key]).max()),
               "bit_equal": bool(np.array_equal(emb["W1"][key] + 0.0, emb["W2"][key] + 0.0))} for key in DEV}
    res["b"] = {"per_set": b, "max_abs": max(x["max_abs"] for x in b.values()), "bit_equal": all(x["bit_equal"] for x in b.values())}
    print("G2(b) W1 vs W2 max|Δ|", res["b"]["max_abs"], flush=True)
    res["c"] = g2c(disable_opt=False)
    res["pass"] = bool(all(a_res[v]["cos_min"] >= 0.99999 for v in ("W1", "W2")) and res["b"]["bit_equal"] and res["c"]["pass"])
    res["g2b_needed"] = any(a_res[v]["n_diff"] > 0 for v in ("W1", "W2"))
    dump("g2.json", res)
    np.savez_compressed(OUT / "g2_emb.npz", **{f"{k}_{key}": emb[k][key] for k in MODELS for key in DEV})
    print("G2", "통과" if res["pass"] else "불통과", "G2b 필요" if res["g2b_needed"] else "G2b 생략", flush=True)
    if not res["pass"]:
        sys.exit(2)


def step_g2c(a):
    """G2(c)만 다시: --disable-opt면 ORT 최적화 끔 (진단, g2c_disable.json)."""
    r = g2c(disable_opt=a.disable_opt)
    dump("g2c_disable.json" if a.disable_opt else "g2c.json", {"date": datetime.now().isoformat(timespec="seconds"),
                                                                 "disable_opt": a.disable_opt, "git": git_state(), **r})


def g2c(disable_opt):
    """2장 중간 출력: 자리마다 출력 = where(m, c, Y) (자기 모델의 Y 기준)."""
    import onnx
    from onnx import numpy_helper as nh
    m0 = onnx.load(str(O))
    init = {i.name: nh.to_array(i) for i in m0.graph.initializer}
    sites = [r for r in RW.find_sites(m0) if r["kind"] in ("simple", "chain")]
    names = sorted({r["out"] for r in sites} | {r["Y"] for r in sites} | {r["X"] for r in sites})
    pix, _, _ = load_dev("pets_dev", 2)
    x = to_x(pix)
    outs = {}
    for k, p in MODELS.items():
        dp = diag_model(p, names)
        s = session(dp, disable_opt=disable_opt)
        on = [o.name for o in s.get_outputs()]
        r = [s.run(None, {s.get_inputs()[0].name: x[i:i + 1]}) for i in range(2)]
        outs[k] = {n: np.concatenate([ri[on.index(n)] for ri in r]) for n in names}
        # 진단 사본의 최종 임베딩이 원래 모델과 같은지 (출력 추가가 계산을 바꾸지 않는지)
        e_diag = np.concatenate([ri[on.index("embedding")] for ri in r])
        e_orig = embed(session(p, disable_opt=disable_opt), pix)
        outs[k]["_diag_same"] = bool(np.array_equal(e_diag, e_orig))
    c_rows = []
    for r in sites:
        mk = np.broadcast_to(init[r["m"]] > 0, outs["O"][r["out"]].shape)
        c = np.broadcast_to(init[r["c"]], outs["O"][r["out"]].shape).astype(np.float32)
        row = {"site": r["sub"], "kind": r["kind"]}
        for k in ("W1", "W2"):
            ref = np.where(mk, c, outs[k][r["Y"]])
            row[f"{k}_eq_select"] = bool(np.array_equal(outs[k][r["out"]] + 0.0, ref + 0.0))
        o, y = outs["O"][r["out"]], outs["O"][r["Y"]]
        row["O_nonsink_eq_Y"] = bool(np.array_equal(o[~mk] + 0.0, y[~mk] + 0.0))
        ulp = np.abs(o[mk].astype(np.float64) - c[mk]) / np.spacing(np.abs(c[mk]).astype(np.float32)).astype(np.float64)
        row["O_sink_max_ulp"] = float(ulp.max()) if ulp.size else 0.0
        big = np.maximum(np.abs(c[mk]), np.abs(np.broadcast_to(outs["O"][r["X"]], o.shape)[mk])).astype(np.float32)
        row["O_sink_max_ulp_of_max_abs_X_c"] = float((np.abs(o[mk].astype(np.float64) - c[mk]) / np.spacing(big).astype(np.float64)).max()) if ulp.size else 0.0
        row["O_sink_frac_ne_c"] = float((o[mk] != c[mk]).mean()) if ulp.size else 0.0
        row["O_vs_W2_nonsink_max_abs"] = float(np.abs(o[~mk] - outs["W2"][r["out"]][~mk]).max())   # 참고: 앞 자리 차이가 퍼진 정도
        c_rows.append(row)
    c = {"disable_opt": disable_opt, "sites": c_rows, "diag_same": {k: outs[k]["_diag_same"] for k in MODELS},
         "pass": all(r["W1_eq_select"] and r["W2_eq_select"] and r["O_nonsink_eq_Y"] and r["O_sink_max_ulp"] <= 1 for r in c_rows)
         and all(outs[k]["_diag_same"] for k in MODELS),
         "pass_ulp_of_max_abs_X_c": all(r["W1_eq_select"] and r["W2_eq_select"] and r["O_nonsink_eq_Y"] and r["O_sink_max_ulp_of_max_abs_X_c"] <= 1
                                        for r in c_rows) and all(outs[k]["_diag_same"] for k in MODELS)}
    print("G2(c)", "disable_opt" if disable_opt else "ENABLE_ALL", "통과" if c["pass"] else "불통과",
          "| 1ulp(max|X|,|c|) 기준", "통과" if c["pass_ulp_of_max_abs_X_c"] else "불통과",
          "| 자리별 불일치", [r["site"] for r in c_rows if not (r["W1_eq_select"] and r["W2_eq_select"] and r["O_nonsink_eq_Y"])], flush=True)
    return c


def boot_hr(diff_by_data, n_boot=1000, seed=0):
    """oslab_g8/siglip2/s2_paired_ci.boot_hr와 같은 계산 (데이터셋별 층화 재표집, 고화질 평균 차이 %p 95% 구간)."""
    rng = np.random.default_rng(seed)
    vals = [np.mean([100 * v[rng.integers(0, len(v), len(v))].mean() for v in diff_by_data.values()]) for _ in range(n_boot)]
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


def step_acc(a):
    pred = {}
    for k, p in MODELS.items():
        s = session(p)
        pred[k] = {}
        for key in DEV:
            pix, y, W = load_dev(key)
            pred[k][key] = (l2n(embed(s, pix)) @ W).argmax(1) == y
        print("acc", k, {key: round(100 * float(v.mean()), 2) for key, v in pred[k].items()}, flush=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state()}
    for k in MODELS:
        res[k] = {key: 100 * float(v.mean()) for key, v in pred[k].items()}
        res[k]["hr"] = float(np.mean([res[k][key] for key in HRK]))
    for v in ("W1", "W2"):
        diff = {key: pred[v][key].astype(float) - pred["O"][key].astype(float) for key in HRK}
        res[v]["hr_delta"] = res[v]["hr"] - res["O"]["hr"]
        res[v]["hr_delta_ci95"] = boot_hr(diff, 1000, 0)
    dump("acc.json", res)


# ---------------------------------------------------------------- mem / speed

def step_mem(a):
    for rep in range(3):
        for k in (["O", "W1", "W2"] if rep % 2 == 0 else ["W2", "W1", "O"]):
            p = subprocess.run([sys.executable, str(HERE / "mem_child.py"), str(MODELS[k]), "C3", "4", "50", "10"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert p.returncode == 0, p.stderr[-2000:]
            r = json.loads(p.stdout.strip().splitlines()[-1])
            append("memory.jsonl", {"date": datetime.now().isoformat(timespec="seconds"), "model": k, "file": MODELS[k].name,
                                    "rep": rep, "combo": "C3", "threads": 4, "n": 50, **r, "git": git_state()})
            print(f"mem rep{rep} {k} peak {r['peak_MiB']} infer {r['infer_wset_max_MiB']} create {r['create_peak_MiB']}", flush=True)


def child(cfg):
    p = subprocess.run([sys.executable, SPEED_CHILD, json.dumps(cfg)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def step_speed(a):
    env, env_info = env_rows()
    env_info.write_env(OUT, env)
    models, threads = ORDERS[a.session % 10]
    order = 0
    for k in models:
        for th in threads:
            t0 = time.time()
            r = child({"mode": "speed", "model": str(MODELS[k]), "tcombo": "A0", "threads": th, "warm": 20, "n": 200, "tail": 0.5})
            row = {"date": datetime.now().isoformat(timespec="seconds"), "session": a.session, "order": order, "model": k,
                   "threads": th, "warm": 20, "n": 200, **r, "wall_s": round(time.time() - t0, 1), "file": MODELS[k].name,
                   "power": env["power"], "git": env["git_oslab_mobile"], **env_info.flat(env)}
            append("speed.jsonl", row)
            print(f"S{a.session} #{order} {k} {th}T p50 {r['p50_ms']} p90 {r['p90_ms']} throttle_off {r['throttle_off']} ({row['wall_s']}s)", flush=True)
            order += 1
    print(f"SESSION {a.session} DONE {datetime.now():%H:%M:%S}", flush=True)


# ---------------------------------------------------------------- summary

def step_summary(a):
    rows = [json.loads(l) for l in (OUT / "speed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    allsess = sorted({r["session"] for r in rows})
    sessions = [x for x in allsess if x < 10 and (x + 10) not in allsess] + [x for x in allsess if x >= 10]
    get = lambda s, k, th, f: next(r[f] for r in rows if r["session"] == s and r["model"] == k and r["threads"] == th)
    g1 = json.loads((OUT / "g1.json").read_text(encoding="utf-8"))
    g2 = json.loads((OUT / "g2.json").read_text(encoding="utf-8"))
    acc = json.loads((OUT / "acc.json").read_text(encoding="utf-8")) if (OUT / "acc.json").exists() else None
    mem = [json.loads(l) for l in (OUT / "memory.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pk = {k: float(np.median([r["peak_MiB"] for r in mem if r["model"] == k])) for k in MODELS}
    res = {"sessions": sessions, "abs": {}, "delta": {}, "peak_MiB": pk, "verdict": {}, "git": git_state()}
    for k in MODELS:
        for th in (1, 2, 4):
            for f in ("p50_ms", "p90_ms"):
                vals = [get(s, k, th, f) for s in sessions]
                res["abs"][f"{k}_{th}T_{f}"] = {"per_session": vals, "median": float(np.median(vals))}
                if k != "O":
                    d = [get(s, k, th, f) / get(s, "O", th, f) - 1 for s in sessions]
                    res["delta"][f"{k}_{th}T_{f}"] = {"per_session": d, "median": float(np.median(d))}
    dm = lambda k, th: res["delta"][f"{k}_{th}T_p50_ms"]["median"]
    for v in ("W1", "W2"):
        c1 = bool(g1["pass"] and g2["pass"])
        c2 = dm(v, 4) <= -0.015
        c3 = dm(v, 1) <= 0.02 and dm(v, 2) <= 0.02
        c4 = (g2["a"][v]["n_diff"] == 0) or (acc is not None and acc[v]["hr_delta_ci95"][0] > -0.3)
        c5 = pk[v] / pk["O"] - 1 <= 0.02
        conds = {"1_gates": c1, "2_4T": c2, "3_1T2T": c3, "4_acc": c4, "5_mem": c5}
        if not (c4 and c5) or not c1:
            vd = "기각" if c1 else "관문 불통과"
        elif c2 and c3:
            vd = "채택 후보"
        elif not c2 and c3:
            vd = "효과 없음"
        else:
            vd = "효과 없음 (1T·2T 조건 불만족)"
        res["verdict"][v] = {"verdict": vd, "conds": conds, "d4T": dm(v, 4), "d2T": dm(v, 2), "d1T": dm(v, 1), "dpeak": pk[v] / pk["O"] - 1}
    cand = [v for v in ("W1", "W2") if res["verdict"][v]["verdict"] == "채택 후보"]
    if len(cand) == 2:
        a4 = {v: res["abs"][f"{v}_4T_p50_ms"]["median"] for v in cand}
        pick = "W2" if abs(a4["W1"] / a4["W2"] - 1) <= 0.01 else min(a4, key=a4.get)
    else:
        pick = cand[0] if cand else None
    res["pick"] = pick
    dump("summary.json", res)
    print(json.dumps({"verdict": res["verdict"], "pick": pick, "peak": pk}, ensure_ascii=False, indent=1))
    for k in MODELS:
        print(k, " / ".join(f"{res['abs'][f'{k}_{th}T_p50_ms']['median']:.2f}" for th in (1, 2, 4)),
              "" if k == "O" else " ".join(f"{th}T {dm(k, th):+.2%}" for th in (1, 2, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["0", "rewrite", "smoke", "g2", "g2c", "acc", "mem", "speed", "summary"])
    ap.add_argument("--session", type=int)
    ap.add_argument("--disable-opt", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"START {' '.join(sys.argv)} git={git_state()}")
    t = time.time()
    {"0": step0, "rewrite": step_rewrite, "smoke": smoke, "g2": step_g2, "g2c": step_g2c, "acc": step_acc, "mem": step_mem,
     "speed": step_speed, "summary": step_summary}[a.step](a)
    log(f"END --step {a.step} {time.time() - t:.0f}s")
    print(f"[{a.step}] {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
