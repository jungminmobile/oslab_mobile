"""run_e14_4.py — E14-4 패치 임베딩 Conv → MatMul 단계별 드라이버 (계획서 results/E14-4_패치임베딩MatMul/E14-4_패치임베딩MatMul.md v3, 7·9절)

--step 0        다른 측정 확인, F2·O sha, G0 ①~⑥, env.json → step0.json
--step rewrite  P1·P2·P2b(+측정용 사본) 생성 + G1 → g1.json
--step smoke    F2·P1·P2·P2b × 2장 (batch 1·2)
--step g2       G2 (a)(b)(c)(d) → g2.json
--step g3 --session k   빠른 속도 (4T, 예열 자식 1회) → g3_speed.jsonl, 2세션 뒤 g3.json
--step acc      G3 통과 모델 dev 전체 + 부트스트랩 (F2 대비) → acc.json
--step mem      O·F2·통과 모델 × 3회 (mem_c3_child.py one, C3, 4T, 50회) → memory.jsonl
--step speed --session k   O·F2·통과 모델 1·2·4T (예열 자식 1회) → speed.jsonl
--step summary  6절 판정 → summary.json
측정·데이터 함수는 run_e14_1.py(R1)를 그대로 쓴다.
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rewrite_mask as RW    # noqa: E402  (qdq_fingerprint, file_sha)
import rewrite_patch as RP   # noqa: E402
import run_e14_1 as R1       # noqa: E402

REPO = HERE.parent
OUT = REPO / "results" / "E14-4_패치임베딩MatMul" / "results"
O, O_SHA, WORK = R1.O, R1.O_SHA, R1.WORK
F2 = WORK / "e14_2_F2_fastgelu_bias.onnx"
F2_SHA8 = "03f4de18"
MODELS = {"O": O, "F2": F2, "P1": WORK / "e14_4_P1_patch_fp32.onnx", "P2": WORK / "e14_4_P2_patch_int8.onnx",
          "P2b": WORK / "e14_4_P2b_patch_int8_qfirst.onnx"}
VARS = ("P1", "P2", "P2b")
DBG = {k: WORK / f"e14_4_{k}_dbg.onnx" for k in ("F2",) + VARS}
DEV, HRK = R1.DEV, R1.HRK
FIRST_Q = "/trunk/blocks.0/norm1/LayerNormalization_output_0_QuantizeLinear_Output"
POS_ADD_OUT = "/trunk/Add_output_0"
MEM_CHILD = str(HERE / "mem_c3_child.py")
git_state = R1.git_state


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def append(name, row):
    with open(OUT / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def alive():
    """G1·G2에서 살아남은 변형 (g1.json·g2.json의 dropped 반영)."""
    v = list(VARS)
    for f in ("g1.json", "g2.json"):
        if (OUT / f).exists():
            v = [x for x in v if x not in load(f).get("dropped", [])]
    return v


def others():
    return [p for p in R1.other_python() if "run_e14_4" not in p["cmd"] and "e12_4_speed" not in p["cmd"] and "mem_c3_child" not in p["cmd"]]


# ---------------------------------------------------------------- 0

def frac_nonint(x):
    v = (np.asarray(x, np.float64) + 1.0) * 127.5
    return float((np.abs(v - np.round(v)) > 1e-3).mean())


def step0(a):
    import onnx
    oth = others()
    print("다른 python (MCP 제외):", oth or "없음", flush=True)
    assert not oth, oth
    sha_f2, sha_o = RW.file_sha(F2), RW.file_sha(O)
    print("F2 sha", sha_f2[:8], "OK" if sha_f2.startswith(F2_SHA8) else "불일치", "| O sha", sha_o[:8], "OK" if sha_o == O_SHA else "불일치", flush=True)
    assert sha_f2.startswith(F2_SHA8) and sha_o == O_SHA
    m = onnx.load(str(F2))
    insp = RP.inspect(m)
    insp_o = RP.inspect(onnx.load(str(O)))
    wf = RP.weight_format(m)
    print("G0 ② F2:", {k: insp[k] for k in ("n_conv", "attrs", "W_shape", "chain", "transpose_perm", "ok")}, flush=True)
    print("G0 ② O 같음:", insp_o["ok"] and insp_o["chain"] == insp["chain"], flush=True)
    print("G0 ③ 위치 임베딩:", insp["pos_name"], "상수" if insp["pos_is_init"] else "상수 아님", insp.get("pos_shape"), "소비 노드", insp["pos_n_consumers"], flush=True)
    print("G0 ④ 가중치 DQ 형식:", wf, flush=True)
    # ⑥ 입력 정수 확인
    six = {}
    nonint = {}
    for key in DEV:
        pix, _, _ = R1.load_dev(key, 200)
        nonint[key] = frac_nonint(R1.to_x(pix))
    six["a_dev_cache"] = {"per_set": nonint, "max": max(nonint.values()), "pix_dtype": str(np.load(R1.ART / "dev256" / "pets_dev_pix.npy", mmap_mode="r").dtype),
                          "path": "C:/s2art_b32/dev256/<set>_pix.npy → run_e14_1.to_x (pix/127.5 − 1)"}
    # (나) 제출 portable_eval 경로: x는 make_xplat_bundle.py가 $SIGLIP2_ARTIFACTS/cache/<set>_x.npy를 그대로 담은 것,
    #      그 캐시는 s2_g6pre.zoo_pre (PIL BICUBIC → uint8 → /127.5 − 1)로 만든다. B/32 256 묶음 파일은 이 노트북에 없으므로
    #      같은 함수를 실제 리사이즈가 일어나도록(원본 256 → 311·197 → 256) 돌려 정수성을 잰다.
    sys.path.insert(0, str(REPO.parent / "oslab_submit" / "siglip2"))
    from PIL import Image
    import importlib
    try:
        g6pre = importlib.import_module("s2_g6pre")
        zp = g6pre.zoo_pre
        src = "oslab_submit/siglip2/s2_g6pre.zoo_pre"
    except Exception as e:     # 모듈 import가 무거우면 같은 식을 그대로 쓴다 (기록)
        def zp(img, size):
            arr = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)
            return (arr.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        src = f"copy of zoo_pre (import 실패: {type(e).__name__}: {str(e)[:120]})"
    pix, _, _ = R1.load_dev("pets_dev", 200)
    xs = []
    for i in range(200):
        im = Image.fromarray(np.asarray(pix[i])).resize((311, 197) if i % 2 else (197, 311), Image.BICUBIC)
        xs.append(zp(im, 256))
    xs = np.stack(xs).astype(np.float32)
    six["b_portable_eval"] = {"frac_nonint": frac_nonint(xs), "x_shape": list(xs.shape), "x_dtype": str(xs.dtype), "func": src,
                              "note": "B/32 256 dev200 묶음(npz)은 이 노트북에 없음 → 묶음을 만드는 전처리 함수를 리사이즈 200장에 적용해 확인"}
    fa, fb = six["a_dev_cache"]["max"], six["b_portable_eval"]["frac_nonint"]
    six["branch"] = "그대로" if fa < 1e-6 and fb < 1e-6 else ("(나)만 0 아님 → 멈춤" if fa < 1e-6 else "둘 다 0 아님 → 기록만")
    print(f"G0 ⑥ frac_nonint (가) dev 캐시 최대 {fa} (나) portable_eval 전처리 {fb} → {six['branch']}", flush=True)
    env, env_info = R1.env_rows()
    env_info.write_env(OUT, env)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "F2_sha256": sha_f2, "O_sha256": sha_o, "inspect_F2": insp,
           "inspect_O_ok": insp_o["ok"], "weight_format": wf, "input_integer": six, "other_python": oth, "git": git_state()}
    res["pass"] = bool(insp["ok"] and insp_o["ok"] and six["branch"] == "그대로"
                       and wf["kinds"] and all(k.startswith("('int8', True, 1, ('int8', True))") for k in wf["kinds"]) and wf["int_min"] >= -64 and wf["int_max"] <= 64)
    dump("step0.json", res)
    print("G0", "통과" if res["pass"] else "불통과", flush=True)
    if not res["pass"]:
        sys.exit(2)


# ---------------------------------------------------------------- rewrite + G1

def ort_ops(path, tag):
    import onnx
    opt = WORK / f"e14_4_{tag}_ortopt.onnx"
    R1.session(path, 4, opt)
    mo = onnx.load(str(opt))
    c = Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in mo.graph.node)
    return {"n_nodes": len(mo.graph.node), "ops": dict(sorted(c.items())), "n_mmitf": c.get("com.microsoft::MatMulIntegerToFloat", 0),
            "fused": dict(sorted({k: v for k, v in c.items() if any(f in k for f in R1.FUSED)}.items())), "file": str(opt), "model": mo}


def crop_transpose_src(mo):
    """최적화 그래프에서 perm (0,2,4,1,3,5) Transpose의 입력이 어디서 오는지 (Reshape는 건너뜀)."""
    import onnx
    prod = {o: n for n in mo.graph.node for o in n.output}
    rows = []
    for n in mo.graph.node:
        if n.op_type == "Transpose":
            perm = [list(onnx.helper.get_attribute_value(x)) for x in n.attribute if x.name == "perm"]
            if perm and perm[0] == [0, 2, 4, 1, 3, 5]:
                t, path = n.input[0], []
                while t in prod and prod[t].op_type == "Reshape":
                    path.append("Reshape")
                    t = prod[t].input[0]
                src = prod[t].op_type if t in prod else "INPUT"
                rows.append({"name": n.name, "src": src, "path": path, "uint8": src in ("QuantizeLinear", "DynamicQuantizeLinear")})
    return rows


def run_shapes(path, names, batch):
    import onnxruntime as ort
    pix, _, _ = R1.load_dev("pets_dev", batch)
    s = R1.session(path, disable_opt=True)
    on = [o.name for o in s.get_outputs()]
    r = s.run(None, {s.get_inputs()[0].name: R1.to_x(pix)})
    return {n: list(r[on.index(n)].shape) for n in names if n in on}


def step_rewrite(a):
    import onnx
    m = onnx.load(str(F2))
    fp0 = RP.qdq_fingerprint_excl(m)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
           "F2": {"qdq": fp0, "n_nodes": len(m.graph.node), "size_MiB": round(F2.stat().st_size / 2 ** 20, 3)}}
    res["O"] = {"n_nodes": len(onnx.load(str(O)).graph.node), "size_MiB": round(O.stat().st_size / 2 ** 20, 3)}
    dbg_f2 = RP.add_outputs(m, [RP.T_OUT, POS_ADD_OUT, FIRST_Q])
    onnx.save(dbg_f2, str(DBG["F2"]))
    sh_f2 = {b: run_shapes(DBG["F2"], [RP.T_OUT, POS_ADD_OUT], b) for b in (1, 2)}
    extras = {}
    for v in VARS:
        mv, extra = RP.rewrite(m, v)
        onnx.save(mv, str(MODELS[v]))
        onnx.save(RP.add_outputs(mv, [RP.MM_OUT, POS_ADD_OUT, FIRST_Q]), str(DBG[v]))
        extras[v] = extra
        np.save(WORK / f"e14_4_{v}_bias_used.npy", extra["bias_used"])
        fp = RP.qdq_fingerprint_excl(mv)
        ops = Counter(n.op_type for n in mv.graph.node)
        sh = {b: run_shapes(DBG[v], [RP.MM_OUT, POS_ADD_OUT], b) for b in (1, 2)}
        res[v] = {"file": str(MODELS[v]), "sha256": RW.file_sha(MODELS[v]), "size_MiB": round(MODELS[v].stat().st_size / 2 ** 20, 3),
                  "n_nodes": len(mv.graph.node), "n_conv": ops["Conv"], "n_matmul": ops["MatMul"], "qdq_excl": fp, "shapes": sh,
                  **{k: x for k, x in extra.items() if k != "bias_used"},
                  "n_qdq_total": {k: ops[k] for k in ("QuantizeLinear", "DequantizeLinear")}}
        res[v]["checks"] = {"1_conv0": ops["Conv"] == 0,
                            "2_qdq_counts_same": fp["counts"] == fp0["counts"],
                            "2_qdq_scale_zp_same": fp["scale_zp_sha256"] == fp0["scale_zp_sha256"],
                            "2_qdq_weight_same": fp["dq_weight_sha256"] == fp0["dq_weight_sha256"],
                            "3_shape_same": all(sh[b][RP.MM_OUT] == sh_f2[b][RP.T_OUT] and sh[b][POS_ADD_OUT] == sh_f2[b][POS_ADD_OUT] for b in (1, 2))}
        print(v, "노드", len(m.graph.node), "→", len(mv.graph.node), "Conv", ops["Conv"], "크기", res[v]["size_MiB"], "MiB",
              "shape", sh[1][POS_ADD_OUT], sh[2][POS_ADD_OUT], res[v]["checks"], flush=True)
    res["F2"]["shapes"] = sh_f2
    for tag in ("O", "F2") + VARS:
        r = ort_ops(MODELS[tag], tag)
        mo = r.pop("model")
        res[tag]["ort_opt"] = r
        if tag in VARS:
            res[tag]["crop_transpose"] = crop_transpose_src(mo)
        print(tag, "ORT 최적화 후 노드", r["n_nodes"], "MatMulIntegerToFloat", r["n_mmitf"], r["fused"], res[tag].get("crop_transpose", ""), flush=True)
    base = res["F2"]["ort_opt"]["n_mmitf"]
    res["P1"]["checks"]["4_session_ok"] = True
    res["P1"]["checks"]["5_mmitf"] = res["P1"]["ort_opt"]["n_mmitf"] == base
    for v in ("P2", "P2b"):
        res[v]["checks"]["4_session_ok"] = True
        res[v]["checks"]["5_mmitf"] = res[v]["ort_opt"]["n_mmitf"] == base + 1
    ct = res["P2b"]["crop_transpose"]
    res["P2b"]["checks"]["5_transpose_uint8"] = bool(ct) and all(x["uint8"] for x in ct)
    dropped = []
    if not all(res["P2b"]["checks"].values()):
        dropped.append("P2b")
    res["dropped"] = dropped
    res["pass_P1"] = all(res["P1"]["checks"].values())
    res["pass_P2"] = all(res["P2"]["checks"].values())
    res["pass"] = bool(res["pass_P1"] and res["pass_P2"])
    dump("g1.json", res)
    print("G1", "통과" if res["pass"] else "불통과", {v: res[v]["checks"] for v in VARS}, "버림", dropped or "없음", flush=True)
    if not res["pass"]:
        sys.exit(2)


def smoke(a):
    for b in (1, 2):
        pix, _, _ = R1.load_dev("pets_dev", b)
        for k in ("F2",) + VARS:
            s = R1.session(MODELS[k])
            e = s.run(None, {s.get_inputs()[0].name: R1.to_x(pix)})[0]
            print("smoke batch", b, k, e.shape, float(np.linalg.norm(e[0])), flush=True)


# ---------------------------------------------------------------- G2

def emb_child(a):
    """G2(b)용 자식: ENABLE_ALL·4T·C3, 2장 최종 임베딩 → npy."""
    pix, _, _ = R1.load_dev("pets_dev", 2)
    e = R1.embed(R1.session(Path(a.model), 4), pix)
    np.save(a.out, e)


def step_g2(a):
    import onnxruntime as ort  # noqa: F401
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state(), "dropped": []}
    vs = alive()
    pix2, _, _ = R1.load_dev("pets_dev", 2)
    x2 = R1.to_x(pix2)
    # (a)(d) DISABLE_ALL, 측정용 사본, 2장 (1장씩)
    outs = {}
    for k in ["F2"] + vs:
        s = R1.session(DBG[k], disable_opt=True)
        on = [o.name for o in s.get_outputs()]
        rr = [s.run(None, {s.get_inputs()[0].name: x2[i:i + 1]}) for i in range(2)]
        outs[k] = {n: np.concatenate([ri[on.index(n)] for ri in rr]) for n in on}
        e_orig = R1.embed(R1.session(MODELS[k], disable_opt=True), pix2)
        outs[k]["_diag_same"] = bool(np.array_equal(outs[k]["embedding"], e_orig))
    y = outs["F2"][RP.T_OUT].astype(np.float64)
    ra = {}
    for v in vs:
        bias = np.load(WORK / f"e14_4_{v}_bias_used.npy").astype(np.float32)
        z = (outs[v][RP.MM_OUT] + bias[None, None, :]).astype(np.float64)
        d = np.abs(z - y)
        yr, zr = y.reshape(-1, 768), z.reshape(-1, 768)
        cos = (yr * zr).sum(1) / np.linalg.norm(yr, axis=1) / np.linalg.norm(zr, axis=1)
        pa, pb = outs["F2"][POS_ADD_OUT].astype(np.float64).reshape(-1, 768), outs[v][POS_ADD_OUT].astype(np.float64).reshape(-1, 768)
        cos_pos = (pa * pb).sum(1) / np.linalg.norm(pa, axis=1) / np.linalg.norm(pb, axis=1)
        qd = float((outs[v][FIRST_Q] != outs["F2"][FIRST_Q]).mean())
        ra[v] = {"rel_maxabs": float(d.max() / np.abs(y).max()), "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                 "pos_add_rel_maxabs": float(np.abs(pa - pb).max() / np.abs(pa).max()), "pos_add_cos_min": float(cos_pos.min()),
                 "first_q_diff_frac": qd, "diag_same": outs[v]["_diag_same"]}
        ok = ra[v]["rel_maxabs"] <= 1e-5 if v == "P1" else ra[v]["cos_min"] >= 0.999
        ra[v]["pass"] = bool(ok and outs[v]["_diag_same"])
        print(f"G2(a) {v}: 패치 출력 max|Δ|/max|y| {ra[v]['rel_maxabs']:.3e} cos 최소 {ra[v]['cos_min']:.6f} | 위치 Add 뒤(참고) "
              f"{ra[v]['pos_add_rel_maxabs']:.3e} cos {ra[v]['pos_add_cos_min']:.6f} | (d) 첫 Q 다른 비율 {qd:.5f} | 사본 같음 {outs[v]['_diag_same']}", flush=True)
    res["a"] = ra
    res["F2_diag_same"] = outs["F2"]["_diag_same"]
    # (b) P2 vs P2b, ENABLE_ALL, 4T, C3, 새 프로세스, 2장
    if "P2b" in vs and "P2" in vs:
        e = {}
        for k in ("P2", "P2b"):
            f = WORK / f"e14_4_g2b_{k}.npy"
            p = subprocess.run([sys.executable, str(Path(__file__)), "--step", "_emb", "--model", str(MODELS[k]), "--out", str(f)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert p.returncode == 0, p.stderr[-2000:]
            e[k] = np.load(f)
        res["b"] = {"bit_equal": bool(np.array_equal(e["P2"], e["P2b"])), "max_abs": float(np.abs(e["P2"] - e["P2b"]).max())}
        print("G2(b) P2 vs P2b 비트 일치", res["b"], flush=True)
        if not res["b"]["bit_equal"]:
            res["dropped"].append("P2b")
    # (c) dev 6×200, ENABLE_ALL, F2 대비
    emb = {}
    for k in ["F2"] + vs:
        s = R1.session(MODELS[k])
        t = time.time()
        emb[k] = {key: R1.embed(s, R1.load_dev(key, 200)[0]) for key in DEV}
        print(f"G2(c) {k} 임베딩 {time.time() - t:.0f}s", flush=True)
    rc = {}
    for v in vs:
        per = {}
        for key in DEV:
            _, yy, W = R1.load_dev(key, 200)
            e0, e1 = R1.l2n(emb["F2"][key]), R1.l2n(emb[v][key])
            p0, p1 = (e0 @ W).argmax(1), (e1 @ W).argmax(1)
            per[key] = {"cos_min": float((e0 * e1).sum(1).min()), "agree": 100 * float((p0 == p1).mean()), "n_diff": int((p0 != p1).sum())}
        rc[v] = {"per_set": per, "cos_min": min(x["cos_min"] for x in per.values()), "n_diff": sum(x["n_diff"] for x in per.values()),
                 "agree": 100 - 100 * sum(x["n_diff"] for x in per.values()) / 1200}
        print(f"G2(c) {v}: 최종 cos 최소 {rc[v]['cos_min']:.6f}, top-1 일치 {rc[v]['agree']:.2f}% (다른 장 {rc[v]['n_diff']}/1200)", flush=True)
    res["c"] = rc
    np.savez_compressed(OUT / "g2_emb.npz", **{f"{k}_{key}": emb[k][key] for k in emb for key in DEV})
    res["pass_P1"] = bool(ra.get("P1", {}).get("pass", False))
    res["pass_P2"] = bool(ra.get("P2", {}).get("pass", False))
    if "P2b" in vs and not ra["P2b"]["pass"]:
        res["dropped"].append("P2b")
    res["pass"] = bool(res["pass_P1"] and res["pass_P2"] and res["F2_diag_same"])
    dump("g2.json", res)
    print("G2", "통과" if res["pass"] else "불통과", "버림", res["dropped"] or "없음", flush=True)
    if not res["pass"]:
        sys.exit(2)


# ---------------------------------------------------------------- speed (G3·3세션 공통)

def speed_child(path, th):
    return R1.child({"mode": "speed", "model": str(path), "tcombo": "A0", "threads": th, "warm": 20, "n": 200, "tail": 0.5})


def run_session(fname, sess, models, threads, kind):
    env, env_info = R1.env_rows()
    env_info.write_env(OUT, env)
    t0 = time.time()
    w = speed_child(MODELS["F2"], 4)
    append(fname, {"date": datetime.now().isoformat(timespec="seconds"), "kind": kind, "session": sess, "order": -1, "model": "F2",
                   "threads": 4, "discard": True, **w, "wall_s": round(time.time() - t0, 1)})
    print(f"{kind} S{sess} 예열(버림) F2 4T p50 {w['p50_ms']}", flush=True)
    order = 0
    for k in models:
        for th in threads:
            t0 = time.time()
            r = speed_child(MODELS[k], th)
            row = {"date": datetime.now().isoformat(timespec="seconds"), "kind": kind, "session": sess, "order": order, "model": k,
                   "threads": th, "warm": 20, "n": 200, "discard": False, **r, "wall_s": round(time.time() - t0, 1), "file": MODELS[k].name,
                   "power": env["power"], "git": env["git_oslab_mobile"], **env_info.flat(env)}
            append(fname, row)
            print(f"{kind} S{sess} #{order} {k} {th}T p50 {r['p50_ms']} p90 {r['p90_ms']} throttle_off {r['throttle_off']} ({row['wall_s']}s)", flush=True)
            order += 1
    print(f"{kind} SESSION {sess} DONE {datetime.now():%H:%M:%S}", flush=True)


def rows_of(fname):
    p = OUT / fname
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def step_g3(a):
    oth = others()
    assert not oth, oth
    vs = alive()
    ms = ["F2"] + vs if a.session == 1 else list(reversed(["F2"] + vs))
    run_session("g3_speed.jsonl", a.session, ms, [4], "G3")
    rows = [r for r in rows_of("g3_speed.jsonl") if not r["discard"]]
    sess = sorted({r["session"] for r in rows})
    if len(sess) >= 2:
        get = lambda s, k: next(r["p50_ms"] for r in rows if r["session"] == s and r["model"] == k)
        res = {"sessions": sess, "delta": {}, "pass_models": [], "stopped": [], "git": git_state()}
        for v in vs:
            d = [get(s, v) / get(s, "F2") - 1 for s in sess]
            res["delta"][v] = d
            (res["stopped"] if all(x > -0.010 for x in d) else res["pass_models"]).append(v)
            print(f"G3 {v}: 4T Δ vs F2 " + " / ".join(f"{x:+.2%}" for x in d), "→", "계속" if v in res["pass_models"] else "효과 없음(멈춤)", flush=True)
        dump("g3.json", res)


def g3_pass():
    return load("g3.json")["pass_models"]


def step_acc(a):
    vs = g3_pass()
    g2 = load("g2.json")
    reuse = "P2b" in vs and "P2" in vs and g2.get("b", {}).get("bit_equal", False)
    run = ["F2"] + [v for v in vs if not (v == "P2b" and reuse)]
    pred = {}
    for k in run:
        s = R1.session(MODELS[k])
        pred[k] = {}
        for key in DEV:
            pix, y, W = R1.load_dev(key)
            pred[k][key] = (R1.l2n(R1.embed(s, pix)) @ W).argmax(1) == y
        print("acc", k, {key: round(100 * float(v.mean()), 2) for key, v in pred[k].items()}, flush=True)
    if reuse:
        pred["P2b"] = pred["P2"]
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state(), "P2b_reused_P2": reuse}
    for k in pred:
        res[k] = {key: 100 * float(v.mean()) for key, v in pred[k].items()}
        res[k]["hr"] = float(np.mean([res[k][key] for key in HRK]))
    for v in vs:
        diff = {key: pred[v][key].astype(float) - pred["F2"][key].astype(float) for key in HRK}
        res[v]["hr_delta"] = res[v]["hr"] - res["F2"]["hr"]
        res[v]["hr_delta_ci95"] = R1.boot_hr(diff, 1000, 0)
        res[v]["cifar_delta"] = res[v]["cifar_A"] - res["F2"]["cifar_A"]
        res[v]["n_changed"] = {key: int((pred[v][key] != pred["F2"][key]).sum()) for key in DEV}
    dump("acc.json", res)
    print(json.dumps({v: {"hr_delta": round(res[v]["hr_delta"], 3), "ci": res[v]["hr_delta_ci95"], "cifar": round(res[v]["cifar_delta"], 2),
                          "changed": res[v]["n_changed"]} for v in vs}, ensure_ascii=False), flush=True)


def step_mem(a):
    oth = others()
    assert not oth, oth
    ms = ["O", "F2"] + g3_pass()
    for rep in range(3):
        for k in (ms if rep % 2 == 0 else list(reversed(ms))):
            p = subprocess.run([sys.executable, MEM_CHILD, str(MODELS[k]), "C3", "4", "50", "10", "one"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert p.returncode == 0, p.stderr[-2000:]
            r = json.loads(p.stdout.strip().splitlines()[-1])
            append("memory.jsonl", {"date": datetime.now().isoformat(timespec="seconds"), "model": k, "file": MODELS[k].name, "rep": rep,
                                    "combo": "C3", "threads": 4, "n": 50, "mode": "one", **r, "git": git_state()})
            print(f"mem rep{rep} {k} peak {r['peak_MiB']} create {r['create_peak_MiB']} infer {r['infer_wset_max_MiB']} priv {r['infer_private_max_MiB']}", flush=True)


def step_speed(a):
    oth = others()
    assert not oth, oth
    L = ["O", "F2"] + g3_pass()
    rot = {1: 0, 2: 1, 3: 2}[a.session]
    ms = L[-rot:] + L[:-rot] if rot else L
    th = {1: [1, 2, 4], 2: [4, 2, 1], 3: [2, 4, 1]}[a.session]
    run_session("speed.jsonl", a.session, ms, th, "speed")


# ---------------------------------------------------------------- summary

def step_summary(a):
    rows = [r for r in rows_of("speed.jsonl") if not r["discard"]]
    sessions = sorted({r["session"] for r in rows})
    get = lambda s, k, th, f: next(r[f] for r in rows if r["session"] == s and r["model"] == k and r["threads"] == th)
    g1, g2, g3 = load("g1.json"), load("g2.json"), load("g3.json")
    vs = g3["pass_models"]
    acc = load("acc.json") if vs else None
    mem = rows_of("memory.jsonl")
    ms = ["O", "F2"] + vs
    med = lambda v: float(np.median(v))
    pk = {k: med([r["peak_MiB"] for r in mem if r["model"] == k]) for k in ms}
    cpk = {k: med([r["create_peak_MiB"] for r in mem if r["model"] == k]) for k in ms}
    res = {"sessions": sessions, "abs": {}, "delta": {}, "peak_MiB": pk, "create_peak_MiB": cpk, "verdict": {}, "git": git_state()}
    for k in ms:
        for th in (1, 2, 4):
            for f in ("p50_ms", "p90_ms"):
                vals = [get(s, k, th, f) for s in sessions]
                res["abs"][f"{k}_{th}T_{f}"] = {"per_session": vals, "median": med(vals)}
                if k != "F2":
                    d = [get(s, k, th, f) / get(s, "F2", th, f) - 1 for s in sessions]
                    res["delta"][f"{k}_{th}T_{f}"] = {"per_session": d, "median": med(d)}
    dm = lambda k, th: res["delta"][f"{k}_{th}T_p50_ms"]["median"]
    for v in VARS:
        if v in g1.get("dropped", []) or v in g2.get("dropped", []):
            res["verdict"][v] = {"verdict": "버림 (P2b 조건 G1⑤/G2(b))"}
            continue
        if v in g3["stopped"]:
            res["verdict"][v] = {"verdict": "효과 없음 (G3에서 멈춤)", "g3_delta": g3["delta"][v]}
            continue
        c1 = bool(g1["pass"] and g2["pass"])
        c2 = dm(v, 4) <= -0.015
        c3 = dm(v, 1) <= 0.02 and dm(v, 2) <= 0.02
        c4 = acc[v]["hr_delta_ci95"][0] > -0.3
        c5 = pk[v] / pk["F2"] - 1 <= 0.02
        if not c1:
            vd = "관문 불통과"
        elif not (c4 and c5):
            vd = "기각"
        elif c2 and c3:
            vd = "채택 후보"
        elif not c2:
            vd = "효과 없음"
        else:
            vd = "효과 없음 (1T·2T 조건 불만족)"
        res["verdict"][v] = {"verdict": vd, "conds": {"1_gates": c1, "2_4T": c2, "3_1T2T": c3, "4_acc": c4, "5_mem": c5},
                             "d4T": dm(v, 4), "d2T": dm(v, 2), "d1T": dm(v, 1), "dpeak": pk[v] / pk["F2"] - 1, "g3_delta": g3["delta"][v]}
    cand = [v for v in VARS if res["verdict"][v]["verdict"] == "채택 후보"]
    pick = None
    if cand:
        a4 = {v: res["abs"][f"{v}_4T_p50_ms"]["median"] for v in cand}
        best = min(a4, key=a4.get)
        pref = [v for v in ("P1", "P2", "P2b") if v in cand and a4[v] / a4[best] - 1 <= 0.01]
        pick = pref[0]
    res["pick"] = pick
    dump("summary.json", res)
    print(json.dumps({"verdict": res["verdict"], "pick": pick, "peak": pk, "create_peak": cpk}, ensure_ascii=False, indent=1))
    for k in ms:
        print(k, " / ".join(f"{res['abs'][f'{k}_{th}T_p50_ms']['median']:.2f}" for th in (1, 2, 4)),
              "" if k == "F2" else " ".join(f"{th}T {dm(k, th):+.2%}" for th in (1, 2, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["0", "rewrite", "smoke", "g2", "g3", "acc", "mem", "speed", "summary", "_emb"])
    ap.add_argument("--session", type=int)
    ap.add_argument("--model")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.step == "_emb":
        return emb_child(a)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"START {' '.join(sys.argv)} git={git_state()}")
    t = time.time()
    {"0": step0, "rewrite": step_rewrite, "smoke": smoke, "g2": step_g2, "g3": step_g3, "acc": step_acc, "mem": step_mem,
     "speed": step_speed, "summary": step_summary}[a.step](a)
    log(f"END --step {a.step} {time.time() - t:.0f}s")
    print(f"[{a.step}] {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
