"""run_e14_2.py — E14-2 GELU 융합 (FastGelu) 단계별 드라이버 (계획서 results/E14-2_GELU융합/E14-2_GELU융합.md 7·9절)

--step 0        다른 측정 확인, O sha, GELU 13곳 패턴·상수 검사, env.json → step0.json
--step rewrite  F1·F2 생성 + G1 → g1.json
--step smoke    O·F1·F2 × 2장
--step g2       G2 (a) dev 6×200 O 대비 (ENABLE_ALL) (b) 곳마다 FastGelu vs 같은 입력의 원래 식 (DISABLE_ALL, 2장) (c) Q 정수 다른 비율 → g2.json
--step acc      (G2b) dev 전체 + 층화 짝 부트스트랩 → acc.json
--step mem / --step speed --session k / --step summary   (E14-1과 같은 방법)
측정·데이터 함수는 run_e14_1.py를 그대로 쓴다.
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
import rewrite_gelu as RG    # noqa: E402
import rewrite_mask as RW    # noqa: E402  (qdq_fingerprint, file_sha)
import run_e14_1 as R1       # noqa: E402  (session, load_dev, embed, l2n, boot_hr, child, env_rows, git_state, ort_ops 방식)

REPO = HERE.parent
OUT = REPO / "results" / "E14-2_GELU융합" / "results"
O, O_SHA, WORK = R1.O, R1.O_SHA, R1.WORK
MODELS = {"O": O, "F1": WORK / "e14_2_F1_fastgelu.onnx", "F2": WORK / "e14_2_F2_fastgelu_bias.onnx"}
VARS = ("F1", "F2")
DEV, HRK = R1.DEV, R1.HRK
ORDERS = {1: (["O", "F1", "F2"], [1, 2, 4]), 2: (["F2", "O", "F1"], [4, 2, 1]), 3: (["F1", "F2", "O"], [2, 4, 1])}
FUSED = R1.FUSED
git_state = R1.git_state


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def append(name, row):
    with open(OUT / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------- 0

def step0(a):
    import onnx
    others = [p for p in R1.other_python() if "run_e14_2" not in p["cmd"]]
    print("다른 python (MCP 제외):", others or "없음", flush=True)
    assert not others, others
    sha = RW.file_sha(O)
    print("O sha", sha, "OK" if sha == O_SHA else "불일치", flush=True)
    assert sha == O_SHA
    m = onnx.load(str(O))
    sites = RG.find_sites(m)
    ops = Counter(n.op_type for n in m.graph.node)
    env, env_info = R1.env_rows()
    env_info.write_env(OUT, env)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "sha256": sha, "nodes": len(m.graph.node), "op_counts": dict(ops),
           "n_tanh": ops["Tanh"], "n_ok": sum(r["ok"] for r in sites), "n_bias_ok": sum(r.get("bias_ok", False) for r in sites),
           "sites": sites, "git": git_state()}
    dump("step0.json", res)
    for r in sites:
        print(r["tanh"], "OK" if r["ok"] else "다름: " + r["why"], r.get("x_from"), "→", r.get("y_to"), "bias", r.get("bias_shape"),
              "mm", r.get("mm_from"), flush=True)
    print("Tanh", ops["Tanh"], "패턴 일치", res["n_ok"], "bias 흡수 가능", res["n_bias_ok"], flush=True)
    assert ops["Tanh"] == 13 and res["n_ok"] > 0


# ---------------------------------------------------------------- rewrite + G1

def ort_ops(path, tag):
    import onnx
    opt = WORK / f"{tag}_ortopt.onnx"
    R1.session(path, 4, opt)
    mo = onnx.load(str(opt))
    c = Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in mo.graph.node)
    return {"n_nodes": len(mo.graph.node), "ops": dict(sorted(c.items())),
            "fused": dict(sorted({k: v for k, v in c.items() if any(f in k for f in FUSED)}.items())), "file": str(opt)}


def step_rewrite(a):
    import onnx
    WORK.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(O))
    fp0 = RW.qdq_fingerprint(m)
    n_ok = sum(r["ok"] for r in RG.find_sites(m))
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
           "O": {"qdq": fp0, "n_nodes": len(m.graph.node), "n_tanh": 13, "size_MiB": round(O.stat().st_size / 2 ** 20, 3)}}
    for v in VARS:
        mv, sites, extra = RG.rewrite(m, v)
        onnx.save(mv, str(MODELS[v]))
        fp = RW.qdq_fingerprint(mv)
        outs = [r["y"] for r in sites]
        sh0, sh1 = R1.shapes(m, outs), R1.shapes(mv, outs)
        ops = Counter(n.op_type for n in mv.graph.node)
        res[v] = {"file": str(MODELS[v]), "sha256": RW.file_sha(MODELS[v]), "size_MiB": round(MODELS[v].stat().st_size / 2 ** 20, 3),
                  "n_nodes": len(mv.graph.node), "n_tanh": ops["Tanh"], "n_fastgelu": ops["FastGelu"], "n_rewritten": len(sites),
                  "qdq": fp, **extra, "shapes": sh1}
        res[v]["checks"] = {"tanh_left_ok": ops["Tanh"] == 13 - len(sites) and len(sites) == n_ok if v == "F1" else ops["Tanh"] == 13 - len(sites),
                            "fastgelu_count": ops["FastGelu"] == len(sites),
                            "qdq_counts_same": fp["counts"] == fp0["counts"],
                            "qdq_scale_zp_same": fp["scale_zp_sha256"] == fp0["scale_zp_sha256"],
                            "qdq_weight_same": fp["dq_weight_sha256"] == fp0["dq_weight_sha256"],
                            "shape_same": all(sh0[o] == sh1[o] and sh0[o] is not None for o in outs)}
        print(v, "노드", len(m.graph.node), "→", len(mv.graph.node), "Tanh", ops["Tanh"], "FastGelu", ops["FastGelu"],
              "크기", res[v]["size_MiB"], res[v]["checks"], flush=True)
    for tag in MODELS:
        res[tag]["ort_opt"] = ort_ops(MODELS[tag], f"e14_2_{tag}")
        print(tag, "ORT 최적화 후 노드", res[tag]["ort_opt"]["n_nodes"], res[tag]["ort_opt"]["fused"],
              {k: v for k, v in res[tag]["ort_opt"]["ops"].items() if "Gelu" in k or "Tanh" in k}, flush=True)
    for v in VARS:
        res[v]["checks"]["ort_fused_same"] = res[v]["ort_opt"]["fused"] == res["O"]["ort_opt"]["fused"]
    res["pass"] = all(all(res[v]["checks"].values()) for v in VARS)
    dump("g1.json", res)
    print("G1", "통과" if res["pass"] else "불통과", {v: res[v]["checks"] for v in VARS}, flush=True)
    if not res["pass"]:
        sys.exit(2)


def smoke(a):
    pix, _, _ = R1.load_dev("pets_dev", 2)
    for k, p in MODELS.items():
        e = R1.embed(R1.session(p), pix)
        print("smoke", k, e.shape, float(np.linalg.norm(e[0])), flush=True)


# ---------------------------------------------------------------- G2

def gelu_ref(x, k):
    """O 그래프와 같은 연산 순서의 float32 GELU (tanh 근사)."""
    x = x.astype(np.float32)
    a, k, one, half = np.float32(0.044715), np.float32(k), np.float32(1), np.float32(0.5)
    s = x * x; cu = s * x; ca = cu * a; inn = x + ca; ik = inn * k
    return (x * (np.tanh(ik) + one)) * half


def diag_model(path, names, tag):
    import onnx
    from onnx import helper, TensorProto
    m = onnx.load(str(path))
    prod = {o: n for n in m.graph.node for o in n.output}
    have = {o.name for o in m.graph.output}
    for n in names:
        if n in prod and n not in have:
            et = TensorProto.UINT8 if prod[n].op_type == "QuantizeLinear" else TensorProto.FLOAT
            if prod[n].op_type == "QuantizeLinear":
                zp = [i for i in m.graph.initializer if len(prod[n].input) > 2 and i.name == prod[n].input[2]]
                if zp:
                    et = zp[0].data_type
            m.graph.output.append(helper.make_tensor_value_info(n, et, None))
    p = WORK / f"diag_e14_2_{tag}.onnx"
    onnx.save(m, str(p))
    return p


def step_g2(a):
    import onnx
    from onnx import numpy_helper as nh
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state()}
    # (a)
    emb = {k: {} for k in MODELS}
    for k, p in MODELS.items():
        s = R1.session(p)
        t = time.time()
        for key in DEV:
            pix, _, _ = R1.load_dev(key, 200)
            emb[k][key] = R1.embed(s, pix)
        print(f"G2(a) {k} 임베딩 {time.time() - t:.0f}s", flush=True)
    a_res = {}
    for v in VARS:
        per = {}
        for key in DEV:
            _, y, W = R1.load_dev(key, 200)
            e0, e1 = R1.l2n(emb["O"][key]), R1.l2n(emb[v][key])
            p0, p1 = (e0 @ W).argmax(1), (e1 @ W).argmax(1)
            per[key] = {"cos_min": float((e0 * e1).sum(1).min()), "cos_mean": float((e0 * e1).sum(1).mean()),
                        "agree": 100 * float((p0 == p1).mean()), "n_diff": int((p0 != p1).sum()),
                        "top1_O": 100 * float((p0 == y).mean()), "top1": 100 * float((p1 == y).mean()),
                        "max_abs_raw": float(np.abs(emb["O"][key] - emb[v][key]).max()),
                        "bit_equal": bool(np.array_equal(emb["O"][key], emb[v][key]))}
        a_res[v] = {"per_set": per, "cos_min": min(x["cos_min"] for x in per.values()), "n_diff": sum(x["n_diff"] for x in per.values()),
                    "bit_equal_all": all(x["bit_equal"] for x in per.values()),
                    "hr_top1_O": float(np.mean([per[k]["top1_O"] for k in HRK])), "hr_top1": float(np.mean([per[k]["top1"] for k in HRK]))}
        print(f"G2(a) {v}: cos 최소 {a_res[v]['cos_min']:.8f}, top-1 다른 장 {a_res[v]['n_diff']}/1200, 비트 일치 {a_res[v]['bit_equal_all']}", flush=True)
    res["a"] = a_res
    # (b)(c) DISABLE_ALL, 2장
    m0 = onnx.load(str(O))
    init = {i.name: nh.to_array(i) for i in m0.graph.initializer}
    sites = [r for r in RG.find_sites(m0) if r["ok"]]
    cons = {}
    for n in m0.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    qout = {r["y"]: [c.output[0] for c in cons[r["y"]] if c.op_type == "QuantizeLinear"] for r in sites}
    kval = {}
    prod = {o: n for n in m0.graph.node for o in n.output}
    for r in sites:
        ik = prod[[n for n in m0.graph.node if n.name == r["tanh"]][0].input[0]]
        kval[r["y"]] = float([init[i] for i in ik.input if i in init][0].reshape(-1)[0])
    pix, _, _ = R1.load_dev("pets_dev", 2)
    x_in = R1.to_x(pix)
    outs = {}
    for k, p in MODELS.items():
        names = sorted({r["y"] for r in sites} | {r["x"] for r in sites} | {r["mm"] for r in sites if "mm" in r}
                       | {q for v in qout.values() for q in v})
        dp = diag_model(p, names, k)
        s = R1.session(dp, disable_opt=True)
        on = [o.name for o in s.get_outputs()]
        rr = [s.run(None, {s.get_inputs()[0].name: x_in[i:i + 1]}) for i in range(2)]
        outs[k] = {n: np.concatenate([ri[on.index(n)] for ri in rr]) for n in on}
        e_orig = R1.embed(R1.session(p, disable_opt=True), pix)
        outs[k]["_diag_same"] = bool(np.array_equal(outs[k]["embedding"], e_orig))
    rows = []
    for i, r in enumerate(sites):
        row = {"site": r["tanh"], "order": i}
        yO = outs["O"][r["y"]]
        refO = gelu_ref(outs["O"][r["x"]], kval[r["y"]])
        row["O_vs_numpy_rel"] = float(np.abs(yO - refO).max() / np.abs(refO).max())
        for v in VARS:
            yF = outs[v][r["y"]]
            xF = outs[v][r["x"]] if r["x"] in outs[v] else (outs[v][r["mm"]] + init[r["bias"]].astype(np.float32))
            ref = gelu_ref(xF, kval[r["y"]])
            d = np.abs(yF - ref)
            row[f"{v}_rel_maxabs"] = float(d.max() / np.abs(ref).max())          # 판정 값 (자기 입력 기준)
            row[f"{v}_elem_rel_max"] = float((d / np.maximum(np.abs(ref), 1e-30)).max())   # 참고
            row[f"{v}_bit_equal_frac"] = float((yF == ref).mean())
            row[f"{v}_vs_O_rel_maxabs"] = float(np.abs(yF - yO).max() / np.abs(yO).max())   # 참고: O와 직접 (앞 곳 영향 섞임)
            q = qout[r["y"]]
            row[f"{v}_q_diff_frac"] = float(np.mean([(outs[v][qq] != outs["O"][qq]).mean() for qq in q])) if q else None
        rows.append(row)
    res["bc"] = {"sites": rows, "diag_same": {k: outs[k]["_diag_same"] for k in MODELS},
                 "max_rel": {v: max(r[f"{v}_rel_maxabs"] for r in rows) for v in VARS},
                 "q_diff_frac_max": {v: max(r[f"{v}_q_diff_frac"] or 0 for r in rows) for v in VARS},
                 "q_diff_frac_first": {v: rows[0][f"{v}_q_diff_frac"] for v in VARS}}
    for v in VARS:
        print(f"G2(b) {v}: 곳별 max|Δ|/max|y| 최대 {res['bc']['max_rel'][v]:.3e}  (c) Q 정수 다른 비율 첫 곳 {res['bc']['q_diff_frac_first'][v]}"
              f" / 최대 {res['bc']['q_diff_frac_max'][v]}", flush=True)
    res["pass_a"] = all(a_res[v]["cos_min"] >= 0.9999 for v in VARS)
    res["pass_b"] = all(res["bc"]["max_rel"][v] <= 1e-5 for v in VARS) and all(res["bc"]["diag_same"].values())
    res["pass"] = bool(res["pass_a"] and res["pass_b"])
    res["g2b_needed"] = any(a_res[v]["n_diff"] > 0 for v in VARS)
    dump("g2.json", res)
    np.savez_compressed(OUT / "g2_emb.npz", **{f"{k}_{key}": emb[k][key] for k in MODELS for key in DEV})
    print("G2", "통과" if res["pass"] else "불통과", "(a)", res["pass_a"], "(b)", res["pass_b"],
          "G2b 필요" if res["g2b_needed"] else "G2b 생략", flush=True)
    if not res["pass"]:
        sys.exit(2)


def step_acc(a):
    pred = {}
    for k, p in MODELS.items():
        s = R1.session(p)
        pred[k] = {}
        for key in DEV:
            pix, y, W = R1.load_dev(key)
            pred[k][key] = (R1.l2n(R1.embed(s, pix)) @ W).argmax(1) == y
        print("acc", k, {key: round(100 * float(v.mean()), 2) for key, v in pred[k].items()}, flush=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "git": git_state()}
    for k in MODELS:
        res[k] = {key: 100 * float(v.mean()) for key, v in pred[k].items()}
        res[k]["hr"] = float(np.mean([res[k][key] for key in HRK]))
    for v in VARS:
        diff = {key: pred[v][key].astype(float) - pred["O"][key].astype(float) for key in HRK}
        res[v]["hr_delta"] = res[v]["hr"] - res["O"]["hr"]
        res[v]["hr_delta_ci95"] = R1.boot_hr(diff, 1000, 0)
        res[v]["n_changed"] = {key: int((pred[v][key] != pred["O"][key]).sum()) for key in DEV}
    dump("acc.json", res)
    print(json.dumps({v: {"hr_delta": res[v]["hr_delta"], "ci": res[v]["hr_delta_ci95"], "changed": res[v]["n_changed"]} for v in VARS},
                     ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- mem / speed / summary

def step_mem(a):
    for rep in range(3):
        for k in (["O", "F1", "F2"] if rep % 2 == 0 else ["F2", "F1", "O"]):
            p = subprocess.run([sys.executable, str(HERE / "mem_child.py"), str(MODELS[k]), "C3", "4", "50", "10"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert p.returncode == 0, p.stderr[-2000:]
            r = json.loads(p.stdout.strip().splitlines()[-1])
            append("memory.jsonl", {"date": datetime.now().isoformat(timespec="seconds"), "model": k, "file": MODELS[k].name,
                                    "rep": rep, "combo": "C3", "threads": 4, "n": 50, **r, "git": git_state()})
            print(f"mem rep{rep} {k} peak {r['peak_MiB']} infer {r['infer_wset_max_MiB']} create {r['create_peak_MiB']}", flush=True)


def step_speed(a):
    env, env_info = R1.env_rows()
    env_info.write_env(OUT, env)
    models, threads = ORDERS[a.session % 10]
    order = 0
    for k in models:
        for th in threads:
            t0 = time.time()
            r = R1.child({"mode": "speed", "model": str(MODELS[k]), "tcombo": "A0", "threads": th, "warm": 20, "n": 200, "tail": 0.5})
            row = {"date": datetime.now().isoformat(timespec="seconds"), "session": a.session, "order": order, "model": k,
                   "threads": th, "warm": 20, "n": 200, **r, "wall_s": round(time.time() - t0, 1), "file": MODELS[k].name,
                   "power": env["power"], "git": env["git_oslab_mobile"], **env_info.flat(env)}
            append("speed.jsonl", row)
            print(f"S{a.session} #{order} {k} {th}T p50 {r['p50_ms']} p90 {r['p90_ms']} throttle_off {r['throttle_off']} ({row['wall_s']}s)", flush=True)
            order += 1
    print(f"SESSION {a.session} DONE {datetime.now():%H:%M:%S}", flush=True)


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
    for v in VARS:
        c1 = bool(g1["pass"] and g2["pass"])
        c2 = dm(v, 4) <= -0.02
        c3 = dm(v, 1) <= 0.02 and dm(v, 2) <= 0.02
        c4 = (g2["a"][v]["n_diff"] == 0) or (acc is not None and acc[v]["hr_delta_ci95"][0] > -0.3)
        c5 = pk[v] / pk["O"] - 1 <= 0.02
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
                             "d4T": dm(v, 4), "d2T": dm(v, 2), "d1T": dm(v, 1), "dpeak": pk[v] / pk["O"] - 1}
    cand = [v for v in VARS if res["verdict"][v]["verdict"] == "채택 후보"]
    if len(cand) == 2:
        a4 = {v: res["abs"][f"{v}_4T_p50_ms"]["median"] for v in cand}
        pick = "F2" if abs(a4["F1"] / a4["F2"] - 1) <= 0.01 else min(a4, key=a4.get)
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
    ap.add_argument("--step", required=True, choices=["0", "rewrite", "smoke", "g2", "acc", "mem", "speed", "summary"])
    ap.add_argument("--session", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"START {' '.join(sys.argv)} git={git_state()}")
    t = time.time()
    {"0": step0, "rewrite": step_rewrite, "smoke": smoke, "g2": step_g2, "acc": step_acc, "mem": step_mem,
     "speed": step_speed, "summary": step_summary}[a.step](a)
    log(f"END --step {a.step} {time.time() - t:.0f}s")
    print(f"[{a.step}] {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
