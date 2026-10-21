"""g6e_diag.py — G6e 1단계 (빌드 없음): compiled HAR에서 층별 컨텍스트·경계 거리·정밀도·양자화 값을 뽑는다
compiled HAR 안의 `<net>.hn`(층 그래프·정밀도), `<net>.auto.alls`(컴파일러가 정한 context(...)·place(...)), `<net>.q.npz`(양자화 값)를 읽는다.
  - auto.alls의 이름(conv41_d0, precision_change19_sdc …)은 hn 층 이름으로 되돌린다 (그대로 있으면 그 이름, 아니면 '<hn이름>_'로 시작하는 가장 긴 hn 이름).
    hn에 없는 보조 층(concat_f_from_…, sh_from_…)은 버린다.
  - 경계 거리: hn 그래프(방향 무시)에서 '다른 컨텍스트에 속한 이웃이 있거나 여러 컨텍스트에 걸친 층'까지의 최소 홉 수 (0 = 경계 층)
사용 (siglip2 venv): python g6e_diag.py --har TAG=<compiled.har> ... --focus TAG=층,층 ... --out diag.json
"""
import argparse
import collections
import io
import json
import re
import tarfile

import numpy as np


def load(har):
    t = tarfile.open(har)
    names = t.getnames()
    get = lambda suf: t.extractfile([n for n in names if n.endswith(suf) and not n.endswith(("fp.hn", "native.hn"))][0]).read()
    hn = json.loads(get(".hn"))
    auto = get(".auto.alls").decode()
    alls = get(".alls").decode() if any(n.endswith(".alls") and not n.endswith("auto.alls") for n in names) else ""
    alls = t.extractfile([n for n in names if n.endswith(".alls") and not n.endswith(".auto.alls")][0]).read().decode()
    q = np.load(io.BytesIO(get(".q.npz")), allow_pickle=True)
    return hn, auto, alls, q


def analyze(hn, auto, q):
    L = {k.split("/")[-1]: v for k, v in hn["layers"].items()}
    H = set(L)
    def base(a):
        if a in H:
            return a
        c = [h for h in H if a.startswith(h + "_")]
        return max(c, key=len) if c else None
    ctx = collections.defaultdict(set)          # hn 층 → 컨텍스트 번호들
    place = collections.defaultdict(set)        # hn 층 → (컨텍스트, cluster)
    nctx = 0
    for m in re.finditer(r"_context_(\d+) = context\(\[([^\]]*)\]\)", auto):
        c = int(m.group(1)); nctx = max(nctx, c + 1)
        for a in m.group(2).split(","):
            b = base(a.strip())
            if b:
                ctx[b].add(c)
    for m in re.finditer(r"_context_(\d+)\.place\((\d+), \[([^\]]*)\]\)", auto):
        for a in m.group(3).split(","):
            b = base(a.strip())
            if b:
                place[b].add((int(m.group(1)), int(m.group(2))))
    nb = collections.defaultdict(set)
    for k, v in L.items():
        for x in v["input"]:
            x = x.split("/")[-1]
            nb[k].add(x); nb[x].add(k)
    edge = {k for k in L if len(ctx[k]) > 1 or any(ctx[n] and ctx[n] != ctx[k] for n in nb[k])}
    dist = {k: 0 for k in edge}
    dq = collections.deque(edge)
    while dq:
        u = dq.popleft()
        for n in nb[u]:
            if n not in dist:
                dist[n] = dist[u] + 1; dq.append(n)
    P = [k for k in q.files if "/" in k][0].split("/")[0]
    def qv(n, s):
        k = f"{P}/{n}/{s}:0"
        return q[k] if k in q.files else None
    return L, ctx, place, dist, nb, nctx, qv


def rng(a):
    return None if a is None else [round(float(a.min()), 4), round(float(a.max()), 4)]


def row(L, ctx, place, dist, nb, qv, k):
    v = L[k]
    return {"layer": k, "type": v["type"], "precision": v["quantization_params"].get("precision_mode"),
            "inputs": [(x.split("/")[-1], L[x.split("/")[-1]]["quantization_params"].get("precision_mode"), sorted(ctx[x.split("/")[-1]])) for x in v["input"]],
            "outputs": [(x.split("/")[-1], sorted(ctx[x.split("/")[-1]])) for x in v["output"]],
            "contexts": sorted(ctx[k]), "place": sorted(place[k]), "boundary_hops": dist.get(k),
            "original_names": v.get("original_names"),
            "limvals_in": rng(qv(k, "limvals_in")), "limvals_out": rng(qv(k, "limvals_out")),
            "output_scale": rng(qv(k, "layer_params/output_scales")), "input_scale": rng(qv(k, "layer_params/input_scales")),
            "zp_out": None if qv(k, "zero_point_out") is None else float(qv(k, "zero_point_out")),
            "mult_shift": None if qv(k, "output_stage/mult_shift") is None else int(qv(k, "output_stage/mult_shift")),
            "slope_e": None if qv(k, "output_stage/piecewise/slopes_e") is None else int(qv(k, "output_stage/piecewise/slopes_e").ravel()[0]),
            "bias": rng(qv(k, "bias")), "kernel": rng(qv(k, "kernel"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", nargs="+", required=True, help="TAG=compiled.har")
    ap.add_argument("--focus", nargs="+", default=[], help="TAG=층,층,…")
    ap.add_argument("--hops", type=int, default=2, help="focus 층 주변 몇 홉까지 표에 넣을지")
    ap.add_argument("--qdiff", default="", help="A,B: 두 TAG의 양자화 값이 다른 층 목록")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    focus = {f.split("=")[0]: f.split("=")[1].split(",") for f in a.focus}
    res, keep = {}, {}
    for spec in a.har:
        tag, path = spec.split("=", 1)
        hn, auto, alls, q = load(path)
        L, ctx, place, dist, nb, nctx, qv = analyze(hn, auto, q)
        keep[tag] = (L, ctx, q)
        sel = set(focus.get(tag, []))
        for _ in range(a.hops):
            sel |= {n for s in list(sel) for n in nb[s] if n in L}
        order = list(L)
        rows = [row(L, ctx, place, dist, nb, qv, k) for k in order if k in sel]
        buckets = {m.group(1): [(x.strip().split("/")[-1], sorted(ctx[x.strip().split("/")[-1]])) for x in m.group(2).split(",")]
                   for m in re.finditer(r"(\w+) = bucket\(\[([^\]]*)\]\)", alls)}
        per_ctx = collections.Counter(c for k in L for c in ctx[k])
        res[tag] = {"har": path, "n_layers": len(L), "n_contexts": nctx, "layers_per_context": dict(sorted(per_ctx.items())),
                    "unassigned": sorted(k for k in L if not ctx[k]), "buckets": buckets, "focus_rows": rows}
        print(f"== {tag}: 층 {len(L)}, 컨텍스트 {nctx}")
        for r in rows:
            print(f"  {r['layer']:42s} {r['type']:10s} {r['precision']:12s} ctx={r['contexts']} 경계홉={r['boundary_hops']} "
                  f"in={[(i[0], i[1], i[2]) for i in r['inputs']]} out_rng={r['limvals_out']}")
    if a.qdiff:
        A, B = a.qdiff.split(",")
        (LA, cA, qA), (LB, cB, qB) = keep[A], keep[B]
        def keys(q):
            P = [k for k in q.files if "/" in k][0].split("/")[0]
            return {k[len(P) + 1:]: k for k in q.files if k.startswith(P + "/")}
        kA, kB = keys(qA), keys(qB)
        diff = collections.defaultdict(list)
        for s in sorted(set(kA) | set(kB)):
            ly = s.split("/")[0]
            if s not in kA or s not in kB:
                diff[ly].append(s.split("/", 1)[1] + (" (A에만)" if s in kA else " (B에만)"))
                continue
            x, y = qA[kA[s]], qB[kB[s]]
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                diff[ly].append(s.split("/", 1)[1])
        moved = sorted(k for k in set(LA) & set(LB) if cA[k] != cB[k])
        res["qdiff"] = {"A": A, "B": B, "layers_with_diff": {k: v for k, v in sorted(diff.items())},
                        "only_A": sorted(set(LA) - set(LB)), "only_B": sorted(set(LB) - set(LA)),
                        "context_changed": {k: [sorted(cA[k]), sorted(cB[k])] for k in moved}}
        print(f"== 양자화 값이 다른 층 ({A} 대 {B}): {sorted(diff)}")
        print(f"   컨텍스트가 바뀐 층 {len(moved)}개")
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
