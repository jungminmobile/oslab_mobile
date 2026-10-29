"""
s2_why_after_qdq.py — G8 H5: 배포 INT8 파일의 실제 눈금·가중치로 층별 출력 SNR (2026-10-22)
==================================================================================================
[계기] E4b(s2_why_after_pct.py)는 dev 64장에서 직접 구한 percentile 눈금으로 활성값만 8비트 흉내를 냈다 (검토 지적:
       배포 모델의 눈금이 아니다, 가중치 양자화·출력 재반올림이 빠졌다).
[방법] (FP32 _pre 그래프, 그 그래프를 양자화한 INT8 QDQ 파일) 짝마다, INT8 파일에서 MatMul 79개 각각의
  - 입력 A: QuantizeLinear의 scale·zero_point (배포 눈금 그대로, U8/S8)
  - 입력 B: 가중치면 INT8 파일의 정수 가중치를 DequantizeLinear 그대로 되돌린 값 (축별 scale 포함), 활성값이면 그쪽 Q 눈금
  - 출력이 다시 QuantizeLinear로 들어가면(출력 재반올림) 그 눈금
  를 읽고, FP32 _pre 그래프로 사진을 돌려 같은 이름의 텐서를 꺼내 MatMul마다 아래를 계산한다 (사진 합, float64 누적).
    snr_act   : A(와 활성값 B)만 배포 눈금으로 양자화, 가중치는 FP32
    snr_w     : 가중치만 배포 정수 가중치로
    snr_both  : 입력 둘 다 배포 그대로
    snr_full  : snr_both + 출력 재반올림 (출력 Q가 있을 때만; 없으면 snr_both와 같음)
  SNR = 10·log10(Σ‖y‖² / Σ‖ŷ − y‖²) (dB). y는 FP32 MatMul 출력.
  ※ 앞 층의 양자화 오차가 쌓이는 효과는 없다(층마다 FP32 입력에서 시작) — 층별 민감도를 보는 분석. 정수 커널 결과가 아니다.
[출력] $RN50_RESULTS/why_after_qdq.jsonl (MatMul마다 한 줄), why_after_qdq_summary.json (짝마다 최솟값·평균·하위 5개)
[사용] RN50_RESULTS=… python s2_why_after_qdq.py --pairs 이름=<FP32 _pre>::<INT8> … (파일 이름은 $RN50_ONNX 기준 또는 절대 경로)
"""
import argparse
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402
from s2_why_after import role, SETS  # noqa: E402

RC = S.RC


def _arr(inits, name):
    from onnx import numpy_helper
    return numpy_helper.to_array(inits[name])


def qdq_plan(int8_path):
    """INT8 QDQ 파일 → {MatMul 노드 이름: {a, b, out}}. a·b는 ('act', 텐서 이름, scale, zp, qmin, qmax) 또는 ('w', 가중치 배열)."""
    import onnx
    m = onnx.load(str(int8_path))
    inits = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def qparams(qnode):
        s = _arr(inits, qnode.input[1]).astype(np.float64)
        z = _arr(inits, qnode.input[2])
        lo, hi = (0, 255) if z.dtype == np.uint8 else (-128, 127)
        return float(s), float(z), lo, hi

    def side(t):
        dq = prod.get(t)
        if dq is None or dq.op_type != "DequantizeLinear":
            return None                                          # 양자화하지 않은 MatMul (예: R0의 FP32로 남긴 fc2)
        src = dq.input[0]
        if src in inits:                                         # 정수 가중치 → 배포 값으로 되돌림
            q = _arr(inits, src).astype(np.float64)
            s = _arr(inits, dq.input[1]).astype(np.float64)
            z = _arr(inits, dq.input[2]).astype(np.float64) if len(dq.input) > 2 and dq.input[2] else np.zeros_like(s)
            axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)
            if s.ndim == 1 and s.size > 1:
                shp = [1] * q.ndim
                shp[axis] = -1
                s, z = s.reshape(shp), z.reshape(shp)
            return ("w", ((q - z) * s).astype(np.float32))
        qn = prod[src]
        assert qn.op_type == "QuantizeLinear", f"{t}: Q 아님"
        return ("act", qn.input[0], *qparams(qn))

    plan = {}
    for n in m.graph.node:
        if n.op_type != "MatMul":
            continue
        a, b = side(n.input[0]), side(n.input[1])
        if a is None or b is None:
            continue
        out = None
        for c in cons.get(n.output[0], []):
            if c.op_type == "QuantizeLinear":
                out = qparams(c)
        plan[n.name] = {"a": a, "b": b, "out": out}
    return plan


def fq(v, s, z, lo, hi):
    return ((np.clip(np.round(v / s) + z, lo, hi) - z) * s).astype(np.float32)


def run(pre_path, plan, x, threads):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    m = onnx.load(str(pre_path))
    inits = {i.name: i for i in m.graph.initializer}
    mm = {n.name: n for n in m.graph.node if n.op_type == "MatMul"}
    use = [k for k in plan if k in mm]
    wfp = {k: numpy_helper.to_array(inits[mm[k].input[1]]).astype(np.float32) for k in use if mm[k].input[1] in inits}
    need = sorted({t for k in use for t in mm[k].input if t not in inits})
    have = {o.name for o in m.graph.output}
    for t in need:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    tmp = Path(tempfile.mkdtemp()) / "with_outputs.onnx"
    onnx.save(m, str(tmp), save_as_external_data=True, location="w.bin")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    acc = {k: {"sig": 0.0, "act": 0.0, "w": 0.0, "both": 0.0, "full": 0.0} for k in use}
    names_ok = {k: True for k in use}
    for i in range(len(x)):
        vals = dict(zip(need, sess.run(need, {"image": x[i:i + 1]})))
        for k in use:
            p, n = plan[k], mm[k]
            A = vals[n.input[0]].astype(np.float32)
            if p["a"][0] != "act" or p["a"][1] != n.input[0]:
                names_ok[k] = False
            Aq = fq(A, *p["a"][2:])
            if p["b"][0] == "w":
                Bf, Bq_w, Bq_a = wfp[k], p["b"][1], wfp[k]
            else:
                Bf = vals[n.input[1]].astype(np.float32)
                Bq_w = Bf
                Bq_a = fq(Bf, *p["b"][2:])
            y = np.matmul(A, Bf).astype(np.float64)
            ya = np.matmul(Aq, Bq_a)
            yw = np.matmul(A, Bq_w)
            yb = np.matmul(Aq, Bq_w if p["b"][0] == "w" else Bq_a)
            yf = fq(yb, *p["out"]) if p["out"] else yb
            a = acc[k]
            a["sig"] += float((y ** 2).sum())
            for key, yy in (("act", ya), ("w", yw), ("both", yb), ("full", yf)):
                a[key] += float(((yy.astype(np.float64) - y) ** 2).sum())
    db = lambda s, e: None if e <= 0 else round(10 * math.log10(s / e), 3)
    rows = []
    for k in use:
        a = acc[k]
        rows.append({"node": k, "role": role(k), "b_kind": plan[k]["b"][0], "out_requant": plan[k]["out"] is not None,
                     "name_match": names_ok[k], "a_scale": plan[k]["a"][2] if plan[k]["a"][0] == "act" else None,
                     **{f"snr_{key}": db(a["sig"], a[key]) for key in ("act", "w", "both", "full")}})
    return rows, len(plan), len(use)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="이름=<FP32 _pre>::<INT8 QDQ>")
    ap.add_argument("--n", type=int, default=32, help="Pets·Caltech dev 앞 n장씩 (E4·E4b와 같은 64장)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    x = np.concatenate([np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n]) for d in SETS]).astype(np.float32)
    summ = {}
    for spec in a.pairs:
        name, files = spec.split("=", 1)
        pre, q = files.split("::")
        plan = qdq_plan(RC.onnx_path(q))
        rows, n_plan, n_use = run(RC.onnx_path(pre), plan, x, a.threads)
        with open(out / "why_after_qdq.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                                    "dirty": env["git"]["dirty"], "pair": name, "pre": Path(pre).name, "int8": Path(q).name,
                                    "n_images": len(x), **r}, ensure_ascii=False) + "\n")
        s = {}
        for key in ("act", "w", "both", "full"):
            v = [r[f"snr_{key}"] for r in rows if r[f"snr_{key}"] is not None]
            s[key] = {"min": round(min(v), 2), "mean": round(float(np.mean(v)), 2),
                      "low5": sorted([(round(r[f'snr_{key}'], 2), r["role"]) for r in rows if r[f"snr_{key}"] is not None])[:5]}
        s["matmul_in_int8"], s["matmul_matched"] = n_plan, n_use
        s["name_mismatch"] = sum(not r["name_match"] for r in rows)
        s["out_requant_n"] = sum(r["out_requant"] for r in rows)
        summ[name] = s
        print(name, f"matched {n_use}/{n_plan}", "requant", s["out_requant_n"],
              {k: (s[k]["min"], s[k]["mean"]) for k in ("act", "w", "both", "full")}, flush=True)
    old = {}
    p = out / "why_after_qdq_summary.json"
    if p.exists():
        old = json.load(open(p, encoding="utf-8"))
    old.update(summ)
    json.dump(old, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
