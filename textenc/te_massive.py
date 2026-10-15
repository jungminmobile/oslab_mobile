"""
te_massive.py — 텍스트 인코더 거대 활성값 점검 (스모크 2단계)
=========================================================

[방법]  이미지 쪽(siglip2/s2_why_massive.py, G1 진단)과 같은 지표를 쓴다.
  FP32 ONNX에서 양자화 대상 층(MatMul·Gemm)의 활성값 입력과 블록 잔차 출력(Add)을 그래프 출력으로 꺼내고,
  보정 문장 앞 64개(tokens_check_s0_64.npy)를 ORT로 흘려 텐서마다:
    ratio = max|x| / 99.9% 분위수|x|, 텐서 전체 min~max 비대칭 8비트에서 0으로 반올림되는 비율(q8_stats, s2_why_massive에서 import)
  (배치, 위치, 채널) 3차원 텐서는 최대값이 나오는 위치·채널이 문장마다 같은지도 센다.
  위치 분류: first(위치 0) / real(문장 토큰) / pad / last(위치 63 = 풀링 위치).

[C 변형 보호 목록 규칙 (결과를 보기 전에 고정)]
  MatMul·Gemm의 활성값 입력 ratio ≥ --protect-ratio(기본 20)인 층은 C에서 FP32로 둔다. 목록을 protect.json으로 저장.

[사용]  python textenc/te_massive.py --out results/textenc/smoke
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_common as T                  # noqa: E402
from s2_why_massive import q8_stats    # noqa: E402  (siglip2/, 이미지 쪽과 같은 8비트 지표)

RC = T.RC


def probe_model(src, dst):
    """양자화 대상 층의 활성값 입력 + 블록 잔차 출력을 그래프 출력으로 추가한 사본. (텐서 이름 → 층 정보) 반환."""
    import onnx
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name for i in g.initializer}
    consts = {o for n in g.node if n.op_type == "Constant" for o in n.output}
    probes = {}
    for n in g.node:
        if n.op_type in ("MatMul", "Gemm"):
            for k, t in enumerate(n.input[:2]):
                if t not in inits and t not in consts:
                    probes.setdefault(t, {"kind": f"{n.op_type}_in{k}", "node": n.name, "nodes": []})["nodes"].append(n.name)
        elif n.op_type == "Add" and re.fullmatch(r".*/resblocks\.\d+/Add_1", n.name):   # 블록 끝 잔차 합
            probes.setdefault(n.output[0], {"kind": "residual_out", "node": n.name, "nodes": []})
    have = {o.name for o in g.output}
    for t in probes:
        if t not in have:
            g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    onnx.save(m, str(dst))
    return probes


def block_of(name):
    return int(name.split("resblocks.")[1].split("/")[0]) if "resblocks." in name else -1


def pos_class(tok_row, p, pad):
    if p == 0:
        return "first"
    if p == len(tok_row) - 1:
        return "last(63)" if tok_row[p] == pad else "last(63,real)"
    return "pad" if tok_row[p] == pad else "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="tokens_check_s0_64.npy")
    ap.add_argument("--bs", type=int, default=8, help="64가 아닌 값이어야 배치·위치 축을 구분할 수 있다")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pad-id", type=int, default=0)
    ap.add_argument("--protect-ratio", type=float, default=20.0)
    a = ap.parse_args()
    assert a.bs != T.CTX
    out = Path(a.out)
    env = T.env_row()
    tk = np.load(T.te_path(a.tokens))
    probe = T.TE_DIR / "siglip2_text_fp32_probe.onnx"
    probes = probe_model(T.te_path(T.FP32), probe)
    sess = RC.session(probe, a.threads)
    names = [o.name for o in sess.get_outputs()]
    acc = {t: [] for t in probes}
    for i in range(0, len(tk), a.bs):
        outs = dict(zip(names, sess.run(None, {"tokens": tk[i:i + a.bs]})))
        for t in probes:
            acc[t].append(outs[t])
    probe.unlink()

    rows = []
    for t, info in probes.items():
        parts = acc[t]
        x = np.concatenate([p.reshape(-1) for p in parts]).astype(np.float32)
        ax = np.abs(x)
        p999 = float(np.percentile(ax, 99.9))
        r = {"tensor": t, "kind": info["kind"], "node": info["node"], "nodes": info["nodes"], "block": block_of(info["node"]),
             "shape_batch0": list(parts[0].shape), "max_abs": round(float(ax.max()), 3), "p99_9": round(p999, 4),
             "ratio": round(float(ax.max()) / max(p999, 1e-12), 1), **{f"q8_{k}": v for k, v in q8_stats(x).items()}}
        # 위치·채널 일관성: (배치, 64, 채널) 또는 (64, 배치, 채널)인 텐서만
        shp = parts[0].shape
        if len(shp) == 3 and T.CTX in shp[:2]:
            pos_ax = shp.index(T.CTX) if shp[0] != shp[1] else 1
            pos_hist, ch_hist, norm_ratio = {}, {}, []
            for bi, p in enumerate(parts):
                v = np.moveaxis(p, pos_ax, 1)                       # (배치, 64, 채널)
                for j in range(v.shape[0]):
                    row_tok = tk[bi * a.bs + j]
                    flat = np.abs(v[j])
                    pp, cc = np.unravel_index(int(flat.argmax()), flat.shape)
                    k = pos_class(row_tok, int(pp), a.pad_id)
                    pos_hist[k] = pos_hist.get(k, 0) + 1
                    ch_hist[int(cc)] = ch_hist.get(int(cc), 0) + 1
                    nrm = np.linalg.norm(v[j], axis=-1)
                    norm_ratio.append(float(nrm.max() / np.median(nrm)))
            top_ch = sorted(ch_hist.items(), key=lambda kv: -kv[1])[:3]
            r.update({"argmax_position": pos_hist, "argmax_channel_top3": top_ch,
                      "token_norm_max_over_median": round(float(np.median(norm_ratio)), 1)})
        rows.append(r)
        T.append_jsonl(out / "massive.jsonl", {**T.base_row(env, vars(a)), "step": "massive", **r})

    protect = sorted({n for r in rows if r["kind"].startswith(("MatMul", "Gemm")) and r["ratio"] >= a.protect_ratio
                      for n in r["nodes"]})
    n_q = len({n for r in rows if r["kind"].startswith(("MatMul", "Gemm")) for n in r["nodes"]})
    (out / "protect.json").write_text(json.dumps({"rule": f"activation input ratio >= {a.protect_ratio}",
                                                  "nodes": protect, "n_quant_nodes": n_q}, indent=1) + "\n")
    print(f"\n{'block':>5s} {'kind':14s} {'ratio':>7s} {'max':>9s} {'q8_zero':>8s}  argmax position / channel  node")
    for r in sorted(rows, key=lambda r: (r["block"] if r["block"] >= 0 else 99, r["node"])):
        print(f"{r['block']:5d} {r['kind']:14s} {r['ratio']:7.1f} {r['max_abs']:9.2f} {r['q8_frac_round_to_zero']:8.3f}  "
              f"{r.get('argmax_position', '')} {r.get('argmax_channel_top3', '')}  {r['node'].split('/text/')[-1]}")
    print(f"\nC 보호 목록 ({len(protect)}/{n_q} 층, ratio ≥ {a.protect_ratio}):", protect)


if __name__ == "__main__":
    main()
