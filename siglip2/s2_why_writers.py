"""
s2_why_writers.py — 거대 활성값은 어느 층이 "써 넣고", 그것을 망가뜨리면 정말 결과가 무너지는가 (G3 원인 분석 2단계)
=====================================================================================================

[배경] s2_why_massive.py 결과 (results/siglip2/G3_why/why_massive.jsonl):
  - 블록 5부터 잔차 흐름의 고정 채널 67·301·145(최대 ≈250), 블록 9부터 채널 483(≈600~716)이 생긴다.
  - 이 값은 이미지마다 같은 위치의 토큰 1~2개(블록 5~8: 패치 55(+블록 5~7 패치 1), 블록 9~11: 패치 55·13)에 몰리고,
    블록 6~11에서는 다른 토큰들이 attention의 27~52%를 그 토큰들에 몰아준다 (균등이면 2/196 ≈ 1%).
    (위치·attention 값은 v2 실행 기준으로 정정, 2026-10-14)
  - G3: fc2를 블록 5·9 두 개만 FP32로 두면 충분하고, 둘 중 하나라도 INT8이면 크게 떨어진다 (fc2L2 vs fc2L1).

[가설] (실행 전에 정함)
  H-write : 블록 5와 9의 MLP(fc2 출력)가 그 거대 채널을 high-norm 토큰에 써 넣는다 (attention 쪽이 아니라).
            → 블록 5·9 fc2만 민감한 이유: 그 층의 입력 이상치(블록 5 채널 1449, 블록 9 채널 1038 ≈ 1766)가
              곧 "거대 값을 만드는 신호"이므로, 텐서별 8비트로 잘리거나 뭉개지면 거대 값 자체가 만들어지지 않는다.
  H-need  : 거대 값과 그 토큰에 몰리는 attention은 모델 기능에 필요하다 (attention sink / register).
            → 거대 값을 지우거나 그 토큰으로 가는 attention을 막으면 최종 임베딩이 크게 바뀐다.

[측정] PyTorch FP32(unfused attention), 고화질 dev 사진 N장(Pets·Caltech 앞쪽, why_massive와 같은 사진)
  A. 쓰는 층: 블록마다 잔차 변화 = attention 가지(norm2 입력 − 블록 입력) + MLP 가지(블록 출력 − norm2 입력).
     high-norm 토큰(블록 출력 노름 > 5×중앙값)에서 채널 67·301·145·483에 각 가지가 더한 값의 평균,
     그리고 fc2 입력의 최대 채널 값이 high-norm 토큰에 있는지(최대값 위치 비율).
  B. 개입(인과 확인) — 최종 이미지 임베딩의 FP32 대비 코사인(이미지 평균·최소), 마지막 블록 high-norm 토큰 유무:
     (범위는 보정처럼 이 사진들 전체에서 한 번 정한 정적 범위. minmax = 전체 min~max, pct = 아래 정의)
     fq_in_fc2_b   : 블록 b의 fc2 입력만 텐서별 비대칭 8비트(minmax)
     fq_io_fc2_b   : 블록 b의 fc2를 ORT QDQ처럼 — 입력 8비트(minmax) + 가중치 채널별 대칭 S8 + MatMul 출력(bias 전) 8비트(minmax)
     pq_io_fc2_b   : fq_io와 같지만 범위가 pct (ORT Percentile 99.999 비대칭과 같은 정의: 양쪽 각 0.0005%, 64장에서 정확히)
     clip_in_fc2_b : 블록 b의 fc2 입력을 pct 범위로 자르기만 (반올림 없음 → 자르기 효과만 분리)
     ort_io_fc2_b / ortclip_in_fc2_b : 범위를 ORT 실제 정수 모델(--ort-model)의 scale·zero point에서 구함 (보정 256장, train)
     fq_{in,out}_fc2_{5,9}_{all,hi_only,normal_only} : 입력 또는 MatMul 출력(bias 전)을 같은 minmax 8비트로,
                     모든 토큰 / high-norm 토큰에만 / 보통 토큰에만 (어느 쪽이 망가뜨리나)
     (ORT 그래프 확인: fc2 MatMul 출력 뒤에 QuantizeLinear가 있다 — only_fc2 모델)
     zero_massive_after5 / after9 : 블록 5(또는 9) 출력에서 high-norm 토큰의 거대 채널을 0으로
     mask_attn_hi_6to11 : 블록 6~11에서 high-norm 토큰으로 가는 attention을 0으로 하고 다시 정규화
  ※ B의 가짜 양자화는 "기전 확인용 디버깅"이다 (CLAUDE.md 3절, D-008). 정확도·선택에는 쓰지 않는다.

[사용]
  RN50_RESULTS=../results/siglip2/G3_why "$SIGLIP2_PYTHON" s2_why_writers.py --n 32
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S                       # noqa: E402
from s2_export_onnx import unfuse_attention  # noqa: E402
from scripts import env_info               # noqa: E402

RC = S.RC
CH = [67, 301, 145, 483]                   # why_massive.jsonl에서 찾은 거대 채널
BS = 16


def fq8r(t, lo, hi):
    """정해진 범위 [lo, hi]의 비대칭 8비트 가짜 양자화 (범위는 0을 포함하게 넓히고, zero point 반올림 — ORT와 같은 방식)."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    s = (hi - lo) / 255.0
    if s == 0:
        return t
    zp = round(-lo / s)
    q = (t / s).round().add(zp).clamp(0, 255)
    return (q - zp) * s


def ort_ranges(path):
    """ORT QDQ 모델에서 블록별 fc2 입력(GELU 출력 Mul_1)·MatMul 출력의 8비트 표현 범위 [(qmin−zp)·s, (qmax−zp)·s]."""
    import onnx
    from onnx import numpy_helper as nh
    m = onnx.load(str(path), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type != "QuantizeLinear":
            continue
        for b in range(12):
            for which, name in (("in", f"/trunk/blocks/blocks.{b}/mlp/act/Mul_1_output_0"), ("out", f"/trunk/blocks/blocks.{b}/mlp/fc2/MatMul_output_0")):
                if n.input[0] == name:
                    s = float(nh.to_array(init[n.input[1]]))
                    zp_arr = nh.to_array(init[n.input[2]])
                    zp = int(zp_arr)
                    qmin, qmax = (-128, 127) if zp_arr.dtype == np.int8 else (0, 255)
                    out[(b, which)] = ((qmin - zp) * s, (qmax - zp) * s)
    assert len(out) == 24, f"fc2 입력·출력 24개를 찾아야 함: {len(out)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="데이터셋당 사진 수 (Pets·Caltech dev 앞쪽)")
    ap.add_argument("--ort-model", default="", help="ORT 정수 모델(onnx 폴더 기준 이름). 주면 그 모델의 fc2 입력·출력 범위로도 개입한다")
    a = ap.parse_args()
    import torch
    ort_rng = ort_ranges(RC.onnx_path(a.ort_model)) if a.ort_model else {}
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _, _ = S.load_model()
    unfuse_attention(model)
    vis, trunk = model.visual, model.visual.trunk
    blocks = trunk.blocks
    for b in blocks:                                               # 잔차 분해가 맞으려면 ls·drop_path가 항등이어야 한다
        assert all(isinstance(m, torch.nn.Identity) for m in (b.ls1, b.ls2, b.drop_path1, b.drop_path2)), b
    xs = np.concatenate([np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n] for d in ("pets_dev", "caltech_dev")])
    x = torch.as_tensor(np.ascontiguousarray(xs))
    N = len(x)
    t0 = time.time()

    # ---------- 기준(FP32) 실행: 블록 입력·attention 뒤·블록 출력, fc2 입력, 최종 임베딩 ----------
    cap = {}

    def keep(name, which):
        def f(_, inp, outp):
            cap.setdefault(name, []).append((inp[0] if which == "in" else outp).detach().float())
        return f
    hs = []
    for i, b in enumerate(blocks):
        hs += [b.register_forward_hook(keep(f"in{i}", "in")), b.norm2.register_forward_hook(keep(f"mid{i}", "in")),
               b.register_forward_hook(keep(f"out{i}", "out")), b.mlp.fc2.register_forward_hook(keep(f"g{i}", "in")),
               b.mlp.fc2.register_forward_hook(lambda m, inp, o, i=i: cap.setdefault(f"y{i}", []).append((o - m.bias).detach().float()))]

    def run():
        with torch.inference_mode():
            e = torch.cat([vis(x[s:s + BS]) for s in range(0, N, BS)])
        return torch.nn.functional.normalize(e.float(), dim=-1)
    ref = run()
    for h in hs:
        h.remove()
    T = {k: torch.cat(v) for k, v in cap.items()}
    hi = {}
    for i in range(len(blocks)):
        nrm = T[f"out{i}"].norm(dim=-1)                            # (N, 196)
        hi[i] = nrm > 5 * nrm.median(dim=1, keepdim=True).values
    assert T["out0"].shape == (N, S.NTOK, 768) and T["g0"].shape == (N, S.NTOK, 3072)

    # ---------- A. 거대 채널을 쓰는 가지 ----------
    rows_a = []
    for i in range(len(blocks)):
        m = hi[i]
        da = T[f"mid{i}"] - T[f"in{i}"]                            # attention 가지가 더한 값
        dm = T[f"out{i}"] - T[f"mid{i}"]                           # MLP 가지가 더한 값
        assert torch.allclose(T[f"in{i}"] + da + dm, T[f"out{i}"], atol=1e-3)
        r = {"block": i, "n_hi_tokens": int(m.sum())}
        if m.any():
            # high-norm 토큰 위치를 모두 센다 (위치별 개수 상위 4개) + 위치별 거대 채널 값 (그 위치가 high-norm인 이미지 평균)
            cnt = m.sum(0)
            top = [int(p) for p in torch.argsort(cnt, descending=True)[:4] if cnt[p] > 0]
            r["hi_positions"] = [[p, int(cnt[p])] for p in top]
            r["per_position"] = {str(p): {f"ch{c}": {"in": round(float(T[f"in{i}"][m[:, p], p, c].mean()), 1),
                                                    "mlp_add": round(float(dm[m[:, p], p, c].mean()), 1),
                                                    "out": round(float(T[f"out{i}"][m[:, p], p, c].mean()), 1)} for c in CH}
                                 for p in top[:2]}
            for c in CH:
                r[f"ch{c}"] = {"in": round(float(T[f"in{i}"][..., c][m].mean()), 2), "attn_add": round(float(da[..., c][m].mean()), 2),
                               "mlp_add": round(float(dm[..., c][m].mean()), 2), "out": round(float(T[f"out{i}"][..., c][m].mean()), 2),
                               "out_other_tokens_absmean": round(float(T[f"out{i}"][..., c][~m].abs().mean()), 3)}
            top_mlp = dm[m].abs().mean(0).topk(3)                  # high-norm 토큰에서 MLP가 가장 크게 바꾼 채널
            r["mlp_add_top_channels_at_hi"] = [[int(c), round(float(v), 1)] for v, c in zip(top_mlp.values, top_mlp.indices)]
        g = T[f"g{i}"]
        gc = int(g.abs().amax(dim=(0, 1)).argmax())                # fc2 입력 최대 채널
        pos = g[..., gc].abs().argmax(dim=1)                       # 이미지마다 그 채널 최대값의 토큰 위치
        r["fc2_in_top_channel"] = gc
        r["fc2_in_top_channel_max"] = round(float(g[..., gc].abs().max()), 1)
        r["fc2_in_top_channel_max_at_hi_token_frac"] = round(float(np.mean([bool(hi[i][n, pos[n]]) for n in range(N)])), 3)
        rows_a.append(r)
        print(f"A block {i:2d} hi {r['n_hi_tokens']:3d} | " + " ".join(
            f"ch{c}: in {r[f'ch{c}']['in']:7.1f} +attn {r[f'ch{c}']['attn_add']:7.1f} +mlp {r[f'ch{c}']['mlp_add']:7.1f}" for c in CH[:2] + CH[3:]
            if f"ch{c}" in r) + f" | fc2in ch{gc} max {r['fc2_in_top_channel_max']} at-hi {r['fc2_in_top_channel_max_at_hi_token_frac']}", flush=True)

    # ---------- B. 개입 ----------
    hi_final = hi[len(blocks) - 1]

    def measure(tag):
        cap.clear()
        hk = blocks[-1].register_forward_hook(keep("last", "out"))
        e = run()
        hk.remove()
        last = torch.cat(cap["last"]).norm(dim=-1)
        ratio = (last.max(1).values / last.median(1).values)
        cos = (e * ref).sum(-1)
        r = {"tag": tag, "cos_mean": round(float(cos.mean()), 4), "cos_min": round(float(cos.min()), 4),
             "last_block_max_over_median_norm": round(float(ratio.median()), 1)}
        print(f"B {tag:24s} cos {r['cos_mean']:.4f} (min {r['cos_min']:.4f}) last max/med-norm {r['last_block_max_over_median_norm']}", flush=True)
        return r
    rows_b = [measure("fp32_again")]
    assert rows_b[0]["cos_mean"] > 0.9999, rows_b[0]               # 같은 입력이면 같은 출력 (결정성)
    # 정적 범위(보정처럼 사진 전체에서 한 번 정함): minmax = 전체 min~max,
    # pct = ORT Percentile 99.999(비대칭)과 같은 정의 — 양쪽에서 각각 (100−99.999)/200 = 0.0005%를 자른 값을 정확히(전체 원소 topk) 구한다.
    # (첫 실행은 1/7 간격 표본으로 분위수를 구했는데, 196 = 28×7이라 모든 사진에서 같은 (토큰, 채널) 위치만 뽑혀
    #  거대 값 위치(패치 55·13의 채널 67·301·483 등)가 통째로 빠졌다 — 독립 검토 지적. why_writers.jsonl 1번째 줄의 pq/clip 행은 무효)
    # ort = ORT 실제 정수 모델(--ort-model, 같은 보정으로 만든 MatMul 전부 INT8 Percentile 모델)의 scale·zero point에서 거꾸로 구한 범위
    def ranges(t, b, which):
        f = t.flatten()
        k = max(1, int(np.ceil(5e-6 * f.numel())))
        r = {"minmax": (float(f.min()), float(f.max())),
             "pct": (float(-(-f).topk(k).values[-1]), float(f.topk(k).values[-1]))}
        if ort_rng:
            r["ort"] = ort_rng[(b, which)]
        return r
    kinds = [("fq_in", "minmax", "in"), ("fq_io", "minmax", "io"), ("pq_io", "pct", "io"), ("clip_in", "pct", "clip")]
    if ort_rng:
        kinds += [("ort_io", "ort", "io"), ("ortclip_in", "ort", "clip")]
    for i, b in enumerate(blocks):
        fc2 = b.mlp.fc2
        rin = ranges(T[f"g{i}"], i, "in")
        rout = ranges(T[f"y{i}"], i, "out")
        w0 = fc2.weight.data.clone()
        wq = (w0 / (w0.abs().amax(1, keepdim=True) / 127)).round().clamp(-127, 127) * (w0.abs().amax(1, keepdim=True) / 127)
        for tag, kind, io in kinds:
            lo, up = rin[kind]
            if io == "clip":
                h1 = fc2.register_forward_pre_hook(lambda _, inp, lo=lo, up=up: (inp[0].clamp(lo, up),))
            else:
                h1 = fc2.register_forward_pre_hook(lambda _, inp, lo=lo, up=up: (fq8r(inp[0], lo, up),))
            h2 = None
            if io == "io":                                         # ORT QDQ와 같이: 가중치 채널별 S8 + MatMul 출력(bias 전)도 8비트
                fc2.weight.data.copy_(wq)
                olo, oup = rout[kind]
                h2 = fc2.register_forward_hook(lambda m, inp, o, olo=olo, oup=oup: fq8r(o - m.bias, olo, oup) + m.bias)
            r = measure(f"{tag}_fc2_{i}")
            r["in_range"] = [round(lo, 3), round(up, 3)]
            if io == "io":
                r["out_range"] = [round(olo, 3), round(oup, 3)]
            rows_b.append(r)
            h1.remove()
            if h2 is not None:
                h2.remove()
            fc2.weight.data.copy_(w0)

    # 어느 토큰에서 망가지나: 블록 5·9 fc2의 입력(또는 MatMul 출력, bias 전)을 같은 minmax 범위로 8비트 하되,
    # 모든 토큰 / high-norm 토큰만 / 보통 토큰만
    for k in (5, 9):
        fc2 = blocks[k].mlp.fc2
        for where in ("in", "out"):
            t_all = T[f"g{k}"] if where == "in" else T[f"y{k}"]
            lo, up = float(t_all.min()), float(t_all.max())
            for part in ("all", "hi_only", "normal_only"):
                if where == "in" and part == "all":
                    continue                                       # = fq_in_fc2_k (위에서 잼)
                state = {"s": 0}

                def pick(t, q, part=part, state=state, k=k):
                    s = state["s"]
                    m = hi[k][s:s + len(t)][..., None]             # (B, 196, 1)
                    state["s"] = s + len(t)
                    return q if part == "all" else (torch.where(m, q, t) if part == "hi_only" else torch.where(m, t, q))
                if where == "in":
                    h = fc2.register_forward_pre_hook(lambda _, inp, lo=lo, up=up, pick=pick: (pick(inp[0], fq8r(inp[0], lo, up)),))
                else:
                    h = fc2.register_forward_hook(
                        lambda m_, inp, o, lo=lo, up=up, pick=pick: pick(o - m_.bias, fq8r(o - m_.bias, lo, up)) + m_.bias)
                r = measure(f"fq_{where}_fc2_{k}_{part}")
                r["range"] = [round(lo, 3), round(up, 3)]
                rows_b.append(r)
                h.remove()

    for k in (5, 9):
        chs = [67, 301, 145] if k == 5 else [67, 301, 145, 483]
        state = {"s": 0}

        def zero(_, inp, outp, k=k, chs=chs, state=state):
            s = state["s"]
            m = hi[k][s:s + len(outp)]
            o = outp.clone()
            for c in chs:
                o[..., c] = torch.where(m, torch.zeros_like(o[..., c]), o[..., c])
            state["s"] = s + len(outp)
            return o
        h = blocks[k].register_forward_hook(zero)
        r = measure(f"zero_massive_after{k}")
        r["channels"] = chs
        rows_b.append(r)
        h.remove()

    state = {"s": 0, "blk": None}

    def mask_attn(i):
        def f(_, inp, outp):
            s = state["s"]
            m = hi[i - 1][s:s + len(outp)]                         # 키 토큰 = 이전 블록 출력의 high-norm 토큰
            p = outp.masked_fill(m[:, None, None, :], 0.0)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            if i == len(blocks) - 1:
                state["s"] = s + len(outp)
            return p
        return f
    hs = [blocks[i].attn.attn_drop.register_forward_hook(mask_attn(i)) for i in range(6, len(blocks))]
    rows_b.append(measure("mask_attn_hi_6to11"))
    for h in hs:
        h.remove()

    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "n_images": N, "data": f"pets_dev[0:{a.n}] + caltech_dev[0:{a.n}]", "sec": round(time.time() - t0),
           "note": "B의 fq/clip은 가짜 양자화(디버깅·기전 확인용). 정확도 비교에 쓰지 않는다.",
           "pct_method": "exact topk, (100-99.999)/200 per side (ORT percentile asymmetric)", "ort_model": a.ort_model,
           "writers": rows_a, "interventions": rows_b}
    with open(out / "why_writers.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
