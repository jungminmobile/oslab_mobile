"""
s2_why_massive.py — 왜 SigLIP2는 블록 5~11에서 8비트 양자화가 무너지는가: 활성값 이상치의 구조 분석 (G3 원인 분석)
==============================================================================================================

[배경]
  G1·G3 결과: 전부 INT8이면 붕괴, 블록 5~11은 하나만 INT8이어도 붕괴, MLP fc2(특히 블록 5·9)가 결정적.
  G1 진단: 블록 5부터 잔차 흐름(블록 출력) 최대값이 99.9% 분위수의 39~68배, fc2 입력은 블록 5·9에서 124배·286배.

[가설] (실행 전에 정함)
  H-ch  (채널 쏠림): 큰 값은 정해진 소수 채널(768 중 몇 개)에만 생긴다 — LLM·ViT의 "massive activation"(Sun et al., 2024)
  H-tok (토큰 쏠림): 큰 값은 소수 토큰(이미지마다 1~몇 개 패치)에 몰린다 — ViT "high-norm token"(Darcet et al., 2023, registers)
  H-attn(attention 집중): 다른 토큰들이 그 high-norm 토큰에 attention을 크게 몰아준다 → 그 토큰 값이 틀리면 전체가 틀린다
  H-q   (8비트 붕괴 기전): 텐서별 8비트 눈금은 최대값에 맞춰지므로, 이상치 텐서에서는 대부분의 값이 0 또는 1~2칸으로 뭉개진다

[측정] PyTorch FP32(unfused attention), 고화질 dev 사진 N장(Pets·Caltech 앞쪽)
  블록마다 잔차 흐름 h(블록 출력, 196 토큰 × 768 채널), fc2 입력 g(GELU 출력, 196 × 3072), attention 확률
  ① 채널: 채널별 최대 |값| 상위 5개 채널과, 그 채널들이 전체 상위 0.01% 큰 값 중 차지하는 비율. 이미지끼리 같은 채널인지(겹침)
  ② 토큰: 토큰 L2 노름의 최대/중앙값 비율, 노름이 중앙값의 5배를 넘는 토큰 수, 그 토큰 위치(패치 번호)가 이미지마다 같은지
  ③ attention: high-norm 토큰이 받는 attention 확률 합(쿼리 평균) — 균등이면 1/196 = 0.0051
  ④ 8비트: 텐서 전체 min~max로 정한 비대칭 8비트(256칸)에서 |값| < 0.5칸이라 0으로 반올림되는 비율, 2칸 이하 비율

[사용]
  RN50_RESULTS=../results/siglip2/G3_why "$SIGLIP2_PYTHON" s2_why_massive.py --n 32
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


def q8_stats(t):
    """텐서 전체 min~max 비대칭 8비트: 한 칸 크기, 0으로 반올림되는 값 비율, 2칸 이하 비율."""
    lo, hi = float(t.min()), float(t.max())
    step = (hi - lo) / 255.0
    a = np.abs(t)
    return {"step": round(step, 5), "max_abs": round(float(a.max()), 3),
            "frac_round_to_zero": round(float((a < 0.5 * step).mean()), 4),
            "frac_le_2_steps": round(float((a <= 2 * step).mean()), 4)}


def channel_stats(t, topk=5):
    """t: (N, T, C). 채널별 최대 |값| 상위 채널과, 전체 상위 0.01% 값 중 그 채널들의 비율, 이미지별 최대 채널 일치도."""
    a = np.abs(t)
    ch_max = a.max(axis=(0, 1))
    top = np.argsort(-ch_max)[:topk]
    thr = np.quantile(a, 0.9999)
    big = a >= thr
    share = float(big[..., top].sum() / max(big.sum(), 1))
    per_img_top = a.max(axis=1).argmax(axis=1)                    # 이미지마다 최대 채널
    vals, cnt = np.unique(per_img_top, return_counts=True)
    return {"top_channels": [int(c) for c in top], "top_channel_max": [round(float(ch_max[c]), 2) for c in top],
            "median_channel_max": round(float(np.median(ch_max)), 3),
            "top5_share_of_top0.01pct": round(share, 3),
            "per_image_argmax_channel_mode": int(vals[cnt.argmax()]), "per_image_mode_frac": round(float(cnt.max() / len(per_img_top)), 3)}


def token_stats(t):
    """t: (N, T, C) → 토큰 노름 기준 high-norm 토큰 (노름 > 5×중앙값)."""
    norm = np.linalg.norm(t, axis=2)                               # (N, T)
    med = np.median(norm, axis=1, keepdims=True)
    hi = norm > 5 * med
    n_hi = hi.sum(1)
    pos = [int(np.argmax(norm[i])) for i in range(len(norm))]
    vals, cnt = np.unique(pos, return_counts=True)
    return {"max_over_median_norm": round(float(np.median((norm.max(1) / med[:, 0]))), 2),
            "n_highnorm_tokens_mean": round(float(n_hi.mean()), 2), "frac_images_with_highnorm": round(float((n_hi > 0).mean()), 3),
            "argmax_token_mode": int(vals[cnt.argmax()]), "argmax_token_mode_frac": round(float(cnt.max() / len(pos)), 3),
            "argmax_tokens_first8": pos[:8]}, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="데이터셋당 사진 수 (Pets·Caltech dev 앞쪽)")
    ap.add_argument("--sets", default="pets_dev,caltech_dev", help="캐시 이름(쉼표). 기본은 처음 분석과 같은 Pets·Caltech dev")
    a = ap.parse_args()
    import torch
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _, _ = S.load_model()
    unfuse_attention(model)
    trunk = model.visual.trunk
    sets = a.sets.split(",")
    xs = np.concatenate([np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n] for d in sets])
    x = torch.as_tensor(np.ascontiguousarray(xs))
    cap = {}

    def hook(name, take_input=False):
        def f(_, inp, out):
            cap.setdefault(name, []).append((inp[0] if take_input else out).detach().float().numpy())
        return f
    hs = []
    for i, b in enumerate(trunk.blocks):
        hs += [b.register_forward_hook(hook(f"h{i}")), b.mlp.fc2.register_forward_hook(hook(f"g{i}", True)),
               b.attn.attn_drop.register_forward_hook(hook(f"attn{i}", True))]
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(x), 16):
            model.visual(x[s:s + 16])
    for h in hs:
        h.remove()
    rows = []
    for i in range(len(trunk.blocks)):
        h = np.concatenate(cap[f"h{i}"]); g = np.concatenate(cap[f"g{i}"]); at = np.concatenate(cap[f"attn{i}"])
        assert h.shape[1:] == (196, 768) and g.shape[1:] == (196, 3072) and at.shape[1:] == (12, 196, 196), (h.shape, g.shape, at.shape)
        ts, hi = token_stats(h)
        # high-norm 토큰이 받는 attention: 쿼리·헤드 평균으로 각 키 토큰이 받는 확률, high-norm 토큰들의 합 (이미지 평균)
        recv = at.mean(axis=(1, 2))                                # (N, 196)
        attn_to_hi = float(np.mean([recv[n][hi[n]].sum() for n in range(len(recv)) if hi[n].any()])) if hi.any() else None
        rows.append({"block": i,
                     "residual": {"channel": channel_stats(h), "token": ts, "q8": q8_stats(h)},
                     "fc2_input": {"channel": channel_stats(g), "q8": q8_stats(g)},
                     "attn_mass_to_highnorm_tokens": None if attn_to_hi is None else round(attn_to_hi, 4),
                     "attn_uniform_per_token": round(1 / 196, 4)})
        r = rows[-1]
        print(f"block {i:2d} | res max/med-norm {ts['max_over_median_norm']:7.2f} hi-tok {ts['n_highnorm_tokens_mean']:4.1f} "
              f"top-ch {r['residual']['channel']['top_channels'][:3]} share {r['residual']['channel']['top5_share_of_top0.01pct']:.2f} "
              f"| q8 zero {r['residual']['q8']['frac_round_to_zero']:.2f} | fc2 q8 zero {r['fc2_input']['q8']['frac_round_to_zero']:.2f} "
              f"| attn->hi {r['attn_mass_to_highnorm_tokens']}", flush=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "n_images": int(len(x)), "data": " + ".join(f"{d}[0:{a.n}]" for d in sets), "sec": round(time.time() - t0),
           "blocks": rows}
    with open(out / "why_massive.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
