"""
s2_why_fix.py — 어떤 8비트 눈금이면 fc2 양자화 손실이 사라지는가 (G3 원인 분석 3단계, 기전 확인용 가짜 양자화)
=========================================================================================================

[배경] G3_why 1·2단계: 블록 5·9의 MLP가 고정 위치 토큰 1~2개에 거대 값을 써 넣고, 텐서별(per-tensor) 8비트에서는
  그 값이 눈금을 정해 "보통 토큰"의 작은 값이 0으로 뭉개진다. 그렇다면 눈금을 토큰별로 따로 잡거나 high-norm 토큰을
  눈금 계산에서 빼면 손실이 거의 사라져야 한다.

[가설] (실행 전에 정함)
  H-fix-tok : 토큰별 눈금(per-token, 각 토큰의 min~max)이면 fc2 5·9를 8비트로 해도 cos가 FP32에 가깝다 (≥ 0.99).
  H-fix-hi  : high-norm 토큰만 FP32로 두고(또는 눈금 계산에서 빼고) 나머지를 텐서별 눈금으로 하면 거의 회복된다.
  H-ch      : 채널별 눈금(per-channel, 입력 채널마다)은 덜 효과적이다. 거대 값이 소수 채널에 있지만 그 채널의 보통 토큰 값도
              함께 뭉개지기 때문이다. (반대로 나오면 채널 쏠림이 더 중요하다는 뜻)

[방법] PyTorch FP32(unfused attention), 고화질 dev 64장(why_massive·why_writers와 같은 사진).
  fc2의 입력과 MatMul 출력(bias 전)에 8비트 비대칭 가짜 양자화, 가중치는 출력 채널별 대칭 S8 (ORT QDQ와 같은 위치).
  범위는 정적(64장 전체에서 한 번) — per-token만은 토큰마다 그 자리에서 정한다(동적, 하드웨어 지원 여부는 별개).
  대상: (a) 블록 5·9의 fc2 두 개를 동시에  (b) fc2 12개 전부를 동시에  (G3 연산 유형 "fc2만 INT8" cos 0.648과 비교)
  눈금 방식: tensor(현재 ORT) / tensor_excl_hi(high-norm 토큰을 눈금 계산에서 빼고 그 토큰은 FP32로 통과)
           / token(토큰별 동적) / channel(입력은 입력 채널별, 출력은 출력 채널별 정적 min~max)
  ※ 가짜 양자화: 기전 확인용 디버깅 (CLAUDE.md 3절, D-008). 정확도·선택에 쓰지 않는다. 새 기법 채택은 팀 결정.

[사용]
  RN50_RESULTS=../results/siglip2/G3_why "$SIGLIP2_PYTHON" s2_why_fix.py --n 32
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
BS = 16


def fq(t, lo, hi):
    """비대칭 8비트 가짜 양자화. lo·hi는 스칼라 또는 t와 브로드캐스트되는 텐서. 범위는 0을 포함하게 넓힌다(ORT와 같음)."""
    import torch
    lo = torch.clamp(torch.as_tensor(lo, dtype=t.dtype), max=0.0)
    hi = torch.clamp(torch.as_tensor(hi, dtype=t.dtype), min=0.0)
    s = (hi - lo) / 255.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    zp = torch.round(-lo / s)
    q = torch.clamp(torch.round(t / s) + zp, 0, 255)
    return (q - zp) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="데이터셋당 사진 수 (Pets·Caltech dev 앞쪽)")
    a = ap.parse_args()
    import torch
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _, _ = S.load_model()
    unfuse_attention(model)
    vis, blocks = model.visual, model.visual.trunk.blocks
    xs = np.concatenate([np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n] for d in ("pets_dev", "caltech_dev")])
    x = torch.as_tensor(np.ascontiguousarray(xs))
    N = len(x)
    t0 = time.time()

    # 기준 실행: fc2 입력 g, MatMul 출력(bias 전) y, 블록 출력 노름(high-norm 토큰), 최종 임베딩
    cap = {}
    hs = []
    def keep_fc2(i):
        def f(m, inp, o):                                          # 값을 돌려주지 않는다 (돌려주면 출력이 바뀜)
            cap.setdefault(f"g{i}", []).append(inp[0].detach())
            cap.setdefault(f"y{i}", []).append((o - m.bias).detach())
        return f

    def keep_norm(i):
        def f(m, inp, o):
            cap.setdefault(f"n{i}", []).append(o.detach().norm(dim=-1))
        return f
    for i, b in enumerate(blocks):
        hs += [b.mlp.fc2.register_forward_hook(keep_fc2(i)), b.register_forward_hook(keep_norm(i))]

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
        nrm = T[f"n{i}"]
        hi[i] = nrm > 5 * nrm.median(dim=1, keepdim=True).values      # (N, 196) — fc2는 블록 출력과 같은 토큰 위치
    assert T["g0"].shape == (N, S.NTOK, 3072) and T["y0"].shape == (N, S.NTOK, 768)

    def ranges(t, m, mode):
        """정적 범위. tensor: 전체 min~max / tensor_excl_hi: high-norm 토큰 제외 min~max / channel: 마지막 축(채널)별 min~max."""
        if mode == "tensor":
            return float(t.min()), float(t.max())
        if mode == "tensor_excl_hi":
            v = t[~m]
            return float(v.min()), float(v.max())
        if mode == "channel":
            return t.amin(dim=(0, 1)), t.amax(dim=(0, 1))
        return None

    def install(bset, mode):
        """bset의 fc2에 가짜 양자화 hook을 달고, (hook 목록, 원래 가중치) 반환."""
        handles, saved = [], {}
        for i in bset:
            fc2 = blocks[i].mlp.fc2
            w0 = fc2.weight.data.clone()
            sc = w0.abs().amax(1, keepdim=True) / 127
            fc2.weight.data.copy_((w0 / sc).round().clamp(-127, 127) * sc)
            saved[i] = w0
            rin, rout = ranges(T[f"g{i}"], hi[i], mode), ranges(T[f"y{i}"], hi[i], mode)
            state = {"s": 0}

            def q(t, rng, m):
                if mode == "token":
                    return fq(t, t.amin(dim=-1, keepdim=True), t.amax(dim=-1, keepdim=True))
                qt = fq(t, rng[0], rng[1])
                if mode == "tensor_excl_hi":                      # high-norm 토큰은 FP32로 통과
                    return torch.where(m[..., None], t, qt)
                return qt

            def pre(_, inp, i=i, rin=rin, state=state):
                s = state["s"]
                m = hi[i][s:s + len(inp[0])]
                return (q(inp[0], rin, m),)

            def post(mod, inp, o, i=i, rout=rout, state=state):
                s = state["s"]
                m = hi[i][s:s + len(o)]
                state["s"] = s + len(o)
                return q(o - mod.bias, rout, m) + mod.bias
            handles += [fc2.register_forward_pre_hook(pre), fc2.register_forward_hook(post)]
        return handles, saved

    rows = []
    sets = {"fc2_5_9": [5, 9], "fc2_all12": list(range(12))}
    for sname, bset in sets.items():
        for mode in ("tensor", "tensor_excl_hi", "token", "channel"):
            handles, saved = install(bset, mode)
            e = run()
            for h in handles:
                h.remove()
            for i, w0 in saved.items():
                blocks[i].mlp.fc2.weight.data.copy_(w0)
            cos = (e * ref).sum(-1)
            r = {"set": sname, "mode": mode, "cos_mean": round(float(cos.mean()), 4), "cos_min": round(float(cos.min()), 4)}
            rows.append(r)
            print(f"{sname:10s} {mode:15s} cos {r['cos_mean']:.4f} (min {r['cos_min']:.4f})", flush=True)
    chk = run()
    assert float((chk * ref).sum(-1).min()) > 0.9999          # 가중치·hook이 모두 원래대로 돌아왔는지
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "n_images": N, "data": f"pets_dev[0:{a.n}] + caltech_dev[0:{a.n}]", "sec": round(time.time() - t0),
           "note": "가짜 양자화(기전 확인용). 정확도 비교·선택에 쓰지 않는다. token 모드는 동적(추론 때 토큰마다) 범위.",
           "rows": rows}
    with open(out / "why_fix.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
