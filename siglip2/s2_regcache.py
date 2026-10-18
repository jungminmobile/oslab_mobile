"""
s2_regcache.py — sink 토큰을 지우고 그 K·V를 상수로 캐시해 붙이는 SigLIP2 그래프 변환 (G3R, 판별 실험)
=========================================================================================================

[왜]  `results/siglip2/G3_why`: 블록 5 MLP가 패치 55에, 블록 9 MLP가 패치 13에 거대 값을 써 넣고(위치 100% 고정),
      그 값이 텐서별 8비트 눈금을 정해 보통 토큰을 뭉갠다. high-norm 토큰만 빼면 fc2 12개를 전부 8비트로 해도 cos 0.988
      (가짜 양자화, `why_fix`). 이 파일은 그 "빼기"를 정수 커널·NPU에서 쓸 수 있는 그래프로 만든다.
      설계와 판정 기준: `results/siglip2/G3R_regcache/README.md` (결과 전에 고정).

[변환]  sinks = [(패치 번호 p, 시작 블록 b), ...]  기본 "55@5,13@9"
  - 블록 b 입력에서 패치 p를 상수 인덱스로 지운다 (Slice + Concat, TopK·Gather 없음).
  - 블록 b 이후의 모든 attention과 MAP 풀링 헤드에서, K·V 뒤에 패치 p의 K·V를 붙인다.
  - sink 토큰이 다른 토큰에 영향을 주는 길은 attention의 K·V뿐이다 (LN·MLP는 토큰별) →
    붙이는 K·V가 "그 사진의 실제 K·V"면 원래 모델과 수학적으로 같다 (mode=oracle, 구현 검증용).
  - mode=mean: 원래 FP32 모델에서 보정 구간(train) 사진으로 sink K·V를 층·헤드별 평균한 상수 (주 구성 R1).
  - mode=none: 지우기만 하고 붙이지 않음 (소거 실험 R2).

[하위 명령]
  selftest  무작위 가중치 SigLIP 구조로 오라클 동치·ONNX export·연산 종류 확인 (데이터·가중치 불필요)
  cache     보정 사진으로 평균 캐시를 만들어 $RN50_ONNX/regcache_<sinks>_<보정>.npz 로 저장       (G1 준비)
  stats     sink K·V가 사진마다 얼마나 같은지 + 잔차 흐름·fc2 범위 비율 (원래 vs 변환)            (G0, G1b 부가)
  fp32      PyTorch FP32로 원래 / oracle / mean / none 의 dev 스크리닝 정확도·cos                    (G1a, G1b)
  export    변환 모델을 ONNX로 (배치 가변 + b1) + ORT vs PyTorch 확인 + 연산 표                       (G2·G3 준비)
  maskcheck 제자리 마스크 그래프(D-041) vs R1 FP32 동치 확인. export --graph mask로 마스크 ONNX
  hcache    sink 잔차를 채울 h_c (보정 train 평균, 비-sink 토큰). hccheck: h_c 마스크 사전 확인 (a)(b)(c)
  hcvcache  분산 맞춤 h_c* (G3b 4번째 시도): h_c의 채널 분산을 보정 train 비-sink LN 입력 분산 중앙값으로
  ucache    M2(--mask-site mlp)의 u_c: 원래 모델 norm2 출력의 보정 train 비-sink 평균. m2check: M2 사전 확인 (a)~(d)
  attn      (기록용) 블록마다 다른 토큰이 sink 패치에 주는 attention 비중 (원래 FP32, dev 사진)

[사용]  (siglip2/ 폴더에서, RN50_RESULTS 지정)
  python s2_regcache.py selftest
  RN50_RESULTS=../results/siglip2/G3R_regcache "$SIGLIP2_PYTHON" s2_regcache.py cache
  RN50_RESULTS=../results/siglip2/G3R_regcache "$SIGLIP2_PYTHON" s2_regcache.py stats --n 64
  RN50_RESULTS=../results/siglip2/G3R_regcache "$SIGLIP2_PYTHON" s2_regcache.py fp32 --k 200 --modes oracle mean none
  RN50_RESULTS=../results/siglip2/G3R_regcache "$SIGLIP2_PYTHON" s2_regcache.py export --verify 64
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
DEFAULT_SINKS = "55@5,13@9"
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]


# --------------------------------------------------------------------------- #
# 1. 변환 모델 (데이터·경로 의존 없음)
# --------------------------------------------------------------------------- #
def parse_sinks(s):
    """'55@5,13@9' → [(55, 5), (13, 9)] (시작 블록 순)."""
    out = [(int(p), int(b)) for p, b in (t.split("@") for t in s.split(",") if t.strip())]
    assert len({p for p, _ in out}) == len(out), f"같은 패치가 두 번: {s}"
    return sorted(out, key=lambda t: (t[1], t[0]))


def sinks_slug(sinks):
    return "-".join(f"{p}at{b}" for p, b in sinks)


def _check_trunk(trunk):
    assert getattr(trunk, "num_prefix_tokens", 0) == 0, "CLS·register 토큰이 있는 구조는 지원하지 않음 (SigLIP2는 0)"
    assert trunk.attn_pool is not None, "MAP 풀링 헤드(attn_pool)가 필요함"
    assert trunk.attn_pool.pos_embed is None, "attn_pool.pos_embed가 있는 구조는 지원하지 않음"
    for blk in trunk.blocks:
        assert getattr(blk.attn, "gate", None) is None, "gated attention은 지원하지 않음"


def _split_heads(t, B, N, n, H, d):
    """(B, N, n*H*d) → n개의 (B, H, N, d)."""
    return t.reshape(B, N, n, H, d).permute(2, 0, 3, 1, 4).unbind(0)


def block_kv(attn, h):
    """블록 attention 입력 h (B,N,C)에서 k_norm 뒤의 K, V (B,H,N,d)."""
    B, N, _ = h.shape
    _, k, v = _split_heads(attn.qkv(h), B, N, 3, attn.num_heads, attn.head_dim)
    return attn.k_norm(k), v


def pool_kv(pool, x):
    B, N, _ = x.shape
    k, v = _split_heads(pool.kv(x), B, N, 2, pool.num_heads, pool.head_dim)
    return pool.k_norm(k), v


def _cat_extra(k, v, kx, vx):
    if kx is None:
        return k, v
    B = k.shape[0]
    if kx.dim() == 3:                                   # 상수 캐시 (H, S, d) → 배치로 늘림
        kx, vx = kx.unsqueeze(0).expand(B, -1, -1, -1), vx.unsqueeze(0).expand(B, -1, -1, -1)
    return torch.cat([k, kx], 2), torch.cat([v, vx], 2)


def attn_forward(attn, h, kx=None, vx=None):
    """timm Attention.forward(unfused)와 같은 계산 + K·V 뒤에 kx·vx를 붙인다."""
    B, N, C = h.shape
    H, d = attn.num_heads, attn.head_dim
    q, k, v = _split_heads(attn.qkv(h), B, N, 3, H, d)
    q, k = attn.q_norm(q), attn.k_norm(k)
    k, v = _cat_extra(k, v, kx, vx)
    a = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    o = (a @ v).transpose(1, 2).reshape(B, N, H * d)
    return attn.proj(attn.norm(o))


def pool_forward(pool, x, kx=None, vx=None):
    """timm AttentionPoolLatent.forward(unfused)와 같은 계산 + K·V 뒤에 kx·vx를 붙인다."""
    B, N, C = x.shape
    H, d, L = pool.num_heads, pool.head_dim, pool.latent_len
    q = pool.q(pool.latent.expand(B, -1, -1)).reshape(B, L, H, d).transpose(1, 2)
    k, v = pool_kv(pool, x)
    q = pool.q_norm(q)
    k, v = _cat_extra(k, v, kx, vx)
    a = ((q * pool.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    o = pool.proj((a @ v).transpose(1, 2).reshape(B, L, C))
    if pool.mlp is not None:
        o = o + pool.mlp(pool.norm(o))
    if pool.pool == "token":
        o = o[:, 0]
    elif pool.pool == "avg":
        o = o.mean(1)
    return o


def _replace_kv(k, v, km, kc, vc):
    """sink 위치의 K·V를 상수로 바꿔 끼운다.
    full : K·km + kc (km = 1−m, kc = m·K_c 를 196토큰 크기로 미리 만든 상수)
    bcast: K + Σ_j m_j ⊙ (K_c,j − K) (m_j (1,1,N,1), K_c,j (1,H,1,d)). 상수끼리 곱하는 부분이 없어 export·ORT·DFC의 상수 접기로
           196토큰 크기 상수가 다시 생기지 않는다. 마스크가 서로 겹치지 않으므로 full과 수학적으로 같다."""
    if isinstance(km, (list, tuple)):
        k0, v0 = k, v
        for m, kj, vj in zip(km, kc, vc):
            k, v = k + m * (kj - k0), v + m * (vj - v0)
        return k, v
    return k * km + kc, v * km + vc


def attn_forward_mask(attn, h, km, kc, vc):
    """attn_forward와 같은 계산인데, K·V의 sink 위치를 상수로 바꿔 끼운다 (K' = K·km + kc). 토큰 수 그대로."""
    B, N, C = h.shape
    H, d = attn.num_heads, attn.head_dim
    q, k, v = _split_heads(attn.qkv(h), B, N, 3, H, d)
    q, k = attn.q_norm(q), attn.k_norm(k)
    k, v = _replace_kv(k, v, km, kc, vc)
    a = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    o = (a @ v).transpose(1, 2).reshape(B, N, H * d)
    return attn.proj(attn.norm(o))


def pool_forward_mask(pool, x, km, kc, vc):
    """pool_forward와 같은 계산인데, kv의 sink 위치를 상수로 바꿔 끼운다."""
    B, N, C = x.shape
    H, d, L = pool.num_heads, pool.head_dim, pool.latent_len
    q = pool.q(pool.latent.expand(B, -1, -1)).reshape(B, L, H, d).transpose(1, 2)
    k, v = pool_kv(pool, x)
    q = pool.q_norm(q)
    k, v = _replace_kv(k, v, km, kc, vc)
    a = ((q * pool.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    o = pool.proj((a @ v).transpose(1, 2).reshape(B, L, C))
    if pool.mlp is not None:
        o = o + pool.mlp(pool.norm(o))
    if pool.pool == "token":
        o = o[:, 0]
    elif pool.pool == "avg":
        o = o.mean(1)
    return o


def _embed_tokens(trunk, x):
    x = trunk.patch_embed(x)
    x = trunk._pos_embed(x)
    x = trunk.patch_drop(x)
    return trunk.norm_pre(x)


def _head(trunk, x):
    """풀링 뒤 forward_head 나머지 (fc_norm → head_drop → head)."""
    return trunk.head(trunk.head_drop(trunk.fc_norm(x)))


@torch.no_grad()
def sink_kv(trunk, x, sinks, return_out=False):
    """원래 모델로 x를 돌리며 sink 토큰의 층별 K·V를 꺼낸다.
    반환: {"b<l>": (K, V) (B,H,S_l,d), S_l = 블록 l에서 활성인 sink 수(시작 블록 순)}, {"pool": (K, V)}"""
    out, feats = {}, {}
    xt = _embed_tokens(trunk, x)
    for l, blk in enumerate(trunk.blocks):
        act = [p for p, b in sinks if b <= l]
        h = blk.norm1(xt)
        if act:
            k, v = block_kv(blk.attn, h)
            out[f"b{l}"] = (k[:, :, act], v[:, :, act])
        xt = xt + blk.drop_path1(blk.ls1(attn_forward(blk.attn, h)))
        xt = xt + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(xt))))
    xt = trunk.norm(xt)
    k, v = pool_kv(trunk.attn_pool, xt)
    act = [p for p, _ in sinks]
    out["pool"] = (k[:, :, act], v[:, :, act])
    if return_out:
        feats = _head(trunk, pool_forward(trunk.attn_pool, xt))
        return out, feats
    return out


class RegCacheTrunk(nn.Module):
    """timm VisionTransformer(trunk)를 감싸 sink 토큰을 지우고 K·V를 붙인다. trunk.forward와 같은 출력 형식."""

    def __init__(self, trunk, sinks, mode="mean", cache=None):
        super().__init__()
        assert mode in ("mean", "none", "oracle"), mode
        _check_trunk(trunk)
        self.t, self.sinks, self.mode = trunk, list(sinks), mode
        n_tok = trunk.patch_embed.num_patches
        assert all(0 <= p < n_tok and 0 < b < len(trunk.blocks) for p, b in self.sinks), self.sinks
        self.n_blocks = len(trunk.blocks)
        self.first = min(b for _, b in self.sinks)
        if mode == "mean":
            assert cache is not None, "mode=mean에는 cache가 필요함 (cache 하위 명령)"
            for key, (k, v) in cache.items():
                k, v = torch.as_tensor(np.asarray(k)), torch.as_tensor(np.asarray(v))
                assert k.dim() == 3 and k.shape == v.shape, (key, k.shape)
                self.register_buffer(f"k_{key}", k.float())
                self.register_buffer(f"v_{key}", v.float())

    def _extra(self, key, orc):
        if self.mode == "none":
            return None, None
        if self.mode == "oracle":
            return orc[key]
        return getattr(self, f"k_{key}"), getattr(self, f"v_{key}")

    def forward(self, x):
        trunk = self.t
        orc = sink_kv(trunk, x, self.sinks) if self.mode == "oracle" else None
        h_tok = _embed_tokens(trunk, x)
        keep = list(range(trunk.patch_embed.num_patches))       # 지금 남아 있는 원래 패치 번호 (파이썬 상수)
        for l, blk in enumerate(trunk.blocks):
            for p, b in self.sinks:
                if b == l:                                      # 상수 인덱스로 지움 → ONNX Slice + Concat
                    i = keep.index(p)
                    h_tok = torch.cat([h_tok[:, :i], h_tok[:, i + 1:]], 1)
                    keep.pop(i)
            kx, vx = self._extra(f"b{l}", orc) if l >= self.first else (None, None)
            h_tok = h_tok + blk.drop_path1(blk.ls1(attn_forward(blk.attn, blk.norm1(h_tok), kx, vx)))
            h_tok = h_tok + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h_tok))))
        h_tok = trunk.norm(h_tok)
        kx, vx = self._extra("pool", orc)
        return _head(trunk, pool_forward(trunk.attn_pool, h_tok, kx, vx))


class MaskRegCacheTrunk(nn.Module):
    """제자리 마스크 그래프 (D-041): 토큰 196개를 유지하고, sink 위치 p의 K·V를 상수 캐시로 바꿔 끼운다.
    K' = K·(1−m) + m·K_c, V' 같음 (m은 블록 l에서 활성인 sink 위치만 1). 풀링 헤드 kv도 같다.
    sink 토큰의 잔차는 시작 블록 입력부터 매 블록 입출력에서 h·(1−m)으로 0 → 거대 값이 생기지 않는다.
    다른 토큰이 보는 key·value 집합이 RegCacheTrunk(mean)와 같으므로 수학적으로 같다. ONNX에는 상수 Mul·Add만 늘어난다 (Concat·Slice 없음)."""

    def __init__(self, trunk, sinks, cache, form="full", hc=None, uc=None):
        super().__init__()
        _check_trunk(trunk)
        assert form in ("full", "bcast"), form
        assert hc is None or uc is None, "hc(잔차 채움)와 uc(MLP 입력 바꿔 끼우기)는 함께 쓰지 않는다"
        self.t, self.sinks, self.form = trunk, list(sinks), form
        self.n_act = {}
        self.hfill = hc is not None          # True: sink 잔차를 0 대신 h_c(보정 평균, 비-sink 토큰)로 채움 (G3b 마지막 시도)
        self.mlp_site = uc is not None       # True (M2, G3b 5번째): 잔차는 건드리지 않고 sink 행의 MLP 입력(norm2 출력)만 u_c로 바꿔 끼움
        N = trunk.patch_embed.num_patches
        assert all(0 <= p < N and 0 < b < len(trunk.blocks) for p, b in self.sinks), self.sinks
        self.first = min(b for _, b in self.sinks)
        H, d = trunk.blocks[0].attn.num_heads, trunk.blocks[0].attn.head_dim
        keys = [(f"b{l}", [p for p, b in self.sinks if b <= l]) for l in range(self.first, len(trunk.blocks))]
        keys.append(("pool", [p for p, _ in self.sinks]))
        if self.mlp_site:
            for l in range(self.first, len(trunk.blocks)):
                act = [p for p, b in self.sinks if b <= l]
                m = torch.zeros(1, N, 1)
                m[0, act] = 1
                self.register_buffer(f"zm_b{l}", m)                                       # MLP 입력을 바꿀 위치 m
                self.register_buffer(f"uc_b{l}", torch.as_tensor(np.asarray(uc[f"b{l}"])).float().reshape(1, 1, -1))
        if self.hfill:
            for l in list(range(self.first, len(trunk.blocks))) + ["final"]:
                key = f"b{l}" if l != "final" else "final"
                act = [p for p, b in self.sinks if l == "final" or b <= l]
                m = torch.zeros(1, N, 1)
                m[0, act] = 1
                v = torch.as_tensor(np.asarray(hc[key])).float().reshape(1, 1, -1)
                self.register_buffer(f"hfm_{key}", m)                                  # 채울 위치 m
                self.register_buffer(f"hc_{key}", v)                                   # h_c (1,1,C)
        for key, act in keys:
            k, v = (torch.as_tensor(np.asarray(t)).float() for t in cache[key])        # (H, S, d), act 순서
            assert k.shape == (H, len(act), d), (key, k.shape)
            keep = torch.ones(N)
            kc, vc = torch.zeros(1, H, N, d), torch.zeros(1, H, N, d)
            for j, p in enumerate(act):
                keep[p] = 0
                kc[0, :, p], vc[0, :, p] = k[:, j], v[:, j]
            self.register_buffer(f"hm_{key}", keep.view(1, N, 1))                      # 잔차 마스크 (1−m)
            self.n_act[key] = len(act)
            if form == "full":
                self.register_buffer(f"km_{key}", keep.view(1, 1, N, 1))               # K·V 마스크 (1−m)
                self.register_buffer(f"kc_{key}", kc)                                  # m·K_c (196토큰 크기)
                self.register_buffer(f"vc_{key}", vc)
            else:
                for j, p in enumerate(act):                                            # sink마다 one-hot m_j와 헤드별 1행 K_c,j
                    m = torch.zeros(1, 1, N, 1)
                    m[0, 0, p] = 1
                    self.register_buffer(f"m_{key}_{j}", m)
                    self.register_buffer(f"kc_{key}_{j}", k[:, j].reshape(1, H, 1, d).clone())
                    self.register_buffer(f"vc_{key}_{j}", v[:, j].reshape(1, H, 1, d).clone())

    def _mlp_in(self, z, l):
        """M2: z + m ⊙ (u_c − z) (sink 행의 MLP 입력만 u_c로). mlp_site가 아니면 그대로."""
        if not self.mlp_site:
            return z
        m = getattr(self, f"zm_b{l}")
        return z + m * (getattr(self, f"uc_b{l}") - z)

    def _fill(self, h, key):
        """hfill=False: h·(1−m) (0으로). hfill=True: h + m ⊙ (h_c − h) (상수 접기로 큰 상수가 생기지 않는 꼴)."""
        if not self.hfill:
            return h * getattr(self, f"hm_{key}")
        m = getattr(self, f"hfm_{key}")
        return h + m * (getattr(self, f"hc_{key}") - h)

    def _kv(self, key):
        if self.form == "full":
            return getattr(self, f"km_{key}"), getattr(self, f"kc_{key}"), getattr(self, f"vc_{key}")
        js = range(self.n_act[key])
        return ([getattr(self, f"m_{key}_{j}") for j in js], [getattr(self, f"kc_{key}_{j}") for j in js],
                [getattr(self, f"vc_{key}_{j}") for j in js])

    def forward(self, x):
        trunk = self.t
        h = _embed_tokens(trunk, x)
        for l, blk in enumerate(trunk.blocks):
            if l < self.first:
                h = blk(h)
                continue
            if not self.mlp_site:
                h = self._fill(h, f"b{l}")                                              # 시작 블록 입력부터 sink 잔차 0 (또는 h_c)
            h = h + blk.drop_path1(blk.ls1(attn_forward_mask(blk.attn, blk.norm1(h), *self._kv(f"b{l}"))))
            h = h + blk.drop_path2(blk.ls2(blk.mlp(self._mlp_in(blk.norm2(h), l))))
            if not self.hfill and not self.mlp_site:
                h = h * getattr(self, f"hm_b{l}")                                       # 블록 출력에서도 0
        if self.hfill:
            h = self._fill(h, "final")
        h = trunk.norm(h)
        return _head(trunk, pool_forward_mask(trunk.attn_pool, h, *self._kv("pool")))


@torch.no_grad()
def build_mean_cache(trunk, x, sinks, bs=16):
    """보정 사진 x (N,3,H,W)로 sink K·V의 사진 평균 → {key: (K (H,S,d), V)} numpy float32."""
    acc, n = {}, 0
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(np.ascontiguousarray(x[i:i + bs]), dtype=torch.float32)
        kv = sink_kv(trunk, xb, sinks)
        for key, (k, v) in kv.items():
            ks, vs = k.sum(0).double(), v.sum(0).double()
            if key in acc:
                acc[key][0].add_(ks)
                acc[key][1].add_(vs)
            else:
                acc[key] = [ks, vs]
        n += len(xb)
    return {key: ((k / n).float().numpy(), (v / n).float().numpy()) for key, (k, v) in acc.items()}, n


@torch.no_grad()
def build_hc(trunk, x, sinks, bs=16):
    """보정 사진 x로 h_c: 블록 l(≥ 첫 시작 블록) 입력 잔차와 마지막 norm 입력을 sink가 아닌 토큰·사진에 대해 평균 → {"b<l>"/"final": (C,)}."""
    first = min(b for _, b in sinks)
    N = trunk.patch_embed.num_patches
    keep = [i for i in range(N) if i not in {p for p, _ in sinks}]
    acc, n = {}, 0
    for i in range(0, len(x), bs):
        h = _embed_tokens(trunk, torch.as_tensor(np.ascontiguousarray(x[i:i + bs]), dtype=torch.float32))
        for l, blk in enumerate(trunk.blocks):
            if l >= first:
                acc[f"b{l}"] = acc.get(f"b{l}", 0) + h[:, keep].double().sum((0, 1))
            h = blk(h)
        acc["final"] = acc.get("final", 0) + h[:, keep].double().sum((0, 1))
        n += len(h)
    return {k: (v / (n * len(keep))).float().numpy() for k, v in acc.items()}, n


def _hc_path(RC, sinks, calib_n, seed):
    return RC.ONNX_DIR / f"regcache_hc_{sinks_slug(sinks)}_highres{calib_n}_s{seed}.npz"


def _hcv_path(RC, sinks, calib_n, seed):
    return RC.ONNX_DIR / f"regcache_hcv_{sinks_slug(sinks)}_highres{calib_n}_s{seed}.npz"


@torch.no_grad()
def build_vtarget(trunk, x, sinks, bs=16):
    """보정 사진 x로 v_target: 블록 l(≥ 첫 시작 블록) 입력(= norm1 입력)과 마지막 norm 입력에서 sink가 아닌 토큰의
    토큰별 채널 분산(편향, LayerNorm과 같음)의 중앙값 → {"b<l>"/"final": float}."""
    first = min(b for _, b in sinks)
    N = trunk.patch_embed.num_patches
    keep = [i for i in range(N) if i not in {p for p, _ in sinks}]
    vs, n = {}, 0
    for i in range(0, len(x), bs):
        h = _embed_tokens(trunk, torch.as_tensor(np.ascontiguousarray(x[i:i + bs]), dtype=torch.float32))
        for l, blk in enumerate(trunk.blocks):
            if l >= first:
                vs.setdefault(f"b{l}", []).append(h[:, keep].var(-1, unbiased=False).flatten())
            h = blk(h)
        vs.setdefault("final", []).append(h[:, keep].var(-1, unbiased=False).flatten())
        n += len(h)
    return {k: float(torch.cat(v).double().median()) for k, v in vs.items()}, n, len(keep)


def _uc_path(RC, sinks, calib_n, seed):
    return RC.ONNX_DIR / f"regcache_uc_{sinks_slug(sinks)}_highres{calib_n}_s{seed}.npz"


@torch.no_grad()
def build_uc(trunk, x, sinks, bs=16):
    """M2의 u_c: 원래 FP32 모델에서 블록 l(≥ 첫 시작 블록)의 norm2 출력(affine 포함)을 sink가 아닌 토큰·사진에 대해 평균 → {"b<l>": (C,)}."""
    first = min(b for _, b in sinks)
    N = trunk.patch_embed.num_patches
    keep = [i for i in range(N) if i not in {p for p, _ in sinks}]
    acc, n = {}, 0
    for i in range(0, len(x), bs):
        h = _embed_tokens(trunk, torch.as_tensor(np.ascontiguousarray(x[i:i + bs]), dtype=torch.float32))
        for l, blk in enumerate(trunk.blocks):
            if l >= first:
                h = h + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(h))))
                z = blk.norm2(h)
                acc[f"b{l}"] = acc.get(f"b{l}", 0) + z[:, keep].double().sum((0, 1))
                h = h + blk.drop_path2(blk.ls2(blk.mlp(z)))
            else:
                h = blk(h)
        n += len(h)
    return {k: (v / (n * len(keep))).float().numpy() for k, v in acc.items()}, n


def match_var(hc, vt):
    """h_c* = μ + (h_c − μ)·√(v_target / v(h_c)) — 채널 평균은 그대로, 채널 분산을 v_target으로."""
    out, info = {}, {}
    for k, v in hc.items():
        v = np.asarray(v, dtype=np.float64)
        mu, var = v.mean(), v.var()
        w = mu + (v - mu) * np.sqrt(vt[k] / var)
        out[k] = w.astype(np.float32)
        info[k] = {"mu": round(float(mu), 6), "v_hc": round(float(var), 6), "v_target": round(float(vt[k]), 6),
                   "v_hc_star": round(float(w.var()), 6)}
    return out, info


def save_cache(path, cache, meta):
    arr = {}
    for key, (k, v) in cache.items():
        arr[f"k_{key}"], arr[f"v_{key}"] = k, v
    np.savez(path, meta=json.dumps(meta, ensure_ascii=False), **arr)


def load_cache(path):
    z = np.load(path)
    keys = sorted({n[2:] for n in z.files if n.startswith("k_")})
    return {key: (z[f"k_{key}"], z[f"v_{key}"]) for key in keys}, json.loads(str(z["meta"]))


def unfuse(module):
    n = 0
    for m in module.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 2. selftest (무작위 가중치, 데이터 불필요)
# --------------------------------------------------------------------------- #
def cmd_selftest(a):
    import timm
    torch.manual_seed(0)
    trunk = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0).eval()
    unfuse(trunk)
    sinks = parse_sinks(a.sinks)
    x = torch.randn(4, 3, 224, 224)
    res = {"timm": timm.__version__, "torch": torch.__version__, "sinks": sinks}
    with torch.no_grad():
        ref = trunk(x)
        kv, ref2 = sink_kv(trunk, x, sinks, return_out=True)
        res["sink_kv_reproduces_trunk_max_abs"] = float((ref - ref2).abs().max())
        orc = RegCacheTrunk(trunk, sinks, "oracle").eval()(x)
        res["oracle_vs_original_max_abs"] = float((orc - ref).abs().max())
        res["oracle_vs_original_cos_min"] = float(torch.nn.functional.cosine_similarity(orc, ref).min())
        cache, n = build_mean_cache(trunk, x.numpy(), sinks, bs=2)
        res["cache_keys"] = {k: list(v[0].shape) for k, v in cache.items()}
        rc = RegCacheTrunk(trunk, sinks, "mean", cache).eval()
        mean_out = rc(x)
        none_out = RegCacheTrunk(trunk, sinks, "none").eval()(x)
        res["mask_vs_mean_max_abs"] = float((MaskRegCacheTrunk(trunk, sinks, cache).eval()(x) - mean_out).abs().max())
        res["maskbc_vs_mean_max_abs"] = float((MaskRegCacheTrunk(trunk, sinks, cache, "bcast").eval()(x) - mean_out).abs().max())
        hc, _ = build_hc(trunk, x.numpy(), sinks, bs=2)
        uc_t, _ = build_uc(trunk, x.numpy(), sinks, bs=2)
        res["maskmlp_vs_mean_max_abs"] = float((MaskRegCacheTrunk(trunk, sinks, cache, "bcast", None, uc_t).eval()(x) - mean_out).abs().max())
        res["maskhc_vs_mean_max_abs"] = float((MaskRegCacheTrunk(trunk, sinks, cache, "bcast", hc).eval()(x) - mean_out).abs().max())
        res["mean_vs_original_cos_min"] = float(torch.nn.functional.cosine_similarity(mean_out, ref).min())
        res["none_vs_original_cos_min"] = float(torch.nn.functional.cosine_similarity(none_out, ref).min())
        # 배치 1 이미지의 평균 캐시 = 그 이미지 자신의 K·V → mean 출력이 원래와 같아야 함 (캐시 경로 검증)
        c1, _ = build_mean_cache(trunk, x[:1].numpy(), sinks)
        res["mean_selfcache_vs_original_max_abs"] = float((RegCacheTrunk(trunk, sinks, "mean", c1).eval()(x[:1]) - ref[:1]).abs().max())
    # ONNX export (배치 가변) + ORT
    import onnx
    import onnxruntime as ort
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "selftest_regcache.onnx"
    with torch.inference_mode():
        torch.onnx.export(rc, (x[:1],), str(p), input_names=["image"], output_names=["embedding"], opset_version=17,
                          dynamo=False, dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}})
    g = onnx.load(str(p)).graph
    ops = {}
    for nd in g.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    res["onnx_ops"] = dict(sorted(ops.items()))
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    y = s.run(None, {"image": x.numpy()})[0]
    res["ort_vs_torch_max_abs"] = float(np.abs(y - mean_out.numpy()).max())
    p.unlink()
    checks = {
        "sink_kv 경로 = timm trunk": res["sink_kv_reproduces_trunk_max_abs"] < 1e-4,
        "oracle = 원래 (G1a 구현 검증)": res["oracle_vs_original_max_abs"] < 1e-4,
        "자기 캐시 mean = 원래": res["mean_selfcache_vs_original_max_abs"] < 1e-4,
        "ORT = PyTorch": res["ort_vs_torch_max_abs"] < 1e-3,
        "제자리 마스크 = mean (D-041)": res["mask_vs_mean_max_abs"] < 1e-4,
        "제자리 마스크 bcast = mean": res["maskbc_vs_mean_max_abs"] < 1e-4,
        "제자리 마스크 bcast + h_c 채움 = mean": res["maskhc_vs_mean_max_abs"] < 1e-4,
        "제자리 마스크 bcast + MLP 입력 u_c (M2) = mean": res["maskmlp_vs_mean_max_abs"] < 1e-4,
        "TopK·NonZero 없음 (NPU 친화)": not any(o in ops for o in ("TopK", "NonZero", "ScatterND")),
    }
    res["checks"] = checks
    print(json.dumps(res, ensure_ascii=False, indent=1))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    print("selftest 통과")


# --------------------------------------------------------------------------- #
# 3. 실제 모델 명령 (s2common: 환경 변수·데이터 필요)
# --------------------------------------------------------------------------- #
def _setup():
    sys.path.insert(0, str(HERE))
    import s2common as S
    from scripts import env_info
    return S, S.RC, env_info


def _load_visual(S, threads):
    torch.set_num_threads(threads)
    model, preprocess, _ = S.load_model()
    unfuse(model)
    return model, preprocess


def _cache_path(RC, sinks, calib_n, seed):
    return RC.ONNX_DIR / f"regcache_{sinks_slug(sinks)}_highres{calib_n}_s{seed}.npz"


def _row(env, a, **kw):
    from scripts import env_info
    return {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
            "dirty": env["git"]["dirty"], **env_info.flat(env), "cpu_flags": env["cpu"]["flags"],
            "options": {k: v for k, v in vars(a).items() if k != "func"}, **kw}


def _append(out, name, row):
    with open(out / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_trunk(S, RC, model, a, mode):
    sinks = parse_sinks(a.sinks)
    cache = None
    if mode == "mean":
        cp = _cache_path(RC, sinks, a.calib_n, a.seed)
        assert cp.exists(), f"{cp.name} 없음 — 먼저 cache 하위 명령"
        cache, meta = load_cache(cp)
        assert meta["sinks"] == [list(s) for s in sinks], (meta["sinks"], sinks)
    return RegCacheTrunk(model.visual.trunk, sinks, mode, cache).eval()


def _make_mask_trunk(RC, model, a):
    sinks = parse_sinks(a.sinks)
    cp = _cache_path(RC, sinks, a.calib_n, a.seed)
    assert cp.exists(), f"{cp.name} 없음 — 먼저 cache 하위 명령"
    cache, meta = load_cache(cp)
    assert meta["sinks"] == [list(s) for s in sinks], (meta["sinks"], sinks)
    hc = None
    if getattr(a, "hfill", "zero") in ("mean", "meanvar"):
        hp = (_hc_path if a.hfill == "mean" else _hcv_path)(RC, sinks, a.calib_n, a.seed)
        assert hp.exists(), f"{hp.name} 없음 — 먼저 hcache / hcvcache 하위 명령"
        z = np.load(hp)
        assert json.loads(str(z["meta"]))["sinks"] == [list(s) for s in sinks]
        hc = {k: z[k] for k in z.files if k != "meta"}
    uc = None
    if getattr(a, "mask_site", "residual") == "mlp":
        assert getattr(a, "hfill", "zero") == "zero", "--mask-site mlp에서는 --hfill을 쓰지 않는다 (잔차를 건드리지 않음)"
        up = _uc_path(RC, sinks, a.calib_n, a.seed)
        assert up.exists(), f"{up.name} 없음 — 먼저 ucache 하위 명령"
        z = np.load(up)
        assert json.loads(str(z["meta"]))["sinks"] == [list(s) for s in sinks]
        uc = {k: z[k] for k in z.files if k != "meta"}
    return MaskRegCacheTrunk(model.visual.trunk, sinks, cache, getattr(a, "mask_form", "full"), hc, uc).eval()


def cmd_cache(a):
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    from s2_quantize import calib_images                 # s2_quantize와 같은 보정 사진 (train, dev 제외)
    model, _ = _load_visual(S, a.threads)
    sinks = parse_sinks(a.sinks)
    x = calib_images("highres", a.calib_n, a.seed)
    t0 = time.time()
    cache, n = build_mean_cache(model.visual.trunk, x, sinks)
    assert n == a.calib_n, f"캐시에 쓴 사진 {n} ≠ 요청 {a.calib_n}"
    cp = _cache_path(RC, sinks, a.calib_n, a.seed)
    meta = {"sinks": [list(s) for s in sinks], "calib": f"highres{a.calib_n}_s{a.seed}", "n": n, "model": S.MODEL_ID,
            "commit": env["git"]["commit"]}
    save_cache(cp, cache, meta)
    row = _row(env, a, file=cp.name, sha256=_sha256(cp), n=n, sec=round(time.time() - t0, 1),
               shapes={k: list(v[0].shape) for k, v in cache.items()})
    _append(out, "cache.jsonl", row)
    print(json.dumps(row, ensure_ascii=False, indent=1))


def _dev_x(RC, n, sets):
    return np.concatenate([np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:n]) for d in sets])


def _range_ratio(t):
    """max|t| / 99.9% 분위수(|t|). torch.quantile 원소 수 한도 때문에 큰 텐서는 kthvalue로."""
    v = t.abs().flatten().float()
    k = max(1, int(np.ceil(0.999 * v.numel())))
    q = v.kthvalue(k).values
    return float(v.max() / q.clamp_min(1e-12)), float(v.max()), float(q)


@torch.no_grad()
def _ranges(trunk, rc, x):
    """블록 출력(잔차 흐름)과 fc2 입력·출력의 max/p99.9 비율. rc=None이면 원래 모델."""
    caps, hs = {}, []
    blocks = trunk.blocks

    def _hook(m, i, o, l):
        # forward hook이 None이 아닌 값을 돌려주면 PyTorch가 그 값으로 출력을 바꾼다 → 반드시 None 반환
        caps.setdefault(f"fc2_in{l}", []).append(i[0].detach())
        caps.setdefault(f"fc2_out{l}", []).append((o - m.bias).detach())

    for l, blk in enumerate(blocks):
        hs.append(blk.mlp.fc2.register_forward_hook(lambda m, i, o, l=l: _hook(m, i, o, l)))
    try:
        if rc is None:
            xt = _embed_tokens(trunk, x)
            for l, blk in enumerate(blocks):
                xt = blk(xt)
                caps.setdefault(f"res{l}", []).append(xt)
        else:
            # RegCacheTrunk.forward와 같은 순서로 잔차 흐름을 꺼내기 위해 블록 출력 hook 대신 직접 재현
            h_tok, keep = _embed_tokens(trunk, x), list(range(trunk.patch_embed.num_patches))
            for l, blk in enumerate(blocks):
                for p, b in rc.sinks:
                    if b == l:
                        i = keep.index(p)
                        h_tok = torch.cat([h_tok[:, :i], h_tok[:, i + 1:]], 1)
                        keep.pop(i)
                kx, vx = rc._extra(f"b{l}", None) if l >= rc.first else (None, None)
                h_tok = h_tok + blk.drop_path1(blk.ls1(attn_forward(blk.attn, blk.norm1(h_tok), kx, vx)))
                h_tok = h_tok + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h_tok))))
                caps.setdefault(f"res{l}", []).append(h_tok)
    finally:
        for h in hs:
            h.remove()
    return {k: dict(zip(("ratio", "max", "p999"), (round(z, 4) for z in _range_ratio(torch.cat(v))))) for k, v in caps.items()}


def cmd_stats(a):
    """G0: sink K·V의 사진 간 일관성 (dev 사진의 K·V vs 보정 평균 캐시) + 잔차·fc2 범위 비율 (원래 vs mean 변환)."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    trunk, sinks = model.visual.trunk, parse_sinks(a.sinks)
    cache, _ = load_cache(_cache_path(RC, sinks, a.calib_n, a.seed))
    x = torch.as_tensor(_dev_x(RC, a.n // 2, ["pets_dev", "caltech_dev"]))     # G3_why와 같은 고화질 dev 사진 구성
    kv = {}
    for i in range(0, len(x), 16):
        for key, (k, v) in sink_kv(trunk, x[i:i + 16], sinks).items():
            kv.setdefault(key, [[], []])
            kv[key][0].append(k)
            kv[key][1].append(v)
    cons = {}
    for key, (ks, vs) in kv.items():
        k, v = torch.cat(ks), torch.cat(vs)                                   # (N,H,S,d)
        kc, vc = torch.as_tensor(cache[key][0]), torch.as_tensor(cache[key][1])
        ck = torch.nn.functional.cosine_similarity(k, kc.unsqueeze(0), dim=-1)   # (N,H,S)
        cv = torch.nn.functional.cosine_similarity(v, vc.unsqueeze(0), dim=-1)
        rk = (k - kc).norm(dim=-1) / kc.norm(dim=-1).clamp_min(1e-12)
        cons[key] = {"k_cos_median_by_sink": ck.median(0).values.median(0).values.tolist(),
                     "k_cos_min_head_median": float(ck.median(0).values.min()),
                     "v_cos_median_by_sink": cv.median(0).values.median(0).values.tolist(),
                     "v_cos_min_head_median": float(cv.median(0).values.min()),
                     "k_relerr_median": float(rk.median())}
    rc = _make_trunk(S, RC, model, a, "mean")
    rng = {"original": _ranges(trunk, None, x), "mean": _ranges(trunk, rc, x)}
    flag = sorted(k for k, c in cons.items() if min(c["k_cos_min_head_median"], c["v_cos_min_head_median"]) < 0.95)
    row = _row(env, a, n=int(len(x)), consistency=cons, ranges=rng, below_095=flag)
    _append(out, "stats.jsonl", row)
    print("층별 (k, v) 최소 헤드 중앙 cos:", {k: (round(c["k_cos_min_head_median"], 4), round(c["v_cos_min_head_median"], 4)) for k, c in cons.items()})
    print("잔차 비율 원래 → 변환:", {l: (rng["original"][f"res{l}"]["ratio"], rng["mean"][f"res{l}"]["ratio"]) for l in range(len(trunk.blocks))})
    print("0.95 미만 층:", flag or "없음")


@torch.no_grad()
def _feat(visual, x, bs=32):
    o = [visual(torch.as_tensor(np.ascontiguousarray(x[i:i + bs]), dtype=torch.float32)).float().numpy()
         for i in range(0, len(x), bs)]
    return np.concatenate(o)


def cmd_fp32(a):
    """G1a·G1b 스크리닝: PyTorch FP32로 원래·변환 모델의 dev 정확도와 원래 FP32 임베딩(ref_fp32 캐시) 대비 cos."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    orig_trunk = model.visual.trunk
    for mode in ["original", *a.modes]:
        t0 = time.time()
        model.visual.trunk = orig_trunk                    # 앞 mode의 RegCacheTrunk를 다시 감싸지 않도록 원래 trunk로 되돌림
        if mode != "original":
            model.visual.trunk = _make_trunk(S, RC, model, a, mode)
        res = {}
        for d in DATA:
            f = lambda s: RC.CACHE_DIR / f"{d}_{s}.npy"
            x = np.asarray(np.load(f("x"), mmap_mode="r")[:a.k])
            y, W, ref = np.load(f("y"))[:a.k], np.load(f("clf")), np.load(f("ref_fp32"))[:a.k]
            feat = RC.l2n(_feat(model.visual, x))
            m = RC.topk_metrics(feat, W, y, ref)
            m["fp32_top1"] = RC.topk_metrics(ref, W, y)["top1"]
            m["cos_min"] = round(float((feat * ref).sum(1).min()), 6)
            res[d] = m
        res["hr_cos"] = round(float(np.mean([res[d]["cos_mean"] for d in HR])), 5)
        res["hr_cos_min"] = round(float(min(res[d]["cos_min"] for d in HR)), 6)
        res["hr_dtop1"] = round(float(np.mean([res[d]["top1"] - res[d]["fp32_top1"] for d in HR])), 2)
        row = _row(env, a, mode=mode, k=a.k, result=res, sec=round(time.time() - t0, 1))
        _append(out, "fp32_screen.jsonl", row)
        print(f"{mode:9s} 고화질 cos {res['hr_cos']:.5f} (최소 {res['hr_cos_min']:.6f})  Δtop-1 {res['hr_dtop1']:+.2f}%p  "
              f"CIFAR A {res['cifar_A']['top1'] - res['cifar_A']['fp32_top1']:+.2f}%p  ({row['sec']}s)")
    model.visual.trunk = orig_trunk


def cmd_export(a):
    """변환 모델(mean 또는 none) → ONNX. 파일: siglip2_image_fp32_rc<sinks>[_none].onnx (+ _b1). ORT vs PyTorch 확인."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, preprocess = _load_visual(S, a.threads)
    sinks = parse_sinks(a.sinks)
    if a.graph != "mask":
        assert a.mask_form == "full" and a.hfill == "zero" and a.mask_site == "residual", \
            "--mask-form·--hfill·--mask-site는 --graph mask에서만 쓴다 (concat에서는 무시되므로 오류로 멈춤)"
    if a.graph == "mask":
        assert a.mode == "mean", "--graph mask는 mode=mean만"
        model.visual.trunk = _make_mask_trunk(RC, model, a)
        stem = (f"siglip2_image_fp32_rcmask{'' if a.mask_form == 'full' else 'bc'}{ {'zero': '', 'mean': 'hc', 'meanvar': 'hcv'}[a.hfill] }"
                f"{'mlp' if a.mask_site == 'mlp' else ''}{sinks_slug(sinks)}")
    else:
        model.visual.trunk = _make_trunk(S, RC, model, a, a.mode)
        stem = f"siglip2_image_fp32_rc{sinks_slug(sinks)}" + ("" if a.mode == "mean" else f"_{a.mode}")
    paths = [RC.ONNX_DIR / f"{stem}.onnx", RC.ONNX_DIR / f"{stem}_b1.onnx"]
    import onnx
    x1 = torch.randn(1, 3, 224, 224)
    files = {}
    for p, dyn in zip(paths, (True, False)):
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p.name} 이미 있음 (--overwrite)")
        kw = dict(dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}) if dyn else {}
        with torch.inference_mode():
            torch.onnx.export(model.visual, (x1,), str(p), input_names=["image"], output_names=["embedding"],
                              opset_version=a.opset, dynamo=False, **kw)
        ops = {}
        for nd in onnx.load(str(p), load_external_data=False).graph.node:
            ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
        files[p.name] = {"MiB": round(p.stat().st_size / 2**20, 2), "sha256": _sha256(p), "ops": dict(sorted(ops.items()))}
        print(p.name, files[p.name]["MiB"], "MiB")
    row = _row(env, a, files=files, cache=_cache_path(RC, sinks, a.calib_n, a.seed).name if a.mode == "mean" else None)
    if a.verify:
        from s2_export_onnx import cifar_dev
        x = cifar_dev(preprocess, a.verify)
        ref = RC.l2n(_feat(model.visual, x.numpy(), 16))
        row["verify"] = {}
        for p in paths:
            s = RC.session(p, a.threads)
            bs = 1 if p.name.endswith("_b1.onnx") else 16
            e = RC.l2n(np.concatenate([s.run(None, {"image": x[i:i + bs].numpy()})[0] for i in range(0, len(x), bs)]))
            c = (e * ref).sum(1)
            row["verify"][p.name] = {"n": a.verify, "cos_min": float(c.min()), "pass_cos_ge_0.9999": bool(c.min() >= 0.9999)}
            print(p.name, row["verify"][p.name])
    _append(out, "export.jsonl", row)
    if a.verify:
        assert all(v["pass_cos_ge_0.9999"] for v in row["verify"].values()), "ORT vs PyTorch cos < 0.9999"


@torch.no_grad()
def sink_attn_share(trunk, x, patches):
    """원래 모델에서 다른 토큰(query)이 각 패치(key)에 주는 attention 확률. 반환 {"b<l>"/"pool": {p: (H,) 사진·query 평균}}."""
    out = {}
    xt = _embed_tokens(trunk, x)
    N = xt.shape[1]
    for l, blk in enumerate(trunk.blocks):
        h = blk.norm1(xt)
        at = blk.attn
        q, k, _ = _split_heads(at.qkv(h), *h.shape[:2], 3, at.num_heads, at.head_dim)
        prob = ((at.q_norm(q) * at.scale) @ at.k_norm(k).transpose(-2, -1)).softmax(-1)      # (B,H,N,N)
        out[f"b{l}"] = {p: prob[:, :, [i for i in range(N) if i != p], p].mean((0, 2)) for p in patches}
        xt = xt + blk.drop_path1(blk.ls1(attn_forward(at, h)))
        xt = xt + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(xt))))
    pool = trunk.attn_pool
    xt = trunk.norm(xt)
    B = xt.shape[0]
    q = pool.q_norm(pool.q(pool.latent.expand(B, -1, -1)).reshape(B, pool.latent_len, pool.num_heads, pool.head_dim).transpose(1, 2))
    k, _ = pool_kv(pool, xt)
    prob = ((q * pool.scale) @ k.transpose(-2, -1)).softmax(-1)                                  # (B,H,L,N)
    out["pool"] = {p: prob[:, :, :, p].mean((0, 2)) for p in patches}
    return out


def cmd_maskcheck(a):
    """D-041 (a): 제자리 마스크 그래프 vs R1(토큰 삭제 + 평균 캐시) PyTorch FP32 임베딩 cos (dev 사진, Pets·Caltech 반씩)."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    orig = model.visual.trunk
    x = _dev_x(RC, a.n // 2, ["pets_dev", "caltech_dev"])
    ref = np.concatenate([np.load(RC.CACHE_DIR / f"{d}_ref_fp32.npy")[:a.n // 2] for d in ["pets_dev", "caltech_dev"]])
    model.visual.trunk = _make_trunk(S, RC, model, a, "mean")
    r1 = RC.l2n(_feat(model.visual, x, 16))
    model.visual.trunk = orig
    model.visual.trunk = _make_mask_trunk(RC, model, a)
    mk = RC.l2n(_feat(model.visual, x, 16))
    model.visual.trunk = orig
    c = (mk * r1).sum(1)
    res = {"n": int(len(x)), "mask_vs_r1_cos_min": float(c.min()), "mask_vs_r1_cos_mean": float(c.mean()),
           "mask_vs_r1_max_abs": float(np.abs(mk - r1).max()),
           "mask_vs_fp32ref_cos_min": float((mk * RC.l2n(ref)).sum(1).min()),
           "r1_vs_fp32ref_cos_min": float((r1 * RC.l2n(ref)).sum(1).min())}
    res["pass_a"] = res["mask_vs_r1_cos_min"] >= 0.99999
    row = _row(env, a, **res)
    _append(out, "maskcheck.jsonl", row)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    assert res["pass_a"], "D-041 (a) 실패: mask vs R1 cos < 0.99999"


def cmd_hcache(a):
    """h_c (G3b 마지막 시도): 보정 구간(train 고화질 calib_n장, seed)으로 비-sink 토큰 잔차 평균. dev는 쓰지 않는다."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    sinks = parse_sinks(a.sinks)
    from s2_quantize import calib_images                 # s2_quantize와 같은 보정 사진 (train, dev 제외)
    x = calib_images("highres", a.calib_n, a.seed)
    hc, n = build_hc(model.visual.trunk, x, sinks)
    assert n == a.calib_n, (n, a.calib_n)
    hp = _hc_path(RC, sinks, a.calib_n, a.seed)
    np.savez(hp, meta=json.dumps({"sinks": [list(s) for s in sinks], "calib": f"highres{a.calib_n}_s{a.seed}", "n": n,
                                  "commit": env["git"]["commit"]}), **hc)
    row = _row(env, a, file=hp.name, sha256=_sha256(hp), n=n,
               norms={k: round(float(np.linalg.norm(v)), 4) for k, v in hc.items()})
    _append(out, "hcache.jsonl", row)
    print(json.dumps(row["norms"]), hp.name, n)


def cmd_hcvcache(a):
    """h_c* (G3b 4번째 시도): hcache의 h_c를 보정 구간(train 고화질 calib_n장, seed)의 비-sink 토큰 LN 입력 분산 중앙값에 맞춘다. dev는 쓰지 않는다."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    sinks = parse_sinks(a.sinks)
    hp = _hc_path(RC, sinks, a.calib_n, a.seed)
    assert hp.exists(), f"{hp.name} 없음 — 먼저 hcache"
    z = np.load(hp)
    hc = {k: z[k] for k in z.files if k != "meta"}
    from s2_quantize import calib_images                 # s2_quantize와 같은 보정 사진 (train, dev 제외)
    x = calib_images("highres", a.calib_n, a.seed)
    vt, n, nk = build_vtarget(model.visual.trunk, x, sinks)
    assert n == a.calib_n, (n, a.calib_n)
    hcs, info = match_var(hc, vt)
    vp = _hcv_path(RC, sinks, a.calib_n, a.seed)
    np.savez(vp, meta=json.dumps({"sinks": [list(s) for s in sinks], "calib": f"highres{a.calib_n}_s{a.seed}", "n": n,
                                  "from": hp.name, "info": info, "commit": env["git"]["commit"]}), **hcs)
    row = _row(env, a, file=vp.name, sha256=_sha256(vp), hc_file=hp.name, hc_sha256=_sha256(hp), n=n,
               tokens_per_image=nk, info=info)
    _append(out, "hcvcache.jsonl", row)
    for k, v in info.items():
        print(k, v)


def cmd_ucache(a):
    """M2 u_c: 보정 구간(train 고화질 calib_n장, seed)에서 원래 모델 norm2 출력의 비-sink 토큰 평균. dev는 쓰지 않는다."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    sinks = parse_sinks(a.sinks)
    from s2_quantize import calib_images                 # s2_quantize와 같은 보정 사진 (train, dev 제외)
    x = calib_images("highres", a.calib_n, a.seed)
    uc, n = build_uc(model.visual.trunk, x, sinks)
    assert n == a.calib_n, (n, a.calib_n)
    up = _uc_path(RC, sinks, a.calib_n, a.seed)
    np.savez(up, meta=json.dumps({"sinks": [list(s) for s in sinks], "calib": f"highres{a.calib_n}_s{a.seed}", "n": n,
                                  "commit": env["git"]["commit"]}), **uc)
    row = _row(env, a, file=up.name, sha256=_sha256(up), n=n,
               norms={k: round(float(np.linalg.norm(v)), 4) for k, v in uc.items()})
    _append(out, "ucache.jsonl", row)
    print(json.dumps(row["norms"]), up.name, n)


@torch.no_grad()
def _trace_m2(rc, x, orig=False):
    """M2 그래프(또는 orig=True면 원래 모델)의 블록 l ≥ first: norm1·norm2 입력, norm2 출력 z, u_c − z, m⊙(u_c − z), z', fc2 입력, 블록 출력, 마지막 norm 입력."""
    trunk, caps = rc.t, {}
    h = _embed_tokens(trunk, x)
    for l, blk in enumerate(trunk.blocks):
        if l < rc.first:
            h = blk(h)
            continue
        caps[f"n1_{l}"] = h
        if orig:
            h = h + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(h))))
        else:
            h = h + blk.drop_path1(blk.ls1(attn_forward_mask(blk.attn, blk.norm1(h), *rc._kv(f"b{l}"))))
        caps[f"n2_{l}"] = h
        z = blk.norm2(h)
        caps[f"z_{l}"] = z
        if not orig:
            m, u = getattr(rc, f"zm_b{l}"), getattr(rc, f"uc_b{l}")
            d = u - z
            caps[f"sub_{l}"], caps[f"mul_{l}"] = d, m * d
            z = z + m * d
            caps[f"add_{l}"] = z
        f1 = blk.mlp.norm(blk.mlp.drop1(blk.mlp.act(blk.mlp.fc1(z))))
        caps[f"fc2in_{l}"] = f1
        h = h + blk.drop_path2(blk.ls2(blk.mlp.drop2(blk.mlp.fc2(f1))))
        caps[f"res_{l}"] = h
    caps["final"] = h
    return caps


def cmd_m2check(a):
    """M2 사전 확인 (a) vs R1 cos (b) sink 행 잔차 max ≤ 일반 ×2 (c) 새 연산 범위 ≤ 원래 같은 위치(norm2 출력) ×2 (d) LN 1/√(분산+ε) 일반 범위 안. dev 사진."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    orig = model.visual.trunk
    sinks = parse_sinks(a.sinks)
    x = _dev_x(RC, a.n // 2, ["pets_dev", "caltech_dev"])
    model.visual.trunk = _make_trunk(S, RC, model, a, "mean")
    r1 = RC.l2n(_feat(model.visual, x, 16))
    model.visual.trunk = orig
    rc = _make_mask_trunk(RC, model, a)
    model.visual.trunk = rc
    e = RC.l2n(_feat(model.visual, x, 16))
    model.visual.trunk = orig
    cm, co = {}, {}
    for i in range(0, len(x), 16):
        xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 16]))
        for k, v in _trace_m2(rc, xb).items():
            cm.setdefault(k, []).append(v)
        for k, v in _trace_m2(rc, xb, orig=True).items():
            co.setdefault(k, []).append(v)
    cm = {k: torch.cat(v) for k, v in cm.items()}
    co = {k: torch.cat(v) for k, v in co.items()}
    N = orig.patch_embed.num_patches
    other = [i for i in range(N) if i not in {p for p, _ in sinks}]
    rng = lambda t: (float(t.abs().max()), float(_range_ratio(t)[2]))      # (max, 99.9% 분위수)
    b, c, d = {}, {}, {}
    for l in range(rc.first, len(orig.blocks)):
        act = [p for p, s0 in sinks if s0 <= l]
        t = cm[f"res_{l}"].abs()
        sm, om = float(t[:, act].max()), float(t[:, other].max())
        b[f"res_{l}"] = {"sink_max": round(sm, 3), "other_max": round(om, 3), "ratio": round(sm / om, 4)}
        ref_max, ref_q = rng(co[f"z_{l}"])                                   # 원래 모델의 같은 위치 텐서 = norm2 출력
        for k in (f"z_{l}", f"sub_{l}", f"mul_{l}", f"add_{l}"):
            mx, q = rng(cm[k])
            c[k] = {"max": round(mx, 4), "p999": round(q, 4), "ref_max": round(ref_max, 4), "ref_p999": round(ref_q, 4),
                    "ok": bool(mx <= 2 * ref_max and q <= 2 * ref_q)}
        for k, eps in ((f"n1_{l}", orig.blocks[l].norm1.eps), (f"n2_{l}", orig.blocks[l].norm2.eps)):
            inv = 1 / torch.sqrt(cm[k].var(-1, unbiased=False) + eps)
            si, oi = inv[:, act], inv[:, other]
            d[k] = {"sink_min": round(float(si.min()), 4), "sink_max": round(float(si.max()), 4),
                    "other_min": round(float(oi.min()), 4), "other_max": round(float(oi.max()), 4),
                    "inside": bool(si.min() >= oi.min() and si.max() <= oi.max())}
    inv = 1 / torch.sqrt(cm["final"].var(-1, unbiased=False) + orig.norm.eps)
    act = [p for p, _ in sinks]
    d["final"] = {"sink_min": round(float(inv[:, act].min()), 4), "sink_max": round(float(inv[:, act].max()), 4),
                  "other_min": round(float(inv[:, other].min()), 4), "other_max": round(float(inv[:, other].max()), 4),
                  "inside": bool(inv[:, act].min() >= inv[:, other].min() and inv[:, act].max() <= inv[:, other].max())}
    cos = (e * r1).sum(1)
    res = {"n": int(len(x)), "a_cos_min_vs_r1": float(cos.min()), "pass_a": bool(cos.min() >= 0.99999),
           "b": b, "pass_b": all(v["ratio"] <= 2 for v in b.values()),
           "c": c, "pass_c_ranges": all(v["ok"] for v in c.values()),
           "d": d, "pass_d": all(v["inside"] for v in d.values())}
    row = _row(env, a, **res)
    _append(out, "m2check.jsonl", row)
    print("(a)", round(float(cos.min()), 7), res["pass_a"], " (b)", res["pass_b"], "최대 비율", max(v["ratio"] for v in b.values()),
          " (c 범위)", res["pass_c_ranges"], "최대 max비", round(max(v["max"] / v["ref_max"] for v in c.values()), 3),
          "최대 p99.9비", round(max(v["p999"] / max(v["ref_p999"], 1e-12) for v in c.values()), 3),
          " (d)", res["pass_d"], "sink 1/√ 최대", max(v["sink_max"] for v in d.values()), "일반 최대", max(v["other_max"] for v in d.values()))


def cmd_m2lnrange(a):
    """M2 확인 (e): 보정 구간(train 고화질 calib_n장, seed — DFC limvals와 같은 사진)에서 M2 FP32와 원래 FP32의 LayerNorm 1/√(분산+ε)
    전체 토큰 범위 (블록 first~11 norm1·norm2, 마지막 norm). DFC `conv_var_inv_layer_normalization*` limvals와 대조한다."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    orig = model.visual.trunk
    rc = _make_mask_trunk(RC, model, a)
    from s2_quantize import calib_images                 # s2_quantize와 같은 보정 사진 (train, dev 제외)
    x = calib_images("highres", a.calib_n, a.seed)
    acc = {}
    for i in range(0, len(x), 16):
        xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 16]))
        for tag, o in (("m2", False), ("orig", True)):
            cap = _trace_m2(rc, xb, orig=o)
            for k in [k for k in cap if k.startswith(("n1_", "n2_"))] + ["final"]:
                l = k.split("_")[-1]
                eps = orig.norm.eps if k == "final" else (orig.blocks[int(l)].norm1.eps if k.startswith("n1") else orig.blocks[int(l)].norm2.eps)
                inv = 1 / torch.sqrt(cap[k].var(-1, unbiased=False) + eps)
                d = acc.setdefault(tag, {}).setdefault(k, [float("inf"), 0.0])
                d[0], d[1] = min(d[0], float(inv.min())), max(d[1], float(inv.max()))
    res = {tag: {k: {"min": round(v[0], 4), "max": round(v[1], 4)} for k, v in d.items()} for tag, d in acc.items()}
    row = _row(env, a, n=int(len(x)), inv_sqrt_range=res)
    _append(out, "m2lnrange.jsonl", row)
    for k in res["m2"]:
        print(f"{k:7s} M2 [{res['m2'][k]['min']:.4f}, {res['m2'][k]['max']:.4f}]   원래 [{res['orig'][k]['min']:.4f}, {res['orig'][k]['max']:.4f}]")


@torch.no_grad()
def _trace_mask(rc, x):
    """MaskRegCacheTrunk.forward와 같은 순서로, 블록 l ≥ first의 norm1·norm2 입력, fc2 입력, 블록 출력과 마지막 norm 입력을 꺼낸다."""
    trunk, caps = rc.t, {}
    h = _embed_tokens(trunk, x)
    for l, blk in enumerate(trunk.blocks):
        if l < rc.first:
            h = blk(h)
            continue
        h = rc._fill(h, f"b{l}")
        caps[f"n1_{l}"] = h
        h = h + blk.drop_path1(blk.ls1(attn_forward_mask(blk.attn, blk.norm1(h), *rc._kv(f"b{l}"))))
        caps[f"n2_{l}"] = h
        f1 = blk.mlp.norm(blk.mlp.drop1(blk.mlp.act(blk.mlp.fc1(blk.norm2(h)))))
        caps[f"fc2in_{l}"] = f1
        h = h + blk.drop_path2(blk.ls2(blk.mlp.drop2(blk.mlp.fc2(f1))))
        if not rc.hfill:
            h = h * getattr(rc, f"hm_b{l}")
        caps[f"res_{l}"] = h
    if rc.hfill:
        h = rc._fill(h, "final")
    caps["final"] = h
    return caps


def cmd_hccheck(a):
    """G3b 마지막 시도 사전 확인 (a) h_c 마스크 vs R1 cos (b) sink 행 거대 값 재발 여부 (c) LN 분해 1/√(분산+ε)의 sink 행 범위. zero 방식도 비교로 계산."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    orig = model.visual.trunk
    sinks = parse_sinks(a.sinks)
    x = _dev_x(RC, a.n // 2, ["pets_dev", "caltech_dev"])
    model.visual.trunk = _make_trunk(S, RC, model, a, "mean")
    r1 = RC.l2n(_feat(model.visual, x, 16))
    model.visual.trunk = orig
    res = {"n": int(len(x))}
    for fill in a.fills:
        a.hfill = fill
        rc = _make_mask_trunk(RC, model, a)
        model.visual.trunk = rc
        e = RC.l2n(_feat(model.visual, x, 16))
        model.visual.trunk = orig
        caps = {}
        for i in range(0, len(x), 16):
            for k, v in _trace_mask(rc, torch.as_tensor(np.ascontiguousarray(x[i:i + 16]))).items():
                caps.setdefault(k, []).append(v)
        caps = {k: torch.cat(v) for k, v in caps.items()}
        b, c = {}, {}
        for l in range(rc.first, len(orig.blocks)):
            act = [p for p, s0 in sinks if s0 <= l]
            other = [i for i in range(orig.patch_embed.num_patches) if i not in {p for p, _ in sinks}]
            for k in (f"res_{l}", f"fc2in_{l}"):
                t = caps[k].abs()
                sm, om = float(t[:, act].max()), float(t[:, other].max())
                b[k] = {"sink_max": round(sm, 3), "other_max": round(om, 3), "ratio": round(sm / om, 4)}
            for k, eps in ((f"n1_{l}", orig.blocks[l].norm1.eps), (f"n2_{l}", orig.blocks[l].norm2.eps)):
                inv = 1 / torch.sqrt(caps[k].var(-1, unbiased=False) + eps)                # (B,N)
                si, oi = inv[:, act], inv[:, other]
                c[k] = {"sink_min": round(float(si.min()), 4), "sink_max": round(float(si.max()), 4),
                        "other_min": round(float(oi.min()), 4), "other_max": round(float(oi.max()), 4),
                        "inside": bool(si.min() >= oi.min() and si.max() <= oi.max())}
        inv = 1 / torch.sqrt(caps["final"].var(-1, unbiased=False) + orig.norm.eps)
        act = [p for p, _ in sinks]
        other = [i for i in range(orig.patch_embed.num_patches) if i not in set(act)]
        c["final"] = {"sink_min": round(float(inv[:, act].min()), 4), "sink_max": round(float(inv[:, act].max()), 4),
                      "other_min": round(float(inv[:, other].min()), 4), "other_max": round(float(inv[:, other].max()), 4),
                      "inside": bool(inv[:, act].min() >= inv[:, other].min() and inv[:, act].max() <= inv[:, other].max())}
        cos = (e * r1).sum(1)
        res[fill] = {"a_cos_min_vs_r1": float(cos.min()), "pass_a": bool(cos.min() >= 0.99999),
                     "b": b, "pass_b": all(v["ratio"] <= 2 for v in b.values()),
                     "c": c, "pass_c": all(v["inside"] for v in c.values())}
        print(fill, "(a)", round(float(cos.min()), 7), res[fill]["pass_a"], " (b)", res[fill]["pass_b"],
              "최대 비율", max(v["ratio"] for v in b.values()), " (c)", res[fill]["pass_c"],
              "sink 1/√ 최대", max(v["sink_max"] for v in c.values()), "일반 최대", max(v["other_max"] for v in c.values()))
    a.hfill = a.fills[0]
    row = _row(env, a, **res)
    _append(out, "hccheck.jsonl", row)


def cmd_attn(a):
    """(기록용, 판정에 안 씀) 블록마다 다른 토큰이 sink 패치에 주는 attention 비중 — 헤드 평균·최대, dev 사진 (원래 FP32)."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model, _ = _load_visual(S, a.threads)
    trunk, sinks = model.visual.trunk, parse_sinks(a.sinks)
    patches = [p for p, _ in sinks]
    x = torch.as_tensor(_dev_x(RC, a.n // 2, ["pets_dev", "caltech_dev"]))     # stats와 같은 사진 구성
    acc, nb = {}, 0
    for i in range(0, len(x), 16):
        for key, d in sink_attn_share(trunk, x[i:i + 16], patches).items():
            for p, v in d.items():
                acc.setdefault(key, {}).setdefault(p, []).append((v * len(x[i:i + 16])).double())
        nb += len(x[i:i + 16])
    assert nb == len(x)
    share = {key: {str(p): {"head_mean": round(float(torch.stack(v).sum(0).div(nb).mean()), 6),
                            "head_max": round(float(torch.stack(v).sum(0).div(nb).max()), 6)} for p, v in d.items()}
             for key, d in acc.items()}
    uni = 1.0 / trunk.patch_embed.num_patches
    row = _row(env, a, n=int(len(x)), uniform=round(uni, 6), share=share)
    _append(out, "attn_share.jsonl", row)
    print(f"균등 분배 기준 1/{trunk.patch_embed.num_patches} = {uni:.4f}")
    for key in share:
        print(key, {p: (v["head_mean"], v["head_max"]) for p, v in share[key].items()})


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data=True):
        p.add_argument("--sinks", default=DEFAULT_SINKS, help="패치번호@시작블록, 쉼표로 (기본 55@5,13@9)")
        p.add_argument("--threads", type=int, default=4)
        if data:
            p.add_argument("--calib-n", type=int, default=256)
            p.add_argument("--seed", type=int, default=0)
        return p

    p = common(sub.add_parser("selftest"), data=False)
    p.add_argument("--out", default=str(HERE / ".selftest_tmp"))
    p.set_defaults(func=cmd_selftest)
    common(sub.add_parser("cache")).set_defaults(func=cmd_cache)
    p = common(sub.add_parser("stats"))
    p.add_argument("--n", type=int, default=64, help="dev 사진 수 (Pets·Caltech 반씩, G3_why와 같음)")
    p.set_defaults(func=cmd_stats)
    p = common(sub.add_parser("fp32"))
    p.add_argument("--k", type=int, default=200, help="데이터셋마다 앞 k장 (스크리닝)")
    p.add_argument("--modes", nargs="+", default=["oracle", "mean", "none"], choices=["oracle", "mean", "none"])
    p.set_defaults(func=cmd_fp32)
    p = common(sub.add_parser("export"))
    p.add_argument("--mode", default="mean", choices=["mean", "none"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--graph", default="concat", choices=["concat", "mask"],
                   help="concat = 토큰 삭제 + K·V Concat (기본, R1). mask = 토큰 196개 유지 + 제자리 마스크 (D-041)")
    p.add_argument("--mask-form", default="full", choices=["full", "bcast"],
                   help="--graph mask의 상수 모양. full = 196토큰 크기 상수(기본, 12cbe01과 같음), bcast = m·(K_c−K) broadcast")
    p.add_argument("--mask-site", default="residual", choices=["residual", "mlp"],
                   help="--graph mask에서 sink 행을 다루는 곳. residual = 잔차를 0/h_c로 채움(기본, 이전과 같음), mlp = 잔차는 그대로, MLP 입력만 u_c (M2, ucache)")
    p.add_argument("--hfill", default="zero", choices=["zero", "mean", "meanvar"],
                   help="--graph mask에서 sink 잔차를 채우는 값. zero = 0(기본, 이전과 같음), mean = h_c (hcache), meanvar = 분산 맞춤 h_c* (hcvcache)")
    p.set_defaults(func=cmd_export)
    p = common(sub.add_parser("maskcheck"))
    p.add_argument("--n", type=int, default=64, help="dev 사진 수 (Pets·Caltech 반씩)")
    p.add_argument("--mask-form", default="full", choices=["full", "bcast"])
    p.set_defaults(func=cmd_maskcheck)
    common(sub.add_parser("hcache")).set_defaults(func=cmd_hcache)
    p = common(sub.add_parser("hccheck"))
    p.add_argument("--n", type=int, default=64, help="dev 사진 수 (Pets·Caltech 반씩)")
    p.add_argument("--mask-form", default="bcast", choices=["full", "bcast"])
    p.add_argument("--fills", nargs="+", default=["mean", "zero"], choices=["meanvar", "mean", "zero"],
                   help="확인할 채움 방식 (첫 번째가 주 구성). 기본 mean zero = 3번째 시도와 같음")
    p.set_defaults(func=cmd_hccheck, hfill="mean")
    common(sub.add_parser("hcvcache")).set_defaults(func=cmd_hcvcache)
    common(sub.add_parser("ucache")).set_defaults(func=cmd_ucache)
    p = common(sub.add_parser("m2check"))
    p.add_argument("--n", type=int, default=64, help="dev 사진 수 (Pets·Caltech 반씩)")
    p.set_defaults(func=cmd_m2check, mask_form="bcast", mask_site="mlp", hfill="zero")
    common(sub.add_parser("m2lnrange")).set_defaults(func=cmd_m2lnrange, mask_form="bcast", mask_site="mlp", hfill="zero")
    p = common(sub.add_parser("attn"))
    p.add_argument("--n", type=int, default=64, help="dev 사진 수 (Pets·Caltech 반씩, stats와 같음)")
    p.set_defaults(func=cmd_attn)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
