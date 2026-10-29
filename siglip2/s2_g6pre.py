"""
s2_g6pre.py — Hailo Zoo SigLIP 두 모델(B/32-256, SigLIP1 B/16)에 M2rot이 들어맞는지 FP32 사전 확인 (G6-pre)
=========================================================================================================

[설계·판정]  `results/siglip2/G6_pre/README.md` (결과 전에 고정). 모델 변경·양자화·컴파일 없음, FP32만.

[모델]  HF `SiglipModel` (attn_implementation="eager"). 비전 쪽 forward를 이 파일에서 직접 풀어 쓴다 (`hf_forward`):
        잔차 흐름·attention 확률·K/V를 꺼내고, M2(s2_regcache.MaskRegCacheTrunk --mask-site mlp)와 같은 치환을 넣기 위해서다.
        풀어 쓴 forward가 HF 원래 출력과 같은지 매번 assert한다 (cos 최소 ≥ 0.99999).
  ref   google/siglip2-base-patch16-224  (우리 timm/ViT-B-16-SigLIP2와 같은 가중치인지 ident에서 확인)
  b32   google/siglip2-base-patch32-256  (Zoo siglip2_b_32_256_image_encoder)
  s1b16 google/siglip-base-patch16-224   (Zoo siglip_b_16_image_encoder)

[M2 치환]  s2_regcache.MaskRegCacheTrunk(form=bcast, mask_site=mlp)와 같은 범위 (README 2-4):
  sinks [(p, k, u)] (표기 p@k:u, p@b = p@b:b): 블록 l ≥ k의 self-attention에서 위치 p의 K·V를 보정 평균 상수로,
  블록 l ≥ u의 sink 행 norm2 출력을 u_c[l]로, MAP 헤드 kv에서 모든 sink 위치의 K·V를 상수로. 잔차는 그대로.
  1단계(M2rot과 같은 범위)는 k = u = b. 2단계 통일 규칙(README 12-2)은 `unify`가 k·u를 정한다.

[하위 명령]  (siglip2/ 폴더에서, RN50_RESULTS=../results/siglip2/G6_pre)
  cache   원본 parquet에서 Zoo 전처리(PIL BICUBIC 바로 맞춤, crop 없음, x/127.5−1) 캐시: 보정 256장(시드 0, 기존 calib_images와 같은 사진) + dev 6×200
  ident   (1) Zoo ONNX vs HF  (2) ref HF vs 우리 timm  (3) 풀어 쓴 forward vs HF  — 32장
  analyze 거대 활성값·sink 통계 (보정 앞 128장) + README 2-4 규칙으로 sink 찾기
  m2      sink K·V 평균 캐시 + u_c (보정 256장) → dev 6×200에서 원래 vs M2 / K·V만 / oracle cos
  m2xchk  ref만: 기존 timm 상수 파일과 timm 전처리 캐시로 내 HF M2 == timm MaskRegCacheTrunk 인지
  top1    HF 텍스트 인코더로 분류기 → dev 6×200 FP32 top-1 (원래, M2)
  confset (2단계) 확인 세트 6×200 (시드 1, dev·보정 후보·test와 SHA-1로 겹침 없음 확인) → Zoo 전처리 캐시
  unify   (2단계) sink 토큰이 high-norm이 되는 첫 지점(입력 / attn 뒤 / MLP 뒤)과 통일 규칙의 p@k:u (README 12-1·12-2)
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "6"                       # G5c와 같은 PC에서 돌므로 6스레드 제한 (시간 측정 없음)

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2common as S                           # noqa: E402
from scripts import env_info                   # noqa: E402

RC, RD = S.RC, S.RD
torch.set_num_threads(6)
torch.set_grad_enabled(False)

MODELS = {
    "ref": dict(hf="google/siglip2-base-patch16-224", size=224),
    "b32": dict(hf="google/siglip2-base-patch32-256", size=256),
    "s1b16": dict(hf="google/siglip-base-patch16-224", size=224),
}
ZOO_ONNX = {
    "b32": ("dl_b32/siglip2-base-patch32-256_vision_encoder.sim.onnx", "1393"),
    "s1b16": ("dl_b16/siglip_base_patch16_224_vision_encoder.onnx", "1725"),
}
WORK = Path(os.environ.get("G6PRE_WORK", Path.home() / "siglip2_work" / "g6pre"))
HR = ["pets", "flowers", "cars", "food", "caltech"]
DEV_SETS = ["cifar_A"] + [f"{n}_dev" for n in HR]
DEV_K = 200
CONF_SETS = ["conf_cifar"] + [f"conf_{n}" for n in HR]   # 2단계 확인 세트 (README 12-3)
CONF_SEED = 1
CIFAR_CONF_RANGE = (47000, 50000)                         # 보정 [0,45000)·dev A·B [45000,47000) 밖, 어떤 규칙도 쓰지 않는 구간
CAL_N, AN_N, SEED = 256, 128, 0
HIGH_NORM = 5.0                                # G3_why: 토큰 노름 > 중앙값의 5배
FIX = 0.90                                     # README 2-4·4절: 고정성 문턱


def cache_dir(size):
    d = WORK / f"cache_zoo{size}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def zoo_pre(img, size):
    """Zoo `classification_preprocessing.clip`: PIL BICUBIC으로 (size,size)에 바로 맞춤(crop 없음) → uint8 → HEF 안 (x−127.5)/127.5."""
    from PIL import Image
    a = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)
    return (a.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def l2n(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def cosr(a, b):
    return (l2n(np.asarray(a, np.float64)) * l2n(np.asarray(b, np.float64))).sum(-1)


def git_info():
    return env_info.collect()["git"]


def out_dir():
    d = S.require_results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def append(name, row):
    row = {"date": datetime.now().isoformat(timespec="seconds"), **git_info(), **row}
    with open(out_dir() / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:2000], flush=True)


# ---------------------------------------------------------------- 데이터 (README 2-3)

def _raw_highres(name):
    """s2_build_dev.build와 같은 순서로 (dev 원본 바이트 목록, dev 정답, 보정 후보 원본 바이트 목록)."""
    cfg = RD.HIGHRES[name]
    p_dev, p_cal = RD.DEV_N / cfg["n_pool"], RD.CAL_N / cfg["n_pool"]
    dev, ys, cal = [], [], []
    for b, lab in RD.parquet_rows(cfg["pool"], False):
        u = RD.u_of(b)
        if u < p_dev:
            dev.append(b)
            ys.append(lab)
        elif u < p_dev + p_cal:
            cal.append(b)
    return dev, ys, cal


def cmd_cache(a):
    from PIL import Image
    import hashlib
    _, timm_pre, _ = S.load_model()
    old = RC.CACHE_DIR                                   # 기존 timm 전처리 캐시 (사진 대조용, 읽기만)
    pool_raw, pool_name = [], []
    dev_raw = {}
    for name in sorted(HR):                              # calib_images는 sorted(glob("calib_*")) = 이름순
        dev, ys, cal = _raw_highres(name)
        sha_old = (old / f"{name}_dev_sha1.txt").read_text().split()
        sha_new = [hashlib.sha1(b).hexdigest() for b in dev]
        assert sha_new == sha_old, f"{name} dev 사진 목록이 기존 캐시와 다름"
        y_old = np.load(old / f"{name}_dev_y.npy")
        assert (np.array(ys) == y_old).all()
        dev_raw[f"{name}_dev"] = (dev[:DEV_K], np.array(ys[:DEV_K]))
        n_old = len(np.load(old / f"calib_{name}_f32.npy", mmap_mode="r"))
        assert len(cal) == n_old, (name, len(cal), n_old)
        pool_raw += cal
        pool_name += [name] * len(cal)
        print(f"  {name}: dev {len(dev)} (sha1 일치), 보정 후보 {len(cal)}", flush=True)
    # s2_quantize.calib_images("highres", 256, 0)와 같은 무작위 선택
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool_raw), CAL_N, replace=False)
    perm = rng.permutation(CAL_N)
    sel = [int(idx[i]) for i in perm]
    # 대조: 고른 사진을 timm 전처리하면 기존 calib_images 결과와 같아야 한다
    pool_old = np.concatenate([np.load(f) for f in sorted(old.glob("calib_*_f32.npy"))])
    ref_old = pool_old[idx][perm]
    mine = np.stack([timm_pre(Image.open(io.BytesIO(pool_raw[i])).convert("RGB")).numpy() for i in sel])
    d = float(np.abs(mine - ref_old).max())
    assert d < 1e-5, f"보정 사진 선택이 기존과 다름 (최대 차이 {d})"
    print(f"  보정 256장: 기존 calib_images(highres,256,0)와 같은 사진·순서 (timm 전처리 최대 차이 {d:.2e})", flush=True)
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=False)
    s0 = RC.VAL_SPLITS["A"]
    cif = [ds[i][0] for i in range(s0, s0 + DEV_K)]
    ycif = np.array([ds.targets[i] for i in range(s0, s0 + DEV_K)])
    assert (ycif == np.load(old / "cifar_A_y.npy")[:DEV_K]).all()
    for size in (224, 256):
        cd = cache_dir(size)
        xc = np.stack([zoo_pre(Image.open(io.BytesIO(pool_raw[i])), size) for i in sel]).astype(np.float32)
        np.save(cd / "calib256_s0_x.npy", xc)
        (cd / "calib256_s0_src.txt").write_text("\n".join(f"{pool_name[i]} {hashlib.sha1(pool_raw[i]).hexdigest()}"
                                                          for i in sel) + "\n")
        np.save(cd / "cifar_A_x.npy", np.stack([zoo_pre(im, size) for im in cif]).astype(np.float32))
        np.save(cd / "cifar_A_y.npy", ycif)
        for key, (bs, ys) in dev_raw.items():
            np.save(cd / f"{key}_x.npy", np.stack([zoo_pre(Image.open(io.BytesIO(b)), size) for b in bs]).astype(np.float32))
            np.save(cd / f"{key}_y.npy", ys)
        for f in sorted(cd.glob("*_x.npy")):
            x = np.load(f, mmap_mode="r")
            assert x.shape[1:] == (3, size, size) and x.min() >= -1 and x.max() <= 1, (f, x.shape)
        print(f"  zoo{size} 캐시: {cd}", flush=True)
    append("cache.jsonl", {"what": "cache", "sizes": [224, 256], "calib_n": CAL_N, "seed": SEED, "dev_k": DEV_K,
                           "calib_match_max_diff": d, "work": str(WORK)})


def load_x(size, key):
    return np.load(cache_dir(size) / f"{key}_x.npy")


def load_y(size, key):
    return np.load(cache_dir(size) / f"{key}_y.npy")


def ident32(size):
    """동일성 확인용 32장: Pets dev 앞 16 + Caltech dev 앞 16 (Zoo 전처리)."""
    return np.concatenate([load_x(size, "pets_dev")[:16], load_x(size, "caltech_dev")[:16]])


# ---------------------------------------------------------------- 모델

def load_hf(name):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODELS[name]["hf"], attn_implementation="eager", dtype=torch.float32).eval()
    return m


def vis_of(m):
    return m.vision_model


def _heads(t, H):
    B, N, C = t.shape
    return t.view(B, N, H, C // H).transpose(1, 2)


def _pool_qkv(head, x):
    """MAP 헤드 (nn.MultiheadAttention): probe query와 x의 k·v를 헤드로 나눠서."""
    mha = head.attention
    C, H = mha.embed_dim, mha.num_heads
    W, bb = mha.in_proj_weight, mha.in_proj_bias
    B = x.shape[0]
    q = head.probe.expand(B, -1, -1) @ W[:C].T + bb[:C]
    k = x @ W[C:2 * C].T + bb[C:2 * C]
    v = x @ W[2 * C:].T + bb[2 * C:]
    return _heads(q, H), _heads(k, H), _heads(v, H)


def _replace(k, v, pos, kc, vc):
    """위치 pos의 K·V를 상수 (H,S,d)로 바꿔 끼운다 (토큰 수 유지)."""
    k, v = k.clone(), v.clone()
    for j, p in enumerate(pos):
        k[:, :, p] = kc[:, j]
        v[:, :, p] = vc[:, j]
    return k, v


def hf_forward(vis, x, sinks=(), kvc=None, uc=None, rec=None):
    """HF SiglipVisionModel과 같은 계산을 풀어 쓴 것.
    sinks [(p, k, u)], kvc {"b<l>"/"pool": (K (H,S,d), V)} (블록 l에서 k ≤ l인 sink 순서), uc {"b<l>": (C,)} 또는 None (K·V만).
    rec(dict)가 있으면 블록별 기록: h (블록 출력), attn mass (B,H,N), sink K·V, norm2 출력, 세 지점 토큰 노름(pts)."""
    h = vis.embeddings(x)
    layers = vis.encoder.layers
    for l, ly in enumerate(layers):
        at = ly.self_attn
        H = at.num_heads
        act = [p for p, kb, _ in sinks if kb <= l]
        act_u = [p for p, _, ub in sinks if ub <= l]
        if rec is not None and rec.get("pts"):
            rec.setdefault(f"n_in_b{l}", []).append(h.norm(dim=-1))
        z = ly.layer_norm1(h)
        q, k, v = _heads(at.q_proj(z), H), _heads(at.k_proj(z), H), _heads(at.v_proj(z), H)
        if rec is not None and "kv_pos" in rec:
            rec.setdefault(f"k_b{l}", []).append(k[:, :, rec["kv_pos"]].clone())
            rec.setdefault(f"v_b{l}", []).append(v[:, :, rec["kv_pos"]].clone())
        if act and kvc is not None:
            k, v = _replace(k, v, act, *kvc[f"b{l}"])
        a = torch.softmax((q @ k.transpose(-1, -2)) * at.scale, dim=-1)
        if rec is not None and rec.get("mass"):
            rec.setdefault(f"mass_b{l}", []).append(a.mean(2))
        o = (a @ v).transpose(1, 2).reshape(h.shape)
        h = h + at.out_proj(o)
        if rec is not None and rec.get("pts"):
            rec.setdefault(f"n_attn_b{l}", []).append(h.norm(dim=-1))
        z2 = ly.layer_norm2(h)
        if act_u and uc is not None:
            z2 = z2.clone()
            for p in act_u:
                z2[:, p] = uc[f"b{l}"]
        if rec is not None and rec.get("z2_sink"):
            rec.setdefault(f"z2_b{l}", []).append(z2.clone())
        h = h + ly.mlp(z2)
        if rec is not None and rec.get("h"):
            rec.setdefault(f"h_b{l}", []).append(h.clone())
        if rec is not None and rec.get("pts"):
            rec.setdefault(f"n_mlp_b{l}", []).append(h.norm(dim=-1))
    h = vis.post_layernorm(h)
    head = vis.head
    q, k, v = _pool_qkv(head, h)
    if rec is not None and "kv_pos" in rec:
        rec.setdefault("k_pool", []).append(k[:, :, rec["kv_pos"]].clone())
        rec.setdefault("v_pool", []).append(v[:, :, rec["kv_pos"]].clone())
    if sinks and kvc is not None:
        k, v = _replace(k, v, [s[0] for s in sinks], *kvc["pool"])
    a = torch.softmax((q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1]), dim=-1)
    if rec is not None and rec.get("mass"):
        rec.setdefault("mass_pool", []).append(a.mean(2))
    o = (a @ v).transpose(1, 2).reshape(h.shape[0], 1, -1)
    o = head.attention.out_proj(o)
    o = o + head.mlp(head.layernorm(o))
    return o[:, 0]


def batched(fn, x, bs=16):
    return torch.cat([fn(torch.as_tensor(np.ascontiguousarray(x[i:i + bs]))) for i in range(0, len(x), bs)]).numpy()


def check_forward(vis, x):
    """풀어 쓴 forward == HF pooler_output (README 3절)."""
    a = batched(lambda t: vis(pixel_values=t).pooler_output, x)
    b = batched(lambda t: hf_forward(vis, t), x)
    c = cosr(a, b)
    assert c.min() >= 0.99999, f"풀어 쓴 forward가 HF와 다름: cos 최소 {c.min()}"
    return float(c.min()), float(np.abs(a - b).max())


# ---------------------------------------------------------------- ident (요청 1·4)

def cmd_ident(a):
    import onnxruntime as ort
    res = {}
    for name in ("b32", "s1b16", "ref"):
        size = MODELS[name]["size"]
        vis = vis_of(load_hf(name))
        x = ident32(size)
        fmin, fdiff = check_forward(vis, x)
        hf = batched(lambda t: vis(pixel_values=t).pooler_output, x)
        row = {"what": "ident", "model": name, "hf": MODELS[name]["hf"], "n": len(x), "input": f"zoo{size}",
               "unrolled_vs_hf_cos_min": fmin, "unrolled_vs_hf_maxabs": fdiff}
        if name in ZOO_ONNX:
            rel, out_name = ZOO_ONNX[name]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 6
            sess = ort.InferenceSession(str(WORK / rel), so, providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            zo = np.concatenate([sess.run([out_name], {iname: x[i:i + 1]})[0] for i in range(len(x))])
            c = cosr(zo, hf)
            row.update(zoo_onnx=rel, zoo_out=out_name, zoo_vs_hf_cos_mean=float(c.mean()), zoo_vs_hf_cos_min=float(c.min()),
                       zoo_vs_hf_maxabs=float(np.abs(zo - hf).max()), ort=ort.__version__)
        else:
            model, _, _ = S.load_model()
            xt = np.concatenate([np.load(RC.CACHE_DIR / "pets_dev_x.npy", mmap_mode="r")[:16],
                                 np.load(RC.CACHE_DIR / "caltech_dev_x.npy", mmap_mode="r")[:16]]).astype(np.float32)
            tm = batched(lambda t: model.encode_image(t), xt)
            hf_t = batched(lambda t: vis(pixel_values=t).pooler_output, xt)
            c = cosr(tm, hf_t)
            row.update(timm=S.MODEL_ID, timm_input="기존 timm 전처리 dev 캐시 (pets 16 + caltech 16)",
                       timm_vs_hf_cos_mean=float(c.mean()), timm_vs_hf_cos_min=float(c.min()),
                       timm_vs_hf_maxabs=float(np.abs(tm - hf_t).max()))
        res[name] = row
        append("ident.jsonl", row)


# ---------------------------------------------------------------- 구조 (요청 2)

def structure(name, m):
    vis = vis_of(m)
    cfg = vis.config
    ly = vis.encoder.layers[0]
    n_ln = sum(isinstance(t, torch.nn.LayerNorm) for t in vis.modules())
    return {"model": name, "hidden": cfg.hidden_size, "blocks": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
            "mlp": cfg.intermediate_size, "patch": cfg.patch_size, "image": cfg.image_size,
            "tokens": vis.embeddings.num_patches, "cls_token": hasattr(vis.embeddings, "class_embedding"),
            "pos_embed": f"학습 절대 위치 nn.Embedding({vis.embeddings.position_embedding.num_embeddings}), 패치 임베딩에 더함",
            "pre_ln": "layer_norm1 → attn → +잔차, layer_norm2 → MLP → +잔차 (HF SiglipEncoderLayer.forward)",
            "n_layernorm": n_ln, "map_head": hasattr(vis, "head") and vis.head is not None,
            "map_head_latent": int(vis.head.probe.shape[1]), "ln_eps": cfg.layer_norm_eps,
            "act": cfg.hidden_act, "q_k_norm": False}


# ---------------------------------------------------------------- analyze (요청 2·3·4, 규칙 2-4)

def cmd_analyze(a):
    name = a.model
    m = load_hf(name)
    vis = vis_of(m)
    size = MODELS[name]["size"]
    x = load_x(size, "calib256_s0")[:AN_N]
    assert len(x) == AN_N
    check_forward(vis, x[:16])
    st = structure(name, m)
    append("structure.jsonl", {"what": "structure", **st})
    L, N = st["blocks"], st["tokens"]
    blocks, norms, mass = [], [], {}
    for i in range(0, AN_N, 16):
        rec = {"h": True, "mass": True}
        hf_forward(vis, torch.as_tensor(x[i:i + 16]), rec=rec)
        for k, v in rec.items():
            if k.startswith("mass_"):
                mass.setdefault(k, []).append(torch.cat(v).numpy())
        norms.append(np.stack([torch.cat(rec[f"h_b{l}"]).norm(dim=-1).numpy() for l in range(L)], 1))   # (B,L,N)
        if i == 0:
            hs = [[] for _ in range(L)]
        for l in range(L):
            hs[l].append(torch.cat(rec[f"h_b{l}"]).abs().numpy().astype(np.float32))
    norms = np.concatenate(norms)                                         # (128, L, N)
    mass = {k: np.concatenate(v) for k, v in mass.items()}               # (128, H, N)
    hn_frac = np.zeros((L, N))
    for l in range(L):
        ab = np.concatenate(hs[l])                                         # (128, N, C)
        hs[l] = None
        med = float(np.median(ab))
        mx = float(ab.max())
        ch = ab.max(axis=(0, 1))
        tok = ab.max(axis=(0, 2))
        nrm = norms[:, l]
        rat_img = nrm / np.median(nrm, axis=1, keepdims=True)
        hn = rat_img > HIGH_NORM
        hn_frac[l] = hn.mean(0)
        top_tok_img = nrm.argmax(1)
        vals, cnts = np.unique(top_tok_img, return_counts=True)
        blocks.append({"block": l, "max_abs": round(mx, 3), "median_abs": round(med, 5), "ratio": round(mx / med, 1),
                       "top5_ch": [[int(c), round(float(ch[c]), 2)] for c in np.argsort(-ch)[:5]],
                       "top5_tok": [[int(t), round(float(tok[t]), 2)] for t in np.argsort(-tok)[:5]],
                       "tok_norm_ratio_max": round(float(rat_img.max()), 2),
                       "tok_norm_ratio_med_img": round(float(np.median(rat_img.max(1))), 2),
                       "high_norm_imgs": int(hn.any(1).sum()), "high_norm_per_img_mean": round(float(hn.sum(1).mean()), 2),
                       "high_norm_pos": [[int(p), int(hn[:, p].sum())] for p in np.argsort(-hn.sum(0))[:5] if hn[:, p].sum() > 0],
                       "argmax_norm_tok_mode": [int(vals[cnts.argmax()]), int(cnts.max())]})
    # sink: 고정성 = 헤드 평균 질량의 상위 2개 안에 p가 드는 사진 비율
    sink_blocks = []
    keys = [f"mass_b{l}" for l in range(L)] + ["mass_pool"]
    fix, fixh = {}, {}
    for k in keys:
        mh = mass[k].mean(1)                                               # (128, N)
        top2 = np.argsort(-mh, 1)[:, :2]
        frac = np.array([(top2 == p).any(1).mean() for p in range(N)])
        fix[k] = frac
        topp = np.argsort(-frac)[:3]
        per_head = []
        fixh[k] = np.zeros((mass[k].shape[1], N))
        for hh in range(mass[k].shape[1]):
            t2 = np.argsort(-mass[k][:, hh], 1)[:, :2]
            fr = np.array([(t2 == p).any(1).mean() for p in range(N)])
            fixh[k][hh] = fr
            pb = int(fr.argmax())
            per_head.append([pb, round(float(fr[pb]), 3), round(float(mass[k][:, hh, pb].mean()), 4)])
        sink_blocks.append({"key": k[5:], "top_fix": [[int(p), round(float(frac[p]), 3), round(float(mh[:, p].mean()), 4)]
                                                      for p in topp],
                            "uniform": round(1 / N, 5), "per_head_best": per_head})
    # README 2-4 규칙 (6절 조정: attention 고정성은 헤드별, 블록 l > b의 어느 헤드든 ≥ 90%)
    sinks = []
    for p in range(N):
        bs = [l for l in range(L) if hn_frac[l, p] >= FIX]
        if not bs:
            continue
        b = bs[0]
        att = [[l, int(hh)] for l in range(b + 1, L) for hh in np.where(fixh[f"mass_b{l}"][:, p] >= FIX)[0]]
        att_avg = [l for l in range(b + 1, L) if fix[f"mass_b{l}"][p] >= FIX]
        if att:
            sinks.append({"p": p, "b": b, "high_norm_frac_b": round(float(hn_frac[b, p]), 3),
                          "attn_fix_block_heads": att, "attn_fix_blocks_headavg": att_avg,
                          "attn_fix_max_head": round(float(max(fixh[f"mass_b{l}"][hh, p] for l, hh in att)), 3)})
    # 기록용: 규칙에서 한 조건만 만족한 후보
    near = []
    for p in range(N):
        if hn_frac[:, p].max() >= 0.5 or max(fix[f"mass_b{l}"][p] for l in range(L)) >= 0.5:
            near.append({"p": p, "hn_frac_max": round(float(hn_frac[:, p].max()), 3), "hn_frac_argmax_block": int(hn_frac[:, p].argmax()),
                         "attn_fix_max": round(float(max(fix[f"mass_b{l}"][p] for l in range(L))), 3),
                         "attn_fix_max_head": round(float(max(fixh[f"mass_b{l}"][:, p].max() for l in range(L))), 3)})
    sinks_s = ",".join(f"{s['p']}@{s['b']}" for s in sinks)
    # K/V 일관성 (기록용): 규칙으로 찾은 sink 위치
    kvcons = []
    if sinks:
        pos = [s["p"] for s in sinks]
        rec = {"kv_pos": pos}
        for i in range(0, AN_N, 16):
            hf_forward(vis, torch.as_tensor(x[i:i + 16]), rec=rec)
        for key in [f"b{l}" for l in range(L)] + ["pool"]:
            K, V = torch.cat(rec[f"k_{key}"]).numpy(), torch.cat(rec[f"v_{key}"]).numpy()   # (128,H,S,d)
            for j, s in enumerate(sinks):
                if key != "pool" and int(key[1:]) < s["b"]:
                    continue
                ck = cosr(K[:, :, j], K[:, :, j].mean(0, keepdims=True))      # (128,H)
                cv = cosr(V[:, :, j], V[:, :, j].mean(0, keepdims=True))
                kvcons.append({"key": key, "p": s["p"], "k_cos_med_minhead": round(float(np.median(ck, 0).min()), 4),
                               "v_cos_med_minhead": round(float(np.median(cv, 0).min()), 4),
                               "k_cos_mean": round(float(ck.mean()), 4), "v_cos_mean": round(float(cv.mean()), 4)})
    ratio_max = max(b["ratio"] for b in blocks)
    append("analyze.jsonl", {"what": "analyze", "model": name, "n": AN_N, "input": f"zoo{size} calib256_s0[:128]",
                             "ratio_max": ratio_max, "ratio_argmax_block": int(np.argmax([b["ratio"] for b in blocks])),
                             "sinks_rule": sinks, "sinks": sinks_s, "near_candidates": near,
                             "blocks": blocks, "sink_attn": sink_blocks, "kv_consistency": kvcons})
    np.savez_compressed(out_dir() / f"analyze_{name}_arrays.npz", hn_frac=hn_frac,
                        **{f"fix_{k[5:]}": v for k, v in fix.items()}, **{f"fixh_{k[5:]}": v for k, v in fixh.items()})


# ---------------------------------------------------------------- m2 (요청 5)

def parse_sinks(s):
    """'p@k:u' 또는 'p@b'(= p@b:b) → [(p, k, u)]."""
    out = []
    for it in (s.split(",") if s else []):
        p, r = it.split("@")
        k, u = (r.split(":") if ":" in r else (r, r))
        out.append((int(p), int(k), int(u)))
    return out


@torch.no_grad()
def build_m2_consts(vis, x, sinks, bs=16):
    """s2_regcache.build_mean_cache + build_uc와 같은 정의 (원래 FP32 모델, 보정 사진 평균)."""
    L = len(vis.encoder.layers)
    N = vis.embeddings.num_patches
    pos = [s[0] for s in sinks]
    first = min(s[1] for s in sinks)                     # K·V 상수가 필요한 첫 블록
    first_u = min(s[2] for s in sinks)                   # u_c가 필요한 첫 블록
    keep = [i for i in range(N) if i not in set(pos)]
    accK, accV, accU, n = {}, {}, {}, 0
    for i in range(0, len(x), bs):
        rec = {"kv_pos": pos, "z2_sink": True}
        hf_forward(vis, torch.as_tensor(np.ascontiguousarray(x[i:i + bs])), rec=rec)
        for key in [f"b{l}" for l in range(first, L)] + ["pool"]:
            act = [j for j, (p, kb, _) in enumerate(sinks) if key == "pool" or kb <= int(key[1:])]
            K, V = torch.cat(rec[f"k_{key}"])[:, :, act].double().sum(0), torch.cat(rec[f"v_{key}"])[:, :, act].double().sum(0)
            accK[key] = accK.get(key, 0) + K
            accV[key] = accV.get(key, 0) + V
        for l in range(first_u, L):
            z = torch.cat(rec[f"z2_b{l}"])
            accU[f"b{l}"] = accU.get(f"b{l}", 0) + z[:, keep].double().sum((0, 1))
        n += len(x[i:i + bs])
    kvc = {k: ((accK[k] / n).float(), (accV[k] / n).float()) for k in accK}
    uc = {k: (v / (n * len(keep))).float() for k, v in accU.items()}
    return kvc, uc, n


def cmd_m2(a):
    name = a.model
    sinks = parse_sinks(a.sinks)
    assert sinks, "sink 없음 — m2를 돌릴 수 없다"
    vis = vis_of(load_hf(name))
    size = MODELS[name]["size"]
    xc = load_x(size, "calib256_s0")
    assert len(xc) == CAL_N
    kvc, uc, n = build_m2_consts(vis, xc, sinks)
    assert n == CAL_N, n
    torch.save({"sinks": sinks, "kvc": kvc, "uc": uc, "n": n}, WORK / f"m2consts_{name}_{a.sinks.replace(',', '-')}.pt")
    pos = [s[0] for s in sinks]
    rows = {}
    sets = DEV_SETS if a.xset == "dev" else CONF_SETS
    tag = a.sinks.replace(",", "-").replace(":", "_")
    for key in sets:
        x = load_x(size, key)
        assert len(x) == DEV_K, (key, len(x))
        base = batched(lambda t: hf_forward(vis, t), x)
        out = {"m2": batched(lambda t: hf_forward(vis, t, sinks, kvc, uc), x),
               "kv_only": batched(lambda t: hf_forward(vis, t, sinks, kvc, None), x)}
        # oracle: 그 사진의 실제 K·V와 실제 norm2 출력으로 바꿔 끼우면 원래와 같아야 한다 (구현 검증)
        ora = []
        for i in range(0, DEV_K, 16):
            t = torch.as_tensor(x[i:i + 16])
            rec = {"kv_pos": pos, "z2_sink": True}
            hf_forward(vis, t, rec=rec)
            L = len(vis.encoder.layers)
            ora.append(_oracle_forward(vis, t, sinks, rec, L))
        out["oracle"] = torch.cat(ora).numpy()
        if a.xset == "dev" and not a.tagged_emb:
            np.save(WORK / f"emb_{name}_{key}_orig.npy", base)          # 1단계 top1이 읽는 이름 (1단계 재현용)
            np.save(WORK / f"emb_{name}_{key}_m2.npy", out["m2"])
        else:
            np.save(WORK / f"emb_{name}_{key}_orig.npy", base)                # conf 세트는 키 이름(conf_*)이 달라 겹치지 않음
            np.save(WORK / f"emb_{name}_{key}_m2_{tag}.npy", out["m2"])
        r = {}
        for mode, e in out.items():
            c = cosr(e, base)
            r[mode] = {"cos_mean": float(c.mean()), "cos_min": float(c.min())}
        rows[key] = r
        print(key, json.dumps(r), flush=True)
    assert all(rows[k]["oracle"]["cos_min"] >= 0.99999 for k in rows), "oracle 동치 실패 — 구현 버그"
    allc = {mode: float(np.mean([rows[k][mode]["cos_mean"] for k in sets])) for mode in ("m2", "kv_only", "oracle")}
    hr = {mode: float(np.mean([rows[k][mode]["cos_mean"] for k in sets[1:]])) for mode in ("m2", "kv_only")}
    append("m2.jsonl", {"what": "m2", "model": name, "sinks": a.sinks, "sinks_pku": [list(s) for s in sinks],
                        "calib_n": n, "seed": SEED, "xset": a.xset, "dev": f"6x{DEV_K}",
                        "input": f"zoo{size}", "cos_mean_1200": allc, "cos_mean_highres5": hr,
                        "cos_min_1200": {mode: float(min(rows[k][mode]["cos_min"] for k in sets)) for mode in ("m2", "kv_only", "oracle")},
                        "per_set": rows})


def _oracle_forward(vis, t, sinks, rec, L):
    """원래 forward에서 기록한 사진별 sink K·V·norm2 행으로 바꿔 끼운다 (배치마다 다른 값)."""
    pos = [s[0] for s in sinks]
    lists = {k: torch.cat(v) for k, v in rec.items() if isinstance(v, list) and k != "kv_pos"}
    K = {k[2:]: v for k, v in lists.items() if k.startswith("k_")}
    V = {k[2:]: v for k, v in lists.items() if k.startswith("v_")}
    Z = {k[3:]: v for k, v in lists.items() if k.startswith("z2_")}
    h = vis.embeddings(t)
    for l, ly in enumerate(vis.encoder.layers):
        at = ly.self_attn
        H = at.num_heads
        act = [j for j, (p, kb, _) in enumerate(sinks) if kb <= l]
        act_u = [j for j, (p, _, ub) in enumerate(sinks) if ub <= l]
        z = ly.layer_norm1(h)
        q, k, v = _heads(at.q_proj(z), H), _heads(at.k_proj(z), H), _heads(at.v_proj(z), H)
        k, v = k.clone(), v.clone()
        for j in act:
            k[:, :, pos[j]] = K[f"b{l}"][:, :, j]
            v[:, :, pos[j]] = V[f"b{l}"][:, :, j]
        a = torch.softmax((q @ k.transpose(-1, -2)) * at.scale, dim=-1)
        h = h + at.out_proj((a @ v).transpose(1, 2).reshape(h.shape))
        z2 = ly.layer_norm2(h).clone()
        for j in act_u:
            z2[:, pos[j]] = Z[f"b{l}"][:, pos[j]]
        h = h + ly.mlp(z2)
    h = vis.post_layernorm(h)
    head = vis.head
    q, k, v = _pool_qkv(head, h)
    k, v = k.clone(), v.clone()
    for j, p in enumerate(pos):
        k[:, :, p] = K["pool"][:, :, j]
        v[:, :, p] = V["pool"][:, :, j]
    a = torch.softmax((q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1]), dim=-1)
    o = head.attention.out_proj((a @ v).transpose(1, 2).reshape(h.shape[0], 1, -1))
    o = o + head.mlp(head.layernorm(o))
    return o[:, 0]


def cmd_m2xchk(a):
    """ref: 기존 timm 상수 파일 + timm 전처리 캐시 → 내 HF M2 == timm MaskRegCacheTrunk(bcast, mlp)."""
    from s2_regcache import MaskRegCacheTrunk, load_cache, parse_sinks as ps
    sinks2 = ps("55@5,13@9")
    sinks = [(p, b, b) for p, b in sinks2]
    cache, meta = load_cache(RC.ONNX_DIR / "regcache_55at5-13at9_highres256_s0.npz")
    z = np.load(RC.ONNX_DIR / "regcache_uc_55at5-13at9_highres256_s0.npz")
    uc = {k: z[k] for k in z.files if k != "meta"}
    model, _, _ = S.load_model()
    tm = MaskRegCacheTrunk(model.visual.trunk, sinks2, cache, "bcast", None, uc).eval()
    vis = vis_of(load_hf("ref"))
    kvc = {k: (torch.as_tensor(np.asarray(K)).float(), torch.as_tensor(np.asarray(V)).float()) for k, (K, V) in cache.items()}
    ucs = {k: torch.as_tensor(v).float() for k, v in uc.items()}
    x = np.concatenate([np.load(RC.CACHE_DIR / "pets_dev_x.npy", mmap_mode="r")[:32],
                        np.load(RC.CACHE_DIR / "caltech_dev_x.npy", mmap_mode="r")[:32]]).astype(np.float32)
    e_t = batched(lambda t: tm(t), x)
    e_h = batched(lambda t: hf_forward(vis, t, sinks, kvc, ucs), x)
    c = cosr(e_t, e_h)
    o_t = batched(lambda t: model.encode_image(t), x)
    o_h = batched(lambda t: hf_forward(vis, t), x)
    c0 = cosr(o_t, o_h)                                  # 원래 모델끼리 (구현과 무관한 바닥 차이)
    eff_t, eff_h = cosr(e_t, o_t), cosr(e_h, o_h)        # M2가 바꾼 양 (사진별 cos(M2, 원래))
    d_eff = np.abs((1 - eff_t) - (1 - eff_h))
    # 내 상수 계산(build_m2_consts)이 기존 파일과 같은지: 같은 timm 전처리 보정 256장으로
    from s2_quantize import calib_images
    xc = calib_images("highres", CAL_N, SEED)
    kvc2, uc2, n = build_m2_consts(vis, xc, sinks)
    dk = max(float(np.abs(kvc2[k][0].numpy() - np.asarray(cache[k][0])).max()) for k in cache)
    dv = max(float(np.abs(kvc2[k][1].numpy() - np.asarray(cache[k][1])).max()) for k in cache)
    du = max(float(np.abs(uc2[k].numpy() - uc[k]).max()) for k in uc)
    row = {"what": "m2xchk", "model": "ref", "n": len(x), "hf_vs_timm_m2_cos_min": float(c.min()),
           "hf_vs_timm_m2_cos_mean": float(c.mean()), "hf_vs_timm_orig_cos_min": float(c0.min()),
           "hf_vs_timm_orig_cos_mean": float(c0.mean()),
           "m2_effect_timm_mean": float(1 - eff_t.mean()), "m2_effect_hf_mean": float(1 - eff_h.mean()),
           "m2_effect_absdiff_max": float(d_eff.max()), "m2_effect_rel_diff_mean": float(abs(eff_t.mean() - eff_h.mean()) / max(1 - eff_t.mean(), 1e-12)),
           "const_maxabs_K": dk, "const_maxabs_V": dv, "const_maxabs_uc": du,
           "const_files": ["regcache_55at5-13at9_highres256_s0.npz", "regcache_uc_55at5-13at9_highres256_s0.npz"]}
    append("m2.jsonl", row)
    # 사후 변경 (README 6절): cos 최소 ≥ 0.99999는 원래 모델끼리도 못 넘으므로, 원래 모델끼리 차이보다 크게 나빠지지 않는지로 본다
    assert c.min() >= c0.min() - 1e-5, "HF M2 구현이 timm MaskRegCacheTrunk와 다름 (원래 모델끼리 차이보다 큼)"


# ---------------------------------------------------------------- top1 (요청 6)

def classifier(m, tok, names, templates):
    """S.zeroshot_classifier와 같은 앙상블. 문장은 우리 timm 분류기(open_clip HFTokenizer)와 같게 `_clean_canonicalize`
    (소문자·구두점 제거, big_vision 규칙)로 정리한 뒤 HF 토크나이저 max_length 64 — 정리 없이 하면 Cars가 무너진다 (README 8절)."""
    from open_clip.tokenizer import _clean_canonicalize
    W = []
    for c in names:
        sents = [_clean_canonicalize(t.format(c, c=c)) for t in templates]
        e = []
        for i in range(0, len(sents), 128):
            ids = tok(sents[i:i + 128], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            f = m.get_text_features(input_ids=ids["input_ids"])
            f = getattr(f, "pooler_output", f)
            e.append(f.float().numpy())
        e = l2n(np.concatenate(e)).mean(0)
        W.append(e / np.linalg.norm(e))
    return np.stack(W, 1).astype(np.float32)


def cmd_top1(a):
    from transformers import AutoTokenizer
    name = a.model
    m = load_hf(name)
    tok = AutoTokenizer.from_pretrained(MODELS[name]["hf"])
    size = MODELS[name]["size"]
    from torchvision.datasets import CIFAR100
    cif_names = [c.replace("_", " ") for c in CIFAR100(str(RC.DATA_ROOT), train=True, download=False).classes]
    res = {}
    for key in DEV_SETS:
        if key == "cifar_A":
            names, templates = cif_names, RC.TEMPLATES
        else:
            n0 = key[:-4]
            names, templates = RD.classnames(n0), RD.HIGHRES[n0]["templates"]
        W = classifier(m, tok, names, templates)
        np.save(WORK / f"clf_{name}_{key}.npy", W)
        y = load_y(size, key)
        r = {}
        for mode in ("orig", "m2"):
            f = WORK / f"emb_{name}_{key}_{mode}.npy"
            if not f.exists():
                if mode == "orig":
                    f_e = batched(lambda t: vis_of(m)(pixel_values=t).pooler_output, load_x(size, key))
                else:
                    continue
            else:
                f_e = np.load(f)
            r[mode] = float(((l2n(f_e) @ W).argmax(1) == y).mean() * 100)
        res[key] = r
        print(key, r, flush=True)
    hr = {mode: float(np.mean([res[k][mode] for k in DEV_SETS if k != "cifar_A" and mode in res[k]]))
          for mode in ("orig", "m2") if all(mode in res[k] for k in DEV_SETS)}
    append("top1.jsonl", {"what": "top1", "model": name, "dev": f"6x{DEV_K}", "input": f"zoo{size}", "per_set": res,
                          "highres5_mean": hr, "tokenizer": "AutoTokenizer, padding=max_length 64, open_clip _clean_canonicalize",
                          "templates": "rn50/common.TEMPLATES (CIFAR), rn50/datasets.HIGHRES[*].templates"})


# ---------------------------------------------------------------- 2단계: 확인 세트 (README 12-3)

def _sha(b):
    import hashlib
    return hashlib.sha1(b).hexdigest()


def cmd_confset(a):
    from PIL import Image
    from torchvision.datasets import CIFAR100
    cd224, cd256 = cache_dir(224), cache_dir(256)
    calib256 = {ln.split()[1] for ln in (cd224 / "calib256_s0_src.txt").read_text().split("\n") if ln.strip()}
    assert len(calib256) == CAL_N
    info = {}
    for name in HR:
        cfg = RD.HIGHRES[name]
        p_dev, p_cal = RD.DEV_N / cfg["n_pool"], RD.CAL_N / cfg["n_pool"]
        had = {loc for loc, _ in cfg["pool"] if (RC.EXTRA_ROOT / loc).exists()}
        # 1회차: 나머지 구간(u ≥ p_dev + p_cal)의 순번과 dev·보정 후보 SHA-1
        dev_sha, cal_sha, rest_n = set(), set(), 0
        for b, _ in RD.parquet_rows(cfg["pool"], True):
            u = RD.u_of(b)
            if u < p_dev:
                dev_sha.add(_sha(b))
            elif u < p_dev + p_cal:
                cal_sha.add(_sha(b))
            else:
                rest_n += 1
        sel = set(np.sort(np.random.default_rng(CONF_SEED).choice(rest_n, DEV_K, replace=False)).tolist())
        # 2회차: 고른 사진
        imgs, ys, shas, i = [], [], [], 0
        for b, lab in RD.parquet_rows(cfg["pool"], True):
            if RD.u_of(b) >= p_dev + p_cal:
                if i in sel:
                    imgs.append(b); ys.append(lab); shas.append(_sha(b))
                i += 1
        assert i == rest_n and len(imgs) == DEV_K
        for loc, _ in cfg["pool"]:                        # 이번에 새로 받은 파일은 지운다 (parquet_rows 기본 동작과 같게)
            if loc not in had and (RC.EXTRA_ROOT / loc).exists():
                (RC.EXTRA_ROOT / loc).unlink()
        test_sha = {_sha(b) for b, _ in RD.parquet_rows(cfg["test"], False)}   # test는 SHA-1만 계산 (모델에 넣지 않음)
        dev_old = set((RC.CACHE_DIR / f"{name}_dev_sha1.txt").read_text().split())
        assert dev_old <= dev_sha, "기존 dev 목록이 dev 구간과 다름"
        ov = {"dev": len(set(shas) & dev_sha), "calib_cands": len(set(shas) & cal_sha), "calib256": len(set(shas) & calib256),
              "test": len(set(shas) & test_sha), "dup_in_set": DEV_K - len(set(shas))}
        info[name] = {"rest_n": rest_n, "n_dev": len(dev_sha), "n_cal_cands": len(cal_sha), "n_test": len(test_sha), "overlap": ov}
        print(name, info[name], flush=True)
        assert not any(ov.values()), f"{name} 확인 세트 겹침 {ov} — 멈춤 (README 12-3)"
        for size, cd in ((224, cd224), (256, cd256)):
            np.save(cd / f"conf_{name}_x.npy", np.stack([zoo_pre(Image.open(io.BytesIO(b)), size) for b in imgs]).astype(np.float32))
            np.save(cd / f"conf_{name}_y.npy", np.array(ys))
        (cd224 / f"conf_{name}_sha1.txt").write_text("\n".join(shas) + "\n")
    tr = CIFAR100(str(RC.DATA_ROOT), train=True, download=False)
    te = CIFAR100(str(RC.DATA_ROOT), train=False, download=False)
    lo, hi = CIFAR_CONF_RANGE
    idx = np.sort(np.random.default_rng(CONF_SEED).choice(hi - lo, DEV_K, replace=False)) + lo
    px = lambda d, i: np.ascontiguousarray(d.data[i]).tobytes()
    shas = [_sha(px(tr, int(i))) for i in idx]
    dev_ab = {_sha(px(tr, i)) for i in range(RC.VAL_SPLITS["A"], RC.VAL_SPLITS["B"] + RC.VAL_SIZE)}
    test_sha = {_sha(px(te, i)) for i in range(len(te.data))}
    calib_rng = {_sha(px(tr, i)) for i in range(*RC.CALIB_RANGE)}                    # 기록용 (이번 보정 256장에는 CIFAR 없음)
    ov = {"dev_AB": len(set(shas) & dev_ab), "test": len(set(shas) & test_sha), "dup_in_set": DEV_K - len(set(shas))}
    info["cifar"] = {"range": [lo, hi], "overlap": ov, "overlap_calib_range_record_only": len(set(shas) & calib_rng)}
    print("cifar", info["cifar"], flush=True)
    assert not any(ov.values()), f"CIFAR 확인 세트 겹침 {ov} — 멈춤"
    all_sha = shas + sum([(cd224 / f"conf_{n}_sha1.txt").read_text().split() for n in HR], [])
    assert len(set(all_sha)) == len(all_sha) == 6 * DEV_K, "확인 세트 1,200장 안에 중복"
    for size, cd in ((224, cd224), (256, cd256)):
        np.save(cd / "conf_cifar_x.npy", np.stack([zoo_pre(tr[int(i)][0], size) for i in idx]).astype(np.float32))
        np.save(cd / "conf_cifar_y.npy", np.array([tr.targets[int(i)] for i in idx]))
    (cd224 / "conf_cifar_sha1.txt").write_text("\n".join(shas) + "\n")
    (cd224 / "conf_cifar_idx.txt").write_text("\n".join(str(int(i)) for i in idx) + "\n")
    append("confset.jsonl", {"what": "confset", "seed": CONF_SEED, "per_set": DEV_K, "n": 6 * DEV_K, "sets": CONF_SETS,
                             "info": info, "preprocess": "zoo (PIL BICUBIC, x/127.5-1)", "sizes": [224, 256]})


# ---------------------------------------------------------------- 2단계: 통일 규칙 (README 12-1·12-2)

def cmd_unify(a):
    name = a.model
    rows = [json.loads(ln) for ln in open(out_dir() / "analyze.jsonl", encoding="utf-8")]
    row = [r for r in rows if r["model"] == name][-1]
    pos = sorted({s["p"] for s in row["sinks_rule"]})
    assert pos, "sink 토큰 없음"
    vis = vis_of(load_hf(name))
    L = len(vis.encoder.layers)
    x = load_x(MODELS[name]["size"], "calib256_s0")[:AN_N]
    rec = {"pts": True}
    for i in range(0, AN_N, 16):
        hf_forward(vis, torch.as_tensor(x[i:i + 16]), rec=rec)
    order = [(l, pt) for l in range(L) for pt in ("in", "attn", "mlp")]
    frac = {}
    for l, pt in order:
        n = torch.cat(rec[f"n_{pt}_b{l}"]).numpy()                         # (128, N)
        hn = n / np.median(n, axis=1, keepdims=True) > HIGH_NORM
        frac[(l, pt)] = hn.mean(0)
    out, table = [], {}
    for p in pos:
        first = next(((l, pt) for l, pt in order if frac[(l, pt)][p] >= FIX), None)
        assert first is not None, p
        l0, pt0 = first
        u = l0                                                                # attn·mlp 뒤 → 그 블록 MLP부터, 블록 0 입력 → 0
        k = next(l for l in range(L) if frac[(l, "in")][p] >= FIX)
        out.append((p, k, u))
        table[p] = {"first": {"block": l0, "point": pt0}, "k": k, "u": u,
                    "frac": {f"b{l}": {pt: round(float(frac[(l, pt)][p]), 3) for pt in ("in", "attn", "mlp")}
                             for l in range(max(0, l0 - 1), min(L, l0 + 3))}}
    out.sort(key=lambda t: (t[2], t[0]))
    s_u = ",".join(f"{p}@{k}:{u}" for p, k, u in out)
    append("unify.jsonl", {"what": "unify", "model": name, "n": AN_N, "analyze_commit": row["commit"],
                           "sink_tokens": pos, "sinks_unified": s_u, "table": {str(p): t for p, t in table.items()}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache").set_defaults(func=cmd_cache)
    sub.add_parser("ident").set_defaults(func=cmd_ident)
    for c, f in (("analyze", cmd_analyze), ("top1", cmd_top1)):
        p = sub.add_parser(c)
        p.add_argument("--model", choices=list(MODELS), required=True)
        p.set_defaults(func=f)
    p = sub.add_parser("m2")
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--sinks", required=True, help="p@b 또는 p@k:u, 쉼표 (analyze·unify의 sinks 값)")
    p.add_argument("--xset", choices=["dev", "conf"], default="dev", help="dev 6×200 (1단계) 또는 확인 세트 6×200 (2단계, 판정용)")
    p.add_argument("--tagged-emb", action="store_true", help="dev 임베딩을 sinks 태그 이름으로 저장 (1단계 top1 파일을 덮지 않음)")
    p.set_defaults(func=cmd_m2)
    sub.add_parser("confset").set_defaults(func=cmd_confset)
    p = sub.add_parser("unify")
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.set_defaults(func=cmd_unify)
    sub.add_parser("m2xchk").set_defaults(func=cmd_m2xchk)
    a = ap.parse_args()
    out_dir()
    env_info.write_env(out_dir(), env_info.collect())
    t0 = time.time()
    a.func(a)
    print(f"[{a.cmd}] {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
