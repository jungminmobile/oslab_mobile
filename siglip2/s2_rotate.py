"""
s2_rotate.py — LayerNorm을 그대로 둔 오프라인 회전: 가중치만 바꾼 수학적 동치 SigLIP2 이미지 인코더 (G3Q)
=========================================================================================================

[왜]  D-031: U8 손실은 QKV 투영 정수 경로(= LayerNorm1 출력)에서 나온다. G3_why: 잔차 흐름의 거대 값은 정해진 채널
      (67·301·145·483)에 몰린다. 잔차 흐름을 직교 행렬 Q로 돌리면 채널 쏠림이 768채널에 퍼진다.
      그래프(연산 종류·개수·노드 이름)는 원래 FP32 ONNX와 같고 가중치만 바뀐다 → ORT·Hailo 경로 그대로.
      설계와 판정 기준: `results/siglip2/G3Q_rotate/README.md` (결과 전에 고정).

[변환]  (계산은 float64, 마지막에 float32)
  1. LN 접기: norm1 → attn.qkv, norm2 → mlp.fc1, trunk.norm → attn_pool.kv.  W ← W·diag(γ), b ← b + W·β, LN은 γ=1, β=0
  2. 중심화: 잔차에 쓰는 것(patch_embed conv·bias, pos_embed, attn.proj, mlp.fc2)의 출력 채널을 C = I − 11ᵀ/n로
  3. 회전: Q (직교, Q·1 = 1). 쓰는 쪽 W ← Q·W (pos ← pos·Qᵀ), 읽는 쪽 W ← W·Qᵀ.
     Q1 = 1 → LN₀(Qx) = Q·LN₀(x) (평균·노름 보존). MLP 내부(3072)·헤드 내부·풀링 헤드 latent 쪽은 그대로.

[Q 후보]  (미리 두 개만, README 3-2절)
  rotrand0  P·diag(1, R)·P,  R = 767×767 무작위 직교 (torch 시드 0), P = e₁ ↔ 1/√n Householder 반사
  rothad    (H/√n)·diag(1, S)·P,  H = H₁₂(Paley) ⊗ H₆₄(Sylvester) 첫 행·열 모두 1, S = 부호 767개 (torch 시드 0)

[하위 명령]
  selftest  무작위 가중치(LN γ·β·bias도 무작위)로 동치·ORT·연산 종류·노드 이름 확인 (Q0, 데이터 불필요)
  build     Q 행렬을 만들어 $RN50_ONNX/rotq_<q>.npy로 저장 + 직교성·Q1=1 확인
  fp32      PyTorch FP32로 원래 vs 회전 (dev 64장) + LN 출력(qkv·fc1·kv 입력) 범위 비율 (Q1 미리보기, 기록용 진단)
  export    회전 모델 → ONNX (배치 가변 + b1) + ORT vs 원래 FP32 임베딩(dev 64장, Q1 판정) + 연산 표 비교

[사용]  (siglip2/ 폴더에서)
  "$SIGLIP2_PYTHON" s2_rotate.py selftest
  RN50_RESULTS=../results/siglip2/G3Q_rotate "$SIGLIP2_PYTHON" s2_rotate.py build
  RN50_RESULTS=../results/siglip2/G3Q_rotate "$SIGLIP2_PYTHON" s2_rotate.py fp32 --q rotrand0 rothad
  RN50_RESULTS=../results/siglip2/G3Q_rotate "$SIGLIP2_PYTHON" s2_rotate.py export --q rothad
  → 양자화: s2_quantize.py --src siglip2_image_fp32_<q>.onnx (README 6절)
"""
import argparse
import collections
import copy
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
QS = ["rotrand0", "rothad"]
DEV64 = ["pets_dev", "caltech_dev"]          # G3_why·G3R과 같은 dev 64장 구성 (앞 32장씩)


# --------------------------------------------------------------------------- #
# 1. Q 행렬 (데이터·경로 의존 없음)
# --------------------------------------------------------------------------- #
def householder_ones(n):
    """e₁과 1/√n을 맞바꾸는 Householder 반사 P (대칭·직교). P·e₁ = 1/√n, P·1 = √n·e₁."""
    v = -np.full(n, 1.0 / np.sqrt(n))
    v[0] += 1.0
    return np.eye(n) - 2.0 * np.outer(v, v) / (v @ v)


def q_rand(n, seed=0):
    """(a) P·diag(1, R)·P. R = (n−1)×(n−1) 무작위 직교 (float64 정규분포 → QR, R 대각 부호로 부호 고정)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n - 1, n - 1, generator=g, dtype=torch.float64)
    Qr, Rr = torch.linalg.qr(A)
    Qr = Qr * torch.sign(torch.diagonal(Rr)).unsqueeze(0)
    D = np.eye(n)
    D[1:, 1:] = Qr.numpy()
    P = householder_ones(n)
    return P @ D @ P


def _normalize_hadamard(H):
    """열·행 부호를 바꿔 첫 행과 첫 열이 모두 1이 되게 (Hadamard 성질은 유지)."""
    H = H * np.sign(H[0:1, :])
    return H * np.sign(H[:, 0:1])


def hadamard_paley12():
    """Paley 구성 I (q = 11 ≡ 3 mod 4): H = I + S, S = [[0, 1ᵀ], [−1, J]], J = Jacobsthal 행렬."""
    q = 11
    sq = {(i * i) % q for i in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    J = np.array([[chi(j - i) for j in range(q)] for i in range(q)], dtype=np.float64)
    S = np.zeros((q + 1, q + 1))
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = J
    return _normalize_hadamard(np.eye(q + 1) + S)


def hadamard_sylvester(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    assert H.shape[0] == n, f"Sylvester 크기는 2의 거듭제곱이어야 함: {n}"
    return H


def q_had(n=768, seed=0):
    """(b) (H/√n)·diag(1, S)·P, H = H₁₂ ⊗ H₆₄, S = 부호 n−1개 (torch 시드 0).
    P·1 = √n·e₁ → diag(1,S)는 e₁ 고정 → H의 첫 열이 1이므로 Q·1 = 1.
    S가 없으면(H/√n·P) H/√n과 P가 같은 두 벡터(e₁ ↔ 1/√n)를 맞바꿔 Q·e₁ = e₁ → 채널 0이 섞이지 않는다 (selftest에서 발견, 10/18)."""
    assert n == 768, "rothad는 768 = 12 × 64 전용"
    H12, H64 = hadamard_paley12(), hadamard_sylvester(64)
    for h in (H12, H64):
        k = h.shape[0]
        assert np.array_equal(h @ h.T, k * np.eye(k)), "Hadamard 아님"
        assert (h[0] == 1).all() and (h[:, 0] == 1).all()
    H = np.kron(H12, H64)
    assert np.array_equal(H @ H.T, n * np.eye(n)) and (H[:, 0] == 1).all()
    g = torch.Generator().manual_seed(seed)
    sgn = (torch.randint(0, 2, (n - 1,), generator=g, dtype=torch.int64) * 2 - 1).double().numpy()
    D = np.diag(np.concatenate([[1.0], sgn]))
    return (H / np.sqrt(n)) @ D @ householder_ones(n)


def make_q(name, n=768):
    if name == "rotrand0":
        return q_rand(n, 0)
    if name == "rothad":
        return q_had(n)
    raise ValueError(name)


def q_checks(Q):
    n = Q.shape[0]
    one = np.ones(n)
    return {"orth_max_abs": float(np.abs(Q.T @ Q - np.eye(n)).max()),
            "q1_minus_1_max_abs": float(np.abs(Q @ one - one).max()),
            "max_abs_entry": float(np.abs(Q).max()),
            "min_abs_entry": float(np.abs(Q).min())}


# --------------------------------------------------------------------------- #
# 2. 회전 (timm VisionTransformer trunk를 제자리에서 바꿈)
# --------------------------------------------------------------------------- #
def check_trunk(trunk):
    """잔차 흐름을 읽는 곳이 norm1·norm2·trunk.norm뿐이고 더하는 곳이 patch_embed·pos_embed·proj·fc2뿐인 구조인지."""
    I = nn.Identity
    assert getattr(trunk, "num_prefix_tokens", 0) == 0 and trunk.cls_token is None and trunk.reg_token is None
    assert trunk.pos_embed is not None and not getattr(trunk, "dynamic_img_size", False)
    assert isinstance(trunk.patch_embed.norm, I) and isinstance(trunk.patch_embed.proj, nn.Conv2d)
    for m in (trunk.norm_pre, trunk.fc_norm, trunk.head, trunk.patch_drop):
        assert isinstance(m, I), f"Identity가 아님: {m}"
    assert trunk.global_pool == "map" and trunk.attn_pool is not None and trunk.attn_pool.pos_embed is None
    for blk in trunk.blocks:
        for m in (blk.ls1, blk.ls2, blk.drop_path1, blk.drop_path2):
            assert isinstance(m, I), f"Identity가 아님: {m}"
        for ln in (blk.norm1, blk.norm2):
            assert isinstance(ln, nn.LayerNorm) and ln.weight is not None and ln.bias is not None
        assert type(blk).__name__ == "Block", type(blk)
    assert isinstance(trunk.norm, nn.LayerNorm)


@torch.no_grad()
def _write(lin_w, lin_b, M):
    """잔차에 쓰는 층: 출력 채널 쪽에 M (= Q·C)을 곱한다. lin_w (n, ...)."""
    W = lin_w.double().reshape(lin_w.shape[0], -1)
    lin_w.copy_((M @ W).reshape(lin_w.shape).to(lin_w.dtype))
    if lin_b is not None:
        lin_b.copy_((M @ lin_b.double()).to(lin_b.dtype))


@torch.no_grad()
def _fold_read(ln, lin, Q):
    """LN γ·β를 다음 Linear에 접고 입력 쪽에 Qᵀ를 흡수. LN은 γ=1, β=0."""
    W = lin.weight.double()
    g, b = ln.weight.double(), ln.bias.double()
    lin.bias.copy_((lin.bias.double() + W @ b).to(lin.bias.dtype))
    lin.weight.copy_(((W * g.unsqueeze(0)) @ Q.T).to(lin.weight.dtype))
    ln.weight.fill_(1.0)
    ln.bias.zero_()


@torch.no_grad()
def rotate_trunk(trunk, Q, center=True):
    """trunk를 제자리에서 회전한다. Q: (n, n) numpy float64, 직교, Q·1 = 1."""
    check_trunk(trunk)
    n = trunk.embed_dim
    assert Q.shape == (n, n)
    c = q_checks(Q)
    assert c["orth_max_abs"] < 1e-10 and c["q1_minus_1_max_abs"] < 1e-10, c
    Qd = torch.as_tensor(Q, dtype=torch.float64)
    C = torch.eye(n, dtype=torch.float64) - 1.0 / n
    M = Qd @ C if center else Qd
    pe = trunk.patch_embed.proj
    _write(pe.weight, pe.bias, M)
    trunk.pos_embed.copy_((trunk.pos_embed.double() @ M.T).to(trunk.pos_embed.dtype))
    for blk in trunk.blocks:
        _fold_read(blk.norm1, blk.attn.qkv, Qd)
        _write(blk.attn.proj.weight, blk.attn.proj.bias, M)
        _fold_read(blk.norm2, blk.mlp.fc1, Qd)
        _write(blk.mlp.fc2.weight, blk.mlp.fc2.bias, M)
    _fold_read(trunk.norm, trunk.attn_pool.kv, Qd)
    return trunk


R2S = ["had64"]


def make_r2(name, d=64, seed=0):
    """(R2, 기본 꺼짐) 헤드 안쪽 회전 R = diag(s)·H_d/√d, s = ±1 부호 d개 (torch 시드 0). 모든 블록·헤드에 같은 R."""
    assert name == "had64" and d == 64
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int64) * 2 - 1).double().numpy()
    R = np.diag(s) @ hadamard_sylvester(d) / np.sqrt(d)
    assert np.abs(R.T @ R - np.eye(d)).max() < 1e-12
    return R


@torch.no_grad()
def rotate_r2(trunk, R):
    """블록마다 헤드마다 V를 R로 돌리고(qkv의 V 출력 행 W_v,h ← Rᵀ·W_v,h, b_v,h ← Rᵀ·b_v,h) attn.proj 입력 열을 같이 돌린다
    (W_o[:, h] ← W_o[:, h]·R). (P·V·R)·(W_o,h·R)ᵀ = P·V·W_o,hᵀ로 동치. 풀링 헤드는 건드리지 않는다.
    timm Attention: qkv 출력 = [q|k|v] × 헤드 × d, attention 출력 = (P·V).transpose(1,2).reshape → proj 입력 열 = 헤드 × d."""
    Rt = torch.as_tensor(R, dtype=torch.float64)
    for blk in trunk.blocks:
        at = blk.attn
        H, d = at.num_heads, at.head_dim
        C = H * d
        assert Rt.shape == (d, d) and at.qkv.out_features == 3 * C and isinstance(at.norm, nn.Identity)
        assert getattr(at, "gate", None) is None and at.proj.in_features == C
        Wq, bq = at.qkv.weight.double(), at.qkv.bias.double()
        Wo = at.proj.weight.double()
        for h in range(H):
            r = slice(2 * C + h * d, 2 * C + (h + 1) * d)      # V 헤드 h의 qkv 출력 행
            Wq[r] = Rt.T @ Wq[r]
            bq[r] = Rt.T @ bq[r]
            c = slice(h * d, (h + 1) * d)                      # proj 입력 열 (헤드 h)
            Wo[:, c] = Wo[:, c] @ Rt
        at.qkv.weight.copy_(Wq.to(at.qkv.weight.dtype))
        at.qkv.bias.copy_(bq.to(at.qkv.bias.dtype))
        at.proj.weight.copy_(Wo.to(at.proj.weight.dtype))
    return trunk


@torch.no_grad()
def _proj_in_ratios(trunk, x):
    """(R2 진단) attn proj 입력(헤드 출력 이어붙인 것)의 max/p99.9와 채널별 max 최대/중앙값."""
    caps, hs = {}, []
    for l, blk in enumerate(trunk.blocks):
        hs.append(blk.attn.proj.register_forward_hook(lambda m, i, o, l=l: caps.setdefault(f"proj{l}", []).append(i[0].detach()) and None))
    try:
        for i in range(0, len(x), 16):
            trunk(torch.as_tensor(np.ascontiguousarray(x[i:i + 16]), dtype=torch.float32))
    finally:
        for h in hs:
            h.remove()
    res = {}
    for k, v in caps.items():
        t = torch.cat(v).reshape(-1, 768).abs()
        flat = t.flatten()
        q = flat.kthvalue(max(1, int(np.ceil(0.999 * flat.numel())))).values
        ch = t.max(0).values
        res[k] = {"max_over_p999": round(float(flat.max() / q), 3), "ch_max_over_median": round(float(ch.max() / ch.median()), 3)}
    return res


def unfuse(module):
    n = 0
    for m in module.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            n += 1
    assert n > 0
    return n


# --------------------------------------------------------------------------- #
# 3. ONNX 공통
# --------------------------------------------------------------------------- #
def export_onnx(module, path, dynamic, opset=17):
    """s2_export_onnx.export와 같은 조건 (dynamo=False, opset 17, 입력 image, 출력 embedding)."""
    x = torch.randn(1, 3, 224, 224)
    kw = dict(dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}) if dynamic else {}
    with torch.inference_mode():
        torch.onnx.export(module, (x,), str(path), input_names=["image"], output_names=["embedding"],
                          opset_version=opset, dynamo=False, **kw)
    return undedup_initializers(path)


def undedup_initializers(path):
    """접은 LN의 γ(모두 1)·β(모두 0)는 25개가 같은 값이라, torch exporter가 initializer 하나로 합치고 Identity 노드로 잇는다
    (원래 export에는 Identity가 없음). initializer를 받는 Identity마다 그 출력 이름으로 initializer 사본을 만들고 노드를 지워
    원래 그래프와 같은 연산 구성으로 되돌린다. 계산은 같다. 지운 노드 수를 돌려준다."""
    import onnx
    m = onnx.load(str(path))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    drop = [n for n in g.node if n.op_type == "Identity" and n.input[0] in inits]
    for n in drop:
        t = onnx.TensorProto()
        t.CopyFrom(inits[n.input[0]])
        t.name = n.output[0]
        g.initializer.append(t)
        g.node.remove(n)
    if drop:
        onnx.checker.check_model(m)
        onnx.save(m, str(path))
    return len(drop)


def graph_sig(path):
    """연산 종류별 개수, 노드 (이름, 종류) 순서 목록, initializer 이름 목록.
    노드 이름은 범위가 붙은 것('/trunk/blocks.5/mlp/fc2/MatMul' 등, --exclude가 쓰는 이름)만 비교하고, 자동 번호 이름
    ('Constant_317' 등)은 빈 문자열로 둔다 — undedup으로 지운 Identity 수만큼 번호만 밀리기 때문 (종류·순서는 비교함)."""
    import onnx
    g = onnx.load(str(path), load_external_data=False).graph
    ops = collections.Counter(n.op_type for n in g.node)
    return dict(sorted(ops.items())), [(n.name if "/" in n.name else "", n.op_type) for n in g.node], sorted(t.name for t in g.initializer)


def ort_run(path, x, bs, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"image": np.ascontiguousarray(x[i:i + bs], np.float32)})[0]
                           for i in range(0, len(x), bs)])


def l2n(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# 4. selftest (Q0)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _randomize(trunk, gen):
    """무작위 init은 LN γ=1·β=0이라 접기 검사가 안 된다 → γ·β·bias·pos_embed를 무작위로, 채널 하나에 큰 bias."""
    for m in trunk.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.copy_(1 + 0.5 * torch.randn(m.weight.shape, generator=gen))
            m.bias.copy_(0.3 * torch.randn(m.bias.shape, generator=gen))
        elif isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=gen))
    trunk.pos_embed.copy_(0.5 * torch.randn(trunk.pos_embed.shape, generator=gen))
    trunk.blocks[5].mlp.fc2.bias[67] += 20.0          # 거대 채널 흉내 (G3_why 채널 67)


def cmd_selftest_r2(a):
    """(--r2) 무작위 가중치: R2 단독·rothad+R2가 원래와 동치, ORT = PyTorch, 연산·이름 같음."""
    import timm
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    base = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0).eval()
    unfuse(base)
    _randomize(base, gen)
    x = torch.randn(4, 3, 224, 224, generator=gen)
    R = make_r2(a.r2)
    with torch.no_grad():
        ref = base(x)
    res, checks = {"r2": a.r2}, {}
    tmp = Path(tempfile.mkdtemp(prefix="s2rot_r2_", dir=a.tmp))
    try:
        sig = {}
        for tag, dyn in (("dyn", True), ("b1", False)):
            p = tmp / f"orig_{tag}.onnx"
            export_onnx(base, p, dyn)
            sig[tag] = graph_sig(p)
            p.unlink()
        for name, qn in (("r2", None), ("rothad_r2", "rothad")):
            m = copy.deepcopy(base)
            if qn:
                rotate_trunk(m, make_q(qn))
            rotate_r2(m, R).eval()
            with torch.no_grad():
                y = m(x)
            r = {"vs_orig_max_abs": float((y - ref).abs().max())}
            for tag, dyn in (("dyn", True), ("b1", False)):
                p = tmp / f"{name}_{tag}.onnx"
                export_onnx(m, p, dyn)
                ops, names, inits = graph_sig(p)
                r[f"graph_equal_{tag}"] = ops == sig[tag][0] and names == sig[tag][1] and inits == sig[tag][2]
                r[f"ort_vs_torch_{tag}"] = float(np.abs(ort_run(p, x.numpy(), 4 if dyn else 1) - y.numpy()).max())
                p.unlink()
            res[name] = r
            checks.update({f"{name}: 회전 = 원래 (< 1e-4)": r["vs_orig_max_abs"] < 1e-4,
                           f"{name}: ORT = PyTorch (< 1e-4)": max(r["ort_vs_torch_dyn"], r["ort_vs_torch_b1"]) < 1e-4,
                           f"{name}: 그래프 같음 (dyn·b1)": r["graph_equal_dyn"] and r["graph_equal_b1"]})
    finally:
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()
    res["checks"] = checks
    print(json.dumps(res, ensure_ascii=False, indent=1))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    print("selftest --r2 통과")


def cmd_selftest(a):
    if getattr(a, "r2", ""):
        return cmd_selftest_r2(a)
    import timm
    import onnxruntime
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    base = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0).eval()
    unfuse(base)
    _randomize(base, gen)
    x = torch.randn(4, 3, 224, 224, generator=gen)
    res = {"timm": timm.__version__, "torch": torch.__version__, "ort": onnxruntime.__version__}
    checks = {}
    with torch.no_grad():
        ref = base(x)
    res["ref_abs_max"] = float(ref.abs().max())
    tmp = Path(tempfile.mkdtemp(prefix="s2rot_", dir=a.tmp))
    try:
        sig = {}
        for tag, dyn in (("dyn", True), ("b1", False)):
            p = tmp / f"orig_{tag}.onnx"
            export_onnx(base, p, dyn)
            sig[tag] = graph_sig(p)
            p.unlink()
        res["orig_ops"] = sig["dyn"][0]
        for qn in QS:
            Q = make_q(qn)
            r = {"q": q_checks(Q)}
            rot = rotate_trunk(copy.deepcopy(base), Q).eval()
            with torch.no_grad():
                y = rot(x)
                # LN 출력이 Q배인지 (블록 0 norm1): rot.norm1(x_rot) == Q·LN₀(x)
                h0 = base.norm_pre(base._pos_embed(base.patch_embed(x)))
                h0r = rot.norm_pre(rot._pos_embed(rot.patch_embed(x)))
                ln0 = torch.nn.functional.layer_norm(h0, (768,), eps=base.blocks[0].norm1.eps)
                r["ln_rot_eq_q_ln_max_abs"] = float((rot.blocks[0].norm1(h0r).double()
                                                     - ln0.double() @ torch.as_tensor(Q).T).abs().max())
                r["resid_token_mean_abs_max"] = float(h0r.mean(-1).abs().max())
            r["rot_vs_orig_max_abs"] = float((y - ref).abs().max())
            r["rot_vs_orig_cos_min"] = float(torch.nn.functional.cosine_similarity(y, ref).min())
            r["ln_gamma1_beta0"] = all(bool((m.weight == 1).all() and (m.bias == 0).all())
                                       for m in rot.modules() if isinstance(m, nn.LayerNorm)
                                       and m is not rot.attn_pool.norm)
            for tag, dyn in (("dyn", True), ("b1", False)):
                p = tmp / f"{qn}_{tag}.onnx"
                export_onnx(rot, p, dyn)
                ops, names, inits = graph_sig(p)
                r[f"ops_equal_{tag}"] = ops == sig[tag][0]
                r[f"node_names_equal_{tag}"] = names == sig[tag][1] and inits == sig[tag][2]
                yo = ort_run(p, x.numpy(), 4 if dyn else 1)
                r[f"ort_vs_torch_max_abs_{tag}"] = float(np.abs(yo - y.numpy()).max())
                p.unlink()
            res[qn] = r
            checks.update({
                f"{qn}: QᵀQ = I, Q1 = 1 (< 1e-10)": r["q"]["orth_max_abs"] < 1e-10 and r["q"]["q1_minus_1_max_abs"] < 1e-10,
                f"{qn}: 고정되는 표준 기저 없음 (max|Q| < 0.5)": r["q"]["max_abs_entry"] < 0.5,
                f"{qn}: ① 회전 = 원래 (< 1e-4)": r["rot_vs_orig_max_abs"] < 1e-4,
                f"{qn}: LN₀(Qx) = Q·LN₀(x) (< 1e-4)": r["ln_rot_eq_q_ln_max_abs"] < 1e-4,
                f"{qn}: 잔차 토큰 평균 0 (중심화, < 1e-4)": r["resid_token_mean_abs_max"] < 1e-4,
                f"{qn}: 접은 LN γ=1 β=0": r["ln_gamma1_beta0"],
                f"{qn}: ② ORT = PyTorch (< 1e-4, dyn·b1)": max(r["ort_vs_torch_max_abs_dyn"], r["ort_vs_torch_max_abs_b1"]) < 1e-4,
                f"{qn}: ③ 연산 종류별 개수 같음 (dyn·b1)": r["ops_equal_dyn"] and r["ops_equal_b1"],
                f"{qn}: ③ 노드·initializer 이름 목록 같음 (dyn·b1)": r["node_names_equal_dyn"] and r["node_names_equal_b1"],
            })
    finally:
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()
    res["checks"] = checks
    print(json.dumps(res, ensure_ascii=False, indent=1))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    print("selftest 통과")


# --------------------------------------------------------------------------- #
# 5. 실제 모델 명령 (s2common: 환경 변수·데이터 필요)
# --------------------------------------------------------------------------- #
def _setup():
    sys.path.insert(0, str(HERE))
    import s2common as S
    from scripts import env_info
    return S, S.RC, env_info


def _row(env, a, **kw):
    from scripts import env_info
    return {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
            "dirty": env["git"]["dirty"], **env_info.flat(env), "cpu_flags": env["cpu"]["flags"],
            "options": {k: v for k, v in vars(a).items() if k != "func"}, **kw}


def _append(out, name, row):
    with open(out / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _q_path(RC, qn):
    return RC.ONNX_DIR / f"rotq_{qn}.npy"


def _load_q(RC, qn):
    p = _q_path(RC, qn)
    assert p.exists(), f"{p.name} 없음 — 먼저 build"
    Q = np.load(p)
    # 저장한 파일이 기준(sha256 기록). QR은 스레드 수에 따라 마지막 자리가 달라지므로 다시 만든 Q와는 근사로만 비교
    assert np.abs(Q - make_q(qn)).max() < 1e-10, f"{p.name}가 지금 코드로 만든 Q와 다름"
    c = q_checks(Q)
    assert c["orth_max_abs"] < 1e-10 and c["q1_minus_1_max_abs"] < 1e-10, c
    return Q, _sha256(p)


def _dev64(RC, n):
    x = np.concatenate([np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:n // 2]) for d in DEV64])
    ref = np.concatenate([np.load(RC.CACHE_DIR / f"{d}_ref_fp32.npy")[:n // 2] for d in DEV64])
    assert len(x) == len(ref) == n
    return x, ref


def _load_visual(S, threads):
    torch.set_num_threads(threads)
    model, preprocess, _ = S.load_model()
    unfuse(model.visual)
    return model


@torch.no_grad()
def _feat(visual, x, bs=16):
    return np.concatenate([visual(torch.as_tensor(np.ascontiguousarray(x[i:i + bs]), dtype=torch.float32)).float().numpy()
                           for i in range(0, len(x), bs)])


def cmd_build(a):
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    for qn in a.q:
        Q = make_q(qn)
        c = q_checks(Q)
        assert c["orth_max_abs"] < 1e-10 and c["q1_minus_1_max_abs"] < 1e-10, (qn, c)
        p = _q_path(RC, qn)
        if p.exists():
            assert np.abs(np.load(p) - Q).max() < 1e-10, f"{p.name}가 지금 코드로 만든 Q와 다름"   # 기존 파일 유지
        else:
            np.save(p, Q)
        row = _row(env, a, q=qn, file=p.name, sha256=_sha256(p), checks=c)
        _append(out, "build.jsonl", row)
        print(qn, p.name, row["sha256"][:12], c)


@torch.no_grad()
def _ln_out_ratios(trunk, x):
    """LN 출력(= qkv·fc1·attn_pool.kv 입력)의 텐서 전체 max|·|/p99.9 비율과 채널별 max의 최대/중앙값 비율."""
    caps, hs = {}, []

    def hook(name):
        def f(m, i, o):            # None 반환 (출력 바꾸지 않음)
            caps.setdefault(name, []).append(i[0].detach())
        return f
    for l, blk in enumerate(trunk.blocks):
        hs.append(blk.attn.qkv.register_forward_hook(hook(f"qkv{l}")))
        hs.append(blk.mlp.fc1.register_forward_hook(hook(f"fc1{l}")))
    hs.append(trunk.attn_pool.kv.register_forward_hook(hook("pool_kv")))
    try:
        for i in range(0, len(x), 16):
            trunk(torch.as_tensor(np.ascontiguousarray(x[i:i + 16]), dtype=torch.float32))
    finally:
        for h in hs:
            h.remove()
    res = {}
    for k, v in caps.items():
        t = torch.cat(v).reshape(-1, 768).abs()
        flat = t.flatten()
        q = flat.kthvalue(max(1, int(np.ceil(0.999 * flat.numel())))).values
        ch = t.max(0).values
        res[k] = {"max_over_p999": round(float(flat.max() / q), 3), "max": round(float(flat.max()), 3),
                  "ch_max_over_median": round(float(ch.max() / ch.median()), 3)}
    return res


def cmd_fp32(a):
    """PyTorch FP32: 원래 vs 회전 (dev 64장) — Q1 미리보기 + LN 출력 범위 진단 (기록용)."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model = _load_visual(S, a.threads)
    x, ref = _dev64(RC, a.n)
    ref = l2n(ref)
    base = _feat(model.visual, x)
    rows = {"original": {"vs_ref_cos_min": float((l2n(base) * ref).sum(1).min()),
                         "ln_out": _ln_out_ratios(model.visual.trunk, x)}}
    if a.r2:                                   # (기본 꺼짐) R2 진단: 원래 모델의 attn proj 입력 범위
        rows["original"]["proj_in"] = _proj_in_ratios(model.visual.trunk, x)
    for qn in a.q:
        t0 = time.time()
        Q, qsha = _load_q(RC, qn) if qn != "none" else (None, None)
        vis = copy.deepcopy(model.visual)
        if Q is not None:
            rotate_trunk(vis.trunk, Q)
        if a.r2:
            rows.setdefault(f"{qn}_noR2", {})["proj_in"] = _proj_in_ratios(vis.trunk, x) if Q is not None else rows["original"]["proj_in"]
            rotate_r2(vis.trunk, make_r2(a.r2))
            qn = f"{qn}_r2{a.r2}"
        f = _feat(vis, x)
        rows[qn] = {"q_sha256": qsha, "vs_torch_orig_max_abs": float(np.abs(f - base).max()),
                    "vs_torch_orig_cos_min": float((l2n(f) * l2n(base)).sum(1).min()),
                    "vs_ref_cos_min": float((l2n(f) * ref).sum(1).min()),
                    "ln_out": _ln_out_ratios(vis.trunk, x), "sec": round(time.time() - t0, 1)}
        if a.r2:
            rows[qn]["proj_in"] = _proj_in_ratios(vis.trunk, x)
        del vis
    row = _row(env, a, n=int(len(x)), result=rows)
    _append(out, "fp32.jsonl", row)
    for k, r in rows.items():
        if "proj_in" in r:
            print(f"   {k}: proj 입력 채널max 최대/중앙:", [r["proj_in"][f"proj{l}"]["ch_max_over_median"] for l in range(12)])
        if "ln_out" not in r:
            continue
        lo = r["ln_out"]
        print(f"{k:9s} vs ref cos 최소 {r['vs_ref_cos_min']:.7f}"
              + (f"  vs 원래 PyTorch 최대 차이 {r['vs_torch_orig_max_abs']:.2e}" if k != "original" else ""))
        print("   qkv 입력 max/p99.9:", [lo[f"qkv{l}"]["max_over_p999"] for l in range(12)])
        print("   qkv 입력 채널max 최대/중앙:", [lo[f"qkv{l}"]["ch_max_over_median"] for l in range(12)])
        print("   fc1 입력 max/p99.9:", [lo[f"fc1{l}"]["max_over_p999"] for l in range(12)], " pool_kv", lo["pool_kv"])


def cmd_export(a):
    """회전 모델 → siglip2_image_fp32_<q>.onnx (+ _b1). Q1 판정: ORT 임베딩 vs ref_fp32 (dev 64장) cos 최소 ≥ 0.99999."""
    S, RC, env_info = _setup()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    model = _load_visual(S, a.threads)
    assert a.q != "none" or a.r2, "--q none은 --r2와 함께만"
    Q, qsha = _load_q(RC, a.q) if a.q != "none" else (None, None)
    stem = f"siglip2_image_fp32_{a.q}"
    if a.r2:                                   # (기본 꺼짐) 파일 이름: _<q>_r2<r2>, q 없음이면 _r2<r2>
        stem = f"siglip2_image_fp32_r2{a.r2}" if a.q == "none" else f"{stem}_r2{a.r2}"
    paths = [RC.ONNX_DIR / f"{stem}.onnx", RC.ONNX_DIR / f"{stem}_b1.onnx"]
    for p in paths:
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p.name} 이미 있음 (--overwrite)")
    # 원래 그래프 서명: 배치 가변은 Release FP32 ONNX, b1은 같은 코드로 임시 export (이 PC에는 원래 b1 파일이 없음)
    rel = RC.ONNX_DIR / "siglip2_image_fp32.onnx"
    sig = {"dyn": graph_sig(rel)}
    with tempfile.TemporaryDirectory(dir=a.tmp) as td:
        pb = Path(td) / "orig_b1.onnx"
        export_onnx(model.visual, pb, False, a.opset)
        sig["b1"] = graph_sig(pb)
    vis = copy.deepcopy(model.visual)
    if Q is not None:
        rotate_trunk(vis.trunk, Q)
    if a.r2:
        rotate_r2(vis.trunk, make_r2(a.r2))
    del model
    files = {}
    for p, dyn, tag in zip(paths, (True, False), ("dyn", "b1")):
        n_undedup = export_onnx(vis, p, dyn, a.opset)
        ops, names, inits = graph_sig(p)
        files[p.name] = {"MiB": round(p.stat().st_size / 2**20, 2), "sha256": _sha256(p), "ops": ops,
                         "ops_equal_orig": ops == sig[tag][0],
                         "node_names_equal_orig": names == sig[tag][1] and inits == sig[tag][2],
                         "n_initializer": len(inits), "n_initializer_orig": len(sig[tag][2]),
                         "identity_removed": n_undedup}
        print(p.name, files[p.name]["MiB"], "MiB", "연산 같음" if files[p.name]["ops_equal_orig"] else "연산 다름!",
              "노드 이름 같음" if files[p.name]["node_names_equal_orig"] else "노드 이름 다름!")
    x, ref = _dev64(RC, a.verify)
    ref = l2n(ref)
    ft = l2n(_feat(vis, x))
    verify = {"torch_rot_vs_ref_cos_min": float((ft * ref).sum(1).min())}
    for p in paths:
        e = l2n(ort_run(p, x, 1 if p.name.endswith("_b1.onnx") else 16, a.threads))
        c = (e * ref).sum(1)
        verify[p.name] = {"n": int(len(x)), "cos_min": float(c.min()), "cos_mean": float(c.mean()),
                          "max_abs_err_normalized": float(np.abs(e - ref).max()),
                          "pass_q1_cos_ge_0.99999": bool(c.min() >= 0.99999)}
        print(p.name, verify[p.name])
    row = _row(env, a, q=a.q, q_sha256=qsha, files=files, verify=verify, **({"r2": a.r2} if a.r2 else {}),
               ref_onnx={"file": rel.name, "sha256": _sha256(rel)})
    _append(out, "export.jsonl", row)
    assert all(f["ops_equal_orig"] and f["node_names_equal_orig"] for f in files.values()), "그래프가 원래와 다름"
    assert all(verify[p.name]["pass_q1_cos_ge_0.99999"] for p in paths), "Q1 실패: cos 최소 < 0.99999 → 구현 버그, 멈춤"
    print("Q1 통과")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--tmp", default=None, help="임시 ONNX 폴더 (기본: 시스템 임시 폴더)")
    p.add_argument("--r2", default="", choices=["", *R2S], help="(기본 꺼짐) 헤드 안쪽 회전 R2만 검사")
    p.set_defaults(func=cmd_selftest)
    p = sub.add_parser("build")
    p.add_argument("--q", nargs="+", default=QS, choices=QS)
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("fp32")
    p.add_argument("--q", nargs="+", default=QS, choices=[*QS, "none"])
    p.add_argument("--r2", default="", choices=["", *R2S], help="(기본 꺼짐) 각 --q 위에 헤드 안쪽 회전 R2를 얹고 attn proj 입력 범위 기록")
    p.add_argument("--n", type=int, default=64, help="dev 사진 수 (Pets·Caltech dev 앞 n/2장씩)")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=cmd_fp32)
    p = sub.add_parser("export")
    p.add_argument("--q", required=True, choices=[*QS, "none"], help="none = 잔차 회전 없음 (--r2와 함께만)")
    p.add_argument("--r2", default="", choices=["", *R2S], help="(기본 꺼짐) 헤드 안쪽 회전 R2를 얹는다 (G3Q-R2)")
    p.add_argument("--verify", type=int, default=64, help="Q1 dev 사진 수 (Pets·Caltech dev 앞 n/2장씩)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tmp", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_export)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
