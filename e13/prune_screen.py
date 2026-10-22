"""prune_screen.py — E13-1 하위층 빼기 사전 확인 (FP32) 공통 코드. 계획서 results/E13-1_하위층빼기_사전확인/*.md

- 모델: open_clip timm `hf-hub:timm/ViT-B-32-SigLIP2-256` FP32 PyTorch CPU (oslab_g8/siglip2/s2common.load_model 재사용)
- 하위층 빼기: trunk.blocks[b].attn 또는 .mlp 의 forward 출력을 0으로 바꾸는 hook (잔차 x는 그대로 흐름).
  가중치·M2·회전·양자화 없음.
- 데이터: dev 원본(oslab_g8 s2_g6pre._raw_highres와 같은 사진 선택) → PIL RGB → BICUBIC 256×256 → uint8 → x/127.5 − 1
- 분류기: s2common.zeroshot_classifier (open_clip 글 인코더 FP32, CLIP 템플릿: CIFAR = RC.TEMPLATES, 고화질 = RD.HIGHRES 템플릿)
팀 원본 코드(oslab_g8)는 import만 하고 고치지 않는다.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                     # oslab_mobile
ROOT = REPO.parent                                     # 2026 모바일 학술대회
G8 = ROOT / "oslab_g8" / "siglip2"                     # 팀 siglip2/g8-candidates (읽기만)
MODEL_ID = "hf-hub:timm/ViT-B-32-SigLIP2-256"
THREADS = 4
ART = Path(os.environ.get("E13_ART", r"C:\s2art_b32"))  # B/32 산출물 (C:\s2art 는 B/16용이라 섞지 않음)
B16_CACHE = Path(r"C:\s2art\cache")                    # 사진 목록 대조용 *_dev_sha1.txt (읽기만)
OUT = REPO / "results" / "E13-1_하위층빼기_사전확인" / "results"

os.environ["S2_MODEL_ID"] = MODEL_ID
os.environ.setdefault("SIGLIP2_DATA", str(ROOT / "s2data"))
os.environ.setdefault("SIGLIP2_ARTIFACTS", str(ART))
os.environ["PYTHONIOENCODING"] = "utf-8"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = str(THREADS)

import numpy as np    # noqa: E402
import torch          # noqa: E402

sys.path.insert(0, str(G8))
import s2common as S            # noqa: E402
RC, RD = S.RC, S.RD
torch.set_num_threads(THREADS)
torch.set_grad_enabled(False)

HR = ["pets", "flowers", "cars", "food", "caltech"]
DEV = ["cifar_A"] + [f"{n}_dev" for n in HR]
HRK = DEV[1:]
SCREEN_N = 200                                         # 스크리닝 = 각 dev 앞 200장 (G6_pre DEV_K와 같음)
D, N_TOK, MLP_H, N_BLK = 768, 64, 3072, 12
CAND_EXCL = {("attn", 0), ("mlp", 0), ("attn", 6), ("attn", 10), ("mlp", 5), ("mlp", 9)}   # 계획서 3절
CANDIDATES = [(k, b) for b in range(N_BLK) for k in ("attn", "mlp") if (k, b) not in CAND_EXCL]
assert len(CANDIDATES) == 18
PLAN_SHARE = {"attn": 2.8, "mlp": 5.5}                 # 계획서 2절 추정 연산 몫 (%)


def name(sub):
    return f"{sub[0]}{sub[1]}"


def git_state():
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    tracked = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    untracked = run("status", "--porcelain", "--untracked-files=all", "--", "e13")
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(tracked or untracked)}


def append(fname, row):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"date": datetime.now().isoformat(timespec="seconds"), **git_state(), "model_id": MODEL_ID,
           "threads": THREADS, **row}
    with open(OUT / fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:1500], flush=True)
    return row


# ---------------------------------------------------------------- 연산·가중치 몫 (해석적 계산)

def macs():
    """B/32-256 이미지 인코더 MAC (곱셈-덧셈) 해석적 계산. 하위층별 몫을 낸다."""
    n, d, h = N_TOK, D, MLP_H
    attn = n * d * 3 * d + 2 * n * n * d + n * d * d
    mlp = 2 * n * d * h
    patch = n * 3 * 32 * 32 * d
    head = d * d + n * d * 2 * d + 2 * n * d + d * d + 2 * d * h     # MAP: q, kv, attn, proj, mlp (latent 1개)
    total = patch + N_BLK * (attn + mlp) + head
    return {"attn": attn, "mlp": mlp, "patch": patch, "head": head, "total": total}


def share(subs, exact=True):
    m = macs()
    return round(100 * sum(m[k] for k, _ in subs) / m["total"], 2) if exact else round(sum(PLAN_SHARE[k] for k, _ in subs), 2)


def sub_params(model, sub):
    blk = model.visual.trunk.blocks[sub[1]]
    mods = (blk.norm1, blk.attn) if sub[0] == "attn" else (blk.norm2, blk.mlp)
    return int(sum(p.numel() for m in mods for p in m.parameters()))


# ---------------------------------------------------------------- 모델·하위층 빼기

def load():
    model, _, tok = S.load_model()
    tr = model.visual.trunk
    assert tuple(tr.patch_embed.img_size) == (256, 256) and len(tr.blocks) == N_BLK
    assert tr.patch_embed.num_patches == N_TOK, tr.patch_embed.num_patches
    for b in tr.blocks:   # 잔차 앞에 아무 것도 없는지 (ls·drop_path가 항등) — 0으로 만든 출력이 그대로 0이 되도록
        assert isinstance(b.ls1, torch.nn.Identity) and isinstance(b.ls2, torch.nn.Identity), type(b.ls1)
    return model, tok


class Skip:
    """with Skip(model, [("mlp", 3), ...]): 해당 하위층 출력을 0으로 (블록 forward의 x = x + 0)."""

    def __init__(self, model, subs):
        self.model, self.subs, self.h = model, list(subs), []

    def __enter__(self):
        for k, b in self.subs:
            mod = getattr(self.model.visual.trunk.blocks[b], k)
            self.h.append(mod.register_forward_hook(lambda m, a, out: torch.zeros_like(out)))
        return self

    def __exit__(self, *e):
        for h in self.h:
            h.remove()


def pix_to_x(pix):
    return (np.asarray(pix, np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2).copy()


def embed(model, pix, bs=32):
    out = []
    for i in range(0, len(pix), bs):
        x = torch.from_numpy(pix_to_x(pix[i:i + bs]))
        out.append(model.encode_image(x).float().numpy())
    return RC.l2n(np.concatenate(out))


class SubCos:
    """논문식 점수(기록용): 하위층 입력 x와 출력 x + f(x)의 토큰별 1−cos 평균. 전체 모델 forward 중에 잰다."""

    def __init__(self, model):
        self.tr, self.h, self.acc, self.cnt = model.visual.trunk, [], {}, 0
        self.cur = {}

    def __enter__(self):
        for b, blk in enumerate(self.tr.blocks):
            self.h.append(blk.register_forward_pre_hook(lambda m, a, b=b: self.cur.__setitem__(("in", b), a[0])))
            self.h.append(blk.norm2.register_forward_pre_hook(lambda m, a, b=b: self.cur.__setitem__(("mid", b), a[0])))
            self.h.append(blk.register_forward_hook(lambda m, a, o, b=b: self._done(b, o)))
        return self

    @staticmethod
    def _1mcos(u, v):
        return (1 - torch.nn.functional.cosine_similarity(u.double(), v.double(), dim=-1)).sum().item()

    def _done(self, b, out):
        xi, xm = self.cur[("in", b)], self.cur[("mid", b)]
        for key, val in ((("attn", b), self._1mcos(xi, xm)), (("mlp", b), self._1mcos(xm, out))):
            self.acc[key] = self.acc.get(key, 0.0) + val
        if b == 0:
            self.cnt += out.shape[0] * out.shape[1]

    def result(self):
        return {name(k): v / self.cnt for k, v in self.acc.items()}

    def __exit__(self, *e):
        for h in self.h:
            h.remove()


# ---------------------------------------------------------------- 데이터·분류기

def dev_dir():
    d = ART / "dev256"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_dev():
    """dev 전체 uint8 NHWC 256 캐시. 사진 목록은 기존 B/16 캐시의 *_dev_sha1.txt와 같아야 한다."""
    import hashlib
    import io
    from PIL import Image
    from torchvision.datasets import CIFAR100
    sys.path.insert(0, str(G8))
    import s2_g6pre as P            # _raw_highres만 쓴다 (import 시 torch 스레드를 6으로 바꾸므로 되돌림)
    torch.set_num_threads(THREADS)
    info = {}
    for n in HR:
        dev, ys, _ = P._raw_highres(n)
        old = (B16_CACHE / f"{n}_dev_sha1.txt").read_text().split()
        assert [hashlib.sha1(b).hexdigest() for b in dev] == old, f"{n} dev 사진 목록이 기존 캐시와 다름"
        pix = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((256, 256), Image.BICUBIC), np.uint8)
                        for b in dev])
        np.save(dev_dir() / f"{n}_dev_pix.npy", pix)
        np.save(dev_dir() / f"{n}_dev_y.npy", np.array(ys))
        info[f"{n}_dev"] = len(dev)
        print(n, pix.shape, flush=True)
    ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=False)
    s0 = RC.VAL_SPLITS["A"]
    pix = np.stack([np.asarray(ds[i][0].convert("RGB").resize((256, 256), Image.BICUBIC), np.uint8) for i in range(s0, s0 + RC.VAL_SIZE)])
    np.save(dev_dir() / "cifar_A_pix.npy", pix)
    np.save(dev_dir() / "cifar_A_y.npy", np.array(ds.targets[s0:s0 + RC.VAL_SIZE]))
    info["cifar_A"] = RC.VAL_SIZE
    sums = {f.name: hashlib.sha1(f.read_bytes()).hexdigest() for f in sorted(dev_dir().glob("*.npy"))}
    (dev_dir() / "SHA1SUMS.json").write_text(json.dumps(sums, indent=1))
    return info, sums


def build_clf(model, tok):
    from torchvision.datasets import CIFAR100
    cif = [c.replace("_", " ") for c in CIFAR100(str(RC.DATA_ROOT), train=True, download=False).classes]
    d = ART / "clf"
    d.mkdir(parents=True, exist_ok=True)
    for key in DEV:
        names, tpl = (cif, RC.TEMPLATES) if key == "cifar_A" else (RD.classnames(key[:-4]), RD.HIGHRES[key[:-4]]["templates"])
        W = S.zeroshot_classifier(model, tok, names, tpl)
        np.save(d / f"{key}_clf.npy", W)
        print(key, W.shape, len(tpl), flush=True)


def load_dev(key, n=None):
    pix = np.load(dev_dir() / f"{key}_pix.npy", mmap_mode="r")
    y = np.load(dev_dir() / f"{key}_y.npy")
    W = np.load(ART / "clf" / f"{key}_clf.npy")
    return (pix, y, W) if n is None else (pix[:n], y[:n], W)


def ref_path(key):
    return ART / "ref" / f"{key}_fp32.npy"


def evaluate(model, subs, n=None, keep=False):
    """subs를 뺀 모델로 dev(앞 n장 또는 전체) 평가. 전체 모델 FP32 기준(ref)과 cos·top-1 일치율·top-1 차이."""
    res, keepd = {}, {}
    with Skip(model, subs):
        for key in DEV:
            pix, y, W = load_dev(key, n)
            e = embed(model, pix)
            ref = np.load(ref_path(key), mmap_mode="r")[:len(y)]
            p, p0 = (e @ W).argmax(1), (ref @ W).argmax(1)
            res[key] = {"n": int(len(y)), "top1": 100 * float((p == y).mean()), "top1_ref": 100 * float((p0 == y).mean()),
                        "cos": float((e * ref).sum(1).mean()), "agree": 100 * float((p == p0).mean())}
            res[key]["dtop1"] = res[key]["top1"] - res[key]["top1_ref"]
            if keep:
                keepd[key] = {"correct": p == y, "correct_ref": p0 == y}
    hr = {m: float(np.mean([res[k][m] for k in HRK])) for m in ("top1", "top1_ref", "dtop1", "cos", "agree")}
    return res, hr, keepd
