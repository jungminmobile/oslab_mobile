"""
te_common.py — SigLIP2 텍스트 인코더 INT8 스모크 공통 부품 (textenc/ 전용)
=====================================================================

[목적]
  팀 규칙(PRD R6)은 "텍스트 인코더는 FP32로 클래스 임베딩을 1회 계산"이다. 이 폴더는 그 규칙을 바꾸지 않고,
  텍스트 인코더 INT8이 가능한지·할 가치가 있는지만 스모크로 확인한다.
  공통 규칙(경로, 데이터셋 템플릿·클래스 이름, ONNX 세션, 지표, 환경 기록)은 siglip2/·rn50/ 코드를 import한다 (복사 없음).

[경로]
  siglip2/s2common.py가 RN50_* 경로를 SigLIP2 쪽으로 맞춘다 (SIGLIP2_DATA·SIGLIP2_ARTIFACTS 필요).
  텍스트 쪽 큰 파일(ONNX, 토큰 .npy)은 $SIGLIP2_ARTIFACTS/textenc/ 에 둔다. 결과(jsonl·README)는 results/textenc/.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "siglip2"))
import s2common as S            # noqa: E402  (RN50_* 경로 설정 + rn50 common·datasets import)
from scripts import env_info    # noqa: E402

RC, RD = S.RC, S.RD
TE_DIR = Path(os.environ["SIGLIP2_ARTIFACTS"]) / "textenc"
FP32 = "siglip2_text_fp32.onnx"
CTX = 64                         # open_clip ViT-B-16-SigLIP2 text_cfg.context_length
EVAL_SETS = ("pets", "food")     # 정확도 스모크 데이터셋 (dev만, test 금지)
DRY = bool(os.environ.get("TE_DRYRUN"))   # 1이면 랜덤 가중치·가짜 토크나이저·가짜 dev로 코드 경로만 확인 (결과로 쓰지 않음)
EXPECTED_PARAMS = {"text": 282_303_744, "token_embedding": 196_608_000, "blocks": 85_054_464,
                   "proj": 590_592, "ln_final": 1_536, "pos": 49_152}


def te_path(name):
    p = Path(name)
    return p if p.is_absolute() or p.exists() else TE_DIR / name


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    """siglip2 env_info.git_state + textenc/ 아래 커밋 안 된 코드도 dirty로 본다 (env_info는 siglip2/만 본다)."""
    g = env_info.git_state(REPO)
    extra = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all", "--", "textenc"],
                           capture_output=True, text=True, check=True).stdout.strip()
    g["dirty"] = bool(g["dirty"] or extra)
    return g


def env_row():
    env = env_info.collect()
    env["git"] = git_state()
    return env


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_row(env, options):
    return {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
            "dirty": env["git"]["dirty"], "options": options, **env_info.flat(env)}


class _FakeTok:
    """TE_DRYRUN 전용: 단어 해시 → id, 끝에 eos(1), 오른쪽 pad(0). 실제 결과에 쓰지 않는다."""
    tokenizer = None

    def __call__(self, sents):
        import torch
        out = torch.zeros(len(sents), CTX, dtype=torch.long)
        for i, s in enumerate(sents):
            ids = [3 + int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 255990 for w in s.split()][:CTX - 1] + [1]
            out[i, :len(ids)] = torch.tensor(ids)
        return out


def load_model():
    if not DRY:
        return S.load_model()
    import open_clip
    import torch
    torch.manual_seed(0)
    m = open_clip.create_model("ViT-B-16-SigLIP2", pretrained=None).eval()
    torch.set_grad_enabled(False)
    return m, None, _FakeTok()


# --------------------------------------------------------------------------- #
# 문장 → 토큰
# --------------------------------------------------------------------------- #
def eval_prompts(name):
    """dev 평가용 클래스 프롬프트: 팀 dev 캐시와 같은 클래스 이름·템플릿 (rn50/datasets.py HIGHRES)."""
    templates = RD.HIGHRES[name]["templates"]
    names = [f"{name} class {i}" for i in range({"pets": 37, "food": 101}[name])] if DRY else RD.classnames(name)
    sents = [t.format(c, c=c) for c in names for t in templates]
    return names, templates, sents


def calib_sentences(n, seed, exclude_names):
    """보정·점검용 문장: CLIP 공식 ImageNet 템플릿(80개) × ImageNet 클래스 이름.
    평가 데이터셋(Pets·Food) 클래스 이름과 겹치는 ImageNet 클래스는 뺀다 (평가 프롬프트와 같은 문장이 보정에 섞이지 않게).
    시드 고정으로 뽑으므로 누가 돌려도 같은 문장."""
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
    ex = [e.lower() for e in exclude_names]
    keep = [c for c in IMAGENET_CLASSNAMES if not any(e in c.lower() or c.lower() in e for e in ex)]
    pool = [t(c) for c in keep for t in OPENAI_IMAGENET_TEMPLATES]
    idx = np.random.default_rng(seed).choice(len(pool), n, replace=False)
    return [pool[i] for i in idx], {"pool": len(pool), "imagenet_classes_kept": len(keep),
                                    "imagenet_classes_dropped": len(IMAGENET_CLASSNAMES) - len(keep)}


def tokenize(tok, sents, bs=256):
    out = np.concatenate([tok(sents[i:i + bs]).numpy() for i in range(0, len(sents), bs)]).astype(np.int64)
    assert out.shape == (len(sents), CTX), out.shape
    return out


# --------------------------------------------------------------------------- #
# ONNX 텍스트 인코더 실행 · 분류기
# --------------------------------------------------------------------------- #
def encode(sess, tokens, bs=32):
    """토큰 (N,64) int64 → 정규화 전 임베딩 (N,768)."""
    return np.concatenate([sess.run(None, {"tokens": tokens[i:i + bs]})[0] for i in range(0, len(tokens), bs)])


def classifier(raw, n_classes, n_templates):
    """s2common.zeroshot_classifier와 같은 앙상블: 문장마다 정규화 → 클래스별 평균 → 다시 정규화. 반환 (768, C)."""
    e = RC.l2n(raw.astype(np.float32)).reshape(n_classes, n_templates, -1).mean(1)
    return RC.l2n(e).T.astype(np.float32)


def load_dev(name):
    """팀 dev 캐시(s2_build_dev.py가 만든 것): 정답, 팀 FP32 분류기, FP32 이미지 임베딩."""
    if DRY:
        rng = np.random.default_rng(0)
        C = {"pets": 37, "food": 101}[name]
        W = np.load(TE_DIR / f"clf_{name}_fp32_ort.npy")
        y = rng.integers(0, C, 300)
        ref = RC.l2n(W[:, y].T + 0.05 * rng.standard_normal((300, W.shape[0])).astype(np.float32))
        return y, W, ref
    f = lambda k: RC.CACHE_DIR / f"{name}_dev_{k}.npy"
    return np.load(f("y")), np.load(f("clf")), np.load(f("ref_fp32"))
