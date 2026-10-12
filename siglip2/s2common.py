"""
s2common.py — SigLIP2 어댑터: RN50 공통 코드를 import하고 모델에 따라 달라지는 부분만 둔다 (D-009)
=============================================================================================

[목적]
  RN50 트랙의 공통 규칙(데이터 구간, 사진 선택 해시, 프롬프트, 지표, ONNX 세션)은 복사하지 않고 import한다.
  SigLIP2에서 달라지는 것만 여기에 둔다.
    - 모델·전처리·토크나이저: open_clip이 모델 설정에서 읽은 값을 그대로 쓴다 (CLIP 평균·표준편차를 쓰지 않음)
    - 제로샷 분류기: RN50과 같은 프롬프트 앙상블(정규화 → 평균 → 정규화), 단 FP32 PyTorch 글 인코더로 계산

[모듈 이름 규칙]
  이 모듈은 rn50/을 sys.path 맨 앞에 넣는다. 그래서 siglip2 쪽 파일이 rn50 파일과 이름이 같으면(export_onnx.py 등)
  rn50 쪽이 불려 온다. siglip2 쪽 모듈은 모두 이름 앞에 s2_를 붙인다 (D-019).

[경로 규칙 (D-013)]
  RN50 코드는 import 시점에 RN50_* 환경변수를 읽는다. 캐시·결과 파일 이름에 모델 이름이 없어서,
  설정하지 않으면 RN50의 파일을 덮어쓴다. 그래서 이 모듈은 import 전에 RN50_*를 SigLIP2 경로로 채우고,
  다른 곳을 가리키면 멈춘다.
    RN50_DATA    = $SIGLIP2_DATA/cifar100          (CIFAR-100 원본, torchvision 형식)
    RN50_EXTRA   = $SIGLIP2_DATA/extra             (고화질 데이터셋 원본 parquet/tar)
    RN50_CACHE   = $SIGLIP2_ARTIFACTS/cache        (SigLIP2 전처리 캐시. RN50 캐시와 섞지 않음)
    RN50_ONNX    = $SIGLIP2_ARTIFACTS/onnx
    RN50_RESULTS = $SIGLIP2_ARTIFACTS/results_tmp  (실험 결과는 실행 때 results/siglip2/<단계>_<주제>로 지정)
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RN50_DIR = REPO / "rn50"
MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP2"


# Windows 콘솔·리디렉션 기본 인코딩(cp949)은 '—', '▁' 같은 문자를 출력하지 못해 스크립트가 멈춘다 → UTF-8로 고정
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def _setup_paths():
    data = Path(os.environ["SIGLIP2_DATA"])          # 없으면 KeyError로 멈춤 (siglip2/CLAUDE.md 10절)
    art = Path(os.environ["SIGLIP2_ARTIFACTS"])
    defaults = {"RN50_DATA": data / "cifar100", "RN50_EXTRA": data / "extra", "RN50_CACHE": art / "cache",
                "RN50_ONNX": art / "onnx", "RN50_RESULTS": art / "results_tmp"}
    global RESULTS_EXPLICIT
    RESULTS_EXPLICIT = "RN50_RESULTS" in os.environ   # 결과를 쓰는 스크립트는 이 값이 True여야 한다 (require_results_dir)
    for k, v in defaults.items():
        os.environ.setdefault(k, str(v))
    # RN50 폴더 안을 가리키면 RN50 캐시·결과를 덮어쓰므로 멈춘다
    for k in defaults:
        p = Path(os.environ[k]).resolve()
        assert RN50_DIR.resolve() not in (p, *p.parents), f"{k}={p} 가 rn50/ 안을 가리킴 — RN50 파일 덮어쓰기 위험"
    for k in ("RN50_CACHE", "RN50_ONNX"):
        p = Path(os.environ[k]).resolve()
        assert art.resolve() in (p, *p.parents), f"{k}={p} 는 SIGLIP2_ARTIFACTS 아래여야 함 (D-013)"
    # RN50 결과는 rn50/ 밖의 results/E*에도 있다. SigLIP2 결과는 results/siglip2 또는 artifacts 아래에만 쓴다
    p = Path(os.environ["RN50_RESULTS"]).resolve()
    ok = [(REPO / "results" / "siglip2").resolve(), art.resolve()]
    assert any(o in (p, *p.parents) for o in ok), f"RN50_RESULTS={p} 는 results/siglip2 또는 SIGLIP2_ARTIFACTS 아래여야 함"


RESULTS_EXPLICIT = False
_setup_paths()
if str(RN50_DIR) not in sys.path:
    sys.path.insert(0, str(RN50_DIR))
import common as RC      # noqa: E402  RN50 공통 규칙 (rn50/common.py)
import datasets as RD    # noqa: E402  RN50 데이터 규칙 (rn50/datasets.py)

# 같은 이름의 다른 패키지(예: Hugging Face datasets)를 잘못 불러오지 않았는지 확인
assert Path(RC.__file__).resolve().parent == RN50_DIR.resolve(), RC.__file__
assert Path(RD.__file__).resolve().parent == RN50_DIR.resolve(), RD.__file__


def require_results_dir():
    """결과를 기록하는 스크립트용: RN50_RESULTS를 명시하지 않으면 멈춘다 (기본값 artifacts/results_tmp에 조용히 쌓이는 것 방지)."""
    assert RESULTS_EXPLICIT, "RN50_RESULTS=../results/siglip2/<단계>_<주제> 를 지정해서 실행 (D-012)"
    return RC.RESULTS_DIR


def load_model(device="cpu"):
    """SigLIP2 모델·전처리·토크나이저. 전처리는 모델 설정(preprocess_cfg)에서 온 것을 그대로 쓴다."""
    import open_clip
    import torch
    model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    torch.set_grad_enabled(False)
    return model, preprocess, tokenizer


def preprocess_cfg(model):
    """기록용: 모델이 쓰는 전처리 설정 (크기, 평균·표준편차, 보간, 리사이즈 방식)."""
    cfg = getattr(model.visual, "preprocess_cfg", None)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()} if cfg else None


def zeroshot_classifier(model, tokenizer, classnames, templates, bs=256):
    """RN50 common.zeroshot_classifier와 같은 프롬프트 앙상블: 문장마다 정규화 → 클래스별 평균 → 다시 정규화.
    templates는 '{}' 또는 '{c}' 자리표시자 어느 쪽이든 받는다. 반환: (임베딩 차원, 클래스 수) float32."""
    import torch
    W = []
    for c in classnames:
        sents = [t.format(c, c=c) for t in templates]
        e = torch.cat([model.encode_text(tokenizer(sents[i:i + bs])) for i in range(0, len(sents), bs)])
        e = RC.l2n(e.float().numpy()).mean(0)
        W.append(e / np.linalg.norm(e))
    return np.stack(W, 1).astype(np.float32)


def embed_torch(model, x, bs=32):
    """전처리된 float 텐서 (N,3,H,W) → 정규화된 이미지 임베딩 (N,D) numpy. FP32 PyTorch 경로."""
    import torch
    x = torch.as_tensor(x)
    out = [model.encode_image(x[i:i + bs]).float().numpy() for i in range(0, len(x), bs)]
    return RC.l2n(np.concatenate(out))
