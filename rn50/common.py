"""
common.py — 모든 스크립트가 공유하는 설정과 도구 모음
=====================================================

[이 파일의 목적]
  여러 스크립트가 "같은 조건"으로 실험하도록 전처리, 토크나이저, 데이터 구간, 프롬프트,
  경로, ONNX 세션 생성 방법을 한곳에 모아 둔다. 조건이 한 군데라도 다르면
  실험 결과를 서로 비교할 수 없기 때문이다(재현성의 핵심).

[담고 있는 것]
  1. 경로 설정 (환경변수로 바꿀 수 있음)
  2. CLIP 전처리 / 토크나이저 / L2 정규화
  3. 데이터 구간 규칙 (보정 / 검증 / 테스트를 절대 섞지 않기 위한 약속)
  4. ONNX Runtime 세션 생성 (스레드 수 고정)
  5. 이미지 임베딩 계산, 제로샷 분류기 생성, 고정 무작위 추출
"""
import json
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# 1. 경로 — 기본값은 이 폴더 기준. 다른 위치를 쓰려면 환경변수로 덮어쓴다.
#    예) RN50_ONNX=../rn50_quant/onnx python evaluate.py ...
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RN50_DATA", HERE / "data"))          # CIFAR-100 (torchvision 형식)
EXTRA_ROOT = Path(os.environ.get("RN50_EXTRA", HERE / "extra_data"))  # 고화질 데이터셋 원본 (parquet/tar)
ONNX_DIR = Path(os.environ.get("RN50_ONNX", HERE / "onnx"))           # ONNX 모델 저장 위치
CACHE_DIR = Path(os.environ.get("RN50_CACHE", HERE / "cache"))        # 전처리된 사진·분류기·기준 임베딩
RESULTS_DIR = Path(os.environ.get("RN50_RESULTS", HERE / "results"))  # 실험 결과 기록(jsonl/json)

# 모델 선택: 환경변수 CLIP_MODEL=RN50(기본) 또는 RN50x4. 입력 크기와 파일 이름 앞부분이 바뀐다.
#   RN50x4는 입력 288×288, 캐시·결과 폴더도 RN50과 섞이지 않게 따로 지정할 것(RN50_CACHE 등).
MODEL = os.environ.get("CLIP_MODEL", "RN50")
IMG_SIZE = {"RN50": 224, "RN101": 224, "RN50x4": 288, "RN50x16": 384}[MODEL]
PREFIX = MODEL.lower()                 # 파일 이름 앞부분: rn50 / rn50x4
FP32_IMAGE = f"{PREFIX}_image_fp32.onnx"   # 원본(양자화 없음) 사진 인코더. 유사도 기준값을 만들 때 항상 이 파일 사용
FP32_TEXT = f"{PREFIX}_text_fp32.onnx"     # 원본 글 인코더. 분류기는 항상 FP32 글 인코더로 만든다(사진 쪽만 비교하기 위해)

# --------------------------------------------------------------------------- #
# 2. CLIP 전처리 — OpenAI CLIP 원본과 동일해야 정확도가 논문과 맞는다.
#    짧은 변을 224로 bicubic 리사이즈 → 가운데 224×224 자르기 → 채널별 평균/표준편차 정규화
# --------------------------------------------------------------------------- #
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess():
    """PIL 이미지 → (3,IMG_SIZE,IMG_SIZE) float 텐서로 바꾸는 변환 함수를 돌려준다."""
    import open_clip
    return open_clip.image_transform(IMG_SIZE, is_train=False, mean=MEAN, std=STD,
                                     interpolation="bicubic", resize_mode="shortest")


def crop_u8(img):
    """PIL 이미지 → 정규화 직전의 224×224 uint8 배열 (3,224,224).
    CLIP 전처리의 Resize·CenterCrop 결과는 원래 uint8이므로, 이것을 저장해 두고 to_input()으로
    정규화하면 preprocess()와 완전히 같은 입력이 된다. float32 저장 대비 디스크 1/4."""
    t = preprocess().transforms
    img = t[1](t[0](img.convert("RGB")))           # Resize(bicubic, 짧은 변 224) → CenterCrop(224)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()


_MEAN = np.array(MEAN, np.float32)[:, None, None]
_STD = np.array(STD, np.float32)[:, None, None]


def to_input(x_u8):
    """uint8 (N,3,224,224) → 모델 입력 float32. ToTensor(/255) + Normalize와 같은 계산."""
    return ((np.asarray(x_u8, np.float32) / 255.0 - _MEAN) / _STD).astype(np.float32)


def tokenizer():
    """문장 → 길이 77 토큰 배열. CLIP-RN50 전용 BPE 토크나이저."""
    import open_clip
    return open_clip.get_tokenizer(MODEL)   # CLIP ResNet 계열은 모두 같은 BPE 토크나이저(길이 77)


def l2n(x):
    """벡터 길이를 1로 맞춘다. 코사인 유사도 = 정규화된 벡터끼리의 내적."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# 3. 데이터 구간 규칙 (누수 방지)
#    - 보정(캘리브레이션): CIFAR-100 train[0:45000] 안에서만 무작위 추출
#    - 설정 비교(검증)   : CIFAR-100 train[45000:46000] = 검증 A, [46000:47000] = 검증 B
#    - 최종 보고         : 각 데이터셋 test 전체. 설정을 고르는 데에는 절대 쓰지 않는다
# --------------------------------------------------------------------------- #
CALIB_RANGE = (0, 45000)
VAL_SPLITS = {"A": 45000, "B": 46000}  # 각 1000장
VAL_SIZE = 1000
SUBSET_SEED = 0                         # 고화질 데이터셋 2000장 부분집합 추출용 고정 시드
SUBSET_N = 2000

# CIFAR-100 제로샷 프롬프트: CLIP 논문 18개 템플릿. "{c}" 자리에 클래스 이름이 들어간다
TEMPLATES = json.load(open(HERE / "prompts" / "cifar100_templates.json"))["templates"]


def cifar100(train=False, download=True, root=None):
    """CIFAR-100을 CLIP 전처리와 함께 불러온다. 클래스 이름의 '_'는 공백으로 바꾼다."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(str(root or DATA_ROOT), train=train, download=download, transform=preprocess())
    return ds, [c.replace("_", " ") for c in ds.classes]


def fixed_sample(n_total, n, seed=SUBSET_SEED):
    """0..n_total-1 중 n개를 시드 고정으로 뽑아 정렬해서 돌려준다(누가 돌려도 같은 사진)."""
    return np.sort(np.random.default_rng(seed).choice(n_total, min(n, n_total), replace=False))


# --------------------------------------------------------------------------- #
# 4. ONNX Runtime 세션 — 스레드 수를 고정해야 속도 비교가 공정하다
# --------------------------------------------------------------------------- #
def session(path, threads=2, quiet=True, fake_quant=False):
    """fake_quant=True: 그래프 최적화를 끄고(ORT_DISABLE_ALL) Q/DQ를 그대로 실행 → 정수 커널 대신
    '반올림만 흉내 낸 FP32 계산'. CPU 종류(VNNI 유무)와 무관하게 같은 결과가 나와서 정확도 평가용으로 쓴다.
    fake_quant=False: 정수 커널(QLinearConv 등)로 융합 → 실제 속도 측정용.
    두 방식의 출력은 같지 않다(검증: 코사인 약 0.993). 정확도와 속도를 어느 방식으로 쟀는지 꼭 함께 기록할 것."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads           # 한 연산 안에서 쓰는 코어 수
    so.inter_op_num_threads = 1                 # 연산 여러 개를 동시에 돌리지 않음(측정 흔들림 감소)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if fake_quant
                                   else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    if quiet:
        so.log_severity_level = 3               # 경고 로그 숨김
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_path(name):
    """파일 이름만 주면 ONNX_DIR 아래 경로로, 경로를 주면 그대로 쓴다."""
    p = Path(name)
    return p if p.exists() else ONNX_DIR / name


# --------------------------------------------------------------------------- #
# 5. 임베딩 / 분류기
# --------------------------------------------------------------------------- #
def embed(model, x, threads=2, bs=50, fake_quant=True):
    """사진 배열 x → 정규화된 사진 임베딩 (N,1024). x는 uint8(캐시) 또는 이미 정규화된 float32.
    기본은 fake-quant 실행(정확도 평가용, CPU와 무관하게 재현).
    주의: 실행 시간에 세션 생성 시간이 섞이므로 속도 측정에 쓰지 말 것(bench_speed.py 사용)."""
    s = session(onnx_path(model), threads, fake_quant=fake_quant)
    conv = to_input if np.asarray(x[:1]).dtype == np.uint8 else (lambda b: np.ascontiguousarray(b, np.float32))
    out = [s.run(None, {"image": conv(x[i:i + bs])})[0] for i in range(0, len(x), bs)]
    return l2n(np.concatenate(out))


def zeroshot_classifier(classnames, templates, threads=2):
    """보기(클래스)마다 '템플릿 × 클래스 이름' 문장들을 FP32 글 인코더에 넣고,
    정규화 → 평균 → 다시 정규화해서 클래스 대표 벡터 1개를 만든다(프롬프트 앙상블).
    templates는 '{}' 또는 '{c}' 자리표시자 어느 쪽이든 받는다.
    반환: (1024, 클래스 수) — 사진 임베딩 @ W 하면 클래스별 점수."""
    s = session(onnx_path(FP32_TEXT), threads)
    tok = tokenizer()
    W = []
    for c in classnames:
        sents = [t.format(c, c=c) for t in templates]
        e = l2n(s.run(None, {"tokens": tok(sents).numpy().astype(np.int64)})[0]).mean(0)
        W.append(e / np.linalg.norm(e))
    return np.stack(W, 1).astype(np.float32)


def topk_metrics(feat, W, y, ref=None):
    """top-1, top-5 정확도(%)와, 기준 임베딩 ref가 있으면 평균 코사인 유사도를 계산."""
    top5 = np.argsort(-(feat @ W), 1)[:, :5]
    r = {"top1": round(100 * float((top5[:, 0] == y).mean()), 2),
         "top5": round(100 * float((top5 == y[:, None]).any(1).mean()), 2)}
    if ref is not None:
        r["cos_mean"] = round(float((feat * ref).sum(1).mean()), 4)
        r["cos_p1"] = round(float(np.percentile((feat * ref).sum(1), 1)), 4)  # 가장 나쁜 1% 지점
    return r


def append_jsonl(name, row):
    """결과 한 줄을 results/<name>.jsonl에 바로 기록(중간에 꺼져도 앞의 결과는 남는다)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.jsonl", "a", buffering=1, encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
