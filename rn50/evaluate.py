"""
evaluate.py — 양자화 모델의 정확도와 "원본과의 유사도" 비교 (속도는 재지 않음)
==========================================================================

[목적]
  datasets.py가 만든 캐시(같은 사진·정답·분류기·FP32 기준 임베딩)를 이용해
  여러 ONNX 사진 인코더를 공정하게 비교한다. 이전의 screen.py(CIFAR 검증)와
  eval_extra.py(고화질 데이터셋)를 하나로 합친 것.

[지표]
  - top1 / top5 : 제로샷 분류 정확도(%). 분류기는 항상 FP32 글 인코더로 만든 것을 써서
                  "사진 인코더만" 바뀐 효과를 본다.
  - cos_mean    : 같은 사진에 대한 FP32 원본 임베딩과의 평균 코사인 유사도(1이면 완전히 같음).
                  정답이 필요 없고 흔들림이 작아서 "어느 층을 보호할지" 고를 때 주 지표로 쓴다.
  - cos_p1      : 유사도가 가장 나쁜 1% 지점(최악의 경우를 보기 위함).

[사용]
  python evaluate.py --data cifar_A pets_dev flowers_dev --models fp32 rn50_image_int8_xxx.onnx
  · "fp32"는 캐시된 기준 임베딩을 그대로 사용(다시 계산하지 않음)
  · 결과는 모델 하나 끝날 때마다 results/evaluate.jsonl 에 바로 기록된다
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np

import common as C


def load(name):
    """캐시 4종을 읽는다. 사진(x)은 메모리를 아끼려고 mmap으로 연다."""
    f = lambda k: C.CACHE_DIR / f"{name}_{k}.npy"
    if not f("x").exists():
        raise SystemExit(f"{f('x')} 없음 — 먼저 `python datasets.py dev {name.removesuffix('_dev')}` 실행")
    return np.load(f("x"), mmap_mode="r"), np.load(f("y")), np.load(f("clf")), np.load(f("ref_fp32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dev 캐시 이름: cifar_A cifar_B pets_dev flowers_dev cars_dev food_dev caltech_dev")
    ap.add_argument("--models", nargs="+", required=True, help="'fp32' 또는 ONNX 파일 이름/경로")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--exec", choices=["fake", "int"], default="fake",
                    help="fake: Q/DQ를 FP32로 흉내(정확도 평가 기본, CPU 무관 재현) / int: 정수 커널(실제 CPU 동작)")
    a = ap.parse_args()
    for d in a.data:
        x, y, W, ref = load(d)
        for m in a.models:
            t0 = time.time()
            feat = ref if m == "fp32" else C.embed(m, x, a.threads, fake_quant=a.exec == "fake")
            row = {"date": datetime.now().isoformat(timespec="seconds"), "data": d, "n": int(len(y)),
                   "model": m if m == "fp32" else C.onnx_path(m).name,
                   "MiB": None if m == "fp32" else round(C.onnx_path(m).stat().st_size / 2**20, 1),
                   "exec": a.exec, **C.topk_metrics(feat, W, y, ref), "eval_sec": round(time.time() - t0)}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            C.append_jsonl("evaluate", row)


if __name__ == "__main__":
    main()
