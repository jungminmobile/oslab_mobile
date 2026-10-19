"""
bootstrap.py — 두 모델의 정확도 차이가 "운"인지 "진짜"인지 판정 (E0 ④, PRD 5절)
=========================================================================

[방법] 짝 부트스트랩(paired bootstrap), 데이터셋별 층화
  - 같은 사진에서 모델 A와 B가 각각 맞았는지(0/1)를 짝지어 둔다
  - 데이터셋마다 따로 사진을 복원추출(층화) → 데이터셋별 정확도 차이 → 여러 데이터셋이면 단순 평균
    ("고화질 평균" = 5종 단순 평균, PRD 5절)
  - 1000번 반복해 차이의 2.5%·97.5% 지점 = 95% 신뢰구간
  - 구간이 0을 포함하면 "유의한 차이 없음"(손실 없음이라고 쓰지 않는다)

[입력] evaluate.py --save-pred 가 만든 results/pred/<모델>__<데이터>.npz
       'fp32'는 캐시의 기준 임베딩으로 바로 계산한다

[사용]  python bootstrap.py --a fp32 --b rn50_image_int8_q1_minmax_highres256_s0_c3 \
                            --data pets_dev flowers_dev cars_dev food_dev caltech_dev
"""
import argparse
import json
from datetime import datetime

import numpy as np

import common as C


def correct(model, data):
    """사진별 정답 여부(0/1)."""
    if model == "fp32":
        ref, W, y = (np.load(C.CACHE_DIR / f"{data}_{k}.npy") for k in ("ref_fp32", "clf", "y"))
        return (np.argmax(ref @ W, 1) == y).astype(np.float64)
    z = np.load(C.RESULTS_DIR / "pred" / f"{model.removesuffix('.onnx')}__{data}.npz")
    return (z["pred"] == z["y"]).astype(np.float64)


def paired_bootstrap(pairs, n_boot=1000, seed=0):
    """pairs: [(a_correct, b_correct), ...] 데이터셋별. 반환: 점추정(%p), 95% 구간, 분포."""
    rng = np.random.default_rng(seed)
    point = np.mean([100 * (b.mean() - a.mean()) for a, b in pairs])
    dist = np.empty(n_boot)
    for i in range(n_boot):
        diffs = []
        for a, b in pairs:
            idx = rng.integers(0, len(a), len(a))          # 같은 인덱스로 A·B를 함께 뽑는다(짝)
            diffs.append(100 * (b[idx].mean() - a[idx].mean()))
        dist[i] = np.mean(diffs)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 모델('fp32' 또는 파일 이름)")
    ap.add_argument("--b", required=True, help="비교 모델")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pairs = [(correct(a.a, d), correct(a.b, d)) for d in a.data]
    for (x, y), d in zip(pairs, a.data):
        assert len(x) == len(y), f"{d}: 사진 수가 다름"
    point, lo, hi = paired_bootstrap(pairs, a.n_boot, a.seed)
    per = {d: round(100 * (y.mean() - x.mean()), 2) for (x, y), d in zip(pairs, a.data)}
    row = {"date": datetime.now().isoformat(timespec="seconds"), "a": a.a, "b": a.b, "data": a.data,
           "diff_pp": round(point, 2), "ci95": [round(lo, 2), round(hi, 2)], "includes_0": bool(lo <= 0 <= hi),
           "per_data_pp": per, "n_boot": a.n_boot, "seed": a.seed,
           "disagree_pct": {d: round(100 * float((x != y).mean()), 1) for (x, y), d in zip(pairs, a.data)}}
    print(json.dumps(row, ensure_ascii=False))
    C.append_jsonl("bootstrap", row)


if __name__ == "__main__":
    main()
