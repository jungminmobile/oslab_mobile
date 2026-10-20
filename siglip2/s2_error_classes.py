"""
s2_error_classes.py — 양자화로 생긴 오답이 어느 클래스에 몰리는가 (보고서 "결과 해석", 10/15 밤 저부하 분석)
================================================================================================

[목적] dev 전체에서 FP32는 맞혔는데 양자화 모델이 틀린 사진(새 오답)과 그 반대(새 정답)를 클래스별로 센다.
  Food(−2.6%p, DFC)처럼 손실이 큰 데이터셋에서 손실이 특정 클래스 쌍(비슷한 음식끼리)에 몰리는지,
  ORT 최종과 DFC가 같은 사진에서 틀리는지 본다. 이미 저장된 사진별 예측(npz)만 쓴다 (모델 실행 없음, 몇 초).
[출력] $RN50_RESULTS/error_classes.json, 표준 출력에 데이터셋별 상위 클래스
[사용]
  RN50_RESULTS=../results/siglip2/G3_error_classes "$SIGLIP2_PYTHON" s2_error_classes.py \
      --preds ort_u8rr=<pred npz> dfc_d256l1=<pred npz> --top 8
"""
import argparse
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
PARQUET = {"pets_dev": "pets", "flowers_dev": "flowers", "cars_dev": "cars", "food_dev": "food", "caltech_dev": "caltech"}


def class_names(d):
    """고화질은 로컬 parquet 메타데이터(파일 끝 스키마만 읽음)에서, 없으면 번호로."""
    try:
        import pyarrow.parquet as pq
        files = sorted((Path(RC.EXTRA_ROOT) / PARQUET[d]).glob("*.parquet"))
        raw = json.loads(pq.read_schema(files[0]).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        return [str(n).replace("_", " ") for n in raw]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="이름=npz (<set>_pred 필요)")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    P = {s.split("=", 1)[0]: np.load(s.split("=", 1)[1]) for s in a.preds}
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "preds": {k: s.split("=", 1)[1] for k, s in zip(P, a.preds)}, "sets": {}}
    for d in DATA:
        y = np.load(RC.CACHE_DIR / f"{d}_y.npy")
        fp = np.argmax(np.load(RC.CACHE_DIR / f"{d}_ref_fp32.npy") @ np.load(RC.CACHE_DIR / f"{d}_clf.npy"), 1)
        names = class_names(d) if d in PARQUET else None
        nm = (lambda c: names[c] if names and c < len(names) else str(int(c)))
        row = {"n": int(len(y)), "fp32_correct": int((fp == y).sum())}
        wrong_sets = {}
        for k, z in P.items():
            p = z[f"{d}_pred"]
            assert len(p) == len(y), (k, d)
            lost = (fp == y) & (p != y)                      # 새 오답
            gained = (fp != y) & (p == y)                    # 새 정답
            wrong_sets[k] = set(np.flatnonzero(lost))
            net = collections.Counter(y[lost].tolist())
            net.subtract(collections.Counter(y[gained].tolist()))
            pairs = collections.Counter(zip(y[lost].tolist(), p[lost].tolist()))
            row[k] = {"lost": int(lost.sum()), "gained": int(gained.sum()), "net_pp": round(100 * (gained.sum() - lost.sum()) / len(y), 2),
                      "top_net_loss_classes": [[nm(c), int(v)] for c, v in net.most_common(a.top) if v > 0],
                      "top_confusions": [[nm(t), nm(q), int(v)] for (t, q), v in pairs.most_common(a.top)],
                      "classes_with_loss": int(sum(v > 0 for v in net.values()))}
            print(f"{d:12s} {k:12s} lost {row[k]['lost']:3d} gained {row[k]['gained']:3d} net {row[k]['net_pp']:+.2f}%p  "
                  f"top: {row[k]['top_net_loss_classes'][:4]}")
        if len(wrong_sets) >= 2:
            ks = list(wrong_sets)
            a0, b0 = wrong_sets[ks[0]], wrong_sets[ks[1]]
            row["lost_overlap"] = {"pair": ks[:2], "both": len(a0 & b0), "only_first": len(a0 - b0), "only_second": len(b0 - a0)}
        res["sets"][d] = row
    with open(out / "error_classes.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
