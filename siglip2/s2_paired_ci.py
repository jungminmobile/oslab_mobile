"""
s2_paired_ci.py — dev 전체 사진별 예측 저장 + 짝 부트스트랩 95% 신뢰구간 (PRD 5절 Δtop-1 규칙)
=========================================================================================

[목적]
  PRD 5절: Δtop-1은 같은 사진끼리 짝지어 비교하고, 데이터셋마다 따로 사진을 뽑는(층화) 부트스트랩 1000회로 95% 구간을 낸다.
  "고화질 평균"은 5종 단순 평균. 구간이 0을 포함하면 "유의한 차이 없음". 두 양자화 구성끼리도 차이를 직접 짝 부트스트랩한다.
  rn50/evaluate.py는 평균만 남기므로, 사진별 정답 여부를 여기서 다시 계산해 저장한다(정수 커널, RN50 공통 함수 사용).

[출력]  $RN50_RESULTS/
  pred_<모델>.npz        데이터셋별 사진 순서대로 top-1 예측·정답 여부·FP32 cos (재분석용)
  paired_ci.jsonl        모델별: 데이터셋 top-1·Δ, 고화질 평균 Δ와 95% 구간, 기준선(0.5·1.0·2.0%p) 분류(점추정)
                         모델 쌍별: 차이와 95% 구간

[사용]
  RN50_RESULTS=../results/siglip2/G3_cumulative/full "$SIGLIP2_PYTHON" s2_paired_ci.py --models a.onnx b.onnx ...
"""
import argparse
import itertools
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
HR = DATA[1:]


def predictions(model, threads):
    """데이터셋별 (예측, 정답 여부, cos). model == 'fp32'면 캐시된 FP32 ONNX 임베딩."""
    out = {}
    for d in DATA:
        f = lambda k: RC.CACHE_DIR / f"{d}_{k}.npy"
        y, W, ref = np.load(f("y")), np.load(f("clf")), np.load(f("ref_fp32"))
        feat = ref if model == "fp32" else RC.embed(RC.onnx_path(model), np.load(f("x"), mmap_mode="r"), threads,
                                                     fake_quant=False)          # 정수 커널 (D-008)
        pred = np.argmax(feat @ W, 1)
        out[d] = {"pred": pred, "correct": pred == y, "cos": (feat * ref).sum(1)}
        print(f"  {model[:60]} {d} top1={100 * out[d]['correct'].mean():.2f}", flush=True)
    return out


def boot_hr(diff_by_data, n_boot=1000, seed=0):
    """diff_by_data: {데이터셋: 사진별 (b 정답 − a 정답)} → 고화질 평균 차이(%p)의 95% 구간 (데이터셋별 층화 재표집)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(np.mean([100 * v[rng.integers(0, len(v), len(v))].mean() for v in diff_by_data.values()]))
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    preds = {"fp32": predictions("fp32", a.threads)}
    for m in a.models:
        preds[m] = predictions(m, a.threads)
        np.savez_compressed(out / f"pred_{Path(m).stem}.npz",
                            **{f"{d}_{k}": v for d in DATA for k, v in preds[m][d].items()})
    base = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
            "dirty": env["git"]["dirty"], "exec": "int", "n_boot": a.n_boot, **env_info.flat(env)}
    rows = []
    for m in a.models:
        top1 = {d: round(100 * float(preds[m][d]["correct"].mean()), 2) for d in DATA}
        dlt = {d: round(top1[d] - 100 * float(preds["fp32"][d]["correct"].mean()), 2) for d in DATA}
        diff = {d: preds[m][d]["correct"].astype(float) - preds["fp32"][d]["correct"] for d in HR}
        # 기준선 분류는 반올림하지 않은 값으로 한다 (반올림 값으로 하면 −0.504가 "≤0.5%p"로 잘못 분류될 수 있음, 독립 검토 지적)
        hr_raw = float(np.mean([100 * (preds[m][d]["correct"].mean() - preds["fp32"][d]["correct"].mean()) for d in HR]))
        hr = round(hr_raw, 2)
        ci = boot_hr(diff, a.n_boot)
        r = {**base, "model": m, "top1": top1, "dtop1": dlt, "hr_dtop1": hr, "hr_dtop1_ci95": ci,
             "hr_cos": round(float(np.mean([preds[m][d]["cos"].mean() for d in HR])), 4),
             "significant": not (ci[0] <= 0 <= ci[1]),
             "hr_dtop1_raw": round(hr_raw, 4),
             "baseline_class": next((f"≤{t}%p" for t in (0.5, 1.0, 2.0) if -hr_raw <= t), ">2.0%p")}
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("model", "hr_dtop1", "hr_dtop1_ci95", "hr_cos", "baseline_class")},
                         ensure_ascii=False), flush=True)
    for m1, m2 in itertools.combinations(a.models, 2):
        diff = {d: preds[m2][d]["correct"].astype(float) - preds[m1][d]["correct"] for d in HR}
        d_hr = round(float(np.mean([100 * v.mean() for v in diff.values()])), 2)
        ci = boot_hr(diff, a.n_boot)
        rows.append({**base, "pair": [m1, m2], "hr_diff_b_minus_a": d_hr, "ci95": ci, "tie": ci[0] <= 0 <= ci[1]})
        print("pair", Path(m1).stem[-20:], Path(m2).stem[-20:], d_hr, ci, flush=True)
    with open(out / "paired_ci.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
