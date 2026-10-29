"""
s2_prompt_compare.py — 프롬프트 템플릿 비교 (CLIP 공식 = RN50과 같은 템플릿 vs SigLIP 계열 권장), dev만 사용 (ROADMAP G2)
=====================================================================================================================

[목적]
  siglip2/CLAUDE.md 4절: "프롬프트 템플릿은 CLIP 공식 템플릿과 SigLIP 계열 권장 템플릿을 dev에서 비교해 정한다".
  이미지 쪽은 s2_build_dev.py가 저장한 FP32 임베딩(<key>_ref_fp32.npy)을 그대로 쓰고, 글 쪽 분류기만 바꿔 top-1을 비교한다.

[비교 대상]
  clip    : RN50 트랙과 같은 템플릿 (CIFAR 18개, Cars 8개, Caltech 34개, Pets·Flowers·Food 각 1개). s2_build_dev.py의 <key>_clf.npy
  siglip  : "This is a photo of {}." 1개 (transformers zero-shot-image-classification 파이프라인 기본값.
            SigLIP 계열에 64토큰 max_length 패딩을 쓰는 같은 코드 경로. open_clip 토크나이저도 64토큰으로 패딩함을 확인)
  두 분류기 모두 같은 프롬프트 앙상블 방식(정규화 → 평균 → 정규화). 클래스 이름은 RN50과 같다.

[판단 규칙 (결과를 보기 전에 정함)]
  고화질 5종 dev 평균 top-1 차이가 짝 부트스트랩(1000회, 데이터셋별 층화) 95% 구간에서 0을 포함하면 "차이 없음" →
  RN50과 같은 clip 템플릿 유지(조건 일치 우선). 구간이 0을 넘어 siglip이 좋으면 사람에게 변경을 제안한다(평가 프로토콜 변경).

[사용]
  RN50_RESULTS=../results/siglip2/G2_prompt_compare "$SIGLIP2_PYTHON" s2_prompt_compare.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC, RD = S.RC, S.RD
SIGLIP_T = ["This is a photo of {}."]
HR = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
DATA = ["cifar_A"] + HR


def classnames(key):
    if key.startswith("cifar"):
        from torchvision.datasets import CIFAR100
        return [c.replace("_", " ") for c in CIFAR100(str(RC.DATA_ROOT), train=True, download=False).classes]
    return RD.classnames(key.removesuffix("_dev"))


def paired_boot(correct_a, correct_b, seed=0, n_boot=1000):
    """데이터셋별 층화 짝 부트스트랩: 고화질 평균 top-1 차이(b − a, %p)의 95% 구간."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        d = []
        for k in correct_a:
            i = rng.integers(0, len(correct_a[k]), len(correct_a[k]))
            d.append(100 * (correct_b[k][i].mean() - correct_a[k][i].mean()))
        diffs.append(np.mean(d))
    return [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]


def main():
    model, _, tok = S.load_model()
    env = env_info.collect()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env_info.write_env(out, env)
    rows, corr = [], {"clip": {}, "siglip": {}}
    for key in DATA:
        f = lambda k: RC.CACHE_DIR / f"{key}_{k}.npy"
        if not f("ref_fp32").exists():
            print(f"skip {key}: 캐시 없음"); continue
        y, ref, W_clip = np.load(f("y")), np.load(f("ref_fp32")), np.load(f("clf"))
        W_sig = S.zeroshot_classifier(model, tok, classnames(key), SIGLIP_T)
        assert W_sig.shape == W_clip.shape
        for name, W in (("clip", W_clip), ("siglip", W_sig)):
            c = np.argmax(ref @ W, 1) == y
            if key in HR:
                corr[name][key] = c
            rows.append({"data": key, "templates": name, "n": int(len(y)), **RC.topk_metrics(ref, W, y)})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
               "dirty": env["git"]["dirty"], "exec": "fp32_ort_enable_all (캐시 임베딩)", "rows": rows}
    if len(corr["clip"]) == len(HR):
        hr = {n: round(float(np.mean([100 * corr[n][k].mean() for k in HR])), 2) for n in corr}
        ci = paired_boot(corr["clip"], corr["siglip"])
        summary["hr_avg_top1"] = hr
        summary["hr_diff_siglip_minus_clip"] = round(hr["siglip"] - hr["clip"], 2)
        summary["hr_diff_ci95"] = ci
        summary["verdict"] = ("차이 없음 → clip 템플릿 유지" if ci[0] <= 0 <= ci[1] else
                              "siglip이 유의하게 좋음 → 사람에게 변경 제안" if ci[0] > 0 else "clip이 유의하게 좋음 → clip 유지")
        print(json.dumps({k: summary[k] for k in ("hr_avg_top1", "hr_diff_siglip_minus_clip", "hr_diff_ci95", "verdict")},
                         ensure_ascii=False))
    with open(out / "prompt_compare.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
