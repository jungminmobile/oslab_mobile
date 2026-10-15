"""
te_eval.py — 텍스트만 INT8로 바꾼 zero-shot 정확도 스모크 + 클래스 임베딩 cos (스모크 4단계)
=====================================================================================

[방법]
  - 이미지 쪽은 팀 dev 캐시의 FP32 이미지 임베딩(<이름>_dev_ref_fp32.npy, s2_build_dev.py)을 그대로 쓴다.
  - 텍스트 쪽만 바꾼다: 각 모델(ORT 기본 설정 = 정수 커널)로 클래스 프롬프트 → 분류기(앙상블 규칙은 s2common과 같음).
  - 기준은 같은 경로의 FP32 ONNX 텍스트 분류기. 팀 캐시 분류기(PyTorch FP32)와의 top-1도 같이 적어 같은 조건임을 확인한다.
  - 지표: 클래스 임베딩 cos(FP32 vs INT8, 클래스별 최소·평균), 보정에 안 쓴 문장 64개의 임베딩 cos,
          dev top-1과 Δtop-1(짝 부트스트랩 1000회 95% 구간, siglip2/s2_paired_ci.boot_hr), 앞 200장 top-1(참고)
  - 데이터: Pets·Food dev만 (test 분할 사용 금지).

[사용]  python textenc/te_eval.py --models siglip2_text_int8_dyn.onnx ... --out results/textenc/smoke
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_common as T            # noqa: E402
from s2_paired_ci import boot_hr  # noqa: E402  (siglip2/)

RC = T.RC


def text_outputs(model, threads, meta):
    s = RC.session(T.te_path(model), threads)
    W = {d: T.classifier(T.encode(s, np.load(T.TE_DIR / f"tokens_{d}.npy")), *meta[d]) for d in T.EVAL_SETS}
    hold = T.encode(s, np.load(T.TE_DIR / "tokens_holdout_s1_64.npy"))
    return W, RC.l2n(hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-screen", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    env = T.env_row()
    meta = {}
    for d in T.EVAL_SETS:
        names, templates, _ = T.eval_prompts(d)
        meta[d] = (len(names), len(templates))
    dev = {d: T.load_dev(d) for d in T.EVAL_SETS}
    W0, H0 = text_outputs(T.FP32, a.threads, meta)
    ok0 = {d: np.argmax(dev[d][2] @ W0[d], 1) == dev[d][0] for d in T.EVAL_SETS}
    team = {d: round(100 * float((np.argmax(dev[d][2] @ dev[d][1], 1) == dev[d][0]).mean()), 2) for d in T.EVAL_SETS}
    fp32_mib = T.te_path(T.FP32).stat().st_size / 2**20
    hdr = f"{'model':44s} {'MiB':>7s} {'Δsize':>7s} {'clsCos min/mean':>16s} {'holdCos':>8s} " + \
          " ".join(f"{d + ' top1(Δ)':>18s}" for d in T.EVAL_SETS) + f" {'meanΔ [95%]':>22s}"
    print(hdr)
    for m in [T.FP32] + a.models:
        if m == T.FP32:
            W, H = W0, H0
        else:
            W, H = text_outputs(m, a.threads, meta)
        r = {**T.base_row(env, vars(a)), "step": "eval", "model": Path(m).name,
             "MiB": round(T.te_path(m).stat().st_size / 2**20, 2), "data": {}}
        r["size_change_pct"] = round(100 * (r["MiB"] / fp32_mib - 1), 1)
        diffs, csum = {}, []
        for d in T.EVAL_SETS:
            y, _, img = dev[d]
            c = (W[d] * W0[d]).sum(0)
            ok = np.argmax(img @ W[d], 1) == y
            diffs[d] = ok.astype(np.float64) - ok0[d]
            n = min(a.n_screen, len(y))
            r["data"][d] = {"n": int(len(y)), "classes": meta[d][0], "cls_cos_min": round(float(c.min()), 5),
                            "cls_cos_mean": round(float(c.mean()), 5), "top1": round(100 * float(ok.mean()), 2),
                            "top1_fp32_ort": round(100 * float(ok0[d].mean()), 2), "top1_team_cache_clf": team[d],
                            "d_top1": round(100 * float(diffs[d].mean()), 2), "d_top1_ci95": boot_hr({d: diffs[d]}, a.n_boot),
                            f"top1_first{n}": round(100 * float(ok[:n].mean()), 2),
                            f"d_top1_first{n}": round(100 * float(diffs[d][:n].mean()), 2),
                            "n_pred_changed": int((np.argmax(img @ W[d], 1) != np.argmax(img @ W0[d], 1)).sum())}
            csum.append(c)
        hc = (H * H0).sum(1)
        r["holdout_cos_min"], r["holdout_cos_mean"] = round(float(hc.min()), 5), round(float(hc.mean()), 5)
        allc = np.concatenate(csum)
        r["cls_cos_min"], r["cls_cos_mean"] = round(float(allc.min()), 5), round(float(allc.mean()), 5)
        r["d_top1_mean"] = round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2)
        r["d_top1_mean_ci95"] = boot_hr(diffs, a.n_boot)
        T.append_jsonl(out / "eval.jsonl", r)
        print(f"{r['model'][:44]:44s} {r['MiB']:7.1f} {r['size_change_pct']:6.1f}% {r['cls_cos_min']:7.4f}/{r['cls_cos_mean']:.4f} "
              f"{r['holdout_cos_mean']:8.4f} " +
              " ".join(f"{r['data'][d]['top1']:8.2f} ({r['data'][d]['d_top1']:+5.2f})" for d in T.EVAL_SETS) +
              f" {r['d_top1_mean']:+6.2f} {str(r['d_top1_mean_ci95']):>15s}", flush=True)
    print("팀 캐시 분류기(PyTorch FP32) top-1:", team)


if __name__ == "__main__":
    main()
