"""
s2_input_shift.py — G8 B7: 입력 분포가 바뀌어도 고정 위치 레지스터 캐시(M2)가 버티는가 (2026-10-22, REVIEW_20261022 한계 3·10)
====================================================================================================================
[계기] M2는 sink 토큰 위치(패치 55·13)와 그 K/V 상수를 보정 사진에서 정해 고정한다. 검토 지적: 좌우 반전·흑백·어두운 사진·
       저화질처럼 분포가 다른 입력에서도 같은 자리에 sink가 생기는지, 정확도가 유지되는지 확인한 적이 없다.
[방법] dev 캐시(전처리 끝난 float, [-1, 1])에 변형을 걸고, 같은 변형 사진에서 FP32 원본 대비
  - 각 모델의 고화질 5종 Δtop-1 (같은 사진 짝, 층화 부트스트랩 1000회 시드 0) · 사진 임베딩 cos
  - FP32 원본 자체의 top-1 (변형이 얼마나 센지)
  모델: 첫 번째가 기준 FP32. 나머지는 M2rot FP32(근사 자체), 배포 INT8, 비교용 R0(M2 없음) 등.
  M2 근사가 깨지면 M2rot FP32의 cos가 떨어지고, M2rot INT8의 R0 대비 이득이 줄어든다.
  변형 (사진 p ∈ [0, 1] 기준): none, hflip(좌우 반전), rot90(90도 회전 — 패치 자리가 바뀜), gray(흑백),
       dark(p×0.4), lowcon(0.5+0.4(p−0.5)), noise(가우시안 σ=0.05, 시드 0), jpeg20(JPEG 품질 20)
[출력] $RN50_RESULTS/input_shift.jsonl (변형×모델), input_shift_summary.json
"""
import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402
from s2_paired_ci import boot_hr, HR  # noqa: E402

RC = S.RC
TRANSFORMS = ["none", "hflip", "rot90", "gray", "dark", "lowcon", "noise", "jpeg20"]


def transform(x, t, seed=0):
    if t == "none":
        return x
    if t == "hflip":
        return np.ascontiguousarray(x[..., ::-1])
    if t == "rot90":
        return np.ascontiguousarray(np.rot90(x, 1, axes=(2, 3)))
    p = (x + 1) / 2
    if t == "gray":
        g = 0.299 * p[:, 0] + 0.587 * p[:, 1] + 0.114 * p[:, 2]
        p = np.repeat(g[:, None], 3, axis=1)
    elif t == "dark":
        p = p * 0.4
    elif t == "lowcon":
        p = 0.5 + 0.4 * (p - 0.5)
    elif t == "noise":
        p = np.clip(p + np.random.default_rng(seed).normal(0, 0.05, p.shape), 0, 1)
    elif t == "jpeg20":
        from PIL import Image
        out = np.empty_like(p)
        for i in range(len(p)):
            u8 = np.clip(np.round(p[i].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(u8).save(buf, format="JPEG", quality=20)
            out[i] = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.float32).transpose(2, 0, 1) / 255
        p = out
    else:
        raise ValueError(t)
    return (p * 2 - 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=파일. 첫 번째가 기준 FP32")
    ap.add_argument("--transforms", nargs="+", default=TRANSFORMS, choices=TRANSFORMS)
    ap.add_argument("--k", type=int, default=200, help="고화질 데이터셋마다 앞 k장")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    specs = [m.split("=", 1) for m in a.models]
    ref_name = specs[0][0]
    data = {d: (np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.k]).astype(np.float32),
                np.load(RC.CACHE_DIR / f"{d}_y.npy")[:a.k], np.load(RC.CACHE_DIR / f"{d}_clf.npy")) for d in HR}
    summ = {}
    for t in a.transforms:
        emb = {n: {} for n, _ in specs}
        for d, (x, y, W) in data.items():
            xt = transform(x, t)
            for n, fn in specs:
                emb[n][d] = RC.embed(RC.onnx_path(fn), xt, a.threads, fake_quant=False)
        corr = {n: {d: (np.argmax(emb[n][d] @ data[d][2], 1) == data[d][1]).astype(float) for d in HR} for n, _ in specs}
        summ[t] = {}
        for n, fn in specs:
            row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
                   "transform": t, "model": n, "file": Path(fn).name, "k": a.k,
                   "top1": {d: round(100 * float(corr[n][d].mean()), 2) for d in HR},
                   "hr_top1": round(float(np.mean([100 * corr[n][d].mean() for d in HR])), 3)}
            if n != ref_name:
                diffs = {d: corr[n][d] - corr[ref_name][d] for d in HR}
                row["hr_dtop1_vs_ref"] = round(float(np.mean([100 * v.mean() for v in diffs.values()])), 3)
                row["hr_dtop1_ci95"] = boot_hr(diffs)
                row["hr_cos_vs_ref"] = round(float(np.mean([(emb[n][d] * emb[ref_name][d]).sum(1).mean() for d in HR])), 5)
                row["hr_cos_p1_vs_ref"] = round(float(np.percentile(np.concatenate([(emb[n][d] * emb[ref_name][d]).sum(1) for d in HR]), 1)), 5)
            with open(out / "input_shift.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summ[t][n] = {k: row[k] for k in ("hr_top1", "hr_dtop1_vs_ref", "hr_dtop1_ci95", "hr_cos_vs_ref") if k in row}
        print(t, json.dumps(summ[t], ensure_ascii=False), flush=True)
        json.dump(summ, open(out / "input_shift_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
