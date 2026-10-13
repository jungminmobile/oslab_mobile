"""
s2_dfc_emulate.py — DFC 에뮬레이터로 SigLIP2 HAR 기능 확인 + 양자화 에뮬레이션 정확도 (G1 확인, S-C 미리보기)
=====================================================================================================

[목적]  (VERIFICATION D-G1, PRD E8a 방식)
  ① SDK_FP_OPTIMIZED(최적화된 float 그래프) 출력이 ONNX FP32와 같은지 → 파싱·그래프 변환이 맞는지 (기능 확인)
  ② SDK_QUANTIZED(8비트 에뮬레이션) 출력의 cos·top-1 → DFC 기본 양자화(S-C)의 손실 미리보기
  두 값을 같이 재면 "양자화 오차"와 "최적화(그래프 변환) 오차"가 분리된다 (PRD E8a).

[입력]  WSL에서 실행. dev 캐시는 Windows 쪽 $SIGLIP2_ARTIFACTS/cache (float32 NCHW, SigLIP2 전처리 끝) → NHWC로 바꿔 넣는다.
  기준 임베딩은 같은 캐시의 <key>_ref_fp32.npy (FP32 ONNX, ORT 기본 설정으로 만든 것).

[사용]  (WSL Ubuntu-22.04, hailo10_venv)
  python s2_dfc_emulate.py --har <optimized.har> --data pets_dev --k 64 --out <결과 jsonl>
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CACHE = Path("/mnt/d/mokerton/artifacts/siglip2/cache")


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True, help="optimize 단계 산출 HAR (양자화 정보 포함)")
    ap.add_argument("--data", default="pets_dev")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantized-only", action="store_true", help="SDK_QUANTIZED만 (float 확인은 G1에서 끝남)")
    a = ap.parse_args()
    from hailo_sdk_client import ClientRunner, InferenceContext
    x = np.load(CACHE / f"{a.data}_x.npy", mmap_mode="r")[:a.k].astype(np.float32)
    y = np.load(CACHE / f"{a.data}_y.npy")[:a.k]
    W = np.load(CACHE / f"{a.data}_clf.npy")
    ref = np.load(CACHE / f"{a.data}_ref_fp32.npy")[:a.k]
    nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    runner = ClientRunner(har=a.har)
    row = {"date": datetime.now().isoformat(timespec="seconds"), "har": Path(a.har).name, "data": a.data, "k": a.k}
    fp_top1 = round(100 * float(np.mean(np.argmax(ref @ W, 1) == y)), 2)
    row["onnx_fp32_top1"] = fp_top1
    ctxs = [("fp_optimized", InferenceContext.SDK_FP_OPTIMIZED), ("quantized", InferenceContext.SDK_QUANTIZED)]
    for name, ctx_type in (ctxs[1:] if a.quantized_only else ctxs):
        t0 = time.time()
        with runner.infer_context(ctx_type) as ctx:
            out = np.concatenate([np.asarray(runner.infer(ctx, nhwc[i:i + a.bs])).reshape(len(nhwc[i:i + a.bs]), -1)
                                  for i in range(0, len(nhwc), a.bs)])
        assert out.shape == (a.k, ref.shape[1]), out.shape
        e = l2n(out.astype(np.float32))
        cos = (e * ref).sum(1)
        row[name] = {"cos_mean": round(float(cos.mean()), 5), "cos_min": round(float(cos.min()), 5),
                     "top1": round(100 * float(np.mean(np.argmax(e @ W, 1) == y)), 2), "sec": round(time.time() - t0)}
        print(name, json.dumps(row[name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
