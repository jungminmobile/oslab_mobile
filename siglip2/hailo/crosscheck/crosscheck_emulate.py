"""
crosscheck_emulate.py — 정민 노트북 DFC 교차 확인 (D-011·D-027): 같은 입력으로 만든 D256L1 HAR의 에뮬레이터 출력이 쭈리 PC와 같은지
=====================================================================================================================
입력: --har (정민 PC에서 optimize한 HAR), --data pets_dev.npz (x NCHW float32 200장, y, clf, ref=FP32 ONNX 임베딩),
      --ref ref_D256L1_pets200.npz (쭈리 PC의 같은 200장 결과: pets_dev_pred, pets_dev_cos)
출력: 표준 출력 + crosscheck_result.json — 사진별 cos 차이, top-1 일치율, 고화질 Pets top-1·cos
판단: cos 차이가 1e-4 안이고 예측이 거의 다 같으면 "같은 결과" (DFC 버전·입력 sha256이 같다는 전제)
사용 (WSL, DFC 5.4.0 venv): python crosscheck_emulate.py --har opt_D256L1.har --data pets_dev.npz --ref ref_D256L1_pets200.npz
"""
import argparse
import json
import platform

import numpy as np


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    from hailo_sdk_client import ClientRunner, InferenceContext
    z, r = np.load(a.data), np.load(a.ref)
    x, y, W, ref = z["x"].astype(np.float32), z["y"], z["clf"], z["ref"]
    runner = ClientRunner(har=a.har)
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:   # 쭈리 PC s2_dfc_screen.py와 같은 방식
        e = np.asarray(runner.infer(ctx, np.ascontiguousarray(x.transpose(0, 2, 3, 1)), batch_size=a.bs))
    e = l2n(e.reshape(len(x), -1).astype(np.float32))
    cos = (e * ref).sum(1)
    pred = np.argmax(e @ W, 1)
    d = np.abs(cos - r["pets_dev_cos"])
    res = {"n": int(len(y)), "machine": platform.node(), "top1": round(100 * float((pred == y).mean()), 2),
           "top1_jjury": round(100 * float((r["pets_dev_pred"] == y).mean()), 2), "cos_mean": round(float(cos.mean()), 5),
           "cos_mean_jjury": round(float(r["pets_dev_cos"].mean()), 5), "cos_absdiff_max": float(d.max()),
           "cos_absdiff_mean": float(d.mean()), "pred_agree": round(float((pred == r["pets_dev_pred"]).mean()), 4)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open("crosscheck_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
