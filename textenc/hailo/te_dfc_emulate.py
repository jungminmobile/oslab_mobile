"""
te_dfc_emulate.py — 텍스트 블록 HAR을 DFC 에뮬레이터로 돌려 정확도 확인 (WSL, ~/hailo10_venv, numpy + hailo_sdk_client만)
=====================================================================================================================

siglip2/hailo/s2_dfc_emulate.py(이미지 쪽)와 같은 방식:
  SDK_FP_OPTIMIZED (그래프 변환만, 양자화 없음)  → ONNX FP32와 같아야 한다 (변환 확인)
  SDK_QUANTIZED    (8/16비트 에뮬레이션)          → 양자화 손실
NPU 출력 (N,1,64,768)에서 위치 63을 고르고 호스트 projection을 곱해 텍스트 임베딩을 만든다 (배포 때 호스트가 하는 일과 같음).
지표: 클래스 임베딩 cos (FP32 ONNX 분류기 대비 최소·평균), dev zero-shot top-1과 Δ (이미지는 FP32 임베딩 그대로),
      확인용 문장 64개 cos. 입력은 te_dfc_evalset.py가 만든 dfc_eval_{pets,food,holdout}.npz.

[사용]  CUDA_VISIBLE_DEVICES=-1 python te_dfc_emulate.py --har t1.har [t2.har ...] --data-dir <npz 폴더> --out emulate.jsonl
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

POS = 63


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def text_emb(runner, ctx, x, proj_w, proj_b, bs):
    out = np.asarray(runner.infer(ctx, x, batch_size=bs)).astype(np.float32)
    h = out.reshape(len(x), -1, 768)              # (N,64,768) — NHWC (N,1,64,768)이든 (N,64,768)이든
    assert h.shape[1] == 64, out.shape
    return h[:, POS] @ proj_w.T + proj_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", nargs="+", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantized-only", action="store_true")
    a = ap.parse_args()
    from hailo_sdk_client import ClientRunner, InferenceContext
    dd = Path(a.data_dir).expanduser()
    sets = {d: dict(np.load(dd / f"dfc_eval_{d}.npz")) for d in ("pets", "food")}
    hold = dict(np.load(dd / "dfc_eval_holdout.npz"))
    ctxs = [("quantized", InferenceContext.SDK_QUANTIZED)]
    if not a.quantized_only:
        ctxs.insert(0, ("fp_optimized", InferenceContext.SDK_FP_OPTIMIZED))
    for har in a.har:
        runner = ClientRunner(har=har)
        row = {"date": datetime.now().isoformat(timespec="seconds"), "har": Path(har).name}
        for name, ctx_type in ctxs:
            t0 = time.time()
            r = {}
            with runner.infer_context(ctx_type) as ctx:
                cos_all, d_all = [], []
                for d, s in sets.items():
                    W = l2n(text_emb(runner, ctx, s["x"], s["proj_w"], s["proj_b"], a.bs)).T   # 템플릿 1개 → 문장 = 클래스
                    c = (W * s["W0"]).sum(0)
                    ok = np.argmax(s["img"] @ W, 1) == s["y"]
                    ok0 = np.argmax(s["img"] @ s["W0"], 1) == s["y"]
                    r[d] = {"cls_cos_min": round(float(c.min()), 5), "cls_cos_mean": round(float(c.mean()), 5),
                            "top1": round(100 * float(ok.mean()), 2), "top1_fp32": round(100 * float(ok0.mean()), 2),
                            "d_top1": round(100 * float(ok.mean() - ok0.mean()), 2), "n": int(len(ok))}
                    cos_all.append(c)
                    d_all.append(r[d]["d_top1"])
                e = l2n(text_emb(runner, ctx, hold["x"], hold["proj_w"], hold["proj_b"], a.bs))
                hc = (e * l2n(hold["ref"])).sum(1)
            allc = np.concatenate(cos_all)
            r.update({"cls_cos_min": round(float(allc.min()), 5), "cls_cos_mean": round(float(allc.mean()), 5),
                      "d_top1_mean": round(float(np.mean(d_all)), 2), "holdout_cos_mean": round(float(hc.mean()), 5),
                      "holdout_cos_min": round(float(hc.min()), 5), "sec": round(time.time() - t0)})
            row[name] = r
            print(Path(har).name, name, json.dumps(r, ensure_ascii=False), flush=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
