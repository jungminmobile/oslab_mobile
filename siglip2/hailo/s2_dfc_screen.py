"""
s2_dfc_screen.py — DFC 양자화 에뮬레이션 스크리닝: dev 6종 각 앞 K장 (ORT s2_screen.py와 같은 형식) (ROADMAP G4)
=====================================================================================================

[목적]
  G4 구성들을 ORT 스크리닝(D-024)과 같은 표본·같은 지표로 비교한다: dev 6종 각 앞 K장(기본 200), 고화질 cos(주)·Δtop-1(참고).
  기준은 같은 캐시의 FP32 ONNX 임베딩(<key>_ref_fp32.npy). 사진별 예측을 npz로 남겨 짝 부트스트랩·ORT와의 비교에 쓴다.
  infer는 데이터셋마다 한 번만 부른다 (배치마다 부르면 메모리가 쌓여 OOM, s2_dfc_emulate.py 주석).

[사용]  (WSL Ubuntu-22.04, hailo10_venv)
  python s2_dfc_screen.py --har opt_D.har --tag D --k 200 --bs 8 --out <결과 폴더>
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CACHE = Path("/mnt/d/mokerton/artifacts/siglip2/cache")
REPO = Path("/mnt/d/mokerton/oslab_mobile")
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--tag", required=True, help="구성 이름 (예: D_official_b9fc2)")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="데이터셋마다 [start, start+k) 구간 (dev 전체를 200장씩 나눠 따로 돌릴 때, 10/15)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True, help="결과 폴더 (screen.jsonl, pred_<tag>.npz)")
    a = ap.parse_args()
    from hailo_sdk_client import ClientRunner, InferenceContext
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # core.autocrlf=true: Windows git이 CRLF로 체크아웃한 파일을 WSL git이 "수정됨"으로 보아 dirty가 잘못 true가 되던 문제 (10/14 확인:
    # WSL에서 51개 파일이 M, Windows·autocrlf=true·--ignore-cr-at-eol로는 0개)
    git = lambda *x: subprocess.run(["git", "-c", "core.autocrlf=true", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "--short", "HEAD"),
           "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")),
           "tag": a.tag, "har": Path(a.har).name, "har_sha256": sha256(a.har), "k": a.k, "start": a.start, "bs": a.bs,
           "exec": "dfc_sdk_quantized"}
    runner = ClientRunner(har=a.har)
    preds = {}
    t0 = time.time()
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for d in DATA:
            sl = slice(a.start, a.start + a.k)
            x = np.load(CACHE / f"{d}_x.npy", mmap_mode="r")[sl].astype(np.float32)
            y, W = np.load(CACHE / f"{d}_y.npy")[sl], np.load(CACHE / f"{d}_clf.npy")
            ref = np.load(CACHE / f"{d}_ref_fp32.npy")[sl]
            if len(x) == 0:
                preds.update({f"{d}_pred": np.zeros(0, np.int64), f"{d}_cos": np.zeros(0, np.float32), f"{d}_y": y})
                row[d] = {"n": 0}
                continue
            e = l2n(np.asarray(runner.infer(ctx, np.ascontiguousarray(x.transpose(0, 2, 3, 1)), batch_size=a.bs))
                    .reshape(len(x), -1).astype(np.float32))
            assert e.shape == ref.shape, (e.shape, ref.shape)
            cos = (e * ref).sum(1)
            pred, fp_pred = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
            preds.update({f"{d}_pred": pred, f"{d}_cos": cos, f"{d}_y": y})
            row[d] = {"n": int(len(y)), "top1": round(100 * float((pred == y).mean()), 2),
                      "fp32_top1": round(100 * float((fp_pred == y).mean()), 2),
                      "cos_mean": round(float(cos.mean()), 4), "cos_p1": round(float(np.percentile(cos, 1)), 4)}
            print(d, json.dumps(row[d]), f"{time.time() - t0:.0f}s", flush=True)
    hr = [d for d in HR if row[d].get("n", 0) > 0]                    # --start 구간에 사진이 없는 데이터셋은 뺀다
    row["hr_sets"] = {d: row[d].get("n", 0) for d in hr}             # 구간 실행(--start>0)이면 데이터셋·장수가 달라 요약이 일반 행과 다르다 (독립 검토 지적)
    row["hr_cos"] = round(float(np.mean([row[d]["cos_mean"] for d in hr])), 4) if hr else None
    row["hr_dtop1"] = round(float(np.mean([row[d]["top1"] - row[d]["fp32_top1"] for d in hr])), 2) if hr else None
    row["sec"] = round(time.time() - t0)
    np.savez_compressed(out / f"pred_{a.tag}.npz", **preds)
    with open(out / "screen.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: row[k] for k in ("tag", "hr_cos", "hr_dtop1", "sec")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
