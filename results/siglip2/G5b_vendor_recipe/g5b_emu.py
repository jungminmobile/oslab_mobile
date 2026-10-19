"""g5b_emu.py — 벤더·M2rot2 optimized HAR를 DFC 에뮬레이터로 실행: SDK_NATIVE(float) · SDK_QUANTIZED (G5b 작업 C, 10/19)
s2_dfc_screen.py와 같은 호출(runner.infer, NHWC, batch_size 8)이다. 다른 점: 입력 영역을 고를 수 있고(--pixel), 두 모드를 한 번에 돌리며,
같은 사진의 실칩 임베딩(--chip-emb)과 사진별로 비교한다.
  --pixel : g5b_hailo_chip.py eval과 같은 변환 x·127.5+127.5 (원본 픽셀). 칩에서는 HailoRT가 quant_in(1, 0)으로 반올림해 uint8로 넣으므로
            여기서도 반올림한 값을 넣는다. 반올림 전후 최대 차이를 로그·결과에 남긴다.
  기본    : 정규화된 값 그대로 (M2rot2, s2_dfc_screen.py와 같음).
사용 (WSL, hailo10_venv, CUDA_VISIBLE_DEVICES=-1, flock 안):
  python g5b_emu.py --har opt.har --tag vendor_v500 --pixel --cache <cache> --chip-emb emb_vendor_v500.npz --out <폴더>
"""
import argparse, hashlib, json, subprocess, time
from datetime import datetime
from pathlib import Path
import numpy as np

SETS = ["cifar_A", "pets_dev"]


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--har", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--chip-emb", default="", help="같은 HEF의 실칩 임베딩 npz (<set>_emb, 앞 k장 이상)")
ap.add_argument("--pixel", action="store_true"); ap.add_argument("--k", type=int, default=200); ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
from hailo_sdk_client import ClientRunner, InferenceContext
import hailo_sdk_client
C, out = Path(a.cache), Path(a.out); out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
row = {"what": "g5b_emu", "date": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "--short", "HEAD"),
       "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "dfc": getattr(hailo_sdk_client, "__version__", "?"),
       "tag": a.tag, "har": Path(a.har).name, "har_sha256": sha256(a.har), "script_sha256": sha256(__file__),
       "input_domain": "pixel_0_255_rounded" if a.pixel else "normalized", "k": a.k, "bs": a.bs, "chip_emb": a.chip_emb, "modes": {}}
X, data = {}, {}
for d in SETS:
    x = np.load(C / f"{d}_x.npy", mmap_mode="r")[: a.k].astype(np.float32).transpose(0, 2, 3, 1)
    if a.pixel:
        p = x * 127.5 + 127.5
        q = np.clip(np.round(p), 0, 255)
        row[f"{d}_round_maxabs"] = float(np.abs(q - p).max())
        print(d, "반올림 전후 최대 차이", row[f"{d}_round_maxabs"], flush=True)
        x = q.astype(np.float32)
    X[d] = np.ascontiguousarray(x)
    data[d] = (np.load(C / f"{d}_y.npy")[: a.k], np.load(C / f"{d}_clf.npy"), l2n(np.load(C / f"{d}_ref_fp32.npy")[: a.k]))
chip = np.load(a.chip_emb) if a.chip_emb else None
runner = ClientRunner(har=a.har)
embs = {}
for mode, ctxk in [("native", InferenceContext.SDK_NATIVE), ("quantized", InferenceContext.SDK_QUANTIZED)]:
    t0, r = time.time(), {}
    with runner.infer_context(ctxk) as ctx:
        for d in SETS:
            y, W, ref = data[d]
            e = l2n(np.asarray(runner.infer(ctx, X[d], batch_size=a.bs)).reshape(len(y), -1).astype(np.float32))
            embs[f"{mode}_{d}"] = e
            cos = (e * ref).sum(1)
            pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
            r[d] = {"n": int(len(y)), "top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                    "dtop1": round(100 * float((pred == y).mean() - (fp == y).mean()), 2),
                    "cos_vs_fp32_mean": round(float(cos.mean()), 5), "cos_vs_fp32_min": round(float(cos.min()), 5)}
            if chip is not None:
                ce = l2n(chip[f"{d}_emb"][: a.k].astype(np.float32))
                assert ce.shape == e.shape, (ce.shape, e.shape)
                cp = np.argmax(ce @ W, 1)
                r[d]["vs_chip_pred_agree"] = round(float((cp == pred).mean()), 4)
                r[d]["vs_chip_emb_cos_mean"] = round(float((ce * e).sum(1).mean()), 5)
                r[d]["vs_chip_emb_cos_min"] = round(float((ce * e).sum(1).min()), 5)
                r[d]["chip_top1"] = round(100 * float((cp == y).mean()), 2)
            print(a.tag, mode, d, json.dumps(r[d]), f"{time.time() - t0:.0f}s", flush=True)
    r["sec"] = round(time.time() - t0)
    row["modes"][mode] = r
np.savez_compressed(out / f"emu_{a.tag}.npz", **embs)
with open(out / "emu.jsonl", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("DONE", a.tag)
