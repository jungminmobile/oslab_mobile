"""b32t_fp32hf.py — README B-3: test 7종 Zoo 픽셀에 HF FP32 기준 (G6 `s2_g6.py fp32`와 같은 계산, M2 없음).
비전: HF google/siglip2-base-patch32-256 FP32 `pooler_output` (입력 pix/127.5−1, 배치 256 → 16씩). 분류기: s2_g6pre.classifier
(HF 텍스트 FP32, _clean_canonicalize, max_length 64), 이름·템플릿은 b32t_data.py가 저장한 meta_<set>.json(준혁 test와 같음).
1회 보장: <out>/ref/done_fp32_<set>.json. 출력: ref/<set>_ref.npy(정규화 전), <set>_clf.npy, <set>_y.npy, 저장소 fp32hf.jsonl.
[사용] python b32t_fp32hf.py --out ~/siglip2_work/b32t --log <B_test 폴더>  (siglip2/ 를 import 경로에, $SIGLIP2_* 설정)"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.argv += ["--model", "b32"] if "--model" not in sys.argv else []          # s2_g6 import 전에 S2_MODEL_ID(b32)
sys.path.insert(0, str(REPO / "siglip2"))
import numpy as np                     # noqa: E402
import s2_g6 as G                       # noqa: E402  (pix_to_x, torch 스레드 6)
P = G.P                                 # s2_g6pre (load_hf, vis_of, classifier, batched, l2n)
SETS = ["pets", "caltech", "flowers", "cars", "food", "cifar_test", "imagenetv2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", default="b32")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    out = Path(a.out)
    rd = out / "ref"
    rd.mkdir(parents=True, exist_ok=True)
    m = P.load_hf("b32")
    vis = P.vis_of(m)
    tok = AutoTokenizer.from_pretrained(P.MODELS["b32"]["hf"])
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    for s in SETS:
        done = rd / f"done_fp32_{s}.json"
        if done.exists():
            print("skip", s, flush=True)
            continue
        t0 = time.time()
        meta = json.loads((out / f"meta_{s}.json").read_text())
        W = P.classifier(m, tok, meta["names"], meta["templates"])
        pix = np.load(out / "test_zoo256" / f"{s}_pix.npy", mmap_mode="r")
        y = np.load(out / "test_zoo256" / f"{s}_y.npy")
        assert len(pix) == len(y) and pix.dtype == np.uint8 and pix.shape[1:] == (256, 256, 3)
        e = np.concatenate([P.batched(lambda t: vis(pixel_values=t).pooler_output, G.pix_to_x(pix[i:i + 256])) for i in range(0, len(y), 256)])
        np.save(rd / f"{s}_ref.npy", e)
        np.save(rd / f"{s}_clf.npy", W)
        np.save(rd / f"{s}_y.npy", y)
        top1 = round(100 * float((np.argmax(P.l2n(e) @ W, 1) == y).mean()), 2)
        row = {"date": datetime.now().isoformat(timespec="seconds"), "set": s, "n": int(len(y)), "classes": int(W.shape[1]),
               "n_templates": len(meta["templates"]), "fp32_hf_top1_argmax": top1, "hf": P.MODELS["b32"]["hf"],
               "commit": commit, "dirty": dirty, "sec": round(time.time() - t0)}
        with open(Path(a.log) / "fp32hf.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done.write_text(json.dumps({"date": row["date"], "n": row["n"], "commit": commit}))
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
