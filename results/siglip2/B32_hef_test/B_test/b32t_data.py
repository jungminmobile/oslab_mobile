"""b32t_data.py — B32 HEF test (README B-2): test 7종을 준혁 ORT test와 같은 코드로 읽어
  (1) Zoo 전처리 uint8 NHWC 256 픽셀(실칩·HF FP32 입력)을 저장하고
  (2) 같은 사진인지 확인하려고 준혁과 같은 FP32 ONNX(be19a8ad)·timm 전처리·open_clip 분류기로 FP32 top-1·top-5를 낸다.
==========================================================================================================
코드는 g8-candidates의 siglip2/ (s2_test_eval.rows_names_templates, s2common.load_model/zeroshot_classifier, rn50 common)를
--code 폴더에서 import한다 (준혁 test 커밋 72784d1과 같은 코드인지 sha256으로 기록).
1회 보장: <out>/done_data_<set>.json 이 있으면 그 데이터셋은 읽지 않는다.
[사용] (환경변수 S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_DATA SIGLIP2_ARTIFACTS(onnx/siglip2_image_fp32.onnx = be19a8ad))
  python b32t_data.py <set> --code ~/oslab_b32p --out ~/siglip2_work/b32t --log <저장소 B_test 폴더>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

FP32_SHA = "be19a8ada01055df00bbb4a59c4d4f14675180dc3ae10a54d69d290707e822ec"
SIZE = 256


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(d, *args):
    return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["cifar_test", "imagenetv2", "pets", "flowers", "cars", "food", "caltech"])
    ap.add_argument("--code", required=True, help="g8-candidates 코드 worktree (siglip2/ 가 있는 저장소 루트)")
    ap.add_argument("--out", required=True, help="큰 파일 폴더 (test_zoo256/ 아래 픽셀)")
    ap.add_argument("--log", required=True, help="저장소 결과 폴더 (data_list.jsonl, fp32_onnx_check.jsonl)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=50)
    a = ap.parse_args()
    assert os.environ.get("S2_MODEL_ID") == "hf-hub:timm/ViT-B-32-SigLIP2-256", "S2_MODEL_ID를 B/32로"
    out, log = Path(a.out), Path(a.log)
    pd = out / f"test_zoo{SIZE}"
    pd.mkdir(parents=True, exist_ok=True)
    done = out / f"done_data_{a.name}.json"
    if done.exists():
        raise SystemExit(f"{done} 있음 — test는 1회만 읽는다")
    code = Path(a.code).resolve()
    sys.path.insert(0, str(code / "siglip2"))
    import s2common as S                                   # noqa: E402
    import s2_test_eval as T                               # noqa: E402
    from PIL import Image                                  # noqa: E402
    RC = S.RC
    fp32 = RC.onnx_path(T.FP32_IMAGE)
    assert sha_file(fp32) == FP32_SHA, f"{fp32} sha256이 be19a8ad가 아님"
    t0 = time.time()
    model, preprocess, tok = S.load_model()
    rows, names, templates = T.rows_names_templates(a.name)
    W = S.zeroshot_classifier(model, tok, names, templates)
    del model
    sess = RC.session(fp32, a.threads, fake_quant=False)
    # 장수는 미리 모르므로 넉넉한 memmap에 쓰고 마지막에 잘라 저장
    cap = {"food": 25250, "cars": 8041, "flowers": 6149, "caltech": 6084, "pets": 3669, "cifar_test": 10000, "imagenetv2": 10000}[a.name]
    tmp = pd / f"{a.name}_pix.tmp.npy"
    pix = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(cap + 64, SIZE, SIZE, 3))
    ys, emb, buf, sha1s = [], [], [], []
    n = 0
    for img, lab in rows:
        rgb = img.convert("RGB")
        p = np.asarray(rgb.resize((SIZE, SIZE), Image.BICUBIC), np.uint8)   # Zoo 전처리 (s2_g6.py data와 같음)
        pix[n] = p
        sha1s.append(hashlib.sha1(p.tobytes()).hexdigest())
        buf.append(preprocess(rgb).numpy())                                 # 준혁 test와 같은 timm 전처리 (s2_test_eval)
        ys.append(lab)
        n += 1
        if len(buf) == a.bs:
            x = np.ascontiguousarray(np.stack(buf).astype(np.float32))
            emb.append(sess.run(None, {"image": x})[0])
            buf.clear()
        if n % 2000 == 0:
            print(f"  {a.name} {n}  {time.time() - t0:.0f}s", flush=True)
    if buf:
        emb.append(sess.run(None, {"image": np.ascontiguousarray(np.stack(buf).astype(np.float32))})[0])
    assert n <= cap + 64
    y = np.array(ys)
    E = RC.l2n(np.concatenate(emb).astype(np.float32))
    met = RC.topk_metrics(E, W, y, E)                       # 준혁 test.jsonl FP32 줄과 같은 함수 (argsort)
    top1_argmax = round(100 * float((np.argmax(E @ W, 1) == y).mean()), 2)
    final = pd / f"{a.name}_pix.npy"
    np.save(final, np.asarray(pix[:n]))
    del pix
    tmp.unlink()
    np.save(pd / f"{a.name}_y.npy", y)
    np.save(out / f"onnxfp32_{a.name}_emb.npy", E)
    (out / f"meta_{a.name}.json").write_text(json.dumps({"names": list(names), "templates": list(templates)}, ensure_ascii=False))   # HF FP32 분류기용 (같은 이름·템플릿)
    # 원본 파일 sha256 (parquet 조각·tar·CIFAR)
    RD = S.RD
    if a.name == "cifar_test":
        srcs = [Path(RC.DATA_ROOT) / "cifar-100-python" / "test"]
    elif a.name == "imagenetv2":
        srcs = [RD.C.EXTRA_ROOT / "imagenetv2-mf.tar"]
    else:
        srcs = [RD.C.EXTRA_ROOT / local for local, _ in RD.HIGHRES[a.name]["test"]]
    row = {"date": datetime.now().isoformat(timespec="seconds"), "set": a.name, "n": n, "classes": len(names),
           "n_templates": len(templates), "y_sha256": hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
           "pix_sha1_list_sha256": hashlib.sha256("\n".join(sha1s).encode()).hexdigest(),
           "pix_file_sha256": sha_file(final), "src_sha256": {str(p.name): sha_file(p) for p in srcs},
           "fp32_onnx": {"file": fp32.name, "sha256": FP32_SHA, **met, "top1_argmax": top1_argmax},
           "code": str(code), "code_commit": git(code, "rev-parse", "--short", "HEAD"),
           "code_dirty": bool(git(code, "status", "--porcelain", "--untracked-files=no")),
           "code_sha256": {f: sha_file(code / "siglip2" / f)[:16] for f in ("s2_test_eval.py", "s2common.py")} | {"rn50/datasets.py": sha_file(code / "rn50" / "datasets.py")[:16], "rn50/common.py": sha_file(code / "rn50" / "common.py")[:16]},
           "script_commit": git(Path(__file__).parent, "rev-parse", "--short", "HEAD"),
           "script_dirty": bool(git(Path(__file__).parent, "status", "--porcelain", "--untracked-files=no")),
           "preprocess_zoo": "img.convert('RGB').resize((256,256), BICUBIC) uint8 NHWC, crop 없음",
           "threads": a.threads, "bs": a.bs, "sec": round(time.time() - t0)}
    with open(log / "data_list.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    done.write_text(json.dumps({"date": row["date"], "n": n, "script_commit": row["script_commit"]}))
    print(json.dumps({k: row[k] for k in ("set", "n", "classes", "n_templates", "fp32_onnx", "sec")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
