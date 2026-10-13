"""
s2_build_dev.py — SigLIP2 dev 캐시·보정 후보 만들기 (rn50/datasets.py의 사진 선택을 그대로 쓰고 전처리만 SigLIP2로)
==========================================================================================================

[목적]
  RN50과 "같은 사진"으로 설정을 비교하려면 dev·보정 사진 선택 규칙이 같아야 한다.
  선택 규칙(SHA-1 해시 구간, CIFAR 구간, 데이터 소스, 템플릿, 클래스 이름)은 rn50/datasets.py·common.py를 import해서 쓰고,
  이 파일에는 SigLIP2 전처리와 분류기 계산만 둔다 (D-009). rn50/datasets.build_dev와 같은 순서로 같은 파일을 만든다.

[캐시 형식]  $RN50_CACHE (= $SIGLIP2_ARTIFACTS/cache, RN50 캐시와 분리. D-013)
  <key>_x.npy          SigLIP2 전처리까지 끝난 float32 (N,3,224,224). rn50/evaluate.py가 그대로 읽는다
                       (uint8로 저장하면 rn50 common.to_input이 CLIP 평균·표준편차로 정규화하므로 float32로 저장)
  <key>_y.npy, <key>_clf.npy (768×클래스), <key>_ref_fp32.npy (FP32 ONNX 임베딩, ORT 기본 설정)
  <key>_sha1.txt       고화질 dev 사진의 SHA-1 목록 (두 트랙이 같은 사진을 쓰는지 대조용. RN50 쪽에는 없는 추가 기록)
  calib_<이름>_f32.npy 보정 후보 (float32). 이름을 _u8로 하지 않아 rn50/quantize_onnx.py가 잘못 읽지 못하게 한다

[명령]
  RN50_RESULTS=../results/siglip2/G2_fp32_dev "$SIGLIP2_PYTHON" s2_build_dev.py all --threads 4
  이름: cifar_A cifar_B pets flowers cars food caltech (all = 전부)
"""
import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from s2_export_onnx import FP32_IMAGE  # noqa: E402
from scripts import env_info  # noqa: E402

RC, RD = S.RC, S.RD


def build(name, model, preprocess, tok, threads, keep_downloads, git):
    import hashlib
    t0 = time.time()
    to_x = lambda img: preprocess(img.convert("RGB")).numpy()
    sha = None
    if name in ("cifar_A", "cifar_B"):
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=True)
        s = RC.VAL_SPLITS[name[-1]]
        xs = np.stack([to_x(ds[i][0]) for i in range(s, s + RC.VAL_SIZE)])
        ys = np.array([ds.targets[i] for i in range(s, s + RC.VAL_SIZE)])
        names, templates = [c.replace("_", " ") for c in ds.classes], RC.TEMPLATES
        n_cal = None
    else:
        cfg = RD.HIGHRES[name]
        p_dev, p_cal = RD.DEV_N / cfg["n_pool"], RD.CAL_N / cfg["n_pool"]
        xs, ys, cal, sha = [], [], [], []
        for b, lab in RD.parquet_rows(cfg["pool"], keep_downloads):
            u = RD.u_of(b)
            if u < p_dev:
                xs.append(to_x(Image.open(io.BytesIO(b)))); ys.append(lab)
                sha.append(hashlib.sha1(b).hexdigest())
            elif u < p_dev + p_cal:
                cal.append(to_x(Image.open(io.BytesIO(b))))
        xs, ys = np.stack(xs), np.array(ys)
        np.save(RC.CACHE_DIR / f"calib_{name}_f32.npy", np.stack(cal).astype(np.float32))
        n_cal = len(cal)
        names, templates = RD.classnames(name), cfg["templates"]
    xs = xs.astype(np.float32)
    assert xs.ndim == 4 and xs.shape[1:] == (3, 224, 224) and np.isfinite(xs).all()
    assert xs.min() >= -1.0001 and xs.max() <= 1.0001, (xs.min(), xs.max())   # 평균·표준편차 0.5 정규화 범위
    W = S.zeroshot_classifier(model, tok, names, templates)
    ref = RC.embed(RC.onnx_path(FP32_IMAGE), xs, threads, fake_quant=False)   # FP32 ONNX, ORT 기본 설정
    key = name if name.startswith("cifar") else f"{name}_dev"
    for k, v in (("x", xs), ("y", ys), ("clf", W), ("ref_fp32", ref)):
        np.save(RC.CACHE_DIR / f"{key}_{k}.npy", v)
    if sha is not None:
        (RC.CACHE_DIR / f"{key}_sha1.txt").write_text("\n".join(sha) + "\n")
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": git["commit"], "dirty": git["dirty"],
           "data": key, "model": "fp32", "onnx": FP32_IMAGE, "exec": "fp32_ort_enable_all", "n": int(len(ys)),
           "classes": len(names), "classes_present": int(len(set(ys.tolist()))), "n_templates": len(templates),
           "n_calib_candidates": n_cal, **RC.topk_metrics(ref, W, ys), "threads": threads,
           "sec": round(time.time() - t0)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    RC.append_jsonl("baselines", row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="cifar_A cifar_B pets flowers cars food caltech 또는 all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-downloads", action="store_true", help="새로 받은 parquet을 지우지 않음")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    RC.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(RC.RESULTS_DIR, env)
    model, preprocess, tok = S.load_model()
    assert RC.onnx_path(FP32_IMAGE).exists(), "먼저 s2_export_onnx.py 실행"
    names = ["cifar_A", "cifar_B"] + list(RD.HIGHRES) if a.name == "all" else [a.name]
    for n in names:
        build(n, model, preprocess, tok, a.threads, a.keep_downloads, env["git"])


if __name__ == "__main__":
    main()
