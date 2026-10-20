"""
repro_fingerprint.py — 두 PC의 빌드가 어디서부터 달라지는지 찾기 위한 지문 (G8, 준용 요청 10/21)
================================================================================================
같은 레시피로 만든 M2rot INT8이 PC마다 다르다 (B/16: 쭈리 009ea9d5… vs 준용 e84c3727…, B/32: 쭈리 빌드 vs 준용 G6b).
같은 PC에서는 다시 만들어도 비트 단위로 같다 (A2). 그래서 PC마다 아래를 뽑아 서로 비교한다.
  1. 버전: python·numpy·torch·onnx·onnxruntime·open_clip·timm, numpy BLAS 설정, CPU 이름·명령어 플래그
  2. 순수 계산 시험: 회전 행렬 rothad를 지금 이 PC에서 다시 계산한 sha256 (파일 없이) + 고정 float64 행렬 곱 sha256
  3. 파일: 보정 사진 calib_<set>_f32.npy, 회전 행렬, M2 캐시(npz는 meta를 빼고 배열만), M2rot FP32·INT8 ONNX
     → 맨 처음 달라지는 파일이 원인 단계다 (보정 사진이 다르면 전처리, 회전 행렬이 다르면 BLAS 반올림 …)
사용 (siglip2 폴더, 모델 실행 없음, 몇 초~1분):
  "$SIGLIP2_PYTHON" scripts/repro_fingerprint.py --art <SIGLIP2_ARTIFACTS> --out fp_<pc>_<b16|b32>.json
  B/32는 S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 를 함께 (모델별 산출물 폴더)
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True, help="산출물 폴더 (cache/, onnx/ 가 있는 곳)")
    ap.add_argument("--onnx-dir", default="", help="ONNX·캐시 폴더 (기본 <art>/onnx)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    art = Path(a.art)
    od = Path(a.onnx_dir) if a.onnx_dir else art / "onnx"
    import io
    import contextlib
    res = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()}
    for m in ["numpy", "torch", "onnx", "onnxruntime", "open_clip", "timm", "PIL"]:
        try:
            res[m] = __import__(m).__version__
        except Exception as e:
            res[m] = f"? {e!r}"
    try:
        import cpuinfo
        ci = cpuinfo.get_cpu_info()
        res["cpu"] = ci.get("brand_raw")
        res["cpu_flags"] = {k: k in ci.get("flags", []) for k in ["avx2", "avx512f", "avx512_vnni", "avx_vnni", "fma"]}
    except Exception as e:
        res["cpu"] = f"? {e!r}"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    res["numpy_blas"] = [l.strip() for l in buf.getvalue().splitlines() if any(k in l.lower() for k in ("name", "openblas", "mkl", "accelerate", "version"))][:12]
    # 2. 순수 계산 시험
    try:
        import s2_rotate
        res["rothad_recomputed_sha"] = sha_arr(s2_rotate.make_q("rothad"))
    except Exception as e:
        res["rothad_recomputed_sha"] = f"? {e!r}"
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((768, 768)), rng.standard_normal((768, 768))
    res["matmul_f64_768_sha"] = sha_arr(A @ B)
    # 3. 파일
    files = {}
    for p in sorted((art / "cache").glob("calib_*_f32.npy")):
        files["cache/" + p.name] = sha_file(p)
    for pat in ["rotq_*.npy", "regcache_*.npz", "siglip2_image_fp32.onnx", "siglip2_image_fp32_rothad.onnx",
                "siglip2_image_fp32_rcmaskbcmlp*.onnx", "siglip2_image_int8_rcmaskbcmlp*_rr_all.onnx"]:
        for p in sorted(od.glob(pat)):
            if p.name.endswith("_b1.onnx") or p.name.endswith("_pre.onnx"):
                continue
            if p.suffix == ".npz":
                z = np.load(p)
                files["onnx/" + p.name + " (배열만)"] = {k: sha_arr(z[k])[:16] for k in z.files if k != "meta"}
            else:
                files["onnx/" + p.name] = sha_file(p)
    res["files"] = files
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
