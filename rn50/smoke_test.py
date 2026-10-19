"""
smoke_test.py — 코드를 고친 뒤 "기존 결과가 안 깨졌는지" 5분 안에 확인 (E0 ⑤)
==========================================================================

[무엇을 확인하나]
  1. 캐시   : Pets dev 캐시의 FP32 정확도가 84.34%인지 (dev 사진·분류기·기준 임베딩이 그대로인지)
  2. 평가   : 10/7에 만든 c3 모델(U8S8)을 정수 커널로 평가 → 10/7·10/12 기록과 같은지
              (CPU마다 정수 커널 결과가 사진 1장 정도 다를 수 있어 CPU 종류별 기준값을 둔다)
  3. 양자화 : 지금 코드로 c3를 새로 양자화(기본 옵션) → 정상 범위인지
              E0 ⑦에서 보정 후보 풀이 311장 → 1292장으로 바뀌어 같은 시드라도 뽑히는 사진이 달라지므로
              "완전히 같은 값"이 아니라 "정상 범위(10/7 c3와 비슷)"인지만 본다

[사용]  python smoke_test.py            (전체, 약 5분)
        python smoke_test.py --skip-quant   (1·2만, 약 1분)
[결과]  마지막 줄 SMOKE PASS / SMOKE FAIL. 결과는 results/smoke_test.jsonl에도 남는다.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

import common as C

C3_OLD = "rn50_image_int8_q1_minmax_highres256_s0_c3.onnx"      # 10/7 모델 (U8S8, 보정 후보 311장 시절)
ATTN_PROJ = "/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm"
# 기준값: Pets dev 990장, 정수 커널, 2스레드. (top1, cos_mean)
EXPECT_FP32_TOP1 = 84.34
EXPECT_C3_OLD = {"avx512": (82.83, 0.9566),   # 클라우드 Xeon (AVX512-VNNI), 10/7·10/12
                 "avxvnni": (82.93, 0.9568)}  # 정민 노트북 Core Ultra 258V (AVX-VNNI), 10/12·10/15
TOL_TOP1, TOL_COS = 0.21, 0.0011              # 사진 2장 미만, cos 소수 셋째 자리
SANE_C3_NEW = {"top1_min": 81.0, "cos_min": 0.950}   # 새 양자화 c3의 정상 범위


def evaluate(model, x, y, W, ref):
    return C.topk_metrics(C.embed(model, x, 2, fake_quant=False), W, y, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quant", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    meta = C.run_meta()
    print("env:", json.dumps({k: meta[k] for k in ("cpu", "vnni", "onnxruntime", "git")}, ensure_ascii=False))
    x = np.load(C.CACHE_DIR / "pets_dev_x.npy", mmap_mode="r")
    y, W, ref = (np.load(C.CACHE_DIR / f"pets_dev_{k}.npy") for k in ("y", "clf", "ref_fp32"))
    checks = []

    fp = C.topk_metrics(ref, W, y)["top1"]
    checks.append(("1 cache fp32 top1", fp, abs(fp - EXPECT_FP32_TOP1) < 0.01))

    if C.onnx_path(C3_OLD).exists():
        r = evaluate(C3_OLD, x, y, W, ref)
        ok = any(abs(r["top1"] - t) <= TOL_TOP1 and abs(r["cos_mean"] - c) <= TOL_COS for t, c in EXPECT_C3_OLD.values())
        checks.append(("2 c3(10/7) int eval", r, ok))
    else:
        checks.append(("2 c3(10/7) int eval", f"{C3_OLD} 없음 (건너뜀)", True))

    if not a.skip_quant:
        cmd = [sys.executable, "quantize_onnx.py", "--enc", "image", "--variant", "_q1", "--calib-source", "highres",
               "--calib-n", "256", "--seed", "0", "--only", "/layer3,/layer4", "--only-exact", ATTN_PROJ,
               "--tag", "c3smoke", "--overwrite"]
        q = subprocess.run(cmd, cwd=C.HERE, capture_output=True, text=True)
        if q.returncode:
            checks.append(("3 quantize c3 (new code)", q.stderr[-500:], False))
        else:
            new = "rn50_image_int8_q1_minmax_highres256_s0_c3smoke.onnx"
            r = evaluate(new, x, y, W, ref)
            ok = r["top1"] >= SANE_C3_NEW["top1_min"] and r["cos_mean"] >= SANE_C3_NEW["cos_min"]
            checks.append(("3 quantize c3 (new code, U8S8)", r, ok))

    passed = all(c[2] for c in checks)
    for name, val, ok in checks:
        print(f"[{'OK ' if ok else 'BAD'}] {name}: {val}")
    row = {"date": datetime.now().isoformat(timespec="seconds"), "pass": passed,
           "checks": [{"name": n, "value": v, "ok": o} for n, v, o in checks], "sec": round(time.time() - t0), "meta": meta}
    C.append_jsonl("smoke_test", row)
    print(f"SMOKE {'PASS' if passed else 'FAIL'}  ({time.time() - t0:.0f}s)")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
