"""
portable_eval.py — 다른 기기(ARM Orin, 다른 x86 PC)에서 SigLIP2 후보 ONNX의 dev 정확도를 재는 독립 스크립트 (numpy + onnxruntime만 필요)
================================================================================================================

[목적] 준용 요청(10/15): ARM은 S8 활성값을 U8로 바꾸지 않고 그대로 실행하므로, 같은 ONNX라도 정확도가 x86과 다를 수 있다.
  같은 파일·같은 사진으로 재서 x86 결과와 짝지어 비교한다. 저장소·RN50 코드·데이터셋 없이 돌도록 만들었다.
[실행 방식] s2_screen.py와 같게 맞췄다: rn50/common.py의 session()·embed()와 같은 설정
  (ORT_ENABLE_ALL, intra_op 스레드 지정, inter_op 1, ORT_SEQUENTIAL, CPU EP, 입력 이름 "image", 배치 50, 출력 L2 정규화).
[입력] --data 폴더의 <set>.npz (x: N×3×224×224 float32 전처리 완료, y: 정답, clf: 클래스 벡터 D×C, ref: FP32 임베딩 N×D)
[출력] --out 폴더에 result.json(데이터셋별 top-1·FP32 top-1·cos, 고화질 평균, 기기·버전 정보)와
       pred_<이름>.npz(<set>_pred, <set>_cos, <set>_y — s2_compare_dfc_ort.py와 같은 형식)
[사용]
  python portable_eval.py --data dev200 --out out_orin --threads 4 \
      --models fp32=models/siglip2_image_fp32.onnx s8=models/<S8>.onnx u8rr=models/<U8rr>.onnx
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = SETS[1:]


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL   # 정수 커널 (fake-quant 아님)
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def peak_rss_mib():
    try:
        import psutil
        p = psutil.Process()
        return round(getattr(p.memory_info(), "peak_wset", p.memory_info().rss) / 2**20, 1)
    except Exception:
        try:
            import resource                                    # Linux: ru_maxrss는 KiB
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="이름=경로")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "machine": platform.machine(), "processor": platform.processor(),
           "platform": platform.platform(), "python": platform.python_version(), "onnxruntime": ort.__version__,
           "numpy": np.__version__, "threads": a.threads, "bs": a.bs, "models": {}}
    data = {d: np.load(Path(a.data) / f"{d}.npz") for d in SETS if (Path(a.data) / f"{d}.npz").exists()}
    assert data, "데이터 npz가 없음"
    for spec in a.models:
        name, path = spec.split("=", 1)
        t0 = time.time()
        s = session(path, a.threads)
        row, preds = {}, {}
        for d, z in data.items():
            x, y, W, ref = z["x"], z["y"], z["clf"], z["ref"]
            assert x.dtype == np.float32 and x.shape[1] == 3 and x.shape[2] == x.shape[3] in (224, 256), (d, x.dtype, x.shape)
            e = l2n(np.concatenate([s.run(None, {"image": np.ascontiguousarray(x[i:i + a.bs])})[0]
                                    for i in range(0, len(x), a.bs)]).astype(np.float32))
            assert e.shape == ref.shape, (e.shape, ref.shape)
            cos = (e * ref).sum(1)
            pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
            row[d] = {"n": int(len(y)), "top1": round(100 * float((pred == y).mean()), 2),
                      "fp32_top1": round(100 * float((fp == y).mean()), 2), "cos_mean": round(float(cos.mean()), 4)}
            preds.update({f"{d}_pred": pred, f"{d}_cos": cos, f"{d}_y": y})
            print(name, d, json.dumps(row[d]), flush=True)
        hr = [d for d in HR if d in row]
        row["hr_cos"] = round(float(np.mean([row[d]["cos_mean"] for d in hr])), 4)
        row["hr_dtop1"] = round(float(np.mean([row[d]["top1"] - row[d]["fp32_top1"] for d in hr])), 2)
        row["file"], row["MiB"], row["sec"] = Path(path).name, round(Path(path).stat().st_size / 2**20, 2), round(time.time() - t0)
        res["models"][name] = row
        np.savez_compressed(out / f"pred_{name}.npz", **preds)
        print(name, "hr_dtop1", row["hr_dtop1"], "hr_cos", row["hr_cos"], flush=True)
    res["peak_rss_mib_process"] = peak_rss_mib()           # 모델 여러 개를 한 프로세스에서 돌린 최대값 (모델별 peak 아님)
    with open(out / "result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
