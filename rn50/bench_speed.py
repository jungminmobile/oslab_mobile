"""
bench_speed.py — 사진 인코더 속도(지연시간) 반복 측정
====================================================

[목적]
  모델끼리 속도를 공정하게 비교한다. 클라우드/PC의 CPU 상태가 시간에 따라 흔들리므로
  모델들을 번갈아(round-robin) 여러 번 돌려 시간대 영향을 고르게 나누고, 평균과 흔들림 폭을 같이 기록한다.

[측정 조건 (보고서에 그대로 적을 것)]
  - ONNX Runtime CPU, intra-op 스레드 고정(--threads), inter-op 1
  - 입력: CIFAR 검증 A 앞 N장(속도는 사진 내용과 무관), batch 1
  - 모델마다 예열(warmup, 기본 20회) 후, 모든 모델을 한 바퀴씩 도는 것을 R번 반복
  - 기록: 장당 평균(ms) / 반복 간 표준편차 / 최소, 개별 호출 지연의 p50·p95

[주의]
  - 같은 측정 안에서 잰 숫자끼리만 비교한다(측정할 때마다 기준 FP32 값도 달라짐).
  - 다른 무거운 작업과 동시에 돌리지 말 것(CPU를 나눠 써서 숫자가 망가짐).
  - 반복 사이에 모델이 바뀌므로 CPU 캐시가 매번 식는다. 모든 모델이 같은 조건이라 비교에는 문제없지만
    "연속 실행 시 최고 속도"보다는 조금 느리게 나온다.

[사용]  python bench_speed.py --models fp32=rn50_image_fp32_q1.onnx D=rn50_image_int8_xxx.onnx --threads 1 2 4
  · 입력은 cache/cifar_A_x.npy(datasets.py dev cifar가 만듦). CIFAR 원본은 캐시가 없을 때만 필요
"""
import argparse
import json
import statistics
import time
from datetime import datetime

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=파일 형식, 여러 개")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, nargs="+", default=[2],
                    help="스레드 수 여러 개 가능(예: 1 2 4). 스레드 수마다 같은 측정을 반복한다")
    a = ap.parse_args()
    models = dict(kv.split("=", 1) for kv in a.models)

    # 입력: dev 캐시(cifar_A)가 있으면 그것을 쓴다(CIFAR 원본 불필요). 없으면 CIFAR 원본에서 같은 사진을 만든다.
    cache = C.CACHE_DIR / "cifar_A_x.npy"
    if cache.exists():
        x = C.to_input(np.load(cache, mmap_mode="r")[: a.n])
    else:
        ds, _ = C.cifar100(train=True, download=False)
        s0 = C.VAL_SPLITS["A"]
        x = np.stack([ds[i][0].numpy() for i in range(s0, s0 + a.n)]).astype(np.float32)
    meta = C.run_meta()
    for th in a.threads:
        bench(models, x, th, a, meta)


def bench(models, x, threads, a, meta):

    sess = {k: C.session(C.onnx_path(v), threads) for k, v in models.items()}
    for s in sess.values():                       # 예열: 첫 호출의 메모리 할당·그래프 준비 비용 제거
        for i in range(a.warmup):
            s.run(None, {"image": x[i:i + 1]})

    per_img = {k: [] for k in models}
    lat = {k: [] for k in models}
    for r in range(a.repeats):
        for k, s in sess.items():                  # 번갈아 측정
            t0 = time.perf_counter()
            for i in range(a.n):
                t1 = time.perf_counter()
                s.run(None, {"image": x[i:i + 1]})
                lat[k].append((time.perf_counter() - t1) * 1000)
            per_img[k].append((time.perf_counter() - t0) * 1000 / a.n)
            print(f"[{threads}T] repeat {r + 1}/{a.repeats} {k:16s} {per_img[k][-1]:7.2f} ms/img", flush=True)

    out = {"date": datetime.now().isoformat(timespec="seconds"),
           "env": {**meta, "threads": threads}, "cond": {**vars(a), "threads": threads}, "results": {}}
    print(f"\n{'model':16s} {'MiB':>7s} {'mean':>8s} {'std':>6s} {'min':>7s} {'p50':>7s} {'p95':>7s}")
    for k, v in models.items():
        L = np.array(lat[k])
        res = {"file": C.onnx_path(v).name, "MiB": round(C.onnx_path(v).stat().st_size / 2**20, 1),
               "ms_per_img_mean": round(statistics.mean(per_img[k]), 2),
               "ms_per_img_std": round(statistics.stdev(per_img[k]), 2) if a.repeats > 1 else 0.0,
               "ms_per_img_min": round(min(per_img[k]), 2),
               "lat_p50": round(float(np.percentile(L, 50)), 2), "lat_p95": round(float(np.percentile(L, 95)), 2)}
        out["results"][k] = res
        print(f"{k:16s} {res['MiB']:7.1f} {res['ms_per_img_mean']:8.2f} {res['ms_per_img_std']:6.2f} "
              f"{res['ms_per_img_min']:7.2f} {res['lat_p50']:7.2f} {res['lat_p95']:7.2f}")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fn = C.RESULTS_DIR / f"speed_{datetime.now():%Y%m%d_%H%M%S}_{threads}T.json"
    json.dump(out, open(fn, "w"), indent=1, ensure_ascii=False)
    print("saved", fn)


if __name__ == "__main__":
    main()
