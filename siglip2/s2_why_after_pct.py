"""
s2_why_after_pct.py — G8 E4b: E4를 실제 보정 방식(Percentile 99.999) 눈금으로 다시 + 큰 값이 어느 토큰에 있는지 (2026-10-21)
==================================================================================================================
[계기] E4(MinMax 눈금): M2를 써도 블록 8~11 fc2 입력은 97~99%가 0으로 반올림된다. 그런데 M2 모델의 실제 INT8 정확도는 좋다(−0.62).
       배포 모델은 MinMax가 아니라 Percentile 99.999로 범위를 정하므로, 드문 큰 값을 잘라낸 눈금에서 다시 잰다.
[방법] E4와 같은 FP32 그래프 4개·같은 사진 64장·같은 MatMul 입력 79개. 텐서마다
  - 범위: ORT HistogramCalibrater percentile(비대칭)과 같은 꼴 — 양쪽에서 (100−99.999)/2 = 0.0005%씩 잘라낸 값
    (64장 전체에서 위·아래 k개를 정확히 추적, k = ceil(전체 원소 수 × 5e-6)). 0을 포함하도록 넓힘
  - 0으로 반올림되는 비율 (|x| < scale/2), 범위 밖으로 잘리는 원소 비율
  - **출력 SNR**: 가중치 MatMul(두 번째 입력이 initializer)이면 활성값만 그 눈금으로 8비트 가짜 양자화했을 때의 출력 오차
    SNR = 10·log10(Σ‖y‖² / Σ‖ŷ − y‖²) (dB, 사진 64장 합). "0으로 반올림되는 비율"이 높아도 그 값들이 원래 0 근처면 출력 오차는 작다 —
    정확도와 직접 이어지는 것은 이쪽이다 (가중치는 FP32 그대로, 활성값 효과만)
  - (fc1·fc2 입력) 사진마다 |x| 최댓값이 있는 토큰 위치 → 자주 나오는 위치 3개와 비율 (sink 토큰 55·13·1인지)
  ※ ORT는 히스토그램(bin 2048, 16장씩 누적)으로 percentile을 근사하므로 실제 눈금과 조금 다를 수 있다 — 같은 기준으로 그래프끼리 비교하는 용도
[출력] $RN50_RESULTS/why_after_pct.jsonl, why_after_pct_summary.json
"""
import argparse
import collections
import json
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402
from s2_why_after import matmul_inputs, role, SETS  # noqa: E402

RC = S.RC
PCT = 99.999


def run(path, x, threads):
    import onnx
    import onnxruntime as ort
    m = onnx.load(str(path))
    mins = matmul_inputs(m)
    from onnx import numpy_helper                  # 가중치는 external data로 저장하기 전에 꺼낸다 (저장하면 참조가 바뀜)
    inits = {i.name: i for i in m.graph.initializer}
    min_names = {t for _, t in mins}
    wmat = {}
    for n in m.graph.node:
        if n.op_type == "MatMul" and n.input[0] in min_names and n.input[1] in inits:
            wmat[n.input[0]] = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
    have = {o.name for o in m.graph.output}
    for _, t in mins:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    tmp = Path(tempfile.mkdtemp()) / "with_outputs.onnx"
    onnx.save(m, str(tmp), save_as_external_data=True, location="w.bin")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    names = [t for _, t in mins]
    roles = {t: role(n) for n, t in mins}
    K, top, bot, ntot = {}, {}, {}, {}
    argtok = {t: collections.Counter() for t in names}
    for i in range(len(x)):                                     # 1회차: 위·아래 k개 추적, 최댓값 토큰
        for t, v in zip(names, sess.run(names, {"image": x[i:i + 1]})):
            v = v.astype(np.float32)
            flat = v.ravel()
            if t not in K:
                K[t] = max(1, math.ceil(flat.size * len(x) * (100 - PCT) / 200))
                top[t] = np.empty(0, np.float32)
                bot[t] = np.empty(0, np.float32)
                ntot[t] = 0
            k = min(K[t], flat.size)
            top[t] = np.concatenate([top[t], np.partition(flat, -k)[-k:]])
            top[t] = np.partition(top[t], -K[t])[-K[t]:] if top[t].size > K[t] else top[t]
            bot[t] = np.concatenate([bot[t], np.partition(flat, k - 1)[:k]])
            bot[t] = np.partition(bot[t], K[t] - 1)[:K[t]] if bot[t].size > K[t] else bot[t]
            ntot[t] += flat.size
            if ("fc1" in roles[t] or "fc2" in roles[t]) and v.ndim == 3:   # (1, 토큰, 채널)
                argtok[t][int(np.abs(v[0]).max(axis=1).argmax())] += 1
    hi = {t: max(float(top[t].min()), 0.0) for t in names}
    lo = {t: min(float(bot[t].max()), 0.0) for t in names}
    scale = {t: (hi[t] - lo[t]) / 255.0 for t in names}
    zc = collections.Counter()
    cl = collections.Counter()
    sig = collections.Counter()
    err = collections.Counter()
    for i in range(len(x)):                                     # 2회차: 그 눈금에서 0으로 반올림·잘림 비율
        for t, v in zip(names, sess.run(names, {"image": x[i:i + 1]})):
            v = v.astype(np.float32)
            zc[t] += int((np.abs(v) < scale[t] / 2).sum())
            cl[t] += int(((v > hi[t]) | (v < lo[t])).sum())
            if t in wmat:
                zp = np.clip(np.round(-lo[t] / scale[t]), 0, 255)
                vq = (np.clip(np.round(v / scale[t]) + zp, 0, 255) - zp) * scale[t]
                y = v.reshape(-1, v.shape[-1]) @ wmat[t]
                yq = vq.reshape(-1, v.shape[-1]) @ wmat[t]
                sig[t] += float((y.astype(np.float64) ** 2).sum())
                err[t] += float(((yq - y).astype(np.float64) ** 2).sum())
    return [{"node": n, "role": roles[t], "n": ntot[t], "k_each_side": K[t], "lo": lo[t], "hi": hi[t], "scale": scale[t],
             "zero_frac": zc[t] / ntot[t], "clip_frac": cl[t] / ntot[t],
             "out_snr_db": (10 * math.log10(sig[t] / err[t]) if err[t] > 0 else None) if t in wmat else None,
             "max_token_top3": [[tok, c / len(x)] for tok, c in argtok[t].most_common(3)]} for n, t in mins]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    x = np.concatenate([np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n]) for d in SETS]).astype(np.float32)
    summary = {}
    with open(out / "why_after_pct.jsonl", "a", encoding="utf-8") as f:
        for spec in a.models:
            name, fn = spec.split("=", 1)
            rows = run(RC.onnx_path(fn), x, a.threads)
            for r in rows:
                f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                                    "dirty": env["git"]["dirty"], "graph": name, "file": fn, "n_images": len(x), "percentile": PCT, **r},
                                   ensure_ascii=False) + "\n")
            summary[name] = {r["role"]: {"zero": round(r["zero_frac"], 4), "clip": round(r["clip_frac"], 6),
                                         "snr_db": None if r["out_snr_db"] is None else round(r["out_snr_db"], 2), "max_tok": r["max_token_top3"]}
                             for r in rows if "fc1" in r["role"] or "fc2" in r["role"]}
            summary[name]["_mean_zero_all"] = round(float(np.mean([r["zero_frac"] for r in rows])), 4)
            print(name, json.dumps({k: (v["zero"], v["snr_db"]) if isinstance(v, dict) else v for k, v in summary[name].items()}, ensure_ascii=False), flush=True)
    json.dump(summary, open(out / "why_after_pct_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
