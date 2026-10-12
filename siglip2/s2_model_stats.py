"""
s2_model_stats.py — SigLIP2 이미지 인코더 파라미터 수·GMACs·ONNX 크기·peak 메모리 (ROADMAP G2, S-A행)
=================================================================================================

[목적]  결과표 S-A행의 모델 지표를 RN50 baseline_rn50.py와 같은 방식으로 구한다.

[파라미터 수]  두 방법으로 계산해 일치 확인 (VERIFICATION B 2단계)
  ① PyTorch model.visual 파라미터 합  ② FP32 ONNX initializer 원소 합
  (ONNX에는 상수 접기로 생긴 작은 상수가 더 있을 수 있어 차이를 함께 기록)

[GMACs]  두 방법
  ① fvcore FlopCountAnalysis (rn50/baseline_rn50.py와 같은 도구, "multiply-accumulate" 기준)
  ② ViT 구조로 직접 계산한 식 (패치 conv, 블록 12개의 QKV·QKᵀ·AV·proj·MLP, MAP 헤드)

[peak 메모리]  rn50/baseline_rn50.py mem_probe와 같은 방식: 모델마다 **새 프로세스**에서 ORT 세션을 만들고,
  1ms 간격으로 RSS를 샘플링해 "세션 생성 중 최대"와 "추론 중 최대"를 따로 기록 (PRD 5절).

[사용]
  RN50_RESULTS=../results/siglip2/G2_fp32_baseline "$SIGLIP2_PYTHON" s2_model_stats.py --models fp32=siglip2_image_fp32.onnx int8=<int8 파일>
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC


def analytic_gmacs(N=196, D=768, L=12, mlp=4, patch=16, C=3):
    """ViT-B/16 + MAP 헤드의 곱셈-누산 수 (LayerNorm·Softmax·GELU·bias 등 원소별 연산 제외 — fvcore와 같은 기준)."""
    patch_embed = N * D * (C * patch * patch)
    block = 3 * N * D * D + N * N * D + N * N * D + N * D * D + 2 * N * D * (mlp * D)
    head = 1 * D * D + 2 * N * D * D + 1 * N * D + 1 * N * D + D * D + 2 * D * (mlp * D)
    return {"patch_embed": patch_embed, "blocks": L * block, "head": head, "total_GMACs": (patch_embed + L * block + head) / 1e9}


def fvcore_gmacs(model):
    import torch
    from fvcore.nn import FlopCountAnalysis
    fca = FlopCountAnalysis(model.visual, torch.randn(1, 3, 224, 224))
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return {"total_GMACs": fca.total() / 1e9, "by_operator": {k: v / 1e9 for k, v in fca.by_operator().items()},
            "unsupported_ops": {k: int(v) for k, v in fca.unsupported_ops().items()}}


def onnx_params(path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    sizes = [int(np.prod(numpy_helper.to_array(t).shape)) for t in m.graph.initializer]
    big = sum(s for s in sizes if s > 16)
    return {"initializer_elements": int(sum(sizes)), "initializer_elements_gt16": int(big), "n_initializers": len(sizes)}


def mem_probe(path, threads, n_iter):
    """새 프로세스에서 실행되는 부분: 세션 생성·추론 중 RSS 최대값(MiB)을 JSON 한 줄로 출력."""
    import psutil
    proc = psutil.Process()
    peak = {"v": proc.memory_info().rss}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["v"] = max(peak["v"], proc.memory_info().rss)
            time.sleep(0.001)
    base = proc.memory_info().rss
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    s = RC.session(path, threads)
    after_load = proc.memory_info().rss
    load_peak = peak["v"]
    peak["v"] = after_load
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for _ in range(n_iter):
        s.run(None, {"image": x})
    stop.set()
    t.join()
    mb = lambda v: round(v / 2**20, 1)
    print(json.dumps({"rss_before_MiB": mb(base), "rss_after_load_MiB": mb(after_load),
                      "peak_during_load_MiB": mb(load_peak), "peak_during_infer_MiB": mb(peak["v"])}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["fp32=siglip2_image_fp32.onnx"], help="이름=파일")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mem-iters", type=int, default=50)
    ap.add_argument("--_mem-probe", dest="mem_probe", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.mem_probe:
        return mem_probe(RC.onnx_path(a.mem_probe), a.threads, a.mem_iters)

    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    model, _, _ = S.load_model()
    from s2_export_onnx import unfuse_attention
    unfuse_attention(model)   # fvcore는 fused SDPA를 세지 못하므로 ONNX와 같은 일반 연산 형태로 센다 (D-019)
    n_torch = sum(p.numel() for p in model.visual.parameters())
    fv = fvcore_gmacs(model)
    an = analytic_gmacs()
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
           "dirty": env["git"]["dirty"], "options": vars(a), "model": S.MODEL_ID,
           "params": {"pytorch_visual": int(n_torch)}, "gmacs": {"fvcore": fv, "analytic": an},
           "gmacs_rel_diff": round(abs(fv["total_GMACs"] - an["total_GMACs"]) / an["total_GMACs"], 5), "files": {}}
    for kv in a.models:
        name, f = kv.split("=", 1)
        p = RC.onnx_path(f)
        info = {"file": p.name, "MiB": round(p.stat().st_size / 2**20, 2)}
        if name == "fp32":
            info.update(onnx_params(p))
            row["params"]["onnx_initializers"] = info["initializer_elements"]
        r = subprocess.run([sys.executable, __file__, "--_mem-probe", f, "--threads", str(a.threads),
                            "--mem-iters", str(a.mem_iters)], capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-2000:]
        info["memory"] = json.loads(r.stdout.strip().splitlines()[-1])
        row["files"][name] = info
        print(name, json.dumps(info, ensure_ascii=False), flush=True)
    print(json.dumps({k: row[k] for k in ("params", "gmacs_rel_diff")}, ensure_ascii=False))
    print("GMACs fvcore", round(fv["total_GMACs"], 3), "analytic", round(an["total_GMACs"], 3))
    with open(out / "model_stats.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
