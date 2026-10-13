"""
s2_diag_int8_collapse.py — INT8 전부(MinMax) 붕괴 원인 진단 (G1 손실 감 잡기 → G3 예비 분석)
==========================================================================================

[배경]  siglip2_image_int8_minmax_highres256_s0_s8_all.onnx: CIFAR dev A top-1 76.9 → 1.6, cos 0.478 (정수 커널).

[가설과 확인 방법]  (결과를 보기 전에 정함)
  H1 활성값 이상치: 잔차 흐름·LayerNorm 입력 등 일부 텐서의 최대값이 99.9% 분위수보다 훨씬 커서 텐서별 8비트 눈금이 성글어짐
     → PyTorch hook으로 텐서별 max|x| / 99.9% 분위수 비율 측정 (ROADMAP G3 "활성값 이상치" 항목과 같은 지표)
  H2 실행 경로(정수 커널) 문제: 같은 QDQ 모델을 fake-quant(ORT_DISABLE_ALL, 디버깅 전용, D-008)로 돌려도 붕괴하면
     정수 커널 문제가 아니라 양자화 오차 자체
  H3 특정 연산 유형: 양자화 대상 연산 유형을 바꾼 변형 모델로 cos 비교
     V_matmul   : MatMul만 양자화 (Add·Softmax·LayerNorm·Mul 등은 float)
     V_no_add   : Add만 빼고 전부
     V_no_softmax: Softmax만 빼고 전부
  평가는 CIFAR dev A 앞 N장의 FP32 대비 cos와 top-1 (캐시 사용). 진단용이라 보고 수치가 아니다.

[2차 진단 (--round 2)]  1차 결과: MatMul만 0.76, QDQRegistry(MatMul+LayerNorm 등) 0.51, 전부 0.475 → LayerNorm과 fc2 입력 이상치 확인
  V2_default_no_ln   : ORT 기본 목록 − LayerNorm
  V2_mm_ln           : MatMul + LayerNorm
  V2_mm_no_fc2       : MatMul − 모든 fc2
  V2_mm_no_fc2_late  : MatMul − 블록 5~11 fc2 (이상치가 큰 블록)

[사용]
  RN50_RESULTS=../results/siglip2/G1_int8_minmax_all/diag "$SIGLIP2_PYTHON" s2_diag_int8_collapse.py --n 200 [--round 2]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S            # noqa: E402
import s2_quantize as SQ        # noqa: E402
from s2_export_onnx import FP32_IMAGE, unfuse_attention  # noqa: E402
from scripts import env_info    # noqa: E402

RC = S.RC
BASE = "siglip2_image_int8_minmax_highres256_s0_s8_all.onnx"
# 1차(ab52093): V_no_add·V_no_softmax는 QDQRegistry와의 교집합으로 목록을 만들어, ORT 기본(QLinearOpsRegistry ∪ QDQRegistry)과
#   달리 Add·Mul·Softmax 등이 처음부터 빠졌다 → 두 변형이 같은 모델("QDQRegistry만")이 됨. 2차부터는 기본 목록에서 뺀다.
VARIANTS = {"V_matmul": dict(op_types=["MatMul"]), "V_no_add": dict(drop="Add"), "V_no_softmax": dict(drop="Softmax")}
VARIANTS_2 = {
    "V2_default_no_ln": dict(drop="LayerNormalization"),             # 기본 목록 − LayerNorm
    "V2_mm_ln": dict(op_types=["MatMul", "LayerNormalization"]),     # MatMul + LayerNorm
    "V2_mm_no_fc2": dict(op_types=["MatMul"], exclude_sub="/mlp/fc2/"),          # MatMul − 모든 fc2
    "V2_mm_no_fc2_late": dict(op_types=["MatMul"], exclude_sub=[f"/blocks.{i}/mlp/fc2/" for i in range(5, 12)]),  # MatMul − 블록 5~11 fc2
}


def eval_model(path, x, y, W, ref, threads, fake):
    feat = RC.embed(path, x, threads, fake_quant=fake)
    m = RC.topk_metrics(feat, W, y, ref)
    return {k: m[k] for k in ("top1", "cos_mean", "cos_p1")}


def build_variant(name, spec, calib, threads):
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
    pre = RC.ONNX_DIR / FP32_IMAGE.replace(".onnx", "_pre.onnx")
    dst = RC.ONNX_DIR / f"diag_{name}.onnx"
    if "op_types" in spec:
        ops = spec["op_types"]
    else:
        present = {n.op_type for n in onnx.load(str(pre), load_external_data=False).graph.node}
        default = set(QDQRegistry) if spec.get("qdq_only") else set(QDQRegistry) | set(QLinearOpsRegistry)
        ops = sorted((default & present) - {spec["drop"]})
    excl = []
    if spec.get("exclude_sub"):
        subs = [spec["exclude_sub"]] if isinstance(spec["exclude_sub"], str) else spec["exclude_sub"]
        excl = [n.name for n in onnx.load(str(pre), load_external_data=False).graph.node if any(t in n.name for t in subs)]
        assert excl, f"제외할 노드 없음: {subs}"
    reader = SQ.CountingCalib(calib, 8)
    quantize_static(str(pre), str(dst), reader, quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=ops, nodes_to_exclude=excl,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    assert reader.counts() and all(c == len(calib) for c in reader.counts()), reader.counts()
    return dst, ops, len(excl)


def outlier_stats(n_img=32):
    """H1: 블록 단위 텐서(블록 출력=잔차 흐름, LayerNorm 입력, fc2 입력(GELU 출력), attention 확률)의 max|x| / p99.9 비율."""
    import torch
    model, preprocess, _ = S.load_model()
    unfuse_attention(model)
    x = torch.as_tensor(np.load(RC.CACHE_DIR / "pets_dev_x.npy", mmap_mode="r")[:n_img].copy())
    stats = {}

    def hook(name):
        def f(_, inp, out):
            t = (inp[0] if name.endswith(("norm1_in", "fc2_in")) else out).detach().abs().flatten().float()
            q = torch.quantile(t[torch.randperm(len(t))[:2_000_000]], 0.999).item()
            stats[name] = {"max": round(t.max().item(), 3), "p999": round(q, 4), "ratio": round(t.max().item() / max(q, 1e-12), 1)}
        return f
    hs = []
    for i, b in enumerate(model.visual.trunk.blocks):
        hs += [b.register_forward_hook(hook(f"block{i}_out")), b.norm1.register_forward_hook(hook(f"block{i}_norm1_in")),
               b.mlp.fc2.register_forward_hook(hook(f"block{i}_fc2_in"))]
    with torch.inference_mode():
        model.visual(x)
    for h in hs:
        h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--round", type=int, default=1, help="1: 1차 진단(H2·H3·H1), 2: 2차(LayerNorm·fc2 위치)")
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    f = lambda k: np.load(RC.CACHE_DIR / f"cifar_A_{k}.npy", mmap_mode="r" if k == "x" else None)
    x, y, W, ref = np.asarray(f("x")[:a.n]), f("y")[:a.n], f("clf"), f("ref_fp32")[:a.n]
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
           "dirty": env["git"]["dirty"], "n": a.n, "data": f"cifar_A[0:{a.n}]", "results": {}}
    t0 = time.time()
    res["round"] = a.round
    if a.round == 1:
        res["results"]["fp32"] = eval_model(RC.onnx_path(FP32_IMAGE), x, y, W, ref, a.threads, False)
        res["results"]["all_int"] = eval_model(RC.onnx_path(BASE), x, y, W, ref, a.threads, False)
        res["results"]["all_fake(H2)"] = eval_model(RC.onnx_path(BASE), x, y, W, ref, a.threads, True)
        print(json.dumps(res["results"], ensure_ascii=False), flush=True)
    calib = SQ.calib_images("highres", 256, 0)
    variants = VARIANTS if a.round == 1 else VARIANTS_2
    if a.round == 1:
        variants = {k: {**v, "qdq_only": True} if "drop" in v else v for k, v in variants.items()}   # 1차 재현용
    for name, spec in variants.items():
        dst, ops, n_ex = build_variant(name, spec, calib, a.threads)
        res["results"][name] = {"op_types": ops, "n_excluded_nodes": n_ex, **eval_model(dst, x, y, W, ref, a.threads, False)}
        print(name, json.dumps(res["results"][name], ensure_ascii=False), flush=True)
    if a.round == 1:
        res["outliers(H1)"] = outlier_stats()
    res["sec"] = round(time.time() - t0)
    if "outliers(H1)" in res:
        print(json.dumps(res["outliers(H1)"], ensure_ascii=False))
    with open(out / "diag.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
