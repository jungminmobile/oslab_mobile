"""fuse_fp32.py — E12-2 2단계: 양자화 전 FP32 그래프에서 Add+LayerNorm 등을 합친다 (계획서 7절)

onnxruntime.transformers.optimizer.optimize_model(model_type="vit", num_heads=12, hidden_size=768, opt_level=0)
  - opt_level=0: ORT 세션 최적화는 하지 않고 파이썬 패턴 합치기만 한다
  - FusionOptions: enable_attention=False (Attention으로 합치면 안의 MatMul이 INT8 대상에서 빠짐, 계획서 2절)
합치기 전후 연산 수, 켠 옵션, MatMul 수(양자화 대상 수 보존 확인)를 --log(json)에 남긴다.
"""
import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402


def count(m):
    return dict(sorted(collections.Counter(f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in m.graph.node).items(),
                       key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    import onnxruntime as ort
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model
    t0 = time.time()
    before = count(onnx.load(a.src, load_external_data=False))
    fo = FusionOptions("vit")
    fo.enable_attention = False
    fo.use_multi_head_attention = False
    fo.enable_embed_layer_norm = False
    opt = optimize_model(a.src, model_type="vit", num_heads=12, hidden_size=768, opt_level=0,
                         optimization_options=fo, use_gpu=False)
    stats = opt.get_fused_operator_statistics()
    opt.save_model_to_file(a.dst, use_external_data_format=False)
    after = count(onnx.load(a.dst, load_external_data=False))
    ln_b = before.get("LayerNormalization", 0)
    fused_ln = after.get("com.microsoft.SkipLayerNormalization", 0) + after.get("SkipLayerNormalization", 0)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "src": Path(a.src).name, "dst": Path(a.dst).name,
           "dst_bytes": Path(a.dst).stat().st_size, "ort": ort.__version__, "onnx": onnx.__version__,
           "fusion_options": {k: v for k, v in vars(fo).items() if isinstance(v, (bool, int, float, str))},
           "optimize_model": {"model_type": "vit", "num_heads": 12, "hidden_size": 768, "opt_level": 0},
           "fused_operator_statistics": stats, "ops_before": before, "ops_after": after,
           "LayerNormalization_before": ln_b, "LayerNormalization_after": after.get("LayerNormalization", 0),
           "SkipLayerNormalization_after": fused_ln,
           "MatMul_before": before.get("MatMul", 0), "MatMul_after": after.get("MatMul", 0),
           "attention_ops_after": {k: v for k, v in after.items() if "Attention" in k},
           "G2a_fused_ln_ge_12": fused_ln >= 12, "matmul_preserved": before.get("MatMul", 0) == after.get("MatMul", 0),
           "git": SO.git_state(), "sec": round(time.time() - t0, 1)}
    Path(a.log).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("LayerNormalization_before", "LayerNormalization_after", "SkipLayerNormalization_after",
                                          "MatMul_before", "MatMul_after", "attention_ops_after", "fused_operator_statistics")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
