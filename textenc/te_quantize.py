"""
te_quantize.py — 텍스트 인코더 INT8 변형 만들기 (스모크 3단계)
=========================================================

[변형]  (모두 FP32 ONNX(te_export.py)에서 출발. 양자화 설정은 이미지 쪽 siglip2/s2_quantize.py와 같게)
  A  동적 INT8: MatMul·Gemm 가중치 INT8(ORT quantize_dynamic 기본 per-tensor), 활성값은 실행 때 범위 계산
  B  정적 QDQ: MatMul·Gemm만 INT8 (텍스트 그래프의 out_proj·text_projection은 Gemm으로 나온다 → 같이 포함),
     가중치 S8 대칭 채널별, 활성값 U8 또는 S8 비대칭 텐서별, Percentile 99.999(비대칭 범위, CalibTensorRangeSymmetric=False),
     보정 문장 256개(tokens_calib_s0_256.npy). 잔차 Add·LayerNorm·GELU·Softmax는 FP32
  C  B + te_massive.py의 protect.json 층을 FP32로 (--exclude-json)
  D  토큰 임베딩 표만 INT8 (가중치만, 행(토큰)마다 눈금 1개, 대칭): Gather(int8 표) → Cast → × Gather(눈금).
     고른 행만 FP32로 바뀐다(표 전체를 펼치지 않음). ORT 기본 연산만 쓴다
  D에 --base <A·B·C 파일>을 주면 그 모델에 D를 덧붙인 조합이 된다 (조합은 따로 만들어 직접 잰다)

[보정 장수 확인]  보정기가 reader에서 가져간 문장 수를 회차마다 세서 요청 장수와 다르면 파일을 지우고 멈춘다
  (이미지 쪽 CountingCalib와 같은 규칙. CalibMaxIntermediateOutputs 옵션 금지).
[메모리]  Percentile 보정은 중간 텐서를 다 쌓는다 → siglip2 s2_quantize.enable_chunked_histogram(묶음 누적)을 import해 쓴다.

[사용]
  python textenc/te_quantize.py --variant A --out results/textenc/smoke
  python textenc/te_quantize.py --variant B --act-type u8 --out results/textenc/smoke
  python textenc/te_quantize.py --variant C --act-type s8 --exclude-json results/textenc/smoke/protect.json --out ...
  python textenc/te_quantize.py --variant D --out results/textenc/smoke
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_common as T                               # noqa: E402
from s2_quantize import enable_chunked_histogram    # noqa: E402  (siglip2/)

RC = T.RC


class CountingTokCalib:
    def __init__(self, tokens, bs):
        self.tk, self.bs, self.i, self.seen, self.passes = tokens, bs, 0, 0, []

    def get_next(self):
        if self.i >= len(self.tk):
            return None
        b = self.tk[self.i:self.i + self.bs]
        self.i += self.bs
        self.seen += len(b)
        return {"tokens": b}

    def rewind(self):
        if self.seen:
            self.passes.append(self.seen)
        self.i, self.seen = 0, 0

    def counts(self):
        return self.passes + ([self.seen] if self.seen else [])


def embed_int8(src, dst):
    """토큰 임베딩 Gather의 FP32 표(256000×768)를 행별 대칭 INT8 + 행별 눈금으로 바꾼다."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    hits = [n for n in g.node if n.op_type == "Gather" and n.input[0] in inits
            and list(inits[n.input[0]].dims) == [256000, 768]]
    assert len(hits) == 1, f"토큰 임베딩 Gather {len(hits)}개 (1개여야 함)"
    n = hits[0]
    W = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
    s = np.abs(W).max(1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    Wq = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    err = float(np.abs(Wq * s - W).max())
    name = n.input[0]
    g.initializer.remove(inits[name])
    g.initializer.extend([numpy_helper.from_array(Wq, name + "_int8"), numpy_helper.from_array(s.astype(np.float32), name + "_scale")])
    out = n.output[0]
    idx = list(g.node).index(n)
    g.node.remove(n)
    new = [helper.make_node("Gather", [name + "_int8", n.input[1]], [out + "_q"], name=n.name + "_q", axis=0),
           helper.make_node("Cast", [out + "_q"], [out + "_f"], name=n.name + "_cast", to=TensorProto.FLOAT),
           helper.make_node("Gather", [name + "_scale", n.input[1]], [out + "_s"], name=n.name + "_s", axis=0),
           helper.make_node("Mul", [out + "_f", out + "_s"], [out], name=n.name + "_deq")]   # 눈금 (B,64,1) 브로드캐스트
    for k, nn in enumerate(new):
        g.node.insert(idx + k, nn)
    onnx.save(m, str(dst))
    return {"embed_max_abs_err": err, "embed_scale_mean": float(s.mean())}


def op_counts(path, threads):
    """ORT 기본 설정(ORT_ENABLE_ALL)으로 최적화한 그래프의 연산 개수 — 정수 커널로 합쳐졌는지 확인."""
    import onnx
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt = Path(str(path).replace(".onnx", ".ortopt.onnx"))
    so.optimized_model_filepath = str(opt)
    so.log_severity_level = 3
    ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    c = collections.Counter(n.op_type for n in onnx.load(str(opt), load_external_data=False).graph.node)
    for f in opt.parent.glob(opt.name + "*"):
        f.unlink()
    opt.unlink(missing_ok=True)
    keys = ("MatMulInteger", "DynamicQuantizeMatMul", "QLinearMatMul", "MatMulIntegerToFloat", "QGemm", "QuantizeLinear",
            "DequantizeLinear", "DynamicQuantizeLinear", "MatMul", "Gemm", "FusedMatMul", "Gather", "Cast")
    return {k: c[k] for k in keys if c.get(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list("ABCD"), required=True)
    ap.add_argument("--act-type", choices=["u8", "s8"], default="s8")
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--calib", default="tokens_calib_s0_256.npy")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--hist-chunk", type=int, default=32, help="문장 수 단위로 히스토그램 누적 (메모리)")
    ap.add_argument("--exclude-json", default="")
    ap.add_argument("--base", default="", help="D 전용: 이 모델(A·B·C 결과)에 임베딩 INT8을 덧붙인 조합을 만든다")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    out = Path(a.out)
    env = T.env_row()
    src = T.TE_DIR / T.FP32
    pre = T.TE_DIR / T.FP32.replace(".onnx", "_pre.onnx")
    if not pre.exists() or pre.stat().st_mtime < src.stat().st_mtime:
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)   # rn50 텍스트와 같은 옵션
    tag = {"A": "dyn", "B": f"qdq_p{str(a.percentile).replace('.', '')}_{a.act_type}",
           "C": f"qdq_p{str(a.percentile).replace('.', '')}_{a.act_type}_protect", "D": "embed8"}[a.variant]
    assert not a.base or a.variant == "D", "--base는 D 전용"
    base = T.te_path(a.base) if a.base else None
    dst = (base.with_name(base.stem + "_embed8.onnx") if base else T.TE_DIR / f"siglip2_text_int8_{tag}.onnx")
    if dst.exists() and not a.overwrite:
        raise SystemExit(f"{dst.name} 이미 있음 (--overwrite)")
    tmp = dst.with_name(dst.stem + ".tmp.onnx")
    t0, info = time.time(), {}
    excl = []
    if a.variant == "C":
        assert a.exclude_json, "C는 --exclude-json 필요 (te_massive.py의 protect.json)"
        excl = json.load(open(a.exclude_json))["nodes"]
        assert excl, "보호할 층이 없음 — C를 만들 이유가 없다 (B와 같음)"
    if a.variant == "D":
        info.update(embed_int8(base or src, tmp))
        if base:
            info["base"] = base.name
    elif a.variant == "A":
        quantize_dynamic(str(pre), str(tmp), weight_type=QuantType.QInt8, per_channel=False,
                         op_types_to_quantize=["MatMul", "Gemm"])
    else:
        tk = np.load(T.te_path(a.calib))[:a.calib_n]
        assert len(tk) == a.calib_n, f"보정 문장 {len(tk)} < {a.calib_n}"
        assert a.hist_chunk % a.calib_bs == 0
        enable_chunked_histogram(a.hist_chunk // a.calib_bs)
        reader = CountingTokCalib(tk, a.calib_bs)
        extra = {"ActivationSymmetric": False, "WeightSymmetric": True, "CalibPercentile": a.percentile,
                 "CalibTensorRangeSymmetric": False}
        assert "CalibMaxIntermediateOutputs" not in extra
        quantize_static(str(pre), str(tmp), reader, quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8 if a.act_type == "s8" else QuantType.QUInt8,
                        weight_type=QuantType.QInt8, per_channel=True, calibrate_method=CalibrationMethod.Percentile,
                        op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl, extra_options=extra)
        seen = reader.counts()
        if not (seen and all(c == a.calib_n for c in seen)):
            tmp.unlink()
            raise AssertionError(f"보정에 쓰인 문장 {seen} ≠ 요청 {a.calib_n} — 모델 파일을 지웠음")
        info["calib_seen"] = seen
    tmp.replace(dst)
    mib = dst.stat().st_size / 2**20
    print(f"saved {dst.name}  {mib:.1f} MiB  ({time.time() - t0:.0f}s) {info}")
    row = {**T.base_row(env, vars(a)), "step": "quantize", "variant": ("D+" + base.name if base else a.variant),
           "model": dst.name, "MiB": round(mib, 2), "sha256": T.sha256(dst), "n_excluded": len(excl), "excluded": excl,
           "ort_optimized_ops": op_counts(dst, a.threads), "sec": round(time.time() - t0), **info}
    print(json.dumps(row["ort_optimized_ops"]))
    T.append_jsonl(out / "quantize_log.jsonl", row)


if __name__ == "__main__":
    main()
