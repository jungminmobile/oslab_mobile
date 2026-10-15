"""
s2_quantize.py — SigLIP2 이미지 인코더 ONNX Runtime 정적 INT8 양자화 (rn50/quantize_onnx.py의 부품을 import)
========================================================================================================

[목적]
  rn50/quantize_onnx.py와 같은 방식(QDQ, 가중치 INT8 대칭 채널별, 활성값 8비트 비대칭 텐서별)으로 SigLIP2를 양자화한다.
  rn50 쪽에서 가져다 쓰는 것: ImageCalib(보정 배치 공급), select_excluded(--exclude/--keep/--only/--only-exact 규칙), METHODS.
  여기에만 있는 것: SigLIP2 전처리로 만든 보정 사진, 보정 장수 assert, 정수 커널 그래프 노드 목록 기록.

[보정 사진 (누수 방지, PRD 5절)]
  --calib-source highres : s2_build_dev.py가 만든 calib_<이름>_f32.npy (고화질 5종 train 중 dev와 겹치지 않는 사진)
  --calib-source cifar   : CIFAR-100 train[0:45000]에서 무작위 (SigLIP2 전처리)
  --calib-source mix     : 반씩
  --method minmax | percentile(--percentile, 비대칭 범위) | entropy
  --op-types MatMul       : 이 연산 유형만 양자화 (기본: ORT 기본 목록 전부). 파일 이름에 _ops-MatMul이 붙는다
  --hist-chunk 16         : percentile·entropy를 16장씩 나눠 히스토그램 누적 (메모리 부족 대책, 파일 이름에 _hc16)
  기본값은 PRD의 E3 전 기본값: highres 256장, 시드 0. 시드 고정으로 섞은 뒤 배치로 나눈다.
  보정 장수 확인 (RN50 "256장 중 16장" 버그 재발 방지):
    ① 보정기가 reader에서 가져간 사진 수 = 요청 장수 (reader를 끝까지 읽었는지)
    ② extra_options에 CalibMaxIntermediateOutputs가 없음 — 이 옵션이 있으면 ORT 1.25 MinMaxCalibrater가 reader는 끝까지 읽되
       중간 결과를 버리므로 ①만으로는 버그를 못 잡는다. 그 버그를 실제로 막는 것은 ②다.
  assert가 실패하면 모델 파일을 남기지 않는다 (임시 이름에 쓴 뒤 통과하면 이름을 바꾼다).
  보정 후보 풀 크기(rn50 datasets.CAL_N에 따라 달라짐)를 로그에 남긴다.

[활성값 형식]  --act-type s8(기본) / u8
  PRD 3절: x86에서는 S8S8도 U8S8로 바뀌어 계산이 같다. PRD E0의 기본값 계획(S8S8)을 따른다 (D-023).

[--dump-ops]  정수 커널 확인 (ROADMAP G3 "먼저 확인")
  ORT 기본 설정(ORT_ENABLE_ALL)으로 세션을 만들 때 최적화된 그래프를 저장하고, 연산 종류별 개수를 기록한다.
  QLinearMatMul / MatMulInteger / QLinearConv 등이 있으면 정수 커널, LayerNormalization·Softmax·Erf가 float으로 남는지 확인.

[사용]
  "$SIGLIP2_PYTHON" s2_quantize.py --calib-source highres --calib-n 256 --tag all --dump-ops
  → $SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_minmax_highres256_s0_s8_all.onnx
"""
import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S                      # noqa: E402  (RN50_* 경로 + rn50 import)
from s2_export_onnx import FP32_IMAGE     # noqa: E402
from scripts import env_info              # noqa: E402
import quantize_onnx as RQ                # noqa: E402  rn50/quantize_onnx.py
assert Path(RQ.__file__).resolve().parent == S.RN50_DIR.resolve(), RQ.__file__

RC = S.RC
RS_CAL_N = lambda: int(S.RD.CAL_N)


def enable_chunked_histogram(chunk_batches):
    """ORT 1.25 HistogramCalibrater.collect_data는 모든 배치의 중간 텐서를 메모리에 쌓은 뒤 한 번에 히스토그램을 만든다
    (calibrate.py 601~651행) → ViT·256장은 16 GB를 넘어 실패. 이 프로세스 안에서만 collect_data를 바꿔
    chunk_batches 배치마다 HistogramCollector.collect(ORT가 제공하는 히스토그램 합치기, merge_histogram)로 누적하고 비운다.
    ORT·RN50 코드는 바꾸지 않는다. 한꺼번에 계산한 것과의 차이는 s2_hist_chunk_check.py로 재서 기록한다."""
    import copy
    from onnxruntime.quantization.calibrate import HistogramCalibrater, HistogramCollector

    def collect_data(self, data_reader):
        input_names_set = {a.name for a in self.infer_session.get_inputs()}
        output_names = [a.name for a in self.infer_session.get_outputs()]
        flushed = 0

        def flush():
            merged = {}
            for out in self.intermediate_outputs:
                for k, v in zip(output_names, out):
                    merged.setdefault(k, []).append(v)
            clean = {k: v for k, v in merged.items() if k in self.tensors_to_calibrate}
            if not self.collector:
                self.collector = HistogramCollector(method=self.method, symmetric=self.symmetric,
                                                    num_bins=self.num_bins, num_quantized_bins=self.num_quantized_bins,
                                                    percentile=self.percentile, scenario=self.scenario)
            self.collector.collect(clean)
            self.clear_collected_data()

        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            outputs = self.infer_session.run(None, inputs)
            self.intermediate_outputs.append([copy.copy(o) if output_names[i] in input_names_set else o
                                              for i, o in enumerate(outputs)])
            if len(self.intermediate_outputs) >= chunk_batches:
                flush()
                flushed += 1
        if self.intermediate_outputs:
            flush()
            flushed += 1
        if flushed == 0:
            raise ValueError("No data is collected.")

    HistogramCalibrater.collect_data = collect_data   # 보정 후보 풀 장수 규칙 (PRD E0 ⑦에서 64 → 256 예정)


class CountingCalib(RQ.ImageCalib):
    """rn50 ImageCalib + 보정기가 실제로 가져간 사진 수 세기."""
    def __init__(self, x, bs):
        super().__init__(x, bs)
        self.seen = 0
        self.passes = []          # 끝까지 읽은(또는 rewind된) 회차마다 가져간 사진 수

    def get_next(self):
        b = super().get_next()
        if b is not None:
            self.seen += len(b["image"])
        return b

    def rewind(self):
        if self.seen:
            self.passes.append(self.seen)
        super().rewind()
        self.seen = 0

    def counts(self):
        return self.passes + ([self.seen] if self.seen else [])


def calib_images(source, n, seed):
    """보정 사진 n장 (n,3,224,224) float32, SigLIP2 전처리. dev·test와 겹치지 않는 사진만."""
    rng = np.random.default_rng(seed)
    parts = []
    n_cifar = {"cifar": n, "mix": n // 2, "highres": 0}[source]
    if n_cifar:
        from torchvision.datasets import CIFAR100
        _, preprocess, _ = S.load_model()
        ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=False)
        idx = rng.choice(RC.CALIB_RANGE[1], n_cifar, replace=False)        # train[0:45000]에서만
        parts.append(np.stack([preprocess(ds[int(i)][0]).numpy() for i in idx]))
    n_hr = n - n_cifar
    if n_hr:
        files = sorted(RC.CACHE_DIR.glob("calib_*_f32.npy"))
        assert len(files) == 5, f"고화질 5종 보정 후보가 필요함, 찾은 것: {[f.name for f in files]} — 먼저 s2_build_dev.py"
        pool = np.concatenate([np.load(f) for f in files])
        assert len(pool) >= n_hr, f"보정 후보 {len(pool)}장 < 요청 {n_hr}장"
        parts.append(pool[rng.choice(len(pool), n_hr, replace=False)])
    x = np.concatenate(parts).astype(np.float32)
    assert len(x) == n
    return x[rng.permutation(len(x))]


def dump_ops(path, threads):
    """ORT_ENABLE_ALL로 최적화된 그래프를 저장해 연산 종류를 센다 (정수 커널 여부 확인)."""
    import onnx
    import onnxruntime as ort
    opt = path.with_name(path.stem + "_ortopt.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.optimized_model_filepath = str(opt)
    so.log_severity_level = 3
    ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    g = onnx.load(str(opt), load_external_data=False).graph
    ops = collections.Counter(f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in g.node)
    opt.unlink()
    for ext in (".data",):
        if opt.with_suffix(opt.suffix + ext).exists():
            opt.with_suffix(opt.suffix + ext).unlink()
    int_ops = {k: v for k, v in ops.items() if any(t in k for t in ("QLinear", "Integer", "QGemm", "QAttention"))}
    qdq_ops = {k: v for k, v in ops.items() if k in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear")}
    float_ops = {k: v for k, v in ops.items() if k in ("MatMul", "Conv", "Gemm", "Softmax", "LayerNormalization", "Erf")}
    # int_kernel_ops: 정수로 계산하는 연산 / qdq_ops: 변환 노드(정수 계산 아님) / float_compute_ops: float으로 남은 계산 연산
    return {"ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])), "int_kernel_ops": int_ops, "qdq_ops": qdq_ops,
            "float_compute_ops": float_ops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-source", choices=["cifar", "highres", "mix"], default="highres")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--method", choices=list(RQ.METHODS), default="minmax")
    ap.add_argument("--act-type", choices=["s8", "u8"], default="s8")
    ap.add_argument("--percentile", type=float, default=99.999,
                    help="--method percentile 전용. 범위는 비대칭(CalibTensorRangeSymmetric=False, PRD 3절)")
    ap.add_argument("--op-types", default="", help="양자화할 연산 유형(쉼표). 비우면 ORT 기본(전부). 예: MatMul")
    ap.add_argument("--hist-chunk", type=int, default=0,
                    help="percentile·entropy 전용: 이 장수마다 히스토그램을 누적하고 중간 텐서를 비움(메모리). 0이면 ORT 표준")
    ap.add_argument("--per-channel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--only-exact", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dump-ops", action="store_true")
    ap.add_argument("--reduce-range", action="store_true",
                    help="(기본 꺼짐) 가중치를 7비트 범위로 (ORT reduce_range). U8S8 정수 커널 포화 대책 (PRD 3절, 10/15)")
    ap.add_argument("--weight-type", choices=["s8", "u8"], default="s8",
                    help="가중치 형식. 기본 s8. u8은 U8U8 (PRD 3절 비-VNNI 대책 후보, 10/15)")
    ap.add_argument("--u8-match", default="",
                    help="(기본 꺼짐) 이름에 이 문자열(쉼표로 여러 개)이 들어간 MatMul의 활성값 입력·출력만 U8, 나머지는 --act-type 그대로. "
                         "예: /attn/qkv/,/attn_pool/kv/ — S8에서 QKV 투영 MatMul이 정수로 합쳐지지 않는 문제 대책 (10/14)")
    ap.add_argument("--attn-u8", action="store_true",
                    help="(기본 꺼짐) 활성값×활성값 MatMul(attention QKᵀ·AV)의 입력·출력만 U8, 나머지는 --act-type 그대로. "
                         "ORT TensorQuantOverrides(quant_type) 사용. (10/14 결과: 효과 없음 — S8에서도 QKᵀ·AV는 이미 정수로 합쳐지고, 합쳐지지 않던 것은 QKV 투영이었다 → --u8-match)")
    ap.add_argument("--w4-regex", default="",
                    help="(기본 꺼짐) 이름이 이 정규식에 맞는 MatMul의 가중치를 4비트 블록 양자화(MatMulNBits)로 바꾼 뒤, 나머지를 평소처럼 INT8 QDQ로 양자화. "
                         r"예: '^/blocks\.\d+/mlp/fc1/' (10/15, 대회 안내 4.1 'INT8/INT4', D-036·D-037)")
    ap.add_argument("--w4-block", type=int, default=32, help="4비트 블록 크기 (--w4-regex일 때)")
    ap.add_argument("--w4-acc", type=int, default=4, help="MatMulNBits accuracy_level (4 = 활성값을 int8로 동적 양자화해 정수 계산)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log", default="", help="기본: $RN50_RESULTS/quantize_log.jsonl (RN50_RESULTS 명시 필요)")
    a = ap.parse_args()
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = RC.ONNX_DIR / FP32_IMAGE
    pre = RC.ONNX_DIR / FP32_IMAGE.replace(".onnx", "_pre.onnx")
    if not pre.exists() or pre.stat().st_mtime < src.stat().st_mtime:
        quant_pre_process(str(src), str(pre))
    meth = f"percentile{str(a.percentile).replace('.', '')}" if a.method == "percentile" else a.method
    ops_tag = f"_ops-{a.op_types.replace(',', '-')}" if a.op_types else ""
    if a.hist_chunk:
        assert a.method in ("percentile", "entropy"), "--hist-chunk는 percentile·entropy 전용"
        assert a.hist_chunk % a.calib_bs == 0, "--hist-chunk는 --calib-bs의 배수"
        enable_chunked_histogram(a.hist_chunk // a.calib_bs)
        ops_tag += f"_hc{a.hist_chunk}"
    if a.reduce_range:
        ops_tag += "_rr"
    if a.weight_type != "s8":
        ops_tag += f"_w{a.weight_type}"
    cal = f"{meth}_{a.calib_source}{a.calib_n}_s{a.seed}_{a.act_type}{ops_tag}"
    dst = RC.ONNX_DIR / f"siglip2_image_int8_{cal}{'_' + a.tag if a.tag else ''}.onnx"
    if dst.exists() and not a.overwrite:
        raise SystemExit(f"{dst.name} 이미 있음 (--overwrite로 덮어쓰기)")
    excl = RQ.select_excluded(str(pre), a)
    w4_nodes = []
    if a.w4_regex:
        import re
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        g = onnx.load(str(pre), load_external_data=False).graph
        inits = {i.name for i in g.initializer}
        w4_nodes = [n.name for n in g.node if n.op_type == "MatMul" and n.name not in excl
                    and re.search(a.w4_regex, n.name) and n.input[1] in inits]
        assert w4_nodes, f"--w4-regex {a.w4_regex!r}에 맞는 가중치 MatMul 없음"
        print(f"--w4-regex: MatMul {len(w4_nodes)}개를 4비트로 (블록 {a.w4_block}, accuracy_level {a.w4_acc})")
        ops_tag += f"_w4b{a.w4_block}a{a.w4_acc}"
        cal = f"{meth}_{a.calib_source}{a.calib_n}_s{a.seed}_{a.act_type}{ops_tag}"
        dst = RC.ONNX_DIR / f"siglip2_image_int8_{cal}{'_' + a.tag if a.tag else ''}.onnx"
        if dst.exists() and not a.overwrite:
            raise SystemExit(f"{dst.name} 이미 있음 (--overwrite로 덮어쓰기)")
        # ORT 1.25: nodes_to_include는 "추가로 포함"이라 기본 op_types(MatMul)와 합쳐져 가중치 MatMul 전부가 4비트가 된다
        #  (10/16 04:38 확인: 12개 요청 → 53개 변환, 아래 개수 assert가 잡음). 그래서 고르지 않은 가중치 MatMul은 nodes_to_exclude로 뺀다
        w_all = [n.name for n in g.node if n.op_type == "MatMul" and n.input[1] in inits]
        q4 = MatMulNBitsQuantizer(onnx.load(str(pre)), bits=4, block_size=a.w4_block, is_symmetric=True,
                                  accuracy_level=a.w4_acc, nodes_to_include=w4_nodes,
                                  nodes_to_exclude=[n for n in w_all if n not in set(w4_nodes)])
        q4.process()
        pre = dst.with_name(dst.stem + ".w4pre.onnx")             # 4비트로 바꾼 중간 모델 → 아래 INT8 QDQ의 입력
        q4.model.save_model_to_file(str(pre), use_external_data_format=False)
        g4 = onnx.load(str(pre), load_external_data=False).graph
        n_nbits = sum(n.op_type == "MatMulNBits" for n in g4.node)
        assert n_nbits == len(w4_nodes), f"MatMulNBits {n_nbits}개 ≠ 요청 {len(w4_nodes)}개"
    attn_tensors = []
    if a.attn_u8:
        import onnx
        assert a.act_type == "s8", "--attn-u8은 S8 활성값에서 attention만 U8로 바꾸는 옵션"
        g = onnx.load(str(pre), load_external_data=False).graph
        inits = {i.name for i in g.initializer}
        consts = {o for n in g.node if n.op_type == "Constant" for o in n.output}
        attn_nodes = [n for n in g.node if n.op_type == "MatMul" and n.name not in excl
                      and not any(i in inits or i in consts for i in n.input)]
        assert attn_nodes, "활성값×활성값 MatMul을 찾지 못함"
        attn_tensors = sorted({t for n in attn_nodes for t in (*n.input, *n.output)})
        ops_tag += "_attnu8"
        cal = f"{meth}_{a.calib_source}{a.calib_n}_s{a.seed}_{a.act_type}{ops_tag}"
        dst = RC.ONNX_DIR / f"siglip2_image_int8_{cal}{'_' + a.tag if a.tag else ''}.onnx"
        if dst.exists() and not a.overwrite:
            raise SystemExit(f"{dst.name} 이미 있음 (--overwrite로 덮어쓰기)")
    log_path = Path(a.log) if a.log else S.require_results_dir() / "quantize_log.jsonl"

    t0 = time.time()
    x = calib_images(a.calib_source, a.calib_n, a.seed)
    pool_n = sum(len(np.load(f, mmap_mode="r")) for f in sorted(RC.CACHE_DIR.glob("calib_*_f32.npy")))
    reader = CountingCalib(x, a.calib_bs)
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    if a.method == "percentile":
        # ORT 1.25 Percentile 보정기 기본은 대칭 범위(−t~t)라 한쪽으로 치우친 활성값에서 눈금 절반을 버린다 → 비대칭으로 (PRD 3절)
        extra.update({"CalibPercentile": a.percentile, "CalibTensorRangeSymmetric": False})
    assert "CalibMaxIntermediateOutputs" not in extra, "보정 결과를 버리는 옵션 금지 (RN50 16장 버그)"
    if a.u8_match:
        import onnx
        assert a.act_type == "s8" and not a.attn_u8, "--u8-match는 S8 활성값에서만, --attn-u8과 함께 쓰지 않음"
        pats = [t for t in a.u8_match.split(",") if t]
        g = onnx.load(str(pre), load_external_data=False).graph
        inits = {i.name for i in g.initializer}
        consts = {o for n in g.node if n.op_type == "Constant" for o in n.output}
        hit = [n for n in g.node if n.op_type == "MatMul" and n.name not in excl and any(t in n.name for t in pats)]
        assert hit, f"--u8-match {pats}에 맞는 MatMul 없음"
        attn_tensors = sorted({t for n in hit for t in (*n.input, *n.output) if t not in inits and t not in consts})
        tag = "_u8m-" + "-".join(t.strip("/").replace("/", ".") for t in pats)
        ops_tag += tag
        cal = f"{meth}_{a.calib_source}{a.calib_n}_s{a.seed}_{a.act_type}{ops_tag}"
        dst = RC.ONNX_DIR / f"siglip2_image_int8_{cal}{'_' + a.tag if a.tag else ''}.onnx"
        if dst.exists() and not a.overwrite:
            raise SystemExit(f"{dst.name} 이미 있음 (--overwrite로 덮어쓰기)")
        print(f"--u8-match: MatMul {len(hit)}개, U8로 둘 텐서 {len(attn_tensors)}개")
    if attn_tensors:
        extra["TensorQuantOverrides"] = {t: [{"quant_type": QuantType.QUInt8}] for t in attn_tensors}
    tmp = dst.with_name(dst.stem + ".tmp.onnx")
    try:
        _quantize_static_checked(quantize_static, pre, tmp, reader, a, excl, extra, QuantFormat, QuantType)
    finally:
        if w4_nodes and pre.exists():
            pre.unlink()                                        # 4비트 중간 모델은 성공·실패 모두 지움 (수백 MB, 10/15 검토 지적)
    seen = reader.counts()
    tmp.replace(dst)
    print(f"saved {dst.name}  {dst.stat().st_size / 2**20:.1f} MiB  ({time.time() - t0:.0f}s), calib seen per pass={seen}")
    _log(a, dst, seen, pool_n, excl, attn_tensors, w4_nodes, t0, log_path)


def _quantize_static_checked(quantize_static, pre, tmp, reader, a, excl, extra, QuantFormat, QuantType):
    quantize_static(str(pre), str(tmp), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8 if a.act_type == "s8" else QuantType.QUInt8,
                    weight_type=QuantType.QInt8 if a.weight_type == "s8" else QuantType.QUInt8, per_channel=bool(a.per_channel),
                    reduce_range=bool(a.reduce_range),
                    calibrate_method=RQ.METHODS[a.method], nodes_to_exclude=excl, extra_options=extra,
                    op_types_to_quantize=[o for o in a.op_types.split(",") if o] or None)
    # 보정기가 데이터를 읽은 회차마다 가져간 사진 수. 모든 회차가 요청 장수와 같아야 한다
    seen = reader.counts()
    if not (seen and all(c == a.calib_n for c in seen)):
        tmp.unlink()
        raise AssertionError(f"보정에 쓰인 사진 {seen} ≠ 요청 {a.calib_n}장 — 모델 파일을 지웠음")


def _log(a, dst, seen, pool_n, excl, attn_tensors, w4_nodes, t0, log_path):
    env = env_info.collect()
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
           "dirty": env["git"]["dirty"], "options": vars(a), "model": dst.name,
           "MiB": round(dst.stat().st_size / 2**20, 2), "calib_seen": seen, "calib_pool_n": int(pool_n),
           "rn50_CAL_N": RS_CAL_N(), "hist_chunk": a.hist_chunk, "n_excluded_nodes": len(excl),
           "qtype": f"{a.act_type.upper()}{a.weight_type.upper()}" + ("+rr" if a.reduce_range else "") + ("+attnU8" if a.attn_u8 else "") + (f"+U8[{a.u8_match}]" if a.u8_match else "") + (f"+W4b{a.w4_block}a{a.w4_acc}" if a.w4_regex else ""), "sec": round(time.time() - t0)}
    if a.attn_u8 or a.u8_match:
        row["u8_override_tensors"] = len(attn_tensors)
    if w4_nodes:
        row["w4_nodes"] = w4_nodes
    if a.dump_ops:
        row["ort_optimized_graph"] = dump_ops(dst, a.threads)
        print(json.dumps(row["ort_optimized_graph"], ensure_ascii=False))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
