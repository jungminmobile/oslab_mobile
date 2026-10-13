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
RS_CAL_N = lambda: int(S.RD.CAL_N)   # 보정 후보 풀 장수 규칙 (PRD E0 ⑦에서 64 → 256 예정)


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
    ap.add_argument("--per-channel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--only-exact", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dump-ops", action="store_true")
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
    cal = f"{meth}_{a.calib_source}{a.calib_n}_s{a.seed}_{a.act_type}{ops_tag}"
    dst = RC.ONNX_DIR / f"siglip2_image_int8_{cal}{'_' + a.tag if a.tag else ''}.onnx"
    if dst.exists() and not a.overwrite:
        raise SystemExit(f"{dst.name} 이미 있음 (--overwrite로 덮어쓰기)")
    excl = RQ.select_excluded(str(pre), a)
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
    tmp = dst.with_name(dst.stem + ".tmp.onnx")
    quantize_static(str(pre), str(tmp), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8 if a.act_type == "s8" else QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=bool(a.per_channel),
                    calibrate_method=RQ.METHODS[a.method], nodes_to_exclude=excl, extra_options=extra,
                    op_types_to_quantize=[o for o in a.op_types.split(",") if o] or None)
    # 보정기가 데이터를 읽은 회차마다 가져간 사진 수. 모든 회차가 요청 장수와 같아야 한다
    seen = reader.counts()
    if not (seen and all(c == a.calib_n for c in seen)):
        tmp.unlink()
        raise AssertionError(f"보정에 쓰인 사진 {seen} ≠ 요청 {a.calib_n}장 — 모델 파일을 지웠음")
    tmp.replace(dst)
    print(f"saved {dst.name}  {dst.stat().st_size / 2**20:.1f} MiB  ({time.time() - t0:.0f}s), calib seen per pass={seen}")

    env = env_info.collect()
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
           "dirty": env["git"]["dirty"], "options": vars(a), "model": dst.name,
           "MiB": round(dst.stat().st_size / 2**20, 2), "calib_seen": seen, "calib_pool_n": int(pool_n),
           "rn50_CAL_N": RS_CAL_N(), "n_excluded_nodes": len(excl),
           "qtype": f"{a.act_type.upper()}S8", "sec": round(time.time() - t0)}
    if a.dump_ops:
        row["ort_optimized_graph"] = dump_ops(dst, a.threads)
        print(json.dumps(row["ort_optimized_graph"], ensure_ascii=False))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
