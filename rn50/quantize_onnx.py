"""
quantize_onnx.py — ONNX Runtime 정적 INT8 양자화(PTQ)
=====================================================

[목적]
  FP32 ONNX 인코더를 INT8로 바꾼다. 다시 학습하지 않고(PTQ), 보정용 사진을 흘려보내
  각 층 값의 범위(최솟값·최댓값 등)만 재서 8비트 눈금(scale, zero-point)을 정한다.

[양자화 형식]  QDQ 형식 (각 연산 앞뒤에 Quantize/Dequantize 노드를 끼워 넣는 방식)
  - 가중치  : INT8, 대칭(0이 가운데), 채널별(도장마다 눈금 따로)  ← 정확도에 유리
  - 활성값  : UINT8(기본) 또는 INT8(--act-type s8), 비대칭, 텐서별(층 전체에 눈금 하나)
             E0(10/15)에 S8을 기본으로 바꿨다가 되돌렸다. 정확도는 S8≈U8(−0.14%p, 구간에 0 포함)이지만
             x86 VNNI CPU에서 S8 활성값 모델은 FP32보다 느려진다(용 측정: Core Ultra 9 285K 4스레드 c3 0.72배, U8은 1.55배).
             ARM(sdot)에서는 둘 다 빠르다
  - 범위 측정: minmax(기본) / percentile / entropy
  - 비-VNNI 대비(E0 ⑧): --reduce-range(가중치 7비트), --act-type u8 --weight-type u8(U8U8)
  - percentile·entropy는 배치 4개(32장)씩 나눠 히스토그램에 누적한다(--calib-stride, 메모리 부족 방지).
    ORT 1.25의 히스토그램 보정은 기본적으로 모든 층 출력을 한꺼번에 쌓아 256장이면 약 6GB 이상이 필요하다
  - percentile은 대칭 범위를 끄고(CalibTensorRangeSymmetric=False) 백분위 값(--percentile)을 명시한다.
    ORT 기본은 대칭(−t~t)이라 ReLU 뒤 활성값에서 눈금 절반을 버린다(PRD 3절, ORT 1.25.0 소스 확인)

[어느 노드를 양자화할지 고르는 3가지 방법] (함께 쓰면 '제외'가 우선)
  --exclude 부분문자열들  : 이름에 포함되면 FP32로 남김. --keep으로 예외를 다시 양자화
  --only    접두사들      : 이 접두사로 시작하는 노드만 양자화, 나머지는 FP32
  --only-exact 정확한이름 : 이 이름의 노드만 양자화 (--only와 합집합)
  예) 실험 D(AttnPool의 attention 계산만 FP32):
      --exclude attnpool --keep /attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm
  예) 새 기준 후보 ④(stem·layer1 FP32, layer2~AttnPool INT8):
      --only /layer2,/layer3,/layer4 --only-exact /attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm

[보정 데이터(누수 방지)]  — test 사진은 보정에 절대 쓰지 않는다
  --calib-source cifar   : CIFAR-100 train[0:45000]에서 무작위 N장
  --calib-source highres : 고화질 5종(pets/flowers/cars/food/caltech)의 train 분할 중 dev와 겹치지 않는
                           보정 후보(datasets.py dev가 만든 cache/calib_*_u8.npy)에서 N장
  --calib-source mix     : 위 둘을 반씩
  모든 보정 사진은 시드 고정으로 섞은 뒤 배치로 나눈다(출처·클래스가 한쪽에 몰린 배치 방지).

[수정 이력 — 2026-10-07 코드 검토]
  (치명) 예전 코드는 메모리를 아끼려고 CalibMaxIntermediateOutputs=3을 줬는데, ORT 1.25의 MinMaxCalibrater는
  N배치가 쌓이면 범위를 계산하지 않고 그냥 버린다(clear_collected_data). 그 결과 256장(32배치) 중
  마지막 2배치 = 16장만으로 눈금이 정해졌다. "No data is collected" 오류도 같은 원인
  (배치 수가 N의 배수면 마지막에 남는 것이 0개). → 옵션 삭제. MinMax의 중간 출력은 텐서마다
  min/max 값뿐이라 메모리 문제가 없다(실측으로 확인). 이전 MinMax 모델은 모두 다시 보정해야 한다.
  (중간) --tag 없이 돌리면 보정 출처·그래프가 달라도 같은 파일명으로 덮어쓰던 문제 → 이름에 모두 포함,
  같은 이름이 있으면 --overwrite 없이는 중단.

[주의]
  - quant_pre_process 결과(_pre.onnx)는 원본보다 오래된 경우에만 다시 만든다(예전 파일 재사용 방지).
  - 정확도 평가는 fake-quant 실행(common.session(fake_quant=True))으로, 속도는 정수 커널로 잰다.
"""
import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

import common as C

METHODS = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
           "percentile": CalibrationMethod.Percentile}
QT = {"s8": QuantType.QInt8, "u8": QuantType.QUInt8}
ATTN_PROJ = "/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm"  # q,k,v 투영 + c_proj


def calib_images(source, n, seed):
    """보정용 사진 n장 (n,3,224,224) float32. 평가(dev/test) 사진과 겹치지 않게 뽑고, 시드 고정으로 섞는다."""
    rng = np.random.default_rng(seed)
    parts = []
    n_cifar = {"cifar": n, "mix": n // 2, "highres": 0}[source]
    if n_cifar:
        ds, _ = C.cifar100(train=True)
        idx = rng.choice(C.CALIB_RANGE[1], n_cifar, replace=False)        # train[0:45000]에서만
        parts.append(np.stack([ds[int(i)][0].numpy() for i in idx]))
    n_hr = n - n_cifar
    if n_hr:
        files = sorted(C.CACHE_DIR.glob("calib_*_u8.npy"))
        assert files, "보정 후보 없음 — 먼저 `python datasets.py dev all`"
        pool = np.concatenate([np.load(f) for f in files])
        assert len(pool) >= n_hr, f"보정 후보 {len(pool)}장 < 요청 {n_hr}장"
        parts.append(C.to_input(pool[rng.choice(len(pool), n_hr, replace=False)]))
    x = np.concatenate(parts).astype(np.float32)
    return x[rng.permutation(len(x))]                                      # 출처·클래스 섞기


class ImageCalib(CalibrationDataReader):
    """ORT가 보정 때 get_next()를 반복 호출해 배치를 받아 간다."""
    def __init__(self, x, bs):
        self.batches = [x[s:s + bs] for s in range(0, len(x), bs)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"image": b}

    def rewind(self):
        self.it = iter(self.batches)

    # CalibStridedMinMax용: ORT가 len()으로 배치 수를 알고, set_range로 일부 배치씩 나눠 받아 간다
    def __len__(self):
        return len(self.batches)

    def set_range(self, start_index, end_index):
        self.it = iter(self.batches[start_index:end_index])


class TextCalib(CalibrationDataReader):
    """글 인코더용: ImageNet 클래스명 × ImageNet 템플릿 무작위 문장(평가 프롬프트와 겹치지 않음)."""
    def __init__(self, n, bs, seed):
        import open_clip
        rng = np.random.default_rng(seed)
        cls, tpl = open_clip.IMAGENET_CLASSNAMES, open_clip.OPENAI_IMAGENET_TEMPLATES
        texts = [tpl[rng.integers(len(tpl))](cls[rng.integers(len(cls))]) for _ in range(n)]
        tok = C.tokenizer()(texts).numpy().astype(np.int64)
        self.batches = [tok[s:s + bs] for s in range(0, n, bs)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"tokens": b}

    def rewind(self):
        self.it = iter(self.batches)


def select_excluded(pre, a):
    """--exclude/--keep/--only/--only-exact 규칙으로 'FP32로 남길 노드 이름 목록'을 만든다."""
    import onnx
    if not (a.exclude or a.only or a.only_exact or a.exclude_exact):
        return []
    nodes = [n.name for n in onnx.load(pre, load_external_data=False).graph.node]
    excl = set()
    if a.exclude:
        pats = [p for p in a.exclude.split(",") if p]
        keep = {k for k in a.keep.split(",") if k}
        excl |= {n for n in nodes if any(p in n for p in pats) and n not in keep}
    if a.only or a.only_exact:
        pre_fx = tuple(p for p in a.only.split(",") if p)
        exact = {p for p in a.only_exact.split(",") if p}
        missing = exact - set(nodes)
        assert not missing, f"--only-exact에 없는 노드 이름: {missing}"   # 오타로 조용히 실패하는 것 방지
        incl = {n for n in nodes if (pre_fx and n.startswith(pre_fx)) or n in exact}
        assert incl, "--only 조건에 맞는 노드가 하나도 없음"
        excl |= set(nodes) - incl
    if a.exclude_exact:   # 정확한 이름으로 FP32 유지 (부분 문자열 --exclude는 /conv1/Conv가 블록 안 conv1까지 잡아서 따로 둠)
        ee = {p for p in a.exclude_exact.split(",") if p}
        missing = ee - set(nodes)
        assert not missing, f"--exclude-exact에 없는 노드 이름: {missing}"
        excl |= ee
    print(f"quantize {len(nodes) - len(excl)} / {len(nodes)} nodes (FP32 유지 {len(excl)})")
    return sorted(excl)


def w4_overrides(pre, prefixes, exact):
    """가중치 4비트(활성값은 8비트 유지 = Hailo의 a8_w4와 같은 방식) 대상 노드의 가중치 이름을 찾아
    TensorQuantOverrides 형식으로 돌려준다. 4비트는 눈금이 16칸뿐이라 채널별(per-channel)·대칭으로 둔다."""
    import onnx
    from onnxruntime.quantization import QuantType
    g = onnx.load(pre, load_external_data=False).graph
    inits = {i.name for i in g.initializer}
    pre_fx = tuple(p for p in prefixes.split(",") if p)
    ex = {p for p in exact.split(",") if p}
    ov = {}
    for n in g.node:
        if n.op_type in ("Conv", "MatMul", "Gemm") and ((pre_fx and n.name.startswith(pre_fx)) or n.name in ex):
            w = n.input[1]
            if w in inits:
                # 채널 축: Conv 가중치 (out,in,kh,kw) → 0 / MatMul 가중치 (in,out) → 1 / Gemm(transB=1) (out,in) → 0
                # (축을 안 주면 텐서 전체에 눈금 하나 → 4비트에서는 모델이 망가짐. 검증: 유사도 0.12)
                trans_b = any(at.name == "transB" and at.i == 1 for at in n.attribute)
                axis = 1 if n.op_type == "MatMul" or (n.op_type == "Gemm" and not trans_b) else 0
                ov[w] = [{"quant_type": QuantType.QInt4, "symmetric": True, "axis": axis}]
    assert ov, "--w4 조건에 맞는 가중치가 없음"
    print(f"4비트 가중치 {len(ov)}개")
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", choices=["image", "text"], required=True)
    ap.add_argument("--mode", choices=["static", "dynamic"], default="static")
    ap.add_argument("--variant", default="", help="FP32 원본 접미사. 예: _q1 → rn50_image_fp32_q1.onnx (AttnPool 최적화본)")
    ap.add_argument("--calib-source", choices=["cifar", "highres", "mix"], default="cifar")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--method", choices=list(METHODS), default="minmax")
    ap.add_argument("--moving-avg", type=int, default=0, help="minmax 전용: 배치별 범위를 이동평균")
    ap.add_argument("--per-channel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--exclude-exact", default="", help="이 이름의 노드만 정확히 FP32로 남김(쉼표). E4-C conv 단위 구성용")
    ap.add_argument("--only", default="")
    ap.add_argument("--only-exact", default="")
    ap.add_argument("--tag", default="", help="파일 이름 뒤에 붙일 설명(노드 선택 이름 등). 보정 조건은 자동으로 붙음")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--w4", default="", help="가중치를 4비트로 할 노드 접두사(쉼표). 활성값은 8비트 유지")
    ap.add_argument("--w4-exact", default="", help="가중치를 4비트로 할 노드 정확한 이름(쉼표)")
    ap.add_argument("--act-type", choices=["s8", "u8"], default="u8",
                    help="활성값 형식. 기본 u8: x86(VNNI)에서 s8 활성값은 FP32보다도 느려진다(용 측정 10/15, 4스레드 0.72배)")
    ap.add_argument("--weight-type", choices=["s8", "u8"], default="s8", help="가중치 형식 (u8은 U8U8 대비책용)")
    ap.add_argument("--reduce-range", action="store_true", help="가중치 7비트 범위(비-VNNI CPU 포화 대비)")
    ap.add_argument("--percentile", type=float, default=99.999, help="percentile 방법의 백분위 값")
    ap.add_argument("--calib-stride", type=int, default=4,
                    help="percentile·entropy 전용: 배치 몇 개씩 나눠 히스토그램에 누적할지(메모리 절약). "
                         "ORT 히스토그램 보정은 모든 층 출력을 한꺼번에 메모리에 쌓아서 256장이면 8GB에서 메모리 부족")
    ap.add_argument("--w4nbits", default="",
                    help="(E11 R3, 기본 꺼짐) 이 이름의 MatMul(쉼표, 정확한 이름) 가중치를 ORT MatMulNBits 4비트 블록 양자화로 "
                         "바꾼 뒤 나머지를 평소처럼 INT8 QDQ로 양자화한다 (SigLIP2 s2_quantize --w4-regex와 같은 방식)")
    ap.add_argument("--w4nbits-block", type=int, default=32)
    ap.add_argument("--w4nbits-acc", type=int, default=4, help="MatMulNBits accuracy_level (4 = 활성값 int8 동적 양자화)")
    a = ap.parse_args()

    src = C.ONNX_DIR / f"{C.PREFIX}_{a.enc}_fp32{a.variant}.onnx"
    pre = C.ONNX_DIR / f"{C.PREFIX}_{a.enc}_fp32{a.variant}_pre.onnx"
    if not pre.exists() or pre.stat().st_mtime < src.stat().st_mtime:
        # 윈도우에서 경로에 한글·공백이 있으면 ONNX 형태 추론(infer_shapes_path)이 파일을 못 찾는다
        # → 영문 임시 폴더에서 처리한 뒤 옮긴다(리눅스에서도 결과는 같음)
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(src, Path(td) / "src.onnx")
            quant_pre_process(str(Path(td) / "src.onnx"), str(Path(td) / "pre.onnx"),
                              skip_symbolic_shape=a.enc == "text")  # 형태 추론 + 그래프 정리
            shutil.move(str(Path(td) / "pre.onnx"), str(pre))
    # 파일 이름에 그래프·보정 조건을 모두 넣어 서로 다른 설정이 같은 이름으로 덮어써지지 않게 한다
    meth = a.method + (f"{a.percentile:g}".replace(".", "") if a.method == "percentile" else "")   # percentile99999
    cal = f"{meth}{'-ma' if a.moving_avg else ''}_{a.calib_source}{a.calib_n}_s{a.seed}" if a.mode == "static" else "dynamic"
    # 형식 꼬리표: 기본(U8S8)은 없음. S8S8 → _s8s8, U8U8 → _u8u8, S8U8 → _s8u8, reduce_range → _rr
    fmt = "" if (a.act_type, a.weight_type) == ("u8", "s8") else f"_{a.act_type}{a.weight_type}"
    fmt += "_rr" if a.reduce_range else ""
    dst = C.ONNX_DIR / f"{C.PREFIX}_{a.enc}_int8{a.variant}_{cal}{'_' + a.tag if a.tag else ''}{fmt}.onnx"
    if dst.exists() and not a.overwrite:
        raise SystemExit(f"{dst.name} 이미 있음 (--overwrite로 덮어쓰기)")
    excl = select_excluded(str(pre), a)

    t0 = time.time()
    if a.mode == "dynamic":   # 활성값 범위를 실행 때마다 재는 방식(NPU에서는 못 씀, 참고용)
        # (E11) 한글 경로에서 ORT가 중간 파일(_pre-inferred)을 못 찾으므로 정적과 같이 영문 임시 폴더에서 처리한다
        td = tempfile.mkdtemp()
        shutil.copy(pre, Path(td) / "pre.onnx")
        quantize_dynamic(str(Path(td) / "pre.onnx"), str(Path(td) / "dst.onnx"), weight_type=QT[a.weight_type],
                         per_channel=bool(a.per_channel), reduce_range=a.reduce_range, nodes_to_exclude=excl)
        shutil.move(str(Path(td) / "dst.onnx"), str(dst))
        shutil.rmtree(td, ignore_errors=True)
    else:
        reader = (ImageCalib(calib_images(a.calib_source, a.calib_n, a.seed), a.calib_bs) if a.enc == "image"
                  else TextCalib(a.calib_n, a.calib_bs, a.seed))
        td = tempfile.mkdtemp()                       # 같은 이유로 영문 임시 폴더에서 양자화 후 옮긴다
        tmp_pre, tmp_dst = Path(td) / "pre.onnx", Path(td) / "dst.onnx"
        shutil.copy(pre, tmp_pre)
        if a.w4nbits:   # 옵션을 끄면 이 블록을 지나지 않아 기존 결과와 같다
            import onnx
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            want = [s for s in a.w4nbits.split(",") if s]
            m0 = onnx.load(str(tmp_pre))
            inits = {i.name for i in m0.graph.initializer}
            w_all = [n.name for n in m0.graph.node if n.op_type == "MatMul" and n.input[1] in inits]
            assert set(want) <= set(w_all), f"가중치 MatMul이 아님: {set(want) - set(w_all)}"
            # ORT 1.25: nodes_to_include는 '추가 포함'이라 고르지 않은 가중치 MatMul은 nodes_to_exclude로 뺀다 (SigLIP2 10/16 기록)
            q4 = MatMulNBitsQuantizer(m0, bits=4, block_size=a.w4nbits_block, is_symmetric=True,
                                      accuracy_level=a.w4nbits_acc, nodes_to_include=want,
                                      nodes_to_exclude=[n for n in w_all if n not in set(want)])
            q4.process()
            q4.model.save_model_to_file(str(tmp_pre), use_external_data_format=False)
            n4 = sum(n.op_type == "MatMulNBits" for n in onnx.load(str(tmp_pre)).graph.node)
            assert n4 == len(want), f"MatMulNBits {n4}개 ≠ 요청 {len(want)}개"
            print(f"--w4nbits: MatMul {n4}개를 4비트로 (블록 {a.w4nbits_block}, accuracy_level {a.w4nbits_acc})")
        quantize_static(str(tmp_pre), str(tmp_dst), reader, quant_format=QuantFormat.QDQ,
                        activation_type=QT[a.act_type], weight_type=QT[a.weight_type],
                        per_channel=bool(a.per_channel), reduce_range=a.reduce_range,
                        calibrate_method=METHODS[a.method],
                        nodes_to_exclude=excl,
                        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                       "CalibMovingAverage": bool(a.moving_avg),
                                       **({"CalibTensorRangeSymmetric": False, "CalibPercentile": a.percentile}
                                          if a.method == "percentile" else {}),
                                       **({"CalibStridedMinMax": a.calib_stride}
                                          if a.method in ("percentile", "entropy") and a.enc == "image" else {}),
                                       **({"TensorQuantOverrides": w4_overrides(str(pre), a.w4, a.w4_exact),
                                           "UseQDQContribOps": True}   # 낮은 opset에서도 int4 Q/DQ 사용
                                          if (a.w4 or a.w4_exact) else {})})
        shutil.move(str(tmp_dst), str(dst))
        shutil.rmtree(td, ignore_errors=True)
    print(f"saved {dst}  {dst.stat().st_size / 2**20:.1f} MiB  ({time.time() - t0:.0f}s)")
    C.append_jsonl("quantize", {"file": dst.name, "MiB": round(dst.stat().st_size / 2**20, 1),
                                "sec": round(time.time() - t0), "opts": vars(a), "meta": C.run_meta()})


if __name__ == "__main__":
    main()
