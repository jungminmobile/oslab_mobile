"""
make_stem_combos.py — stem 내부 양자화 조합 15개 모델 만들기 (실험 2-2)
=====================================================================

[목적]
  stem(conv1~3 + avgpool) 안에서 어느 부분의 8비트화가 정확도에 영향을 주는지 나눠 보기 위해,
  "stem 전체 INT8" 모델에서 고르지 않은 묶음의 Q/DQ만 제거(FP32로 되돌림)한 모델을 만든다.
  재보정을 하지 않으므로 15개 모델 모두 완전히 같은 눈금(scale)을 쓰고, 차이는 "어디를 켰나"뿐이다.

[묶음]
  a: 입력 사진(image) 활성값
  b: conv1 가중치·편향 + act1 출력
  c: conv2 가중치·편향 + act2 출력
  d: conv3 가중치·편향 + act3 출력 + avgpool 출력

[중요 함정 — 검토 중 발견해 고친 버그]
  ORT QDQ 양자화는 Conv 뒤의 Relu 노드를 지운다(UINT8, zero-point=0 클램프가 ReLU 역할을 대신함).
  그래서 활성값 Q/DQ만 빼면 ReLU가 사라져 결과가 망가진다(정확도 1%대). Q/DQ를 뺄 때 Relu를 다시 넣는다.
  검증: 모든 묶음을 끄면 FP32와 출력 차이 0.0, abcd는 기존 stem 전체 모델과 같은 정확도.

[입력]  --src <stem만 INT8 모델>  (quantize_onnx.py --variant _q1 --only /conv1,... --tag only-stem 으로 생성)
        onnx/rn50_image_fp32_q1_pre.onnx        (FP32 가중치 복원용)
[출력]  onnx/<src 이름에서 _only-stem을 뺀 것>_sc_<조합>.onnx  (a, b, ..., abcd)
"""
import argparse
import itertools

import onnx
from onnx import numpy_helper

import common as C

SRC = None                                                  # main()에서 --src로 지정
FP32 = C.ONNX_DIR / f"{C.PREFIX}_image_fp32_q1_pre.onnx"
ACT = {"a": ["image"], "b": ["/act1/Relu_output_0"], "c": ["/act2/Relu_output_0"],
       "d": ["/act3/Relu_output_0", "/avgpool/AveragePool_output_0"]}
CONV_OUT = {"b": "/act1/Relu_output_0", "c": "/act2/Relu_output_0", "d": "/act3/Relu_output_0"}


def rewire(g, old, new):
    """old 텐서를 입력으로 쓰던 모든 노드가 new 텐서를 쓰도록 바꾼다."""
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == old:
                n.input[i] = new


def drop_act_qdq(g, t):
    """활성값 t에 붙은 Quantize→Dequantize 쌍을 제거하고, 필요하면 지워졌던 Relu를 복원."""
    q = next(n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == t)
    dq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == q.output[0])
    out = dq.output[0]
    g.node.remove(q)
    g.node.remove(dq)
    rewire(g, out, t)
    prod = [i for i, n in enumerate(g.node) if t in n.output]
    if prod and g.node[prod[0]].op_type == "Conv" and "Relu" in t:
        conv = g.node[prod[0]]
        conv.output[0] = t + "_pre_relu"
        g.node.insert(prod[0] + 1, onnx.helper.make_node("Relu", [t + "_pre_relu"], [t], name=t + "_restored_relu"))


def drop_weight_qdq(g, conv_out, fp32_inits):
    """conv의 가중치·편향 Dequantize를 지우고 FP32 원본 값을 다시 넣는다."""
    conv = next(n for n in g.node if n.op_type == "Conv" and n.output[0] in (conv_out, conv_out + "_pre_relu"))
    for k in (1, 2):  # 1=가중치, 2=편향
        dq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.output[0] == conv.input[k])
        orig = dq.input[0].removesuffix("_quantized")
        g.node.remove(dq)
        g.initializer.append(numpy_helper.from_array(fp32_inits[orig], orig))
        conv.input[k] = orig


def prune_inits(g):
    """더 이상 쓰이지 않는 상수(눈금·양자화 가중치)를 지워 파일을 깔끔하게."""
    used = {x for n in g.node for x in n.input}
    for init in list(g.initializer):
        if init.name not in used:
            g.initializer.remove(init)


def build(on, fp32_inits):
    m = onnx.load(str(SRC))
    for x in "abcd":
        if x in on:
            continue
        for t in ACT[x]:
            drop_act_qdq(m.graph, t)
        if x in CONV_OUT:
            drop_weight_qdq(m.graph, CONV_OUT[x], fp32_inits)
    prune_inits(m.graph)
    onnx.checker.check_model(m)
    return m


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="stem만 INT8인 모델 (quantize_onnx.py ... --tag only-stem)")
    a = ap.parse_args()
    SRC = C.onnx_path(a.src)
    stem_tag = SRC.stem.replace("_only-stem", "")
    fp32_inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(FP32)).graph.initializer}
    for r in range(1, 5):
        for combo in itertools.combinations("abcd", r):
            dst = C.ONNX_DIR / f"{stem_tag}_sc_{''.join(combo)}.onnx"
            onnx.save(build(set(combo), fp32_inits), str(dst))
            print("saved", dst.name)


if __name__ == "__main__":
    main()
