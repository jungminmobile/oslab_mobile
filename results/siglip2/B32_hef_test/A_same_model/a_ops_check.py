"""a_ops_check.py — (독립 검토 반영, 사후 보강) A-2 보충: 두 ONNX의 opset과, 모양 계산 op를 뺀 노드의 (op, 속성) 다중집합을 비교한다.
모양 계산 op = Shape·Gather·Unsqueeze·Concat·Reshape·Constant·ConstantOfShape·Equal·Where·Expand·Identity (배치 가변 export가 늘리는 op).
출력: a_ops_check.json. [사용] python a_ops_check.py --b1 <b1.onnx> --dyn <dyn.onnx> --out a_ops_check.json"""
import argparse
import json
from collections import Counter

import onnx
from onnx import helper

SHAPE_OPS = {"Shape", "Gather", "Unsqueeze", "Concat", "Reshape", "Constant", "ConstantOfShape", "Equal", "Where", "Expand", "Identity"}


def sig(n):
    return n.op_type + "|" + json.dumps(sorted((a.name, str(helper.get_attribute_value(a))) for a in n.attribute))


def main():
    ap = argparse.ArgumentParser()
    for k in ("b1", "dyn", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    m1, m2 = onnx.load(a.b1, load_external_data=False), onnx.load(a.dyn, load_external_data=False)
    c1 = Counter(sig(n) for n in m1.graph.node if n.op_type not in SHAPE_OPS)
    c2 = Counter(sig(n) for n in m2.graph.node if n.op_type not in SHAPE_OPS)
    o1, o2 = Counter(n.op_type for n in m1.graph.node), Counter(n.op_type for n in m2.graph.node)
    res = {"opset_b1": [(o.domain, o.version) for o in m1.opset_import], "opset_dyn": [(o.domain, o.version) for o in m2.opset_import],
           "compute_nodes_b1": sum(c1.values()), "compute_nodes_dyn": sum(c2.values()),
           "compute_op_attr_multiset_equal": c1 == c2, "only_b1": dict(c1 - c2), "only_dyn": dict(c2 - c1),
           "op_count_diff": {k: [o1.get(k, 0), o2.get(k, 0)] for k in sorted(set(o1) | set(o2)) if o1.get(k, 0) != o2.get(k, 0)}}
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
