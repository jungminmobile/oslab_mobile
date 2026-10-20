"""g6_alls.py — D256L1 원칙(README 3-2)을 HAR의 층 이름으로 옮겨 .alls를 만든다 (hailo10_venv, 저장소 밖 폴더에서 실행)
규칙 (층 이름이 아니라 그래프 구조·원래 ONNX 노드 이름으로 정한다):
  - 패치 임베딩: original_names에 'patch_embed' 또는 'patch_embedding'이 있는 층 전부 (conv1과 그 입력 경로 층)
  - 블록 k fc1·fc2: original_names가 'mlp/fc1(_n)?/MatMul'·'mlp/fc2(_n)?/MatMul'인 conv를 HN 순서(위상 순서)로 센 k번째 (13번째는 풀링 헤드)
  - 잔차 덧셈: 입력 층 중 하나가 attention 출력 투영(…/proj/MatMul, attn_pool 제외는 아님)·fc2 conv인 ew_add + 위치 임베딩 덧셈
    (입력이 패치 임베딩 층인 ew_add). M 그래프의 마스크 연산 덧셈(K/V 바꿔 끼우기, u_c)은 입력이 proj·fc2가 아니라서 빠진다.
사용: python g6_alls.py <har> <out.alls> <out_map.csv> [--check <기존 .alls>]  (--check: 같은 규칙으로 만든 16비트 목록이 기존 .alls와 같은지)
"""
import csv
import re
import sys

from hailo_sdk_client import ClientRunner

har, out_alls, out_map = sys.argv[1:4]
check = sys.argv[sys.argv.index("--check") + 1] if "--check" in sys.argv else None
norm_line = "--no-norm" not in sys.argv
hn = ClientRunner(har=har).get_hn_dict()
L = hn["layers"]
short = lambda s: s.split("/", 1)[-1]
names = lambda n: ";".join(L[n].get("original_names", []))
order = list(L)
patch = [n for n in order if re.search(r"patch_embed|patch_embedding", names(n))]
# --input-layer X: 패치 conv 말고 Zoo .alls가 입력 경로로 16비트에 둔 층 이름 (B32 format_conversion1, S1B16 conv1_s2d — conv1_s2d는 optimize 때 생김)
inp = sys.argv[sys.argv.index("--input-layer") + 1] if "--input-layer" in sys.argv else None
patch_names = [short(n) for n in patch if L[n]["type"] == "conv"] + ([inp] if inp else [short(n) for n in patch if L[n]["type"] != "conv"])
fc1 = [n for n in order if L[n]["type"] == "conv" and re.search(r"mlp/fc1(_\d+)?/MatMul", names(n))]
fc2 = [n for n in order if L[n]["type"] == "conv" and re.search(r"mlp/fc2(_\d+)?/MatMul", names(n))]
proj = {n for n in order if L[n]["type"] == "conv" and re.search(r"(attn|attention)/(proj|out_proj)(_\d+)?/MatMul|/proj(_\d+)?/MatMul", names(n))
        and "patch" not in names(n)}
assert len(fc1) == 13 and len(fc2) == 13, (len(fc1), len(fc2))
inputs = lambda n: L[n].get("input", [])
res = [n for n in order if L[n]["type"] == "ew_add" and any(i in proj or i in set(fc2) for i in inputs(n))]
pos = [n for n in order if L[n]["type"] == "ew_add" and any(i in set(patch) for i in inputs(n))]
assert len(pos) == 1, [short(n) for n in pos]
resid = pos + res
assert len(resid) == 26, (len(resid), [short(n) for n in resid])
sel = {"patch": [n for n in patch if L[n]["type"] == "conv"], "b0_fc1": [fc1[0]], "b0_fc2": [fc2[0]], "b5_fc1": [fc1[5]], "b5_fc2": [fc2[5]], "b9_fc2": [fc2[9]]}
adds = [short(n) for n in resid]
lines = []
if norm_line:
    lines.append("norm1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])")
lines += ["model_optimization_flavor(optimization_level=1, compression_level=0)",
          "pre_quantization_optimization(layer_norm_decomposition, equalization=disabled, bit_decomposition_mode=uniform_precision)",
          "model_optimization_config(calibration, calibset_size=256)",
          f"quantization_param([{', '.join((['norm1'] if norm_line else []) + patch_names)}], precision_mode=a16_w16)",
          f"quantization_param([{', '.join(adds)}], precision_mode=a16_w16)",
          f"quantization_param([{', '.join(short(n) for k in ('b0_fc1', 'b0_fc2', 'b5_fc1', 'b5_fc2', 'b9_fc2') for n in sel[k])}], precision_mode=a16_w16)"]
open(out_alls, "w").write("\n".join(lines) + "\n")
with open(out_map, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["role", "dfc_layer", "onnx_nodes"])
    for k, v in sel.items():
        for n in v:
            w.writerow([k, short(n), names(n)[:200]])
    for i, n in enumerate(resid):
        w.writerow(["residual_add" if i else "pos_embed_add", short(n), names(n)[:200]])
print("patch", patch_names, "fc", {k: short(v[0]) for k, v in sel.items() if k != "patch"}, "resid", len(adds))
if check:
    txt = open(check).read()
    got = set(re.findall(r"[a-z_]+\d+(?:_s2d)?", ",".join(re.findall(r"quantization_param\(\[([^\]]*)\]", txt))))
    mine = set(adds) | {short(n) for k in ("b0_fc1", "b0_fc2", "b5_fc1", "b5_fc2", "b9_fc2") for n in sel[k]} | {"conv1"}
    print("check vs", check, "same" if got == mine else f"DIFF only_old={sorted(got - mine)} only_new={sorted(mine - got)}")
    assert got == mine
