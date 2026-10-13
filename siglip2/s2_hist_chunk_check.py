"""
s2_hist_chunk_check.py — 히스토그램 보정 "나눠서 누적"(--hist-chunk)이 ORT 표준(한꺼번에)과 같은지 확인
==================================================================================================

[목적]  (쭈리 승인 10/13: "나눠서 누적 + 동등성 확인")
  ORT 1.25 Percentile·Entropy 보정기는 모든 배치의 중간 텐서를 쌓은 뒤 히스토그램을 만들어 ViT·256장에서 메모리가 부족하다.
  s2_quantize.py --hist-chunk는 ORT의 HistogramCollector 합치기(merge_histogram)로 나눠서 누적한다.
  합칠 때 범위가 커지면 구간을 다시 나누므로(rebin) 결과가 완전히 같다는 보장이 없다 → 메모리에 들어가는 작은 장수에서 직접 비교한다.

[방법]  MatMul만 INT8 구성, 보정 N장(기본 32), 방법 percentile(99.999)·entropy 각각
  A: ORT 표준 (--hist-chunk 0)   B: 나눠서 누적 (--hist-chunk C, 기본 8 → 4번 누적)
  비교 ① 모든 활성값 QuantizeLinear의 scale 상대 차이(최대·중앙값·1% 넘는 텐서 수), zero point 다른 개수
       ② s2_screen과 같은 스크리닝(dev 각 앞 K장, 정수 커널)의 고화질 cos·top-1
  판단 규칙(실행 전 고정): ② 고화질 cos 차이 ≤ 0.002 이고 ① scale 중앙값 상대 차이 ≤ 1%면 "동등"으로 보고 256장은 나눠서 진행.
  아니면 결과와 함께 사람에게 보고한다.

[사용]
  RN50_RESULTS=../results/siglip2/G3_calib_method/hist_chunk_check "$SIGLIP2_PYTHON" s2_hist_chunk_check.py
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
import s2_screen as SS        # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC


def act_scales(path):
    """활성값 QuantizeLinear 노드: 입력 텐서 이름 → (scale, zero_point)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits:
            out[n.input[0]] = (float(inits[n.input[1]]), int(inits[n.input[2]]) if len(n.input) > 2 else 0)
    return out


def quantize(method, n, chunk, tag, out):
    args = [sys.executable, "-u", "s2_quantize.py", "--calib-source", "highres", "--calib-n", str(n), "--op-types",
            "MatMul", "--method", method, "--tag", tag, "--overwrite", "--log", str(out / "quantize_log.jsonl")]
    if method == "percentile":
        args += ["--percentile", "99.999"]
    if chunk:
        args += ["--hist-chunk", str(chunk)]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-3000:]
    name = [l for l in r.stdout.splitlines() if l.startswith("saved ")][-1].split()[1]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    rows = []
    for method in ("percentile", "entropy"):
        fa = quantize(method, a.n, 0, "hcchk", out)
        fb = quantize(method, a.n, a.chunk, "hcchk", out)
        sa, sb = act_scales(RC.onnx_path(fa)), act_scales(RC.onnx_path(fb))
        assert sa.keys() == sb.keys() and sa, "두 모델의 양자화 텐서 목록이 다름"
        rel = np.array([abs(sb[k][0] - sa[k][0]) / max(sa[k][0], 1e-12) for k in sa])
        zp_diff = sum(sa[k][1] != sb[k][1] for k in sa)
        ra, rb = SS.screen(fa, a.k, a.threads), SS.screen(fb, a.k, a.threads)
        row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
               "dirty": env["git"]["dirty"], "method": method, "calib_n": a.n, "chunk": a.chunk, "k": a.k,
               "model_standard": fa, "model_chunked": fb, "n_act_tensors": len(sa),
               "scale_rel_diff": {"max": round(float(rel.max()), 5), "median": round(float(np.median(rel)), 5),
                                  "n_over_1pct": int((rel > 0.01).sum())}, "zero_point_diff_count": int(zp_diff),
               "standard": {"hr_cos": ra["hr_cos"], "hr_dtop1": ra["hr_dtop1"]},
               "chunked": {"hr_cos": rb["hr_cos"], "hr_dtop1": rb["hr_dtop1"]}}
        row["hr_cos_diff"] = round(rb["hr_cos"] - ra["hr_cos"], 4)
        row["equivalent"] = bool(abs(row["hr_cos_diff"]) <= 0.002 and row["scale_rel_diff"]["median"] <= 0.01)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
        with open(out / "hist_chunk_check.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("ALL EQUIVALENT" if all(r["equivalent"] for r in rows) else "NOT EQUIVALENT — 사람에게 보고")


if __name__ == "__main__":
    main()
