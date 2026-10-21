"""g6e_bitexact_cmp.py — 4단계 판정: SDK_BIT_EXACT 출력 vs 저장된 실칩 출력 vs SDK_QUANTIZED 출력 (사진별 cos, README 4단계 규칙)
재현됨     : 모든 사진에서 cos(BE_both, 실칩_both) ≥ 0.999 그리고 cos(BE_rb, 실칩_rb) ≥ 0.999
재현 안 됨 : 모든 사진에서 cos(BE_both, QUANT_both) ≥ 0.999
둘 다 아니면 '판정 불가'로 기록한다.
실칩: B16_both_bk(G6c eval 전체), B16_rb_bk(G6c eval200, M2rot2와 비트 동일). QUANT: G6c g6c_emulate.py 출력(앞 200장).
사용 (siglip2 venv): python g6e_bitexact_cmp.py --be-both be8_b16_both.npz --be-rb be8_b16_rb.npz --g6 ~/siglip2_work/g6 --out bitexact.jsonl
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

TH = 0.999


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--be-both", required=True); ap.add_argument("--be-rb", required=True)
    ap.add_argument("--g6", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = Path(a.g6)
    chip = {"both": np.load(G / "g6c_chip/eval_b16_both_bk/emb_b16_both_bk.npz"), "rb": np.load(G / "g6c_chip/eval200_b16_rb_bk/emb_b16_rb_bk.npz")}
    quant = {"both": np.load(G / "g6c_emu/emu_b16_both.npz"), "rb": np.load(G / "g6c_emu/emu_b16_rb.npz")}
    be = {"both": np.load(a.be_both), "rb": np.load(a.be_rb)}
    tags = [str(t) for t in be["both"]["tags"]]
    assert tags == [str(t) for t in be["rb"]["tags"]]
    rows = []
    for i, t in enumerate(tags):
        d, j = t.split(":"); j = int(j)
        r = {"pick": t}
        for m in ("both", "rb"):
            b, c, q = be[m]["emb"][i], chip[m][f"{d}_emb"][j], quant[m][f"{d}_emb"][j]
            r.update({f"{m}_be_chip": round(cos(b, c), 5), f"{m}_be_quant": round(cos(b, q), 5), f"{m}_quant_chip": round(cos(q, c), 5),
                      f"{m}_be_eq_chip_bit": bool(np.array_equal(b, c)), f"{m}_be_sec": round(float(be[m]["sec"][i]), 1)})
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    repro = all(r["both_be_chip"] >= TH and r["rb_be_chip"] >= TH for r in rows)
    norepro = all(r["both_be_quant"] >= TH for r in rows)
    verdict = "재현됨" if repro else ("재현 안 됨" if norepro else "판정 불가")
    git = lambda *c: subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), *c], capture_output=True, text=True).stdout.strip()
    out = {"date": datetime.now().isoformat(timespec="seconds"), "what": "g6e_bitexact", "commit": git("rev-parse", "--short", "HEAD"),
           "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "th": TH, "verdict": verdict, "n": len(rows),
           "summary": {k: [round(min(r[k] for r in rows), 5), round(max(r[k] for r in rows), 5)] for k in rows[0] if k.endswith(("_chip", "_quant"))},
           "rows": rows}
    print(f"== 판정: {verdict}  " + json.dumps(out["summary"], ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
