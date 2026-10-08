"""
summarize.py — evaluate.jsonl을 표(마크다운)로 요약
==================================================
[목적] 모델마다 dev 데이터셋별 top-1, FP32 대비 변화, 평균 유사도, 용량을 한 표로 본다.
[사용] python summarize.py --results results --filter c4 --exec int
  · 고화질 평균(HR avg)은 pets/flowers/cars/food/caltech dev의 평균 (CIFAR는 따로 표시)
"""
import argparse
import json
from pathlib import Path

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--filter", default="")
    ap.add_argument("--exec", default="int")
    ap.add_argument("--strip", default="_image_int8_q1_minmax_")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.results) / "evaluate.jsonl")]
    base = {r["data"]: r for r in rows if r["model"] == "fp32"}
    tab = {}
    for r in rows:
        if r["model"] == "fp32" or r.get("exec") != a.exec or a.filter not in r["model"]:
            continue
        tab.setdefault(r["model"], {})[r["data"]] = r
    hdr = ["model", "MiB"] + [d.replace("_dev", "") for d in DATA] + ["HR Δtop1", "HR cos", "CIFAR cos"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    print("| fp32 | — | " + " | ".join(f"{base[d]['top1']:.1f}" if d in base else "-" for d in DATA) + " | 0 | 1.000 | 1.000 |")
    for m, d in tab.items():
        name = m.split(a.strip)[-1].replace(".onnx", "") if a.strip in m else m
        dl = [d[x]["top1"] - base[x]["top1"] for x in HR if x in d and x in base]
        cs = [d[x]["cos_mean"] for x in HR if x in d]
        mib = next(iter(d.values())).get("MiB")
        print(f"| {name} | {mib} | " + " | ".join(f"{d[x]['top1']:.1f}" if x in d else "-" for x in DATA)
              + f" | {sum(dl)/len(dl):+.2f} | {sum(cs)/len(cs):.3f} | "
              + (f"{d['cifar_A']['cos_mean']:.3f}" if "cifar_A" in d else "-") + " |" if dl else " |")


if __name__ == "__main__":
    main()
