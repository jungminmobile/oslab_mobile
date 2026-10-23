"""b32t_check_data.py — README B-2 사진 대조: data_list.jsonl(이 PC) vs 준혁 test.jsonl FP32 줄 (g8-candidates)과 ImageNetV2 tar sha256.
전부 같으면 exit 0, 하나라도 다르면 exit 3 (실칩 전에 멈춤). 결과 data_check.json.
[사용] python b32t_check_data.py --ref <준혁 test.jsonl> --log <B_test 폴더>"""
import argparse
import json
from pathlib import Path

IMNV2_SHA = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"   # results/siglip2/G6_test/data_sha256.txt
SETS = ["pets", "caltech", "flowers", "cars", "food", "cifar_test", "imagenetv2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    ref = {r["data"]: r for r in map(json.loads, open(a.ref, encoding="utf-8")) if r["model"] == "fp32"}
    mine = {r["set"]: r for r in map(json.loads, open(Path(a.log) / "data_list.jsonl", encoding="utf-8"))}
    rows, ok = {}, True
    for s in SETS:
        m, r = mine.get(s), ref.get(s)
        if m is None or r is None:
            rows[s] = {"ok": False, "why": "없음"}
            ok = False
            continue
        c = {"n": (m["n"], r["n"]), "classes": (m["classes"], r["classes"]), "n_templates": (m["n_templates"], r["n_templates"]),
             "fp32_top1": (m["fp32_onnx"]["top1"], r["top1"]), "fp32_top5": (m["fp32_onnx"]["top5"], r["top5"])}
        if s == "imagenetv2":
            c["tar_sha256"] = (m["src_sha256"].get("imagenetv2-mf.tar"), IMNV2_SHA)
        good = all(x == y for x, y in c.values())
        rows[s] = {"ok": good, **{k: {"mine": x, "junhyuk": y} for k, (x, y) in c.items()}}
        ok &= good
    res = {"all_ok": ok, "sets": rows, "ref": a.ref}
    (Path(a.log) / "data_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
