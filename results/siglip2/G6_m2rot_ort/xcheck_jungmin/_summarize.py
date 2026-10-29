"""지연 정리: speed_*.json(9개 = 3세션 × 1·2·4스레드, 시간순) → 세션별 평균(3회 반복 평균) → 3세션 중앙값. 정확도는 out_jungmin/result.json."""
import json
import statistics as st
from pathlib import Path

W = Path(r"D:\siglip2_xcheck")
S = W / "art" / "results" / "speed"
files = sorted(S.glob("speed_*.json"))
assert len(files) == 9, [f.name for f in files]
lat = {}
for i, f in enumerate(files):
    d = json.loads(f.read_text(encoding="utf-8"))
    t = d["cond"]["threads"] if "cond" in d else d.get("threads")
    for m, v in d["results"].items():
        lat.setdefault(m, {}).setdefault(t, []).append(v["ms_per_img_mean"])
    if i == 0:
        env = d.get("env", {})
        print("env:", json.dumps({k: env.get(k) for k in ("cpu", "vnni", "onnxruntime", "os", "power", "git")}, ensure_ascii=False))
sizes = {m: v["MiB"] for m, v in json.loads(files[0].read_text(encoding="utf-8"))["results"].items()}
med = {m: {t: st.median(v) for t, v in d.items()} for m, d in lat.items()}
print("\n세션별 평균 ms (s1, s2, s3)")
for m, d in lat.items():
    print(m, {t: [round(x, 1) for x in v] for t, v in sorted(d.items())})
print("\n중앙값 ms (FP32 대비 감소율)")
for m in med:
    row = []
    for t in (1, 2, 4):
        red = 100 * (1 - med[m][t] / med["fp32"][t])
        row.append(f"t{t} {med[m][t]:.1f} ({-red:+.1f}%)")
    print(f"{m:10s} {sizes[m]:6.1f} MiB  " + "  ".join(row))
print("\n① M2rot_ort < FP32 (모든 스레드):", all(med["M2rot_ort"][t] < med["fp32"][t] for t in (1, 2, 4)))
print("② M2rot_ort < R0 (스레드별):", {t: (round(med["M2rot_ort"][t], 1), round(med["R0_5eee"][t], 1), med["M2rot_ort"][t] < med["R0_5eee"][t]) for t in (1, 2, 4)})
print("② 세션별 M2rot < R0:", {t: [a < b for a, b in zip(lat["M2rot_ort"][t], lat["R0_5eee"][t])] for t in (1, 2, 4)})
acc = json.loads((W / "out_jungmin" / "result.json").read_text(encoding="utf-8"))
print("\n정확도 result.json keys:", list(acc)[:10])
json.dump({"latency_session_means_ms": lat, "latency_median_ms": med, "MiB": sizes}, open(S / "summary_jungmin.json", "w", encoding="utf-8"), indent=1)
