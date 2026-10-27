"""p2_vs_oab.py — test 사진별 예측으로 P2 − O-AB 짝 차이 (참고, 판정 아님; test_p2/README 설계대로). 같은 사진·같은 정답·같은 FP32 예측인지 먼저 확인.
층화 짝 부트스트랩 1,000회, 시드 0 (s2_paired_ci.boot_hr와 같은 방식). 사용: "$SIGLIP2_PYTHON" results/siglip2/G8_candidates/b32/test_p2/p2_vs_oab.py"""
import json
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
OAB = "siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all"
P2 = "siglip2_image_int8_g6b_b32_m2rot_o1b_e14P2_fastgelu_patchint8"
HR = ["pets", "flowers", "cars", "food", "caltech"]
res, diff = {}, {}
for d in HR + ["cifar_test", "imagenetv2"]:
    a, b = np.load(H.parent / "test" / f"test_{d}.npz"), np.load(H / f"test_{d}.npz")
    assert np.array_equal(a["y"], b["y"]) and np.array_equal(a["fp32_pred"], b["fp32_pred"]), d
    ca, cb = a[OAB + "_pred"] == a["y"], b[P2 + "_pred"] == b["y"]
    diff[d] = cb.astype(float) - ca
    res[d] = {"n": int(len(ca)), "oab": round(100 * ca.mean(), 2), "p2": round(100 * cb.mean(), 2), "p2_minus_oab": round(100 * (cb.mean() - ca.mean()), 2),
              "pred_diff": int((a[OAB + "_pred"] != b[P2 + "_pred"]).sum()), "fp32_same": True}
rng = np.random.default_rng(0)
def boot(sets):
    v = [np.mean([100 * diff[s][rng.integers(0, len(diff[s]), len(diff[s]))].mean() for s in sets]) for _ in range(1000)]
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
res["hr"] = {"p2_minus_oab": round(float(np.mean([100 * diff[s].mean() for s in HR])), 2), "ci95": boot(HR)}
for s in ("cifar_test", "imagenetv2"):
    res[s]["ci95"] = boot([s])
json.dump(res, open(H / "p2_vs_oab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in res.items(): print(k, v)
