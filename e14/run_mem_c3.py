"""run_mem_c3.py — B/32 O / F2 메모리 C0 vs C3 재측정 (E12-1 방법, 4T, n=50, 3회, 측정마다 새 프로세스)
반복 r 바깥, (모델 × 입력방식 × 조합) 안쪽. 판정 = 같은 반복끼리 C3 < C0 이 3회 모두인지 + 중앙값 차이.
"""
import json, subprocess, sys, os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "C3_메모리재측정" / "mem_c3.jsonl"
MODELS = {"O": r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx",
          "F2": r"C:\s2art_b32\e14\e14_2_F2_fastgelu_bias.onnx"}
MODES, COMBOS, REPS = ["held50", "one"], ["C0", "C3"], 3


def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def main():
    if len(sys.argv) > 1:
        MODELS["O"] = sys.argv[1]
    for p in MODELS.values():
        assert Path(p).exists(), p
    OUT.parent.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    import onnxruntime as ort
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    runs = {}
    for rep in range(REPS):
        for mk, mp in MODELS.items():
            for mode in MODES:
                for c in COMBOS:
                    p = subprocess.run([sys.executable, str(HERE / "mem_c3_child.py"), mp, c, "4", "50", "10", mode],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
                    assert p.returncode == 0, p.stderr[-2000:]
                    r = json.loads(p.stdout.strip().splitlines()[-1])
                    runs.setdefault((mk, mode, c), []).append(r)
                    row = {"type": "run", "date": datetime.now().isoformat(timespec="seconds"), "model": mk,
                           "file": Path(mp).name, "mode": mode, "combo": c, "rep": rep, "ort": ort.__version__, "git": git, **r}
                    with open(OUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"rep{rep} {mk} {mode} {c} peak={r['peak_MiB']} infer={r['infer_wset_max_MiB']} "
                          f"create={r['create_peak_MiB']} priv={r['infer_private_max_MiB']} base={r['base_MiB']}", flush=True)
    for mk in MODELS:
        for mode in MODES:
            c0, c3 = runs[(mk, mode, "C0")], runs[(mk, mode, "C3")]
            keys = [k for k in c0[0] if not k.startswith("t_")]
            m0 = {k: round(med([r[k] for r in c0]), 1) for k in keys}
            m3 = {k: round(med([r[k] for r in c3]), 1) for k in keys}
            lower = all(c3[i]["peak_MiB"] < c0[i]["peak_MiB"] for i in range(REPS))
            d = round(100 * (m3["peak_MiB"] - m0["peak_MiB"]) / m0["peak_MiB"], 2)
            row = {"type": "summary", "model": mk, "mode": mode, "C0": m0, "C3": m3, "peak_delta_pct": d,
                   "peak_lower_all_reps": lower, "ort": ort.__version__, "git": git}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("SUMMARY", mk, mode, "C0", m0["peak_MiB"], "C3", m3["peak_MiB"], f"{d}%", "all_lower", lower,
                  "| base", m0["base_MiB"], m3["base_MiB"], "| infer", m0["infer_wset_max_MiB"], m3["infer_wset_max_MiB"],
                  "| priv", m0["infer_private_max_MiB"], m3["infer_private_max_MiB"], flush=True)


if __name__ == "__main__":
    main()
