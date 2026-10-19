"""
e10_eval.py — E10 3단계: ilp_plan.json의 고유 구성을 만들고 B2와 같은 조건으로 측정
사전 등록: results/E10_ilp/PREREG.md (0e99334)

  build : 구성별 양자화 (e10_cost.quantize, --exclude-exact = 선택 단위 노드 합집합)
  acc   : dev 전체(CIFAR A + 고화질 5종) 정수 커널 4T --save-pred, B2 대비 짝 부트스트랩 (bootstrap.py)
  speed : 3세션 × 1·2·4T, 한 실행에서 FP32·B2·구성들을 번갈아 (n=200, warmup 20, repeats 3)
  mem   : bench_mem 4T
"""
import json
import subprocess
import sys
import time

import common as C
import e10_cost as E

OUT = C.RESULTS_DIR
HR = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
B2 = "rn50_image_int8_q1_percentile99999_highres256_s0_B2.onnx"
FP32 = "rn50_image_fp32_q1.onnx"


def run(cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=C.HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-1500:]
    return r.stdout


def configs():
    plan = json.loads((OUT / "ilp_plan.json").read_text(encoding="utf-8"))
    U = json.loads((OUT / "units_rn50.json").read_text(encoding="utf-8"))
    out = {}
    for cid, units in plan["configs"].items():
        nodes = [n for u in units for n in U[u]["nodes"]]
        out[cid] = E.quantize(f"ilp_{cid}", nodes)
    return out


def main(step):
    cf = configs()
    if step in ("acc", "all"):
        models = [B2] + [f.name for f in cf.values()]
        run(["evaluate.py", "--exec", "int", "--threads", "4", "--save-pred", "--data", "cifar_A", *HR, "--models", *models])
        for f in cf.values():
            run(["bootstrap.py", "--a", B2[:-5], "--b", f.stem, "--data", *HR])
            run(["bootstrap.py", "--a", "fp32", "--b", f.stem, "--data", *HR])
        run(["bootstrap.py", "--a", "fp32", "--b", B2[:-5], "--data", *HR])
        print("acc done", flush=True)
    if step in ("speed", "all"):
        named = [f"fp32={FP32}", f"B2={B2}"] + [f"{k}={f.name}" for k, f in cf.items()]
        for s in (1, 2, 3):
            for t in (1, 2, 4):
                run(["bench_speed.py", "--threads", str(t), "--n", "200", "--repeats", "3", "--warmup", "20", "--models", *named])
                C.append_jsonl("e10_speed_sessions", {"session": s, "threads": t, "meta": C.run_meta()})
                print(f"speed s{s} t{t} done {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(120)                                   # 세션 사이 2분 냉각 (계획서 7절)
    if step in ("mem", "all"):
        run(["bench_mem.py", "--threads", "4", "--models", f"fp32={FP32}", f"B2={B2}", *[f"{k}={f.name}" for k, f in cf.items()]])
        print("mem done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
