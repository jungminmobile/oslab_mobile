"""
e5b_run.py — E5b 0~2단계 실행 (사전 등록 results/E5b_bias_corr/PREREG.md, 커밋 b5c4b7f)
  check : B0·B2 sha 확인 → zero-delta(sha=B0) → B0+BC(시드 0, 정수 확인 포함) → 구조 비교 → ③ 관문
  build : B0+BC′(시드 1), B2+BC(시드 0)
  acc   : dev 전체 정확도 + 짝 부트스트랩
"""
import json
import subprocess
import sys

import onnx
from onnx import numpy_helper as nh

import bias_correct as BC
import common as C

D = C.RESULTS_DIR
B0 = "rn50_image_int8_q1_percentile99999_highres256_s0_B0.onnx"
B2 = "rn50_image_int8_q1_percentile99999_highres256_s0_B2.onnx"
SHA = {B0: "9d99f92d2bc3013dc388d05afa25d7efce6721247ef9a2fe01ba72bf4670ad3f",
       B2: "687097928a9d82146d486f440e9956ab51d0ebacfffdf34c37e2c432dd3f19cd"}
HR = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
out = lambda src, tag: src[:-5] + f"_{tag}.onnx"


def py(*args):
    r = subprocess.run([sys.executable, *args], cwd=C.HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-3000:], flush=True)
    assert r.returncode == 0, r.stderr[-2000:]


def last_bc(tag, src):
    rows = [json.loads(l) for l in (D / "bc_log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["tag"] == tag and r["src"] == src][-1]


def same_but_bias(a, b, T):
    """그래프 노드·입출력이 같고, 다른 initializer는 bias(int32 q / Add 상수)뿐인지."""
    ma, mb = onnx.load(C.onnx_path(a).as_posix()), onnx.load(C.onnx_path(b).as_posix())
    same_nodes = [n.SerializeToString() for n in ma.graph.node] == [n.SerializeToString() for n in mb.graph.node]
    ia = {i.name: nh.to_array(i) for i in ma.graph.initializer}
    ib = {i.name: nh.to_array(i) for i in mb.graph.initializer}
    allowed = {t.get("q") or t.get("f") for t in T}
    diff = [k for k in ia if k not in ib or ia[k].shape != ib[k].shape or not (ia[k] == ib[k]).all()]
    return {"same_nodes": same_nodes, "same_init_keys": set(ia) == set(ib), "n_diff": len(diff),
            "non_bias_diff": [k for k in diff if k not in allowed]}


def check():
    rec = {"sha_ok": {f: BC.sha(C.onnx_path(f)) == s for f, s in SHA.items()}}
    assert all(rec["sha_ok"].values()), rec
    zf = C.onnx_path(out(B0, "BCzero"))
    if not (zf.exists() and BC.sha(zf) == SHA[B0]):        # 04:44 1차 실행에서 이미 통과(버그와 무관) → 다시 하지 않음
        py("bias_correct.py", "--src", B0, "--tag", "BCzero", "--zero-delta")
    rec["zero_delta_sha_eq_B0"] = BC.sha(C.onnx_path(out(B0, "BCzero"))) == SHA[B0]
    py("bias_correct.py", "--src", B0, "--tag", "BC", "--calib-seed", "0", "--check-int")
    T = BC.targets(onnx.load(C.onnx_path(B0).as_posix()))
    rec["structure"] = same_but_bias(B0, out(B0, "BC"), T)
    r = last_bc("BC", B0)
    rec["reduction_fake"], rec["reduction_int"] = r["reduction_fake"], r["reduction_int"]
    rec["gate"] = (rec["zero_delta_sha_eq_B0"] and rec["structure"]["same_nodes"] and rec["structure"]["same_init_keys"]
                   and not rec["structure"]["non_bias_diff"] and r["reduction_fake"] >= 0.9 and r["reduction_int"] >= 0.5)
    C.append_jsonl("e5b_check", {**rec, "meta": C.run_meta()})
    print("CHECK", json.dumps(rec, ensure_ascii=False), flush=True)
    if not rec["gate"]:
        sys.exit(2)


def build():
    py("bias_correct.py", "--src", B0, "--tag", "BCp", "--calib-seed", "1")
    py("bias_correct.py", "--src", B2, "--tag", "BC", "--calib-seed", "0")


def acc():
    models = [B0, B2, out(B0, "BC"), out(B0, "BCp"), out(B2, "BC")]
    py("evaluate.py", "--exec", "int", "--threads", "4", "--save-pred", "--data", "cifar_A", *HR, "--models", *models)
    s = lambda f: f[:-5]
    pairs = [(s(B2), s(out(B0, "BC"))), (s(B2), s(out(B0, "BCp"))), (s(B0), s(out(B0, "BC"))), (s(B2), s(out(B2, "BC"))),
             (s(B2), s(B0))] + [("fp32", s(f)) for f in models]
    for a, b in pairs:
        py("bootstrap.py", "--a", a, "--b", b, "--data", *HR)


def speed():
    """계획서 2단계 속도: 3세션 × 1·2·4T, 한 실행에서 번갈아 (E10과 같은 설정, 세션 사이 2분 냉각)."""
    import time
    named = ["fp32=rn50_image_fp32_q1.onnx", f"B2={B2}", f"B0={B0}", f"B0BC={out(B0, 'BC')}", f"B0BCp={out(B0, 'BCp')}", f"B2BC={out(B2, 'BC')}"]
    for s in (1, 2, 3):
        for t in (1, 2, 4):
            py("bench_speed.py", "--threads", str(t), "--n", "200", "--repeats", "3", "--warmup", "20", "--models", *named)
            C.append_jsonl("e5b_speed_sessions", {"session": s, "threads": t, "meta": C.run_meta()})
            print(f"speed s{s} t{t} done {time.strftime('%H:%M:%S')}", flush=True)
        if s < 3:
            time.sleep(120)


def mem():
    py("bench_mem.py", "--threads", "4", "--models", "fp32=rn50_image_fp32_q1.onnx", f"B2={B2}", f"B0={B0}",
       f"B0BC={out(B0, 'BC')}", f"B0BCp={out(B0, 'BCp')}", f"B2BC={out(B2, 'BC')}")


if __name__ == "__main__":
    {"check": check, "build": build, "acc": acc, "speed": speed, "mem": mem}[sys.argv[1]]()
