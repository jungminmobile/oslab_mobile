"""e12_3_check_env.py — E12-3 0단계 (계획서 6절 0, 0-2)

1) env.json (env_info + oslab_mobile git), 2) S·F sha256 비교(S=계획서 고정값, F=Release SHA256SUMS.txt),
3) ORT CPU EP 정수 커널 유무: 작은 모델을 만들어 세션 생성이 되는지로 확인 (등록 목록 API가 없으므로 실측),
4) P코어/E코어 논리 번호 (GetLogicalProcessorInformationEx, EfficiencyClass) → 0-based 와 ORT용(+1),
5) 번호 확인: S 4T를 P코어 고정 / E코어 고정으로 짧게 돌리며 코어별 사용률(psutil)과 지연을 기록.
결과: <out>/env.json, <out>/check_env.json
"""
import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402

S_SHA = "7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def core_classes():
    """논리 프로세서(0-based) → EfficiencyClass. 값이 클수록 성능 코어."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    n = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))   # RelationProcessorCore = 0
    buf = ctypes.create_string_buffer(n.value)
    assert k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)), ctypes.get_last_error()
    raw, off, out = buf.raw, 0, {}
    while off < n.value:
        rel, size = int.from_bytes(raw[off:off + 4], "little"), int.from_bytes(raw[off + 4:off + 8], "little")
        eff = raw[off + 9]
        gcount = int.from_bytes(raw[off + 30:off + 32], "little")
        for g in range(gcount):
            base = off + 32 + g * 16
            mask = int.from_bytes(raw[base:base + 8], "little")
            group = int.from_bytes(raw[base + 8:base + 10], "little")
            for bit in range(64):
                if mask >> bit & 1:
                    out[group * 64 + bit] = eff
        off += size
    return dict(sorted(out.items()))


def probe_kernels():
    """ORT CPU EP에서 세션 생성이 되면 정수 커널 있음."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto as T, helper as H
    f32, u8 = T.FLOAT, T.UINT8
    sc = lambda n: H.make_tensor(n, f32, [], [0.1])
    zp = lambda n: H.make_tensor(n, u8, [], [128])
    inits = [sc("s"), zp("z")]
    two_in = ["A", "s", "z", "B", "s", "z", "s", "z"]
    one_in = ["A", "s", "z", "s", "z"]
    cands = {  # op: (domain, inputs, attrs, n_data_inputs)
        "QLinearAdd": ("com.microsoft", two_in, {}, 2), "QLinearMul": ("com.microsoft", two_in, {}, 2),
        "QLinearSub": ("com.microsoft", two_in, {}, 2), "QLinearSoftmax": ("com.microsoft", one_in, {"axis": -1, "opset": 13}, 1),
        "QLinearSigmoid": ("com.microsoft", one_in, {}, 1), "QLinearLeakyRelu": ("com.microsoft", one_in, {}, 1),
        "QLinearGelu": ("com.microsoft", one_in, {}, 1), "QGelu": ("com.microsoft", one_in, {}, 1),
        "QLinearLayerNormalization": ("com.microsoft", one_in, {}, 1), "QLinearReduceMean": ("com.microsoft", one_in, {"axes": [-1], "keepdims": 1}, 1),
        "QLinearConcat": ("com.microsoft", ["s", "z", "A", "s", "z", "B", "s", "z"], {"axis": -1}, 2),
        "QLinearWhere": ("com.microsoft", None, {}, 0),
    }
    res = {}
    for op, (dom, ins, attrs, nd) in cands.items():
        try:
            if ins is None:
                raise RuntimeError("probe 생략 (Where는 이 모델에 없음)")
            data = [H.make_tensor_value_info(nm, u8, [1, 4, 8]) for nm in ["A", "B"][:nd]]
            node = H.make_node(op, ins, ["Y"], domain=dom, **attrs)
            g = H.make_graph([node], "p", data, [H.make_tensor_value_info("Y", u8, None)], inits)
            m = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)])
            m.ir_version = 9   # ORT 1.25 최대 IR 13, onnx 1.23 기본 14
            s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
            s.run(None, {nm: np.full((1, 4, 8), 130, np.uint8) for nm in ["A", "B"][:nd]})
            res[op] = {"available": True}
        except Exception as e:  # noqa: BLE001
            res[op] = {"available": False, "error": str(e).splitlines()[0][:200]}
    return res


AFF_CHILD = r'''
import sys, json, time, threading
import numpy as np, psutil
sys.path.insert(0, sys.argv[1]); import session_opts as SO
model, cores0 = sys.argv[2], [int(c) for c in sys.argv[3].split(",")]
psutil.Process().cpu_affinity(cores0)
so = SO.make_options("C3", 4)
so.add_session_config_entry("session.intra_op_thread_affinities", ";".join(str(c + 1) for c in cores0[1:4]))
import onnxruntime as ort
s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"]); nm = s.get_inputs()[0].name
x = np.random.default_rng(0).standard_normal((1, 3, 256, 256)).astype(np.float32)
for _ in range(10): s.run(None, {nm: x})
psutil.cpu_percent(percpu=True); t = []; t_end = time.time() + 4
while time.time() < t_end:
    t0 = time.perf_counter(); s.run(None, {nm: x}); t.append((time.perf_counter() - t0) * 1000)
busy = psutil.cpu_percent(percpu=True)
print(json.dumps({"p50_ms": round(float(np.median(t)), 2), "n": len(t), "percpu_busy": busy}))
'''


def affinity_check(model, pcores, ecores):
    out = {}
    for tag, cores in (("P", pcores), ("E", ecores)):
        p = subprocess.run([sys.executable, "-c", AFF_CHILD, str(SO.HERE), str(model), ",".join(map(str, cores))],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out[tag] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-800:]}
        out[tag]["cores0"] = cores
        time.sleep(5)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", required=True)
    ap.add_argument("--F", required=True)
    ap.add_argument("--F_sums", required=True, help="Release siglip2-b32-candidate-20261020의 SHA256SUMS.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SO.S2_SIGLIP))
    from scripts import env_info
    env = env_info.collect()
    env["git_oslab_mobile"] = SO.git_state()
    env_info.write_env(out, env)

    S, F = Path(a.S), Path(a.F)
    s_sha, f_sha = sha(S), sha(F)
    sums = {ln.split()[-1].lstrip("*"): ln.split()[0] for ln in Path(a.F_sums).read_text(encoding="utf-8").splitlines() if ln.strip()}
    f_exp = sums.get(F.name)
    shas = {"S": {"file": S.name, "bytes": S.stat().st_size, "sha256": s_sha, "expect": S_SHA, "ok": s_sha == S_SHA},
            "F": {"file": F.name, "bytes": F.stat().st_size, "sha256": f_sha, "expect": f_exp, "ok": f_sha == f_exp}}
    print("SHA", json.dumps(shas, ensure_ascii=False), flush=True)

    cls = core_classes()
    top = max(cls.values())
    p0 = [c for c, e in cls.items() if e == top]
    e0 = [c for c, e in cls.items() if e != top]
    cores = {"efficiency_class": cls, "pcores_0based": p0, "ecores_0based": e0,
             "pcores_ort": [c + 1 for c in p0], "ecores_ort": [c + 1 for c in e0],
             "note": "ORT intra_op_thread_affinities 번호 = 0-based + 1 (가정; affinity_check로 코어별 사용률 확인)"}
    print("CORES", json.dumps(cores), flush=True)

    kern = probe_kernels()
    print("KERNELS", json.dumps({k: v["available"] for k, v in kern.items()}), flush=True)

    aff = affinity_check(S, p0, e0) if shas["S"]["ok"] and len(p0) >= 4 and len(e0) >= 4 else {"skipped": True}
    print("AFFINITY", json.dumps(aff), flush=True)

    pw = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, errors="replace").stdout.strip()
    import os
    res = {"date": datetime.now().isoformat(timespec="seconds"), "sha": shas, "G0_pass": shas["S"]["ok"] and shas["F"]["ok"],
           "cores": cores, "int_kernels": kern, "affinity_check": aff, "powercfg": pw,
           "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "power": env.get("power")}
    (out / "check_env.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("G0_pass", res["G0_pass"])
