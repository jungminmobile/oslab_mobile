"""e12_4_speed.py — E12-4 자식 프로세스 1회분 (계획서 4절)

speed: 세션 생성 → 워밍업 W장 → cpu_times → N장(장마다 perf_counter) → cpu_times → sleep(tail) → cpu_times
       출력: p50·p90·p99, 장당 CPU ms(꼬리 포함 = 판정 값), 루프만 CPU ms, 꼬리 CPU ms
equal: pets 캐시 앞 50장을 256으로 맞춰 실행, 출력 전체를 npz로 저장
입력(speed): E12-3과 같음 — pets_dev_x[0] (224 캐시) → [0,255] → PIL bilinear 256 → x/127.5-1, 1장 반복
사용: python e12_4_speed.py '<json cfg>'   (cfg: mode, model, tcombo, threads, warm, n, tail, save)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402


def to256(x224, img=256):
    from PIL import Image
    u8 = np.clip((np.transpose(x224, (1, 2, 0)) + 1) * 127.5, 0, 255).round().astype(np.uint8)
    im = np.asarray(Image.fromarray(u8).resize((img, img), Image.BILINEAR), np.float32)
    return np.ascontiguousarray(np.transpose(im / 127.5 - 1, (2, 0, 1))[None], np.float32)


def no_power_throttle():
    """Windows 전원 절약(EcoQoS) 끄기: 백그라운드·최소화 창 프로세스가 E코어로 밀리는 것 방지 (10/23 1차 시도에서 1T가 E코어 속도로 측정됨)."""
    import ctypes
    from ctypes import wintypes

    class PPT(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    st = PPT(1, 1, 0)   # CURRENT_VERSION=1, EXECUTION_SPEED 제어, 상태 0 = 절약 끔
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))  # ProcessPowerThrottling=4


def main():
    cfg = json.loads(sys.argv[1])
    throttle_off = no_power_throttle() if cfg.get("no_throttle", True) else None
    import onnxruntime as ort
    import psutil
    proc = psutil.Process()
    cache = np.load(SO.CACHE / "pets_dev_x.npy", mmap_mode="r")
    so = SO.make_thread_options(cfg["tcombo"], cfg["threads"])
    s = ort.InferenceSession(cfg["model"], so, providers=["CPUExecutionProvider"])
    nm = s.get_inputs()[0].name
    keys = {k: so.get_session_config_entry(k) for k in SO.THREAD_COMBOS[cfg["tcombo"]]}
    if cfg["mode"] == "equal":
        outs = [s.run(None, {nm: to256(cache[i])}) for i in range(cfg["n"])]
        np.savez(cfg["save"], **{f"o{j}": np.concatenate([o[j] for o in outs]) for j in range(len(outs[0]))})
        print(json.dumps({"saved": cfg["save"], "keys": keys}))
        return
    x = to256(cache[0])
    for _ in range(cfg["warm"]):
        s.run(None, {nm: x})
    c0 = proc.cpu_times()
    t = []
    for _ in range(cfg["n"]):
        t0 = time.perf_counter()
        s.run(None, {nm: x})
        t.append((time.perf_counter() - t0) * 1000)
    c1 = proc.cpu_times()
    time.sleep(cfg["tail"])
    c2 = proc.cpu_times()
    cpu = lambda a, b: (b.user + b.system) - (a.user + a.system)
    t = np.array(t)
    n = cfg["n"]
    print(json.dumps({"p50_ms": round(float(np.median(t)), 3), "p90_ms": round(float(np.percentile(t, 90)), 3),
                      "p99_ms": round(float(np.percentile(t, 99)), 3), "mean_ms": round(float(t.mean()), 3),
                      "min_ms": round(float(t.min()), 3), "loop_wall_s": round(float(t.sum() / 1000), 3),
                      "cpu_ms_per_img": round(cpu(c0, c2) * 1000 / n, 3), "cpu_ms_per_img_loop": round(cpu(c0, c1) * 1000 / n, 3),
                      "tail_cpu_ms": round(cpu(c1, c2) * 1000, 3), "keys": keys, "throttle_off": throttle_off}))


if __name__ == "__main__":
    main()
