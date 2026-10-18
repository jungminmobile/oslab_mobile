"""s2_hailo_temp.py — Hailo 칩 온도 읽기 (읽기 전용, 설정 변경 없음, G3Q-H 10/18). 사용: python3 s2_hailo_temp.py [라벨]  → JSON 한 줄"""
import sys
import json, time
import hailo_platform as hpf
out = {"time": time.strftime("%H:%M:%S"), "label": sys.argv[1] if len(sys.argv) > 1 else ""}
try:
    ids = hpf.Device.scan()
    d = hpf.Device(ids[0])
    t = d.control.get_chip_temperature()
    out.update({"ts0_c": round(float(t.ts0_temperature), 2), "ts1_c": round(float(t.ts1_temperature), 2)})
    d.release()
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:200]
print(json.dumps(out), flush=True)
