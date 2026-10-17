"""
mem_guard_wsl.py — SigLIP2 WSL 작업이 공용 잠금을 잡고 있는 동안 PC 여유 메모리를 지켜보다가, 너무 적어지면 WSL을 내린다 (10/17 밤)
개인연구 세션 요청: 밤새 폰 측정(adb)이 돌고, PC 메모리가 꽉 차면 adb 시간 초과로 측정을 잃는다 → WSL을 10 GB 안쪽으로.
.wslconfig(메모리 상한)는 사용자 시스템 설정이라 바꾸지 않고, 대신 여유 메모리가 1.2 GB 미만으로 20초 이어지면 `wsl -t Ubuntu-22.04`.
사용: "$SIGLIP2_PYTHON" scripts/mem_guard_wsl.py --until-file <끝 표시 로그> --done-text "TONIGHT_1018 DONE"
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

OWNER = Path(r"D:\mokerton\locks\heavy.lock\owner.json")


def our_wsl_job():
    try:
        o = json.loads(OWNER.read_text(encoding="utf-8"))
        return o.get("owner") == "siglip2" and "WSL" in o.get("job", "")
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until-file", required=True)
    ap.add_argument("--done-text", required=True)
    ap.add_argument("--min-avail-gb", type=float, default=1.2)
    a = ap.parse_args()
    low = 0
    while True:
        p = Path(a.until_file)
        if p.exists() and a.done_text in p.read_text(encoding="utf-8", errors="ignore"):
            print(datetime.now().strftime("%T"), "done 표시 확인 — 감시 끝", flush=True)
            return
        if our_wsl_job():
            av = psutil.virtual_memory().available / 2**30
            low = low + 1 if av < a.min_avail_gb else 0
            if low == 1 or (low == 0 and int(time.time()) % 600 < 10):
                print(datetime.now().strftime("%T"), f"avail {av:.2f} GB", flush=True)
            if low >= 2:
                print(datetime.now().strftime("%T"), f"여유 메모리 {av:.2f} GB < {a.min_avail_gb} — WSL 종료", flush=True)
                subprocess.run(["wsl.exe", "-t", "Ubuntu-22.04"])
                low = 0
        else:
            low = 0
        time.sleep(10)


if __name__ == "__main__":
    main()
