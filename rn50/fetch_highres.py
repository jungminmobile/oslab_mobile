"""fetch_highres.py — datasets.py dev에 필요한 parquet(보정·dev 풀 전체 + test 첫 조각)를 curl 이어받기·재시도로 미리 받는다.
urllib 다운로드가 연결 끊김(WinError 10054)으로 실패하는 환경 대비. 받은 파일은 pyarrow로 열어 확인하고, 깨진 파일은 다시 받는다."""
import subprocess
import sys
from datetime import datetime

import pyarrow.parquet as pq

import common as C
import datasets as D

names = sys.argv[1:] or list(D.HIGHRES)
for n in names:
    cfg = D.HIGHRES[n]
    for local, url in cfg["pool"] + cfg["test"][:1]:
        p = C.EXTRA_ROOT / local
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            if p.exists():
                try:
                    pq.ParquetFile(p).metadata
                    break
                except Exception:
                    p.unlink()
            tmp = p.with_suffix(".part")
            rc = subprocess.run(["curl.exe", "-sSL", "--ssl-no-revoke", "-C", "-", "--retry", "20", "--retry-all-errors",
                                 "--retry-delay", "3", "-o", str(tmp), url]).returncode
            if rc == 0:
                tmp.replace(p)
        ok = p.exists()
        print(f"{datetime.now():%H:%M:%S} {'ok  ' if ok else 'FAIL'} {local} {p.stat().st_size // 2**20 if ok else 0} MiB", flush=True)
