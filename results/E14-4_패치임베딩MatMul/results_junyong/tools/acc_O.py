"""O 정확도: run_e14_4.step_acc와 같은 경로(R1.session 4T, embed, l2n @ W)로 O만 평가 → $E14_OUT/acc_O.json"""
import json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\e14work\oslab_mobile\e14")
import run_e14_1 as R1
s = R1.session(R1.O)
res = {"date": datetime.now().isoformat(timespec="seconds"), "git": R1.git_state(), "model": "O", "via": "scratch acc_O.py (run_e14_4.step_acc와 같은 함수)"}
for key in R1.DEV:
    pix, y, W = R1.load_dev(key)
    res[key] = 100 * float(((R1.l2n(R1.embed(s, pix)) @ W).argmax(1) == y).mean())
res["hr"] = float(np.mean([res[k] for k in R1.HRK]))
Path(os.environ["E14_OUT"], "acc_O.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
