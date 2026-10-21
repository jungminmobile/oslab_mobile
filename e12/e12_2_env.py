"""e12_2_env.py — E12-2 0단계: env.json, g0_sha.json 기록 (계획서 4절 G0)"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402
G8 = SO.REPO.parent / "oslab_g8" / "siglip2"
sys.path.insert(0, str(G8))
from scripts import env_info  # noqa: E402

ONNX = Path(r"C:\s2art\onnx")
FILES = {"M2rot_fp32": ("siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx", 371605898, "16ef696e"),
         "O": ("siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx", None, "e84c3727")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env["git_oslab_mobile"] = SO.git_state()
    env_info.write_env(out, env)
    rows = {}
    for k, (name, size, pre) in FILES.items():
        p = ONNX / name
        s = sha(p)
        rows[k] = {"file": name, "bytes": p.stat().st_size, "sha256": s, "expect_prefix": pre, "expect_bytes": size,
                   "ok": s.startswith(pre) and (size is None or p.stat().st_size == size)}
    res = {"date": datetime.now().isoformat(timespec="seconds"), "files": rows, "G0_pass": all(r["ok"] for r in rows.values())}
    (out / "g0_sha.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
