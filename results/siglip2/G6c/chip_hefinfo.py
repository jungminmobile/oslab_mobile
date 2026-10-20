"""chip_hefinfo.py — G6c: HEF가 호스트(CPU)에서 실행하는 연산이 있는지 확인 (용 지시)
HEF마다:
  1) `hailortcli parse-hef --parse-streams --parse-vstreams <hef>` 전문 → <out>/parsehef_full_<이름>.txt
  2) hailo_platform.HEF API로 vstream(사용자 쪽)·stream(칩 쪽) 정보 전부 → <out>/hefinfo_<이름>.json
     형식(type·order·flags), 모양, quant_info(qp_scale, qp_zp, limvals), data_bytes, nms_shape, 원래 층 이름
확인 방법 (README 5절에 근거 줄과 함께 적음):
  - 후처리 연산(NMS 등 HEF에 붙은 ops): vstream이 입력 1개·출력 1개이고 nms_shape가 없으며, parse-hef 전문에 ops/NMS 항목이 없으면 "없음"
  - 형식 변환(호스트 transform): vstream과 stream의 format type·order·shape가 같으면, 사용자 형식을 HEF 기본값으로 두었을 때 호스트 변환이 없다
  - 사용자가 FLOAT32를 고르면(eval) HailoRT가 호스트에서 양자화·역양자화한다 — 이것은 HEF가 아니라 호출 쪽 선택이다 (bench는 UINT8 입력·기본 출력)
사용: python3 chip_hefinfo.py --out out/hefinfo --models b16_M2rot2=b16_M2rot2.hef ...
"""
import argparse, hashlib, json, subprocess
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fmt(f):
    out = {}
    for k in ("type", "order", "flags"):
        v = getattr(f, k, None)
        out[k] = str(v) if v is not None else None
    return out


def qi(q):
    return {k: float(getattr(q, k)) for k in ("qp_scale", "qp_zp", "limvals_min", "limvals_max") if hasattr(q, k)}


def nms(i):
    """NMS 순서가 아닌 vstream에서 nms_shape를 읽으면 HailoRT가 예외를 낸다 → 그 경우 None (NMS 아님)."""
    try:
        return str(i.nms_shape)
    except Exception:
        return None


def info(i, h):
    d = {"name": i.name, "direction": str(i.direction), "format": fmt(i.format), "shape": list(i.shape) if i.shape else None,
         "quant_info": qi(i.quant_info), "nms_shape": nms(i)}
    if hasattr(i, "data_bytes"):
        d["data_bytes"] = int(i.data_bytes)
        d["sys_index"] = int(i.sys_index)
    else:
        try:
            d["original_names"] = list(h.get_original_names_from_vstream_name(i.name))
            d["stream_names"] = list(h.get_stream_names_from_vstream_name(i.name))
        except Exception as e:
            d["names_error"] = str(e)[:100]
    return d


def main():
    import hailo_platform as hpf
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for kv in a.models:
        name, hef = kv.split("=", 1)
        txt = subprocess.run(["hailortcli", "parse-hef", "--parse-streams", "--parse-vstreams", hef], capture_output=True, text=True, timeout=120)
        full = txt.stdout + txt.stderr
        (out / f"parsehef_full_{name}.txt").write_text(full)
        h = hpf.HEF(hef)
        d = {"hef": name, "file": Path(hef).name, "sha256": sha256(hef), "bytes": Path(hef).stat().st_size,
             "network_groups": list(h.get_network_group_names()), "networks": list(h.get_networks_names()),
             "input_vstreams": [info(i, h) for i in h.get_input_vstream_infos()],
             "output_vstreams": [info(i, h) for i in h.get_output_vstream_infos()],
             "input_streams": [info(i, h) for i in h.get_input_stream_infos()],
             "output_streams": [info(i, h) for i in h.get_output_stream_infos()]}
        try:
            d["bottleneck_fps"] = float(h.bottleneck_fps())
        except Exception as e:
            d["bottleneck_fps_error"] = str(e)[:100]
        same = lambda v, s: (v["format"]["type"], v["format"]["order"], v["shape"]) == (s["format"]["type"], s["format"]["order"], s["shape"])
        d["check"] = {
            "n_in_vstreams": len(d["input_vstreams"]), "n_out_vstreams": len(d["output_vstreams"]),
            "n_in_streams": len(d["input_streams"]), "n_out_streams": len(d["output_streams"]),
            "any_nms": any(v["nms_shape"] for v in d["input_vstreams"] + d["output_vstreams"]),
            "parsehef_mentions_ops": [ln.strip() for ln in full.splitlines() if any(w in ln.lower() for w in ("nms", "op ", "ops", "post", "pre-process", "preprocess"))],
            "in_vstream_eq_stream": len(d["input_vstreams"]) == len(d["input_streams"]) and all(same(v, s) for v, s in zip(d["input_vstreams"], d["input_streams"])),
            "out_vstream_eq_stream": len(d["output_vstreams"]) == len(d["output_streams"]) and all(same(v, s) for v, s in zip(d["output_vstreams"], d["output_streams"])),
        }
        (out / f"hefinfo_{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
        summary.append({"hef": name, **d["check"], "in": d["input_vstreams"][0]["format"] | {"shape": d["input_vstreams"][0]["shape"], "q": d["input_vstreams"][0]["quant_info"]},
                        "in_stream": d["input_streams"][0]["format"] | {"shape": d["input_streams"][0]["shape"]},
                        "out": d["output_vstreams"][0]["format"] | {"shape": d["output_vstreams"][0]["shape"], "q": d["output_vstreams"][0]["quant_info"]},
                        "out_stream": d["output_streams"][0]["format"] | {"shape": d["output_streams"][0]["shape"]}})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
