"""
s2_g6.py — G6: Zoo B32·S1B16에 M2rot을 적용한 ONNX export·FP32 기준·데이터 준비 (데스크탑, DFC·AGX 밖)
=========================================================================================================

[설계·판정]  `results/siglip2/G6/README.md` (결과 전에 고정).
[모델]  --model b32 | s1b16 → open_clip timm 모델 (`S2_MODEL_ID` 환경변수를 s2common import 전에 정한다)
  b32   hf-hub:timm/ViT-B-32-SigLIP2-256   (Zoo siglip2_b_32_256, HF google/siglip2-base-patch32-256)
  s1b16 hf-hub:timm/ViT-B-16-SigLIP        (Zoo siglip_b_16, HF google/siglip-base-patch16-224)
[하위 명령]  (siglip2/ 폴더에서, RN50_RESULTS=../results/siglip2/G6)
  data     Zoo 전처리 dev 전체(고화질 5종 dev 전부 + CIFAR A 1000) uint8 NHWC 픽셀, 보정 256·64장 NHWC float32 0~255, bench용 1장
  ident    open_clip timm 모델 vs Zoo ONNX(ORT) cos (G6_pre ident32와 같은 32장)
  export   --cfg plain|rot|m2rot → b1 ONNX (DFC parse용) + FP32 동치 확인 (README 4-4)
  fp32     HF FP32 기준 임베딩·분류기 (dev 전체) + (b32) 통일 규칙 M2 FP32
[출력]  ~/siglip2_work/g6/ (큰 파일), results/siglip2/G6/*.jsonl (기록)
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

MODEL_IDS = {"b32": "hf-hub:timm/ViT-B-32-SigLIP2-256", "s1b16": "hf-hub:timm/ViT-B-16-SigLIP",
             "b16": "hf-hub:timm/ViT-B-16-SigLIP2"}   # b16: G6b, 우리 B16 (FP32 기준·분류기만, HF 키 ref)
SIZE = {"b32": 256, "s1b16": 224, "b16": 224}
HFKEY = {"b32": "b32", "s1b16": "s1b16", "b16": "ref"}   # s2_g6pre.MODELS 키
ZOO_ONNX = {"b32": ("dl_b32/siglip2-base-patch32-256_vision_encoder.sim.onnx", "1393"),
            "s1b16": ("dl_b16/siglip_base_patch16_224_vision_encoder.onnx", "1725")}
UNIFIED = {"b32": ([(15, 6), (7, 10)], [5, 9])}       # G6_pre 13-2: K/V 시작, u_c 시작 (15@6:5, 7@10:9)


def _pre_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", choices=list(MODEL_IDS))
    return p.parse_known_args()[0]


_m = _pre_args().model
if _m:
    os.environ["S2_MODEL_ID"] = MODEL_IDS[_m]
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "6")

import numpy as np    # noqa: E402
import torch          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2common as S                    # noqa: E402
import s2_g6pre as P                    # noqa: E402  (Zoo 전처리, HF forward, 분류기 — G6_pre와 같은 코드)
from scripts import env_info            # noqa: E402

RC, RD = S.RC, S.RD
torch.set_num_threads(6)
torch.set_grad_enabled(False)
WORK = Path(os.environ.get("G6_WORK", Path.home() / "siglip2_work" / "g6"))
PRE = P.WORK                             # ~/siglip2_work/g6pre (Zoo ONNX, 보정 캐시)
HR = P.HR
DEV = ["cifar_A"] + [f"{n}_dev" for n in HR]


def dev_dir(size):
    d = WORK / f"dev_zoo{size}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def append(name, row):
    env = env_info.collect()
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "model_id": S.MODEL_ID, **row}
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:1500], flush=True)


def pix_to_x(pix):
    """uint8 NHWC 픽셀 → Zoo 전처리 정규화 float32 NCHW (HEF 안 normalization과 같은 식)."""
    return (np.asarray(pix, np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2).copy()


# ---------------------------------------------------------------- data (README 2·5-1)

def cmd_data(a):
    from PIL import Image
    import hashlib
    from torchvision.datasets import CIFAR100
    info = {}
    for name in HR:
        dev, ys, _ = P._raw_highres(name)
        sha_old = (RC.CACHE_DIR / f"{name}_dev_sha1.txt").read_text().split()
        assert [hashlib.sha1(b).hexdigest() for b in dev] == sha_old, f"{name} dev 사진 목록이 기존 캐시와 다름"
        for size in (224, 256):
            pix = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((size, size), Image.BICUBIC), np.uint8)
                            for b in dev])
            np.save(dev_dir(size) / f"{name}_dev_pix.npy", pix)
            np.save(dev_dir(size) / f"{name}_dev_y.npy", np.array(ys))
            # G6_pre 앞 200장 캐시와 같은 값인지 (같은 전처리 함수)
            old = np.load(PRE / f"cache_zoo{size}" / f"{name}_dev_x.npy")
            assert np.abs(pix_to_x(pix[:200]) - old).max() < 1e-6, (name, size)
        info[name] = len(dev)
        print(name, len(dev), flush=True)
    ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=False)
    s0 = RC.VAL_SPLITS["A"]
    for size in (224, 256):
        pix = np.stack([np.asarray(ds[i][0].convert("RGB").resize((size, size), Image.BICUBIC), np.uint8) for i in range(s0, s0 + RC.VAL_SIZE)])
        np.save(dev_dir(size) / "cifar_A_pix.npy", pix)
        np.save(dev_dir(size) / "cifar_A_y.npy", np.array(ds.targets[s0:s0 + RC.VAL_SIZE]))
        old = np.load(PRE / f"cache_zoo{size}" / "cifar_A_x.npy")
        assert np.abs(pix_to_x(pix[:200]) - old).max() < 1e-6
        # 보정: G6_pre calib256_s0 (Zoo 전처리, 시드 0) → NHWC float32 0~255 (DFC 입력), 앞 64장 (V)
        xc = np.load(PRE / f"cache_zoo{size}" / "calib256_s0_x.npy")
        pc = np.round((xc.transpose(0, 2, 3, 1) + 1.0) * 127.5)
        assert np.abs(pc / 127.5 - 1.0 - xc.transpose(0, 2, 3, 1)).max() < 1e-5 and pc.min() >= 0 and pc.max() <= 255
        np.save(WORK / f"calib256_s0_zoo{size}_nhwc.npy", pc.astype(np.float32))
        np.save(WORK / f"calib64_s0_zoo{size}_nhwc.npy", pc[:64].astype(np.float32))
    info["cifar_A"] = RC.VAL_SIZE
    sums = {}
    for size in (224, 256):
        for f in sorted(dev_dir(size).glob("*.npy")):
            sums[f"dev_zoo{size}/{f.name}"] = P._sha(f.read_bytes())
        for n in (256, 64):
            f = WORK / f"calib{n}_s0_zoo{size}_nhwc.npy"
            sums[f.name] = P._sha(f.read_bytes())
    (WORK / "SHA1SUMS.json").write_text(json.dumps(sums, indent=1))
    append("data.jsonl", {"what": "data", "n": info, "hr_n": sum(info[n] for n in HR), "sha1": sums,
                          "preprocess": "PIL BICUBIC 바로 맞춤, crop 없음, uint8 (Zoo clip)"})


def load_dev(size, key):
    return np.load(dev_dir(size) / f"{key}_pix.npy", mmap_mode="r"), np.load(dev_dir(size) / f"{key}_y.npy")


def dev64(size):
    """FP32 동치 확인용: Pets dev 앞 32 + Caltech dev 앞 32 (Zoo 전처리)."""
    return np.concatenate([pix_to_x(load_dev(size, "pets_dev")[0][:32]), pix_to_x(load_dev(size, "caltech_dev")[0][:32])])


# ---------------------------------------------------------------- 모델 (open_clip timm)

def load_visual(rot):
    from s2_regcache import unfuse, apply_rot
    model, _, _ = S.load_model()
    unfuse(model)
    qsha = apply_rot(model, "rothad") if rot else None
    return model, qsha


def feat(visual, x, bs=16):
    return np.concatenate([visual(torch.as_tensor(np.ascontiguousarray(x[i:i + bs]))).float().numpy() for i in range(0, len(x), bs)])


def zoo_onnx(model, x):
    import onnxruntime as ort
    rel, out = ZOO_ONNX[model]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 6
    s = ort.InferenceSession(str(PRE / rel), so, providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return np.concatenate([s.run([out], {n: x[i:i + 1]})[0] for i in range(len(x))])


def cmd_ident(a):
    model, _ = load_visual(False)
    size = SIZE[a.model]
    assert tuple(model.visual.trunk.patch_embed.img_size) == (size, size)
    x = P.ident32(size)
    c = P.cosr(feat(model.visual, x), zoo_onnx(a.model, x))
    row = {"what": "ident", "model": a.model, "n": len(x), "cos_mean": float(c.mean()), "cos_min": float(c.min())}
    append("ident.jsonl", row)
    assert c.mean() >= 0.9999, "open_clip timm 모델이 Zoo ONNX와 다름 — 멈춤 (README 2)"


# ---------------------------------------------------------------- export (README 4-2·4-4)

def m2_trunk(model, xcal):
    from s2_regcache import MaskRegCacheTrunk, build_mean_cache, build_uc
    kv_sinks, u_starts = UNIFIED["b32"]
    trunk = model.visual.trunk
    cache, n1 = build_mean_cache(trunk, xcal, kv_sinks)
    uc, n2 = build_uc(trunk, xcal, [(p, u) for (p, _), u in zip(kv_sinks, u_starts)])
    assert n1 == n2 == len(xcal) == 256, (n1, n2, len(xcal))
    return MaskRegCacheTrunk(trunk, kv_sinks, cache, "bcast", None, uc, u_starts=u_starts).eval(), cache, uc


def cmd_export(a):
    import onnx
    from s2_rotate import export_onnx
    size = SIZE[a.model]
    model, qsha = load_visual(a.cfg in ("rot", "m2rot"))
    x = dev64(size)
    ref_hf = None
    extra = {}
    if a.cfg == "m2rot":
        assert a.model == "b32", "M은 B32만 (README 2)"
        xcal = np.load(PRE / f"cache_zoo{size}" / "calib256_s0_x.npy")
        rc, cache, uc = m2_trunk(model, xcal)
        model.visual.trunk = rc
        # HF G6_pre 구현(같은 sinks·같은 상수 정의, 회전 없음)과 비교 — 회전은 FP32에서 같은 변환
        vis = P.vis_of(P.load_hf("b32"))
        sinks = [(p, k, u) for (p, k), u in zip(*UNIFIED["b32"])]
        kvc, ucs, n = P.build_m2_consts(vis, xcal, sinks)
        ref_hf = P.batched(lambda t: P.hf_forward(vis, t, sinks, kvc, ucs), x)
        extra["unified"] = "15@6:5,7@10:9"
    e_t = feat(model.visual, x)
    stem = f"g6_{a.model}_{a.cfg}"
    p = WORK / "onnx" / (f"{stem}_dyn.onnx" if a.dynamic else f"{stem}_b1.onnx")   # --dynamic (G6b, 기본 꺼짐): 배치 가변
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not a.overwrite:
        raise SystemExit(f"{p} 이미 있음 (--overwrite)")
    if a.cfg == "m2rot":       # M2rot과 같은 export (s2_regcache cmd_export: torch.onnx.export 직접, undedup 없음)
        with torch.inference_mode():
            kw = dict(dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}) if a.dynamic else {}
            torch.onnx.export(model.visual, (torch.randn(1, 3, size, size),), str(p), input_names=["image"],
                              output_names=["embedding"], opset_version=17, dynamo=False, **kw)
        n_undedup = None
    else:                      # rothad·D256L1과 같은 export (s2_rotate.export_onnx: undedup 포함)
        n_undedup = export_onnx(model.visual, p, a.dynamic, 17)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 6
    s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    e_o = np.concatenate([s.run(None, {"image": x[i:i + 1]})[0] for i in range(len(x))])
    chk = {"ort_vs_torch_cos_min": float(P.cosr(e_o, e_t).min())}
    if ref_hf is not None:
        chk["torch_vs_hf_m2_cos_min"] = float(P.cosr(e_t, ref_hf).min())
    else:
        chk["torch_vs_zoo_onnx_cos_min"] = float(P.cosr(e_t, zoo_onnx(a.model, x)).min())
    ops = {}
    for nd in onnx.load(str(p), load_external_data=False).graph.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    import hashlib
    row = {"what": "export", "model": a.model, "cfg": a.cfg, "dynamic": a.dynamic, "onnx": str(p), "bytes": p.stat().st_size,
           "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rotq_sha256": qsha, "n_undedup": n_undedup,
           "ops": dict(sorted(ops.items())), "check_n": len(x), **chk, **extra}
    append("export.jsonl", row)
    assert all(v >= 0.9999 for v in chk.values()), f"FP32 동치 실패 {chk} — 멈춤 (README 4-4)"


# ---------------------------------------------------------------- fp32 기준 (README 5-1)

def cmd_fp32(a):
    from transformers import AutoTokenizer
    from torchvision.datasets import CIFAR100
    name = a.model
    size = SIZE[name]
    m = P.load_hf(HFKEY[name])
    vis = P.vis_of(m)
    tok = AutoTokenizer.from_pretrained(P.MODELS[HFKEY[name]]["hf"])
    cif = [c.replace("_", " ") for c in CIFAR100(str(RC.DATA_ROOT), train=True, download=False).classes]
    out = WORK / "ref"
    out.mkdir(parents=True, exist_ok=True)
    sinks = consts = None
    if name == "b32":
        xcal = np.load(PRE / f"cache_zoo{size}" / "calib256_s0_x.npy")
        sinks = [(p, k, u) for (p, k), u in zip(*UNIFIED["b32"])]
        consts = P.build_m2_consts(vis, xcal, sinks)
        assert consts[2] == 256
    res = {}
    for key in DEV:
        names, templates = (cif, RC.TEMPLATES) if key == "cifar_A" else (RD.classnames(key[:-4]), RD.HIGHRES[key[:-4]]["templates"])
        W = P.classifier(m, tok, names, templates)
        pix, y = load_dev(size, key)
        e = np.concatenate([P.batched(lambda t: vis(pixel_values=t).pooler_output, pix_to_x(pix[i:i + 256])) for i in range(0, len(y), 256)])
        np.save(out / f"{name}_{key}_clf.npy", W)
        np.save(out / f"{name}_{key}_ref_fp32.npy", e)
        np.save(out / f"{name}_{key}_y.npy", y)
        r = {"n": int(len(y)), "fp32_top1": round(100 * float(((P.l2n(e) @ W).argmax(1) == y).mean()), 2)}
        if sinks:
            e2 = np.concatenate([P.batched(lambda t: P.hf_forward(vis, t, sinks, consts[0], consts[1]), pix_to_x(pix[i:i + 256]))
                                 for i in range(0, len(y), 256)])
            np.save(out / f"{name}_{key}_m2_fp32.npy", e2)
            c = P.cosr(e2, e)
            r.update(m2_top1=round(100 * float(((P.l2n(e2) @ W).argmax(1) == y).mean()), 2), m2_cos_mean=float(c.mean()), m2_cos_min=float(c.min()))
        res[key] = r
        print(key, r, flush=True)
    hr = {k: round(float(np.mean([res[d][k] for d in DEV[1:]])), 3) for k in ("fp32_top1", "m2_top1") if k in res["pets_dev"]}
    append("fp32.jsonl", {"what": "fp32", "model": name, "hf": P.MODELS[HFKEY[name]]["hf"], "per_set": res, "hr_mean": hr,
                          "unified": "15@6:5,7@10:9" if sinks else None,
                          "text": "HF 텍스트 인코더 FP32, _clean_canonicalize, max_length 64 (G6_pre와 같음)"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODEL_IDS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").set_defaults(func=cmd_data)
    sub.add_parser("ident").set_defaults(func=cmd_ident)
    p = sub.add_parser("export")
    p.add_argument("--cfg", choices=["plain", "rot", "m2rot"], required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dynamic", action="store_true", help="(G6b, 기본 꺼짐) 배치 가변 ONNX (ORT 양자화·eval용)")
    p.set_defaults(func=cmd_export)
    sub.add_parser("fp32").set_defaults(func=cmd_fp32)
    a = ap.parse_args()
    if a.cmd != "data":
        assert a.model, "--model 필요 (하위 명령 앞에)"
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a.func(a)
    print(f"[{a.cmd}] {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
