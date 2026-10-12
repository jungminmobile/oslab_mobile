"""
s2_export_onnx.py — SigLIP2 이미지 인코더 PyTorch → ONNX(FP32) 변환 + 동일성 확인 + 연산 목록 (ROADMAP G1)
=====================================================================================================

[목적]
  양자화 도구(ONNX Runtime)와 Hailo DFC는 float ONNX를 입력으로 받는다. 이미지 인코더만 내보낸다
  (텍스트는 FP32 PyTorch로 분류기를 1회 계산, D-004). 출력은 정규화 전 768차원 임베딩이고,
  L2 정규화와 유사도 계산은 바깥에서 한다 (rn50/export_onnx.py와 같은 약속).

[출력 파일] ($RN50_ONNX = $SIGLIP2_ARTIFACTS/onnx)
  siglip2_image_fp32.onnx     배치 크기 가변 (ORT 평가·양자화용. rn50 common.embed가 배치로 넣음)
  siglip2_image_fp32_b1.onnx  입력 1×3×224×224 고정 (DFC parse용, ROADMAP G1)
  두 파일은 배치 축만 다르다. --verify가 두 파일 모두 PyTorch와 비교한다.

[변환 조건]  rn50/export_onnx.py와 같게: torch.onnx.export(dynamo=False), opset 17, 입력 이름 image, 출력 이름 embedding
  timm fused attention은 끄고 내보낸다(기본, D-019). 켜고 내보내려면 --fused-attn.

[--verify]  CIFAR-100 dev A 앞 N장(SigLIP2 전처리)으로 PyTorch vs ONNX(ORT_ENABLE_ALL) 비교
  기준: 정규화 임베딩 코사인 최소값 ≥ 0.9999 (ROADMAP G1). 최대 절대 오차(정규화 전·후)도 기록.

[--ops]  ONNX 연산 종류별 개수와 GELU 형태(Erf / Tanh 근사), LayerNorm·Softmax·pooling head 구조를 JSON으로 저장.

[사용]
  "$SIGLIP2_PYTHON" s2_export_onnx.py --verify 64 --ops --out ../results/siglip2/G1_onnx_export
"""
import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC
PREFIX = "siglip2"
FP32_IMAGE = f"{PREFIX}_image_fp32.onnx"
FP32_IMAGE_B1 = f"{PREFIX}_image_fp32_b1.onnx"


def unfuse_attention(model):
    """timm의 fused attention(F.scaled_dot_product_attention)을 끄고 일반 연산(MatMul·Softmax)으로 계산하게 한다.
    fused 상태로 내보내면 배치 1 고정 그래프에도 Shape·Sqrt·Where 같은 동적 연산이 남아 DFC 파싱에 불리하다.
    계산은 같다 (--verify의 코사인으로 확인). 바꾼 모듈 수를 돌려준다."""
    n = 0
    for m in model.visual.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            n += 1
    assert n > 0, "fused_attn 속성을 가진 모듈이 없음 — timm 구조가 바뀌었는지 확인"
    return n


def export(model, path, dynamic, opset):
    import torch
    x = torch.randn(1, 3, 224, 224)
    kw = dict(dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}) if dynamic else {}
    with torch.inference_mode():
        torch.onnx.export(model.visual, (x,), str(path), input_names=["image"], output_names=["embedding"],
                          opset_version=opset, dynamo=False, **kw)
    print(f"saved {path}  {path.stat().st_size / 2**20:.1f} MiB")


def op_table(path):
    import onnx
    g = onnx.load(str(path), load_external_data=False).graph
    ops = collections.Counter(n.op_type for n in g.node)
    names = [n.name for n in g.node]
    inp = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input]
    out = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output]
    return {"file": path.name, "n_nodes": len(g.node), "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
            "inputs": inp, "outputs": out,
            "gelu_form": ("Erf (정확한 GELU)" if ops.get("Erf") else "Tanh 근사" if ops.get("Tanh") else
                          "Gelu 단일 연산" if ops.get("Gelu") else "없음"),
            "layernorm": "LayerNormalization 연산" if ops.get("LayerNormalization") else "분해된 형태(ReduceMean 등)",
            "pool_nodes": [n for n in names if "attn_pool" in n][:40],
            "n_pool_nodes": sum("attn_pool" in n for n in names)}


def cifar_dev(preprocess, n):
    import torch
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=False)
    s0 = RC.VAL_SPLITS["A"]
    x = torch.stack([preprocess(ds[i][0]) for i in range(s0, s0 + n)])
    assert x.shape == (n, 3, 224, 224)
    return x


def torch_embed(model, x):
    import torch
    with torch.inference_mode():
        return torch.cat([model.visual(x[i:i + 16]) for i in range(0, len(x), 16)]).float().numpy()


def fused_reference(model, preprocess, n):
    return RC.l2n(torch_embed(model, cifar_dev(preprocess, n)))


def verify(model, preprocess, paths, n, threads):
    x = cifar_dev(preprocess, n)
    ref_raw = torch_embed(model, x)
    ref = RC.l2n(ref_raw)
    res = {}
    for p in paths:
        s = RC.session(p, threads)                     # ORT_ENABLE_ALL (정수 커널 경로와 같은 기본 설정)
        bs = 1 if p.name == FP32_IMAGE_B1 else 16
        raw = np.concatenate([s.run(None, {"image": x[i:i + bs].numpy()})[0] for i in range(0, n, bs)])
        e = RC.l2n(raw)
        cos = (e * ref).sum(1)
        res[p.name] = {"n": n, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                       "max_abs_err_raw": float(np.abs(raw - ref_raw).max()),
                       "max_abs_err_normalized": float(np.abs(e - ref).max()),
                       "pass_cos_ge_0.9999": bool(cos.min() >= 0.9999)}
        print(p.name, json.dumps(res[p.name]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", type=int, default=0, help="CIFAR dev A 앞 N장으로 PyTorch vs ONNX 비교 (0이면 생략)")
    ap.add_argument("--ops", action="store_true", help="연산 목록 JSON 저장")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True, help="결과 폴더 (예: ../results/siglip2/G1_onnx_export)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fused-attn", action="store_true", help="timm fused attention을 그대로 내보냄 (기본: 끔, D-019)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    model, preprocess, _ = S.load_model()
    fused_ref = fused_reference(model, preprocess, a.verify) if a.verify else None   # fused 상태 PyTorch 출력 (D-019 확인용)
    n_unfused = 0 if a.fused_attn else unfuse_attention(model)
    print(f"fused attention 끈 모듈 수: {n_unfused}")
    RC.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    paths = [RC.ONNX_DIR / FP32_IMAGE, RC.ONNX_DIR / FP32_IMAGE_B1]
    t0 = time.time()
    reused = []
    for p, dyn in zip(paths, (True, False)):
        if p.exists() and not a.overwrite:
            import onnx
            file_opset = next(o.version for o in onnx.load(str(p), load_external_data=False).opset_import if o.domain in ("", "ai.onnx"))
            assert file_opset == a.opset and not a.fused_attn, \
                f"{p.name}(opset {file_opset})를 다른 설정으로 재사용할 수 없음 — --overwrite로 다시 만들기"
            print(f"{p.name} 이미 있음 — 재사용 (opset {file_opset}, --overwrite로 다시 만들기)")
            reused.append(p.name)
        else:
            export(model, p, dyn, a.opset)
    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
           "dirty": env["git"]["dirty"], "options": vars(a), "model": S.MODEL_ID, "opset": a.opset,
           "unfused_attention_modules": n_unfused, "reused_files": reused,
           "export_sec": round(time.time() - t0, 1),
           "files": {p.name: {"MiB": round(p.stat().st_size / 2**20, 2), "sha256": _sha256(p)} for p in paths}}
    if a.ops:
        row["ops"] = {p.name: op_table(p) for p in paths}
        (out / "onnx_ops.json").write_text(json.dumps(row["ops"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.verify:
        row["verify"] = verify(model, preprocess, paths, a.verify, a.threads)
        unf = RC.l2n(torch_embed(model, cifar_dev(preprocess, a.verify)))
        c = (unf * fused_ref).sum(1)
        row["verify"]["pytorch_fused_vs_unfused"] = {"n": a.verify, "cos_min": float(c.min()),
                                                     "max_abs_err_normalized": float(np.abs(unf - fused_ref).max())}
        print("pytorch fused vs unfused", json.dumps(row["verify"]["pytorch_fused_vs_unfused"]))
    with open(out / "export.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in row.items() if k != "ops"}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "ops"}, ensure_ascii=False, indent=1))
    if a.verify:
        assert all(v["pass_cos_ge_0.9999"] for v in row["verify"].values()), "PyTorch vs ONNX 코사인 < 0.9999"


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    main()
