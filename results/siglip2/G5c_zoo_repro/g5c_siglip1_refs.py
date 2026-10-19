"""g5c_siglip1_refs.py — SigLIP 1세대 B/16 기준값: (1) FP32 이미지 임베딩 = 공개 ONNX를 ORT FP32(ORT_ENABLE_ALL)로,
(2) 클래스 임베딩 = google/siglip-base-patch16-224 텍스트 인코더 FP32 (HF transformers). G5c 4-1.
- 사진: dev 캐시 6종 각 앞 K장(기본 200), SigLIP2용 전처리 후 정규화 값 그대로 (두 모델 모두 평균·표준편차 0.5).
- 클래스 이름·템플릿: s2_build_dev.py와 같은 출처 (CIFAR = torchvision CIFAR100.classes + RC.TEMPLATES, 고화질 = RD.classnames·cfg["templates"]).
- 프롬프트 앙상블: s2common.zeroshot_classifier와 같은 방식 (문장마다 정규화 → 클래스 평균 → 다시 정규화).
  토크나이저는 Zoo create_clip_vision_tfrecord.py와 같이 padding="max_length", max_length=64.
사용: "$SIGLIP2_PYTHON" g5c_siglip1_refs.py --onnx <공개 ONNX> --s2dir ~/oslab_m2rot/siglip2 --out <폴더>
출력: <set>_ref_s1.npy (K×768, 정규화 전), <set>_clf_s1.npy (768×클래스), refs_info.json
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True); ap.add_argument("--s2dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=200); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--model", default="google/siglip-base-patch16-224")
a = ap.parse_args()
sys.path.insert(0, a.s2dir)
import s2common as S   # noqa: E402
RC, RD = S.RC, S.RD
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
info = {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "onnx_sha256": hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest(), "k": a.k, "model": a.model, "sets": {}}

import onnxruntime as ort
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
info["ort"] = ort.__version__
t0 = time.time()
for d in DATA:
    x = np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[: a.k].astype(np.float32)
    e = np.concatenate([sess.run(["1725"], {"input.1": x[i:i + 1]})[0] for i in range(len(x))])
    np.save(out / f"{d}_ref_s1.npy", e.astype(np.float32))
    print(d, "ref", e.shape, f"{time.time() - t0:.0f}s", flush=True)

import torch
from transformers import AutoModel, AutoTokenizer
torch.set_grad_enabled(False)
model = AutoModel.from_pretrained(a.model).eval()
tok = AutoTokenizer.from_pretrained(a.model)
info["transformers"] = __import__("transformers").__version__


def clf(names, templates):
    W = []
    for c in names:
        sents = [t.format(c, c=c) for t in templates]
        inp = tok(sents, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
        e = model.get_text_features(input_ids=inp["input_ids"])
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        e = e.float().numpy(); e = (e / np.linalg.norm(e, axis=-1, keepdims=True)).mean(0)
        W.append(e / np.linalg.norm(e))
    return np.stack(W, 1).astype(np.float32)


for d in DATA:
    if d.startswith("cifar"):
        from torchvision.datasets import CIFAR100
        names, templates = [c.replace("_", " ") for c in CIFAR100(str(RC.DATA_ROOT), train=True, download=False).classes], RC.TEMPLATES
    else:
        name = d[:-4]
        names, templates = RD.classnames(name), RD.HIGHRES[name]["templates"]
    W = clf(names, templates)
    W2 = np.load(RC.CACHE_DIR / f"{d}_clf.npy")
    assert W.shape == W2.shape, (d, W.shape, W2.shape)
    np.save(out / f"{d}_clf_s1.npy", W)
    ref = np.load(out / f"{d}_ref_s1.npy"); y = np.load(RC.CACHE_DIR / f"{d}_y.npy")[: a.k]
    r = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
    info["sets"][d] = {"classes": int(W.shape[1]), "n_templates": len(templates), "fp32_top1": round(100 * float((np.argmax(r @ W, 1) == y).mean()), 2)}
    print(d, "clf", W.shape, info["sets"][d], f"{time.time() - t0:.0f}s", flush=True)
(out / "refs_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
print("DONE", json.dumps(info["sets"]))
