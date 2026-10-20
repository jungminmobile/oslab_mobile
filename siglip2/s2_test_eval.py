"""
s2_test_eval.py — SigLIP2 최종 test 평가 (G6, test는 최종 1회만)
====================================================================

[목적] 최종 동결 모델(D-030 ORT fc2L2 U8 + reduce_range)과 FP32 베이스라인을 test 7종으로 한 번만 잰다 (PRD 5절, CLAUDE.md 3절).
  사진 선택·클래스 이름·프롬프트는 rn50/datasets.py를 import해 RN50 트랙과 같게 쓰고(D-009),
  전처리·텍스트 분류기는 dev 캐시(s2_build_dev.py)와 같은 SigLIP2 것을 쓴다. 정확도는 정수 커널(ORT_ENABLE_ALL)로만 (D-008).
  rn50 datasets.eval_test는 CLIP 전처리·RN50 텍스트 인코더·fake-quant 실행이라 쓰지 않는다.
[test 7종] cifar_test(CIFAR-100 test 1만), imagenetv2(matched-frequency 1만), pets·flowers·cars·food·caltech 공식 test 전체
[1회 보장] 데이터셋마다 $RN50_RESULTS/done_<set>.json을 남기고, 있으면 다시 돌리지 않는다 (--force 금지 — 사람 확인 뒤 파일을 직접 지워야 함)
[출력] $RN50_RESULTS/test.jsonl (데이터셋·모델별 top-1·top-5·FP32 cos, 커밋·dirty·환경),
       test_<set>.npz (y, <모델>_pred, <모델>_cos — 짝 부트스트랩용), done_<set>.json
[사용]
  RN50_RESULTS=../results/siglip2/G6_test "$SIGLIP2_PYTHON" s2_test_eval.py pets --models <U8rr onnx> --threads 4
"""
import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S                      # noqa: E402  (RN50_* 경로 설정 + 확인)
from s2_export_onnx import FP32_IMAGE     # noqa: E402
from scripts import env_info              # noqa: E402
import datasets as RD                     # noqa: E402  rn50/datasets.py

RC = S.RC
SETS = ["cifar_test", "imagenetv2", "pets", "flowers", "cars", "food", "caltech"]


def rows_names_templates(name):
    """(이미지 PIL을 내는 반복자, 클래스 이름, 템플릿) — rn50 datasets.eval_test와 같은 사진·이름·템플릿."""
    if name == "cifar_test":
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(str(RC.DATA_ROOT), train=False, download=True)
        rows = ((ds[i][0], ds.targets[i]) for i in range(len(ds)))
        return rows, [c.replace("_", " ") for c in ds.classes], RC.TEMPLATES
    if name == "imagenetv2":
        names, templates = RD.imagenet_meta()
        return ((Image.open(io.BytesIO(b)), lab) for b, lab in RD.imagenetv2_rows()), names, templates
    cfg = RD.HIGHRES[name]
    rows = ((Image.open(io.BytesIO(b)), lab) for b, lab in RD.parquet_rows(cfg["test"], keep_downloads=True))
    return rows, RD.classnames(name), cfg["templates"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=SETS)
    ap.add_argument("--models", nargs="+", required=True, help="양자화 ONNX 파일 이름 (FP32는 항상 함께)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=50)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"done_{a.name}.json"
    if done.exists():
        raise SystemExit(f"{a.name} test는 이미 평가했다 ({done}). test는 1회만 쓴다 — 다시 하려면 사람 확인 뒤 파일을 직접 지울 것")
    env = env_info.collect()
    env_info.write_env(out, env)
    t0 = time.time()
    model, preprocess, tok = S.load_model()
    rows, names, templates = rows_names_templates(a.name)
    W = S.zeroshot_classifier(model, tok, names, templates)
    del model
    models = ["fp32"] + a.models
    sess = {m: RC.session(RC.onnx_path(FP32_IMAGE if m == "fp32" else m), a.threads, fake_quant=False) for m in models}
    emb = {m: [] for m in models}
    ys, buf = [], []

    def flush():
        x = np.ascontiguousarray(np.stack(buf).astype(np.float32))
        assert x.shape[1:] == (3, 224, 224) and x.min() >= -1.0001 and x.max() <= 1.0001, (x.shape, x.min(), x.max())
        for m, s in sess.items():
            emb[m].append(s.run(None, {"image": x})[0])
        buf.clear()

    for i, (img, lab) in enumerate(rows):
        buf.append(preprocess(img.convert("RGB")).numpy())
        ys.append(lab)
        if len(buf) == a.bs:
            flush()
        if i % 2000 == 0:
            print(f"  {a.name} {i}  {time.time() - t0:.0f}s", flush=True)
    if buf:
        flush()
    y = np.array(ys)
    E = {m: RC.l2n(np.concatenate(v).astype(np.float32)) for m, v in emb.items()}
    ref = E["fp32"]
    npz = {"y": y}
    with open(out / "test.jsonl", "a", encoding="utf-8") as f:
        for m in models:
            met = RC.topk_metrics(E[m], W, y, ref)
            pred = np.argmax(E[m] @ W, 1)
            key = "fp32" if m == "fp32" else Path(m).stem
            npz[f"{key}_pred"], npz[f"{key}_cos"] = pred, (E[m] * ref).sum(1)
            row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
                   "data": a.name, "model": "fp32" if m == "fp32" else Path(m).name, "n": int(len(y)), "classes": len(names),
                   "n_templates": len(templates), "exec": "int", "threads": a.threads, **met, **env_info.flat(env),
                   "sec": round(time.time() - t0)}
            print(json.dumps({k: row[k] for k in ("data", "model", "n", "top1", "top5", "cos_mean")}, ensure_ascii=False), flush=True)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(out / f"test_{a.name}.npz", **npz)
    done.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                                "models": models, "n": int(len(y))}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
