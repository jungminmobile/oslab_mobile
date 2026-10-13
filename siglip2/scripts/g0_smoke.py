"""
g0_smoke.py — G0 완료 기준 확인: SigLIP2 FP32(PyTorch) 이미지 임베딩 + CIFAR-100 dev 앞 N장 zero-shot
=================================================================================================

[목적]  (ROADMAP G0 완료 기준)
  SigLIP2를 불러와 이미지 한 장의 임베딩이 나오고, PyTorch FP32 zero-shot이 CIFAR-100 dev(A 구간) 앞 N장에서 도는지 확인한다.
  정확도 비교용 수치가 아니다 (100장은 표본이 작다). 파이프라인 점검용이다.

[같은 조건]
  - 사진: CIFAR-100 train[45000:45000+N] = RN50과 같은 dev A 구간 (rn50/common.py VAL_SPLITS)
  - 프롬프트: CIFAR-100 18개 템플릿 (rn50/prompts), 클래스 이름 '_' → 공백 (rn50 common.cifar100과 같음)
  - 전처리만 SigLIP2 모델 설정을 따른다 (s2common.load_model)

[확인 (VERIFICATION.md)]
  1. 자기 검증: 임베딩 shape·길이 1·유한값, 사진 수·클래스 수 assert, 같은 사진을 두 번 넣어 비트 단위 일치
  2. 독립 재계산: ① top-1을 topk_metrics와 다른 방법(직접 argmax 비교)으로 다시 계산
                 ② SigLIP 점수 sigmoid(scale·cos + bias)의 argmax가 코사인 argmax와 같은지 (CLAUDE.md 4절의 가정)

[사용]
  "$SIGLIP2_PYTHON" scripts/g0_smoke.py --n 100 --threads 4 --out ../results/siglip2/G0_smoke
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s2common as S       # noqa: E402  (RN50_* 경로 설정 + rn50 공통 코드 import)
from scripts import env_info  # noqa: E402

RC = S.RC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True, help="결과 폴더 (예: ../results/siglip2/G0_smoke)")
    a = ap.parse_args()
    import torch
    from torchvision.datasets import CIFAR100
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)

    t0 = time.time()
    model, preprocess, tok = S.load_model()
    n_img = sum(p.numel() for p in model.visual.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"model loaded {time.time() - t0:.0f}s  image params {n_img / 1e6:.2f}M / total {n_all / 1e6:.2f}M")
    print("preprocess:", preprocess)

    # 1) 이미지 한 장
    ds = CIFAR100(str(RC.DATA_ROOT), train=True, download=True)
    s0 = RC.VAL_SPLITS["A"]
    x1 = preprocess(ds[s0][0]).unsqueeze(0)
    e1 = S.embed_torch(model, x1)
    D = e1.shape[1]
    assert e1.shape == (1, D) and np.isfinite(e1).all() and abs(np.linalg.norm(e1) - 1) < 1e-5, e1.shape
    print(f"one image: embedding shape {e1.shape}, dim {D}")

    # 2) dev A 앞 N장 zero-shot
    idx = range(s0, s0 + a.n)
    assert s0 + a.n <= s0 + RC.VAL_SIZE, "dev A 구간(1000장)을 넘음"
    x = torch.stack([preprocess(ds[i][0]) for i in idx])
    y = np.array([ds.targets[i] for i in idx])
    names = [c.replace("_", " ") for c in ds.classes]
    assert len(x) == a.n and len(names) == 100 and len(RC.TEMPLATES) == 18
    t1 = time.time()
    W = S.zeroshot_classifier(model, tok, names, RC.TEMPLATES)
    t_clf = time.time() - t1
    assert W.shape == (D, 100) and np.allclose(np.linalg.norm(W, axis=0), 1, atol=1e-5)
    t1 = time.time()
    feat = S.embed_torch(model, x)
    t_emb = time.time() - t1
    feat2 = S.embed_torch(model, x)
    deterministic = bool(np.array_equal(feat, feat2))
    m = RC.topk_metrics(feat, W, y)

    # 독립 재계산 ① top-1 직접 계산
    top1_direct = round(100 * float(np.mean(np.argmax(feat @ W, 1) == y)), 2)
    # 독립 재계산 ② SigLIP 점수(sigmoid(scale·cos + bias))의 argmax = 코사인 argmax
    scale, bias = float(model.logit_scale.exp()), float(model.logit_bias)
    prob = 1 / (1 + np.exp(-(scale * (feat @ W) + bias)))
    rank_same = bool(np.array_equal(np.argmax(prob, 1), np.argmax(feat @ W, 1)))

    row = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
           "dirty": env["git"]["dirty"], "options": vars(a), "model": S.MODEL_ID, "data": f"cifar_A[0:{a.n}]",
           "n": int(len(y)), **m, "exec": "torch_fp32", "qtype": None, "calib": None,
           "cpu": env["cpu"]["model"], "vnni": env["cpu"]["flags"]["avx512_vnni"] or env["cpu"]["flags"]["avx_vnni"],
           "ort": (env["onnxruntime"] or {}).get("version"), "os": env["os"], "threads": a.threads,
           "embed_dim": D, "image_params_M": round(n_img / 1e6, 3), "total_params_M": round(n_all / 1e6, 3),
           "preprocess_cfg": S.preprocess_cfg(model), "logit_scale": round(scale, 4), "logit_bias": round(bias, 4),
           "checks": {"top1_direct": top1_direct, "top1_match": top1_direct == m["top1"],
                      "sigmoid_rank_equals_cos_rank": rank_same, "embedding_deterministic": deterministic},
           "sec": {"classifier": round(t_clf, 1), "embed": round(t_emb, 1)}}
    print(json.dumps(row, ensure_ascii=False, indent=1))
    with open(out / "g0_smoke.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert row["checks"]["top1_match"] and rank_same and deterministic, row["checks"]


if __name__ == "__main__":
    main()
