"""
te_dfc_evalset.py — DFC 에뮬레이션 정확도용 입력 묶음 만들기 (torch 환경, x86)
========================================================================

블록 ONNX(te_export_blocks.py)는 입력이 "토큰 임베딩 + 위치 임베딩"이라, DFC 쪽에서 토크나이저·임베딩 표 없이 돌리려면
입력을 미리 계산해 넘겨야 한다. 데이터셋마다 npz 하나 (30 MiB 안쪽, 대화로 보낼 수 있게):
  x        (N,1,64,768)  DFC NHWC 입력 (평가 프롬프트, 템플릿 1개라 N = 클래스 수)
  W0       (768,C)       FP32 ONNX 텍스트 분류기 (te_export.py가 저장한 clf_<d>_fp32_ort.npy)
  y, img   dev 정답·FP32 이미지 임베딩 (팀 dev 캐시)
  proj_w, proj_b         호스트 쪽 projection (위치 63 → 768)
확인용 문장 64개는 holdout.npz (x, ref = FP32 텍스트 임베딩 정규화 전).

[사용]  python textenc/te_dfc_evalset.py   → $SIGLIP2_ARTIFACTS/textenc/dfc_eval_{pets,food,holdout}.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_common as T                    # noqa: E402
from te_export_blocks import host_front  # noqa: E402


def main():
    import torch
    model, _, _ = T.load_model()
    t = model.text
    head = {"proj_w": t.text_projection.weight.detach().numpy(), "proj_b": t.text_projection.bias.detach().numpy()}
    with torch.inference_mode():
        for d in T.EVAL_SETS:
            tk = np.load(T.TE_DIR / f"tokens_{d}.npy")
            x = np.concatenate([host_front(model, torch.as_tensor(r[None])) for r in tk])[:, None].astype(np.float32)
            y, _, img = T.load_dev(d)
            W0 = np.load(T.TE_DIR / f"clf_{d}_fp32_ort.npy")
            assert len(tk) == W0.shape[1], "템플릿 1개 가정 (문장 수 = 클래스 수)"
            f = T.TE_DIR / f"dfc_eval_{d}.npz"
            np.savez(f, x=x, W0=W0, y=y, img=img.astype(np.float32), **head)
            print(f.name, x.shape, f"{f.stat().st_size / 2**20:.1f} MiB")
        tk = np.load(T.TE_DIR / "tokens_holdout_s1_64.npy")
        x = np.concatenate([host_front(model, torch.as_tensor(r[None])) for r in tk])[:, None].astype(np.float32)
        ref = model.encode_text(torch.as_tensor(tk)).numpy().astype(np.float32)
        f = T.TE_DIR / "dfc_eval_holdout.npz"
        np.savez(f, x=x, ref=ref, **head)
        print(f.name, x.shape, f"{f.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
