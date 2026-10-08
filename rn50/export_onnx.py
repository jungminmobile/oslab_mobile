"""
export_onnx.py — PyTorch CLIP-RN50 → ONNX(FP32) 변환 (+ AttnPool 그래프 최적화, 동일성 검증)
==========================================================================================

[목적]
  양자화 도구(ONNX Runtime)와 Hailo 컴파일러는 ONNX 파일을 입력으로 받는다.
  OpenAI 원본 가중치(RN50.pt)를 사진 인코더 / 글 인코더 두 파일로 나눠 내보낸다.
  출력은 정규화 전 1024차원 임베딩이며, L2 정규화와 유사도 계산은 바깥(호스트)에서 한다.

[--attnpool-q1 : 로드맵 1번 그래프 최적화]
  open_clip의 AttentionPool2d는 50칸(평균 칸 + 7×7) 모두를 질문(query)으로 계산한 뒤
  0번 칸 결과만 쓰고 49칸은 버린다. 처음부터 0번 칸만 query로 계산하도록 forward를 바꾼다.
  → 계산량 6.12G → 5.80 GMAC(−5%), 출력은 수학적으로 동일.
  --verify 를 주면 원본 FP32 ONNX와 출력을 비교해 동일한지 확인한다(최대 차이 ~1e-6이면 정상).

[출력 파일]
  onnx/rn50_image_fp32.onnx, onnx/rn50_text_fp32.onnx, (옵션) onnx/rn50_image_fp32_q1.onnx
"""
import argparse
import types

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def attnpool_q1_forward(self, x):
    """AttentionPool2d.forward 대체본. 원본과 다른 곳은 query=x[:1] 한 줄뿐."""
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW → (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)                              # 평균 칸을 맨 앞에 추가
    x = x + self.positional_embedding[:, None, :].to(x.dtype)                           # 위치 표시
    x, _ = F.multi_head_attention_forward(
        query=x[:1], key=x, value=x,                                                    # ← 0번 칸만 질문
        embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
        q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight,
        in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.,
        out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
        use_separate_proj_weight=True, training=False, need_weights=False)
    return x[0]


class TextEnc(torch.nn.Module):
    """model.encode_text를 독립 모듈로 감싸 글 인코더만 내보내기 위한 래퍼."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens):
        return self.m.encode_text(tokens)


def verify(a_path, b_path, n=200):
    """두 사진 인코더의 출력이 같은지 CIFAR 검증 A 앞 n장으로 확인."""
    ds, _ = C.cifar100(train=True, download=False)
    x = np.stack([ds[i][0].numpy() for i in range(C.VAL_SPLITS["A"], C.VAL_SPLITS["A"] + n)]).astype(np.float32)
    ya, yb = C.embed(a_path, x), C.embed(b_path, x)
    print(f"[verify] max|diff| of normalized emb = {np.abs(ya - yb).max():.2e}, min cos = {(ya * yb).sum(1).min():.7f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="OpenAI 가중치 경로 (RN50.pt sha256 afeb0e10...b6762). 모델 종류는 CLIP_MODEL 환경변수")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--attnpool-q1", action="store_true")
    ap.add_argument("--image-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    import open_clip
    C.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model = open_clip.load_openai_model(a.weights, precision="fp32", device="cpu").eval()
    suffix = ""
    if a.attnpool_q1:
        model.visual.attnpool.forward = types.MethodType(attnpool_q1_forward, model.visual.attnpool)
        suffix = "_q1"
    img = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    tok = C.tokenizer()(["a photo of a dog."])
    out = C.ONNX_DIR / f"{C.PREFIX}_image_fp32{suffix}.onnx"
    with torch.inference_mode():
        # dynamo=False: 기존 TorchScript 기반 내보내기(노드 이름 /attnpool/MatMul 등이 이 방식 기준)
        torch.onnx.export(model.visual, (img,), str(out), input_names=["image"], output_names=["embedding"],
                          dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=a.opset, dynamo=False)
        if not a.image_only:
            torch.onnx.export(TextEnc(model), (tok,), str(C.ONNX_DIR / C.FP32_TEXT), input_names=["tokens"],
                              output_names=["embedding"], dynamic_axes={"tokens": {0: "batch"}, "embedding": {0: "batch"}},
                              opset_version=a.opset, dynamo=False)
    print("saved", out)
    if a.verify and suffix:
        verify(C.FP32_IMAGE, out.name)


if __name__ == "__main__":
    main()
