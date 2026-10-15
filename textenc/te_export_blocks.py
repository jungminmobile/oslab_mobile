"""
te_export_blocks.py — Hailo DFC parse 시험용: SigLIP2 텍스트 인코더에서 "블록 12개"만 떼어 낸 FP32 ONNX (배치 1 고정)
=================================================================================================================

[왜]  전에 직접 만든 CLIP 텍스트 HEF는 Gather·ArgMax·Shuffle 미지원으로 실패했다.
  Hailo Model Zoo처럼 토큰 임베딩(Gather)·풀링·projection은 호스트에서, 트랜스포머 본체만 NPU에 두는 구성을 시험한다.
  te_export.py의 전체 그래프에는 nn.MultiheadAttention이 만든 동적 shape 연산(Shape·Gather·Reshape 수십 개)이 섞여 있어서,
  여기서는 같은 가중치로 attention을 고정 shape 연산(MatMul·Reshape·Transpose·Softmax)만으로 다시 쓴다.

[출력]  ($SIGLIP2_ARTIFACTS/textenc/)
  siglip2_text_blocks_b1.onnx       입력 x (1,64,768) = 토큰 임베딩 + 위치 임베딩 (호스트 계산)
                                    출력 h (1,64,768) = 블록 12개 → ln_final  (위치 63 선택·projection은 호스트)
  siglip2_text_blocks_pool_b1.onnx  같은 입력, 출력 (1,768) = 위치 63 Slice → projection까지 NPU (Slice가 받아들여지는지 시험)
  dfc_text_calib64_s0.npy           DFC 보정용 입력 x (64,1,64,768) — DFC NHWC 형식 = 보정 문장 앞 64개 (tokens_check_s0_64.npy)
  host_text_head.npz                호스트 쪽 가중치: 위치 임베딩, projection (임베딩 표는 FP32 ONNX/INT8 D 쪽을 쓴다)

[검증]  PyTorch encode_text와 "호스트 앞부분 + 블록 ONNX(ORT) + 호스트 뒷부분" 결과 cos ≥ 0.9999 (평가 프롬프트 + 확인 문장)

[사용]  python textenc/te_export_blocks.py --out results/textenc/hailo_parse
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_common as T   # noqa: E402

RC = T.RC
L, D, H = T.CTX, 768, 12


def build(model, pool):
    import torch
    import torch.nn.functional as F
    t = model.text

    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = t.transformer.resblocks
            self.ln_final = t.ln_final
            self.proj = t.text_projection

        def attn(self, blk, x):
            a = blk.attn
            qkv = F.linear(x, a.in_proj_weight, a.in_proj_bias)          # (1,64,2304)
            q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
            sh = lambda z: z.reshape(1, L, H, D // H).transpose(1, 2)    # (1,12,64,64)
            q, k, v = sh(q) * (D // H) ** -0.5, sh(k), sh(v)
            o = torch.softmax(q @ k.transpose(-1, -2), -1) @ v
            return a.out_proj(o.transpose(1, 2).reshape(1, L, D))

        def forward(self, x):
            for blk in self.blocks:
                x = x + self.attn(blk, blk.ln_1(x))
                x = x + blk.mlp(blk.ln_2(x))
            x = self.ln_final(x)
            if pool:
                return self.proj(x[:, L - 1:L].reshape(1, D))              # 위치 63 (pool_type last). Gather 대신 Slice
            return x
    return Blocks().eval()


def drop_float_casts(path):
    """open_clip LayerNorm의 float→float Cast(값이 그대로)를 지운다. DFC가 Cast를 받지 않을 수 있어서. 지운 개수 반환."""
    import onnx
    m = onnx.load(str(path))
    g = m.graph
    outs = {o.name for o in g.output}
    casts = [n for n in g.node if n.op_type == "Cast"
             and next(a.i for a in n.attribute if a.name == "to") == onnx.TensorProto.FLOAT]
    rename = {}
    for n in casts:
        src, dst = n.input[0], n.output[0]
        if dst in outs:   # 그래프 출력인 Cast: 앞 노드의 출력 이름을 그래프 출력 이름으로 바꾼다
            rename[src] = dst
        else:
            rename[dst] = src
    keep = [n for n in g.node if n not in casts]
    def r(x):
        while x in rename:
            x = rename[x]
        return x
    for n in keep:
        for i, x in enumerate(n.input):
            n.input[i] = r(x)
        for i, x in enumerate(n.output):
            if x in rename and rename[x] in outs:
                n.output[i] = rename[x]
    del g.node[:]
    g.node.extend(keep)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return len(casts)


def host_front(model, tokens):
    t = model.text
    return (t.token_embedding.weight[tokens] + t.positional_embedding).detach().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import onnx
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    env = T.env_row()
    T.env_info.write_env(out, env)
    model, _, _ = T.load_model()
    t = model.text
    assert t.attn_mask is None and t.pool_type == "last" and not t.use_pad_mask and t.cls_emb is None
    files = {False: T.TE_DIR / "siglip2_text_blocks_b1.onnx", True: T.TE_DIR / "siglip2_text_blocks_pool_b1.onnx"}
    tok = {k: np.load(T.TE_DIR / f) for k, f in (("pets", "tokens_pets.npy"), ("food", "tokens_food.npy"),
                                                  ("holdout", "tokens_holdout_s1_64.npy"))}
    row = {**T.base_row(env, vars(a)), "step": "export_blocks", "files": {}, "verify": {}}
    for pool, f in files.items():
        m = build(model, pool)
        x0 = torch.as_tensor(host_front(model, torch.as_tensor(tok["pets"][:1])))
        t0 = time.time()
        with torch.inference_mode():
            torch.onnx.export(m, (x0,), str(f), input_names=["x"], output_names=["h"], opset_version=a.opset, dynamo=False)
        n_cast = drop_float_casts(f)
        g = onnx.load(str(f), load_external_data=False).graph
        ops = collections.Counter(n.op_type for n in g.node)
        row["files"][f.name] = {"MiB": round(f.stat().st_size / 2**20, 2), "sha256": T.sha256(f), "ops": dict(ops), "float_casts_removed": n_cast,
                                "inputs": [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.input],
                                "outputs": [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in g.output]}
        assert not {"Shape", "ConstantOfShape", "Where", "ArgMax", "Gather", "Cast"} & set(ops), f"동적 연산 남음: {ops}"
        print(f"saved {f.name} {row['files'][f.name]['MiB']} MiB ({time.time() - t0:.0f}s) ops={dict(ops)}")
        s = RC.session(f, a.threads)
        for k, tk in tok.items():
            with torch.inference_mode():
                ref = torch.cat([model.encode_text(torch.as_tensor(tk[i:i + 32])) for i in range(0, len(tk), 32)]).numpy()
                got = []
                for i in range(len(tk)):
                    h = s.run(None, {"x": host_front(model, torch.as_tensor(tk[i:i + 1]))})[0]
                    got.append(h[0] if pool else t.text_projection(torch.as_tensor(h[:, L - 1])).numpy()[0])
            c = (RC.l2n(ref) * RC.l2n(np.stack(got))).sum(1)
            row["verify"][f"{f.name}:{k}"] = {"n": len(tk), "cos_min": float(c.min())}
    print(json.dumps(row["verify"]))
    with torch.inference_mode():
        cal = np.concatenate([host_front(model, torch.as_tensor(tk1[None])) for tk1 in np.load(T.TE_DIR / "tokens_check_s0_64.npy")])
    cal = cal[:, None].astype(np.float32)   # DFC는 (N,H,W,C)로 읽는다: 입력 (1,64,768) → (1,1,64,768). 3차원이면 Slice 층에서 IndexError
    np.save(T.TE_DIR / "dfc_text_calib64_s0.npy", cal)
    np.savez(T.TE_DIR / "host_text_head.npz", positional_embedding=t.positional_embedding.detach().numpy(),
             proj_w=t.text_projection.weight.detach().numpy(), proj_b=t.text_projection.bias.detach().numpy())
    row["calib"] = {"file": "dfc_text_calib64_s0.npy", "shape": list(cal.shape)}
    T.append_jsonl(out / "export_blocks.jsonl", row)
    assert all(v["cos_min"] >= 0.9999 for v in row["verify"].values()), "블록 ONNX 경로 cos < 0.9999"


if __name__ == "__main__":
    main()
