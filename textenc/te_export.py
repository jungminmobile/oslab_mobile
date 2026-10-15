"""
te_export.py — SigLIP2 텍스트 인코더 FP32 ONNX export + 구조 확인 + PyTorch 대비 검증 (스모크 1·0단계)
====================================================================================================

[하는 일]
  0. 구조 확인: 파라미터 분포(임베딩 표 / 블록 / 헤드)를 실제 모델에서 세고 기대값과 assert,
     토크나이저 종류·padding_side·특수 토큰 id, 평가 프롬프트의 실제 길이, 풀링 위치(63)가 pad인지 기록
  1. export: tokens(int64, B×64) → embedding(정규화 전 768), opset 17, dynamo=False, 배치 가변
  2. 검증: 평가 프롬프트(Pets·Food 전부) + 보정 문장 64개로 PyTorch vs ORT(ORT_ENABLE_ALL) cos ≥ 0.9999,
     팀 dev 캐시의 FP32 분류기(<이름>_dev_clf.npy, PyTorch로 계산됨)와 ORT 분류기 비교
  3. 토큰 .npy 저장: 평가 프롬프트, 보정 문장(시드 0, 256개), 점검용 앞 64개, 보정과 겹치지 않는 확인용 문장(시드 1, 64개)

[사용]
  python textenc/te_export.py --out results/textenc/smoke
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


def param_split(model):
    t = model.text
    n = lambda m: sum(p.numel() for p in m.parameters())
    r = {"text": n(t), "token_embedding": t.token_embedding.weight.numel(), "blocks": n(t.transformer),
         "proj": n(t.text_projection), "ln_final": n(t.ln_final), "pos": t.positional_embedding.numel(),
         "image": n(model.visual)}
    for k, v in T.EXPECTED_PARAMS.items():
        assert r[k] == v, f"{k}: {r[k]} ≠ 기대 {v} — 모델 구조가 open_clip 3.3.0 설정과 다름"
    r["share_pct"] = {k: round(100 * r[k] / r["text"], 2) for k in ("token_embedding", "blocks", "proj")}
    r["MiB_fp32"] = {k: round(r[k] * 4 / 2**20, 1) for k in ("text", "token_embedding", "blocks", "proj")}
    return r


def tokenizer_info(tok, toks_eval):
    hf = getattr(tok, "tokenizer", None)
    info = {"wrapper": type(tok).__name__, "hf_class": type(hf).__name__ if hf else None,
            "vocab_size": getattr(hf, "vocab_size", None), "padding_side": getattr(hf, "padding_side", None),
            "pad_id": getattr(hf, "pad_token_id", None), "bos_id": getattr(hf, "bos_token_id", None),
            "eos_id": getattr(hf, "eos_token_id", None)}
    pad = info["pad_id"] if info["pad_id"] is not None else 0
    for name, tk in toks_eval.items():
        L = (tk != pad).sum(1)
        info[name] = {"n": len(tk), "len_min": int(L.min()), "len_max": int(L.max()), "len_mean": round(float(L.mean()), 1),
                      "pos63_is_pad_frac": round(float((tk[:, -1] == pad).mean()), 3),
                      "first_ids": tk[0, :12].tolist(), "last_ids": tk[0, -4:].tolist()}
    return info


class TextEnc:
    """tokens → model.encode_text(tokens) (정규화 전). torch.nn.Module로 감싸 export한다."""
    @staticmethod
    def build(model):
        import torch

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, tokens):
                return self.model.encode_text(tokens)
        return M().eval()


def op_table(path):
    import onnx
    g = onnx.load(str(path), load_external_data=False).graph
    ops = collections.Counter(n.op_type for n in g.node)
    return {"n_nodes": len(g.node), "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--n-check", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    env = T.env_row()
    T.env_info.write_env(out, env)
    T.TE_DIR.mkdir(parents=True, exist_ok=True)

    model, _, tok = T.load_model()
    params = param_split(model)
    print("params", json.dumps(params))

    # 평가 프롬프트 · 보정 문장 토큰
    toks, prompt_meta, names_all = {}, {}, []
    for d in T.EVAL_SETS:
        names, templates, sents = T.eval_prompts(d)
        toks[d] = T.tokenize(tok, sents)
        prompt_meta[d] = {"classes": len(names), "templates": templates}
        names_all += names
        np.save(T.TE_DIR / f"tokens_{d}.npy", toks[d])
    cal, cal_meta = T.calib_sentences(a.n_calib, 0, names_all)
    hold, _ = T.calib_sentences(a.n_calib + a.n_check, 1, names_all)
    hold = [s for s in hold if s not in set(cal)][:a.n_check]          # 보정 문장과 겹치지 않는 확인용
    assert len(hold) == a.n_check
    tk_cal, tk_hold = T.tokenize(tok, cal), T.tokenize(tok, hold)
    np.save(T.TE_DIR / f"tokens_calib_s0_{a.n_calib}.npy", tk_cal)
    np.save(T.TE_DIR / f"tokens_check_s0_{a.n_check}.npy", tk_cal[:a.n_check])
    np.save(T.TE_DIR / f"tokens_holdout_s1_{a.n_check}.npy", tk_hold)
    (out / "calib_sentences_s0.txt").write_text("\n".join(cal) + "\n", encoding="utf-8")
    (out / "holdout_sentences_s1.txt").write_text("\n".join(hold) + "\n", encoding="utf-8")
    tinfo = tokenizer_info(tok, {**toks, "calib": tk_cal, "holdout": tk_hold})
    print("tokenizer", json.dumps(tinfo, ensure_ascii=False))

    # export
    dst = T.TE_DIR / T.FP32
    t0 = time.time()
    if dst.exists() and not a.overwrite:
        print(f"{dst.name} 이미 있음 — 재사용 (--overwrite로 다시 만들기)")
    else:
        enc = TextEnc.build(model)
        # nn.MultiheadAttention의 추론 fast path(aten::_native_multi_head_attention)는 ONNX로 못 내보낸다 → 일반 경로로 (계산 같음, 아래 cos로 확인)
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            torch.onnx.export(enc, (torch.as_tensor(toks["pets"][:2]),), str(dst), input_names=["tokens"],
                              output_names=["embedding"], dynamic_axes={"tokens": {0: "batch"}, "embedding": {0: "batch"}},
                              opset_version=a.opset, dynamo=False)
        print(f"saved {dst}  {dst.stat().st_size / 2**20:.1f} MiB ({time.time() - t0:.0f}s)")

    # 검증: PyTorch vs ORT
    sess = RC.session(dst, a.threads)
    verify = {}
    for k, tk in {**toks, "check": tk_cal[:a.n_check], "holdout": tk_hold}.items():
        with torch.inference_mode():
            ref = torch.cat([model.encode_text(torch.as_tensor(tk[i:i + 32])) for i in range(0, len(tk), 32)]).float().numpy()
        got = T.encode(sess, tk)
        c = (RC.l2n(ref) * RC.l2n(got)).sum(1)
        verify[k] = {"n": len(tk), "cos_min": float(c.min()), "max_abs_err_raw": float(np.abs(ref - got).max()),
                     "pass_cos_ge_0.9999": bool(c.min() >= 0.9999)}
    # 팀 dev 캐시 분류기(PyTorch FP32)와 ORT FP32 분류기
    for d in T.EVAL_SETS:
        W_ort = T.classifier(T.encode(sess, toks[d]), prompt_meta[d]["classes"], len(prompt_meta[d]["templates"]))
        f = RC.CACHE_DIR / f"{d}_dev_clf.npy"
        if f.exists():
            Wc = np.load(f)
            c = (Wc * W_ort).sum(0)
            verify[f"{d}_clf_vs_team_cache"] = {"cos_min": float(c.min()), "shape": list(Wc.shape)}
        np.save(T.TE_DIR / f"clf_{d}_fp32_ort.npy", W_ort)
    print("verify", json.dumps(verify))

    row = {**T.base_row(env, vars(a)), "step": "export", "model": T.S.MODEL_ID, "params": params,
           "tokenizer": tinfo, "prompts": prompt_meta, "calib": {"n": a.n_calib, "seed": 0, **cal_meta},
           "file": {"name": dst.name, "MiB": round(dst.stat().st_size / 2**20, 2), "sha256": T.sha256(dst)},
           "ops": op_table(dst), "verify": verify}
    T.append_jsonl(out / "export.jsonl", row)
    assert all(v["pass_cos_ge_0.9999"] for k, v in verify.items() if "pass_cos_ge_0.9999" in v), "PyTorch vs ONNX cos < 0.9999"


if __name__ == "__main__":
    main()
