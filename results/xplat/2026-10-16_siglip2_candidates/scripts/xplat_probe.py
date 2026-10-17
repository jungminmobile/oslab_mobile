#!/usr/bin/env python3
"""
xplat_probe.py — 이미지 인코더 INT8 이득이 플랫폼(x86 / ARM)마다 다른가: 속도 파일럿

  build : (torch 필요, Orin에서 1번만) FP32 ONNX export + 팀 구성과 같은 모양의 정적 QDQ INT8 모델 생성
  bench : (onnxruntime·numpy만 필요) models/ 안의 모든 .onnx를 같은 조건으로 측정 → 어느 PC에서나 같은 파일로 실행

※ 파일럿이다. 보정은 랜덤 사진 16장이라 정확도는 의미가 없고(cos는 참고만), 속도만 본다.
   정적 QDQ 그래프 모양·정수 커널 선택은 보정값과 무관하므로 속도 비교에는 충분하다.
   본 측정은 정민(c3·c4)·준혁(fc2L2) 실제 파일과 dev 사진으로 다시 한다.

구성 (팀 PRD/STATUS 이름과 맞춤)
  RN50  : fp32(원본), fp32_q1(AttnPool 최적화), c3·c4·Fon(q1 기반) × 활성값 {u8, s8}
  SigLIP2 (--siglip2): fp32, mm_all(MatMul 전부 INT8), fc2L2(블록 5·9 fc2만 FP32) × {u8, s8}

사용
  python3 xplat_probe.py build --out ~/ssd/junyong/xplat [--siglip2]
  python3 xplat_probe.py bench --out ~/ssd/junyong/xplat --label orin_a78ae [--threads 1,2,4]
"""
import argparse, json, os, platform, statistics, subprocess, time, types
from pathlib import Path

import numpy as np

ATTN_PROJ = ["/attnpool/MatMul", "/attnpool/MatMul_1", "/attnpool/MatMul_2", "/attnpool/Gemm"]


# ----------------------------------------------------------------------------- build
def attnpool_q1_forward(self, x):
    """정민 export_onnx.py의 --attnpool-q1과 같은 forward (query = 0번 칸만)."""
    import torch, torch.nn.functional as F
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
    x = x + self.positional_embedding[:, None, :].to(x.dtype)
    x, _ = F.multi_head_attention_forward(
        query=x[:1], key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
        q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight,
        in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.,
        out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
        use_separate_proj_weight=True, training=False, need_weights=False)
    return x[0]


def export(module, path, size=224):
    import torch
    x = torch.randn(1, 3, size, size)
    kw = dict(input_names=["image"], output_names=["embedding"],
              dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    with torch.inference_mode():
        try:
            torch.onnx.export(module, (x,), str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(module, (x,), str(path), **kw)


class RandCalib:
    """랜덤 사진 16장 (정규화 공간). 속도 파일럿 전용."""
    def __init__(self, size=224, n=16, bs=8):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, 3, size, size)).astype(np.float32)
        self.b = [x[i:i + bs] for i in range(0, n, bs)]
        self.it = iter(self.b)

    def get_next(self):
        v = next(self.it, None)
        return None if v is None else {"image": v}

    def rewind(self):
        self.it = iter(self.b)


def node_names(path):
    import onnx
    return [n for n in onnx.load(str(path), load_external_data=False).graph.node]


def quant(pre, dst, act, excl, op_types=None):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rc = RandCalib()

        def get_next(self):
            return self.rc.get_next()

        def rewind(self):
            self.rc.rewind()

    reader = Reader()
    quantize_static(str(pre), str(dst), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8 if act == "u8" else QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=sorted(excl),
                    op_types_to_quantize=op_types,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    print(f"  {dst.name}: {dst.stat().st_size / 2**20:.1f} MiB, FP32 유지 노드 {len(excl)}", flush=True)


def build_rn50(md):
    import torch, clip
    from onnxruntime.quantization.shape_inference import quant_pre_process
    model, _ = clip.load("RN50", device="cpu", jit=False)
    vis = model.float().eval().visual
    export(vis, md / "rn50_fp32.onnx")
    vis.attnpool.forward = types.MethodType(attnpool_q1_forward, vis.attnpool)
    export(vis, md / "rn50_fp32_q1.onnx")
    pre = md.parent / "work" / "rn50_fp32_q1_pre.onnx"
    pre.parent.mkdir(exist_ok=True)
    quant_pre_process(str(md / "rn50_fp32_q1.onnx"), str(pre))
    names = [n.name for n in node_names(pre)]
    assert any(n in names for n in ATTN_PROJ), "AttnPool 투영 노드 이름이 예상과 다름"

    def only(prefixes):   # 이 접두사 + AttnPool 투영만 INT8, 나머지 FP32
        return {n for n in names if not (n.startswith(tuple(prefixes)) or n in ATTN_PROJ)}

    cfgs = {
        "c3": only(["/layer3", "/layer4"]),                       # stem·layer1·2 FP32
        "c4": only(["/layer2", "/layer3", "/layer4"]),            # stem·layer1 FP32
        "Fon": {n for n in names if n.startswith("/attnpool") and n not in ATTN_PROJ},  # attention 계산만 FP32
    }
    for c, excl in cfgs.items():
        for act in ("u8", "s8"):
            quant(pre, md / f"rn50_q1_int8_{c}_{act}.onnx", act, excl)


def build_siglip2(md):
    import torch, open_clip
    from onnxruntime.quantization.shape_inference import quant_pre_process
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2")
    vis = m.float().eval().visual
    export(vis, md / "siglip2_fp32.onnx")
    pre = md.parent / "work" / "siglip2_fp32_pre.onnx"
    pre.parent.mkdir(exist_ok=True)
    quant_pre_process(str(md / "siglip2_fp32.onnx"), str(pre))
    nodes = node_names(pre)
    fc2_59 = {n.name for n in nodes if n.op_type == "MatMul" and "mlp/fc2" in n.name
              and ("blocks.5/" in n.name or "blocks.9/" in n.name)}
    print(f"  fc2(블록 5·9) MatMul 노드: {sorted(fc2_59)}")
    assert len(fc2_59) == 2, "fc2 노드 이름을 못 찾음 — 위 출력 확인"
    for act in ("u8", "s8"):   # 준혁 구성과 같이 MatMul만 INT8 (잔차 Add·LayerNorm·GELU는 FP32)
        quant(pre, md / f"siglip2_int8_mmall_{act}.onnx", act, set(), op_types=["MatMul"])
        quant(pre, md / f"siglip2_int8_fc2L2_{act}.onnx", act, fc2_59, op_types=["MatMul"])


# ----------------------------------------------------------------------------- bench
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"ERR {e}"


def env_info(label):
    import onnxruntime as ort
    flags = sh("grep -m1 -E '^(flags|Features)' /proc/cpuinfo")
    keys = ["avx512_vnni", "avx_vnni", "avx512f", "avx2", "asimddp", "i8mm", "sve", "bf16"]
    return {"label": label, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine": platform.machine(), "os": platform.platform(),
            "cpu_model": sh("lscpu | grep -m1 'Model name' | sed 's/.*: *//'") or platform.processor(),
            "cpu_count": os.cpu_count(),
            "isa_flags": [k for k in keys if f" {k}" in f" {flags} "],
            "cpu0_cur_freq_khz": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
            "nvpmodel": sh("nvpmodel -q 2>/dev/null | head -1"),
            "onnxruntime": ort.__version__, "python": platform.python_version()}


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL   # 채점 기본 설정
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ref_name(name):
    """INT8 모델이 비교할 FP32 기준 (PRD: 원본 FP32)."""
    return "rn50_fp32.onnx" if name.startswith("rn50") else "siglip2_fp32.onnx"


def bench(md, label, threads, warmup, n, reps, match=""):
    files = sorted(f for f in md.glob("*.onnx") if f.name.startswith(match))
    assert files, f"{md}에 ONNX 없음"
    x = np.random.default_rng(1).standard_normal((1, 3, 224, 224)).astype(np.float32)
    feed = {"image": x}
    res = {"env": env_info(label), "protocol": dict(threads=threads, warmup=warmup, n=n, reps=reps), "models": {}}
    for f in files:
        res["models"][f.name] = {"MiB": round(f.stat().st_size / 2**20, 2), "lat": {}}

    # 출력 cos (FP32 대비, 랜덤 입력·랜덤 보정이라 참고만)
    outs = {f.name: session(f, 4).run(None, feed)[0].ravel() for f in files}
    for k, v in outs.items():
        r = outs.get(ref_name(k))
        if r is not None:
            res["models"][k]["cos_vs_fp32_random"] = round(float(v @ r / np.linalg.norm(v) / np.linalg.norm(r)), 4)

    for t in threads:
        print(f"\n[threads={t}] 세션 생성·워밍업", flush=True)
        sess = {f.name: session(f, t) for f in files}
        for s in sess.values():
            for _ in range(warmup):
                s.run(None, feed)
        per = {k: [] for k in sess}
        for r in range(reps):                      # 모델을 번갈아 돌려 시간에 따른 drift(열·클럭)를 고르게
            for k, s in sess.items():
                ts = []
                for _ in range(n):
                    t0 = time.perf_counter(); s.run(None, feed); ts.append((time.perf_counter() - t0) * 1e3)
                per[k].append(ts)
            print(f"  rep {r + 1}/{reps} 완료", flush=True)
        for k, reps_ts in per.items():
            means = [statistics.mean(v) for v in reps_ts]
            allv = sorted(sum(reps_ts, []))
            res["models"][k]["lat"][str(t)] = {"mean": round(statistics.mean(means), 2),
                                               "std": round(statistics.pstdev(means), 2),
                                               "p50": round(allv[len(allv) // 2], 2),
                                               "p99": round(allv[int(len(allv) * .99) - 1], 2)}
        del sess

    out = md.parent / f"bench_{label}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print_table(res)
    print(f"\n저장: {out}")


def print_table(res):
    e = res["env"]
    print("\n" + "=" * 90)
    print(f"{e['label']} | {e['cpu_model']} | {e['machine']} | ISA {e['isa_flags']} | ORT {e['onnxruntime']}")
    ths = res["protocol"]["threads"]
    hdr = f"{'model':34}{'MiB':>8}{'cos*':>7}" + "".join(f"{'t' + str(t) + ' ms':>11}{'x':>7}" for t in ths)
    print(hdr); print("-" * len(hdr))
    M = res["models"]
    for k, v in M.items():
        ref = M.get(ref_name(k))
        row = f"{k.replace('.onnx', ''):34}{v['MiB']:>8.1f}{v.get('cos_vs_fp32_random', float('nan')):>7.3f}"
        for t in ths:
            ms = v["lat"][str(t)]["mean"]
            sp = ref["lat"][str(t)]["mean"] / ms if ref else float("nan")
            row += f"{ms:>11.1f}{sp:>6.2f}x"
        print(row)
    print("x = FP32 원본 / 이 모델 (1보다 크면 빨라짐). cos* = 랜덤 입력·랜덤 보정, 참고만")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "bench", "table"])
    ap.add_argument("--out", default=os.path.expanduser("~/ssd/junyong/xplat"))
    ap.add_argument("--siglip2", action="store_true")
    ap.add_argument("--label", default=platform.machine())
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--match", default="", help="이 접두사로 시작하는 모델만 (rn50 / siglip2)")
    a = ap.parse_args()
    root = Path(a.out); md = root / "models"; md.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        print("[RN50]", flush=True); build_rn50(md)
        if a.siglip2:
            print("[SigLIP2]", flush=True)
            try:
                build_siglip2(md)
            except Exception as ex:
                print("SigLIP2 생략:", repr(ex))
    elif a.cmd == "bench":
        bench(md, a.label + (f"_{a.match}" if a.match else ""), [int(t) for t in a.threads.split(",")], a.warmup, a.n, a.reps, a.match)
    else:
        for f in sorted(root.glob("bench_*.json")):
            print_table(json.loads(f.read_text()))


if __name__ == "__main__":
    main()
