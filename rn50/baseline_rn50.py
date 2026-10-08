#!/usr/bin/env python3
"""
baseline_rn50.py — PyTorch 원본 CLIP-RN50 FP32 베이스라인 (양자화·ONNX 이전의 기준점)
==================================================================================

[목적]
  모든 최적화 결과를 비교할 "출발점"을 PyTorch 원본 그대로 측정한다.
  ONNX로 바꾼 FP32 모델이 이 값과 같은지(변환 손실 없음)도 이 결과로 확인한다.
  결과: CIFAR-100 test 10,000장 top-1 41.94% (Hailo 공식 float 42.06%, CLIP 논문 41.6%와 일치)

[검토 후 수정 사항]
  - --weights 경로가 없을 때 조용히 인터넷에서 다른 가중치를 받던 동작을 오류로 바꿈(재현성).

측정 항목
  1) 정확도   : CIFAR-100 test(10,000장) zero-shot top-1 / top-5
                - ensemble : OpenAI CLIP 논문 18개 프롬프트 평균 (주 지표)
                - single   : "a photo of a {c}." 1개 (보조 지표)
  2) 모델     : 파라미터 수, GMACs, FP32 가중치 파일 크기 (image / text / 전체)
  3) 지연     : image encoder, text encoder 각각 batch=1 (+ 선택 배치)
                mean / p50 / p90 / p99 / std, throughput
  4) 메모리   : 인코더별 독립 프로세스에서 모델 적재 후 / 추론 중 peak RSS

사용 예
  python baseline_rn50.py --weights ./weights/RN50.pt --data-root ./data --device cpu --threads 4
  python baseline_rn50.py --random-init --skip-accuracy          # 가중치 없이 파이프라인 점검(스모크)
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# 인자
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLIP-RN50 FP32 baseline")
    p.add_argument("--weights", default="openai",
                   help="OpenAI RN50.pt 경로. 'openai'면 open_clip이 자동 다운로드")
    p.add_argument("--random-init", action="store_true",
                   help="가중치 없이 무작위 초기화 (지연/크기/메모리 스모크 테스트 전용, 정확도 무의미)")
    p.add_argument("--data-root", default=str(HERE / "data"),
                   help="CIFAR-100 저장 위치 (torchvision 형식: <root>/cifar-100-python)")
    p.add_argument("--no-download", action="store_true", help="데이터셋 자동 다운로드 금지")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4, help="CPU intra-op 스레드 수 (고정 필수)")
    p.add_argument("--acc-batch", type=int, default=64, help="정확도 평가 배치")
    p.add_argument("--acc-limit", type=int, default=0, help="정확도 평가 이미지 수 제한(0=전체 10k)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lat-batches", default="1", help="지연 측정 배치 목록, 예: 1,8,32")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--runs", type=int, default=1,
                   help="지연 측정 반복 run 수 (팀 규칙: 5, run1=cold 별도 보고)")
    p.add_argument("--skip-accuracy", action="store_true")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--skip-memory", action="store_true")
    p.add_argument("--save-embeddings", action="store_true",
                   help="FP32 이미지 임베딩·텍스트 분류기 저장 (양자화 후 코사인 충실도 비교용)")
    p.add_argument("--out", default=str(HERE / "results"))
    # 내부용: 메모리 측정 서브프로세스
    p.add_argument("--_mem-probe", dest="mem_probe", choices=["image", "text"], help=argparse.SUPPRESS)
    return p.parse_args(argv)


def set_threads(n):
    # torch import 전에 설정해야 OpenMP/MKL/OpenBLAS 풀에 적용됨
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)


# --------------------------------------------------------------------------- #
# 모델
# --------------------------------------------------------------------------- #
def load_model(args):
    import torch
    import open_clip

    if args.random_init:
        model = open_clip.create_model("RN50", pretrained=None, force_quick_gelu=True)
    elif args.weights != "openai" and not os.path.isfile(args.weights):
        # 경로 오타 시 다른 가중치로 조용히 바뀌면 재현성이 깨지므로 즉시 중단
        raise SystemExit(f"가중치 파일 없음: {args.weights}")
    elif args.weights == "openai":
        model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai",
                                                            precision="fp32")
    else:
        # OpenAI 원본 JIT 체크포인트(fp16 저장) → fp32로 적재
        model = open_clip.load_openai_model(args.weights, precision="fp32", device="cpu")

    model = model.float().eval().to(args.device)
    preprocess = open_clip.image_transform(
        224, is_train=False,
        mean=open_clip.OPENAI_DATASET_MEAN, std=open_clip.OPENAI_DATASET_STD,
        interpolation="bicubic", resize_mode="shortest",
    )
    tokenizer = open_clip.get_tokenizer("RN50")
    return model, preprocess, tokenizer


class TextEncoder:
    """model.encode_text 래퍼 (visual 제외 전부 = 텍스트 인코더)"""
    def __init__(self, model):
        self.m = model

    def __call__(self, tokens):
        return self.m.encode_text(tokens)


def text_modules(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("visual.")}


def file_mb(sd, path):
    import torch
    torch.save(sd, path)
    mb = os.path.getsize(path) / 2**20
    os.remove(path)
    return mb


def model_stats(model, device):
    import torch
    sd_all = model.state_dict()
    sd_img = model.visual.state_dict()
    sd_txt = text_modules(model)
    p_img = sum(p.numel() for p in model.visual.parameters())
    p_all = sum(p.numel() for p in model.parameters())
    tmp = Path(tempfile.gettempdir()) / f"_rn50_sz_{os.getpid()}.pt"
    stats = {
        "params_M": {  # 학습 파라미터만 (BN running stats 등 buffer 제외)
            "image": round(p_img / 1e6, 2),
            "text": round((p_all - p_img) / 1e6, 2),
            "total": round(p_all / 1e6, 2),
        },
        "fp32_file_MiB": {  # torch.save(state_dict) 크기, 1 MiB = 2^20 B
            "image": round(file_mb(sd_img, tmp), 1),
            "text": round(file_mb(sd_txt, tmp), 1),
            "total": round(file_mb(sd_all, tmp), 1),
        },
        "embed_dim": int(model.text_projection.shape[1]),
    }
    try:
        from fvcore.nn import FlopCountAnalysis

        def macs(mod, x):
            f = FlopCountAnalysis(mod, x)
            f.unsupported_ops_warnings(False)
            f.uncalled_modules_warnings(False)
            return round(f.total() / 1e9, 2)

        class _T(torch.nn.Module):
            def __init__(s, m):
                super().__init__()
                s.m = m

            def forward(s, t):
                return s.m.encode_text(t)

        with torch.inference_mode():
            stats["GMACs"] = {
                "image": macs(model.visual, torch.randn(1, 3, 224, 224, device=device)),
                "text": macs(_T(model), torch.zeros(1, 77, dtype=torch.long, device=device)),
            }
        stats["GMACs"]["note"] = "fvcore 기준 multiply-accumulate. FLOPs ≈ 2 × MACs"
    except ImportError:
        stats["GMACs"] = "fvcore 미설치로 생략"
    return stats


# --------------------------------------------------------------------------- #
# 정확도
# --------------------------------------------------------------------------- #
def build_classifier(model, tokenizer, classnames, templates, device):
    import torch
    with torch.inference_mode():
        weights = []
        for c in classnames:
            toks = tokenizer([t.format(c=c) for t in templates]).to(device)
            e = model.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            weights.append(e / e.norm())
        return torch.stack(weights, dim=1)  # [D, C]


def eval_accuracy(model, preprocess, tokenizer, args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR100

    ds = CIFAR100(args.data_root, train=False, download=not args.no_download, transform=preprocess)
    classnames = [c.replace("_", " ") for c in ds.classes]
    if args.acc_limit:
        ds = Subset(ds, range(args.acc_limit))
    templates = json.load(open(HERE / "prompts" / "cifar100_templates.json"))["templates"]

    t0 = time.time()
    clf = {
        "ensemble18": build_classifier(model, tokenizer, classnames, templates, args.device),
        "single": build_classifier(model, tokenizer, classnames, ["a photo of a {c}."], args.device),
    }
    t_text = time.time() - t0

    loader = DataLoader(ds, batch_size=args.acc_batch, shuffle=False, num_workers=args.workers)
    correct = {k: {"top1": 0, "top5": 0} for k in clf}
    n = 0
    feats, labels = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            for k, W in clf.items():
                top5 = (100.0 * f @ W).topk(5, dim=-1).indices
                correct[k]["top1"] += (top5[:, 0] == y).sum().item()
                correct[k]["top5"] += (top5 == y[:, None]).any(dim=-1).sum().item()
            n += y.numel()
            if args.save_embeddings:
                feats.append(f.cpu().numpy().astype(np.float32))
                labels.append(y.cpu().numpy())
            if i % 20 == 0:
                print(f"  [acc] {n}/{len(ds)}  top1(ens)={100*correct['ensemble18']['top1']/n:.2f}%"
                      f"  {time.time()-t0:.0f}s", flush=True)

    res = {
        "dataset": "CIFAR-100 test",
        "n_images": n,
        "classnames": "torchvision CIFAR100.classes, '_'→' '",
        "preprocess": "resize shortest 224 bicubic → center crop 224 → OpenAI mean/std",
        "text_classifier_build_s": round(t_text, 1),
        "image_pass_s": round(time.time() - t0, 1),
    }
    for k, c in correct.items():
        res[k] = {"top1": round(100 * c["top1"] / n, 2), "top5": round(100 * c["top5"] / n, 2)}

    if args.save_embeddings:
        emb = Path(args.out) / "embeddings"
        emb.mkdir(parents=True, exist_ok=True)
        np.save(emb / "rn50_fp32_image_emb.npy", np.concatenate(feats))
        np.save(emb / "rn50_fp32_labels.npy", np.concatenate(labels))
        for k, W in clf.items():
            np.save(emb / f"rn50_fp32_text_clf_{k}.npy", W.cpu().numpy().astype(np.float32))
        res["embeddings_dir"] = str(emb)
    return res


# --------------------------------------------------------------------------- #
# 지연
# --------------------------------------------------------------------------- #
def _stats_ms(ts):
    import numpy as np
    a = np.asarray(ts) * 1e3
    return {
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
    }


def time_fn(fn, x, warmup, iters, device):
    import torch
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        sync()
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            fn(x)
            sync()
            ts.append(time.perf_counter() - t)
    return ts


def eval_latency(model, tokenizer, args):
    import torch
    out = {"warmup": args.warmup, "iters": args.iters, "runs": args.runs, "image": {}, "text": {}}
    txt_enc = TextEncoder(model)
    for b in [int(s) for s in args.lat_batches.split(",")]:
        img = torch.randn(b, 3, 224, 224, device=args.device)
        tok = tokenizer(["a photo of a dog."] * b).to(args.device)
        for name, fn, x in (("image", model.encode_image, img), ("text", txt_enc, tok)):
            runs = []
            for r in range(args.runs):
                ts = time_fn(fn, x, args.warmup, args.iters, args.device)
                s = _stats_ms(ts)
                s["throughput_per_s"] = round(b / (s["mean"] / 1e3), 2)
                runs.append(s)
                print(f"  [lat] {name:5s} b={b:<3d} run{r+1}: p50={s['p50']:.2f}ms "
                      f"p99={s['p99']:.2f}ms", flush=True)
            out[name][f"batch{b}"] = {"cold_run1": runs[0], "all_runs": runs}
            if len(runs) > 1:
                hot = runs[1:]
                out[name][f"batch{b}"]["hot_mean_of_p50"] = round(
                    sum(h["p50"] for h in hot) / len(hot), 3)
    return out


# --------------------------------------------------------------------------- #
# 메모리 (인코더별 독립 프로세스)
# --------------------------------------------------------------------------- #
def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def mem_probe(args):
    """서브프로세스: 인코더 하나만 생성해 적재·추론 시 RSS 측정 후 JSON 한 줄 출력.
    메모리 점유는 가중치 값과 무관하고 구조·dtype에만 의존하므로 해당 타워만 구성한다
    (전체 모델을 올린 뒤 일부를 지우면 allocator가 OS에 반환하지 않아 과대 측정됨)."""
    import torch
    import open_clip
    from open_clip import model as M
    torch.set_num_threads(args.threads)
    base = rss_mb()
    cfg = open_clip.get_model_config("RN50")
    if args.mem_probe == "image":
        tower = M._build_vision_tower(cfg["embed_dim"], M.CLIPVisionCfg(**cfg["vision_cfg"]), quick_gelu=True)
        x = torch.randn(1, 3, 224, 224)
    else:
        tower = M._build_text_tower(cfg["embed_dim"], M.CLIPTextCfg(**cfg["text_cfg"]), quick_gelu=True)
        x = open_clip.get_tokenizer("RN50")(["a photo of a dog."])
    tower = tower.float().eval().to(args.device)
    x = x.to(args.device)
    after_load = rss_mb()
    # 추론 중 RSS를 1ms 간격으로 샘플링 (ru_maxrss는 import 시점 스파이크까지 포함해 부정확)
    import threading
    peak = [after_load]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb())
            time.sleep(0.001)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time_fn(tower, x, 5, 20, args.device)
    stop.set()
    th.join()
    res = {
        "params_M": round(sum(p.numel() for p in tower.parameters()) / 1e6, 2),
        "rss_runtime_only_MB": round(base, 1),
        "rss_after_load_MB": round(after_load, 1),
        "weights_resident_MB": round(after_load - base, 1),
        "peak_rss_during_inference_MB": round(peak[0], 1),
        "peak_minus_runtime_MB": round(peak[0] - base, 1),
    }
    if args.device == "cuda":
        res["cuda_max_allocated_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print("MEMJSON " + json.dumps(res))


def eval_memory(args):
    out = {"batch": 1, "unit": "MiB",
           "note": "인코더별 새 프로세스. rss_runtime_only = torch import 직후(모델 없음). "
                   "peak_minus_runtime = 가중치 + 추론 활성값 + 작업 버퍼"}
    for enc in ("image", "text"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_mem-probe", enc,
               "--device", args.device, "--threads", str(args.threads)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        line = [l for l in r.stdout.splitlines() if l.startswith("MEMJSON ")]
        out[enc] = json.loads(line[-1][8:]) if line else {"error": r.stderr[-800:]}
        print(f"  [mem] {enc}: {out[enc]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 환경 기록
# --------------------------------------------------------------------------- #
def env_info(args):
    import numpy
    import torch
    import open_clip
    import torchvision
    cpu = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            for l in open("/proc/cpuinfo"):
                if l.startswith("model name"):
                    cpu = l.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "threads_used": args.threads,
        "device": args.device,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "open_clip": open_clip.__version__,
        "numpy": numpy.__version__,
        "weights": "RANDOM-INIT (smoke test)" if args.random_init else args.weights,
    }
    if args.device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_threads(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    if args.mem_probe:
        mem_probe(args)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"model": "CLIP-RN50 (OpenAI)", "precision": "fp32", "env": env_info(args)}
    print(json.dumps(result["env"], indent=1, ensure_ascii=False))

    model, preprocess, tokenizer = load_model(args)
    result["model_stats"] = model_stats(model, args.device)
    print("[model]", json.dumps(result["model_stats"], ensure_ascii=False))

    if not args.skip_latency:
        result["latency_ms"] = eval_latency(model, tokenizer, args)
    if not args.skip_memory:
        result["memory"] = eval_memory(args)
    if not args.skip_accuracy:
        result["accuracy"] = eval_accuracy(model, preprocess, tokenizer, args)
        print("[acc]", json.dumps(result["accuracy"], ensure_ascii=False))

    tag = "smoke" if args.random_init else args.device
    path = out_dir / f"rn50_fp32_baseline_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(result, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
