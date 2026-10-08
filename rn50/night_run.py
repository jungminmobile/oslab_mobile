"""
night_run.py — 2026-10-07 야간 실험 자동 실행기 (실험계획표PRD_2026-10-07.md 4절의 순서)
=========================================================================================

[목적]
  PRD에 적은 실험(R0~R8, X0~X8)을 우선순위 순서대로 차례로 실행한다.
  - 이미 만들어진 모델·이미 기록된 평가는 건너뛴다(중간에 꺼져도 다시 실행하면 이어서 진행).
  - 마감 시각(기본 15:00 KST)이 가까우면 남은 저우선 단계를 건너뛴다.
  - 진행 상황은 results/night_progress.log 에 한 줄씩 남긴다.

[사용]  python night_run.py [--deadline 15:00] [--only R1,R2]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
def _p(env, default):
    """환경변수가 있으면 그 경로, 없으면 이 폴더 기준 기본 경로."""
    return str(Path(os.environ.get(env, HERE / default)))


# 경로 기본값은 이 폴더 기준(common.py와 같은 규칙). 다른 위치는 환경변수로 지정.
ONNX = Path(_p("RN50_ONNX", "onnx"))
RESULTS50 = Path(_p("RN50_RESULTS", "results"))
WEIGHTS = Path(_p("RN50_WEIGHTS", "weights"))
ENV50 = dict(os.environ, CLIP_MODEL="RN50", RN50_ONNX=str(ONNX), RN50_CACHE=_p("RN50_CACHE", "cache"),
             RN50_RESULTS=str(RESULTS50), RN50_DATA=_p("RN50_DATA", "data"),
             RN50_EXTRA=_p("RN50_EXTRA", "extra_data"))
# RN50x4는 캐시·결과를 RN50과 섞지 않도록 별도 폴더
ENVX4 = dict(ENV50, CLIP_MODEL="RN50x4", RN50_CACHE=_p("RN50X4_CACHE", "cache_x4"),
             RN50_RESULTS=_p("RN50X4_RESULTS", "results_x4"))
LOG = RESULTS50 / "night_progress.log"

STEM = "/conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool"
AP = "/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm"
DEV = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HIGHRES = ["pets", "flowers", "cars", "food", "caltech"]

# 구성 이름 → quantize_onnx.py 노드 선택 옵션
CONFIGS = {
    "only-stem": ["--only", STEM],
    "only-layer1": ["--only", "/layer1"], "only-layer2": ["--only", "/layer2"],
    "only-layer3": ["--only", "/layer3"], "only-layer4": ["--only", "/layer4"],
    "only-attnpool": ["--only-exact", AP],                                   # = 누적 c1
    "c2": ["--only", "/layer4", "--only-exact", AP],
    "c3": ["--only", "/layer3,/layer4", "--only-exact", AP],
    "c4": ["--only", "/layer2,/layer3,/layer4", "--only-exact", AP],
    "c5": ["--only", "/layer1,/layer2,/layer3,/layer4", "--only-exact", AP],
    "Fon": ["--exclude", "attnpool", "--keep", AP],                          # 전부 INT8 (AttnPool D 방식)
    **{f"stem+{s}": ["--only", f"{STEM},/{s}"] for s in ["layer1", "layer2", "layer3", "layer4"]},
    "stem+attnpool": ["--only", STEM, "--only-exact", AP],
    # 4비트: c4 구성에서 큰 층 가중치만 4비트(활성값 8비트)
    "c4_w4-AP": ["--only", "/layer2,/layer3,/layer4", "--only-exact", AP, "--w4-exact", AP],
    "c4_w4-L4AP": ["--only", "/layer2,/layer3,/layer4", "--only-exact", AP, "--w4", "/layer4", "--w4-exact", AP],
    "c4_w4-L3L4AP": ["--only", "/layer2,/layer3,/layer4", "--only-exact", AP, "--w4", "/layer3,/layer4", "--w4-exact", AP],
}
R1 = ["only-stem", "only-layer1", "only-layer2", "only-layer3", "only-layer4", "only-attnpool"]
R2 = ["c2", "c3", "c4", "c5", "Fon"]
R3 = ["stem+layer1", "stem+layer2", "stem+layer3", "stem+layer4", "stem+attnpool"]
R4_CFG = ["only-attnpool", "c2", "c3", "c4", "c5", "Fon"]
R6 = ["c4_w4-AP", "c4_w4-L4AP", "c4_w4-L3L4AP"]


def log(msg):
    line = f"{datetime.now(KST):%m-%d %H:%M} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, env):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  ERROR {' '.join(cmd[:3])}...: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
    return r


def prefix(env):
    return env["CLIP_MODEL"].lower()


def model_name(env, cfg, src="highres"):
    return f"{prefix(env)}_image_int8_q1_minmax_{src}256_s0_{cfg}.onnx"


def quantize(env, cfg, src="highres"):
    name = model_name(env, cfg, src)
    if not (ONNX / name).exists():
        # RN50x4는 활성값이 커서 보정 배치 8이면 메모리 부족(OOM, 05:26 실패) → 배치 2
        bs = "2" if env["CLIP_MODEL"] == "RN50x4" else "8"
        run(["quantize_onnx.py", "--enc", "image", "--variant", "_q1", "--calib-source", src,
             "--calib-bs", bs, "--tag", cfg] + CONFIGS[cfg], env)
        for d in Path(tempfile.gettempdir()).glob("ort.quant.*"):          # 실패 시 남는 임시 파일 정리(각 300MB+)
            if time.time() - d.stat().st_mtime > 600:
                subprocess.run(["rm", "-rf", str(d)])
    if not (ONNX / name).exists():
        log(f"  양자화 실패: {name}")
        return None
    return name


def evaluated(env, model, data, exec_):
    f = Path(env["RN50_RESULTS"]) / "evaluate.jsonl"
    if not f.exists():
        return False
    for line in open(f):
        r = json.loads(line)
        if r["model"] == model and r["data"] == data and r.get("exec") == exec_:
            return True
    return False


def evaluate(env, models, exec_="int", data=DEV):
    models = [m for m in models if m]                        # 양자화 실패(None) 제외
    for d in data:
        todo = [m for m in models if not evaluated(env, m, d, exec_)]
        if todo:
            run(["evaluate.py", "--exec", exec_, "--data", d, "--models"] + todo, env)


def cleanup(env, names, keep=()):
    """디스크 절약: 평가가 끝난 모델 중 이후 단계(속도·fake 확인)에 필요 없는 것을 지운다.
    지운 모델은 README 명령으로 언제든 다시 만들 수 있다."""
    for n in [x for x in names if x]:
        if n not in keep and (ONNX / n).exists():
            (ONNX / n).unlink()


def stage(name, est_min, deadline):
    left = (deadline - datetime.now(KST)).total_seconds() / 60
    if left < est_min:
        log(f"SKIP {name} (남은 {left:.0f}분 < 예상 {est_min}분)")
        return False
    log(f"START {name} (예상 {est_min}분, 남은 {left:.0f}분)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="15:00")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    now = datetime.now(KST)
    if a.deadline == "none":                      # 마감 없이 끝까지
        deadline = now + timedelta(days=7)
    else:
        h, m = map(int, a.deadline.split(":"))
        deadline = now.replace(hour=h, minute=m, second=0)
        if deadline < now:
            deadline += timedelta(days=1)
    only = set(a.only.split(",")) if a.only else None
    want = lambda s: only is None or s in only
    t0 = time.time()

    if want("R1R2") or want("R1") or want("R2"):
        if stage("R1+R2 (RN50 단계별·누적)", 40, deadline):
            ms = [quantize(ENV50, c) for c in R1 + R2]
            evaluate(ENV50, ["fp32"] + ms)
            log("DONE R1+R2")
    if want("R3") and stage("R3 (stem+단계)", 20, deadline):
        ms = [quantize(ENV50, c) for c in R3]
        evaluate(ENV50, ms); cleanup(ENV50, ms); log("DONE R3")
    if want("R4") and stage("R4 (보정 출처 cifar·mix)", 40, deadline):
        ms = [quantize(ENV50, c, s) for s in ["cifar", "mix"] for c in R4_CFG]
        evaluate(ENV50, ms); cleanup(ENV50, ms, keep={model_name(ENV50, "c4", "cifar")}); log("DONE R4")
    if want("X0") and stage("X0 (RN50x4 준비)", 60, deadline):
        w = WEIGHTS / "RN50x4.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if not w.exists():
            subprocess.run(["curl", "-sSL", "-o", str(w), "https://openaipublic.azureedge.net/clip/models/"
                            "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"])
        if not (ONNX / "rn50x4_image_fp32.onnx").exists():
            run(["export_onnx.py", "--weights", str(w)], ENVX4)
        if not (ONNX / "rn50x4_image_fp32_q1.onnx").exists():
            r = run(["export_onnx.py", "--weights", str(w), "--attnpool-q1", "--image-only", "--verify"], ENVX4)
            log("  " + " ".join(l for l in r.stdout.splitlines() if "verify" in l))
        Path(ENVX4["RN50_CACHE"]).mkdir(parents=True, exist_ok=True)
        for n in ["cifar_A"] + HIGHRES:
            if not (Path(ENVX4["RN50_CACHE"]) / f"{n if n.startswith('cifar') else n + '_dev'}_x.npy").exists():
                run(["datasets.py", "dev", n], ENVX4)
        log("DONE X0")
    if want("X1X2") and stage("X1+X2 (RN50x4 단계별·누적)", 100, deadline):
        ms = [quantize(ENVX4, c) for c in R1 + R2]
        evaluate(ENVX4, ["fp32"] + ms)
        cleanup(ENVX4, [model_name(ENVX4, c) for c in R1 if c != "only-attnpool"]); log("DONE X1+X2")
    if want("R6") and stage("R6 (RN50 4비트)", 20, deadline):
        evaluate(ENV50, [quantize(ENV50, c) for c in R6]); log("DONE R6")
    if want("R7") and stage("R7 (RN50 속도)", 20, deadline):
        m50 = ["fp32=rn50_image_fp32_q1.onnx"] + [f"{c}={model_name(ENV50, c)}" for c in R1 + R2 + R6]
        run(["bench_speed.py", "--models"] + m50, ENV50)
        log("DONE R7")
    if want("R8") and stage("R8 (RN50 fake-quant 상위 후보)", 55, deadline):
        top = [model_name(ENV50, c) for c in ["c3", "c4", "c5", "Fon"]] + [model_name(ENV50, "c4", "cifar")]
        evaluate(ENV50, top, "fake", DEV + ["cifar_B"]); log("DONE R8")
    if want("X4") and stage("X4 (RN50x4 보정 출처)", 150, deadline):
        for s_ in ["cifar", "mix"]:                      # 출처별로 만들고 평가한 뒤 바로 지움(디스크)
            ms = [quantize(ENVX4, c, s_) for c in ["only-attnpool", "c3", "c4", "c5", "Fon"]]
            evaluate(ENVX4, ms); cleanup(ENVX4, ms)
        log("DONE X4")
    if want("X6") and stage("X6 (RN50x4 4비트)", 60, deadline):
        evaluate(ENVX4, [quantize(ENVX4, c) for c in R6]); log("DONE X6")
    if want("X8") and stage("X8 (RN50x4 fake-quant)", 90, deadline):
        evaluate(ENVX4, [model_name(ENVX4, c) for c in ["c4", "Fon"]], "fake"); log("DONE X8")
    if want("X7") and stage("X7 (RN50x4 속도)", 30, deadline):
        mx4 = ["fp32=rn50x4_image_fp32_q1.onnx"] + [f"{c}={model_name(ENVX4, c)}" for c in ["only-attnpool"] + R2 + R6]
        run(["bench_speed.py", "--models"] + [x for x in mx4 if (ONNX / x.split('=')[1]).exists()], ENVX4)
        log("DONE X7")
    if want("R5") and stage("R5 (stem 내부 15조합)", 50, deadline):
        src = quantize(ENV50, "only-stem")
        run(["make_stem_combos.py", "--src", src], ENV50)
        combos = sorted(p.name for p in ONNX.glob(src.replace("_only-stem.onnx", "_sc_*.onnx")))
        evaluate(ENV50, combos); cleanup(ENV50, combos); log("DONE R5")
    log(f"ALL DONE ({(time.time() - t0) / 3600:.1f}h)")


if __name__ == "__main__":
    main()
