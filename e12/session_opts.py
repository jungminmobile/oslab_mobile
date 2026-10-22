"""session_opts.py — E12-1 ORT 세션 메모리 설정 조합 C0~C4 → SessionOptions (계획서 4절)

모델 파일(ONNX)은 바꾸지 않고, 세션 설정 3개만 켜고 끈다.
  S-P prepack     : session.disable_prepacking = "1"   (가중치 정리본 사본 제거)
  S-A arena       : enable_cpu_mem_arena = False        (큰 메모리 창고 미리 잡기 제거)
  S-M mem pattern : enable_mem_pattern = False          (배치표 미리 잡기 제거)
고정: ORT_ENABLE_ALL, inter_op 1, ORT_SEQUENTIAL (rn50/common.session과 같음), CPU EP만.
이 모듈은 numpy·onnxruntime 외에는 아무것도 import하지 않는다 (메모리 기준선 base를 키우지 않기 위해).
"""
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                   # oslab_mobile
S2_SIGLIP = REPO.parent / "oslab_s2" / "siglip2"     # s2common.py, scripts/env_info.py
CACHE = Path(os.environ.get("SIGLIP2_ARTIFACTS", r"C:\s2art")) / "cache"

# 조합: (prepack 끔, arena 끔, mem pattern 끔)
COMBOS = {"C0": (False, False, False), "C1": (True, False, False), "C2": (False, True, False),
          "C3": (False, False, True), "C4": (True, True, True)}


def describe(combo):
    p, a, m = COMBOS[combo]
    return {"disable_prepacking": p, "cpu_mem_arena": not a, "mem_pattern": not m}


def make_options(combo, threads):
    import onnxruntime as ort
    no_prepack, no_arena, no_pattern = COMBOS[combo]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    if no_prepack:
        so.add_session_config_entry("session.disable_prepacking", "1")
    if no_arena:
        so.enable_cpu_mem_arena = False
    if no_pattern:
        so.enable_mem_pattern = False
    return so


def make_session(path, combo, threads):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), make_options(combo, threads), providers=["CPUExecutionProvider"])


def git_state():
    """oslab_mobile 커밋 해시와 dirty(추적 파일 변경 또는 e12/ 아래 커밋 안 된 파일)."""
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    tracked = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    untracked = run("status", "--porcelain", "--untracked-files=all", "--", "e12")
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(tracked or untracked)}


def parse_models(items):
    """['m2rot=C:\\..\\a.onnx', ...] → [(이름, Path)]"""
    out = []
    for kv in items:
        k, f = kv.split("=", 1)
        assert Path(f).exists(), f
        out.append((k, Path(f)))
    return out


# ---- E12-4 스레드 풀 설정 조합 (계획서 3절). C3 위에 세션 설정 키만 추가. 키는 ORT v1.25.0 헤더로 확인 ----
SPIN, BLOCK, FSTOP = "session.intra_op.allow_spinning", "session.dynamic_block_base", "session.force_spinning_stop"
THREAD_COMBOS = {
    "A0": {},                                   # 기준 (현재 제출 설정, 키 미지정)
    "A0_1": {SPIN: "1"},                        # 0단계 확인용: spinning 명시 1
    "A1": {SPIN: "0", BLOCK: "0"},
    "A2": {SPIN: "1", BLOCK: "4"},
    "A3": {SPIN: "0", BLOCK: "4"},
    "A4": {SPIN: "1", BLOCK: "0", FSTOP: "1"},
}


def make_thread_options(tcombo, threads, base="C3"):
    so = make_options(base, threads)
    for k, v in THREAD_COMBOS[tcombo].items():
        so.add_session_config_entry(k, v)
    return so
