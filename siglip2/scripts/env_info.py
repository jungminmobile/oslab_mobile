"""
env_info.py — 측정 환경 기록 (CPU·명령어 플래그·OS·패키지 버전·git 상태)
=====================================================================

[목적]
  정수 커널 정확도와 속도는 CPU 명령어 집합(VNNI 유무)과 ONNX Runtime 버전에 따라 달라질 수 있다.
  그래서 모든 결과 폴더에 이 스크립트가 만든 env.json을 함께 남긴다 (siglip2/CLAUDE.md 3절).

[기록 항목]
  - CPU 모델명, 물리·논리 코어 수, 명령어 플래그 avx2 / avx512f / avx512_vnni / avx_vnni
  - OS, Python, 주요 패키지 버전, ORT 실행 공급자
  - 전원 연결 여부(노트북), 스레드 관련 환경변수
  - git 커밋 해시와 dirty 여부 (커밋되지 않은 변경이 있으면 보고서에 쓰지 않는다)

[교차 확인]  --cross-check-wsl
  Windows에서는 py-cpuinfo(CPUID 직접 읽기)로 플래그를 얻는다. 같은 값을 WSL의 /proc/cpuinfo에서도 읽어
  두 경로가 일치하는지 기록한다 (VERIFICATION.md 2단계).

[사용]
  "$SIGLIP2_PYTHON" scripts/env_info.py --out results/siglip2/<단계>_<주제>/env.json [--cross-check-wsl]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

FLAGS = ["avx2", "avx512f", "avx512_vnni", "avx_vnni"]
PACKAGES = ["torch", "torchvision", "open_clip_torch", "timm", "numpy", "onnx", "onnxruntime",
            "pillow", "pyarrow", "psutil", "fvcore", "py-cpuinfo", "transformers", "tokenizers"]
THREAD_ENV = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]


def _norm(flag):
    """플래그 이름 표기 차이 제거 (py-cpuinfo는 avx512vnni, /proc/cpuinfo는 avx512_vnni)."""
    return flag.replace("_", "").lower()


def cpu_info():
    import cpuinfo
    import psutil
    info = cpuinfo.get_cpu_info()
    have = {_norm(f) for f in info.get("flags", [])}
    return {"model": info.get("brand_raw") or platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "flags": {f: _norm(f) in have for f in FLAGS},
            "flags_source": "py-cpuinfo (CPUID)"}


def wsl_flags():
    """WSL /proc/cpuinfo에서 같은 플래그를 읽는다. 실패하면 오류 문자열을 돌려준다(숨기지 않음)."""
    try:
        out = subprocess.run(["wsl", "-e", "bash", "-lc", "grep -m1 '^flags' /proc/cpuinfo"],
                             capture_output=True, timeout=120).stdout.replace(b"\0", b"").decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": repr(e)}
    have = {_norm(f) for f in out.split(":", 1)[-1].split()}
    if not have:
        return {"error": "flags 줄을 찾지 못함"}
    return {f: _norm(f) in have for f in FLAGS}


def packages():
    v = {}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = None
    return v


def ort_info():
    try:
        import onnxruntime as ort
        return {"version": ort.__version__, "providers": ort.get_available_providers()}
    except ImportError:
        return None


def power():
    import psutil
    b = psutil.sensors_battery()
    return None if b is None else {"plugged": b.power_plugged, "percent": round(b.percent)}


def git_state(repo):
    """커밋 해시와 dirty 여부. dirty = 추적 파일 변경이 있거나, siglip2/ 아래에 커밋 안 된 코드 파일이 있음.
    (results/ 아래 파일은 실행이 만들거나 덧붙이는 산출물이라 추적 여부와 관계없이 dirty로 보지 않는다.) git 실패는 예외로 멈춘다."""
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
    tracked = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    untracked_code = run("status", "--porcelain", "--untracked-files=all", "--", "siglip2")
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(tracked or untracked_code)}


def collect(cross_check_wsl=False):
    env = {"date": datetime.now().isoformat(timespec="seconds"),
           "cpu": cpu_info(),
           "os": platform.platform(),
           "python": platform.python_version(),
           "python_executable": sys.executable,
           "packages": packages(),
           "onnxruntime": ort_info(),
           "power": power(),
           "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
           "git": git_state(Path(__file__).resolve().parents[2])}
    if cross_check_wsl:
        w = wsl_flags()
        env["cpu"]["flags_wsl_proc_cpuinfo"] = w
        env["cpu"]["flags_match"] = w == env["cpu"]["flags"]
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="env.json 저장 경로. 없으면 화면에만 출력")
    ap.add_argument("--cross-check-wsl", action="store_true", help="WSL /proc/cpuinfo로 플래그 교차 확인")
    a = ap.parse_args()
    env = collect(a.cross_check_wsl)
    text = json.dumps(env, ensure_ascii=False, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
