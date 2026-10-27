"""make_models_e14.py — 다른 PC 재현용: 제출 모델 O 하나에서 F2(E14-2)와 P1·P2·P2b(E14-4)를 만들고 sha256을 원본과 비교한다.
결과 폴더(results/)는 건드리지 않는다. 경로는 환경변수 E14_O_ONNX, E14_ART, E14_WORK (README_E14.md).
사용: python e14\\make_models_e14.py
"""
import sys
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rewrite_gelu as RG    # noqa: E402
import rewrite_mask as RW    # noqa: E402
import rewrite_patch as RP   # noqa: E402
import run_e14_1 as R1       # noqa: E402

# 정민 노트북(onnx 패키지 같은 버전)에서 만든 파일의 sha256 앞 8자리
EXPECT = {"F2": "03f4de18", "P1": "450db00f", "P2": "e465d10d", "P2b": "6df4ac09"}
NAMES = {"F2": "e14_2_F2_fastgelu_bias.onnx", "P1": "e14_4_P1_patch_fp32.onnx", "P2": "e14_4_P2_patch_int8.onnx",
         "P2b": "e14_4_P2b_patch_int8_qfirst.onnx"}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else R1.WORK
    work.mkdir(parents=True, exist_ok=True)
    assert R1.O.exists(), f"O 없음: {R1.O} (환경변수 E14_O_ONNX)"
    sha = RW.file_sha(R1.O)
    print("O", R1.O, sha[:8], "OK" if sha == R1.O_SHA else "불일치!", flush=True)
    assert sha == R1.O_SHA, "제출 모델 O가 다른 파일이다"
    m = onnx.load(str(R1.O))
    f2, _, _ = RG.rewrite(m, "F2")
    onnx.save(f2, str(work / NAMES["F2"]))
    ok = True
    for v in ("P1", "P2", "P2b"):
        mv, _ = RP.rewrite(f2, v)
        onnx.save(mv, str(work / NAMES[v]))
    for k, n in NAMES.items():
        s = RW.file_sha(work / n)[:8]
        same = s == EXPECT[k]
        ok &= same
        print(f"{k:4} {n:40} sha {s} {'= 원본' if same else '≠ 원본 ' + EXPECT[k] + ' (onnx 버전 차이 가능 → 정확도로 확인)'}", flush=True)
    print("모두 원본과 같음" if ok else "일부 sha 다름: 값은 같을 수 있음. run_e14_4 --step 0은 F2 sha가 다르면 멈추므로 README 참고", flush=True)


if __name__ == "__main__":
    main()
