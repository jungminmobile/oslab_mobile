# E12-1 ORT 세션 메모리 설정

- 계획서: [`E12-1_ORT세션메모리설정.md`](E12-1_ORT세션메모리설정.md) / 결과: [`results/REPORT.md`](results/REPORT.md)
- 코드: [`e12/`](../../e12) (`session_opts.py`, `check_equal.py`, `bench_mem_session.py`, `bench_speed_session.py`, `analyze_e12_1.py`, `run_e12_1.bat`)

## 결론
**C3 (`so.enable_mem_pattern = False`) 채택.** SigLIP2 M2rot_ort 추론 peak 메모리 226.5 → 206.5 MiB (−8.8%), 지연 1T 0.0% / 2T +0.5% / 4T −1.6% (차이 없음), 출력 비트 동일.
arena 끔은 −10%지만 4T +57% → 기본값 X. prepack 끔은 효과 없음. FP32는 모든 설정 −1% 이하.
예선(ORT CPU) 전용. 설정은 ONNX에 저장되지 않으므로 제출 실행 코드에 따로 넣어야 한다 (아직 반영 안 함).

## 준비물 (다른 PC에서 재현할 때)
| 항목 | 내용 |
|---|---|
| 모델 | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` (Release `siglip2-m2rot-ort-20261019`, sha256 `e84c3727…0f9e`), `siglip2_image_fp32.onnx` (Release `siglip2-xplat-20261015`, sha256 `333cad08…40e0`) |
| SigLIP2 코드 | 이 저장소의 SigLIP2 코드를 `oslab_mobile` **옆 폴더 `oslab_s2`** 에 체크아웃 (측정 때 커밋 `1cc17c4`, `origin/siglip2/g3rq-m2rot-ort` 등에 포함). `e12/session_opts.py`의 `S2_SIGLIP`가 이 위치를 본다 |
| 캐시 | `$SIGLIP2_ARTIFACTS/cache/*_x.npy` (SigLIP2 전처리 캐시). 메모리·속도는 `pets_dev_x.npy`만 필요, G1은 dev 5종 `_x/_y/_clf.npy` |
| 패키지 | numpy, onnxruntime(측정 1.25.0), psutil, py-cpuinfo (env 기록용) |

## 실행 (Windows cmd, 경로는 각자 맞춤)
```bat
cd oslab_mobile
set "PY=<SigLIP2 venv>\Scripts\python.exe"
set "SIGLIP2_DATA=<s2data 폴더>"
set "SIGLIP2_ARTIFACTS=<캐시가 있는 폴더, 예: C:\s2art>"
set OPENBLAS_NUM_THREADS=1
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set "OUT=results\E12-1_ORT세션메모리설정\results_<이름>"
set "M2=<M2rot onnx 전체 경로>"
set "FP=<FP32 onnx 전체 경로>"

"%PY%" -u e12\check_equal.py --model "%M2%" --combos C0 C1 C2 C3 C4 --n 200 --threads 4 --out "%OUT%\equal.json"
"%PY%" -u e12\bench_mem_session.py --models "m2rot=%M2%" --combos C0 C1 C2 C3 C4 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl"
"%PY%" -u e12\bench_mem_session.py --models "fp32=%FP%" --combos C0 C1 C2 C3 C4 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl"
"%PY%" -u e12\bench_speed_session.py --session 1 --models "m2rot=%M2%" "fp32=%FP%" --combos C0 C1 C2 C3 C4 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl"
:: 2분 휴식
"%PY%" -u e12\bench_speed_session.py --session 2 --models "m2rot=%M2%" "fp32=%FP%" --combos C2 C1 C4 C3 C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl"
:: 2분 휴식
"%PY%" -u e12\bench_speed_session.py --session 3 --models "m2rot=%M2%" "fp32=%FP%" --combos C4 C3 C2 C1 C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl"
"%PY%" e12\analyze_e12_1.py "%OUT%"
```
- 정민 PC 기준 총 약 1시간 10분 (G1 7분, 메모리 7분, 속도 세션 3 × 17분 + 휴식).
- 전원 연결, 절전 끄기, 무거운 앱 종료 후 실행. 결과 폴더는 `results_<이름>`으로 따로 둔다.
- 리눅스·맥: 메모리 측정이 Windows `peak_wset` 기준이라 `bench_mem_session.py`는 Windows 전용. 다른 OS는 속도·G1만 비교.
