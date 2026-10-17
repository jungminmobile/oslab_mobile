# SigLIP2 후보 플랫폼 교차 측정 (x86 · ARM) — 2026-10-16, 용(준용)

준혁이 공유한 최종 후보 3개(GitHub Release `siglip2-xplat-20261015`, `SHA256SUMS.txt` 확인)를 **같은 파일·같은 사진**으로 다른 CPU에서 측정한 결과다.
모델 파일과 dev 사진은 Release에 있으므로 여기에는 결과만 둔다.

## 측정 환경
| 이름 | CPU | 명령어 | OS | ORT |
|---|---|---|---|---|
| `orin_a78ae` | Jetson AGX Orin CPU, Cortex-A78AE 12코어 (MAXN) | ARM asimddp(sdot), i8mm 없음 | Ubuntu (JetPack 6.1) | 1.23.2 (Python 3.10, ORT 1.24부터 3.10 휠 없음) |
| `labpc` | Intel Core Ultra 9 285K | AVX-VNNI, AVX2 (AVX-512 없음) | WSL2 Ubuntu | 1.23.2 / 1.25.0 |
| (기준) 준혁 PC | Intel i7-1165G7 | AVX-512 VNNI | Windows | 1.25.0 |

## 폴더
| 경로 | 내용 | 만든 방법 |
|---|---|---|
| `out_orin_a78ae/` | ARM 정확도 (result.json, pred_*.npz) | `portable_eval.py --data dev200 --threads 4` (README 명령 그대로) |
| `out_labpc_ort1232/`, `out_labpc_ort1250/` | x86 정확도, ORT 두 버전 | 같음 |
| `bench/bench_orin_a78ae_jh_siglip2.json` | ARM 지연 | `xplat_probe.py bench` — 배치 1, 워밍업 20, 200장×3회(모델 번갈아), 스레드 1·2·4, ORT_ENABLE_ALL |
| `bench/bench_labpc_ort1232_jh_siglip2.json` | x86 지연 | 같음 |
| `peak_mem_*.jsonl` | 모델별 메모리 (있을 때) | `peak_mem.py` — 모델마다 새 프로세스, 세션 생성 직후·추론 20회 후 RSS, VmHWM |
| `scripts/` | `xplat_probe.py`, `peak_mem.py` | |

bench는 FP32 기준 파일을 이름으로 찾으므로 `bench/models/`에 짧은 이름의 심볼릭 링크를 두고 쟀다: `siglip2_fp32.onnx`, `siglip2_int8_s8.onnx`, `siglip2_int8_u8rr.onnx` → Release의 원본 3개.

## 요약
### 정확도 (dev 6종 × 200장, 정수 커널)
| 환경 | 후보 | 고화질 cos | 고화질 Δtop-1 |
|---|---|---:|---:|
| x86 285K (1.23.2 = 1.25.0, 값 완전히 같음) | S8 | 0.9557 | +0.3 |
| x86 285K | U8+rr | 0.9545 | +0.1 |
| ARM Orin | S8 | 0.9519 | +0.6 |
| ARM Orin | U8+rr | 0.9547 | +0.1 |

- FP32 top-1은 세 환경 모두 `ref`와 같다.
- S8만 ARM에서 출력이 달라진다(cos −0.004). ARM은 S8 활성값을 U8로 바꾸지 않고 그대로 계산하기 때문으로 보인다.
- 200장 표본이라 top-1 차이는 흔들림 범위다. 플랫폼 간 차이의 유의성은 pred 파일로 짝 부트스트랩해서 판단한다.

### 지연 (배율 = FP32 / 후보, t1 / t2 / t4)
| 환경 | FP32 ms | S8 | U8+rr |
|---|---|---|---|
| x86 285K, ORT 1.23.2 | 240.6 / 124.9 / 68.3 | 1.53 / 1.34 / 1.12 | 2.50 / 2.36 / 2.21 |
| ARM Orin, ORT 1.23.2 | 1145.4 / 577.2 / 293.9 | 2.52 / 2.43 / 2.27 | 2.52 / 2.43 / 2.27 |

### 해석
- U8+rr은 두 플랫폼에서 모두 약 2.2배 빠르고 출력도 거의 같다.
- S8은 x86에서 속도 이득이 거의 없고(1.12배), ARM에서는 출력이 달라진다.
- ORT 버전(1.23.2 vs 1.25.0) 효과는 정확도·속도 모두 없다.

### peak 메모리 (peak_mem.py — 모델마다 새 프로세스, 4스레드, 배치 1 × 20회 후 VmHWM, ORT 1.23.2)
| 환경 | FP32 | S8 | U8+rr |
|---|---:|---:|---:|
| x86 285K (2회: 첫 값 / jsonl 값) | 516.8 / 498.5 MiB | 269.0 / 263.5 (−47%) | 243.2 / 248.4 (−50%) |
| ARM Orin | 470.4 MiB | 252.0 (−46%) | 210.1 (−55%) |

- 프로세스 전체 RSS 기준(Python·ORT 기본 약 40~47 MiB 포함). 감소율은 jsonl 값 기준.
- 두 플랫폼 모두 U8+rr < S8 < FP32. 준혁 PC(Windows peak_wset) 결과와 순서가 같다. 절대값은 측정 방식이 달라 직접 비교하지 않는다.
- x86 2회 측정 차이는 ±6 MiB.
