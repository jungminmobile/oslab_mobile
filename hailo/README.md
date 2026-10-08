# hailo — 본선 이식 (Hailo-10H)

PRD의 **E1(DFC 호환성 관문)**, **E8a·E8b(Hailo 사전 검증·최종 패키지)**, 본선 H1~H3 작업 공간이다.

## 전제

- Hailo Dataflow Compiler **5.x**, 대상 `--hw-arch hailo10h`. Model Zoo 공개 `.hef`는 DFC 5.4.0으로 컴파일됐다.
  Hailo-8/8L용 `.hef`는 Hailo-10H에서 동작하지 않는다.
- DFC는 **float ONNX**에서 시작해 직접 양자화한다. 예선의 QDQ ONNX를 넣는 것이 아니다.
  입력은 `rn50/export_onnx.py --attnpool-q1 --image-only`로 만든 float ONNX다.
- 환경: x86 리눅스 (정민 PC의 WSL2 Ubuntu 22.04 또는 Docker). Hailo Developer Zone 계정으로 DFC를 받는다.

## 예정 파일

| 파일 | 내용 |
|---|---|
| `compile.sh` | parse → optimize → compile 한 번에 (`hailo parser onnx` / `hailo optimize` / `hailo compiler`) |
| `rn50_<구성>.alls` | 예선 구성과 같은 층별 정밀도. INT8 층은 `a8_w8`, FP32로 남긴 층은 `a16_w16` |
| `layer_map_rn50.csv` | ONNX 노드 이름 ↔ DFC 층 이름(`conv23` 등) 매핑 |
| `emulate_eval.py` | DFC 에뮬레이터(SDK_QUANTIZED·SDK_FP_OPTIMIZED)로 dev 정확도 측정 (E8a) |

## 참고: Hailo 공식 RN50 이미지 인코더 설정 (Model Zoo)

```
normalization1 = normalization([122.77, 116.75, 104.09], [68.50, 66.63, 70.32])
model_optimization_flavor(optimization_level=0, compression_level=0)
quantization_param({ew_add*}, precision_mode=a16_w16)
quantization_param({conv*}, precision_mode=a16_w16)
quantization_param([format_conversion1], precision_mode=a16_w16)
quantization_param([conv56, conv57, conv58], precision_mode=a8_w8)
```

공식 수치(Hailo-10H, CIFAR-100): RN50 42.1 → 41.2, 142 FPS · RN50x4 50.3 → 50.9, 62.6 FPS
