# Hailo-10H (본선: Raspberry Pi 5 + AI HAT+ 2)

예정 작업:

1. Hailo Dataflow Compiler로 ONNX → `.har` → `.hef` 컴파일 (**Hailo-10H 대상**. Hailo-8/8L용 .hef는 호환 안 됨)
2. 예선 결론(c3/c4 구성)을 DFC 설정으로 옮기기: stem·layer1은 16비트, 나머지 8비트
3. Pi 5에서 예선과 같은 dev/test 규칙으로 정확도·속도·메모리 실측 → 데스크톱 결과와 차이 분석

`.hef`, `.har` 파일은 용량이 커서 `.gitignore`로 제외한다. 컴파일 명령과 설정 파일만 커밋한다.
