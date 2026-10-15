# Hailo 4비트(a8_w4) 구성 메모 (10/15 밤, D-037)

- `g4_D256L1_w4A.alls`: D256L1 + 블록 1~4·6~11의 MLP fc1을 `a8_w4` (블록 0·5 MLP는 공식 패턴대로 a16_w16)
- `g4_D256L1_w4B.alls`: A + 블록 1~4·6~8·10·11의 MLP fc2도 `a8_w4` (블록 9 fc2 `conv41`은 a16_w16 그대로)
- ORT 4비트(`results/siglip2/G3_w4`)의 A·B 단계와 같은 층을 고른 것이다. 단, ORT는 블록 0·5 MLP도 4비트(A·B)인데 Hailo는 그 층을 16비트로 둔다.
- 층 이름 확인: `layer_map_siglip2.csv`(DFC 층 → 원래 ONNX 노드)로 19개 층이 모두 의도한 블록의 `mlp/fc1`·`mlp/fc2` MatMul임을 확인했다 (10/15). 규칙: 블록 b의 fc1·fc2 = `conv(4+4b)`·`conv(5+4b)`.
- 실행 조건: ORT 4비트 결과에서 A·B의 손실이 작을 때만 돌린다 (WSL 약 1시간 반 최적화 + 50분 스크리닝).
