# SigLIP2 Hailo 교차 확인 묶음 (쭈리 → 정민, 2026-10-17)

목적: SigLIP2 Hailo 최종 구성(D256L1)을 정민 노트북 DFC 5.4.0에서 같은 입력으로 다시 만들어, 쭈리 PC(WSL Ubuntu-22.04, DFC 5.4.0)와 같은 결과가 나오는지 확인한다 (D-011·D-027). **제출본이 아님 (pre-release).**

## 실행 (정민 노트북 WSL, DFC 가상환경 켠 상태)
```bash
gh release download siglip2-hailo-crosscheck-20261017 --repo jungminmobile/oslab_mobile --dir siglip2_crosscheck
cd siglip2_crosscheck
bash run_crosscheck.sh              # parse → optimize(약 1시간, GPU 불필요) → 에뮬레이터 200장 비교
COMPILE=1 bash run_crosscheck.sh    # (선택) HEF 컴파일·프로파일러까지, 추가 약 1시간 반
```
끝나면 `crosscheck_steps.txt`, `crosscheck_env.txt`, `crosscheck_result.json`, `*.log`를 쭈리에게 보내 주세요.

## 쭈리 PC 기준값 (같아야 하는 것)
- optimize HAR sha256: `84e981f75fff02da12a4ea5e8aaa1d026003a4d0c82d05b45bb090337975e6a7` (DFC 결정성에 따라 다를 수 있음 — 다르면 아래 출력 비교로 판단)
- 에뮬레이터 Pets dev 200장: top-1 92.0%, FP32 임베딩 대비 cos 평균 0.9752 (`crosscheck_result.json`의 `*_jjury` 값). **사진별 cos 차이가 1e-4 안이고 예측이 거의 다 같으면 같은 결과**
- (컴파일 시) HEF 99,831,808 B, 컨텍스트 17개(프로파일러 `Number of Contexts`), sha256 `b33a266f694ad40b1498ca4a19ec4dfccc96620e5127586ba5e55c987e81d99a`

## 파일
- `siglip2_image_fp32_b1.onnx`: FP32 이미지 인코더(배치 1 고정, opset 17)
- `calib_highres256_s0_nhwc.npy`: 보정 사진 256장(train 고화질, 시드 0, NHWC float32)
- `g4_D256L1_official_b9fc2.alls`: 모델 스크립트(공식 패턴 + 블록 9 fc2 a16_w16 + 보정 256장 + 최적화 수준 1)
- `pets_dev.npz`: 비교용 Pets dev 앞 200장(x NCHW, y, clf, ref) / `ref_D256L1_pets200.npz`: 쭈리 PC 결과
- 저장소: `siglip2/hailo/crosscheck/`, 결과 기록은 `results/siglip2/G4_crosscheck/`에 할 예정
