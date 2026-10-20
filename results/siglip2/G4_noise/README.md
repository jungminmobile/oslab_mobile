# G4 층별 잡음 분석 (DFC `hailo analyze-noise`) — 실패, 보류 (2026-10-14)

**결론: G1 기본 8비트와 구성 D 모두 같은 DFC 내부 오류로 실패했다. 같은 문제가 2회 나서 규칙(siglip2/CLAUDE.md 7절)에 따라 3회째 시도 전에 보고했고, 쭈리가 "보류"로 결정했다 (10/14 오전).** G4 결론(구성별 정확도 `../G4_dfc_screen`, 컴파일 `../G4_compile`)에는 이 분석이 필요하지 않다.

## 목적

DFC가 보고하는 층별 양자화 잡음(SNR)을 ORT 쪽 민감도(G3: 블록 5~11, fc2 5·9)와 비교하는 것이 목적이었다. 같은 위치가 Hailo에서도 민감한지 확인하려고 했다.

## 실행 (`siglip2/hailo/overnight_wsl2.sh`의 `noise` 함수, WSL Ubuntu-22.04, DFC 5.4.0, 공용 잠금 안에서)

```bash
hailo analyze-noise noise_<구성>.har --data-path /mnt/d/mokerton/artifacts/siglip2/dfc/pets_dev64_nhwc.npy \
    --calib-data-path ~/siglip2_dfc/20261013_124213/calib_highres64_s0_nhwc.npy --batch-size 2 --data-count 64
```

원본 HAR은 그대로 두고 복사본(`~/siglip2_dfc/noise_20261014/noise_<구성>.har`)으로 실행했다.

## 결과

| 구성 | 시각 | 결과 |
|---|---|---|
| G1 기본 8비트 | 10:22~10:25 | 실패: `ValueError: Dimensions must be equal, but are 768 and 780` (`lat_model_1/sub_25`, `conv_feature_splitter1_2_1/act_op_1` vs `conv_feature_splitter1_2_3/output_op_1`, 모양 [2,1,196,768] vs [2,1,196,780]) |
| D 공식 패턴 + 블록 9 fc2 | 10:37~10:40 | 같은 오류 |

- 오류는 "Full Quant Analysis" 첫 반복(0/32)에서 났다. 분석 결과는 만들어지지 않았다 (로그로 확인). 당시 로그의 `rc=0` 표시는 스크립트 버그였다 (`echo "$(date) rc=$?"`가 date의 종료 코드를 찍음, 10/14 19:00 수정). 실제 종료 코드는 기록되지 않았다.
- 780 = 768 + 12다. DFC가 내부에서 채널을 나누는 층(feature splitter)에 정렬용 채움을 넣었고, 잡음 비교 단계에서 원래 크기와 맞지 않는 것으로 보인다. **이것은 추정이고 확인하지 않았다.**
- 로그: `analyze_G1_default.log`, `analyze_D_official_b9fc2.log`. 같은 단계의 프로파일러(최적화 HAR)는 `prof_G1_default.log`에 있다. csv는 비어 있다.

## 다음 (보류 중, 다시 할 경우)

- `analyze-noise`의 다른 모드나 옵션으로 1회 시도한다.
- 또는 이 분석 없이 G4 스크리닝의 구성별 비교(어느 층을 16비트로 두면 좋아지는가)로 대신한다.
- DFC 버그로 Hailo 커뮤니티에 문의할 수 있다.
