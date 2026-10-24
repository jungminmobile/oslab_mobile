# B32 본선 준비: (A) 본선 HEF 바탕 = 제출 ONNX 바탕인지 확인, (B) 본선 HEF 확정 기록 + test 1회 (2026-10-24, 용 요청)

> 모델은 새로 만들지 않는다. 판정 기준은 결과 전에 이 파일에 고정·커밋한다. 결과를 보고 기준·HEF 선택을 바꾸지 않는다.
> 브랜치 `siglip2/b32-hef-test` (`siglip2/g6f-ceiling` 8b4eff5 기준).

## A. 같은 모델인지 확인 (CPU만) — 설계 (결과 전 고정)

**확인할 문장 (보고서):** "본선 HEF 바탕(`a0eb502b…`, 배치 1 export)과 제출 ONNX 바탕(`23236123…`, 배치 가변 export)은 같은 모델(회전 행렬 `3a6391a2…`, 같은 sink 규칙, 같은 M2 상수)".

1. **계보:** GF HEF(G6f) → 빌드 기록(`build.jsonl`, `run_info.txt`) → HAR 메타데이터(`original_model_meta.json`)로 parse에 쓴 ONNX를 거슬러 올라가 sha256을 적는다. 두 ONNX를 만든 명령·커밋·인자 표.
2. **가중치:** 두 ONNX의 initializer(+ Constant 노드 값)를 이름으로 대응, 이름이 다르면(`onnx::MatMul_*`) 소비 노드 이름·입력 위치로 대응. 비트 동일/다름 개수. 회전된 가중치(MatMul·Conv), M2 상수(sink K/V 평균 `kc_*`·`vc_*`, `u_c` `uc_*`, 마스크 `zm_*`·`m_*`), LN 파라미터를 따로 센다. 배치 축 계산에만 쓰이는 Constant(모양 계산)는 따로 분리한다.
3. **회전 행렬:** 두 export 기록의 `rotq_sha256`와 파일 `rotq_rothad.npy`의 sha256.
4. **sink 규칙:** 두 그래프에서 마스크 상수(`zm_*`, `m_*`)의 0/1 위치와 그 상수가 쓰이는 블록(노드 이름)을 직접 읽는다. 기대: K/V 대체 15@블록 6, 7@블록 10, u_c 대체 시작 블록 5·9.
5. **출력:** dev 64장(pets_dev 앞 64장, Zoo 전처리 256 캐시)을 두 ONNX(ORT FP32 `ORT_ENABLE_ALL`, 배치 1)로 돌려 사진별 출력 cos 최소, 최대 절대 차이.

**판정:** 2에서 배치 축과 무관한 initializer가 **모두 비트 동일**, 3·4가 같고, 5의 cos 최소 ≥ 0.999999 → **"같은 모델"**. 하나라도 다르면 무엇이 다른지와 보고서 문장을 어떻게 고칠지 적는다. B(test)는 "같은 모델"일 때만 진행한다.

## A. 결과 — **판정: "같은 모델"** (2026-10-24, `A_same_model/a_result.json`, 스크립트 커밋 `6b228e5` dirty 아님)

### A-1. 계보 (GF HEF → parse한 ONNX)
| 단계 | 파일 | sha256 | 근거 |
|---|---|---|---|
| GF HEF (본선) | `~/siglip2_work/g6/dfc/b32_GF/comp/g6_b32_G.hef` | `0c2e04a6ef9bfb60…` | G6f `build.jsonl` (`tag b32_GF`, compile만, 커밋 `d696320` dirty 0) |
| ← G optimized HAR | `~/siglip2_work/g6/dfc/b32_G/g6_b32_G_optimized.har` | `409b0344…` | 위 줄의 `src_har_sha256`, 오늘 파일 sha256 다시 계산해 같음 |
| ← G parse | `g6_b32_G.har` ← `hailo parser onnx …/g6_b32_m2rot_b1.onnx` | — | G6d `build.jsonl`·`run_info.txt` (`tag b32_G`, 커밋 `ddf9b5f` **dirty 1**), `parse.log` |
| ← **ONNX** | `~/siglip2_work/g6/onnx/g6_b32_m2rot_b1.onnx` | **`a0eb502b…`** | `run_info.txt`의 `onnx_sha256`, optimized HAR 안 `original_model_meta.json`의 `original_model_path`, 오늘 파일 sha256 → **맞음** |
- 참고: DFC parser가 첫 시도 뒤 자동으로 단순화한 `g6_b32_m2rot_b1.sim.onnx`로 다시 parse했다(`parse.log` "Simplified ONNX model for a parsing retry", HAR 메타 `simplified_model_path`). 끝 노드 `/trunk/Gather`는 ONNX의 마지막 노드(출력 `embedding`)라 HEF가 그래프 전체를 덮는다.
- G 빌드 시작 때 저장소가 dirty 1이었다(G6d 기록 그대로). 입력 ONNX·.alls·보정 파일 sha256은 기록되어 있다.

| | 본선 HEF 바탕 (b1) | 제출 ONNX 바탕 (dyn) |
|---|---|---|
| 파일 | `g6_b32_m2rot_b1.onnx` | `g6_b32_m2rot_dyn.onnx` → 이름만 바꿔 `siglip2_image_fp32_g6b_b32_m2rot.onnx` |
| sha256 | `a0eb502b…` | `23236123…` |
| 명령 | `s2_g6.py --model b32 export --cfg m2rot` | `s2_g6.py --model b32 export --cfg m2rot --dynamic` |
| 커밋 (dirty) | `eb78616` (False), G6 `export.jsonl` | `e7cd97c` (False), G6b `export.jsonl` |
| 모델 ID | `hf-hub:timm/ViT-B-32-SigLIP2-256` | 같음 |
| 회전 `rotq_sha256` | `3a6391a2…` | `3a6391a2…` |
| sink 규칙 `unified` | `15@6:5,7@10:9` | 같음 |
| export 확인 (64장) | ORT↔torch cos 최소 0.99999999697, torch↔HF M2 0.99999999879 | 0.99999999829, 0.99999999879 |
| 두 커밋 사이 `s2_g6.py` 차이 | — | `--dynamic` 옵션 추가뿐 (`dynamic_axes={"image":{0:"batch"},"embedding":{0:"batch"}}`, 파일 이름 `_dyn`). 나머지 차이는 b16 키 추가(B32 경로와 무관) |
| 입력/출력 | `image [1,3,256,256]` → `embedding [1,768]` | `[batch,…]` → `[batch,768]` |
| 노드 / initializer / Constant | 750 / 145 / 151 | 998 / 145 / 261 |

### A-2. 가중치 (initializer 145개 + Constant)
| 구분 | 비트 동일 | 다름 |
|---|---|---|
| 회전된 가중치 (MatMul 53 + patch Conv 1) | **54** | 0 |
| M2 상수: sink K/V 평균 `kc_*`·`vc_*` | **20** | 0 |
| M2 상수: `u_c` `uc_b5…b11` | **7** | 0 |
| M2 상수: 마스크 `zm_b5`·`zm_b9`·`m_b6_0`·`m_b10_1` | **4** | 0 |
| LN 파라미터 (`blocks.0.norm1`, `attn_pool.norm` weight·bias; 나머지 LN의 γ·β는 회전 때 뒤 Linear에 접혀 없음) | **4** | 0 |
| 그 밖의 파라미터 (bias, pos_embed, latent 등) | **56** | 0 |
| **initializer 합계** | **145 / 145** | **0** |
| float Constant (attention 0.125 배율 등) | 65 | 0 |
| int Constant, 짝이 맞은 것 (Split 크기·Squeeze 축·Gather 인덱스 52 + 모양 경로의 공통 상수 5) | 57 (양쪽 같은 값) | 1 (아래 줄의 Reshape 목표 모양) |
| int Constant, **배치 축 모양 경로** | — | b1에만 28 (Reshape 목표 모양 `[1,64,768]` 등 26 + latent Expand용 ConstantOfShape·Equal 2), dyn에만 138 (Shape→Gather→Unsqueeze→Concat 모양 계산), 짝이 맞았지만 값 다름 1 (b1 `[1,64,768]` vs dyn `[-1]`, 둘 다 Reshape 목표 모양) |
- Constant 노드는 이름 번호가 dyn의 모양 계산 노드 때문에 밀려서, (첫 소비 op·입력 위치·블록·순번)으로 대응했다. 처음 실행(이름 대응, `a_result_v1_name_mismatch.json`)에서 나온 "다름"은 이 이름 밀림 때문이었다.
- 판정 조건 "배치 축과 무관한 initializer가 모두 비트 동일" → **충족 (145/145)**. 다른 Constant는 모두 배치 축 모양 계산이다.
- `a_result.json`의 `tensor_summary`에서 `|only_b1`·`|only_dyn` 칸의 "same"은 "한쪽에만 있는 개수"를 센 것이다(비교한 것 아님).
- **사후 변경 (A, 독립 검토에서 지적):** Constant 대응 방식은 첫 실행(이름 대응) 결과를 본 뒤 (소비 op·입력 위치·블록·순번)으로 바꿨다. 첫 결과는 `a_result_v1_name_mismatch.json`에 남겼다.
- **보충 (독립 검토 반영, `a_ops_check.py` → `a_ops_check.json`):** 두 그래프 모두 opset 17. 모양 계산 op(Shape·Gather·Unsqueeze·Concat·Reshape·Constant 등)를 뺀 계산 노드 503개의 (op, 속성) 다중집합이 **같다**. op 개수가 다른 것은 Concat 1/29, Constant 151/261, Gather 1/28, Reshape 28/29, Shape 1/29, Unsqueeze 0/54뿐이다.

### A-3. 회전 행렬
- 두 export 기록 `rotq_sha256` = `3a6391a2091db60d…` (같음), 파일 `$SIGLIP2_ARTIFACTS/onnx/rotq_rothad.npy` sha256 `3a6391a2…` (수정 시각 10/18, 두 export(10/20)보다 앞) → **같음**.

### A-4. sink 규칙 (그래프에서 직접 읽음, 블록 번호 = 그 노드 앞에 있는 Softmax 수. 블록 b의 attention 뒤 노드는 b+1로 셈)
| 상수 | 값이 1인 토큰 행 | 쓰는 노드 (블록, op) | 해석 |
|---|---|---|---|
| `m_b6_0` | 15 | (6, Mul) ×2 | 블록 6 attention의 K/V를 토큰 15에서 대체 |
| `m_b10_1` | 7 | (10, Mul) ×2 | 블록 10 attention의 K/V를 토큰 7에서 대체 |
| `zm_b5` | 15 | (6, Mul) | 블록 5 attention 뒤 토큰 15 잔차를 u_c로 |
| `zm_b9` | 7, 15 | (10, Mul) | 블록 9 attention 뒤 토큰 7·15 |
| `kc/vc_b6…b11`, `kc/vc_pool` | — | 블록 6~11·MAP 헤드의 Sub | |
| `uc_b5…b11` | — | 블록 5~11 attention 뒤의 Sub | |
- 두 그래프에서 이 표 전체(값 sha256·쓰는 블록·op)가 **같다** (`sink_equal: true`). 노드 이름 번호와 Identity 별칭 개수만 다르다. → **15@6:5, 7@10:9 확인.**

### A-5. 출력 (pets_dev 앞 64장, Zoo 전처리 256 캐시, ORT 1.25.0 FP32 `ORT_ENABLE_ALL`, 배치 1)
- 사진별 출력 cos 최소 **0.99999999981** (평균 0.9999999999959), 최대 절대 차이 6.7e-5 (원 출력), 정규화 뒤 3.2e-6. 비트 동일은 아니다(ORT가 두 그래프를 다르게 최적화하는 부동소수점 순서 차이로 보임, 확인 안 함).
- 기준 ≥ 0.999999 → **충족**.

### A 판정
- 2 (145/145 비트 동일), 3 (같음), 4 (같음), 5 (0.99999999981 ≥ 0.999999) → **"같은 모델"**. 보고서 문장은 그대로 써도 된다.
- 덧붙일 것(선택): "두 ONNX는 배치 축 처리(모양 계산 Constant)만 다르고 initializer 145개가 비트 동일하다. HEF는 DFC parser가 단순화한 `a0eb502b`의 `.sim.onnx`로 parse했다."

---

## B. 본선 HEF 확정 기록 + test 1회 — 설계 (결과 전 고정, 2026-10-24)

> A가 "같은 모델"로 판정되어 진행한다. test 사진은 이 설계 커밋 **뒤에만** 처음 읽는다.

### B-1. 결정 기록 문안 — 상태 "제안" (`siglip2/DECISIONS.md`에 붙일 안 — **그 파일은 고치지 않는다**)
여러 브랜치(`g8-candidates`·`report-b32`·`b32-portable-junyong`)에서 마지막 번호는 D-049다 → 다음 번호 D-050·D-051.

```markdown
## D-050 본선 Hailo HEF(B/32) = GF (G6f, `0c2e04a6…`) — D-047(B16_rb_bk)을 B/32로 바꿈
- 상태: **제안** (10/24, 용) / 승인자: 
- 준혁 동의를 받으면 DECISIONS.md로 옮기면서 승인자를 채운다.
- **결정: 본선 HEF는 G6f GF** `g6_b32_G.hef` (GF 빌드, sha256 `0c2e04a6ef9bfb6055c249ad969f07546711d6130ce401800fcc7fdc61c8b29a`, 99,876,864 B, 컨텍스트 13). G optimized HAR(`409b0344…`) + G .alls + F의 compile 두 줄(`allocator_param`·`context_switch_param`), compile만. 바탕 ONNX `g6_b32_m2rot_b1.onnx` (`a0eb502b…`) = 예선 제출 ONNX 바탕 `23236123…`와 같은 모델 (initializer 145/145 비트 동일, 회전 `3a6391a2…`, sink 15@6:5·7@10:9; `results/siglip2/B32_hef_test/README.md` A절).
- 근거: GF는 G와 출력이 비트 동일(G6f, 6×200 = 1,200장)이므로 정확도는 G(dev 전체 G − Z +2.67 [+1.92, +3.46]%p 우위, G6d)와 같고, 배치 1 14.93 ms로 Zoo 공개 HEF(Z) 16.49 ms 대비 −9.5% (G6f 같은 세션). 더 빠른 ALL8F(13.38 ms)는 dev 전체에서 GF 대비 −0.87 [−1.46, −0.28]%p 열위이고 CIFAR A 손실도 더 크다(FP32 대비 −5.3, G6f 9절) → 채택하지 않음.
- test 1회: `results/siglip2/B32_hef_test/` B절 (GF·Z 실칩, 결과는 HEF 선택에 쓰지 않음).

## D-051 예선 ONNX(O-AB)의 float 출력(QKᵀ·Softmax)은 Hailo에서 a16으로 맞추지 않는다 — 기록만
- 상태: **제안** (10/24, 용) / 승인자: 
- 준혁 동의를 받으면 DECISIONS.md로 옮기면서 승인자를 채운다. 팀 규칙("ORT에서 FP32로 남긴 층은 Hailo에서 a16_w16")의 예외다. 확정되면 D-049 위험 ③을 닫는다.
- **결정:** O-A(`--no-matmul-out-q`)로 ORT에서 float이 된 QKᵀ 출력·Softmax를 본선 HEF에서 `a16_w16`으로 맞추지 않는다. 차이는 보고서에 기록만 한다.
- 근거: Hailo는 float ONNX(`a0eb502b`)에서 DFC로 따로 양자화하므로 ORT 양자화 그래프의 정밀도 배치를 따를 필요가 없다. 맞추려면 HEF를 다시 빌드(optimize·compile 약 1시간 + 실칩 재측정)해야 하고, D-050의 GF 근거(정확도·지연)가 다시 확인되어야 한다.
- 팀 규칙("ORT에서 FP32로 남긴 층은 Hailo에서 a16_w16") 예외로 보고서에 적는다.
```

### B-2. test 데이터 (데스크탑 WSL, `b32t_data.py`)
- **사진:** 준혁 ORT test(`siglip2/g8-candidates`, 커밋 `72784d1`, `results/siglip2/G8_candidates/b32/test/`)와 같은 7종을 **같은 코드**로 고른다: `s2_test_eval.rows_names_templates()` (g8-candidates `092a4ab` 코드, worktree `~/oslab_b32p`에서 import). 고화질 5종 공식 test 전체(Food는 validation = 공식 test), CIFAR-100 test 1만, ImageNetV2 matched-frequency 1만.
- **같은 사진인지 확인 (`siglip2/g8-candidates`에 사진별 목록·sha256 파일이 없어서 아래로 대신한다):**
  1. 데이터셋별 장수·클래스 수·템플릿 수 = 준혁 `test.jsonl`
  2. ImageNetV2 tar sha256 = `f0c37fdf…` (`results/siglip2/G6_test/data_sha256.txt`, g8-candidates)
  3. **준혁 FP32 재현:** 같은 FP32 ONNX(`siglip2_image_fp32.onnx` B/32, `be19a8ad…`), 같은 timm 전처리·open_clip 분류기·ORT 4스레드·배치 50으로 FP32 top-1·top-5를 내서 준혁 `test.jsonl`의 FP32 값과 **소수 둘째 자리까지 같아야** 한다. 7종 중 하나라도 다르면 **실칩 전에 멈추고 보고**한다. (FP32는 후보가 아니므로 선택에 쓰이지 않는다.)
  4. 기록: parquet 조각·tar·CIFAR 파일 sha256, 사진별 Zoo 픽셀 sha1 목록의 sha256, 정답 배열 sha256 → `data_list.json` (다음 재현용)
- **Zoo 전처리 (G6 `s2_g6.py data`와 같음):** `img.convert("RGB").resize((256, 256), BICUBIC)` → uint8 NHWC, crop 없음. `test_zoo256/<set>_pix.npy`, `<set>_y.npy`.
- 1회 보장: 데이터셋마다 `done_data_<set>.json`. 있으면 그 데이터셋은 다시 읽지 않는다.

### B-3. FP32 기준 (데스크탑, `b32t_fp32hf.py`)
- G6 `s2_g6.py fp32`와 같은 계산: HF `google/siglip2-base-patch32-256` FP32 비전 인코더 `pooler_output`, 입력 = 위 Zoo 픽셀 → `x/127.5 − 1`. 분류기 = HF 텍스트 인코더 FP32(`s2_g6pre.classifier`, G6과 같음), 클래스 이름·템플릿 = B-2와 같은 것(준혁 test와 같음). M2 없음(원래 FP32).
- 출력: `ref/<set>_ref.npy`, `<set>_clf.npy`, `<set>_y.npy`, `done_fp32_<set>.json`.

### B-4. 실칩 (AGX, `~/ssd/junyong/hailo_b32t/`)
- HEF: **GF** `b32_GF.hef` (`0c2e04a6…`), **Z** Zoo v5.4.0 공개 `b32_Z.hef` (`c1612329…`, G6과 같은 파일). 시작 전 sha256 대조, 다르면 멈춤.
- 코드: G6 `g6_hailo_chip.py eval` 그대로 (입력 FLOAT32 픽셀 0~255, HEF 안 정규화, 배치 1, 출력 정규화 전 임베딩). 추가 옵션 `--subdir`(기본 꺼짐 = 기존 `dev_zoo<H>`)로 `test_zoo256/`을 읽는다.
- 순서: GF 7종 → Z 7종 (데이터셋마다 따로 실행).
- **AGX 규칙:** 시작 전과 데이터셋마다 다른 Hailo 프로세스 0개(있으면 대기, 3시간 넘으면 멈춤). 데이터셋마다 칩 max(ts0, ts1) < **64.6 °C**까지 냉각(90분 넘으면 멈춤). **실행 중 온도 감시**: 별도 프로세스가 15초마다 온도를 읽어 **95 °C를 넘으면 즉시 eval을 죽이고 멈춘다**(`HOT_STOP` 기록). tmux 금지, `setsid nohup` + 로그 + `timeout`.
- 감시 점검(test 전, dev만): GF로 pets_dev 앞 200장을 감시를 켠 채 eval → G6f GF eval 임베딩(같은 200장)과 비트 동일해야 하고, 온도 읽기 오류가 없어야 한다. 실패하면 감시 방식을 바꾼 뒤 다시 점검하고, test는 시작하지 않는다.
- 1회 보장: HEF·데이터셋마다 `done_<HEF>_<set>.json`.

### B-5. 판정·보고 (결과 전 고정, **결과를 보고 HEF 선택을 바꾸지 않는다**)
- top-1: 각 임베딩을 L2 정규화 → HF 분류기(B-3)와 곱해 argmax. FP32 = HF FP32 기준 임베딩.
- 지표 (`b32t_summary.py`): **GF − Z, GF − FP32, Z − FP32**
  - 고화질 5종 평균 Δtop-1 (주 지표)과 쌍 비교 95% 구간: `s2_paired_ci.boot_hr` (층화 짝 부트스트랩, 시드 0, 1,000회 — 준혁 `s2_test_summary.py`와 같은 방법)
  - CIFAR-100, ImageNetV2 따로 (Δ와 95% 구간)
  - 데이터셋별 top-1, FP32 대비 cos 평균
- 문구: 구간 하한 > 0 "우위", 상한 < 0 "열위"(FP32 대비면 "유의한 손실"), 0 포함 "동률(유의차 없음)".
- 함께 적는 것(비교용, 판정 아님): 준혁 ORT test(O-AB −0.23 [−0.36, −0.08], ONNX FP32 기준이라 FP32 정의가 다름), dev 결과(G − Z +2.67, G FP32 대비 −1.55).
- 속도는 다시 재지 않는다: GF 배치 1 14.93 ms, Z 16.49 ms (G6f 같은 세션), GF 99.9 MB·Z 102.6 MB.
- 결과가 dev와 달라도(예: GF − Z가 동률·열위) 그대로 보고하고, HEF는 D-050대로 GF.

### B-6. 결과 위치
`B_test/` (data_list.json, fp32 기준 요약, chip eval_info·온도 로그, test_summary.json). 큰 파일(픽셀·임베딩)은 `~/siglip2_work/b32t/`와 AGX에 두고 커밋하지 않는다 (사진별 예측 npz는 작으면 커밋).

### B-7. 사후 변경 (test 실칩 전, 결과 전 — 2026-10-24)
1. **덩어리 eval + 덩어리마다 냉각 (B-4 보강, 안전 목적):** 감시 점검(dev pets·cars·food 전체 2,942장, GF) 중 칩이 1분 만에 76 °C까지 올랐다. 과거 긴 연속 eval(G6/G6d, 5,900장)은 94~96 °C까지 갔다. 데이터셋 단위로만 식히면 Food(25,250장, 약 23분 연속)에서 95 °C 중단이 거의 확실하다. 그래서 **데이터셋마다 2,000장 덩어리로 나눠** 덩어리마다 다른 Hailo 작업 확인·64.6 °C 미만 냉각을 하고, 끝나면 이어 붙인다. 실칩 출력은 결정적이고(덩어리·순서와 무관) 아래 점검으로 확인했으므로 결과 값은 바뀌지 않는다.
   - `g6_hailo_chip.py eval --start`(기본 0 = 기존과 같음; 0이면 `eval_info.json`에 키도 추가하지 않음)를 추가했다.
2. **감시 점검 결과 (dev만, test 아님):**
   - ① 200장(pets_dev): G6f GF eval 임베딩과 비트 동일. 그런데 eval이 4초라 온도 읽기와 겹치지 않아 감시 점검으로는 무효 → ② 다시 함.
   - ② pets·cars·food dev 전체 2,942장(약 55초): 실행 중 온도 읽기 3회(64.2 → 71.0 → 76.0 °C), 읽기 오류 0, **G6d G dev 전체 임베딩과 비트 동일**(GF = G).
   - ③ 덩어리 경로(`--start 0/500 --k 500`, pets_dev 990장): 이어 붙인 것이 G 임베딩과 비트 동일.
3. **사진 대조 결과 (B-2, 실칩 전):** `B_test/data_check.json` **all_ok = true**. 7종 모두 장수·클래스 수·템플릿 수와 FP32 ONNX top-1·top-5가 준혁 `test.jsonl`과 소수 둘째 자리까지 같고, ImageNetV2 tar sha256 = `f0c37fdf…`. ImageNetV2 tar는 이 PC에 없어 HF `vaishaal/ImageNetV2`의 `imagenetv2-matched-frequency.tar.gz`(실제로는 압축 안 된 tar)를 받아 sha256을 대조했다. 데이터 준비 커밋 `f187204`, 코드 `~/oslab_b32p` `3356672`(siglip2 코드 = g8-candidates `092a4ab`), 모두 dirty 아님.
   - ImageNetV2 FP32 top-1은 `topk_metrics`(argsort) 65.83 = 준혁, argmax로는 65.85 (G6_test에 적힌 같은 이름 클래스 동점 문제). B-5 요약은 세 모델 모두 argmax로 계산한다.

## B. 결과 — test 1회 (2026-10-24 06:12~07:40 AGX, `B_test/test_summary.json`, 요약 커밋 `a235e40` dirty 아님)

**결론: 본선 HEF GF는 test에서도 Zoo 공개 HEF(Z)보다 고화질 5종 평균 +2.73 [+2.46, +3.02]%p 우위다 (dev G − Z +2.67과 같은 크기). FP32 대비 손실은 −1.04 [−1.24, −0.85]%p로 유의하다 (dev −1.55보다 작음).** HEF 선택은 D-050대로 GF (결과로 바꾸지 않음).

| 데이터셋 | 장수 | FP32 (HF) | **GF** | Z | GF − Z [95%] | GF − FP32 [95%] | Z − FP32 [95%] | cos GF / Z (FP32 대비) |
|---|---|---|---|---|---|---|---|---|
| Pets | 3,669 | 93.54 | 92.42 | 90.35 | +2.07 [+1.23, +2.89] | −1.12 [−1.66, −0.55] | −3.19 [−3.95, −2.45] | 0.982 / 0.950 |
| Flowers | 6,149 | 84.49 | 83.31 | 80.29 | +3.02 [+2.31, +3.74] | −1.17 [−1.69, −0.75] | −4.20 [−4.94, −3.54] | 0.983 / 0.947 |
| Cars | 8,041 | 90.80 | 88.94 | 84.40 | +4.54 [+3.86, +5.24] | −1.85 [−2.36, −1.39] | −6.39 [−7.06, −5.74] | 0.973 / 0.925 |
| Food | 25,250 | 88.90 | 87.49 | 83.47 | +4.02 [+3.65, +4.40] | −1.41 [−1.68, −1.15] | −5.43 [−5.80, −5.07] | 0.973 / 0.925 |
| Caltech | 6,084 | 84.83 | 85.21 | 85.22 | −0.02 [−0.43, +0.39] | +0.38 [+0.07, +0.71] | +0.39 [+0.02, +0.77] | 0.968 / 0.931 |
| **고화질 5종 평균 (주 지표)** | | **88.51** | **87.47** | **84.75** | **+2.73 [+2.46, +3.02] 우위** | **−1.04 [−1.24, −0.85] 유의한 손실** | **−3.76 [−4.05, −3.52] 유의한 손실** | 0.976 / 0.935 |
| CIFAR-100 (별도) | 10,000 | 76.05 | 72.18 | 70.98 | +1.20 [+0.48, +1.95] 우위 | −3.87 [−4.50, −3.18] | −5.07 [−5.80, −4.36] | 0.948 / 0.879 |
| ImageNetV2 (별도) | 10,000 | 65.85 | 63.84 | 60.76 | +3.08 [+2.39, +3.77] 우위 | −2.01 [−2.57, −1.50] | −5.09 [−5.75, −4.45] | 0.973 / 0.929 |

- top-1은 argmax(L2 정규화 임베딩 @ HF 분류기), 95% 구간은 층화 짝 부트스트랩(시드 0, 1,000회). 사진별 예측·cos는 `B_test/preds_<set>.npz`.
- **FP32 기준:** HF FP32(Zoo 전처리)의 고화질 평균 88.51은 준혁 ONNX FP32(timm 전처리) 88.51과 같다. 고화질 5종 중에서는 Caltech만 84.83 vs 84.81로 다르고, ImageNetV2도 65.85(argmax) vs 65.83(준혁, argsort)로 다르다(전처리·분류기·동점 처리가 달라 같을 필요는 없음).
- **Caltech만 GF ≈ Z이고 둘 다 FP32보다 약간 높다** (+0.38, +0.39, 유의). 다른 4종은 GF가 Z보다 2~4.5%p 높다. 원인은 확인하지 않았다.
- **CIFAR-100 손실이 가장 크다** (GF −3.87). dev CIFAR A에서도 GF(= G)가 가장 큰 손실을 낸 데이터셋이었다(G6d).
- **dev와 비교 (판정 아님):** GF(= G) dev 전체 FP32 대비 −1.55 [−2.09, −1.03], G − Z +2.67 [+1.92, +3.46] → test −1.04, +2.73. 우위 크기는 같고, FP32 대비 손실은 test 점추정이 작다(두 구간이 겹쳐 차이가 있다고 할 수는 없음).
- **ORT 예선 제출 모델과 (FP32 정의가 달라 직접 비교 아님):** 준혁 ORT test O-AB −0.23 [−0.36, −0.08] (ONNX FP32 기준), CIFAR-100 +0.93, ImageNetV2 −0.20. Hailo GF는 ORT O-AB보다 FP32 대비 손실이 약 0.8%p 크다(고화질). 같은 FP32 기준(88.51)으로 보면 GF 87.47, O-AB 88.28.
- **속도 (다시 재지 않음, G6f 같은 세션):** GF 배치 1 14.93 ms, Z 16.49 ms (−9.5%). HEF 크기 GF 99,876,864 B, Z 102,645,760 B (−2.7%).
- **FP32 대비 (CLAUDE.md):** 정확도 GF −1.04%p (위). 크기는 FP32 ONNX(`a0eb502b`, 378,243,038 B) 대비 GF HEF −73.6%. Hailo에는 FP32 HEF가 없어 지연 감소율은 FP32 기준으로 낼 수 없다 (ORT FP32 CPU 지연과는 장치가 달라 비교하지 않음).

### 실칩 실행 기록 (`B_test/chip/out/`)
- 시작 전·데이터셋·덩어리마다 다른 Hailo 프로세스 0개. 덩어리(2,000장) **76개(HEF 2 × 38: Pets 2·Caltech 4·Flowers 4·Cars 5·Food 13·CIFAR 5·ImageNetV2 5)**마다 64.6 °C 미만에서 시작.
- **최고 온도 83.05 °C** (07:23:23, 감시가 냉각 시간까지 포함해 세션 내내 15초마다 읽은 342회, 오류 0), 덩어리 끝 최고 80.26 °C. HOT_STOP 없음, eval 실패 없음.
- HEF sha256 시작 전 대조 (GF `0c2e04a6…`, Z `c1612329…`), 입력 FLOAT32 픽셀 0~255(HEF 안 정규화), test 픽셀 14개 파일 sha256을 AGX에서 다시 확인(`~/siglip2_work/b32t/test_zoo256_SHA256SUMS`). 임베딩 파일 sha256 `chip/out/emb_sha256.txt` (임베딩은 커밋하지 않음, 데스크탑 `~/siglip2_work/b32t/chip_out/`, AGX `~/ssd/junyong/hailo_b32t/out/`).
- 걸린 시간: GF 06:12~06:55, Z 06:55~07:40 (냉각 포함).
- test 사용: 데스크탑에서 사진 준비 1회(+ 준혁 FP32 재현 대조), HF FP32 1회, 실칩 HEF마다 1회. 모두 done 파일로 1회만.

### 확인 못 한 것 / 주의
- 준혁의 사진별 test 예측(`test_*.npz`)은 커밋되지 않아 사진 단위 대조는 하지 않았다(데이터셋별 FP32 top-1·top-5 재현으로 대신).
- 감시(별도 프로세스의 `hpf.Device` 온도 읽기)가 eval 중 VDevice와 동시에 열려도 출력이 같다는 것은 dev 2,942장으로만 확인했다.
