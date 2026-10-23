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
| int Constant, 계산 경로 (Split 크기·Squeeze 축·Gather 인덱스) | 52 (양쪽 같은 값) | 0 |
| int Constant, **배치 축 모양 경로** | — | b1에만 28 (Reshape 목표 모양 `[1,64,768]` 등 26 + latent Expand용 ConstantOfShape·Equal 2), dyn에만 138 (Shape→Gather→Unsqueeze→Concat 모양 계산), 짝이 맞았지만 값 다름 1 (b1 `[1,64,768]` vs dyn `[-1]`, 둘 다 Reshape 목표 모양) |
- Constant 노드는 이름 번호가 dyn의 모양 계산 노드 때문에 밀려서, (첫 소비 op·입력 위치·블록·순번)으로 대응했다. 처음 실행(이름 대응, `a_result_v1_name_mismatch.json`)에서 나온 "다름"은 이 이름 밀림 때문이었다.
- 판정 조건 "배치 축과 무관한 initializer가 모두 비트 동일" → **충족 (145/145)**. 다른 Constant는 모두 배치 축 모양 계산이다.

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
