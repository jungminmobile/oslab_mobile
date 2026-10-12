# Hailo DFC 5.x 설치 가이드 (Windows 노트북 + WSL2, Hailo-10H용)

> 작성일 2026-10-12. "확인 필요"가 붙은 내용은 공식 문서에서 직접 확인하지 못한 것이니, 설치하면서 꼭 확인하세요.

## 1. 요약
- **무엇을 설치하나**: Hailo Dataflow Compiler(DFC) **5.x**. ONNX 모델을 Hailo-10H용 `.hef` 파일로 바꾸는 프로그램입니다.
- **어디에 설치하나**: 노트북의 **WSL2 `Ubuntu-22.04`** 안에 만든 Python 가상환경(venv). 기본 배포판(Ubuntu-20.04)은 쓰지 않습니다.
- **버전**: Pi 쪽 HailoRT 버전과 **같은 5.x 버전**을 고르세요(6장). 정해진 게 없으면 최신 **5.4.0**(Model Zoo 공식 HEF가 이 버전으로 컴파일됨)을 쓰세요.
- **주의**: 노트북에 NVIDIA GPU가 없습니다. 그래서 변환은 되지만 **고급 양자화 최적화는 자동으로 꺼지거나 아주 느립니다**(7장).

## 2. 준비물
| 항목 | 현재 노트북 상태 (2026-10-12 확인) | 필요 조건 |
|---|---|---|
| WSL 배포판 | `Ubuntu-22.04` (WSL2, 지금은 Stopped), 기본은 `Ubuntu-20.04` | Ubuntu 22.04 또는 24.04 (64bit) |
| Python | 3.10.12 (Ubuntu 22.04 기본) | 3.10 / 3.11 / 3.12 |
| 디스크 | 946 GB 남음 | 넉넉함 |
| WSL 메모리 | 15 GB (+ 스왑 4 GB), CPU 8코어 | 16 GB 이상 권장 (확인 필요) |
| GPU | 없음 (Intel Core Ultra 7 258V) | 선택 사항. 있으면 NVIDIA + CUDA 12.5.1 + cuDNN 9.10 |
| Hailo 계정 | 직접 가입 필요 | Developer Zone 로그인 |

**메모리 늘리기 (선택, 권장)**: 기본값이면 WSL은 PC 메모리(32 GB)의 절반만 씁니다. CLIP처럼 큰 모델은 부족할 수 있습니다. Windows에서 `C:\Users\<내이름>\.wslconfig` 파일을 만들고 아래 내용을 넣은 다음, PowerShell에서 `wsl --shutdown`을 실행하세요.
```ini
[wsl2]
memory=24GB
swap=16GB
```

## 3. 버전 선택과 이유
| 라인 | 대상 칩 | 비고 |
|---|---|---|
| DFC **3.x** (+ Model Zoo v2.x) | Hailo-8 / 8L | **Hailo-10H에는 쓸 수 없음** |
| DFC **5.x** (+ Model Zoo master) | Hailo-10H / Hailo-15 | 우리가 쓸 것 |

- Model Zoo README에는 "Dataflow Compiler v5.4.0, HailoRT v5.4.0"이라고 적혀 있고, HAILO10H 공개 모델 표에도 "Hailo Dataflow Compiler Version v5.4.0"이라고 되어 있습니다. 같은 저장소의 GETTING_STARTED 문서에는 아직 v5.3.0이라고 남아 있습니다.
- HailoRT 5.x는 지금까지 5.0.0, 5.0.1, 5.1.0, 5.1.1, 5.2.0, 5.3.0, 5.4.0이 나왔습니다(GitHub 릴리스 기준).
- Python 3.12는 DFC 5.1.0부터 지원합니다(Hailo 직원 답변). Ubuntu 22.04에 들어 있는 **Python 3.10**으로 충분합니다.
- **규칙: DFC 버전 = Pi의 HailoRT 버전.** HEF와 HailoRT 버전이 다르면 Hailo-10H에서 "HEF file corrupted (91)" 오류가 나는 사례가 있습니다(커뮤니티, Hailo 직원은 "버전 불일치 가능성이 높다"고 답함).

## 4. 설치 방법 A: venv (추천)
GPU가 없으므로 Docker를 써도 이득이 거의 없습니다. 가장 단순한 방법인 venv를 추천합니다.

**① Ubuntu-22.04 켜기** (Windows PowerShell)
```powershell
wsl -d Ubuntu-22.04
```
**② 필요한 프로그램 설치** (지금부터는 Ubuntu 안에서 실행)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-dev python3-pip python3-venv python3-tk \
  graphviz graphviz-dev libgraphviz-dev build-essential unzip
```
**③ DFC 파일 받기** (Windows 브라우저)
1. https://hailo.ai/developer-zone/ 에 로그인한 뒤 **Software Downloads**로 갑니다.
2. 필터를 **Hailo-10H** / **Dataflow Compiler** / **Linux** / **Python 3.10**으로 맞춥니다(메뉴 이름은 확인 필요).
3. `hailo_dataflow_compiler-5.X.Y-py3-none-linux_x86_64.whl`을 받습니다. 파일 이름에 5.x가 들어 있어야 합니다.

**④ 파일을 WSL로 옮기기** (Windows 다운로드 폴더는 WSL에서 `/mnt/c/...` 경로로 보입니다)
```bash
mkdir -p ~/hailo && cd ~/hailo
cp /mnt/c/Users/<윈도우사용자명>/Downloads/hailo_dataflow_compiler-5.*-py3-none-linux_x86_64.whl .
```
**⑤ 가상환경을 만들고 설치**
```bash
python3 -m venv ~/hailo/dfc_venv
source ~/hailo/dfc_venv/bin/activate
pip install --upgrade pip wheel
pip install ./hailo_dataflow_compiler-5.*-py3-none-linux_x86_64.whl
```
새 터미널을 열 때마다 `source ~/hailo/dfc_venv/bin/activate`를 다시 실행해야 합니다.

**⑥ (선택) Model Zoo 설치** (CLIP용 레퍼런스 설정을 볼 때 씁니다)
```bash
git clone https://github.com/hailo-ai/hailo_model_zoo.git ~/hailo/hailo_model_zoo
cd ~/hailo/hailo_model_zoo && pip install -e .
```
DFC 버전과 맞는 태그를 고르세요. 예를 들어 DFC 5.4.0이면 Model Zoo도 5.4.0 계열을 씁니다(태그 이름은 확인 필요).

## 5. 설치 방법 B: Docker (Hailo AI Software Suite)
- Developer Zone에서 "Hailo AI Software Suite – Docker"(Hailo-10H용)를 받아 압축을 풀고, 안에 있는 `hailo_ai_sw_suite_docker_run.sh`를 실행합니다.
- **장점**: DFC, HailoRT, Model Zoo의 버전이 이미 서로 맞춰져 있습니다.
- **단점**: 파일이 수 GB로 큽니다. Hailo는 Docker를 **Ubuntu 호스트에서만 검증**했습니다(Windows의 Docker Desktop은 지원 대상 아님, 확인 필요). GPU가 없으니 속도 이득도 없습니다.
- **결론**: 이 노트북에서는 **A(venv)** 를 쓰세요. B는 A가 의존성 충돌로 실패할 때만 쓰는 예비 방법입니다.

## 6. 설치 확인 (스모크 테스트)
```bash
source ~/hailo/dfc_venv/bin/activate
hailo --version                 # DFC 5.x.y가 나와야 정상
pip show hailo_dataflow_compiler | head -3
hailo parser onnx --help | grep -i hw-arch   # hailo10h가 선택지에 있는지 확인
hailo --help                    # profiler, optimize, compiler 등 하위 명령 목록 (profiler가 있는지 확인 필요)
```
작은 ONNX 모델로 파싱부터 컴파일까지 해 봅니다.
```bash
mkdir -p ~/hailo/smoke && cd ~/hailo/smoke
python - <<'EOF'
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
W = numpy_helper.from_array(np.random.randn(8,3,3,3).astype(np.float32), "W")
node = helper.make_node("Conv", ["x","W"], ["y"], pads=[1,1,1,1])
relu = helper.make_node("Relu", ["y"], ["z"])
g = helper.make_graph([node, relu], "tiny",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,3,32,32])],
    [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1,8,32,32])], [W])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("",13)]), "tiny.onnx")
EOF
hailo parser onnx tiny.onnx --hw-arch hailo10h            # 결과: tiny.har
hailo optimize tiny.har --hw-arch hailo10h --use-random-calib-set   # 결과: tiny_optimized.har (옵션 이름은 확인 필요)
hailo compiler tiny_optimized.har --hw-arch hailo10h      # 결과: tiny.hef
ls -lh *.har *.hef
```
`tiny.hef`가 만들어지면 설치가 성공한 것입니다. 파일 이름은 버전마다 조금 다를 수 있습니다.

## 7. GPU가 없을 때 생기는 제한
| 항목 | GPU 있음 | 이 노트북 (CPU만) |
|---|---|---|
| 파싱, 컴파일 (`parser`, `compiler`) | 가능 | **가능** (CPU만 사용) |
| 기본 양자화 (calibration) | 빠름 | 가능, 느림 |
| optimization_level 1~4 | 사용 가능 | 자동으로 **0으로 내려감**. 경고: "Reducing optimization level to 0 … because there's no available GPU" |
| Finetune/QFT, AdaRound, IBC | 권장 | 사실상 불가. 모델 스크립트로 강제로 켤 수는 있어도 매우 느림 (확인 필요) |
| 에뮬레이터로 정확도 측정 | 빠름 | 느림 (CLIP 전체 평가는 오래 걸림) |

대응 방법:
- 노트북은 **파싱/컴파일과 동작 확인용**으로 씁니다.
- 최종 양자화(최적화 레벨 2 이상, finetune)는 NVIDIA GPU가 있는 Ubuntu 서버(CUDA 12.5.1, cuDNN 9.10, 드라이버 555 이상)에서 **같은 DFC 버전**으로 실행합니다.
- 보고서에는 CPU에서 레벨 0으로 낸 결과와 GPU에서 레벨 2 이상으로 낸 결과를 따로 적습니다.

## 8. Pi 쪽 HailoRT 버전 맞추기
Pi 5 + AI HAT+ 2에서는 Raspberry Pi OS **Trixie(64bit)** 를 쓰고 `sudo apt install dkms hailo-h10-all`로 설치합니다. `hailo-all`은 Hailo-8용이라 함께 깔면 안 됩니다.
```bash
hailortcli --version
hailortcli fw-control identify          # 펌웨어 버전, Device Architecture: HAILO10H
apt-cache policy hailo-h10-all h10-hailort
```
- 2026년 초 커뮤니티 기록에서는 `hailo-h10-all`, `h10-hailort`가 **5.1.1**이었습니다. 지금 apt에 올라온 버전은 **확인 필요**합니다(위 명령으로 확인).
- **Pi가 5.1.1이라면**: DFC 5.1.x를 받거나, Pi의 HailoRT를 DFC와 같은 버전(예: 5.4.0, Developer Zone에서 arm64 .deb 받기)으로 올리세요. 둘 중 하나로 **버전을 꼭 맞춥니다**. 새 DFC로 만든 HEF가 옛 HailoRT에서 돌아가는지(하위 호환)는 공식 문서가 없어 **확인 필요**입니다.
- 정한 버전 조합은 `requirements`/README에 적어 둡니다(재현성 확보).

## 9. 막혔을 때
| 증상 | 해결 |
|---|---|
| `pygraphviz` 빌드 실패 | `sudo apt install graphviz graphviz-dev libgraphviz-dev python3-dev` 후 다시 설치 |
| `... is not a supported wheel on this platform` | Python 버전이나 CPU 종류가 맞지 않음. `python3 --version` 확인, x86_64용 wheel인지 확인 |
| `hailo: command not found` | venv를 켜지 않은 상태. `source ~/hailo/dfc_venv/bin/activate` 실행 |
| 실수로 Ubuntu-20.04에서 설치함 | 20.04는 DFC 5.x 지원 목록에 없음. `wsl -d Ubuntu-22.04`로 들어가서 다시 설치 |
| 컴파일 중 프로세스가 강제 종료됨 (Killed) | 메모리 부족. `.wslconfig`에서 memory/swap 늘리기 |
| hash mismatch, 다운로드 오류 | 네트워크 문제. 다시 받기 |
| Pi에서 `HEF file corrupted (91)` / `xxhash does not match` | DFC와 HailoRT 버전 불일치. 8장 참고 |
| Windows 폴더(`/mnt/c`)에서 작업할 때 느림 | 작업 폴더를 `~/hailo`(리눅스 쪽)에 두기 |

## 10. 출처 링크
- Hailo Model Zoo README (master = Hailo-10/15, DFC/HailoRT v5.4.0): https://github.com/hailo-ai/hailo_model_zoo
- Model Zoo GETTING_STARTED (Ubuntu 22.04/24.04, WSL2, Python 3.10–3.12, CUDA 12.5.1/cuDNN 9.10): https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst
- HAILO10H 공개 모델 (DFC v5.4.0): https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_classification.rst
- HailoRT 릴리스: https://github.com/hailo-ai/hailort/releases
- WSL2에 DFC 설치하기 (커뮤니티): https://community.hailo.ai/t/how-to-install-the-hailo-dataflow-compiler-dfc-on-wsl2/2890
- Python 3.12 지원 (DFC 5.1.0부터): https://community.hailo.ai/t/hailo-dataflow-compiler-with-python-3-12/19129
- 호스트 요구사항, GPU 없을 때 제한: https://community.hailo.ai/t/trying-to-understand-the-software-suite-or-dfc-requirements-and-installation/9017
- GPU가 없으면 최적화 레벨이 0으로 내려가는 경고: https://community.hailo.ai/t/problem-wigh-gpu-on-hailo-warning-reducing-optimization-level-to-0-the-accuracy-won-t-be-optimized-and-compression-won-t-be-used-because-there-s-no-available-gpu/18264
- HEF/HailoRT 버전 불일치 사례 (Hailo-10H): https://community.hailo.ai/t/hailo-10h-failed-to-configure-network-groups-7-and-hef-file-corrupted-91-errors-when-testing-tiny-yolov4-hef/18171
- AI HAT+ 2 패키지 버전 (5.1.1): https://community.hailo.ai/t/hailort-not-ready-brand-new-raspi-5-ai-hat-2/19683
- Raspberry Pi 공식 AI 문서 (`hailo-h10-all`, Trixie): https://www.raspberrypi.com/documentation/computers/ai.html
