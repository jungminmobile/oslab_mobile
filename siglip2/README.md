# SigLIP2 (담당: 준혁)

SigLIP2 ViT-B/16 또는 ViT-B/32 양자화 실험 코드를 이 폴더에 둔다.

RN50과 결과를 비교하려면 루트 README의 "실험 규칙"을 맞춘다:
데이터 구간(dev/test), 보정 256장·시드 0, 프롬프트 템플릿, Caltech 클래스 처리, 정수 커널 기준 정확도.
`rn50/common.py`, `rn50/datasets.py`의 데이터 규칙을 그대로 가져다 써도 된다.
