"""g6c_emulate.py — optimized HAR의 양자화 에뮬레이터(SDK_QUANTIZED) 출력을 dev 6×200에 대해 저장 (README 1-2 비트 확인)
B16 M2rot 계열은 정규화가 HEF 밖이라 입력은 Zoo 전처리 픽셀을 x/127.5 − 1 (NHWC float32)로 넣는다 (g6_hailo_chip.py eval과 같은 값).
사용 (hailo10_venv, CUDA_VISIBLE_DEVICES=-1, 저장소 밖 폴더에서): python g6c_emulate.py <optimized.har> <dev_zoo224 폴더> <out.npz>
"""
import sys
import time

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

har, data, out = sys.argv[1:4]
SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
r = ClientRunner(har=har)
res = {}
t0 = time.time()
with r.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    for d in SETS:
        x = np.load(f"{data}/{d}_pix.npy", mmap_mode="r")[:200].astype(np.float32) / 127.5 - 1.0
        y = r.infer(ctx, x, batch_size=8)
        res[f"{d}_emb"] = np.asarray(y, np.float32).reshape(len(x), -1)
        print(d, res[f"{d}_emb"].shape, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(out, **res)
