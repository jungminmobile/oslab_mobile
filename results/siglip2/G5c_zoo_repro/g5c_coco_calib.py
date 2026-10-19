"""g5c_coco_calib.py — Zoo COCO 보정 세트 근사 (G5c R-coco)
Zoo(v5.4.0): create_clip_vision_tfrecord.py calib가 COCO val2017/*.jpg를 glob 순서로 원본 크기 uint8 RGB로 담고(섞지 않음),
clip 전처리(classification_preprocessing.clip)가 PIL bicubic으로 224×224에 바로 맞춘다(crop 없음, 0~255). DFC는 앞 calibset_size=64장을 쓴다.
Hailo 쪽 glob 순서는 알 수 없으므로 **파일 이름 정렬 순서의 앞 N장**으로 근사한다.
사용: python g5c_coco_calib.py --dir val2017 --n 64 --out coco_calib64_nhwc.npy
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--n", type=int, default=64); ap.add_argument("--out", required=True)
a = ap.parse_args()
files = sorted(Path(a.dir).glob("*.jpg"))
assert len(files) == 5000, len(files)
xs = [np.array(Image.fromarray(np.array(Image.open(f).convert("RGB"), np.uint8)).resize((224, 224), Image.BICUBIC), np.uint8) for f in files[: a.n]]
x = np.stack(xs).astype(np.float32)
np.save(a.out, x)
Path(a.out).with_suffix(".files.json").write_text(json.dumps([f.name for f in files[: a.n]]))
print(f"coco calib n={len(x)} shape={x.shape} min={x.min()} max={x.max()} mean={x.mean():.2f} first={files[0].name} last={files[a.n - 1].name} "
      f"sha256={hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")
