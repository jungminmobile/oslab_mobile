"""
datasets.py — 평가 데이터 준비(개발용 dev / 최종 test 분리) + FP32 베이스라인 + 최종 평가
=========================================================================================

[왜 dev와 test를 나누나] (코드 검토 3번 지적 반영)
  설정(어느 층을 양자화할지 등)을 고를 때 test 사진을 쓰면, test에 우연히 잘 맞는 설정을 고르게 되어
  최종 숫자가 부풀려진다. 그래서
    - dev  : 설정 비교·선택용. 각 데이터셋의 train(또는 validation) 분할에서만 뽑는다.
    - test : 최종 보고용. 설정이 다 정해진 뒤 딱 한 번, 테스트 전체로 잰다.
    - 보정 : dev와 겹치지 않는 train 사진. test 사진은 보정에도 절대 쓰지 않는다.
  ImageNetV2는 test만 있는 데이터셋이라 dev·보정에 쓰지 않고 최종 보고에만 쓴다.

[사진 선택 방법]
  고화질 데이터셋의 dev·보정 사진은 "사진 파일 내용의 SHA-1 해시값"으로 고른다.
  해시를 0~1 사이 수 u로 바꿔 u < p_dev 이면 dev, p_dev ≤ u < p_dev + p_cal 이면 보정.
  → 파일을 어떤 순서로 읽든, 일부만 내려받든 항상 같은 사진이 선택된다(재현성).
  Food-101·Stanford Cars의 train parquet은 클래스 순으로 나뉘어 있어 일부 파일만 쓰면 클래스가 빠지므로,
  모든 파일을 하나씩 내려받아 필요한 사진만 뽑고 원본은 지운다(디스크 절약).

[캐시 형식]  cache/<이름>_x.npy 는 정규화 직전 uint8 (3,224,224). common.to_input()으로 정규화하면
  preprocess()와 완전히 같은 값(검증: 차이 0.0). float32 대비 디스크 1/4.

[명령]
  python datasets.py dev  <이름|all>            # dev 캐시 + 보정 후보 + FP32 dev 정확도
  python datasets.py test <이름> --models fp32 a.onnx b.onnx   # 테스트 전체 스트리밍 평가(최종 보고용)
  이름: cifar_A cifar_B cifar_test imagenetv2 pets flowers cars food caltech
"""
import argparse
import glob
import hashlib
import io
import json
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import common as C

HF = "https://huggingface.co/datasets"
# CLIP 논문(prompts.md) 공식 템플릿
CARS_T = ["a photo of a {}.", "a photo of the {}.", "a photo of my {}.", "i love my {}!",
          "a photo of my dirty {}.", "a photo of my clean {}.", "a photo of my new {}.", "a photo of my old {}."]
CALTECH_T = ["a photo of a {}.", "a painting of a {}.", "a plastic {}.", "a sculpture of a {}.", "a sketch of a {}.",
             "a tattoo of a {}.", "a toy {}.", "a rendition of a {}.", "a embroidered {}.", "a cartoon {}.",
             "a {} in a video game.", "a plushie {}.", "a origami {}.", "art of a {}.", "graffiti of a {}.",
             "a drawing of a {}.", "a doodle of a {}.", "a photo of the {}.", "a painting of the {}.",
             "the plastic {}.", "a sculpture of the {}.", "a sketch of the {}.", "a tattoo of the {}.",
             "the toy {}.", "a rendition of the {}.", "the embroidered {}.", "the cartoon {}.",
             "the {} in a video game.", "the plushie {}.", "the origami {}.", "art of the {}.",
             "graffiti of the {}.", "a drawing of the {}.", "a doodle of the {}."]
CALTECH_RENAME = {"background_google": "background", "faces": "off-center face", "faces_easy": "centered face",
                  "leopards": "leopard", "motorbikes": "motorbike", "airplanes": "airplane",
                  "car_side": "side of a car", "yin_yang": "yin and yang"}


def _shards(local_dir, repo, split, n):
    """(로컬 경로, 다운로드 URL) 목록. 로컬에 없으면 그때 내려받는다."""
    return [(f"{local_dir}/{split}-{i}.parquet", f"{HF}/{repo}/resolve/main/data/{split}-{i:05d}-of-{n:05d}.parquet")
            for i in range(n)]


# 고화질 데이터셋: dev/보정 후보(train 계열) 파일, test 파일, 템플릿, 이름 변환, train 후보 전체 장수
HIGHRES = {
    "pets": dict(pool=_shards("pets", "timm/oxford-iiit-pet", "train", 1),
                 test=_shards("pets", "timm/oxford-iiit-pet", "test", 1),
                 templates=["a photo of a {}, a type of pet."], rename={}, n_pool=3680),
    "flowers": dict(pool=_shards("flowers", "dpdl-benchmark/oxford_flowers102", "train", 1)
                    + _shards("flowers", "dpdl-benchmark/oxford_flowers102", "validation", 1),
                    test=_shards("flowers", "dpdl-benchmark/oxford_flowers102", "test", 6),
                    templates=["a photo of a {}, a type of flower."], rename={}, n_pool=2040),
    "cars": dict(pool=_shards("cars", "tanganke/stanford_cars", "train", 2),
                 test=_shards("cars", "tanganke/stanford_cars", "test", 2),
                 templates=CARS_T, rename={}, n_pool=8144),
    "food": dict(pool=_shards("food", "ethz/food101", "train", 8),
                 test=_shards("food", "ethz/food101", "validation", 3),   # food101의 validation = 공식 test
                 templates=["a photo of {}, a type of food."], rename={}, n_pool=75750),
    "caltech": dict(pool=_shards("caltech", "dpdl-benchmark/caltech101", "train", 1),
                    test=_shards("caltech", "dpdl-benchmark/caltech101", "test", 2),
                    templates=CALTECH_T, rename=CALTECH_RENAME, n_pool=3060),
}
DEV_N = 1000           # 데이터셋당 dev 목표 장수(해시 선택이라 ±3% 정도 흔들림)
CAL_N = 256            # 데이터셋당 보정 후보 장수 → 5개 합계 약 1280장 (E0 ⑦: 64→256)
                       # 해시 구간 [DEV, DEV+CAL)을 넓히는 방식이라 이전 64장 후보는 새 후보에 그대로 포함되고 dev는 바뀌지 않는다


def u_of(b):
    """사진 바이트의 SHA-1 → [0,1) 실수. 같은 사진이면 언제나 같은 값."""
    return int(hashlib.sha1(b).hexdigest()[:12], 16) / 16**12


def ensure(local, url):
    """파일이 없으면 내려받고, (경로, 이번에 내려받았는지)를 돌려준다."""
    p = C.EXTRA_ROOT / local
    if p.exists():
        return p, False
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download {url}", flush=True)
    urllib.request.urlretrieve(url, p)
    return p, True


def parquet_rows(entries, keep_downloads):
    """parquet 파일들에서 (이미지 바이트, 정답)을 차례로 낸다. 새로 받은 파일은 다 읽은 뒤 지운다."""
    import pyarrow.parquet as pq
    for local, url in entries:
        p, fresh = ensure(local, url)
        for rb in pq.ParquetFile(p).iter_batches(batch_size=64, columns=["image", "label"]):
            for img, lab in zip(rb.column("image").to_pylist(), rb.column("label").to_pylist()):
                yield img["bytes"], lab
        if fresh and not keep_downloads:
            p.unlink()


def classnames(name):
    import pyarrow.parquet as pq
    cfg = HIGHRES[name]
    p, _ = ensure(*cfg["test"][0])
    raw = json.loads(pq.read_schema(p).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    return [cfg["rename"].get(n, n.replace("_", " ")) for n in raw]


def imagenetv2_rows():
    """ImageNetV2 matched-frequency(1만 장, test 전용). 폴더 이름(0~999)이 ImageNet 클래스 번호."""
    tf = tarfile.open(C.EXTRA_ROOT / "imagenetv2-mf.tar")
    for m in sorted((m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg", ".png"))),
                    key=lambda m: m.name):
        yield tf.extractfile(m).read(), int(m.name.split("/")[-2])


def imagenet_meta():
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, SIMPLE_IMAGENET_TEMPLATES
    # SIMPLE 7개 = CLIP 저자들이 추천한 80개 템플릿의 핵심 부분집합(계산 시간 절약). 논문과 직접 비교할 땐 80개 사용
    return list(IMAGENET_CLASSNAMES), [t("{}") for t in SIMPLE_IMAGENET_TEMPLATES]


def decode(b):
    return C.crop_u8(Image.open(io.BytesIO(b)))


# --------------------------------------------------------------------------- #
# dev 캐시 만들기
# --------------------------------------------------------------------------- #
def build_dev(name, threads, keep_downloads):
    t0 = time.time()
    if name in ("cifar_A", "cifar_B"):
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(str(C.DATA_ROOT), train=True, download=True)          # transform 없이 PIL 이미지
        s = C.VAL_SPLITS[name[-1]]
        xs = np.stack([C.crop_u8(ds[i][0]) for i in range(s, s + C.VAL_SIZE)])
        ys = np.array([ds.targets[i] for i in range(s, s + C.VAL_SIZE)])
        names, templates = [c.replace("_", " ") for c in ds.classes], C.TEMPLATES
    else:
        cfg = HIGHRES[name]
        p_dev, p_cal = DEV_N / cfg["n_pool"], CAL_N / cfg["n_pool"]
        xs, ys, cal = [], [], []
        for b, lab in parquet_rows(cfg["pool"], keep_downloads):
            u = u_of(b)
            if u < p_dev:
                xs.append(decode(b)); ys.append(lab)
            elif u < p_dev + p_cal:
                cal.append(decode(b))
        xs, ys = np.stack(xs), np.array(ys)
        np.save(C.CACHE_DIR / f"calib_{name}_u8.npy", np.stack(cal))
        names, templates = classnames(name), cfg["templates"]
    W = C.zeroshot_classifier(names, templates, threads)
    ref = C.embed(C.FP32_IMAGE, xs, threads)
    key = name if name.startswith("cifar") else f"{name}_dev"
    for k, v in (("x", xs), ("y", ys), ("clf", W), ("ref_fp32", ref)):
        np.save(C.CACHE_DIR / f"{key}_{k}.npy", v)
    row = {"date": datetime.now().isoformat(timespec="seconds"), "data": key, "model": "fp32", "n": int(len(ys)),
           "classes_present": int(len(set(ys.tolist()))), **C.topk_metrics(ref, W, ys), "sec": round(time.time() - t0)}
    print(json.dumps(row, ensure_ascii=False))
    C.append_jsonl("baselines", row)


# --------------------------------------------------------------------------- #
# test 전체 스트리밍 평가 (최종 보고용): 사진을 한 번만 디코딩하고 여러 모델을 같은 배치로 돌린다
# --------------------------------------------------------------------------- #
def eval_test(name, models, threads, bs=32, keep_downloads=True, exec_mode="int"):
    t0 = time.time()
    if name == "cifar_test":
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(str(C.DATA_ROOT), train=False, download=True)
        rows = ((ds[i][0], ds.targets[i]) for i in range(len(ds)))
        to_u8 = C.crop_u8
        names, templates = [c.replace("_", " ") for c in ds.classes], C.TEMPLATES
    elif name == "imagenetv2":
        rows, to_u8 = imagenetv2_rows(), decode
        names, templates = imagenet_meta()
    else:
        rows, to_u8 = parquet_rows(HIGHRES[name]["test"], keep_downloads), decode
        names, templates = classnames(name), HIGHRES[name]["templates"]
    W = C.zeroshot_classifier(names, templates, threads)
    # FP32 기준은 dev 기준 임베딩과 같은 방식(그래프 최적화 끔)으로, 양자화 모델은 exec_mode(기본 정수 커널)로 실행
    sess = {m: C.session(C.onnx_path(C.FP32_IMAGE if m == "fp32" else m), threads,
                         fake_quant=(m == "fp32" or exec_mode == "fake")) for m in models}
    meta = C.run_meta()
    emb = {m: [] for m in models}
    ys, buf = [], []

    def flush():
        x = C.to_input(np.stack(buf))
        for m, s in sess.items():
            emb[m].append(s.run(None, {"image": x})[0])
        buf.clear()

    for i, (img, lab) in enumerate(rows):
        buf.append(to_u8(img)); ys.append(lab)
        if len(buf) == bs:
            flush()
        if i % 2000 == 0:
            print(f"  {i}  {time.time() - t0:.0f}s", flush=True)
    if buf:
        flush()
    y = np.array(ys)
    ref = C.l2n(np.concatenate(emb["fp32"])) if "fp32" in emb else None
    for m in models:
        E = C.l2n(np.concatenate(emb[m]))
        row = {"date": datetime.now().isoformat(timespec="seconds"), "data": f"{name}_test" if name in HIGHRES else name,
               "model": m if m == "fp32" else C.onnx_path(m).name, "n": int(len(y)),
               "exec": "fp32" if m == "fp32" else exec_mode, "threads": threads,
               **C.topk_metrics(E, W, y, ref), "sec": round(time.time() - t0), "meta": meta}
        print(json.dumps(row, ensure_ascii=False))
        C.append_jsonl("final_test", row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "test"])
    ap.add_argument("name")
    ap.add_argument("--models", nargs="+", default=["fp32"], help="test 전용. 'fp32'를 넣으면 유사도도 계산")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--keep-downloads", action="store_true", help="새로 받은 parquet을 지우지 않음")
    ap.add_argument("--exec", choices=["fake", "int"], default="int", help="test 전용: 양자화 모델 실행 방식 (E0부터 기본 int)")
    a = ap.parse_args()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "dev":
        for n in (["cifar_A", "cifar_B"] + list(HIGHRES) if a.name == "all" else [a.name]):
            build_dev(n, a.threads, a.keep_downloads)
    else:
        eval_test(a.name, a.models, a.threads, keep_downloads=True, exec_mode=a.exec)


if __name__ == "__main__":
    main()
