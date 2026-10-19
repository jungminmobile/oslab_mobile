"""E4-D 속도·메모리 (노트북): C1~C6, C1a~C6a, B2, B2a와 같은 노드 선택으로 양자화(보정은 CIFAR 64장 MinMax:
그래프 구조가 같으면 속도도 같으므로 빠르게 만든다) → bench_speed 4스레드 + bench_mem.
사용: rn50/.venv 의 python 으로 python e4d_speed.py   (결과: RN50_RESULTS)"""
import subprocess, sys
PY = sys.executable
STEM = "/conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool"
BLK = [f"/layer{s}/layer{s}.{i}/" for s, n in [(1, 3), (2, 4), (3, 6), (4, 3)] for i in range(n)]
AP = ["/attnpool/MatMul", "/attnpool/MatMul_1", "/attnpool/MatMul_2", "/attnpool/Gemm"]
STEPS = [["/conv1/Conv", "/conv2/Conv"], ["/layer1/layer1.0/downsample/downsample.0/Conv"], ["/layer1/layer1.0/conv1/Conv"],
         ["AP:/attnpool/MatMul_2"], ["/layer2/layer2.0/downsample/downsample.0/Conv"],
         ["AP:/attnpool/MatMul_1", "/layer4/layer4.0/downsample/downsample.0/Conv"]]
cfgs = {}
for add in (0, 1):
    ex, ap = [], list(AP)
    for k, st in enumerate(STEPS):
        for it in st:
            (ap.remove(it[3:]) if it.startswith("AP:") else ex.append(it))
        a = ["--only", ",".join([STEM] + BLK), "--only-exact", ",".join(ap), "--exclude-exact", ",".join(ex)]
        cfgs[f"C{k+1}{'a' if add else ''}"] = a + (["--exclude", "/Add"] if add else [])
B2 = ["--only", ",".join(b for b in BLK if not b.startswith("/layer1/layer1.0/")), "--only-exact", ",".join(AP)]
cfgs["B2"] = B2; cfgs["B2a"] = B2 + ["--exclude", "/Add"]
models = ["fp32=rn50_image_fp32_q1.onnx"]
for tag, a in cfgs.items():
    subprocess.run([PY, "quantize_onnx.py", "--enc", "image", "--variant", "_q1", "--calib-source", "cifar", "--calib-n", "64",
                    "--seed", "0", "--tag", f"spd{tag}", "--overwrite", *a], check=True)
    models.append(f"{tag}=rn50_image_int8_q1_minmax_cifar64_s0_spd{tag}.onnx")
subprocess.run([PY, "bench_speed.py", "--models", *models, "--threads", "4", "--n", "200", "--repeats", "3", "--warmup", "20"], check=True)
subprocess.run([PY, "bench_mem.py", "--models", *models, "--threads", "4"], check=True)
print("ALL DONE")
