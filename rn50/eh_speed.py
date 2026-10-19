"""H 시리즈 속도 (노트북 4스레드): 같은 노드 선택을 CIFAR 64장 MinMax로 만들어(구조가 같으면 속도도 같음) 비교.
비교군: FP32, B0, C1, B2, B4, H1~H4. 결과: RN50_RESULTS"""
import subprocess, sys
PY = sys.executable
STEM = "/conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool"
BLK = [f"/layer{s}/layer{s}.{i}/" for s, n in [(1, 3), (2, 4), (3, 6), (4, 3)] for i in range(n)]
AP = "/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm"
onlyH = ",".join([STEM] + [b for b in BLK if b != "/layer1/layer1.0/"])
EX1 = "/conv1/Conv,/conv2/Conv"
EX3 = EX1 + ",/layer2/layer2.0/downsample/downsample.0/Conv,/layer2/layer2.0/Add"
EX4 = EX3 + ",/layer4/layer4.0/downsample/downsample.0/Conv,/layer4/layer4.0/Add"
cfgs = {
    "H1": ["--only", onlyH, "--only-exact", AP, "--exclude-exact", EX1],
    "H2": ["--only", onlyH, "--only-exact", "/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/Gemm", "--exclude-exact", EX1],
    "H3": ["--only", onlyH, "--only-exact", "/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/Gemm", "--exclude-exact", EX3],
    "H4": ["--only", onlyH, "--only-exact", "/attnpool/MatMul,/attnpool/Gemm", "--exclude-exact", EX4],
    "B4": ["--only", ",".join(b for b in BLK if b not in ("/layer1/layer1.0/", "/layer2/layer2.0/"))],
}
models = ["fp32=rn50_image_fp32_q1.onnx", "B0=rn50_image_int8_q1_minmax_cifar64_s0_spdB0.onnx",
          "C1=rn50_image_int8_q1_minmax_cifar64_s0_spdC1.onnx", "B2=rn50_image_int8_q1_minmax_cifar64_s0_spdB2.onnx"]
for tag, a in cfgs.items():
    subprocess.run([PY, "quantize_onnx.py", "--enc", "image", "--variant", "_q1", "--calib-source", "cifar", "--calib-n", "64",
                    "--seed", "0", "--tag", f"spd{tag}", "--overwrite", *a], check=True)
    models.append(f"{tag}=rn50_image_int8_q1_minmax_cifar64_s0_spd{tag}.onnx")
subprocess.run([PY, "bench_speed.py", "--models", *models, "--threads", "4", "--n", "200", "--repeats", "3", "--warmup", "20"], check=True)
subprocess.run([PY, "bench_mem.py", "--models", *models, "--threads", "4"], check=True)
print("ALL DONE")
