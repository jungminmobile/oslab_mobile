#!/bin/bash
# E1 0단계: 작업 폴더 준비 + DFC 스모크 테스트 (WSL Ubuntu-22.04)
set -u
SRC=$(ls -d /mnt/c/study/3/2/2026*/ | head -1)
W=~/hailo10_work
mkdir -p $W/onnx $W/calib $W/smoke $W/logs
cp -u "$SRC"xfer/calib_*_nhwc_u8.npy $W/calib/
cp -u "$SRC"oslab_mobile/rn50/onnx/rn50_image_fp32_q1.onnx $W/onnx/
ls -la $W/onnx $W/calib
source ~/hailo10_venv/bin/activate
cd $W/smoke
python - <<'PY'
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
W = numpy_helper.from_array(np.random.default_rng(0).standard_normal((8,3,3,3)).astype(np.float32), "W")
g = helper.make_graph([helper.make_node("Conv", ["x","W"], ["y"], pads=[1,1,1,1]), helper.make_node("Relu", ["y"], ["z"])],
    "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,3,32,32])],
    [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1,8,32,32])], [W])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("",13)]), "tiny.onnx")
np.save("calib_tiny.npy", np.random.default_rng(0).uniform(0,1,(8,32,32,3)).astype(np.float32))
print("tiny.onnx ok")
PY
echo "== hailo version"; hailo --version 2>&1 | tail -1
echo "== optimize help (calib flags)"; hailo optimize --help 2>&1 | grep -iE "calib|random|model-script|hw-arch" | head -10
echo "== parse"; hailo parser onnx tiny.onnx --hw-arch hailo10h -y 2>&1 | tail -5
echo "== optimize"; hailo optimize tiny.har --hw-arch hailo10h --calib-set-path calib_tiny.npy 2>&1 | tail -8
echo "== compile"; hailo compiler tiny_optimized.har --hw-arch hailo10h 2>&1 | tail -8
ls -la $W/smoke
echo "== subcommands"; hailo --help 2>&1 | sed -n 1,40p
echo "== STEP0 SMOKE DONE"
