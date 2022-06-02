import openvino.runtime
import torch
from ppq import *
from tqdm import tqdm

MODEL          = 'models\\resnet18.onnx'
INPUT_SHAPE    = [1, 3, 224, 224]
SAMPLES        = [torch.rand(size=INPUT_SHAPE) for _ in range(256)] # rewirte this to use real data.

# -------------------------------------------------------------------
# 启动 openvino 进行推理
# -------------------------------------------------------------------
openvino_executor = openvino.runtime.Core()
openvino_results = []
model = openvino_executor.compile_model(
    model = openvino_executor.read_model(MODEL), device_name="CPU")
for sample in tqdm(SAMPLES, desc='OPENVINO GENERATEING OUTPUTS'):
    openvino_results.append(model([convert_any_to_numpy(sample)]))