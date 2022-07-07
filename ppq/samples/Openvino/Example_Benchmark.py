# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 openvino 对 PPQ 导出的模型进行推理
# 你需要注意，openvino 也可以运行各种各样的量化方案，你甚至可以用 tensorRT 的 policy
# 但总的来说，openvino 需要非对称量化的 activation 和对称量化的 weights
# 现在的写法针对单输入网络哦，多输入的你得自己改改
# ---------------------------------------------------------------

# For this onnx inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
import numpy as np
import openvino
import torch
from tqdm import tqdm
import time

from ppq import *
from ppq.api import *

QUANT_PLATFROM    = TargetPlatform.PPL_CUDA_INT8
BATCHSIZE         = 1
DEVICE            = 'cuda'
INPUTSHAPE        = [BATCHSIZE, 3, 640, 640]
SAMPLES           = [torch.rand(size=INPUTSHAPE) for _ in range(256)]
BENCHMARK_SAMPLES = 512
MODEL_PATH        = 'yolov7.onnx'
VALIDATION        = False

with ENABLE_CUDA_KERNEL():
    quantized = quantize_onnx_model(
        onnx_import_file=MODEL_PATH, calib_dataloader=SAMPLES, collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=INPUTSHAPE,
        setting=QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM)

    if VALIDATION:
        executor = TorchExecutor(graph=quantized)
        ref_results = []
        for sample in tqdm(SAMPLES, desc='PPQ GENERATEING REFERENCES', total=len(SAMPLES)):
            result = executor.forward(inputs=sample.to(DEVICE))[0]
            result = result.cpu().reshape([1, -1])
            ref_results.append(result)

    fp32_input_names  = [name for name, _ in quantized.inputs.items()]
    fp32_output_names = [name for name, _ in quantized.outputs.items()]

    graphwise_error_analyse(graph=quantized, running_device='cuda', 
                            dataloader=SAMPLES, collate_fn=lambda x: x.cuda(), steps=32)

    export_ppq_graph(
        graph=quantized, platform=TargetPlatform.OPENVINO_INT8,
        graph_save_to='model_int8.onnx')

int8_input_names  = [name for name, _ in quantized.inputs.items()]
int8_output_names = [name for name, _ in quantized.outputs.items()]

import openvino.runtime
openvino_executor = openvino.runtime.Core()

# run with openvino.
# do not use Tensorrt provider to run quantized model.
# TensorRT provider needs another qdq format.
if VALIDATION:
    model = openvino_executor.compile_model(
        model = openvino_executor.read_model(model="model_int8.onnx"), device_name="CPU")
    openvino_results = []
    for sample in tqdm(SAMPLES, desc='OPENVINO GENERATEING OUTPUTS', total=len(SAMPLES)):
        result = model([convert_any_to_numpy(sample)])
        for key, value in result.items():
            result = convert_any_to_torch_tensor(value).reshape([1, -1])
        openvino_results.append(result)

    # compute simulating error
    error = []
    for ref, real in zip(ref_results, openvino_results):
        error.append(torch_snr_error(ref, real))
    error = sum(error) / len(error) * 100
    print(f'PPQ INT8 Simulating Error: {error: .3f} %')

# benchmark with openvino int8
print(f'Start Benchmark with openvino (Batchsize = {BATCHSIZE})')
benchmark_sample = np.zeros(shape=INPUTSHAPE, dtype=np.float32)
benchmark_sample = convert_any_to_numpy(benchmark_sample)

model = openvino_executor.compile_model(
    model = openvino_executor.read_model(model="modelzoo\yolo.onnx"), device_name="CPU")
tick = time.time()
for iter in tqdm(range(BENCHMARK_SAMPLES), desc='FP32 benchmark...'):
    result = model([benchmark_sample])
tok  = time.time()
print(f'Time span (FP32 MODE): {tok - tick : .4f} sec')

model = openvino_executor.compile_model(
    model = openvino_executor.read_model(model="model_int8.onnx"), device_name="CPU")
tick = time.time()
for iter in tqdm(range(BENCHMARK_SAMPLES), desc='INT8 benchmark...'):
    result = model([benchmark_sample])
tok  = time.time()
print(f'Time span (INT8 MODE): {tok - tick  : .4f} sec')
