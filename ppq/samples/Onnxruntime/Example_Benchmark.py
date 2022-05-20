# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 onnxruntime 对 PPQ 导出的模型进行推理
# 你需要注意，Onnxruntime 可以运行各种各样的量化方案，但模型量化对 Onnxruntime 而言几乎无法起到加速作用
# 你可以使用 Onnxruntime 来验证量化方案以及 ppq 量化的正确性，但这不是一个合理的部署平台
# 修改 QUANT_PLATFROM 来使用不同的量化方案。

# This Script export ppq internal graph to onnxruntime,
# you should notice that onnx is designed as an Open Neural Network Exchange format.
# It has the capbility to describe most of ppq's quantization policy including combinations of:
#   Symmtrical, Asymmtrical, POT, Per-channel, Per-Layer
# However onnxruntime can not accelerate quantized model in most cases,
# you are supposed to use onnxruntime for verifying your network quantization result only.
# ---------------------------------------------------------------

# For this onnx inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES

import time

import numpy as np
import onnxruntime
import torch
import torchvision
import torchvision.models
from tqdm import tqdm

from ppq import *
from ppq.api import *

QUANT_PLATFROM = TargetPlatform.TRT_INT8
BATCHSIZE = 1
MODELS = {
    'resnet50': torchvision.models.resnet50,
    'mobilenet_v2': torchvision.models.mobilenet.mobilenet_v2,
    'mnas': torchvision.models.mnasnet0_5,
    'shufflenet': torchvision.models.shufflenet_v2_x1_0}
SAMPLES = [torch.rand(size=[BATCHSIZE, 3, 224, 224]) for _ in range(256)]
DEVICE  = 'cuda'

for mname, model_builder in MODELS.items():
    print(f'Ready for run quantization with {mname}')
    model = model_builder(pretrained = True).to(DEVICE)
    
    # quantize model with ppq.
    quantized = quantize_torch_model(
        model=model, calib_dataloader=SAMPLES, collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=[BATCHSIZE, 3, 224, 224],
        setting=QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM,
        onnx_export_file='model_fp32.onnx')

    # collect ppq execution result for validation
    executor = TorchExecutor(graph=quantized)
    ref_results = []
    for sample in tqdm(SAMPLES, desc='PPQ GENERATEING REFERENCES', total=len(SAMPLES)):
        result = executor.forward(inputs=sample.to(DEVICE))[0]
        result = result.cpu().reshape([BATCHSIZE, 1000])
        ref_results.append(result)
    
    # record input & output name
    fp32_input_names  = [name for name, _ in quantized.inputs.items()]
    fp32_output_names = [name for name, _ in quantized.outputs.items()]
    
    # quantization error analyse
    graphwise_error_analyse(graph=quantized, running_device='cuda', 
                            dataloader=SAMPLES, collate_fn=lambda x: x.cuda(), steps=32)
    
    # export model to disk.
    export_ppq_graph(
        graph=quantized, 
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to='model_int8.onnx')

    # record input & output name
    int8_input_names  = [name for name, _ in quantized.inputs.items()]
    int8_output_names = [name for name, _ in quantized.outputs.items()]

    # run with onnxruntime.
    # compare onnxruntime result with ppq.
    session = onnxruntime.InferenceSession('model_int8.onnx', providers=['CUDAExecutionProvider'])
    onnxruntime_results = []
    for sample in tqdm(SAMPLES, desc='ONNXRUNTIME GENERATEING OUTPUTS', total=len(SAMPLES)):
        result = session.run([int8_output_names[0]], {int8_input_names[0]: convert_any_to_numpy(sample)})
        result = convert_any_to_torch_tensor(result).reshape([BATCHSIZE, 1000])
        onnxruntime_results.append(result)

    # compute simulating error
    error = []
    for ref, real in zip(ref_results, onnxruntime_results):
        error.append(torch_snr_error(ref, real))
    error = sum(error) / len(error) * 100
    print(f'PPQ INT8 Simulating Error: {error: .3f} %')

    # -------------------------------
    print(f'Start Benchmark with onnxruntime (Batchsize = {BATCHSIZE})')
    benchmark_samples = [np.zeros(shape=[BATCHSIZE, 3, 224, 224], dtype=np.float32) for _ in range(512)]
    
    # benchmark with onnxruntime fp32
    session = onnxruntime.InferenceSession('model_fp32.onnx', providers=['CUDAExecutionProvider'])    
    tick = time.time()
    for sample in tqdm(benchmark_samples, desc='FP32 benchmark...'):
        session.run([fp32_output_names[0]], {fp32_input_names[0]: sample})
    tok  = time.time()
    print(f'Time span (FP32 MODE): {tok - tick : .4f} sec')
    
    # benchmark with onnxruntime int8
    session = onnxruntime.InferenceSession('model_int8.onnx', providers=['CUDAExecutionProvider'])    
    tick = time.time()
    for sample in tqdm(benchmark_samples, desc='INT8 benchmark...'):
        session.run([int8_output_names[0]], {int8_input_names[0]: sample})
    tok  = time.time()
    print(f'Time span (INT8 MODE): {tok - tick  : .4f} sec')
