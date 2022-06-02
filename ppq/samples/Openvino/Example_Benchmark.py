# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 openvino 对 PPQ 导出的模型进行推理
# 你需要注意，openvino 也可以运行各种各样的量化方案，你甚至可以用 tensorRT 的 policy
# 但总的来说，openvino 需要非对称量化的 activation 和对称量化的 weights
# ---------------------------------------------------------------

# For this onnx inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
import numpy as np
import openvino
import torch
import torchvision
import torchvision.models
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppq import *
from ppq.api import *

QUANT_PLATFROM = TargetPlatform.PPL_CUDA_INT8
BATCHSIZE = 16
MODELS = {
    'resnet50': torchvision.models.resnet50,
    'mobilenet_v2': torchvision.models.mobilenet.mobilenet_v2,
    'mnas': torchvision.models.mnasnet0_5,
    'shufflenet': torchvision.models.shufflenet_v2_x1_0}
DEVICE  = 'cuda'
SAMPLES = [torch.rand(size=[BATCHSIZE, 3, 224, 224]) for _ in range(256)]

for mname, model_builder in MODELS.items():
    print(f'Ready for run quantization with {mname}')
    model = model_builder(pretrained = True).to(DEVICE)
    
    quantized = quantize_torch_model(
        model=model, calib_dataloader=SAMPLES, collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=[BATCHSIZE, 3, 224, 224],
        setting=QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM,
        onnx_export_file='model_fp32.onnx')

    executor = TorchExecutor(graph=quantized)
    ref_results = []
    for sample in tqdm(SAMPLES, desc='PPQ GENERATEING REFERENCES', total=len(SAMPLES)):
        result = executor.forward(inputs=sample.to(DEVICE))[0]
        result = result.cpu().reshape([BATCHSIZE, 1000])
        ref_results.append(result)
    
    fp32_input_names  = [name for name, _ in quantized.inputs.items()]
    fp32_output_names = [name for name, _ in quantized.outputs.items()]
    
    graphwise_error_analyse(graph=quantized, running_device='cuda', 
                            dataloader=SAMPLES, collate_fn=lambda x: x.cuda(), steps=32)
    
    export_ppq_graph(graph=quantized, platform=TargetPlatform.OPENVINO_INT8,
                     graph_save_to='model_int8.onnx')

    int8_input_names  = [name for name, _ in quantized.inputs.items()]
    int8_output_names = [name for name, _ in quantized.outputs.items()]

    # run with openvino.
    # do not use Tensorrt provider to run quantized model.
    # TensorRT provider needs another qdq format.
    import openvino.runtime
    openvino_executor = openvino.runtime.Core()
    model = openvino_executor.compile_model(
        model = openvino_executor.read_model(model="model_int8.onnx"), device_name="CPU")
    openvino_results = []
    for sample in tqdm(SAMPLES, desc='OPENVINO GENERATEING OUTPUTS', total=len(SAMPLES)):
        result = model([convert_any_to_numpy(sample)])
        for key, value in result.items():
            result = convert_any_to_torch_tensor(value).reshape([BATCHSIZE, 1000])
        openvino_results.append(result)

    # compute simulating error
    error = []
    for ref, real in zip(ref_results, openvino_results):
        error.append(torch_snr_error(ref, real))
    error = sum(error) / len(error) * 100
    print(f'PPQ INT8 Simulating Error: {error: .3f} %')
    
    # benchmark with openvino int8
    print(f'Start Benchmark with openvino (Batchsize = {BATCHSIZE})')
    benchmark_samples = [np.zeros(shape=[BATCHSIZE, 3, 224, 224], dtype=np.float32) for _ in range(512)]
    
    model = openvino_executor.compile_model(
        model = openvino_executor.read_model(model="model_fp32.onnx"), device_name="CPU")
    tick = time.time()
    for sample in tqdm(benchmark_samples, desc='FP32 benchmark...'):
        result = model([convert_any_to_numpy(sample)])
    tok  = time.time()
    print(f'Time span (FP32 MODE): {tok - tick : .4f} sec')
    
    model = openvino_executor.compile_model(
        model = openvino_executor.read_model(model="model_int8.onnx"), device_name="CPU")
    tick = time.time()
    for sample in tqdm(benchmark_samples, desc='INT8 benchmark...'):
        result = model([convert_any_to_numpy(sample)])
    tok  = time.time()
    print(f'Time span (INT8 MODE): {tok - tick  : .4f} sec')