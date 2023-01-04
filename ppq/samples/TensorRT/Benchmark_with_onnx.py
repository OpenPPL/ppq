# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理，并进行速度测试
# 目前 GPU 上 tensorRT 是跑的最快的部署框架 ...
# ---------------------------------------------------------------

import time

import numpy as np
import tensorrt as trt
import torch
import torchvision
import torchvision.models
from tqdm import tqdm

import trt_infer
from ppq import *
from ppq.api import *

# Nvidia Nsight Performance Profile
QUANT_PLATFROM   = TargetPlatform.TRT_INT8
BATCHSIZE        = 1
SAMPLES          = [torch.zeros(size=[BATCHSIZE, 3, 640, 640]) for _ in range(32)]
DEVICE           = 'cuda'
MODEL_PATH       = 'models/yolov6s.onnx'
CFG_VALID_RESULT = False

def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    if CFG_VALID_RESULT:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            for sample in tqdm(samples, desc='TensorRT is running...'):
                inputs[0].host = convert_any_to_numpy(sample)
                [output] = trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)[0]
                results.append(convert_any_to_torch_tensor(output).reshape([-1, 1000]))
    else:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            inputs[0].host = convert_any_to_numpy(samples[0])
            for sample in tqdm(samples, desc='TensorRT is running...'):
                trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)
    return results

with ENABLE_CUDA_KERNEL():
    # export non-quantized model to tensorRT for benchmark
    non_quantized = quantize_onnx_model(
        onnx_import_file=MODEL_PATH, calib_dataloader=SAMPLES, 
        collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=[BATCHSIZE, 3, 640, 640],
        setting=QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM,
        do_quantize=False)

    export_ppq_graph(
        graph=non_quantized, 
        platform=TargetPlatform.ONNX,
        graph_save_to='model_fp32.onnx')
    builder = trt_infer.EngineBuilder()
    builder.create_network('model_fp32.onnx')
    builder.create_engine(engine_path='model_fp32.engine', precision="fp16")

    # quantize model with ppq.
    quantized = quantize_onnx_model(
        onnx_import_file=MODEL_PATH, calib_dataloader=SAMPLES, 
        collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=[BATCHSIZE, 3, 640, 640],
        setting=QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM)

    if CFG_VALID_RESULT:
        executor = TorchExecutor(graph=quantized)
        ref_results = []
        for sample in tqdm(SAMPLES, desc='PPQ GENERATEING REFERENCES', total=len(SAMPLES)):
            result = executor.forward(inputs=sample.to(DEVICE))[0]
            result = result.cpu().reshape([-1, 1000])
            ref_results.append(result)

    # export model to disk.
    export_ppq_graph(
        graph=quantized, 
        platform=TargetPlatform.TRT_INT8,
        graph_save_to='model_int8.onnx')

    if CFG_VALID_RESULT:
        # compute simulating error
        trt_outputs = infer_trt(
            model_path='model_int8.engine', 
            samples=[convert_any_to_numpy(sample) for sample in SAMPLES])

        error = []
        for ref, real in zip(ref_results, trt_outputs):
            ref = convert_any_to_torch_tensor(ref).float()
            real = convert_any_to_torch_tensor(real).float()
            error.append(torch_snr_error(ref, real))
        error = sum(error) / len(error) * 100
        print(f'Simulating Error: {error: .4f}%')

    # benchmark with onnxruntime int8
    benchmark_samples = [np.zeros(shape=[BATCHSIZE, 3, 224, 224], dtype=np.float32) for _ in range(512)]
    print(f'Start Benchmark with tensorRT (Batchsize = {BATCHSIZE})')
    tick = time.time()
    infer_trt(model_path='model_fp32.engine', samples=benchmark_samples)
    tok  = time.time()
    print(f'Time span (FP32 MODE): {tok - tick : .4f} sec')

    tick = time.time()
    infer_trt(model_path='model_int8.engine', samples=benchmark_samples)
    tok  = time.time()
    print(f'Time span (INT8 MODE): {tok - tick  : .4f} sec')
