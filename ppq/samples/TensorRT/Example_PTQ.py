import os

import numpy as np
import torch
import torchvision

from ppq import TargetPlatform, graphwise_error_analyse, TorchExecutor
from ppq.api import ENABLE_CUDA_KERNEL, export_ppq_graph, load_torch_model
from ppq.core import convert_any_to_numpy
from ppq.quantization.optim import *
import ppq.lib as PFL

calibration_dataloader = []
for file in os.listdir('imagenet'):
    path = os.path.join('imagenet', file)
    arr  = np.fromfile(path, dtype=np.dtype('float32')).reshape([1, 3, 224, 224])
    calibration_dataloader.append(torch.tensor(arr))

with ENABLE_CUDA_KERNEL():
    model = torchvision.models.mnasnet1_0(pretrained=True).cuda()
    graph = load_torch_model(model=model, sample=torch.zeros(size=[1, 3, 224, 224]).cuda())
    # ------------------------------------------------------------
    # 我们首先进行标准的量化流程，为所有算子初始化量化信息，并进行 Calibration
    # ------------------------------------------------------------
    quantizer   = PFL.Quantizer(platform=TargetPlatform.TRT_INT8, graph=graph) # 取得 TRT_INT8 所对应的量化器
    dispatching = PFL.Dispatcher(graph=graph).dispatch(          # 生成调度表
        quant_types=quantizer.quant_operation_types)

    # 为算子初始化量化信息
    for op in graph.operations.values():
        quantizer.quantize_operation(
            op_name = op.name, platform = dispatching[op.name])

    # 初始化执行器
    collate_fn = lambda x: x.to('cuda')
    executor = TorchExecutor(graph=graph, device='cuda')
    executor.tracing_operation_meta(inputs=torch.zeros(size=[1, 3, 224, 224]).cuda())
    executor.load_graph(graph=graph)

    # ------------------------------------------------------------
    # 创建优化管线，由于后续还要继续训练我们的模型，我们不能在此处调用
    # ParameterBakingPass()，一旦模型权重完成烘焙，则它们不能被进一步调整
    # ------------------------------------------------------------
    pipeline = PFL.Pipeline([
        QuantizeSimplifyPass(),
        QuantizeFusionPass(
            activation_type=quantizer.activation_fusion_types),
        ParameterQuantizePass(),
        RuntimeCalibrationPass(),
        PassiveParameterQuantizePass(),
        QuantAlignmentPass(force_overlap=True),
    ])

with ENABLE_CUDA_KERNEL():
    # 调用管线完成量化
    pipeline.optimize(
        graph=graph, dataloader=calibration_dataloader, verbose=True, 
        calib_steps=32, collate_fn=collate_fn, executor=executor)

    graphwise_error_analyse(
        graph=graph, running_device='cuda', dataloader=calibration_dataloader, 
        collate_fn=lambda x: x.cuda())

    export_ppq_graph(
        graph=graph, platform=TargetPlatform.TRT_INT8, 
        graph_save_to='Output/quantized.onnx', 
        config_save_to='Output/quantized.json')
    
    results, executor = [], TorchExecutor(graph=graph)
    for idx, data in enumerate(calibration_dataloader):
        arr = convert_any_to_numpy(executor(data.cuda())[0])
        arr.tofile(f'Output/Result/{idx}.bin')

    from ppq.utils.TensorRTUtil import build_engine
    build_engine(onnx_file='Output/quantized.onnx', int8_scale_file='Output/quantized.json', engine_file='Output/INT8.engine', int8=True, fp16=True)
    build_engine(onnx_file='Output/quantized.onnx', engine_file='Output/FP16.engine', int8=False, fp16=True)
    build_engine(onnx_file='Output/quantized.onnx', engine_file='Output/FP32.engine', int8=False, fp16=False)

