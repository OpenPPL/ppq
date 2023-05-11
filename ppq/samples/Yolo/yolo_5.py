
import numpy as np
import torch

import ppq.lib as PFL
from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
from ppq.api import ENABLE_CUDA_KERNEL, export_ppq_graph, load_onnx_graph
from ppq.quantization.optim import *

calibration_dataloader = [torch.rand([1, 3, 640, 640]) for _ in range(32)]

with ENABLE_CUDA_KERNEL():
    graph = load_onnx_graph('Models/yolov5s.v5.onnx')

    quantizer   = PFL.Quantizer(platform=TargetPlatform.OPENVINO_INT8, graph=graph) # 取得 OPENVINO_INT8 所对应的量化器
    dispatching = PFL.Dispatcher(graph=graph).dispatch(                             # 生成调度表
        quant_types=quantizer.quant_operation_types)

    # ------------------------------------------------------------
    # Yolo5 前面可能有一坨 Slice, Concat 算子
    # 后面可能有一坨后处理算子，我们不希望量化它们，你可用下面的方法将它们解除量化
    # 不同模型中层的名字可能不一样，你需要按照你的模型对它们进行手动修改
    # ------------------------------------------------------------

    # Concat_40 往前的所有算子不量化
    from ppq.IR import SearchableGraph
    search_engine = SearchableGraph(graph)
    for op in search_engine.opset_matching(
        sp_expr=lambda x: x.name == 'Concat_40',
        rp_expr=lambda x, y: True,
        ep_expr=None, direction='up'
    ):
        dispatching[op.name] = TargetPlatform.FP32

    # Sigmoid_280 往后的所有算子不量化
    # Sigmoid_246 往后的所有算子不量化
    # Sigmoid_314 往后的所有算子不量化
    for op in search_engine.opset_matching(
        sp_expr=lambda x: x.name in {'Sigmoid_246', 'Sigmoid_280', 'Sigmoid_314'},
        rp_expr=lambda x, y: True,
        ep_expr=None, direction='down'
    ):
        dispatching[op.name] = TargetPlatform.FP32

    # 为算子初始化量化信息
    for op in graph.operations.values():
        quantizer.quantize_operation(
            op_name = op.name, platform = dispatching[op.name])

    # 初始化执行器
    collate_fn = lambda x: x.to('cuda')
    executor = TorchExecutor(graph=graph, device='cuda')
    executor.tracing_operation_meta(inputs=torch.zeros(size=[1, 3, 640, 640]).cuda())
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
        ParameterBakingPass()
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

from ppq.utils.TensorRTUtil import Benchmark, Profiling, build_engine

build_engine(
    onnx_file='Output/quantized.onnx', 
    engine_file='Output/quantized.engine', 
    int8=True, int8_scale_file='Output/quantized.json')
Benchmark('Output/quantized.engine')
Profiling('Output/quantized.engine')