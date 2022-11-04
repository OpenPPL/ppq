import torch
from torchvision import models
from tqdm import tqdm
from ppq.IR.quantize import QuantableOperation
from ppq.api.interface import DEQUANTIZE_GRAPH

import ppq.lib as PFL
from ppq import (BaseGraph, TargetPlatform, TorchExecutor,
                 graphwise_error_analyse)
from ppq.api import ENABLE_CUDA_KERNEL, load_torch_model
from ppq.core.quant import (QuantizationPolicy, QuantizationProperty,
                            RoundingPolicy)
from ppq.quantization.optim import (LearnedStepSizePass, ParameterBakingPass,
                                    ParameterQuantizePass, QuantAlignmentPass,
                                    QuantizeFusionPass, QuantizeSimplifyPass,
                                    RuntimeCalibrationPass)

# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何在 PPQ 中对你的网络进行量化感知训练
# 在 PPQ 中，所有的训练过程均无需带标签数据，训练的目标是使得浮点模型结果与量化模型结果尽可能的接近
# 因此我们会先运行浮点模型，收集浮点模型的结果，并以此指导量化模型的训练
# PPQ 模型的训练过程与 Pytorch 遵循相同的逻辑，你可以使用 Pytorch 中的技巧来获得更好的训练效果
# ------------------------------------------------------------
model = models.mobilenet_v2(pretrained=True)
graph = load_torch_model(
    model, sample=torch.zeros(size=[1, 3, 224, 224]).cuda())

calibration_dataset = [torch.rand(size=[1, 3, 224, 224]) for _ in range(64)]
training_dataset    = [torch.rand(size=[1, 3, 224, 224]) for _ in range(64)]

# ------------------------------------------------------------
# 我们首先进行标准的量化流程，为所有算子初始化量化信息，并进行 Calibration
# 完成标准流程后，我们测算此时模型的误差情况
# ------------------------------------------------------------
collate_fn  = lambda x: x.to('cuda')
quantizer   = PFL.Quantizer(platform=TargetPlatform.TRT_INT8, graph=graph) # 取得 TRT_INT8 所对应的量化器
dispatching = PFL.Dispatcher(graph=graph).dispatch(                      # 生成调度表
    quant_types=quantizer.quant_operation_types)

# 为算子初始化量化信息
for op in graph.operations.values():
    quantizer.quantize_operation(
        op_name = op.name, platform = dispatching[op.name])

# 初始化执行器
executor = TorchExecutor(graph=graph, device='cuda')
executor.tracing_operation_meta(inputs=collate_fn(calibration_dataset[0]))
executor.load_graph(graph=graph)

# 创建优化管线，由于后续还要继续训练我们的模型，我们不能在此处调用
# ParameterBakingPass()，一旦模型权重完成烘焙，则它们不能被进一步调整
pipeline = PFL.Pipeline([
    QuantizeSimplifyPass(),
    QuantizeFusionPass(
        activation_type=quantizer.activation_fusion_types),
    ParameterQuantizePass(),
    RuntimeCalibrationPass(),
    QuantAlignmentPass(force_overlap=True),
])

with ENABLE_CUDA_KERNEL():
    # 调用管线完成量化
    pipeline.optimize(
        graph=graph, dataloader=calibration_dataset, verbose=True, 
        calib_steps=32, collate_fn=collate_fn, executor=executor)

    # 执行量化误差分析
    graphwise_error_analyse(
        graph=graph, running_device='cuda', 
        dataloader=calibration_dataset, collate_fn=collate_fn)

    # ------------------------------------------------------------
    # 完成量化后，我们将开始着手收集浮点模型的执行结果
    # 我们将指定网络最后的输出变量作为我们的关注对象
    # 你可以使用可视化工具打开你的网络模型，查看网络输出的名字
    # 我们将收集网络末端卷积层的输出结果，并将其作为标准，以此来训练量化模型
    # ------------------------------------------------------------
    output_names  = [k for k, _ in graph.outputs.items()]
    loss_fn       = torch.nn.MSELoss().cuda()
    optimizer     = torch.optim.Adam(params=graph.parameters(), lr=1e-5)
    max_grad_norm = 1
    for parameter in graph.parameters(): parameter.requires_grad = True

    print(f'Network has {len(output_names)} outputs variable: {output_names}, '
          'PPQ will collect their value for finetuning your network.')

    for epoch in range(20):
        for batch in tqdm(training_dataset):
            batch = collate_fn(batch)

            with DEQUANTIZE_GRAPH(graph=graph):
                fp_outputs = executor.forward(batch, output_names=output_names)
            qt_outputs = executor.forward_with_gradient(batch, output_names=output_names)

            loss = 0
            for fp_output, qt_output in zip(fp_outputs, qt_outputs):
                loss += loss_fn(fp_output, qt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(graph.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        print('Training Loss : %.4f ' % loss.item())

    # 执行量化误差分析
    graphwise_error_analyse(
        graph=graph, running_device='cuda', 
        dataloader=calibration_dataset, collate_fn=collate_fn)