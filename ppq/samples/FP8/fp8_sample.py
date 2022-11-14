import torch
from torchvision import models

import ppq.lib as PFL
from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
from ppq.api import ENABLE_CUDA_KERNEL, load_torch_model
from ppq.core.quant import (QuantizationPolicy, QuantizationProperty,
                            RoundingPolicy)
from ppq.quantization.optim import (LearnedStepSizePass, ParameterBakingPass,
                                    ParameterQuantizePass,
                                    RuntimeCalibrationPass)

# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何使用 FP8 量化一个 Pytorch 模型
# 我们使用随机数据进行量化，这并不能得到好的量化结果。
# 在量化你的网络时，你应当使用真实数据和正确的预处理。
# ------------------------------------------------------------
model = models.efficientnet_b0(pretrained=True)
graph = load_torch_model(
    model, sample=torch.zeros(size=[1, 3, 224, 224]).cuda())
dataset = [torch.rand(size=[1, 3, 224, 224]) for _ in range(64)]

# -----------------------------------------------------------
# 我们将借助 PFL - PPQ Foundation Library, 即 PPQ 基础类库完成量化
# 这是 PPQ 自 0.6.6 以来推出的新的量化 api 接口，这一接口是提供给
# 算法工程师、部署工程师、以及芯片研发人员使用的，它更为灵活。
# 我们将手动使用 Quantizer 完成算子量化信息初始化, 并且手动完成模型的调度工作
# ------------------------------------------------------------
collate_fn  = lambda x: x.to('cuda')

# ------------------------------------------------------------
# 在开始之前，我需要向你介绍量化器、量化信息以及调度表
# 量化信息在 PPQ 中是由 TensorQuantizationConfig(TQC) 进行描述的
# 这个结构体描述了我要如何去量化一个数据，其中包含了量化位宽、量化策略、
# 量化 Scale, offset 等内容。
# ------------------------------------------------------------
from ppq import TensorQuantizationConfig as TQC

MyTQC = TQC(
    policy = QuantizationPolicy(
        QuantizationProperty.SYMMETRICAL + 
        QuantizationProperty.FLOATING +
        QuantizationProperty.PER_TENSOR + 
        QuantizationProperty.POWER_OF_2),
    rounding=RoundingPolicy.ROUND_HALF_EVEN,
    num_of_bits=8, quant_min=-448.0, quant_max=448.0, 
    exponent_bits=3, channel_axis=None,
    observer_algorithm='minmax'
)
# ------------------------------------------------------------
# 作为示例，我们创建了一个 "浮点" "对称" "Tensorwise" 的量化信息
# 这三者皆是该量化信息的 QuantizationPolicy 的一部分
# 同时要求该量化信息使用 ROUND_HALF_EVEN 方式进行取整
# 量化位宽为 8 bit，其中指数部分为 3 bit
# 量化上限为 448.0，下限则为 -448.0
# 这是一个 Tensorwise 的量化信息，因此 channel_axis = None
# observer_algorithm 表示在未来使用 minmax calibration 方法确定该量化信息的 scale

# 上述例子完成了该 TQC 的初始化，但并未真正启用该量化信息
# MyTQC.scale, MyTQC.offset 仍然为空，它们必须经过 calibration 才会具有有意义的值
# 并且他目前的状态 MyTQC.state 仍然是 Quantization.INITIAL，这意味着在计算时该 TQC 并不会参与运算。
# ------------------------------------------------------------

# ------------------------------------------------------------
# 接下来我们向你介绍量化器，这是 PPQ 中的一个核心类型
# 它的职责是为网络中所有处于量化区的算子初始化量化信息(TQC)
# PPQ 中实现了一堆不同的量化器，它们分别适配不同的情形
# 在这个例子中，我们分别创建了 TRT_INT8, GRAPHCORE_FP8, TRT_FP8 三种不同的量化器
# 由它们所生成的量化信息是不同的，为此你可以访问它们的源代码
# 位于 ppq.quantization.quantizer 中，查看它们初始化量化信息的逻辑。
# ------------------------------------------------------------
_ = PFL.Quantizer(platform=TargetPlatform.TRT_INT8, graph=graph)      # 取得 TRT_INT8 所对应的量化器
_ = PFL.Quantizer(platform=TargetPlatform.GRAPHCORE_FP8, graph=graph) # 取得 GRAPHCORE_FP8 所对应的量化器
quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # 取得 TRT_FP8 所对应的量化器

# ------------------------------------------------------------
# 调度器是 PPQ 中另一核心类型，它负责切分计算图
# 在量化开始之前，你的计算图将被切分成可量化区域，以及不可量化区域
# 不可量化区域往往就是那些执行 Shape 推断的算子所构成的子图
# *** 量化器只为量化区的算子初始化量化信息 ***
# 调度信息将被写在算子的属性中，你可以通过 op.platform 来访问每一个算子的调度信息
# ------------------------------------------------------------
dispatching = PFL.Dispatcher(graph=graph).dispatch(                       # 生成调度表
    quant_types=quantizer.quant_operation_types)

for op in graph.operations.values():
    # quantize_operation - 为算子初始化量化信息，platform 传递了算子的调度信息
    # 如果你的算子被调度到 TargetPlatform.FP32 上，则该算子不量化
    # 你可以手动修改调度信息
    dispatching['Op1'] = TargetPlatform.FP32        # 将 Op1 强行送往非量化区
    dispatching['Op2'] = TargetPlatform.UNSPECIFIED # 将 Op2 强行送往量化区
    
    # 你可能已经注意到了，我们并没有将 Op2 送往 TRT_FP8，而是将其送往 UNSPECIFIED 平台
    # 其含义是告诉量化器我们 "建议" 量化器对算子初始化量化信息，如果此时 Op2 不是量化器所支持的类型，则该算子仍然不会被量化
    # 但如果我们直接将 Op2 送往 TRT_FP8，不论如何该算子都将被量化
    
    quantizer.quantize_operation(
        op_name = op.name, platform = dispatching[op.name])

# ------------------------------------------------------------
# 在创建量化管线之前，我们需要初始化执行器，它用于模拟硬件并执行你的网络
# 请注意，执行器需要对网络结果进行分析并缓存分析结果，如果你的网络结构发生变化
# 你必须重新建立新的执行器。在上一步操作中，我们对算子进行了量化，这使得
# 普通的算子被量化算子替代，这一步操作将会改变网络结构。因此我们必须在其后建立执行器。
# ------------------------------------------------------------
executor = TorchExecutor(graph=graph, device='cuda')
executor.tracing_operation_meta(inputs=collate_fn(dataset[0]))
executor.load_graph(graph=graph)

# ------------------------------------------------------------
# 下面的过程将创建量化管线，它还是一个 PPQ 的核心类型
# 在 PPQ 中，模型的量化是由一个一个的量化过程(QuantizationOptimizationPass)完成的
# 量化管线 是 量化过程 的集合，在其中的量化过程将被逐个调用
# 从而实现对 TQC 中内容的修改，最终实现模型的量化
# 在这里我们为管线中添加了 4 个量化过程，分别处理不同的内容

# ParameterQuantizePass - 用于为模型中的所有参数执行 Calibration, 生成它们的 scale，并将对应 TQC 的状态调整为 ACTIVED
# RuntimeCalibrationPass - 用于为模型中的所有激活执行 Calibration, 生成它们的 scale，并将对应 TQC 的状态调整为 ACTIVED
# LearnedStepSizePass - 用于训练微调模型的权重，从而降低量化误差
# ParameterBakingPass - 用于执行模型参数烘焙

# 在 PPQ 中我们提供了数十种不同的 QuantizationOptimizationPass
# 你可以组合它们从而实现自定义的功能，也可以继承 QuantizationOptimizationPass 基类
# 从而创造出新的量化优化过程
# ------------------------------------------------------------
pipeline = PFL.Pipeline([
    ParameterQuantizePass(),
    RuntimeCalibrationPass(),
    LearnedStepSizePass(
        steps=1000, is_scale_trainable=False, 
        lr=1e-4, block_size=4, collecting_device='cuda'),
    ParameterBakingPass()
])

with ENABLE_CUDA_KERNEL():
    # 调用管线完成量化
    pipeline.optimize(
        graph=graph, dataloader=dataset, verbose=True, 
        calib_steps=32, collate_fn=collate_fn, executor=executor)

    # 执行量化误差分析
    graphwise_error_analyse(
        graph=graph, running_device='cuda', 
        dataloader=dataset, collate_fn=collate_fn)

# ------------------------------------------------------------
# 在最后，我们导出计算图
# 同样地，我们根据不同推理框架的需要，写了一堆不同的网络导出逻辑
# 你通过参数 platform 告诉 PPQ 你的模型最终将部署在何处，
# PPQ 则会返回一个对应的 GraphExporter 对象，它将负责将 PPQ 的量化信息
# 翻译成推理框架所需的内容。你也可以自己写一个 GraphExporter 类并注册到 PPQ 框架中来。
# ------------------------------------------------------------
exporter = PFL.Exporter(platform=TargetPlatform.TRT_FP8)
exporter.export(file_path='Quantized.onnx', graph=graph)
