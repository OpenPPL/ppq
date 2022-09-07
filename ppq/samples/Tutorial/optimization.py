from typing import Callable, Iterable

import torch
import torchvision
from ppq import QuantizationSettingFactory, TargetPlatform
from ppq.api import (ENABLE_CUDA_KERNEL, QuantizationSettingFactory,
                     quantize_torch_model)
from ppq.core import QuantizationStates
from ppq.executor.torch import TorchExecutor
from ppq.IR.quantize import QuantableOperation

# ------------------------------------------------------------
# 在这个例子中，我们将向你介绍如何自定义量化优化过程，以及如何手动调用优化过程
# ------------------------------------------------------------

BATCHSIZE   = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE      = 'cuda'
PLATFORM    = TargetPlatform.TRT_INT8

# ------------------------------------------------------------
# 和往常一样，我们要创建 calibration 数据，以及加载模型
# ------------------------------------------------------------
def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
CALIBRATION = load_calibration_dataset()

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# ------------------------------------------------------------
# 下面，我们将向你展示如何不借助 QSetting 来自定义优化过程
# QSetting 中包含了 PPQ 官方量化过程的配置参数，你可以借助它来调用所有系统内置优化过程
# 但如果你设计了新的优化过程，你将必须在合适的时机手动启动他们
# ------------------------------------------------------------
QSetting = QuantizationSettingFactory.default_setting()
# 不要进行 Parameter Baking 操作，一旦 Parameter 完成 Baking，后续任何对于参数的修改都是不被允许的
# 你可以设置 baking_parameter = True 并再次执行这个脚本，PPQ 系统会拒绝后续修改 scale 的请求
QSetting.quantize_parameter_setting.baking_parameter = False

# ------------------------------------------------------------
# 定义我们自己的优化过程，继承 QuantizationOptimizationPass 基类，实现 optimize 接口
# 在 optimize 接口函数中，你可以修改图的属性从而实现特定目的
# 在这个例子中，我们将图中所有卷积的输入 scale 变换为原来的两倍
# 同时，我们解除最后一个 Gemm 的输入量化
# ------------------------------------------------------------
from ppq import BaseGraph, QuantizationOptimizationPass, TorchExecutor
class MyOptim(QuantizationOptimizationPass):
    def optimize(self, graph: BaseGraph, dataloader: Iterable, 
                 collate_fn: Callable, executor: TorchExecutor, **kwargs) -> None:
        # graph.operations 是一个包含了图中所有 op 的字典
        for name, op in graph.operations.items():
            
            # 从图中找出所有已经量化的卷积算子
            # 对于你的网络而言，并非所有算子最终都会被量化，他们会受到 调度策略 和 Quantizer策略 的双重限制
            # 因此我们要使用 isinstance(op, QuantableOperation) 来判断它是否是一个量化的算子
            if op.type == 'Conv' and isinstance(op, QuantableOperation):
                
                # 对于卷积算子，它可能有 2-3 个输入，其中第二个输入为权重，第三个输入为 bias
                # 我们修改权重量化信息的 scale
                op.input_quant_config[1].scale *= 2                
                print(f'Input scale of Op {name} has been enlarged.')

            # 我们接下来解除 Gemm 的量化，在这里 mobilenet_v2 网络只有一个 Gemm 层
            # 所以我们将所有遇到的 Gemm 的层全部解除量化
            if op.type == 'Gemm' and isinstance(op, QuantableOperation):

                # config_with_variable 接口将返回量化算子的所有量化信息————包括输入与输出
                for cfg, _ in op.config_with_variable:

                    # 在 PPQ 中有许多方法可以切换算子的量化状态
                    # 将量化状态直接设置为 FP32，即解除了算子的量化
                    cfg.state = QuantizationStates.FP32

                # 也可以直接调用算子的 dequantize 方法
                # op.dequantize()

# ------------------------------------------------------------
# 如果你使用 ENABLE_CUDA_KERNEL 方法
# PPQ 将会尝试编译自定义的高性能量化算子，这一过程需要编译环境的支持
# 如果你在编译过程中发生错误，你可以删除此处对于 ENABLE_CUDA_KERNEL 方法的调用
# 这将显著降低 PPQ 的运算速度；但即使你无法编译这些算子，你仍然可以使用 pytorch 的 gpu 算子完成量化
# ------------------------------------------------------------
with ENABLE_CUDA_KERNEL():
    quantized = quantize_torch_model(
        model=model, calib_dataloader=CALIBRATION,
        calib_steps=32, input_shape=INPUT_SHAPE,
        setting=QSetting, collate_fn=collate_fn, platform=PLATFORM,
        onnx_export_file='Output/model.onnx', device=DEVICE, verbose=0)

    # ------------------------------------------------------------
    # 在完成量化流程之后，我们调用我们自定义的量化优化过程从而修改量化参数
    # ------------------------------------------------------------
    optim = MyOptim(name='My Optimization Procedure')
    optim.optimize(graph=quantized, dataloader=CALIBRATION, 
                   collate_fn=INPUT_SHAPE, executor=TorchExecutor(quantized, device=DEVICE))
