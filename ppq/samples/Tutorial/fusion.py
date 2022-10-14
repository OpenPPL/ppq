from typing import Callable, Iterable

import torch
import torchvision

from ppq import (BaseGraph, QuantizationOptimizationPass,
                 QuantizationOptimizationPipeline, QuantizationSetting,
                 TargetPlatform, TorchExecutor)
from ppq.api import ENABLE_CUDA_KERNEL
from ppq.executor.torch import TorchExecutor
from ppq.IR.quantize import QuantableOperation
from ppq.IR.search import SearchableGraph
from ppq.quantization.optim import (ParameterQuantizePass,
                                    PassiveParameterQuantizePass,
                                    QuantAlignmentPass,
                                    QuantizeSimplifyPass,
                                    RuntimeCalibrationPass)
from ppq.quantization.quantizer import TensorRTQuantizer

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
# 下面，我们将向你展示如何自定义图融合过程
# 图融合过程将改变量化方案，PPQ 使用 Tensor Quantization Config
# 来描述图融合的具体规则，其底层由并查集进行实现
# ------------------------------------------------------------

# ------------------------------------------------------------
# 定义我们自己的图融合过程，在这里我们将尝试进行 Conv - Clip 的融合
# 但与平常不同的是，我们将关闭 Clip 之后的量化点，保留 Conv - Clip 中间的量化
# 对于更为复杂的模式匹配，你可以参考 ppq.quantization.optim.refine.SwishFusionPass
# ------------------------------------------------------------
class MyFusion(QuantizationOptimizationPass):
    def optimize(self, graph: BaseGraph, dataloader: Iterable,
                 collate_fn: Callable, executor: TorchExecutor, **kwargs) -> None:
        
        # 图融合过程往往由图模式匹配开始，让我们建立一个模式匹配引擎
        search_engine = SearchableGraph(graph=graph)
        for pattern in search_engine.pattern_matching(patterns=['Conv', 'Clip'], edges=[[0, 1]], exclusive=True):
            conv, relu = pattern

            # 匹配到图中的 conv - relu 对，接下来关闭不必要的量化点
            # 首先我们检查 conv - relu 是否都是量化算子，是否处于同一平台
            is_quantable = isinstance(conv, QuantableOperation) and isinstance(relu, QuantableOperation)
            is_same_plat = conv.platform == relu.platform

            if is_quantable and is_same_plat:
                # 将 relu 输入输出的量化全部指向 conv 输出
                # 一旦调用 dominated_by 完成赋值，则调用 dominated_by 的同时
                # PPQ 会将 relu.input_quant_config[0] 与 relu.output_quant_config[0] 的状态置为 OVERLAPPED
                # 在后续运算中，它们所对应的量化不再起作用
                relu.input_quant_config[0].dominated_by = conv.output_quant_config[0]
                relu.output_quant_config[0].dominated_by = conv.output_quant_config[0]

# ------------------------------------------------------------
# 自定义图融合的过程将会干预量化器逻辑，我们需要新建量化器
# 此处我们继承 TensorRT Quantizer，算子的量化逻辑将使用 TensorRT 的配置
# 但在生成量化管线时，我们将覆盖量化器原有的逻辑，使用我们自定义的管线
# 这样我们就可以把自定义的图融合过程放置在合适的位置上，而此时 QuantizationSetting 也不再起作用
# ------------------------------------------------------------
class MyQuantizer(TensorRTQuantizer):
    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        return QuantizationOptimizationPipeline([
            QuantizeSimplifyPass(),
            ParameterQuantizePass(),
            MyFusion(name='My Optimization Procedure'),
            RuntimeCalibrationPass(),
            QuantAlignmentPass(),
            PassiveParameterQuantizePass()])

from ppq.api import quantize_torch_model, register_network_quantizer
register_network_quantizer(quantizer=MyQuantizer, platform=TargetPlatform.EXTENSION)

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
        collate_fn=collate_fn, platform=TargetPlatform.EXTENSION,
        onnx_export_file='model.onnx', device=DEVICE, verbose=0)
