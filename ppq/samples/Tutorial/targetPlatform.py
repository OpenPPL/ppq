from typing import Iterable

import torch
import torchvision
from ppq import (BaseQuantizer, Operation, OperationQuantizationConfig,
                 TargetPlatform)
from ppq.api import (ENABLE_CUDA_KERNEL, export_ppq_graph,
                     quantize_torch_model, register_network_exporter,
                     register_network_quantizer)
from ppq.core import (QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, convert_any_to_numpy)
from ppq.IR import BaseGraph, Operation, QuantableOperation

# ------------------------------------------------------------
# 在这个脚本中，我们将创建一个新的量化平台，定义我们自己的量化规则
# 这意味着我们将创建自己的 Quantizer，并调节 QuantSetting 中的各项属性
# ------------------------------------------------------------

BATCHSIZE   = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE      = 'cuda'
PLATFORM    = TargetPlatform.TRT_INT8

def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
CALIBRATION = load_calibration_dataset()

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# ------------------------------------------------------------
# 定义自己的量化器，并完成注册
# 自定义量化器需要你完成所有接口函数与接口属性（接口属性必须以@property修饰）
# ------------------------------------------------------------
class MyQuantizer(BaseQuantizer):
    
    # ------------------------------------------------------------
    # quant_operation_types 是一个类型枚举，在这里你需要写下所有该量化器所需要量化的算子
    # ------------------------------------------------------------
    @ property
    def quant_operation_types(self) -> set:
        return {'Conv'}

    # ------------------------------------------------------------
    # 一旦你确定了那些算子需要量化，则需要在 init_quantize_config 为他们初始化量化信息
    # 然而你需要注意的是，由于手动调度的存在，用户可以强制调度一个类型不在 quant_operation_types 中的算子来到量化平台
    # 我建议你针对这类情况进行回应。或者，在探测到算子类型并非可量化类型后进行报错
    # ------------------------------------------------------------
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        # ------------------------------------------------------------
        # 为卷积算子初始化量化信息，只量化卷积算子的输出
        # ------------------------------------------------------------
        if operation.type == 'Conv':
            config = self.create_default_quant_config(
                op                 = operation, 
                num_of_bits        = 8,
                quant_max          = 127, 
                quant_min          = -128,
                observer_algorithm = 'percentile', 
                policy             = QuantizationPolicy(
                    QuantizationProperty.PER_TENSOR +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.SYMMETRICAL),
                rounding           = RoundingPolicy.ROUND_HALF_EVEN)

            # ------------------------------------------------------------
            # 关闭所有输入量化，状态设置为fp32
            # ------------------------------------------------------------
            for tensor_quant_config in config.input_quantization_config:
                tensor_quant_config.state = QuantizationStates.FP32

            return config
        else:
            raise TypeError(f'Unsupported Op Type: {operation.type}')

    # ------------------------------------------------------------
    # 当前量化器进行量化的算子都将被发往一个指定的目标平台
    # 这里我们选择 TargetPlatform.EXTENSION 作为目标平台
    # ------------------------------------------------------------
    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.EXTENSION

    @ property
    def activation_fusion_types(self) -> set:
        # 列举此处的算子会与他们之前的 卷积 和 矩阵乘法执行激活函数图融合
        # 从而消去一些错误的量化行为，对于更为复杂的图融合，你必须手写一个单独的优化过程
        # 不过我们这里没有声明 relu, clip 是量化类型，因此图融合并不会起作用
        
        # 图融合起作用的前提是参与融合的所有算子，全部都需要被量化
        return {'Relu', 'Clip'}

# ------------------------------------------------------------
# 注册我们的量化器，目标平台为 TargetPlatform.EXTENSION
# 后面我们将以这个平台来调用量化器
# ------------------------------------------------------------
register_network_quantizer(MyQuantizer, TargetPlatform.EXTENSION)

with ENABLE_CUDA_KERNEL():
    # ------------------------------------------------------------
    # 以 TargetPlatform.EXTENSION 作为目标平台调用量化器
    # ------------------------------------------------------------
    quantized = quantize_torch_model(
        model=model, calib_dataloader=CALIBRATION,
        calib_steps=32, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn, platform=TargetPlatform.EXTENSION,
        onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

import onnx
# ------------------------------------------------------------
# 完成量化后，我们开始导出量化信息，这里我们注册一个网络量化信息导出器
# 对于一个网络导出器而言，你需要关注以下两件事：
#   1. 对于一些场景而言，你需要导出 PPQ 的量化信息到特定的文件
#   2. 有些时候你使用 PPQ 训练过你的网络，你可能还需要导出训练后的网络权重
# 我们这里向你展示如何导出量化信息到文本文档，而网络权重的导出则继承于 onnx
# ------------------------------------------------------------
from ppq.parser import OnnxExporter


class MyExporter(OnnxExporter):
    def convert_value(self, value: torch.Tensor) -> str:
        if type(value) in {int, float}: return value
        else:
            value = convert_any_to_numpy(value, accept_none=True)
            if value is None: return value # SOI config has Nona as its scale and
            return value.tolist()
    
    def export(self, file_path: str, graph: BaseGraph, config_path: str, **kwargs):
        # ------------------------------------------------------------
        # 接下来我们将导出量化信息，在 PPQ 中所有的量化信息都绑定在 Op 上
        # 因此我们需要遍历图中所有的 Op, 将绑定在其上的量化信息导出到文件
        # ------------------------------------------------------------
        with open(config_path, 'w') as file:
            for name, op in graph.operations.items():
                if not isinstance(op, QuantableOperation): continue

                for cfg, var in op.config_with_variable:
                    file.write(f"{name}: {var.name}\n")
                    file.write(f"Quant State: {cfg.state.name}\n")
                    file.write(f"Scale:  {self.convert_value(cfg.scale)}\n")
                    file.write(f"Offset: {self.convert_value(cfg.offset)}\n")

        # ------------------------------------------------------------
        # 最后我们导出完整的计算图到 onnx
        # ------------------------------------------------------------
        onnx.save(self.export_graph(graph=graph), file_path)


# ------------------------------------------------------------
# 注册导出器并导出模型
# ------------------------------------------------------------
register_network_exporter(exporter=MyExporter, platform=TargetPlatform.EXTENSION)
export_ppq_graph(graph=quantized, platform=TargetPlatform.EXTENSION, 
                 graph_save_to='Output/model.onnx', config_save_to='Output/model.json')
