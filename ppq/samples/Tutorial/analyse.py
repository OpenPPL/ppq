from typing import Iterable

import torch
import torchvision
from ppq import TargetPlatform, graphwise_error_analyse
from ppq.api import quantize_torch_model
from ppq.api.interface import ENABLE_CUDA_KERNEL
from ppq.quantization.analyse.graphwise import statistical_analyse
from ppq.quantization.analyse.layerwise import layerwise_error_analyse

# ------------------------------------------------------------
# 在 PPQ 中我们提供许多方法帮助你进行误差分析，误差分析是量化网络的第一步
# 这个脚本将以随机数据和 mobilenet v2 网络为例向你展示它们的使用方法
# ------------------------------------------------------------

BATCHSIZE   = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE      = 'cuda'
PLATFORM    = TargetPlatform.TRT_INT8

def load_calibration_dataset() -> Iterable:
    # ------------------------------------------------------------
    # 让我们从创建 calibration 数据开始做起， PPQ 需要你送入 32 ~ 1024 个样本数据作为校准数据集
    # 它们应该尽可能服从真实样本的分布，量化过程如同训练过程一样存在可能的过拟合问题
    # 你应当保证校准数据是经过正确预处理的、有代表性的数据，否则量化将会失败；校准数据不需要标签；数据集不能乱序
    # ------------------------------------------------------------
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
CALIBRATION = load_calibration_dataset()

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

# ------------------------------------------------------------
# 我们使用 mobilenet v2 作为一个样例模型
# PPQ 将会使用 torch.onnx.export 函数 把 pytorch 的模型转换为 onnx 模型
# 对于复杂的 pytorch 模型而言，你或许需要自己完成 pytorch 模型到 onnx 的转换过程
# ------------------------------------------------------------
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

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
        collate_fn=collate_fn, platform=PLATFORM,
        onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

    # ------------------------------------------------------------
    # graphwise_error_analyse 是最常用的分析方法，它分析网络中的量化误差情况，它的结果将直接打印在屏幕上
    # 对于 graphwise_error_analyse 而言，算子的误差直接衡量了量化网络与浮点网络之间的输出误差
    # 这一误差是累积的，意味着网络后面的算子总是会比网络前面的算子拥有更高的输出误差
    # 留意网络输出的误差情况，如果你想获得一个精度较高的量化网络，那么那些靠近输出的节点误差不应超过 10%
    # 该方法只衡量 Conv, Gemm 算子的误差情况，如果你对其余算子的误差情况感兴趣，需要手动修改方法逻辑
    # ------------------------------------------------------------
    reports = graphwise_error_analyse(
        graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
        dataloader=CALIBRATION)

    # ------------------------------------------------------------
    # layerwise_error_analyse 是更为强大的分析方法，它分析算子的量化敏感性
    # 与 graphwise_error_analyse 不同，该方法分析的误差不是累计的
    # 该方法首先解除网络中所有算子的量化，而后单独地量化每一个 Conv, Gemm 算子
    # 以此来衡量量化单独一个算子对网络输出的影响情况，该方法常被用来决定网络调度与混合精度量化
    # 你可以将那些误差较大的层送往 TargetPlatform.FP32
    # ------------------------------------------------------------
    reports = layerwise_error_analyse(
        graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
        dataloader=CALIBRATION)

    # ------------------------------------------------------------
    # statistical_analyse 是强有力的统计分析方法，该方法统计每一层的输入、输出以及参数的统计分布情况
    # 使用这一方法，你将更清晰地了解网络的量化情况，并能够有针对性地选择优化方案
    # 推荐在网络量化情况不佳时，使用 statistical_analyse 辅助你的分析
    # 该方法不打印任何数据，你需要手动将数据保存到硬盘并进行分析
    # ------------------------------------------------------------
    report = statistical_analyse(
        graph=quantized, running_device=DEVICE, 
        collate_fn=collate_fn, dataloader=CALIBRATION)

    from pandas import DataFrame

    report = DataFrame(report)
    report.to_csv('1.csv')
