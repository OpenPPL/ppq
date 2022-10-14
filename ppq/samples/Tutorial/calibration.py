from typing import Iterable

import torch
import torchvision
from ppq import TargetPlatform, graphwise_error_analyse
from ppq.api.interface import (ENABLE_CUDA_KERNEL, dump_torch_to_onnx,
                               load_onnx_graph, quantize_native_model)
from ppq.api.setting import QuantizationSettingFactory

# ------------------------------------------------------------
# 在 PPQ 中我们提供许多校准方法，这些校准方法将计算出网络的量化参数
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
# PPQ 提供 kl, mse, minmax, isotone, percentile(默认) 五种校准方法
# 每一种校准方法还有更多参数可供调整，PPQ 也允许你单独调整某一层的量化校准方法
# 在这里我们首先展示以 QSetting 的方法调整量化校准参数(推荐)
# ------------------------------------------------------------
QSetting = QuantizationSettingFactory.default_setting()
QSetting.quantize_activation_setting.calib_algorithm = 'kl'
QSetting.quantize_parameter_setting.calib_algorithm  = 'minmax'
# ------------------------------------------------------------
# 更进一步地，当你选择了某种校准方法，你可以进入 ppq.core.common
# OBSERVER_KL_HIST_BINS, OBSERVER_PERCENTILE, OBSERVER_MSE_HIST_BINS 皆是与校准方法相关的可调整参数
# OBSERVER_KL_HIST_BINS - KL 算法相关的箱子个数，你可以试试将其调整为 512, 1024, 2048, 4096, 8192 ...
# OBSERVER_PERCENTILE - Percentile 算法相关的百分比，你可以试试将其调整为 0.9999, 0.9995, 0.99999, 0.99995 ...
# OBSERVER_MSE_HIST_BINS - MSE 算法相关的箱子个数，你可以试试将其调整为 512, 1024, 2048, 4096, 8192 ...
# ------------------------------------------------------------

# ------------------------------------------------------------
# 如果你使用 ENABLE_CUDA_KERNEL 方法
# PPQ 将会尝试编译自定义的高性能量化算子，这一过程需要编译环境的支持
# 如果你在编译过程中发生错误，你可以删除此处对于 ENABLE_CUDA_KERNEL 方法的调用
# 这将显著降低 PPQ 的运算速度；但即使你无法编译这些算子，你仍然可以使用 pytorch 的 gpu 算子完成量化
# ------------------------------------------------------------
with ENABLE_CUDA_KERNEL():
    dump_torch_to_onnx(model=model, onnx_export_file='Output/model.onnx', 
                       input_shape=INPUT_SHAPE, input_dtype=torch.float32)
    graph = load_onnx_graph(onnx_import_file='Output/model.onnx')
    quantized = quantize_native_model(
        model=graph, calib_dataloader=CALIBRATION,
        calib_steps=32, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn, platform=PLATFORM,
        device=DEVICE, verbose=0)

    # ------------------------------------------------------------
    # graphwise_error_analyse 是最常用的分析方法，它分析网络中的量化误差情况，它的结果将直接打印在屏幕上
    # 对于 graphwise_error_analyse 而言，算子的误差直接衡量了量化网络与浮点网络之间的输出误差
    # 这一误差是累积的，意味着网络后面的算子总是会比网络前面的算子拥有更高的输出误差
    # 留意网络输出的误差情况，如果你想获得一个精度较高的量化网络，那么那些靠近输出的节点误差不应超过 10%
    # 该方法只衡量 Conv, Gemm 算子的误差情况，如果你对其余算子的误差情况感兴趣，需要手动修改方法逻辑
    # 你可以使用该方法对比不同校准方法对量化结果所产生的影响
    # ------------------------------------------------------------
    reports = graphwise_error_analyse(
        graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
        dataloader=CALIBRATION)
