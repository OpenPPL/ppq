from typing import Iterable

import torch
import torchvision

from ppq import (QuantizationSettingFactory, TargetPlatform,
                 graphwise_error_analyse)
from ppq.api import QuantizationSettingFactory, quantize_torch_model
from ppq.api.interface import ENABLE_CUDA_KERNEL
from ppq.executor.torch import TorchExecutor

# ------------------------------------------------------------
# 在 PPQ 中我们目前提供两种不同的算法帮助你微调网络
# 这些算法将使用 calibration dataset 中的数据，对网络权重展开重训练
# 1. 经过训练的网络不保证中间结果与原来能够对齐，在进行误差分析时你需要注意这一点
# 2. 在训练中使用 with ENABLE_CUDA_KERNEL(): 子句将显著加速训练过程
# 3. 训练过程的缓存数据将被贮存在 gpu 上，这可能导致你显存溢出，你可以修改参数将缓存设备改为 cpu
# ------------------------------------------------------------

BATCHSIZE   = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE      = 'cuda'
PLATFORM    = TargetPlatform.PPL_CUDA_INT8

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
# PPQ 提供基于 LSQ 的网络微调过程，这是推荐的做法
# 你将使用 Quant Setting 来调用微调过程，并调整微调参数
# ------------------------------------------------------------
QSetting = QuantizationSettingFactory.default_setting()
QSetting.lsq_optimization                            = True
QSetting.lsq_optimization_setting.block_size         = 4
QSetting.lsq_optimization_setting.lr                 = 1e-5
QSetting.lsq_optimization_setting.gamma              = 0
QSetting.lsq_optimization_setting.is_scale_trainable = True
QSetting.lsq_optimization_setting.collecting_device  = 'cuda'

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
    # 当我们完成训练后，我们将调用 graphwise_error_analyse 方法分析网络误差
    # 经过训练的中间层误差可能很大，但这不是我们所关心的 —— 训练方法只优化最终输出的误差
    # 一个量化良好的网络，最后输出层的误差不应大于 10%
    # ------------------------------------------------------------
    graphwise_error_analyse(
        graph=quantized, 
        running_device=DEVICE, 
        dataloader=CALIBRATION,
        collate_fn=collate_fn)


model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

QSetting = QuantizationSettingFactory.default_setting()

with ENABLE_CUDA_KERNEL():
    quantized = quantize_torch_model(
        model=model, calib_dataloader=CALIBRATION,
        calib_steps=32, input_shape=INPUT_SHAPE,
        setting=QSetting, collate_fn=collate_fn, platform=PLATFORM,
        onnx_export_file='Output/model.onnx', device=DEVICE, verbose=0)

    graphwise_error_analyse(
        graph=quantized, 
        running_device=DEVICE, 
        dataloader=CALIBRATION,
        collate_fn=collate_fn)
