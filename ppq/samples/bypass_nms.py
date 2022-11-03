# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用绕过那些跟量化没什么关系的算子
#   当你的算子处于网络的最后，其后也没有什么需要量化的算子了
#   你就可以给它定义一个假的 forward 函数，从而帮助 PPQ 完成量化
#   PPQ 不再需要收集其后的数据信息，所以错误额计算过程也能得到正确的量化结果
#   
#   当然如果你的自定义算子如果会干涉到量化过程，你还是需要向 PPQ 提供一个的执行函数
# ---------------------------------------------------------------

# For this inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
import torch
from torch.utils.data import DataLoader

from ppq import *
from ppq.api import *

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.TRT_INT8  # identify a target platform for your network.

def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

def happy_forward_func(op: Operation, values, ctx, **kwargs):
    """你必须保证函数签名满足要求，即函数的输入和返回值都满足 PPQ 的系统要求
    
    你的执行函数将接收 op, values, ctx 三个元素作为输入
    其中 op 反应了当前执行的算子信息，values是一个数组，包含了算子所有输入
    ctx 是 PPQ 执行上下文

    你将返回一个 torch.Tensor 或者多个 torch.Tensor 作为结果
    这取决于你的算子在onnx中有多少个输出
    """
    return torch.zeros(size=[1, 100, 5]), torch.zeros(size=[1, 100])

# ---------------------------------------------
# 注册一个假的函数让我们绕过 nms
# ---------------------------------------------
register_operation_handler(
    happy_forward_func, 
    operation_type="TRTBatchedNMS", 
    platform=TargetPlatform.FP32)

quant_setting = QuantizationSettingFactory.default_setting()

# For pytorch user, just dump your network to disk with onnx first
unquantized = load_onnx_graph(onnx_import_file='Output/onnx.model')

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=BATCHSIZE, shuffle=False)

# quantize your model.
quantized = quantize_native_model(
    model=unquantized, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(graph=quantized, platform=PLATFORM,
                 graph_save_to='Output/quantized(onnx).onnx',
                 config_save_to='Output/quantized(onnx).json')
