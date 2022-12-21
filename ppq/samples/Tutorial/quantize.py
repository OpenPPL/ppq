from typing import Iterable, Tuple

import torch

from ppq import (BaseGraph, QuantizationSettingFactory, TargetPlatform,
                 convert_any_to_numpy, torch_snr_error)
from ppq.api import (dispatch_graph, export_ppq_graph, load_onnx_graph,
                     quantize_onnx_model)
from ppq.core.data import convert_any_to_torch_tensor
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse

INPUT_SHAPES     = {'input.1': [1, 3, 224, 224]}
DEVICE           = 'cuda'
QUANT_PLATFORM   = TargetPlatform.TRT_INT8
ONNX_PATH        = 'model.onnx'
ONNX_OUTPUT_PATH = 'Output/model.onnx'

# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何量化一个 onnx 模型，执行误差分析，并与 onnxruntime 对齐结果
# 在这个例子中，我们特别地为你展示如何量化一个多输入的模型
# 此时你的 Calibration Dataset 应该是一个 list of dictionary
# ------------------------------------------------------------
def generate_calibration_dataset(graph: BaseGraph, num_of_batches: int = 32) -> Tuple[Iterable[dict], torch.Tensor]:
    dataset = []
    for i in range(num_of_batches):
        sample = {name: torch.rand(INPUT_SHAPES[name]) for name in graph.inputs}
        dataset.append(sample)
    return dataset, sample # last sample

def collate_fn(batch: dict) -> torch.Tensor:
    return {k: v.to(DEVICE) for k, v in batch.items()}

# ------------------------------------------------------------
# 在这里，我们仍然创建一个 QuantizationSetting 对象用来管理量化过程
# 我们将调度方法修改为 conservative，并且要求 PPQ 启动量化微调
# ------------------------------------------------------------
QSetting = QuantizationSettingFactory.default_setting()
QSetting.lsq_optimization = False

# ------------------------------------------------------------
# 准备好 QuantizationSetting 后，我们加载模型，并且要求 ppq 按照规则完成图调度
# ------------------------------------------------------------
graph = load_onnx_graph(onnx_import_file=ONNX_PATH)
graph = dispatch_graph(graph=graph, platform=QUANT_PLATFORM)
for name in graph.inputs:
    if name not in INPUT_SHAPES:
        raise KeyError(f'Graph Input {name} needs a valid shape.')

if len(graph.outputs) != 1:
    raise ValueError('This Script Requires graph to have only 1 output.')

# ------------------------------------------------------------
# 生成校准所需的数据集，我们准备开始完成网络量化任务
# ------------------------------------------------------------
calibration_dataset, sample = generate_calibration_dataset(graph)
quantized = quantize_onnx_model(
    onnx_import_file=ONNX_PATH, calib_dataloader=calibration_dataset,
    calib_steps=32, input_shape=None, inputs=collate_fn(sample),
    setting=QSetting, collate_fn=collate_fn, platform=QUANT_PLATFORM,
    device=DEVICE, verbose=0)

# ------------------------------------------------------------
# 在 PPQ 完成网络量化之后，我们特别地保存一下 PPQ 网络执行的结果
# 在本样例的最后，我们将对比 PPQ 与 Onnxruntime 的执行结果是否相同
# ------------------------------------------------------------
executor, reference_outputs = TorchExecutor(quantized), []
for sample in calibration_dataset:
    reference_outputs.append(executor.forward(collate_fn(sample)))

# ------------------------------------------------------------
# 执行网络误差分析，并导出计算图
# ------------------------------------------------------------
graphwise_error_analyse(
    graph=quantized, running_device=DEVICE, 
    collate_fn=collate_fn, dataloader=calibration_dataset)

export_ppq_graph(graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
                 graph_save_to=ONNX_OUTPUT_PATH)

# -----------------------------------------
# 在最后，我们启动 onnxruntime 并比对结果
# -----------------------------------------
try:
    import onnxruntime
except ImportError as e:
    raise Exception('Onnxruntime is not installed.')

sess = onnxruntime.InferenceSession(ONNX_OUTPUT_PATH, providers=['CUDAExecutionProvider'])
output_name = sess.get_outputs()[0].name

onnxruntime_outputs = []
for sample in calibration_dataset:
    onnxruntime_outputs.append(sess.run(
        output_names=[output_name], 
        input_feed={k: convert_any_to_numpy(v) for k, v in sample.items()}))

y_pred, y_real = [], []
for reference_output, onnxruntime_output in zip(reference_outputs, onnxruntime_outputs):
    y_pred.append(convert_any_to_torch_tensor(reference_output[0], device='cpu').unsqueeze(0))
    y_real.append(convert_any_to_torch_tensor(onnxruntime_output[0], device='cpu').unsqueeze(0))
y_pred = torch.cat(y_pred, dim=0)
y_real = torch.cat(y_real, dim=0)
print(f'Simulating Error For {output_name}: {torch_snr_error(y_pred=y_pred, y_real=y_real).item() * 100 :.4f}%')
