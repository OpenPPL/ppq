# ------------------------------------------------------------
# PPQ 最佳实践示例工程，在这个工程中，我们将向你展示如何充分调动 PPQ 的各项功能
# ------------------------------------------------------------
import torch
from ppq import *
from ppq.api import *
import torchvision
# ------------------------------------------------------------
# Step - 1. 加载校准集与模型
# ------------------------------------------------------------
BATCHSIZE   = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE      = 'cuda'
PLATFORM    = TargetPlatform.TRT_INT8
CALIBRATION = [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
QS          = QuantizationSettingFactory.default_setting()
def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# ------------------------------------------------------------
# Step - 2. 执行首次量化，完成逐层误差分析
# ------------------------------------------------------------
with ENABLE_CUDA_KERNEL():
    quantized = quantize_torch_model(
        model=model, calib_dataloader=CALIBRATION,
        calib_steps=32, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn, platform=PLATFORM, setting=QS,
        onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

    reports = layerwise_error_analyse(
        graph=quantized, running_device=DEVICE, 
        collate_fn=collate_fn, dataloader=CALIBRATION)

# ------------------------------------------------------------
# Step - 3. 根据逐层误差情况，将部分难以量化的层直接送到非量化平台
# 在这个例子中，我们解除前十个误差最大的层的量化，这只是一个示例
# 为了确保量化精度达到较高水准，通常只有个别层需要解除量化。
# 不要妄图单纯使用调度解决所有精度问题，调度体现了运行效率与网络精度之间的权衡
# 我们后续还可以通过调节校准算法与量化参数来提升精度
# ------------------------------------------------------------

# 从大到小排序单层误差
sensitivity = [(op_name, error) for op_name, error in reports.items()]
sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)

# 将前十个误差最大的层送上 FP32
for op_name, _ in sensitivity[: 10]:
    QS.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)

# ------------------------------------------------------------
# Step - 4. 选择一个合适的校准算法，最小化量化误差
# 这一过程需要你手动调整 QS 中的校准算法，需要反复执行和对比
# 在这里我们只推荐你更改激活值的校准算法，对于参数而言
# 在 INT8 的量化场景中，minmax 往往都是最好的校准策略
# 在这个场景下，不能使用 isotone 策略
# ------------------------------------------------------------
for calib_algo in {'minmax', 'percentile', 'kl', 'mse'}:
    QS.quantize_activation_setting.calib_algorithm = calib_algo

    with ENABLE_CUDA_KERNEL():
        quantized = quantize_torch_model(
            model=model, calib_dataloader=CALIBRATION,
            calib_steps=32, input_shape=INPUT_SHAPE,
            collate_fn=collate_fn, platform=PLATFORM, setting=QS,
            onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

        print(f'Error Report of Algorithm {calib_algo}: ')
        reports = graphwise_error_analyse(
            graph=quantized, running_device=DEVICE, 
            collate_fn=collate_fn, dataloader=CALIBRATION)

# 在确定了一种校准算法之后，你还可以修改 ppq.core.common 中的相关属性来取得更优结果
# 下列参数将影响校准效果：
    # Observer 中 hist 箱子的个数
    # OBSERVER_KL_HIST_BINS = 4096
    # Observer 中 percentile 的参数
    # OBSERVER_PERCENTILE = 0.9999
    # Observer 中 mse 校准方法 hist 箱子的个数
    # OBSERVER_MSE_HIST_BINS = 2048
    # Observer 中 mse 计算的间隔，间隔越小，所需时间越长
    # OBSERVER_MSE_COMPUTE_INTERVAL = 8

# 在完成测试后，选择一种最为合适的校准算法，我们此处以 percentile 为例
QS.quantize_activation_setting.calib_algorithm = 'percentile'

# ------------------------------------------------------------
# Step - 5. 再次检查我们的量化误差，如果与预期仍有差距
# 则我们可以进一步使用优化过程来调节网络参数
# ------------------------------------------------------------
with ENABLE_CUDA_KERNEL():
    
    # 首先我们调节 equalization 算法的参数
    # 调节时关闭 lsq_optimization 以缩短流程执行时间
    QS.equalization                         = True # 试试 True 或 False
    QS.equalization_setting.opt_level       = 1    # 试试 1 或 2
    QS.equalization_setting.iterations      = 10   # 试试 3, 10, 100
    QS.equalization_setting.value_threshold = 0.5  # 试试 0, 0.5, 2

    # 之后我们调节 LSQ 算法的参数
    QS.lsq_optimization                            = True
    QS.lsq_optimization_setting.block_size         = 4       # 试试 1, 4, 6
    QS.lsq_optimization_setting.collecting_device  = 'cuda'  # 如果显存不够你就写 cpu
    QS.lsq_optimization_setting.is_scale_trainable = True    # 试试 True 或者 False
    QS.lsq_optimization_setting.lr                 = 1e-5    # 试试 1e-5, 3e-5, 1e-4
    QS.lsq_optimization_setting.steps              = 500     # 试试 300, 500, 2000

    quantized = quantize_torch_model(
        model=model, calib_dataloader=CALIBRATION, 
        calib_steps=32, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn, platform=PLATFORM, setting=QS,
        onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

    reports = graphwise_error_analyse(
        graph=quantized, running_device=DEVICE, 
        collate_fn=collate_fn, dataloader=CALIBRATION)


# ------------------------------------------------------------
# Step - 6. 最后，我们导出模型，并在 onnxruntime 上完成校验
# ------------------------------------------------------------
ONNX_OUTPUT_PATH = 'Output/model.onnx'
executor, reference_outputs = TorchExecutor(quantized), []
for sample in CALIBRATION:
    reference_outputs.append(executor.forward(collate_fn(sample)))

export_ppq_graph(
    graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
    graph_save_to=ONNX_OUTPUT_PATH)

try:
    import onnxruntime
except ImportError as e:
    raise Exception('Onnxruntime is not installed.')

sess = onnxruntime.InferenceSession(ONNX_OUTPUT_PATH, providers=['CUDAExecutionProvider'])
onnxruntime_outputs = []
for sample in CALIBRATION:
    onnxruntime_outputs.append(sess.run(
        output_names=[name for name in quantized.outputs], 
        input_feed={'input.1': convert_any_to_numpy(sample)}))

name_of_output = [name for name in quantized.outputs]
for oidx, output in enumerate(name_of_output):
    y_pred, y_real = [], []
    for reference_output, onnxruntime_output in zip(reference_outputs, onnxruntime_outputs):
        y_pred.append(convert_any_to_torch_tensor(reference_output[oidx], device='cpu').unsqueeze(0))
        y_real.append(convert_any_to_torch_tensor(onnxruntime_output[oidx], device='cpu').unsqueeze(0))
    y_pred = torch.cat(y_pred, dim=0)
    y_real = torch.cat(y_real, dim=0)
    print(f'Simulating Error For {output}: {torch_snr_error(y_pred=y_pred, y_real=y_real).item() :.4f}')
