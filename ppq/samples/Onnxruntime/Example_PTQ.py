# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 onnxruntime 对 PPQ 导出的模型进行推理
# 你需要注意，Onnxruntime 可以运行各种各样的量化方案，但模型量化对 Onnxruntime 而言几乎无法起到加速作用
# 你可以使用 Onnxruntime 来验证量化方案以及 ppq 量化的正确性，但这不是一个合理的部署平台
# 修改 QUANT_PLATFROM 来使用不同的量化方案。

# This Script export ppq internal graph to onnxruntime,
# you should notice that onnx is designed as an Open Neural Network Exchange format.
# It has the capbility to describe most of ppq's quantization policy including combinations of:
#   Symmtrical, Asymmtrical, POT, Per-channel, Per-Layer
# However onnxruntime can not accelerate quantized model in most cases,
# you are supposed to use onnxruntime for verifying your network quantization result only.
# ---------------------------------------------------------------

# For this onnx inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
import onnxruntime
import torch
from ppq import *
from ppq.api import *
from tqdm import tqdm

QUANT_PLATFROM = TargetPlatform.TRT_INT8
MODEL          = 'model.onnx'
INPUT_SHAPE    = [1, 3, 224, 224]
SAMPLES        = [torch.rand(size=[INPUT_SHAPE]) for _ in range(256)] # rewirte this to use real data.
DEVICE         = 'cuda'
FINETUNE       = True
QS             = QuantizationSettingFactory.default_setting()
EXECUTING_DEVICE = 'cuda'
REQUIRE_ANALYSE  = True

# -------------------------------------------------------------------
# 下面向你展示了常用参数调节选项：
# -------------------------------------------------------------------
if USING_CUDA_KERNEL:
    QS.advanced_optimization = FINETUNE                             # 启动网络再训练过程，降低量化误差
    QS.advanced_optimization_setting.steps = 2500                   # 再训练步数，影响训练时间，2500步大概几分钟
    QS.advanced_optimization_setting.collecting_device = 'executor' # 缓存数据放在那，executor 就是放在gpu，如果显存超了你就换成 'cpu'
    QS.advanced_optimization_setting.auto_check = False             # 打开这个选项则训练过程中会防止过拟合，以及意外情况，通常不需要开。
else:
    QS.lsq_optimization = FINETUNE                                  # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.epochs = 30                         # 再训练轮数，影响训练时间，30轮大概几分钟
    QS.lsq_optimization_setting.collecting_device = 'cuda'          # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'

if QUANT_PLATFROM in {TargetPlatform.PPL_DSP_INT8,                  # 这些平台是 per tensor 量化的
                       TargetPlatform.HEXAGON_INT8,
                       TargetPlatform.SNPE_INT8,
                       TargetPlatform.METAX_INT8_T,
                       TargetPlatform.FPGA_INT8}:
    QS.equalization = True                                          # per tensor 量化平台需要做 equalization

if QUANT_PLATFROM in {TargetPlatform.ACADEMIC_INT8,                 # 把量化的不太好的算子送回 FP32
                       TargetPlatform.PPL_CUDA_INT8,                # 注意做这件事之前你需要确保你的执行框架具有混合精度执行的能力，以及浮点计算的能力
                       TargetPlatform.TRT_INT8}:
    QS.dispatching_table.append(operation='OP NAME', platform=TargetPlatform.FP32)

print('正准备量化你的网络，检查下列设置:')
print(f'TARGET PLATFORM      : {QUANT_PLATFROM.name}')
print(f'NETWORK INPUTSHAPE   : {INPUT_SHAPE}')

qir = quantize_onnx_model(
    onnx_import_file=MODEL, calib_dataloader=SAMPLES, calib_steps=128, setting=QS,
    input_shape=INPUT_SHAPE, collate_fn=None, platform=QUANT_PLATFROM, do_quantize=True)

# -------------------------------------------------------------------
# PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量
# 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
# 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
# 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
# 你需要使用 layerwise_error_analyse 逐层分析误差的来源
# -------------------------------------------------------------------
print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
reports = graphwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, steps=32,
    dataloader=SAMPLES, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
for op, snr in reports.items():
    if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

if REQUIRE_ANALYSE:
    print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
    layerwise_error_analyse(graph=qir, running_device=EXECUTING_DEVICE,
                            interested_outputs=None,
                            dataloader=SAMPLES, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

print('网络量化结束，正在生成目标文件:')
export_ppq_graph(
    graph=qir, platform=QUANT_PLATFROM,
    graph_save_to = 'model_int8.onnx')

# -------------------------------------------------------------------
# 记录一下输入输出的名字，onnxruntime 跑的时候需要提供这些名字
# 我写的只是单输出单输入的版本，多输出多输入你得自己改改
# -------------------------------------------------------------------
int8_input_names  = [name for name, _ in qir.inputs.items()]
int8_output_names = [name for name, _ in qir.outputs.items()]

# -------------------------------------------------------------------
# 启动 onnxruntime 进行推理
# 截止 2022.05， onnxruntime 跑 int8 很慢的，你就别期待它会很快了。
# 如果你知道怎么让它跑的快点，或者onnxruntime更新了，你可以随时联系我。
# -------------------------------------------------------------------
session = onnxruntime.InferenceSession('model_int8.onnx', providers=['CUDAExecutionProvider'])
onnxruntime_results = []
for sample in tqdm(SAMPLES, desc='ONNXRUNTIME GENERATEING OUTPUTS', total=len(SAMPLES)):
    result = session.run([int8_output_names[0]], {int8_input_names[0]: convert_any_to_numpy(sample)})
    onnxruntime_results.append(result)