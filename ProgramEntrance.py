"""
    This file will show you how to quantize your network with PPQ
        You should prepare your model and calibration dataset as follow:

        ~/working/model.onnx  <--  your model
        ~/working/data/*.npy  <--  your dataset

    if you are using caffe model:
        ~/working/model.caffemdoel  <--  your model
        ~/working/model.prototext   <--  your model

    ### MAKE SURE YOUR INPUT LAYOUT IS [N, C, H, W] or [C, H, W] ###

    quantized model will be generated at: ~/working/quantized.onnx
"""
from ppq import *
from ppq.api import *
from Util import *

# modify configuration below:
WORKING_DIRECTORY = 'working/'                            # choose your working directory
TARGET_PLATFORM   = TargetPlatform.PPL_CUDA_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
NETWORK_INPUTSHAPE    = [1, 3, 224, 224]                  # input shape of your network
CALIBRATION_BATCHSIZE = 16                                # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu', 'cpu' is untested.
REQUIRE_ANALYSE       = True

# -------------------------------------------------------------------
# SETTING 对象用于控制 PPQ 的量化逻辑 
# 当你的网络量化误差过高时，你需要修改 SETTING 对象中的参数来进行特定的优化
# -------------------------------------------------------------------
SETTING = QuantizationSettingFactory.default_setting()
if TARGET_PLATFORM == TargetPlatform.PPL_CUDA_INT8:
    SETTING = QuantizationSettingFactory.pplcuda_setting()
if TARGET_PLATFORM == TargetPlatform.DSP_INT8:
    SETTING = QuantizationSettingFactory.dsp_setting()
if TARGET_PLATFORM == TargetPlatform.NXP_INT8:
    SETTING = QuantizationSettingFactory.nxp_setting()

print('正准备量化你的网络，检查下列设置:')
print(f'WORKING DIRECTORY    : {WORKING_DIRECTORY}')
print(f'TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')

dataloader = load_calibration_dataset(
    directory=WORKING_DIRECTORY,
    input_shape=NETWORK_INPUTSHAPE,
    batchsize=CALIBRATION_BATCHSIZE)

print('网络正量化中，根据你的量化配置，这将需要一段时间:')
quantized = quantize(
    working_directory=WORKING_DIRECTORY, setting=SETTING,
    model_type=MODEL_TYPE, executing_device=EXECUTING_DEVICE,
    input_shape=NETWORK_INPUTSHAPE, target_platform=TARGET_PLATFORM,
    dataloader=dataloader)

print('网络量化结束，正在生成目标文件:')
if MODEL_TYPE == NetworkFramework.ONNX:
    export(working_directory=WORKING_DIRECTORY,
           quantized=quantized, platform=TARGET_PLATFORM)

elif MODEL_TYPE == NetworkFramework.CAFFE:
    export(working_directory=WORKING_DIRECTORY,
           quantized=quantized, platform=TARGET_PLATFORM)
    
# -------------------------------------------------------------------
# PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量 
# 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
# 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
# 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
# 你需要使用 layerwise_error_analyse 逐层分析误差的来源
# -------------------------------------------------------------------
print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
reports = graphwise_error_analyse(
    graph=quantized, running_device=EXECUTING_DEVICE,
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
for op, snr in reports.items():
    if snr > 0.10: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

if REQUIRE_ANALYSE:
    print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
    layerwise_error_analyse(graph=quantized, running_device=EXECUTING_DEVICE,
                            interested_outputs=[var for var in quantized.outputs],
                            dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))