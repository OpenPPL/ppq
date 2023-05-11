"""这是一个高度自动化的 PPQ 量化的入口脚本，将你的模型和数据按要求进行打包:

在自动化 API 中，我们使用 QuantizationSetting 对象传递量化参数。

This file will show you how to quantize your network with PPQ
    You should prepare your model and calibration dataset as follow:

    ~/working/model.onnx                          <--  your model
    ~/working/data/*.npy or ~/working/data/*.bin  <--  your dataset

if you are using caffe model:
    ~/working/model.caffemdoel  <--  your model
    ~/working/model.prototext   <--  your model

### MAKE SURE YOUR INPUT LAYOUT IS [N, C, H, W] or [C, H, W] ###

quantized model will be generated at: ~/working/quantized.onnx
"""
from ppq import *                                       
from ppq.api import *
import os

# modify configuration below:
WORKING_DIRECTORY = 'working'                             # choose your working directory
TARGET_PLATFORM   = TargetPlatform.PPL_CUDA_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
NETWORK_INPUTSHAPE    = [1, 3, 224, 224]                  # input shape of your network
CALIBRATION_BATCHSIZE = 16                                # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = False
TRAINING_YOUR_NETWORK = True                              # 是否需要 Finetuning 一下你的网络

# -------------------------------------------------------------------
# 加载你的模型文件，PPQ 将会把 onnx 或者 caffe 模型文件解析成自己的格式
# 如果你正使用 pytorch, tensorflow 等框架，你可以先将模型导出成 onnx
# 使用 torch.onnx.export 即可，如果你在导出 torch 模型时发生错误，欢迎与我们联系。
# -------------------------------------------------------------------
graph = None
if MODEL_TYPE == NetworkFramework.ONNX:
    graph = load_onnx_graph(onnx_import_file = os.path.join(WORKING_DIRECTORY, 'model.onnx'))
if MODEL_TYPE == NetworkFramework.CAFFE:
    graph = load_caffe_graph(
        caffemodel_path = os.path.join(WORKING_DIRECTORY, 'model.caffemodel'),
        prototxt_path = os.path.join(WORKING_DIRECTORY, 'model.prototxt'))
assert graph is not None, 'Graph Loading Error, Check your input again.'

# -------------------------------------------------------------------
# SETTING 对象用于控制 PPQ 的量化逻辑，主要描述了图融合逻辑、调度方案、量化细节策略等
# 当你的网络量化误差过高时，你需要修改 SETTING 对象中的属性来进行特定的优化
# -------------------------------------------------------------------
QS = QuantizationSettingFactory.default_setting()

# -------------------------------------------------------------------
# 下面向你展示了如何使用 finetuning 过程提升量化精度
# 在 PPQ 中我们提供了十余种算法用来帮助你恢复精度
# 开启他们的方式都是 QS.xxxx = True
# 按需使用，不要全部打开，容易起飞
# -------------------------------------------------------------------
if TRAINING_YOUR_NETWORK:
    QS.lsq_optimization = True                                      # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.steps = 500                         # 再训练步数，影响训练时间，500 步大概几分钟
    QS.lsq_optimization_setting.collecting_device = 'cuda'          # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'

# -------------------------------------------------------------------
# 你可以把量化很糟糕的算子送回 FP32
# 当然你要先确认你的硬件支持 fp32 的执行
# 你可以使用 layerwise_error_analyse 来找出那些算子量化的很糟糕
# -------------------------------------------------------------------
QS.dispatching_table.append(operation='OP NAME', platform=TargetPlatform.FP32)

print('正准备量化你的网络，检查下列设置:')
print(f'WORKING DIRECTORY    : {WORKING_DIRECTORY}')
print(f'TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')

# -------------------------------------------------------------------
# load_calibration_dataset 函数针对的是单输入模型，输入数据必须是图像数据 layout : [n c h w]
# 如果你的模型具有更复杂的输入格式，你可以重写下面的 load_calibration_dataset 函数
# 请注意，任何可遍历对象都可以作为 ppq 的数据集作为输入
# 比如下面这个 dataloader = [torch.zeros(size=[1,3,224,224]) for _ in range(32)]
# 当前这个函数的数据将从 WORKING_DIRECTORY/data 文件夹中进行数据加载
# 
# 如果你的数据不在这里
# 你同样需要自己写一个 load_calibration_dataset 函数
# -------------------------------------------------------------------
dataloader = load_calibration_dataset(
    directory    = WORKING_DIRECTORY,
    input_shape  = NETWORK_INPUTSHAPE,
    batchsize    = CALIBRATION_BATCHSIZE,
    input_format = INPUT_LAYOUT)

# ENABLE CUDA KERNEL 会加速量化效率 3x ~ 10x，但是你如果没有装相应编译环境的话是编译不了的
# 你可以尝试安装编译环境，或者在不启动 CUDA KERNEL 的情况下完成量化：移除 with ENABLE_CUDA_KERNEL(): 即可
with ENABLE_CUDA_KERNEL():
    print('网络正量化中，根据你的量化配置，这将需要一段时间:')
    quantized = quantize_native_model(
        setting=QS,                     # setting 对象用来控制标准量化逻辑
        model=graph,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=NETWORK_INPUTSHAPE, # 如果你的网络只有一个输入，使用这个参数传参
        inputs=None,                    # 如果你的网络有多个输入，使用这个参数传参，就是 input_shape=None, inputs=[torch.zeros(1,3,224,224), torch.zeros(1,3,224,224)]
        collate_fn=lambda x: x.to(EXECUTING_DEVICE),  # collate_fn 跟 torch dataloader 的 collate fn 是一样的，用于数据预处理，
                                                      # 你当然也可以用 torch dataloader 的那个，然后设置这个为 None
        platform=TARGET_PLATFORM,
        device=EXECUTING_DEVICE,
        do_quantize=True)
    
    # -------------------------------------------------------------------
    # 如果你需要执行量化后的神经网络并得到结果，则需要创建一个 executor
    # 这个 executor 的行为和 torch.Module 是类似的，你可以利用这个东西来获取执行结果
    # 请注意，必须在 export 之前执行此操作。
    # -------------------------------------------------------------------
    executor = TorchExecutor(graph=quantized, device=EXECUTING_DEVICE)
    # output = executor.forward(input)

    # -------------------------------------------------------------------
    # PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量
    # 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
    # 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
    # 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
    # 你需要使用 layerwise_error_analyse 逐层分析误差的来源
    # -------------------------------------------------------------------
    print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
    reports = graphwise_error_analyse(
        graph=quantized, running_device=EXECUTING_DEVICE, steps=32,
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    for op, snr in reports.items():
        if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

    if REQUIRE_ANALYSE:
        print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
        layerwise_error_analyse(graph=quantized, running_device=EXECUTING_DEVICE,
                                interested_outputs=None,
                                dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    # -------------------------------------------------------------------
    # 使用 export_ppq_graph 函数来导出量化后的模型
    # PPQ 会根据你所选择的导出平台来修改模型格式
    # -------------------------------------------------------------------
    print('网络量化结束，正在生成目标文件:')
    export_ppq_graph(
        graph=quantized, platform=TARGET_PLATFORM,
        graph_save_to = os.path.join(WORKING_DIRECTORY, 'quantized.onnx'),
        config_save_to = os.path.join(WORKING_DIRECTORY, 'quant_cfg.json'))
