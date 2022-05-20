"""这是一个 PPQ 量化的入口脚本，将你的模型和数据按要求进行打包:

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
DUMP_RESULT           = False

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
if TARGET_PLATFORM == TargetPlatform.PPL_CUDA_INT8:
    QS = QuantizationSettingFactory.pplcuda_setting()
if TARGET_PLATFORM in {TargetPlatform.PPL_DSP_INT8, TargetPlatform.HEXAGON_INT8, TargetPlatform.SNPE_INT8}:
    QS = QuantizationSettingFactory.dsp_setting()
if TARGET_PLATFORM == TargetPlatform.METAX_INT8_T:
    QS = QuantizationSettingFactory.metax_pertensor_setting()
if TARGET_PLATFORM == TargetPlatform.NXP_INT8:
    QS = QuantizationSettingFactory.nxp_setting()
if TARGET_PLATFORM == TargetPlatform.FPGA_INT8:
    QS = QuantizationSettingFactory.fpga_setting()
if TARGET_PLATFORM == TargetPlatform.TRT_INT8:
    QS = QuantizationSettingFactory.trt_setting()

# -------------------------------------------------------------------
# 下面向你展示了常用参数调节选项：
# -------------------------------------------------------------------
if USING_CUDA_KERNEL:
    QS.advanced_optimization = True                                 # 启动网络再训练过程，降低量化误差
    QS.advanced_optimization_setting.steps = 2500                   # 再训练步数，影响训练时间，2500步大概几分钟
    QS.advanced_optimization_setting.collecting_device = 'executor' # 缓存数据放在那，executor 就是放在gpu，如果显存超了你就换成 'cpu'
    QS.advanced_optimization_setting.auto_check = False             # 打开这个选项则训练过程中会防止过拟合，以及意外情况，通常不需要开。
else:
    QS.lsq_optimization = True                                      # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.epochs = 30                         # 再训练轮数，影响训练时间，30轮大概几分钟
    QS.lsq_optimization_setting.collecting_device = 'cuda'          # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'

if TARGET_PLATFORM in {TargetPlatform.PPL_DSP_INT8,                 # 这些平台是 per tensor 量化的
                       TargetPlatform.HEXAGON_INT8,
                       TargetPlatform.SNPE_INT8,
                       TargetPlatform.METAX_INT8_T,
                       TargetPlatform.FPGA_INT8}:
    QS.equalization = True                                          # per tensor 量化平台需要做 equalization

if TARGET_PLATFORM in {TargetPlatform.ACADEMIC_INT8,                # 把量化的不太好的算子送回 FP32
                       TargetPlatform.PPL_CUDA_INT8,                # 注意做这件事之前你需要确保你的执行框架具有混合精度执行的能力，以及浮点计算的能力
                       TargetPlatform.TRT_INT8}:
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
# 当前这个函数的数据将从 WORKING_DIRECTORY/data 文件夹中进行数据加载
# -------------------------------------------------------------------
dataloader = load_calibration_dataset(
    directory    = WORKING_DIRECTORY,
    input_shape  = NETWORK_INPUTSHAPE,
    batchsize    = CALIBRATION_BATCHSIZE,
    input_format = INPUT_LAYOUT)

# -------------------------------------------------------------------
# CustimizedPass 创建了一个自定义的量化优化过程
# 在 PPQ 中，你的网络是被一个个的 QuantizationOptimizationPass 所量化的
# 你可以在 CustimizedPass 中书写自己的逻辑，或者添加新的 QuantizationOptimizationPass 子类
# 在 manop 函数中调用你定义的方法从而完成自定义的量化优化过程。
# -------------------------------------------------------------------
class CustimizedPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Custimized Optimization Pass')
    def optimize(self, processor: GraphCommandProcessor,
                 dataloader: Iterable, executor: TorchExecutor, **kwargs) -> None:
        graph = processor.graph

# -----------------------------------------------------------------
# 我们允许你在标准量化流程之前添加自定义量化逻辑，通常在标准量化流程之前添加的逻辑用于：
# 标准化网络格式，分裂或融合网络算子，变更网络结构以降低量化误差等
# 在标准量化流程开始之前，你不能访问或修改网络中的量化参数 -- 因为它们还未正确初始化。
# 当然，你也可以调用我们提供的任意 QuantizationOptimizationPass 来完成你的自定义量化逻辑。

# 在量化开始之前，你可以使用的 QuantizationOptimizationPass 包括
# LayerwiseEqualizationPass -- 用于调整网络权重以降低量化误差
# MetaxGemmSplitPass        -- 用于将 Gemm 分解成 matmul + add
# GRUSplitPass              -- 用于将 Gru  分解成 Gemm
# MatrixFactorizationPass   -- 用于执行纵向算子分裂，降低量化误差
# WeightSplitPass           -- 用于执行横向算子分裂，降低量化误差
# ChannelSplitPass          -- 用于执行通道分裂，降低量化误差
# -------------------------------------------------------------------
custimized_prequant_passes = [CustimizedPass()]
manop(graph=graph, list_of_passes=custimized_prequant_passes,
      calib_dataloader=dataloader, executor=TorchExecutor(graph, device=EXECUTING_DEVICE),
      collate_fn=None)

print('网络正量化中，根据你的量化配置，这将需要一段时间:')
quantized = quantize_native_model(
    setting=QS,                     # setting 对象用来控制标准量化逻辑
    model=graph,
    calib_dataloader=dataloader,
    calib_steps=32,
    input_shape=NETWORK_INPUTSHAPE, # 如果你的网络只有一个输入，使用这个参数传参
    inputs=None,                    # 如果你的网络有多个输入，使用这个参数传参
    collate_fn=lambda x: x.to(EXECUTING_DEVICE),  # collate_fn 跟 torch dataloader 的 collate fn 是一样的，用于数据预处理，
                                                  # 你当然也可以用 torch dataloader 的那个，然后设置这个为 None
    platform=TARGET_PLATFORM,
    device=EXECUTING_DEVICE,
    do_quantize=True)

# -----------------------------------------------------------------
# 我们允许你在标准量化流程之后添加自定义量化逻辑，通常在标准量化流程之后添加的逻辑用于：
# 调整量化信息，再训练网络权重
# 当然，你也可以调用我们提供的任意 QuantizationOptimizationPass 来完成你的自定义量化逻辑。

# 在量化开始之前，你可以使用的 QuantizationOptimizationPass 包括
# ParameterBakingPass           -- 用于将权重直接烘焙成量化后的值
# RuntimeCalibrationPass        -- 用于重新确定量化参数
# MishFusionPass                -- 用于执行 Mish 算子联合定点
# SwishFusionPass               -- 用于执行 Swish 算子联合定点
# AdvancedQuantOptimization     -- 用于再训练网络，降低量化误差
# LearningStepSizeOptimization  -- 用于再训练网络，降低量化误差
# -------------------------------------------------------------------
custimized_postquant_passes = [CustimizedPass()]
manop(graph=graph, list_of_passes=custimized_postquant_passes,
      calib_dataloader=dataloader, executor=TorchExecutor(graph, device=EXECUTING_DEVICE),
      collate_fn=None)

# -------------------------------------------------------------------
# 如果你需要执行量化后的神经网络并得到结果，则需要创建一个 executor
# 这个 executor 的行为和 torch.Module 是类似的，你可以利用这个东西来获取执行结果
# 请注意，必须在 export 之前执行此操作。
# -------------------------------------------------------------------
executor = TorchExecutor(graph=quantized, device=EXECUTING_DEVICE)
# output = executor.forward(input)

# -------------------------------------------------------------------
# 导出 PPQ 执行网络的所有中间结果，该功能是为了和硬件对比结果
# 中间结果可能十分庞大，所有的中间结果将被产生在 WORKING_DIRECTORY 中
# -------------------------------------------------------------------
if DUMP_RESULT:
    dump_internal_results(
        graph=quantized, dataloader=dataloader, sample=False,
        dump_dir=WORKING_DIRECTORY, executing_device=EXECUTING_DEVICE)

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
# PPQ 会根据你所选择的导出平台来修改模型格式，请知悉：
# 如果你选择 tensorRT 作为导出平台，则我们将直接导出量化后的engine文件，tensorRT可以直接运行；
# 如果你选择 ppl 系列作为导出平台，则我们将优化后的onnx文件以及json格式的量化参数，ppl需要这两个东西以运行量化模型；
# 如果你选择 snpe 作为导出平台，量化参数将被写入caffe proto文件，snpe好像...可以直接运行；
# 如果你选择 metax 作为导出平台，他们还在写后端框架所以短时间内可能不能执行；
# 如果你选择 tengine 作为导出平台，他们还在写接口来读取ppq的输入，所以好像也不能执行；
# 如果你选择 nxp 作为导出格式，量化参数将被写入onnx，nxp可以直接运行；
# 如果你选择 onnxruntime 作为导出格式，我们将在网络中插入quant以及dequant节点，onnxruntime可以直接运行。
# 如果你选择 onnxruntime OOS 作为导出格式，我们将会弄出一些 com.microsoft 定义量化算子，onnxruntime可以直接运行。
# 如果你选择 onnx 作为导出格式，我们将导出一个ppq原生的格式，这个格式只是用来debug的。

# 如果你想最快速的看到结果，选择onnxruntime作为导出格式即可，你就可以在导出的onnx中看到量化结果。
# 请不要使用 onnxruntime 执行量化后的文件，那样无法加速。
# 如果你想加速你的网络，请使用 tensorRT, ncnn, openvino, openppl, tengine 等

# 所有导出平台被列举在ppq.api.interface.py文件中：
# EXPORTERS = {
# TargetPlatform.PPL_DSP_INT8:  PPLDSPCaffeExporter,
# TargetPlatform.PPL_DSP_TI_INT8: PPLDSPTICaffeExporter,
# TargetPlatform.QNN_DSP_INT8:  QNNDSPExporter,
# TargetPlatform.PPL_CUDA_INT8: PPLBackendExporter,
# TargetPlatform.SNPE_INT8:     SNPECaffeExporter,
# TargetPlatform.NXP_INT8:      NxpExporter,
# TargetPlatform.ONNX:          OnnxExporter,
# TargetPlatform.ONNXRUNTIME:   ONNXRUNTIMExporter,
# TargetPlatform.OPENVINO_INT8: ONNXRUNTIMExporter,
# TargetPlatform.CAFFE:         CaffeExporter,
# TargetPlatform.NATIVE:        NativeExporter,
# TargetPlatform.EXTENSION:     ExtensionExporter,
# TargetPlatform.ORT_OOS_INT8:  ORTOOSExporter,
# TargetPlatform.METAX_INT8_C:  MetaxExporter,
# TargetPlatform.METAX_INT8_T:  MetaxExporter,
# TargetPlatform.TRT_INT8:      TensorRTExporter,
# TargetPlatform.NCNN_INT8:     NCNNExporter
# }
# -------------------------------------------------------------------
print('网络量化结束，正在生成目标文件:')
export_ppq_graph(
    graph=quantized, platform=TARGET_PLATFORM,
    graph_save_to = os.path.join(WORKING_DIRECTORY, 'quantized'),
    config_save_to = os.path.join(WORKING_DIRECTORY, 'quant_cfg.json'))

# 如果你需要导出 CAFFE 模型，使用下面的语句，caffe exporter 需要一个 input_shapes 参数。
# export(working_directory=WORKING_DIRECTORY,
#        quantized=quantized, platform=TARGET_PLATFORM,
#        input_shapes=[NETWORK_INPUTSHAPE])
