# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理

# This script shows you how to export ppq internal graph to tensorRT
# ---------------------------------------------------------------

# For this inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
import onnxruntime
import torch
from ppq import *
from ppq.api import *
from tqdm import tqdm

QUANT_PLATFROM = TargetPlatform.TRT_INT8
MODEL          = 'model.onnx'
INPUT_SHAPE    = [1, 3, 224, 224]
SAMPLES        = [torch.rand(size=INPUT_SHAPE) for _ in range(256)] # rewirte this to use real data.
DEVICE         = 'cuda'
FINETUNE       = True
QS             = QuantizationSettingFactory.default_setting()
EXECUTING_DEVICE = 'cuda'
REQUIRE_ANALYSE  = True

# -------------------------------------------------------------------
# 下面向你展示了常用参数调节选项：
# -------------------------------------------------------------------
if PPQ_CONFIG.USING_CUDA_KERNEL:
    QS.advanced_optimization = FINETUNE                             # 启动网络再训练过程，降低量化误差
    QS.advanced_optimization_setting.steps = 2500                   # 再训练步数，影响训练时间，2500步大概几分钟
    QS.advanced_optimization_setting.collecting_device = 'executor' # 缓存数据放在那，executor 就是放在gpu，如果显存超了你就换成 'cpu'
    QS.advanced_optimization_setting.auto_check = False             # 打开这个选项则训练过程中会防止过拟合，以及意外情况，通常不需要开。
else:
    QS.lsq_optimization = FINETUNE                                  # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.epochs = 30                         # 再训练轮数，影响训练时间，30轮大概几分钟
    QS.lsq_optimization_setting.collecting_device = 'cuda'          # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'

QS.dispatching_table.append(operation='OP NAME', platform=TargetPlatform.FP32) # 把量化的不太好的算子送回 FP32

print('正准备量化你的网络，检查下列设置:')
print(f'TARGET PLATFORM      : {QUANT_PLATFROM.name}')
print(f'NETWORK INPUTSHAPE   : {INPUT_SHAPE}')

# ENABLE CUDA KERNEL 会加速量化效率 3x ~ 10x，但是你如果没有装相应编译环境的话是编译不了的
# 你可以尝试安装编译环境，或者在不启动 CUDA KERNEL 的情况下完成量化：移除 with ENABLE_CUDA_KERNEL(): 即可
with ENABLE_CUDA_KERNEL():
    qir = quantize_onnx_model(
        onnx_import_file=MODEL, calib_dataloader=SAMPLES, calib_steps=128, setting=QS,
        input_shape=INPUT_SHAPE, collate_fn=lambda x: x.to(EXECUTING_DEVICE), 
        platform=QUANT_PLATFROM, do_quantize=True)

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
    # 启动 tensorRT 进行推理，你先装一下 trt
    # -------------------------------------------------------------------
    import tensorrt as trt
    import trt_infer

    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]
    logger = trt.Logger(trt.Logger.INFO)
    with open('model_int8.engine', 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        for sample in tqdm(samples, desc='TensorRT is running...'):
            inputs[0].host = convert_any_to_numpy(sample)
            [output] = trt_infer.do_inference(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream, batch_size=1)
            results.append(convert_any_to_torch_tensor(output).reshape([-1, 1000]))
