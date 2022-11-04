import torch
import torchvision
from ppq import *
from ppq.api import *

# ------------------------------------------------------------
# 在这个脚本中，我们将向你展示如何自由调度算子，并实现混合精度推理
# 在开始之前，我们首先设计一个支持混合精度的量化器
# ------------------------------------------------------------
class MyQuantizer(BaseQuantizer):
    
    # ------------------------------------------------------------
    # quant_operation_types 是一个类型枚举，在这里你需要写下所有该量化器所需要量化的算子
    # ------------------------------------------------------------
    @ property
    def quant_operation_types(self) -> set:
        return {'Conv'}

    # ------------------------------------------------------------
    # 一旦你确定了那些算子需要量化，则需要在 init_quantize_config 为他们初始化量化信息
    # 然而你需要注意的是，由于手动调度的存在，用户可以强制调度一个类型不在 quant_operation_types 中的算子来到量化平台
    # 我建议你针对这类情况进行回应。或者，在探测到算子类型并非可量化类型后进行报错
    # ------------------------------------------------------------
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        # ------------------------------------------------------------
        # 为卷积算子初始化量化信息，只量化卷积算子的输入(input & weight)，bias 不做量化
        # ------------------------------------------------------------
        if operation.type == 'Conv':
            config = self.create_default_quant_config(
                op                 = operation, 
                num_of_bits        = 4,
                quant_max          = 15, 
                quant_min          = -16,
                observer_algorithm = 'percentile', 
                policy             = QuantizationPolicy(
                    QuantizationProperty.PER_TENSOR +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.SYMMETRICAL),
                rounding           = RoundingPolicy.ROUND_HALF_EVEN)

            # ------------------------------------------------------------
            # 关闭所有输出量化，状态设置为fp32
            # ------------------------------------------------------------
            for tensor_quant_config in config.output_quantization_config:
                tensor_quant_config.state = QuantizationStates.FP32
                
            # ------------------------------------------------------------
            # 关闭 bias 量化，状态设置为fp32
            # ------------------------------------------------------------
            if operation.num_of_input == 3:
                config.input_quantization_config[-1].state = QuantizationStates.FP32

            # ------------------------------------------------------------
            # 如果算子被调度到 INT8 平台上，执行 INT8 的量化
            # ------------------------------------------------------------
            if operation.platform == TargetPlatform.ACADEMIC_INT8:
                print(f'{operation.name} has been dispatched to INT8')
                config.input_quantization_config[0].num_of_bits = 8
                config.input_quantization_config[0].quant_max   = 127
                config.input_quantization_config[0].quant_min   = -128

                config.input_quantization_config[1].num_of_bits = 8
                config.input_quantization_config[1].quant_max   = 127
                config.input_quantization_config[1].quant_min   = -128

            return config
        else:
            raise TypeError(f'Unsupported Op Type: {operation.type}')

    # ------------------------------------------------------------
    # 当前量化器进行量化的算子都将被发往一个指定的目标平台
    # 这里我们选择 TargetPlatform.ACADEMIC_INT4 作为目标平台
    # ------------------------------------------------------------
    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ACADEMIC_INT4


# 注册我们的量化器
register_network_quantizer(MyQuantizer, platform=TargetPlatform.ACADEMIC_INT4)

# ------------------------------------------------------------
# 下面，我们向你展示 PPQ 的手动调度逻辑
# 我们仍然以 MobilenetV2 举例，向你展示如何完成混合精度调度
# ------------------------------------------------------------

BATCHSIZE   = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE      = 'cuda'
PLATFORM    = TargetPlatform.ACADEMIC_INT4
CALIBRATION = [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
def collate_fn(batch: torch.Tensor) -> torch.Tensor: return batch.to(DEVICE)

model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# ------------------------------------------------------------
# 为了执行手动调度，你必须首先创建 QuantizationSetting
# 使用 QuantizationSetting.dispatch_table 属性来传递调度方案
# 大多数预制量化器没有针对 INT4 写过量化方案，因此只支持 FP32 - INT8 的相互调度
# ------------------------------------------------------------
QS = QuantizationSettingFactory.default_setting()

# 示例语句：下面的语句将把 Op1 调度到 FP32 平台
# QS.dispatching_table.append(operation='Op1', platform=TargetPlatform.FP32)

# ------------------------------------------------------------
# 我如何知道调度那些层到高精度会得到最优的性能表现？
# layerwise_error_analyse 函数正是为此设计的，我们调用该方法
# 然后选择那些误差较高的层调度到高精度平台

# 我们首先将所有层全部送往 INT4，然后执行误差分析
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
        device=DEVICE, verbose=0, setting=QS)

    # ------------------------------------------------------------
    # 使用 graphwise_analyse 衡量调度前的量化误差
    # ------------------------------------------------------------
    reports = graphwise_error_analyse(
        graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
        dataloader=CALIBRATION)

    # ------------------------------------------------------------
    # 执行逐层分析，结果是一个字典，该字典内写入了所有算子的单层量化误差
    # ------------------------------------------------------------
    reports = layerwise_error_analyse(
        graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
        dataloader=CALIBRATION, verbose=False)

    # 从大到小排序单层误差
    sensitivity = [(op_name, error) for op_name, error in reports.items()]
    sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------
    # 将前十个误差最大的层送上INT8，并重新量化
    # ------------------------------------------------------------
    for op_name, _ in sensitivity[: 10]:
        QS.dispatching_table.append(operation=op_name, platform=TargetPlatform.ACADEMIC_INT8)
    graph = load_onnx_graph(onnx_import_file='Output/model.onnx')
    quantized = quantize_native_model(
        model=graph, calib_dataloader=CALIBRATION,
        calib_steps=32, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn, platform=PLATFORM,
        device=DEVICE, verbose=0, setting=QS)

    # ------------------------------------------------------------
    # 使用 graphwise_analyse 衡量最终的量化误差
    # ------------------------------------------------------------
    reports = graphwise_error_analyse(
        graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
        dataloader=CALIBRATION)