
import torch
import torchvision
from trainer import ImageNetTrainer
from Utilities.Imagenet import *  # check ppq.samples.imagenet.Utilities
from Utilities.Imagenet.imagenet_util import \
    load_imagenet_from_directory  # check ppq.samples.imagenet.Utilities

import ppq.lib as PFL
from ppq.api import ENABLE_CUDA_KERNEL, load_native_graph, load_torch_model
from ppq.core import (OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates,
                      QuantizationVisibility, RoundingPolicy, TargetPlatform)
from ppq.executor import TorchExecutor
from ppq.IR import Operation
from ppq.quantization.optim import *

"""
    使用这个脚本来尝试在 Imagenet 数据集上执行量化感知训练
        使用 imagenet 中的数据测试量化精度与 calibration
        默认的 imagenet 数据集位置:Assets/Imagenet_Train, Assets/Imagenet_Valid
        你可以通过软连接创建它们:
            ln -s /home/data/Imagenet/val Assets/Imagenet_Valid
            ln -s /home/data/Imagenet/train Assets/Imagenet_Train
"""

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 32                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = 'Assets/Imagenet_Valid'   # 用来读取 validation dataset
CFG_TRAIN_DIR = 'Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_PLATFORM = TargetPlatform.TRT_INT8         # 用来指定目标平台

# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何在 PPQ 中对你的网络进行量化感知训练
# 你可以使用带标签的数据执行正常的训练流程，也可以使用类似蒸馏的方式进行无标签训练
# PPQ 模型的训练过程与 Pytorch 遵循相同的逻辑，你可以使用 Pytorch 中的技巧来获得更好的训练效果
# ------------------------------------------------------------
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
graph = load_torch_model(model=model, sample=torch.zeros([1,3,224,224]).cuda())

# ------------------------------------------------------------
# 创建自定义的 Quantizer, 并完成量化
# 你需要参考 ppq.quantization.quantizer 文件夹里其他的 quantizer 定义来设计你的量化规则
# Quantizer 负责为所有算子初始化量化信息，你需要实现它的所有接口函数
# ------------------------------------------------------------
from ppq.quantization.quantizer import BaseQuantizer


class MyQuantizer(BaseQuantizer):
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        """
        函数 init_quantize_config 是 Quantizer 类的核心接口函数。PPQ 使用这个函数为所有算子初始化 TQC(Tensor Quantization Config)。        
        通常而言，我们将根据算子的类型、算子的调度情况（算子上的 platform 属性）来决定如何量化给定的算子。
        
        你需要认识到一个事实，在 PPQ 中所有的逻辑都围绕着 TQC(Tensor Quantization Config) 这一控制结构体展开
            - Quantizer 负责初始化 TQC
            - Optim Passes 负责调整 TQC 的参数和状态
            - Exporter 负责导出 TQC
        
        init_quantize_config 函数只有一个参数，即需要被量化的算子，返回一个量化算子的量化控制结构体。

        Args:
            operation (Operation): 需要被量化的算子

        Returns:
            OperationQuantizationConfig: 返回的控制结构体
        """
        # ------------------------------------------------------------
        # 成员函数 create_default_quant_config 负责提供默认的 OQC (Operation Quantization Config)
        # 数据结构 OQC 只是 TQC 的一个简单集合，对于任何一个 onnx 算子而言，它都会具有 n 个输入变量和 m 个输出变量
        # 因此对应的 OQC 内将包含 n 的关于输入变量的 TQC, 以及 m 个关于输出变量的 TQC
        
        # 以卷积算子为例，它可以有 2 - 3 个输入变量，以及至多一个输出变量
        # 因此它的 OQC 内将含有 2 - 3 个输入 TQC，分别对应 输入量化、权重量化、bias 量化
        
        # OQC 一旦创建，则图结构不能发生改变，因此图的变换需要发生在量化之前
        # 因为图结构一旦变换，则可能导致算子的输入个数变化，从而导致 OQC 不匹配
        
        # 下面的代码将把算子上所有的 TQC 初始化为非对称 PER_TENSOR 量化
        # ------------------------------------------------------------
        OQC = self.create_default_quant_config(
            policy=QuantizationPolicy(
                QuantizationProperty.ASYMMETRICAL + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.PER_TENSOR), 
            rounding=RoundingPolicy.ROUND_HALF_EVEN,
            op=operation, num_of_bits=8, exponent_bits=0,
            quant_max=255, quant_min=0,
            observer_algorithm='percentile')

        # 对于卷积和矩阵乘算子而言，它的 bias 量化通常需要使用 scale = input scale * weight scale
        # 此时我们希望你将卷积的 bias 量化状态设置为 PASSIVE_INIT, 而后 PassiveParameterQuantizePass 将会正确处理 bias 的 scale 问题
        # 于此同时你还需要将 bias 的量化位宽调整到 32 位，并且相应的调整 quant min, quant max
        # 同时将 bias 的量化调整为 对称 PER_TENSOR 量化
        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            if operation.num_of_input == 3:
                TQC = OQC.input_quantization_config[-1]
                TQC.state = QuantizationStates.PASSIVE_INIT
                TQC.num_of_bits = 32
                TQC.quant_max = 1 << 30
                TQC.quant_min = -1 << 30
                TQC.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL + 
                    QuantizationProperty.LINEAR + 
                    QuantizationProperty.PER_TENSOR)
                TQC.observer_algorithm = 'minmax'

        # 对于卷积和矩阵乘算子而言，它们的权重可以进行 PER_CHANNEL 量化，因此你可以将权重量化的策略调整为 PER_CHANNEL
        # 于此同时我们将权重量化策略调整为 对称 PER_CHANNEL
        # 相应地修改 quant min, quant max
        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:

            # 有些时候，矩阵乘法算子并没有权重，而是两路激活值相乘，因此我们需要单独判断这种情况
            if not operation.inputs[1].is_parameter:
                TQC = OQC.input_quantization_config[1]
                TQC.num_of_bits = 8
                TQC.quant_max = 127
                TQC.quant_min = -128
                TQC.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL + 
                    QuantizationProperty.LINEAR + 
                    QuantizationProperty.PER_CHANNEL)
                TQC.observer_algorithm = 'minmax'

        # 你也可以按算子的名字为算子创建不一样的量化信息
        # 下面的代码把名字叫 MyLayer 的层指定为 PER TENSOR 量化
        if operation.name == 'MyLayer' and operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:
            TQC = OQC.input_quantization_config[1]
            TQC.num_of_bits = 8
            TQC.quant_max = 127
            TQC.quant_min = -128
            TQC.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.PER_TENSOR)
            TQC.observer_algorithm = 'minmax'

        # 如果算子类型不在支持的范围内，我们给出警告
        if operation.type not in self.quant_operation_types:
            print(f'Operation {operation.name}({operation.type}) is not supported with current Quantizer.')

        return OQC

    def quant_operation_types(self) -> set:
        """ quant_operation_types 返回一个类型集合，该集合被用于调度和量化 """
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Softmax',
            'Mul', 'Add', 'Max', 'Sub', 'Div', 'Reshape',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Interp',
            'ReduceMean', 'Transpose', 'Slice', 'Flatten',
            'HardSwish', 'HardSigmoid', 'MatMul'}
    
    def activation_fusion_types(self) -> set:
        """ 
            quant_operation_types 返回一个类型集合，该集合被用于图融合
            
            对于任何一个类型在 activation_fusion_types 中的算子而言
            PPQ QuantizeFusionPass 会探测 Conv + activation, Gemm + activation 等结构
            会取消 Conv, Gemm 的输出量化及 activation 的输入量化，从而模拟推理时可能发生的激活算子融合
            
        """
        return {'Clip', 'Relu', 'Sigmoid', 'Swish', 'SoftPlus', 'Gelu', 'LeakyRelu'}

quantizer = MyQuantizer(graph=graph) 
dispatching_table = PFL.Dispatcher(graph=graph).dispatch(quantizer.quant_operation_types)

# 为算子初始化量化信息
for op in graph.operations.values():
    quantizer.quantize_operation(
        op_name=op.name, platform=dispatching_table[op.name])

executor = TorchExecutor(graph=graph)
executor.tracing_operation_meta(inputs=torch.zeros([1,3,224,224]).cuda())

calib_dataloader = load_imagenet_from_directory(
    directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
    shuffle=True, subset=1280, require_label=True,
    num_of_workers=8)

training_dataloader = load_imagenet_from_directory(
    directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
    shuffle=True, require_label=True,
    num_of_workers=8)

eval_dataloader = load_imagenet_from_directory(
    directory=CFG_VALIDATION_DIR, batchsize=CFG_BATCHSIZE,
    shuffle=True, require_label=True,
    num_of_workers=8)

with ENABLE_CUDA_KERNEL():
    # ------------------------------------------------------------
    # 创建优化管线，由于后续还要继续训练我们的模型，我们不能在此处调用
    # ParameterBakingPass()，一旦模型权重完成烘焙，则它们不能被进一步调整
    # ------------------------------------------------------------
    pipeline = PFL.Pipeline([
        QuantizeSimplifyPass(),
        QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
        ParameterQuantizePass(),
        RuntimeCalibrationPass(method='kl'),
        PassiveParameterQuantizePass(),
        QuantAlignmentPass(),
        # LearnedStepSizePass(steps=500, block_size=5)
    ])
    pipeline.optimize(
        calib_steps=8, collate_fn=lambda x: x[0].cuda(),
        graph=graph, dataloader=calib_dataloader, executor=executor)
    
    # ------------------------------------------------------------
    # 完成量化后，我们将开始进行 QAT 的模型训练，我们希望你能够注意到：
    #
    # 1. 不能从零开始使用 QAT 的方法完成训练，你应当先训练好浮点的模型，或者在一个预训练的模型基础上进行 QAT finetuning.
    # 2. 你必须完成标准量化流程
    # 3. PPQ Executor 长得很像 Pytorch Module，单机训练应该不会遇到太多困难，但它不支持多卡训练
    
    # 训练的代码我们封装进了一个叫做 ImageNetTrainer 的东西
    # 你可以打开它看到具体的训练逻辑
    # ------------------------------------------------------------
    trainer = ImageNetTrainer(graph=graph)

    best_acc = 0
    for epoch in range(20):
        trainer.epoch(training_dataloader)
        current_acc = trainer.eval(eval_dataloader)
        if current_acc > best_acc:
            trainer.save('Best.native')

    graph = load_native_graph(import_file='Best.native')
    PFL.Exporter(platform=CFG_PLATFORM).export(
        file_path='export.onnx', graph=graph, config_path='export.json')
