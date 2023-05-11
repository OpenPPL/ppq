
import ppq.lib as PFL
import torch
import torchvision
from ppq.api import ENABLE_CUDA_KERNEL, load_native_graph, load_torch_model
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.quantization.optim import *
from ppq.quantization.quantizer import TensorRTQuantizer
from Utilities.Imagenet import *  # check ppq.samples.imagenet.Utilities
from Utilities.Imagenet.imagenet_util import \
    load_imagenet_from_directory  # check ppq.samples.imagenet.Utilities

from trainer import ImageNetTrainer

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
# 我们首先进行标准的量化流程，为所有算子初始化量化信息，并进行 Calibration
# ------------------------------------------------------------
quantizer = TensorRTQuantizer(graph=graph) 
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
