# Test Quantization System Performace on Image Classification Models with ILSVRC2012 Dataset
#
#   1. How to use: 
#      Run this script with python directly.

# Quantizer Configuration
SYMMETRICAL = True
POWER_OF_2  = True
PER_CHANNEL = False
FP8         = True
BIT_WIDTH   = 8

# Should contains model file(.onnx)
MODEL_DIR = 'QuantZoo/Model/Imagenet'

# Should contains Calib & Test Img Folder
CALIB_DIR = 'QuantZoo/Data/Imagenet/Calib'
TEST_DIR  = 'QuantZoo/Data/Imagenet/Test'

# calibration & test batchsize
BATCHSIZE = 32

# write report to here
REPORT_DIR = 'QuantZoo/Reports'

CONFIGS = [
{
    'Model': 'repvgg',
    'Output': ['input.172']
},
{
    'Model': 'efficientnet_v1_b0',
    'Output': ['/features/features.8/features.8.2/Mul_output_0']
},
{
    'Model': 'efficientnet_v1_b1',
    'Output': ['/features/features.8/features.8.2/Mul_output_0']
},
{
    'Model': 'efficientnet_v2_s',
    'Output': ['/features/features.7/features.7.2/Mul_output_0']
},
{
    'Model': 'mnasnet0_5',
    'Output': ['/layers/layers.16/Relu_output_0']
},
{
    'Model': 'mnasnet1_0',
    'Output': ['/layers/layers.16/Relu_output_0']
},
{
    'Model': 'mobilenet_v2',
    'Output': ['/features/features.18/features.18.2/Clip_output_0']
},
{
    'Model': 'resnet18',
    'Output': ['/layer4/layer4.1/relu_1/Relu_output_0']
},
{
    'Model': 'resnet50',
    'Output': ['/layer4/layer4.2/relu_2/Relu_output_0']
},

{
    'Model': 'mobilenet_v3_large',
    'Output': ['/classifier/classifier.1/Mul_output_0']
},
{
    'Model': 'mobilenet_v3_small',
    'Output': ['/classifier/classifier.1/Mul_output_0']
},
{
    'Model': 'v100_gpu64@5ms_top1@71.6_finetune@25',
    'Output': ['471']
},
{
    'Model': 'v100_gpu64@6ms_top1@73.0_finetune@25',
    'Output': ['471']
},
{
    'Model': 'lcnet_050',
    'Output': ['/act2/Mul_output_0']
},
{
    'Model': 'lcnet_100',
    'Output': ['/act2/Mul_output_0']
},


{
    # vit_b_16 requires BATCHSIZE = 1!
    'Model': 'vit_b_16',
    'Output': ['onnx::Gather_1703']
},
]

import os

import torch

import ppq.lib as PFL
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.quantization.optim import (LayerwiseEqualizationPass,
                                    LearnedStepSizePass, ParameterQuantizePass,
                                    RuntimeCalibrationPass)
from QuantZoo.Data.Imagenet.Eval import (evaluate_ppq_module_with_imagenet,
                                         load_imagenet_from_directory)
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze


calib_loader = load_imagenet_from_directory(
    directory=CALIB_DIR, batchsize=BATCHSIZE,
    shuffle=False, require_label=False,
    num_of_workers=8)


test_loader = load_imagenet_from_directory(
    directory=TEST_DIR, batchsize=BATCHSIZE,
    shuffle=False, require_label=True, 
    num_of_workers=8)


with ENABLE_CUDA_KERNEL():
    for config in CONFIGS:
        model = config['Model']
        monitoring_vars = config['Output']

        print(f"Ready to run quant benchmark on {model}")
        graph = load_onnx_graph(onnx_import_file=os.path.join(MODEL_DIR, model + '.onnx'))

        if model == 'vit_b_16':
            if BATCHSIZE == 32:
                raise Exception('To Evaluate vit_b_16, change batchsize to 1, change calibration method to minmax.')
            from ppq.IR import GraphMerger
            processor = GraphMerger(graph)
            processor.fuse_matmul_add()
            processor.fuse_layernorm()
            processor.fuse_gelu()

        quantizer = MyInt8Quantizer(
            graph=graph, sym=SYMMETRICAL, power_of_2=POWER_OF_2, 
            num_of_bits=BIT_WIDTH, per_channel=PER_CHANNEL)
        if FP8: quantizer = MyFP8Quantizer(graph=graph, calibration='floating')

        # convert op to quantable-op
        for name, op in graph.operations.items():
            if op.type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 
                           'PPQBiasFusedMatMul', 'LayerNormalization'}:
                quantizer.quantize_operation(name, platform=TargetPlatform.INT8)

        # build quant pipeline.
        pipeline = PFL.Pipeline([
            # LayerwiseEqualizationPass(iteration=10),
            ParameterQuantizePass(),
            RuntimeCalibrationPass(),
            # LearnedStepSizePass(steps=500, collecting_device='cuda', block_size=5)
        ])

        # call pipeline.
        executor = TorchExecutor(graph=graph)
        executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 224, 224]).cuda())

        pipeline.optimize(
            graph=graph, dataloader=calib_loader, verbose=True,
            calib_steps=32, collate_fn=lambda x: x.to('cuda'), executor=executor)

        # evaluation
        acc = evaluate_ppq_module_with_imagenet(
            model=graph, imagenet_validation_loader=test_loader,
            batchsize=BATCHSIZE, device='cuda', verbose=False)
        print(f'Model Classify Accurarcy = {acc: .4f}%')

        # error analyze
        performance = error_analyze(
            graph=graph,
            outputs=monitoring_vars,
            dataloader=test_loader, 
            collate_fn=lambda x: x[0].to('cuda'),
            verbose=True
        )