# Test Quantization System Performace on MMEditing models with CityScapes

# Quantizer Configuration
SYMMETRICAL = True
PERCHANNEL  = True
POWER_OF_2  = False
BIT_WIDTH   = 8
FP8         = True

# Should contains model file(.onnx)
MODEL_DIR = 'QuantZoo/Model/mmseg'

# Should contains Calib & Test Img Folder
CALIB_DIR = 'QuantZoo/Data/Cityscapes/Calib'
TEST_DIR  = 'QuantZoo/Data/Cityscapes/Test'

# calibration & test batchsize
BATCHSIZE = 1

# write report to here
REPORT_DIR = 'QuantZoo/Reports'

CONFIGS = [

{
    'Model': 'stdc1_512x1024_80k_cityscapes',
    'Output': ['/convs/convs.0/activate/Relu_output_0']
},

{
    'Model': 'pspnet_r50-d8_512x1024_40k_cityscapes',
    'Output': ['/bottleneck/activate/Relu_output_0']
},

{
    'Model': 'pointrend_r50_512x1024_80k_cityscapes', # complex model
    'Output': ['/Concat_60_output_0']
},

{
    'Model': 'fpn_r50_512x1024_80k_cityscapes',
    'Output': ['/Add_2_output_0']
},

{
    'Model': 'icnet_r18-d8_832x832_80k_cityscapes',
    'Output': ['/convs/convs.0/activate/Relu_output_0']
},

{
    'Model': 'fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes',
    'Output': ['/convs/convs.0/activate/Relu_output_0']
},

{
    'Model': 'fast_scnn_lr0.12_8x4_160k_cityscapes',
    'Output': ['/convs/convs.1/pointwise_conv/activate/Relu_output_0']
},

{
    'Model': 'fcn_r50-d8_512x1024_40k_cityscapes',
    'Output': ['/conv_cat/activate/Relu_output_0']
},

{
    'Model': 'bisenetv2_fcn_4x4_1024x1024_160k_cityscapes',
    'Output': ['/convs/convs.0/activate/Relu_output_0']
},

{
    'Model': 'deeplabv3_r50-d8_512x1024_40k_cityscapes',
    'Output': ['/bottleneck/activate/Relu_output_0']
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
from QuantZoo.Data.Cityscapes.Data import load_cityscapes_dataset
from QuantZoo.Data.Cityscapes.Eval import evaluation
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze, report

calib_loader = load_cityscapes_dataset(img_folder=CALIB_DIR)
test_loader = load_cityscapes_dataset(img_folder=TEST_DIR)


with ENABLE_CUDA_KERNEL():
    for config in CONFIGS:
        model = config['Model']
        monitoring_vars = config['Output']

        print(f"Ready to run quant benchmark on {model}")
        graph = load_onnx_graph(onnx_import_file=os.path.join(MODEL_DIR, model + '.onnx'))
        quantizer = MyInt8Quantizer(
            graph=graph, sym=SYMMETRICAL, power_of_2=POWER_OF_2, 
            num_of_bits=BIT_WIDTH, per_channel=PERCHANNEL)
        if FP8: quantizer = MyFP8Quantizer(graph=graph, calibration='floating')
        
        # convert op to quantable-op
        for name, op in graph.operations.items():
            if op.type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm'}:
                quantizer.quantize_operation(name, platform=TargetPlatform.INT8)

        # build quant pipeline.
        pipeline = PFL.Pipeline([
            # LayerwiseEqualizationPass(iteration=10),
            ParameterQuantizePass(),
            RuntimeCalibrationPass(),
            # LearnedStepSizePass(steps=500, collecting_device='cpu')
        ])

        # call pipeline.
        executor = TorchExecutor(graph=graph)
        executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 1024, 2048]).cuda())

        pipeline.optimize(
            graph=graph, dataloader=calib_loader, verbose=True,
            calib_steps=32, collate_fn=lambda x: x[0].to('cuda'), executor=executor)

        miou = evaluation(graph=graph, dataloader=test_loader, working_directory=TEST_DIR)
        print(f'Model Performance on CityScapes Miou: {miou * 100: .4f}%')

        # error analyze
        performance = error_analyze(
            graph=graph,
            outputs=monitoring_vars,
            dataloader=test_loader, 
            collate_fn=lambda x: x[0].cuda(),
            verbose=True
        )