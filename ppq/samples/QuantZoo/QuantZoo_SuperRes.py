# Test Quantization System Performance on SuperRes Task(DIV2K)

# Quantizer Configuration
SYMMETRICAL = True
PER_CHANNEL = True
POWER_OF_2  = False
BIT_WIDTH   = 8
FP8         = True

# Should contains model file(.onnx)
MODEL_DIR = 'QuantZoo/Model/mmedit'

# Should contains Calib & Test Img Folder
TRAIN_HR_DIR = 'QuantZoo/Data/DIV2K/DIV2K_train_HR'
TRAIN_LR_DIR = 'QuantZoo/Data/DIV2K/DIV2K_train_LR_bicubic'
VALID_HR_DIR = 'QuantZoo/Data/DIV2K/DIV2K_valid_HR'
VALID_LR_DIR = 'QuantZoo/Data/DIV2K/DIV2K_valid_LR_bicubic'

# calibration & test batchsize
# super resolution model must have batchsize = 1
BATCHSIZE = 1

# write report to here
REPORT_DIR = 'QuantZoo/Reports'

CONFIGS = [
{
    'Model': 'srcnn_x4k915_g1_1000k_div2k',
    'Output': ['output'],
},
{
    'Model': 'srgan_x4c64b16_g1_1000k_div2k',
    'Output': ['output'],
},
{
    'Model': 'rdn_x4c64b16_g1_1000k_div2k',
    'Output': ['output'],
},
{
    'Model': 'edsr_x4c64b16_g1_300k_div2k',
    'Output': ['/generator/conv_last/Conv_output_0'],
},
]

import os
from typing import Iterable

import torch
from tqdm import tqdm

import ppq.lib as PFL
from ppq import convert_any_to_numpy
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph
from ppq.quantization.optim import (LayerwiseEqualizationPass,
                                    LearnedStepSizePass, ParameterQuantizePass,
                                    RuntimeCalibrationPass)
from QuantZoo.Data.DIV2K.Data import load_div2k_dataset
from QuantZoo.Data.DIV2K.Eval import psnr, ssim
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze, report


def evaluation(graph: BaseGraph, dataloader: Iterable, method: str='psnr'):
    if method not in {'psnr', 'ssim'}: raise Exception('Evaluation method not understood.')
    executor = TorchExecutor(graph)
    ret_collector = []
    
    for lr_img, hr_img in tqdm(dataloader):
        pred = executor.forward(lr_img.cuda())[0]
        real = hr_img
        
        # post processing
        pred = convert_any_to_numpy((pred.squeeze(0) * 255).round())
        real = convert_any_to_numpy((real.squeeze(0) * 255).round())
        
        if method == 'psnr': sample_ret = psnr(img1=real, img2=pred, input_order='CHW')
        else: sample_ret = ssim(img1=real, img2=pred, input_order='CHW')
        ret_collector.append(sample_ret)

    return sum(ret_collector) / len(ret_collector)

calib_loader = load_div2k_dataset(
    lr_folder = TRAIN_LR_DIR, 
    hr_folder = TRAIN_HR_DIR)

test_loader = load_div2k_dataset(
    lr_folder = VALID_LR_DIR,
    hr_folder = VALID_HR_DIR)

with ENABLE_CUDA_KERNEL():
    for config in CONFIGS:
        model = config['Model']
        monitoring_vars = config['Output']

        print(f"Ready to run quant benchmark on {model}")
        graph = load_onnx_graph(onnx_import_file=os.path.join(MODEL_DIR, model + '.onnx'))

        quantizer = MyInt8Quantizer(graph=graph, sym=SYMMETRICAL, per_channel=PER_CHANNEL,
                                    power_of_2=POWER_OF_2, num_of_bits=BIT_WIDTH)
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
        executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 480, 640]).cuda())

        pipeline.optimize(
            graph=graph, dataloader=calib_loader, verbose=True,
            calib_steps=32, collate_fn=lambda x: x[0].cuda(), 
            executor=executor)

        result = evaluation(graph=graph, dataloader=test_loader, method='psnr')
        print(f'Model Performance on DIV2K PSNR: {result}')

        # error analyze
        performance = error_analyze(
            graph=graph,
            outputs=monitoring_vars,
            dataloader=test_loader, 
            collate_fn=lambda x: x[0].cuda(),
            verbose=True
        )