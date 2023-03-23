# Test Quantization System Performace on OCR Models with IC15 Dataset
#
#   1. How to use: 
#      Run this script with python directly.
#

# Quantizer Configuration
SYMMETRICAL = True
PERCHANNEL  = True
POWER_OF_2  = False
FP8         = False
BIT_WIDTH   = 8

# Should contains model file(.onnx)
MODEL_DIR = 'QuantZoo/Model/ocr'

# Should contains Calib & Test Img Folder
CALIB_DIR   = 'QuantZoo/Data/IC15'
CALIB_LABEL = 'QuantZoo/Data/IC15/rec_gt_train.txt'
TEST_DIR    = 'QuantZoo/Data/IC15'
TEST_LABEL  = 'QuantZoo/Data/IC15/rec_gt_test.txt'
CHAR_DIR    = 'QuantZoo/Data/IC15/ic15_dict.txt'

# calibration & test batchsize
BATCHSIZE = 32

# write report to here
REPORT_DIR = 'QuantZoo/Reports'

CONFIGS = [
    
{
    'Model': 'en_PP-OCRv3_rec_infer',
    'Output': ['swish_13.tmp_0'],
    'Dictionary': 'en_dict.txt',
    'Reshape': [3, 48, 320],
    'Language': 'en',
},
{
    'Model': 'en_number_mobile_v2.0_rec_infer',
    'Output': ['save_infer_model/scale_0.tmp_1'],
    'Dictionary': 'en_dict.txt',
    'Reshape': [3, 32, 320],
    'Language': 'en',
},
{
    'Model': 'ch_PP-OCRv2_rec_infer',
    'Output': ['p2o.LSTM.5'],
    'Dictionary': 'ppocr_keys_v1.txt',
    'Reshape': [3, 32, 320],
    'Language': 'ch',
},
{
    'Model': 'ch_PP-OCRv3_rec_infer',
    'Output': ['swish_27.tmp_0'],
    'Dictionary': 'ppocr_keys_v1.txt',
    'Reshape': [3, 48, 320],
    'Language': 'ch',
},
{
    'Model': 'ch_ppocr_mobile_v2.0_rec_infer',
    'Output': ['p2o.LSTM.5'],
    'Dictionary': 'ppocr_keys_v1.txt',
    'Reshape': [3, 32, 320],
    'Language': 'ch',
},
{
    'Model': 'ch_ppocr_server_v2.0_rec_infer',
    'Output': ['p2o.LSTM.5'],
    'Dictionary': 'ppocr_keys_v1.txt',
    'Reshape': [3, 32, 320],
    'Language': 'ch',
},
]

import os

import torch

import ppq.lib as PFL
from ppq import convert_any_to_torch_tensor
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.quantization.optim import (LayerwiseEqualizationPass,
                                    LearnedStepSizePass, ParameterQuantizePass,
                                    RuntimeCalibrationPass)
from QuantZoo.Data.IC15.Data import IC15_PaddleOCR
from QuantZoo.Data.IC15.Eval import evaluate_ppq_module_with_ic15
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze, report


with ENABLE_CUDA_KERNEL():
    for config in CONFIGS:
        model = config['Model']
        monitoring_vars = config['Output']
        dictionary = config['Dictionary']
        shape      = config['Reshape']
        chinese    = config['Language'] == 'ch'

        calib_loader = IC15_PaddleOCR(
            images_path=CALIB_DIR, 
            label_path=CALIB_LABEL,
            input_shape=shape, 
            is_chinese_version=chinese).dataloader(
                batchsize=BATCHSIZE, shuffle=False)

        test_loader = IC15_PaddleOCR(
            images_path=TEST_DIR, 
            label_path=TEST_LABEL,
            input_shape=shape,
            is_chinese_version=chinese).dataloader(
                batchsize=BATCHSIZE, shuffle=False)

        print(f"Ready to run quant benchmark on {model}")
        graph = load_onnx_graph(onnx_import_file=os.path.join(MODEL_DIR, model + '.onnx'))

        quantizer = MyInt8Quantizer(graph=graph, sym=SYMMETRICAL, power_of_2=POWER_OF_2, 
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
            # LearnedStepSizePass(steps=500, collecting_device='cuda')
        ])

        # call pipeline.
        executor = TorchExecutor(graph=graph)
        executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 32, 100]).cuda())

        pipeline.optimize(
            graph=graph, dataloader=calib_loader, verbose=True,
            calib_steps=32, collate_fn=lambda x: x[0].to('cuda'), executor=executor)

        acc = evaluate_ppq_module_with_ic15(
            executor=executor, character_dict_path=os.path.join(MODEL_DIR, dictionary),
            dataloader=test_loader, collate_fn=lambda x: convert_any_to_torch_tensor(x).cuda())
        print(f'Model Performace on IC15: {acc * 100 :.4f}%')

        # error analyze
        performance = error_analyze(
            graph=graph,
            outputs=monitoring_vars,
            dataloader=test_loader, 
            collate_fn=lambda x: x[0].to('cuda'),
            verbose=True
        )