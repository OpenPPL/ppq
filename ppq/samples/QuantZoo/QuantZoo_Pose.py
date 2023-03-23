# Test Quantization System Performace on MMPose Models with Coco 2017 Dataset
#
#   1. How to use: 
#      Run this script with python directly.
#

# Quantizer Configuration
SYMMETRICAL = True
POWER_OF_2  = False
PERCHANNEL  = True
BIT_WIDTH   = 8
FP8         = False

# Should contains model file(.onnx)
MODEL_DIR = 'QuantZoo/Model/mmpose'

# Should contains Calib & Test Img Folder
CALIB_DIR = 'QuantZoo/Data/Pose/Calib'
TEST_DIR  = 'QuantZoo/Data/Pose/Test'

# calibration & test batchsize
BATCHSIZE = 1

# write report to here
REPORT_DIR = 'QuantZoo/Reports'

CONFIGS = [

{
    'Model': 'deeppose_res50_coco_256x192',
    'Output': ['/backbone/layer4/layer4.2/relu_2/Relu_output_0']
},

{
    'Model': 'hrnet_w32',
    'Output': ['onnx::Conv_2946']
},

{
    'Model': 'simcc-mobilenet_v2',
    'Output': ['537']
},

{
    'Model': 'rtmpose-s',
    'Output': ['onnx::Conv_439']
},

]

import os

import torch

import ppq.lib as PFL
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph, SearchableGraph
from ppq.quantization.optim import (LayerwiseEqualizationPass,
                                    LearnedStepSizePass, ParameterQuantizePass,
                                    RuntimeCalibrationPass)
from QuantZoo.Data.Pose.Data import load_coco_keypoint_dataset
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze, report

calib_dataloader = load_coco_keypoint_dataset(data_dir=CALIB_DIR)
test_dataloader = load_coco_keypoint_dataset(data_dir=TEST_DIR)


def output_ops(graph: BaseGraph, output_vars: list):
    """
    Return all ops that under the given output variable,
    We perfer to dequantize those ops.
    """
    serach_engine = SearchableGraph(graph)
    source_ops = [graph.variables[var].source_op for var in output_vars]
    return serach_engine.opset_matching(
        sp_expr=lambda x: x in source_ops,
        rp_expr=lambda x, y: True,
        ep_expr=None, direction='down',
    )


with ENABLE_CUDA_KERNEL():
    for config in CONFIGS:
        model = config['Model']
        monitoring_vars = config['Output']

        print(f"Ready to run quant benchmark on {model}")
        graph = load_onnx_graph(onnx_import_file=os.path.join(MODEL_DIR, model + '.onnx'))
        quantizer = MyInt8Quantizer(
            graph=graph, sym=SYMMETRICAL, power_of_2=POWER_OF_2, 
            num_of_bits=BIT_WIDTH, per_channel=PERCHANNEL)
        if FP8: quantizer = MyFP8Quantizer(graph=graph)

        # convert op to quantable-op
        non_quantable_ops = output_ops(graph=graph, output_vars=monitoring_vars)
        for name, op in graph.operations.items():
            if op.type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm'} and op not in non_quantable_ops:
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
        executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 256, 192]).cuda())

        pipeline.optimize(
            graph=graph, dataloader=calib_dataloader, verbose=True,
            calib_steps=32, collate_fn=lambda x: x[0].to('cuda'),
            executor=executor)

        # error analyze
        performance = error_analyze(
            graph=graph,
            outputs=monitoring_vars,
            dataloader=test_dataloader, 
            collate_fn=lambda x: x[0].cuda(),
            verbose=True
        )