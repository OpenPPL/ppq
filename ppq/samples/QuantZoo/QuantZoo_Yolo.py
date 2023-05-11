# Test Quantization System Performance on Detection Models with Coco Dataset

# Quantizer Configuration
SYMMETRICAL = True
PER_CHANNEL = False
POWER_OF_2  = True
BIT_WIDTH   = 8
FP8         = False

# Should contains model file(.onnx)
MODEL_DIR = 'QuantZoo/Model/yolo'

# Should contains Calib & Test Img Folder
CALIB_DIR      = 'QuantZoo/Data/Coco/Calib'
TEST_DIR       = 'QuantZoo/Data/Coco/Test'
CALIB_ANN_FILE = 'QuantZoo/Data/Coco/Calib/DetectionAnnotation.json'
TEST_ANN_FILE  = 'QuantZoo/Data/Coco/Test/DetectionAnnotation.json'
PRED_ANN_FILE  = 'QuantZoo/Data/Coco/Test/DetectionPrediction.json'
VALID_DIR      = '/mnt/hpc/share/wangpeiqi/COCO/val2017'
VALID_ANN_FILE = '/mnt/hpc/share/wangpeiqi/COCO/annotations/instances_val2017.json'
EVAL_MODE      = True # only for evaluation, it will slow down the system.

# calibration & test batchsize
# yolo requires batchsize = 1
BATCHSIZE   = 1

# write report to here
REPORT_DIR = 'QuantZoo/Reports'

CONFIGS = [
{
    'Model': 'yolov6p5_n',
    'Output': ['/Concat_5_output_0', '/Concat_4_output_0'],
    'collate_fn': lambda x: x[0].cuda() # img preprocessing function
},
{
    'Model': 'yolov6p5_t',
    'Output': ['/Concat_5_output_0', '/Concat_4_output_0'],
    'collate_fn': lambda x: x[0].cuda() # img preprocessing function
},
{
    'Model': 'yolov5s6_n',
    'Output': ['/baseModel/head_module/convs_pred.1/Conv_output_0', '/baseModel/head_module/convs_pred.2/Conv_output_0', '/baseModel/head_module/convs_pred.0/Conv_output_0'],
    'collate_fn': lambda x: x[0].cuda() # img preprocessing function
},
{
    'Model': 'yolov5s6_s',
    'Output': ['/baseModel/head_module/convs_pred.1/Conv_output_0', '/baseModel/head_module/convs_pred.2/Conv_output_0', '/baseModel/head_module/convs_pred.0/Conv_output_0'],
    'collate_fn': lambda x: x[0].cuda() # img preprocessing function
},
{
    'Model': 'yolov7p5_tiny',
    'Output': ['/Concat_4_output_0', '/Concat_5_output_0', '/Concat_6_output_0'],
    'collate_fn': lambda x: x[0].cuda() # img preprocessing function
},
{
    'Model': 'yolov7p5_l',
    'Output': ['/Concat_4_output_0', '/Concat_5_output_0', '/Concat_6_output_0'],
    'collate_fn': lambda x: x[0].cuda() # img preprocessing function
},
{
    'Model': 'yolox_s',
    'Output': ['/Concat_4_output_0', '/Concat_5_output_0', '/Concat_6_output_0'],
    'collate_fn': lambda x: x[0].cuda() * 255 # img preprocessing function
},
{
    'Model': 'yolox_tiny',
    'Output': ['/Concat_4_output_0', '/Concat_5_output_0', '/Concat_6_output_0'],
    'collate_fn': lambda x: x[0].cuda() * 255 # img preprocessing function
},
{
    'Model': 'ppyoloe_m',
    'Output': ['/Concat_4_output_0', '/Concat_5_output_0'],
    'collate_fn': lambda x: (
        x[0].cuda() * 255 - torch.tensor([103.53, 116.28, 123.675]).reshape([1, 3, 1, 1]).cuda()
    ) / 255 # img preprocessing function
},
{
    'Model': 'ppyoloe_s',
    'Output': ['/Concat_4_output_0', '/Concat_5_output_0'],
    'collate_fn': lambda x: (
        x[0].cuda() * 255 - torch.tensor([103.53, 116.28, 123.675]).reshape([1, 3, 1, 1]).cuda()
    ) / 255 # img preprocessing function
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
from ppq.IR import GraphFormatter

from QuantZoo.Data.Coco.Data import load_coco_detection_dataset
from QuantZoo.Data.Coco.Eval import evaluate_ppq_module_with_coco
from QuantZoo.Quantizers import MyFP8Quantizer, MyInt8Quantizer
from QuantZoo.Util import error_analyze


calib_loader = load_coco_detection_dataset(
    data_dir=CALIB_DIR,
    batchsize=BATCHSIZE)

test_loader = load_coco_detection_dataset(
    data_dir=TEST_DIR,
    batchsize=BATCHSIZE)


with ENABLE_CUDA_KERNEL():
    for config in CONFIGS:
        model           = config['Model']
        monitoring_vars = config['Output']
        collate_fn      = config['collate_fn']
        
        print(f"Ready to run quant benchmark on {model}")
        graph = load_onnx_graph(onnx_import_file=os.path.join(MODEL_DIR, model + '.onnx'))

        # if EVAL_MODE == False, truncate graph
        if EVAL_MODE == False:
            graph.outputs.clear()
            editor = GraphFormatter(graph)
            for var in monitoring_vars:
                graph.mark_variable_as_graph_output(graph.variables[var])
            editor.delete_isolated()
        else:
            if model in {'rtmdet_s', 'rtmdet_tiny'}:
                # 别问我为什么，只是因为大家的输出都叫 '/Split_output_1'，但是 rtmdet 的不叫这个
                # rename variable 'onnx::Shape_1182' to '/Split_output_1'
                graph.variables['/Split_output_1'] = graph.variables['onnx::Shape_1182']
                graph.variables['/Split_output_1']._name = '/Split_output_1'

            editor = GraphFormatter(graph)
            graph.outputs.pop('scores')
            graph.outputs.pop('num_dets')
            graph.mark_variable_as_graph_output(graph.variables['/Split_output_1'])
            editor.delete_isolated()

        quantizer = MyInt8Quantizer(graph=graph, sym=SYMMETRICAL, 
                                    per_channel=PER_CHANNEL, power_of_2=POWER_OF_2, 
                                    num_of_bits=BIT_WIDTH)
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
        executor.tracing_operation_meta(torch.zeros(size=[BATCHSIZE, 3, 640, 640]).cuda())

        pipeline.optimize(
            graph=graph, dataloader=calib_loader, verbose=True,
            calib_steps=32, collate_fn=collate_fn, 
            executor=executor)

        if EVAL_MODE:
            # evaluation 好像 batchsize != 1 会错
            evaluate_ppq_module_with_coco(
                ann_file=TEST_ANN_FILE,
                output_file=PRED_ANN_FILE,
                executor=executor, 
                dataloader=test_loader,
                collate_fn=collate_fn)

        # error analyze
        performance = error_analyze(
            graph=graph,
            outputs=monitoring_vars,
            dataloader=test_loader, 
            collate_fn=collate_fn,
            verbose=True
        )
