# This example shows how to make a dynamic-shape network
# dynamic shape is only supported by onnx

# first of all, load your model from anywhere
from ppq import *
from ppq.api import *

YOU_WANT_TO_QUANTIZE_IT = True

ir = load_onnx_graph('onnx model path')
input_shape = [1, 3, 224, 224]
samples     = [torch.zeros(size=input_shape).cuda()]

if YOU_WANT_TO_QUANTIZE_IT:
    ir = dispatch_graph(ir, platform=TargetPlatform.NCNN_INT8, 
                        setting=QuantizationSettingFactory.ncnn_setting())

    ir = quantize_native_model(
        model=ir, calib_dataloader=samples, calib_steps=32, 
        input_shape=input_shape, setting=QuantizationSettingFactory.ncnn_setting())

# You are supposed to set dynamic shape input/output variable just before export.
# Get variable instance from ir by its name, set shape attribute as your wish.
var = ir.variables['input variable name']
var.shape = ['Batch', 3, 'Width', 'Height']
# text, None, int are both acceptable here.

export_ppq_graph(graph=ir, platform=TargetPlatform.NCNN_INT8, 
                 graph_save_to='onnx model save to', config_save_to='config save to')