"""这个脚本将教会你如何使用 PPQ 量化自定义算子"""

import torch
from ppq import *
from ppq.api import *
from ppq.quantization.quantizer import TensorRTQuantizer

B = 1
T = 64
MODEL_PATH = 'models\encoder_ln.onnx'

def generate_samples(num_of_samples: int = 32):
    """生成样本数据，把这个函数改成真实数据读入就可以完成量化了
    这个语音数据量很小 我建议你把整个数据集直接全部送上CUDA
    """
    sample = {
        'speech': torch.rand(size=[B, T, 80]).float().cuda(),
        'speech_lengths': torch.ones(size=[B]).int().cuda()}
    samples = [sample for _ in range(num_of_samples)]
    return samples
SAMPLES = generate_samples()

# 定义一个自己的量化器，定制量化行为，继承于 TensorRTQuantizer 量化器
class MyTensorRTQuantizer(TensorRTQuantizer):
    @ property
    def quant_operation_types(self) -> set:
        """覆盖 quant_operation_types  自定义需要量化的算子"""
        return {'LayerNormPlugin'}

    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        config = super().init_quantize_config(operation=operation)
        """针对 LayerNormPlugin 生成量化配置信息"""
        if operation.type == 'LayerNormPlugin': 
            wconfig = config.input_quantization_config[1] # weight config
            bconfig = config.input_quantization_config[2]
            
            wconfig.policy = QuantizationPolicy(    # weight 做 Per channel 量化
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.PER_TENSOR)
            bconfig.state = QuantizationStates.FP32 # bias 不量化
            '''
            # 将 weight config 升级为 ChannelwiseTensorQuantizationConfig
            config.input_quantization_config[1] = ( 
                ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                    convert_from = wconfig,
                    offsets = None, scales = None, 
                    channel_axis = 0))
            '''
            config.input_quantization_config[1].observer_algorithm = 'Minmax'
        return config

def layernorm_forward(op: Operation, values: List[torch.Tensor], ctx = None, **kwargs):
    """自定义算子的前向传播函数"""
    return values[0]

register_operation_handler(
    handler=layernorm_forward, 
    operation_type='LayerNormPlugin', 
    platform=TargetPlatform.TRT_INT8)

register_network_quantizer(
    quantizer=MyTensorRTQuantizer,
    platform=TargetPlatform.TRT_INT8)

with ENABLE_CUDA_KERNEL():
    QS = QuantizationSettingFactory.trt_setting()
    ir = load_onnx_graph(onnx_import_file=MODEL_PATH)
    # 默认调度失效，直接手动调度所有 LayerNormPlugin 送上量化区
    for op in ir.operations.values():
        if op.type == 'LayerNormPlugin':
            QS.dispatching_table.append(operation=op.name, platform=TargetPlatform.TRT_INT8)
    
    qir = quantize_native_model(
        model=ir, calib_dataloader=SAMPLES, calib_steps=32, 
        input_shape=None, inputs=SAMPLES[0], 
        platform=TargetPlatform.TRT_INT8, setting=QS)

    graphwise_error_analyse(
        graph=qir, running_device='cuda', 
        dataloader=SAMPLES)

    export_ppq_graph(
        graph=qir, platform=TargetPlatform.ONNXRUNTIME, 
        graph_save_to='quantized.onnx')
