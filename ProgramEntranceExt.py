""" 这是一个 PPQ 量化的
"""

import ppq
from ppq import Operation, OperationQuantizationConfig, BaseGraph, QuantizationPolicy, QuantizationProperty, RoundingPolicy, QuantizationStates

class Int8Quantizer(ppq.BaseQuantizer):
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        op_quant_config = self.create_default_quant_config(
            op=operation, num_of_bits=8, 
            policy=QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.PER_TENSOR
            ), quant_min=-128, quant_max=127, 
            observer_algorithm='percentile',
            rounding=RoundingPolicy.ROUND_HALF_EVEN
        )
        
        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            # 这些算子的 weight 通常是 per-channel 量化的
            W_TQC = op_quant_config.input_quantization_config[1]
            W_TQC.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.PER_CHANNEL
            )
            W_TQC.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)

            # 这些算子的 bias 通常是 32 位量化，且 scale = input scale * weight scale
            if operation.num_of_input == 3:
                B_TQC = op_quant_config.input_quantization_config[2]
                B_TQC.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.PER_CHANNEL
                )
                B_TQC.num_of_bits = 32
                B_TQC.quant_min   = -(1 << 30)
                B_TQC.quant_max   = (1 << 30)
                
                # 当 Bias 的初始状态为 PASSIVE_INIT，则 PPQ 认为当前 Bias 被动量化，需要与 input, weight 共享 scale
                # 当 Bias 的初始状态为 INIT，则 PPQ 认为当前 Bias 独立量化，会单独计算 scale
                # 当 Bias 的初始状态为 FP32，则 PPQ 认为当前 Bias 不量化
                B_TQC.state       = QuantizationStates.PASSIVE_INIT
        
        if operation.type in {'LayerNormalization'}:
            # Layer Normalization 的参数不量化，它们在推理时使用 FP32 或者 FP16
            # 但它的输入输出将被量化
            for TQC in op_quant_config.input_quantization_config[1: ]:
                TQC.state = QuantizationStates.FP32

    def quant_operation_types(self) -> set:
        # 告知 PPQ 那些算子需要量化，区分大小写
        return {''}
    
    def activation_fusion_types(self) -> set:
        # 告知 PPQ 那些激活函数需要融合，区分大小写
        return {'Relu', 'Swish', 'Gelu', 'LeakyRelu', 'Sigmoid', 'Tanh'}

class FP8Quantizer(ppq.BaseQuantizer):
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_config = self.create_default_quant_config(op=operation, num_of_bits=8,)

class MyOptimPass(ppq.QuantizationOptimizationPass):
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        return super().optimize(graph, **kwargs)

def MyOpForward()

    