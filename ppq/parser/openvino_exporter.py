from .onnxruntime_exporter import ONNXRUNTIMExporter
import torch
from ppq.core import (QuantizationProperty,
                      TensorQuantizationConfig, 
                      convert_any_to_torch_tensor)
from ppq.IR import (BaseGraph, Operation,
                    Variable)
from ppq.utils.round import ppq_tensor_round


class OpenvinoExporter(ONNXRUNTIMExporter):
    """Openvino 喜欢所有 QuantizeLinear 上面都带 axis 属性，所以这里需要单独处理一下"""
    
    def insert_quantize_node(
        self, graph: BaseGraph, 
        var: Variable, config: TensorQuantizationConfig, 
        op: Operation) -> Operation:
        """
        Insert a Quantize Node on given variable, according to given TensorQuantizationConfig.
        There is two basic type of Quantize Node: QuantizeLinear and QuantizeFloating.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            # Following code will export Linear Quantization Config
            # That is for FP32 -> INT
            offset_dtype, value_type = self.infer_qtype(config)
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

            created = graph.create_operation(op_type='QuantizeLinear', attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = 0
            
            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')

            graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

            created.outputs[0].dtype = value_type
            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape
            return created
        
        elif config.policy.has_property(QuantizationProperty.FLOATING):
            # Following code will export Linear Quantization Config
            # That is for FP32 -> FP8
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = convert_any_to_torch_tensor(config.offset.clone(), dtype=torch.float32)

            created = graph.create_operation(
                op_type='QuantizeFloating', 
                attributes={
                    'min': config.quant_min, 
                    'max': config.quant_max, 
                    'exponent': config.exponent_bits, 
                    'mantissa': config.mantissa_bits})

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = None
            
            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')

            graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape
            return created
        
        else:
            raise TypeError(
                f'PPQ Can not export quantization information with variable {var.name}, '
                'Unexpected Quantization property.')

    def insert_dequantize_node(
        self, graph: BaseGraph, 
        var: Variable, config: TensorQuantizationConfig, 
        op: Operation) -> Operation:
        """
        Insert a DeQuantize Node on given variable, according to given TensorQuantizationConfig.
        There is two basic type of DeQuantize Node: DeQuantizeLinear and DeQuantizeFloating.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            offset_dtype, value_type = self.infer_qtype(config)
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)
            
            created = graph.create_operation(op_type='DequantizeLinear', attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = 0
            
            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')
            
            graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

            created.inputs[0].dtype = value_type
            created.inputs[0].shape = var.shape
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = torch.float32

            return created

        elif config.policy.has_property(QuantizationProperty.FLOATING):
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = convert_any_to_torch_tensor(config.offset.clone(), dtype=torch.float32)

            created = graph.create_operation(
                op_type='DequantizeFloating', 
                attributes={
                    'min': config.quant_min, 
                    'max': config.quant_max, 
                    'exponent': config.exponent_bits, 
                    'mantissa': config.mantissa_bits})

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes['axis'] = config.channel_axis
            else: created.attributes['axis'] = None
            
            if var in op.inputs:  graph.insert_op_before(A=created, B=op, input_idx=op.inputs.index(var))
            elif var in op.outputs: graph.insert_op_after(A=created, B=op, output_idx=op.outputs.index(var))
            else: raise ValueError(f'Unexpected Error in Exporting Op {op.name}({op.type}).')

            graph.create_variable(name=None, value=scale, is_parameter=True, dest_ops=[created])
            graph.create_variable(name=None, value=offset, is_parameter=True, dest_ops=[created])

            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape

            return created

        else:

            raise TypeError(
                f'PPQ Can not export quantization information with variable {var.name}, '
                'Unexpected Quantization property.')
