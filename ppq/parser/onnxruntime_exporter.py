from ast import Continue
from typing import Dict, List

import onnx
import torch
from onnx import helper
from ppq.core import (EXPORT_DEVICE_SWITCHER, PPQ_NAME, DataType,
                      OperationMeta, QuantizationProperty, QuantizationStates,
                      TensorMeta, TensorQuantizationConfig,
                      convert_any_to_torch_tensor)
from ppq.IR import (BaseGraph, Operation, QuantableOperation,
                    QuantableVariable, Variable)
from ppq.IR.base.command import GraphCommand, GraphCommandType
from ppq.IR.morph import GraphDeviceSwitcher, GraphFormatter
from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt
from ppq.utils.round import ppq_tensor_round

from .onnx_exporter import OnnxExporter


class ONNXRUNTIMExporter(OnnxExporter):
    """ONNXRUNTIME int8 QDQ format exporter, no further actions should be applied to the graph because we will modify the graph
    in-place, and the modified graph can't be executed. We remove Clip and Relu ops(fuse into computing op) here when asym quantization
    for activation is applied, and following the official implementation, when an variable has multiple outputs, we assume the same
    quantization scales and offset. For parameters, we pre-quantize the value and only insert DequantizeLinear op, both per-layer/per-channel
    and asym/sym quantizations are supported for export, the exported onnx model is tested to align with PPQ monitor when CUDAExecutionProvider
    is applied in onnxruntime-gpu >= 1.8.1, i.e., to run the model correctly if you have gpu and onnxruntime-gpu version installed
    
    X     W      b             X        quant(W)   quant(b)
    \     |     /               \         |          /
     \    |    /                quant    dequant  dequant
        Conv             ->       \       |        /
          |                      dequant  |       / 
          |                         \     |      / 
                                         Conv
                                          |
                                        quant
                                          |
                                        dequant
                                          |

    ```
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(file_path, sess_options, providers=['CUDAExecutionProvider'])
    res = sess.run(None, {sess.get_inputs()[0].name : dummy_input.cpu().numpy()})
    ```
    """

    def __init__(self, removed_activation_types: List[str] = ['Relu', 'Clip']) -> None:
        super().__init__()
        self._num_of_generated_ops  = 1
        self._num_of_generated_vars = 0
        self.removed_activation_types = removed_activation_types
    
    @ property
    def num_of_generated_vars(self):
        self._num_of_generated_vars += 1
        return self._num_of_generated_vars

    @ property
    def num_of_generated_ops(self):
        self._num_of_generated_ops += 1
        return self._num_of_generated_ops

    def insert_quant_dequant_on_variable(
        self, graph: BaseGraph, var: QuantableVariable, 
        config: TensorQuantizationConfig, related_op: Operation) -> None:
        """insert quant and dequant op on common quantable variables, by default a pair of quant
        and dequant ops will be inserted on var, i.e., all destinations of original var will be
        replaced by output of dequant op, but you can also insert on single var--dest_op branch
        by setting single_branch=True, in this case you should give the desired dest_op as the
        destination op of dequant op 

        Args:
            graph (BaseGraph): PPQ IR graph.
            var (Variable): quantable variables, parameters assumed.
            config (TensorQuantizationConfig, optional): quantization config. Defaults to None.
            single_branch (bool, optional): whether to insert on var(replace all destinations)
                                            or insert on just single branch. Defaults to False.
            dest_op (Operation, optional): shouldn't be None when single_branch is True. Defaults to None.
        """
        meta = var.meta

        # For SYMMETRICAL Quantization:
        #   offset - int8, quant value - int8
        # For ASYMMETRICAL Quantization:
        #   offset - uint8, quant value - uint8
        # For Bias Quantization:
        #   offset - int32, quant value - int32
        offset_dtype, value_dtype = torch.int8, torch.int8 
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL): 
            offset_dtype = torch.uint8
            value_dtype  = torch.int8
        if config.num_of_bits > 16: 
            offset_dtype = torch.int32
            value_dtype  = torch.int32
        
        scale  = convert_any_to_torch_tensor(config.scale, dtype=torch.float32)
        offset = ppq_tensor_round(config.offset).type(offset_dtype)

        qt_svar = Variable(name=f'scale_{self.num_of_generated_vars}', value=scale, is_parameter=True)
        qt_zvar = Variable(name=f'zeropoint_{self.num_of_generated_vars}', value=offset, is_parameter=True)
        dq_svar = Variable(name=f'scale_{self.num_of_generated_vars}', value=scale, is_parameter=True)
        dq_zvar = Variable(name=f'zeropoint_{self.num_of_generated_vars}', value=offset, is_parameter=True)

        qt_op = Operation(name=f'Quantize_{self.num_of_generated_ops}', op_type='QuantizeLinear', attributes={})
        dq_op = Operation(name=f'Dequantize_{self.num_of_generated_ops}', op_type='DequantizeLinear', attributes={})

        if var in related_op.inputs:
            graph.insert_op_between_var_and_op(dq_op, up_var=var, down_op=related_op)
            graph.insert_op_between_var_and_op(qt_op, up_var=var, down_op=dq_op)
        else:
            graph.insert_op_on_var(dq_op, var=var.name)
            graph.insert_op_on_var(qt_op, var=var.name)

        qt_op.inputs.extend([qt_svar, qt_zvar])
        dq_op.inputs.extend([dq_svar, dq_zvar])

        qt_svar.dest_ops.append(qt_op)
        qt_zvar.dest_ops.append(qt_op)
        dq_svar.dest_ops.append(dq_op)
        dq_zvar.dest_ops.append(dq_op)

        graph.append_variable(qt_svar)
        graph.append_variable(qt_zvar)
        graph.append_variable(dq_svar)
        graph.append_variable(dq_zvar)

        qt_meta = OperationMeta(
            input_metas    = [TensorMeta(dtype=DataType.FP32, shape=meta.shape), 
                              TensorMeta(dtype=DataType.FP32, shape=config.scale.shape), 
                              TensorMeta(dtype=DataType.convert_from_torch(offset_dtype), shape=config.offset.shape)],
            output_metas   = [TensorMeta(dtype=DataType.convert_from_torch(value_dtype), shape=meta.shape)],
            operation_name = qt_op.name, operation_type=qt_op.type, executing_order=-1)
        dq_meta = OperationMeta(
            input_metas    = [TensorMeta(dtype=DataType.convert_from_torch(value_dtype), shape=meta.shape), 
                              TensorMeta(dtype=DataType.FP32, shape=config.scale.shape), 
                              TensorMeta(dtype=DataType.convert_from_torch(offset_dtype), shape=config.offset.shape)],
            output_metas   = [TensorMeta(dtype=DataType.FP32, shape=meta.shape)],
            operation_name = dq_op.name, operation_type=dq_op.type, executing_order=-1)

        qt_op.meta_data = qt_meta
        dq_op.meta_data = dq_meta

        if var.is_parameter:
            var.value = PPQLinearQuant_toInt(var.value, config=config)

    def remove_activation_ops(self, graph: BaseGraph, activation_ops: List[Operation]) -> None:
        """
        For Asymmetric Quantization Policy, Activations like Relu & Clip can be
            removed from your network safely. Their function can be replaced by
            quant & dequant operations.
        
        So to say those activation is unnecessary for Asymmetric quantized network.

        Args:
            graph (BaseGraph): Processing Graph
            activation_ops (List[Operation]): Removing activations.
        """
        
        # Activation op can only be relu and clip,
        # so it is safe to access op.inputs[0], op.outputs[0] as their input and output.
        for op in activation_ops:
            if not isinstance(op, QuantableOperation): continue
            if len(graph.get_upstream_operations(op)) == 0: Continue

            upstream_op = graph.get_upstream_operations(op)[0]
            if not isinstance(upstream_op, QuantableOperation): continue
            input_var, input_cfg = op.inputs[0], op.config.input_quantization_config[0]
            if not input_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL): continue

            # PATCH 20220304 Removing graph output op might cause error.
            if op.outputs[0].name in graph.outputs:
                graph.outputs.pop(op.outputs[0].name)
                graph.outputs[input_var.name] = input_var

            if op.type == 'Clip':
                for var in op.inputs[1:]:
                    var.dest_ops.clear()
                    graph.delete_variable(var.name)
                while len(op.inputs) > 1:
                    op.inputs.pop()

            graph.remove_operation(op)
        formater = GraphFormatter(graph)
        formater(GraphCommand(GraphCommandType.DELETE_ISOLATED))

    def remove_duplicated_quant_op(self, graph: BaseGraph):
        """
        Some time there will be more than 1 quant operation inserted with a single variable.
        This function will remove duplicated quant operation from variable if it is possible.
        
        If inserted quant operations do not share a same zeropoint and scale,
        Then there is no way to remove any one of them.

        Args:
            graph (BaseGraph): Processing Graph

        Returns:
            _type_: Processed Graph
        """
        interested_pairs = []
        for qt_op in graph.operations.values():
            if qt_op.type == 'QuantizeLinear':
                if len(graph.get_upstream_operations(qt_op)) != 1: continue
                if graph.get_upstream_operations(qt_op)[0].type != 'DequantizeLinear': continue
                interested_pairs.append((qt_op, graph.get_upstream_operations(qt_op)[0]))

        mark_to_remove = set()
        for qt_op, dq_op in interested_pairs:
            assert isinstance(dq_op, Operation)
            assert isinstance(qt_op, Operation)

            scale_diff     = torch.max(torch.abs(dq_op.inputs[1].value - qt_op.inputs[1].value)).item()
            zeropoint_diff = torch.max(torch.abs(dq_op.inputs[2].value - qt_op.inputs[2].value)).item()
            if scale_diff < 1e-5 and zeropoint_diff < 0.5: # zero point 是整数，所以只要误差小于1就行了。
                # mark quant operation and its following operation(suppose to be another dequantization op)
                mark_to_remove.add(qt_op)
                assert len(graph.get_downstream_operations(qt_op)) == 1, 'Oops, that should never happen.'
                mark_to_remove.add(graph.get_downstream_operations(qt_op)[0])
        
        for op in mark_to_remove: graph.remove_operation(op)
        return graph

    @ property
    def required_opsets(self) -> Dict[str, int]:
        extra_domain_versions = [
            ("ai.onnx", 13),
        ]
        return dict(extra_domain_versions)

    def convert_operation_from_opset11_to_opset13(self, graph:BaseGraph) -> None:
        """
        Convert your network from opset 11 standard towards opset 13
        With Onnx defination, per-channel quant operation requires opset 13.

        Args:
            graph (BaseGraph): Processing graph.
        """
        # this func transform representation of certain op from opset 11 to 13
        for op in graph.operations.values():
            if op.type == 'ReduceSum' or op.type == 'Squeeze' or op.type == 'Unsqueeze':
                axes = convert_any_to_torch_tensor(op.attributes.pop('axes'), dtype=torch.int64)
                var = Variable(name=op.name+'_axes', value=axes, is_parameter=True, dest_ops=[op])
                graph.append_variable(var)
                op.inputs.append(var)
                op.meta_data.input_metas.append(TensorMeta.parsing_from_torch_tensor(var.value, var.name))

            elif op.type == 'Split':
                split = convert_any_to_torch_tensor(op.attributes.pop('split'), dtype=torch.int64)
                var = Variable(name=op.name+'_axes', value=split, is_parameter=True, dest_ops=[op])
                graph.append_variable(var)
                op.inputs.append(var)
                op.meta_data.input_metas.append(TensorMeta.parsing_from_torch_tensor(var.value, var.name))

    def prepare_graph(self, graph: BaseGraph) -> BaseGraph:
        """
        Prepare your graph for exporting.
        
        There are many works to do with your graph:

            1. Insert Quant and Dequant operation within your graph.
            
            2. Quantize all parameters of your graph, convert them to int8.
            
            3. Remove all unnecessary activations.

        Args:
            graph (BaseGraph): Processing Graph

        Returns:
            BaseGraph: Processed Graph
        """
        self.convert_operation_from_opset11_to_opset13(graph)

        # remove switchers.
        if not EXPORT_DEVICE_SWITCHER:
            processer = GraphDeviceSwitcher(graph)
            processer.remove_switcher()

        # mark quantable variables
        for op in [op for op in graph.operations.values()]:
            if not isinstance(op, QuantableOperation): continue
            # collect quantable vars, where we need to insert quant and dequant op
            for config, var in op.config_with_variable:
                if not QuantizationStates.is_activated(config.state): continue
                if var.is_parameter: 
                    assert len(var.dest_ops) == 1, (
                    f'Can not export variable {var.name}, cause it has more than 1 destination operations. '
                    'PPQ require all parameters to have only 1 destination operation.')

                    if QuantizationStates.is_activated(config.state):
                        self.insert_quant_dequant_on_variable(
                            graph=graph, var=var, config=config, related_op=op)

                elif QuantizationStates.is_activated(config.state): 
                    self.insert_quant_dequant_on_variable(
                        graph=graph, var=var, config=config, related_op=op)

        removed_activations = []
        for op in graph.operations.values(): 
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'Relu', 'Clip'}:
                # Only ASYMMETRICAL quantized activations can be safely removed.
                if op.config.input_quantization_config[0].policy.has_property(QuantizationProperty.ASYMMETRICAL):
                    removed_activations.append(op)

        # remove useless activation.
        self.remove_activation_ops(graph, removed_activations)

        return self.remove_duplicated_quant_op(graph)

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None) -> None:
        graph = self.prepare_graph(graph)
        
        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            super().export_quantization_config(config_path, graph)
        
        name = graph._name
        if not name: name = 'PPL Quantization Tool - Onnx Export'

        # Ready to export onnx graph defination.
        _inputs, _outputs, _initilizers, _nodes = [], [], [], []
        for operation in graph.topological_sort():
            _nodes.append(super().export_operation(operation))

        for variable in graph.variables.values():
            tensor_proto = super().export_var(variable)
            if variable.name in graph.inputs:
                _inputs.append(tensor_proto)
            if variable.name in graph.outputs:
                _outputs.append(tensor_proto)
            if variable.is_parameter:
                _initilizers.append(tensor_proto)

        graph_def = helper.make_graph(
            name=name,
            nodes=_nodes,
            inputs=_inputs,
            outputs=_outputs,
            initializer=_initilizers,
        )
        extra_opsets = self.required_opsets

        if 'opsets' in graph._detail:
            opsets = []
            for opset in graph._detail['opsets']:
                if opset['domain'] in extra_opsets or opset['domain'] == '':
                    continue
                op = onnx.OperatorSetIdProto()
                op.domain = opset['domain']
                op.version = opset['version']
                opsets.append(op)

        for key, value in extra_opsets.items():
            op = onnx.OperatorSetIdProto()
            op.domain = key
            op.version = value
            opsets.append(op)

        onnx_model = helper.make_model(
            graph_def, producer_name=PPQ_NAME, opset_imports=opsets)
        onnx_model.ir_version = 7
        # onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, file_path)
