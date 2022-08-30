from typing import Dict, List

import onnx
import torch
from onnx import helper
from ppq.core import (GRAPH_OPSET_ATTRIB, PPQ_CONFIG,
                      ChannelwiseTensorQuantizationConfig, DataType,
                      OperationMeta, QuantizationProperty, QuantizationStates,
                      TensorMeta, TensorQuantizationConfig,
                      convert_any_to_torch_tensor, ppq_warning)
from ppq.IR import (BaseGraph, Operation, QuantableOperation,
                    QuantableVariable, Variable)
from ppq.IR.morph import GraphDeviceSwitcher
from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt
from ppq.utils.round import ppq_tensor_round

from .onnx_exporter import OnnxExporter


class QDQHelper():
    """Helper class for processing onnx qdq format"""
    @ staticmethod
    def TQC_Exportable_Check(
        TQC: TensorQuantizationConfig, bounded_var: Variable) -> bool:
        if not TQC.can_export(): return False
        meta_check = bounded_var.meta is not None

        if TQC.num_of_bits == 8:
            if TQC.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                range_check = (TQC.quant_max <= 255 and TQC.quant_min >= 0)
            else: range_check = (TQC.quant_max <= 127 and TQC.quant_min >= -128)
        else: range_check = True

        if not range_check:
            ppq_warning(f'Is it not safe to export TQC({bounded_var.name}) to Onnx, '
                        f'INT8 value range must be [-128, 127] or [0, 255], '
                        f'however [{TQC.quant_min, TQC.quant_max}] was given.')
            return False

        if not meta_check:
            raise ValueError(f'Meta Data is missing! Graph Export Failed. '
                             f'(Check Meta For Varaible: {bounded_var.name})')
        return True


class ONNXRUNTIMExporter(OnnxExporter):
    """ONNXRUNTIME int8 QDQ format exporter, no further actions should be
    applied to the graph because we will modify the graph in-place, and the
    modified graph can't be executed. We remove Clip and Relu ops(fuse into
    computing op) here when asym quantization for activation is applied, and
    following the official implementation, when an variable has multiple
    outputs, we assume the same quantization scales and offset. For parameters,
    we pre-quantize the value and only insert DequantizeLinear op, both per-
    layer/per-channel and asym/sym quantizations are supported for export, the
    exported onnx model is tested to align with PPQ monitor when
    CUDAExecutionProvider is applied in onnxruntime-gpu >= 1.8.1, i.e., to run
    the model correctly if you have gpu and onnxruntime-gpu version installed.

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
        self.removed_activation_types = removed_activation_types

    def infer_qtype(self, config: TensorQuantizationConfig):
        offset_dtype, value_dtype = torch.int8, torch.int8
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            offset_dtype = torch.uint8
            value_dtype  = torch.uint8
        if config.num_of_bits > 16:
            offset_dtype = torch.int32
            value_dtype  = torch.int32
        return offset_dtype, value_dtype

    def insert_quant_on_variable(
        self, graph: BaseGraph, var: QuantableVariable,
        config: TensorQuantizationConfig, related_op: Operation,
        meta: TensorMeta = None) -> Operation:
        if meta is None: meta = var.meta
        offset_dtype, value_dtype = self.infer_qtype(config)
        scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
        offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

        s_var = graph.create_variable(name=None, value=scale, is_parameter=True)
        z_var = graph.create_variable(name=None, value=offset, is_parameter=True)
        created = graph.create_operation(op_type='QuantizeLinear', attributes={})

        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            assert isinstance(config, ChannelwiseTensorQuantizationConfig)
            created.attributes['axis'] = config.channel_axis

        # PATCH 20220803, OPSET 13 REQUIRES AXIS = 0
        if config.policy.has_property(QuantizationProperty.PER_TENSOR):
            created.attributes['axis'] = 0

        if related_op is not None and var in related_op.inputs:
            graph.insert_op_between_var_and_op(created, up_var=var, down_op=related_op)
        else: graph.insert_op_on_var(created, var=var.name)

        graph.create_link_with_op(variable=s_var, upstream_op=None, downstream_op=created)
        graph.create_link_with_op(variable=z_var, upstream_op=None, downstream_op=created)

        meta = OperationMeta(
        input_metas    = [TensorMeta(dtype=DataType.FP32, shape=meta.shape),
                          TensorMeta(dtype=DataType.FP32, shape=config.scale.shape),
                          TensorMeta(dtype=DataType.convert_from_torch(offset_dtype), shape=config.offset.shape)],
        output_metas   = [TensorMeta(dtype=DataType.convert_from_torch(value_dtype), shape=meta.shape)],
        operation_name = created.name, operation_type=created.type, executing_order=-1)
        created.meta_data = meta
        return created

    def insert_dequant_on_variable(
        self, graph: BaseGraph, var: QuantableVariable,
        config: TensorQuantizationConfig, related_op: Operation,
        meta: TensorMeta = None) -> Operation:
        if meta is None: meta = var.meta
        offset_dtype, value_dtype = self.infer_qtype(config)
        scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
        offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

        s_var = graph.create_variable(name=None, value=scale.clone(), is_parameter=True)
        z_var = graph.create_variable(name=None, value=offset.clone(), is_parameter=True)
        created = graph.create_operation(op_type='DequantizeLinear', attributes={})

        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            assert isinstance(config, ChannelwiseTensorQuantizationConfig)
            created.attributes['axis'] = config.channel_axis

        # PATCH 20220803, OPSET 13 REQUIRES AXIS = 0
        if config.policy.has_property(QuantizationProperty.PER_TENSOR):
            created.attributes['axis'] = 0

        if var in related_op.inputs:
            graph.insert_op_between_var_and_op(created, up_var=var, down_op=related_op)
        else: graph.insert_op_on_var(created, var=var.name)

        graph.create_link_with_op(variable=s_var, upstream_op=None, downstream_op=created)
        graph.create_link_with_op(variable=z_var, upstream_op=None, downstream_op=created)

        dq_meta = OperationMeta(
        input_metas    = [TensorMeta(dtype=DataType.convert_from_torch(value_dtype), shape=meta.shape),
                          TensorMeta(dtype=DataType.FP32, shape=config.scale.shape),
                          TensorMeta(dtype=DataType.convert_from_torch(offset_dtype), shape=config.offset.shape)],
        output_metas   = [TensorMeta(dtype=DataType.FP32, shape=meta.shape)],
        operation_name = created.name, operation_type=created.type, executing_order=-1)
        created.meta_data = dq_meta
        return created

    def remove_activation_ops(self, graph: BaseGraph) -> BaseGraph:
        """For Asymmetric Quantization Policy, Activations like Relu & Clip can
        be removed from your network safely. Their function can be replaced by
        quant & dequant operations.

        Those activation is unnecessary for Asymmetric quantized network.

        Args:
            graph (BaseGraph): Processing Graph
            activation_ops (List[Operation]): Removing activations.
        """
        removed_activations = []
        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'Relu', 'Clip'}:
                config = op.config.output_quantization_config[0]
                # Only ASYMMETRICAL quantized activations can be safely removed.
                if config.policy.has_property(QuantizationProperty.SYMMETRICAL): continue

                if not isinstance(config.scale, torch.Tensor): continue
                if not isinstance(config.offset, torch.Tensor): continue

                range_min = (config.scale * (config.quant_min - config.offset)).min().item()
                range_max = (config.scale * (config.quant_max - config.offset)).max().item()

                if op.type == 'Relu':
                    if range_min >= 0:
                        removed_activations.append(op)

                if op.type == 'Clip':
                    if op.num_of_input == 3:
                        clip_min = op.inputs[1].value
                        clip_max = op.inputs[2].value
                        if clip_min is not None: clip_min = clip_min.item()
                        else: clip_min = float('-inf')
                        if clip_max is not None: clip_max = clip_max.item()
                        else: clip_max = float('+inf')

                        if range_min >= clip_min and range_max <= clip_max:
                            removed_activations.append(op)

        # Activation op can only be relu and clip,
        # so it is safe to access op.inputs[0], op.outputs[0] as their input and output.
        for op in removed_activations:
            if not isinstance(op, QuantableOperation): continue
            if len(graph.get_upstream_operations(op)) == 0: continue
            quant_config = op.config.output_quantization_config[0]

            upstream_op = graph.get_upstream_operations(op)[0]
            if not isinstance(upstream_op, QuantableOperation): continue
            if len(graph.get_downstream_operations(upstream_op)) != 1: continue
            input_var, input_cfg = op.inputs[0], op.config.input_quantization_config[0]
            if not input_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL): continue

            # PATCH 20220304 Removing graph output op might cause error.
            if op.outputs[0].name in graph.outputs:
                graph.outputs.pop(op.outputs[0].name)
                graph.outputs[input_var.name] = input_var

            input_var, output_var = op.inputs[0], op.outputs[0]
            graph.remove_operation(op)
            graph.create_link_with_var(input_var, output_var)

            # insert quant & dequant op on var
            self.insert_dequant_on_variable(
                graph=graph, var=input_var, config=quant_config, 
                related_op=upstream_op, meta=input_var.meta)
            self.insert_quant_on_variable(
                graph=graph, var=input_var, config=quant_config, 
                related_op=upstream_op, meta=input_var.meta)

        # formatter = GraphFormatter(graph)
        # formatter(GraphCommand(GraphCommandType.DELETE_ISOLATED))
        return graph

    def remove_duplicated_quant_op(self, graph: BaseGraph) -> BaseGraph:
        """
        Pattern:        Quant - Dequant - Quant - Dequant

        Can reduced to: Quant - Dequant

        Some time there will be more than 1 quant operation inserted with a
        single variable. This function will remove duplicated quant operation
        from variable if it is possible.

        If inserted quant operations do not share a same zeropoint and scale,
        Then there is no way to remove any one of them.
        """
        interested_pairs = []
        for qt_op in graph.operations.values():
            if qt_op.type == 'QuantizeLinear':
                if len(graph.get_upstream_operations(qt_op)) != 1: continue
                if graph.get_upstream_operations(qt_op)[0].type != 'DequantizeLinear': continue
                interested_pairs.append((qt_op, graph.get_upstream_operations(qt_op)[0]))

        mark_to_remove = set()
        for qt_op, dq_op in interested_pairs:
            assert isinstance(dq_op, Operation) and isinstance(qt_op, Operation)
            scale_diff     = torch.max(torch.abs(dq_op.inputs[1].value - qt_op.inputs[1].value)).item()
            zeropoint_diff = torch.max(torch.abs(dq_op.inputs[2].value - qt_op.inputs[2].value)).item()
            if scale_diff < 1e-5 and zeropoint_diff < 0.5: # zero point 是整数，所以只要误差小于1就行了。
                # mark quant operation and its following operation(suppose to be another dequantization op)
                mark_to_remove.add(qt_op)
                assert len(graph.get_downstream_operations(qt_op)) == 1, 'Oops, that should never happen.'
                mark_to_remove.add(graph.get_downstream_operations(qt_op)[0])

        for op in mark_to_remove:
            assert isinstance(op, Operation)
            input_var, output_var = op.inputs[0], op.outputs[0]
            graph.remove_operation(op)
            graph.create_link_with_var(input_var, output_var)

        """
        There is another type of fusion:
        Pattern:        Quant +-- Dequant
                              |
                              +-- Dequant

        Can reduce to:  Quant - Dequant +--
                                        |
                                        +--
                
        Not implemented.
        """
        return graph

    @ property
    def required_opsets(self) -> Dict[str, int]:
        extra_domain_versions = [('ai.onnx', 13)]
        return dict(extra_domain_versions)

    def convert_operation_from_opset11_to_opset13(self, graph:BaseGraph) -> None:
        """Convert your network from opset 11 standard towards opset 13 With
        Onnx definition, per-channel quant operation requires opset 13.

        Args:
            graph (BaseGraph): Processing graph.
        """
        # this func transform representation of certain op from opset 11 to 13
        for op in graph.operations.values():
            if op.type == 'ReduceSum' or op.type == 'Squeeze' or op.type == 'Unsqueeze':
                if 'axes' not in op.attributes: continue # is already v13
                axes = convert_any_to_torch_tensor(op.attributes.pop('axes'), dtype=torch.int64)
                var = graph.create_variable(name=None, value=axes, is_parameter=True)
                graph.create_link_with_op(variable=var, upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.append(TensorMeta.parsing_from_torch_tensor(var.value, var.name))

            elif op.type == 'Split':
                if 'split' not in op.attributes: continue # split is already v13
                split = convert_any_to_torch_tensor(op.attributes.pop('split'), dtype=torch.int64)
                var = graph.create_variable(name=None, value=split, is_parameter=True)
                graph.create_link_with_op(variable=var, upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.append(TensorMeta.parsing_from_torch_tensor(var.value, var.name))

    def convert_operation(self, graph: BaseGraph, op: QuantableOperation,
                          process_activation: bool, process_parameter: bool,
                          quant_param_to_int: bool):
        """Convert an operation to onnx quant & dequant format by inserting
        necessary quant & dequant op around it. There are 2 ways to represent
        quantized ONNX models:

        Operator Oriented. All the quantized operators have their own ONNX definitions,
            like QLinearConv, MatMulInteger and etc.

        Tensor Oriented, aka Quantize and DeQuantize (QDQ).
            This format uses DQ(Q(tensor)) to simulate the quantize and dequantize process,
            and QuantizeLinear and DeQuantizeLinear operators also carry the quantization parameters.

        Quantization-Aware training (QAT) models converted from Tensorflow or exported from PyTorch.

        Quantized models converted from tflite and other framework.

        Args:
            graph (BaseGraph): PPQ IR
            op (Operation): Converting op
            process_activation (bool): Converting op's activation
            process_parameter (bool): Converting op's parameter
            quant_param_to_int (bool): Quant op's parameter to int8
        """
        # collect quantable vars, where we need to insert quant and dequant op
        for config, var in op.config_with_variable:
            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var): continue

            meta = var.meta
            if var.is_parameter and process_parameter:
                # we do not want to process clip value here.
                if op.type in {'Clip', 'Pad'}: continue                
                assert len(var.dest_ops) == 1, (
                f'Can not export variable {var.name}, cause it has more than 1 destination operations. '
                'PPQ require all parameters to have only 1 destination operation.')

                # override quantization state, so that we can export parameter correctly.
                if config.state == QuantizationStates.BAKED:
                    config.state = QuantizationStates.ACTIVATED
                if config.state == QuantizationStates.PASSIVE_BAKED:
                    config.state = QuantizationStates.PASSIVE

                # if not quant parameter to int, all parameter should export as fp32.
                # needs insert both quant and dequant op for them
                if not quant_param_to_int:
                    created = self.insert_quant_on_variable(
                        graph=graph, var=var, config=config, related_op=op, meta=meta)
                    var = created.outputs[0]

                self.insert_dequant_on_variable(
                    graph=graph, var=var, config=config, related_op=op, meta=meta)
                if quant_param_to_int:
                    var.value = PPQLinearQuant_toInt(tensor=var.value, config=config)

            elif (not var.is_parameter) and process_activation:
                created = self.insert_quant_on_variable(
                    graph=graph, var=var, config=config, related_op=op, meta=meta)
                self.insert_dequant_on_variable(
                    graph=graph, var=created.outputs[0], config=config, 
                    related_op=op, meta=meta)

    def prepare_graph(
        self, graph: BaseGraph,
        process_activation: bool = True,
        process_parameter: bool = True,
        remove_activation_fn: bool = True,
        quant_parameter_to_int: bool = True) -> BaseGraph:
        """Prepare your graph for exporting.

        There are many works to do with your graph:

            1. Insert Quant and Dequant operation within your graph.

            2. Remove all unnecessary activations.

            3. Quantize all parameters of your graph, convert them to int8.

        Args:
            graph (BaseGraph): Processing Graph

        Returns:
            BaseGraph: Processed Graph
        """
        self.convert_operation_from_opset11_to_opset13(graph)

        # remove switchers.
        if not PPQ_CONFIG.EXPORT_DEVICE_SWITCHER:
            processor = GraphDeviceSwitcher(graph)
            processor.remove_switcher()

        # mark quantable variables
        for op in [op for op in graph.operations.values()]:
            if not isinstance(op, QuantableOperation): continue
            self.convert_operation(
                graph=graph, op=op,
                process_activation=process_activation,
                process_parameter=process_parameter,
                quant_param_to_int=quant_parameter_to_int)

        # remove activations
        if remove_activation_fn:
            # remove useless activation.
            self.remove_activation_ops(graph)

        return self.remove_duplicated_quant_op(graph)

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None) -> None:
        graph = self.prepare_graph(graph)

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            super().export_quantization_config(config_path, graph)

        name = graph._name
        if not name: name = 'PPL Quantization Tool - Onnx Export'

        # Ready to export onnx graph definition.
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
            initializer=_initilizers)
        extra_opsets = self.required_opsets

        opsets = []
        if GRAPH_OPSET_ATTRIB in graph._detail:
            for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
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
            graph_def, producer_name=PPQ_CONFIG.NAME, opset_imports=opsets)
        onnx_model.ir_version = 7
        # onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, file_path)
