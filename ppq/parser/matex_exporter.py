from ast import Continue
from typing import Dict, List, Tuple

import onnx
import torch
from onnx import helper
from ppq.core import (COMPELING_OP_TYPES, PPQ_CONFIG,
                      ChannelwiseTensorQuantizationConfig, DataType,
                      OperationMeta, QuantizationProperty, QuantizationStates,
                      TensorMeta, TensorQuantizationConfig,
                      convert_any_to_torch_tensor, ppq_legacy)
from ppq.IR import (BaseGraph, Operation, QuantableOperation,
                    QuantableVariable, Variable)
from ppq.IR.base.command import GraphCommand, GraphCommandType
from ppq.IR.morph import GraphDeviceSwitcher, GraphFormatter
from ppq.core.common import GRAPH_OPSET_ATTRIB
from ppq.utils.round import ppq_tensor_round

from .onnx_exporter import OnnxExporter


# legacy exporter since ppq 0.6.4
# use onnxruntime exporter instead.
class MetaxExporter(OnnxExporter):
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
        ppq_legacy('Metax Exporter', version='0.6.4', adapt_to='ONNXRUNTIME Exporter')
        self.removed_activation_types = removed_activation_types

    def inplace_quantization(self, var: QuantableVariable, is_bias: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        config = var.dest_op_configs[0]
        assert isinstance(config, TensorQuantizationConfig)
        tensor = var.value
        scale, offset = config.scale, config.offset
        axis = 1
        if config.policy.has_property(QuantizationProperty.PER_TENSOR):
            tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
            tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        else:
            assert isinstance(config, ChannelwiseTensorQuantizationConfig)
            shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
            scale, offset = scale.view(shape), offset.view(shape)
            tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
            tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
            axis = config.channel_axis
        if is_bias:
            var.value = tensor.type(torch.int32)
        elif config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            var.value = tensor.type(torch.uint8)
        else:
            var.value = tensor.type(torch.int8)
        return (convert_any_to_torch_tensor(config.scale, dtype=torch.float32),
            convert_any_to_torch_tensor(config.offset, dtype=var.value.dtype), axis)

    def insert_quant_dequant_on_var(
        self, graph: BaseGraph, var: QuantableVariable,
        config: TensorQuantizationConfig=None, single_branch: bool=False,
        dest_op: Operation=None) -> None:
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
        if config is None:
            configs = [cfg for cfg in [var.source_op_config] + var.dest_op_configs if cfg is not None]
            config = configs[0]

        offset_dtype = torch.int8
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL): offset_dtype = torch.uint8
        scale  = convert_any_to_torch_tensor(config.scale, dtype=torch.float32)
        offset = convert_any_to_torch_tensor(config.offset, dtype=offset_dtype)

        qt_svar = graph.create_variable(name=None, value=scale.clone(), is_parameter=True)
        qt_zvar = graph.create_variable(name=None, value=offset.clone(), is_parameter=True)
        dq_svar = graph.create_variable(name=None, value=scale.clone(), is_parameter=True)
        dq_zvar = graph.create_variable(name=None, value=offset.clone(), is_parameter=True)
        qt_op   = graph.create_operation(op_type='QuantizeLinear', attributes={})
        dq_op   = graph.create_operation(op_type='DequantizeLinear', attributes={})

        if single_branch:
            upstream_op, downstream_op = var.source_op, dest_op
            graph.insert_op_between_ops(qt_op, up_op=upstream_op, down_op=downstream_op)
            graph.insert_op_between_ops(dq_op, up_op=qt_op, down_op=downstream_op)

        if not single_branch:
            graph.insert_op_on_var(dq_op, var=var.name)
            graph.insert_op_on_var(qt_op, var=var.name)

        graph.create_link_with_op(variable=qt_svar, upstream_op=None, downstream_op=qt_op)
        graph.create_link_with_op(variable=qt_zvar, upstream_op=None, downstream_op=qt_op)

        graph.create_link_with_op(variable=dq_svar, upstream_op=None, downstream_op=dq_op)
        graph.create_link_with_op(variable=dq_zvar, upstream_op=None, downstream_op=dq_op)

    def insert_dequant_param(self, graph: BaseGraph, var: Variable, is_bias: bool) -> None:
        # apply inplace quantization for parameters and only insert dequant op
        # on pre-quant var
        scale, offset, axis = self.inplace_quantization(var, is_bias)
        dequant_op = graph.create_operation(op_type='DequantizeLinear', attributes={'axis':axis})
        graph.insert_op_on_var(dequant_op, var.name)

        dq_svar = graph.create_variable(name=None, value=scale.clone(), is_parameter=True)
        dq_zvar = graph.create_variable(name=None, value=offset.clone(), is_parameter=True)
        graph.create_link_with_op(dq_svar, upstream_op=None, downstream_op=dequant_op)
        graph.create_link_with_op(dq_zvar, upstream_op=None, downstream_op=dequant_op)

    def correct_param_meta(self, graph: BaseGraph) -> None:
        # correct parameter meta data
        for var in graph.variables.values():
            if var.is_parameter:
                for op in var.dest_ops:
                    if op.meta_data is None:
                        op.meta_data = OperationMeta([TensorMeta(DataType.FP32, None, v.name) for v in
                            op.inputs], [TensorMeta(DataType.FP32, None, v.name) for v in
                            op.outputs], op.name, op.type, -1)

                    if torch.is_tensor(var.value):
                        op.meta_data.input_metas[op.inputs.index(var)] = (
                            TensorMeta.parsing_from_torch_tensor(var.value, var.name))
                    else:
                        op.meta_data.input_metas[op.inputs.index(var)] = (
                            TensorMeta.parsing_from_numpy_ndarray(var.value, var.name))

        # add variable meta info in topo order
        for op in graph.topological_sort():
            if op.type == 'QuantizeLinear' and op.inputs[0].source_op is not None:
                input_var = op.inputs[0]
                op.meta_data.input_metas[0] = input_var.meta
                op.meta_data.output_metas[0].shape = input_var.meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[2].dtype
                dequant_op = op.outputs[0].dest_ops[0]
                dequant_op.meta_data.input_metas[0] = op.meta_data.output_metas[0]
                dequant_op.meta_data.output_metas[0].shape = input_var.meta.shape
                dequant_op.meta_data.output_metas[0].dtype = dequant_op.meta_data.input_metas[1].dtype
            # must be input
            elif op.type == 'QuantizeLinear' and op.inputs[0].value is None:
                var = op.outputs[0]
                dest_op = var.dest_ops[0]
                dest_idx = var.dest_idx[0]
                meta = dest_op.meta_data.input_metas[dest_idx]
                # meta can't be None itself because we have built TensorMeta
                # for every input when we correct param meta
                while meta.shape is None or meta.dtype is None:
                    assert isinstance(dest_op, Operation)
                    var = dest_op.outputs[0]
                    dest_op = var.dest_ops[0]
                    dest_idx = var.dest_idx[0]
                    meta = dest_op.meta_data.input_metas[dest_idx]

                dequant_op = op.outputs[0].dest_ops[0]
                dequant_op.meta_data.output_metas[0] = meta
                dequant_op.meta_data.input_metas[0].shape = meta.shape
                dequant_op.meta_data.input_metas[0].dtype = dequant_op.meta_data.input_metas[2].dtype
                op.meta_data.input_metas[0] = meta
                op.meta_data.output_metas[0].shape = meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[2].dtype
            elif op.type == 'DequantizeLinear' and op.inputs[0].source_op is None:
                op.meta_data.output_metas[0].shape = op.meta_data.input_metas[0].shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[1].dtype

    def remove_activation(self, graph: BaseGraph, activation_ops: List[Operation]) -> None:
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

            input_var, output_var = op.inputs[0], op.outputs[0]
            graph.remove_operation(op)
            graph.create_link_with_var(input_var, output_var)

        formatter = GraphFormatter(graph)
        formatter(GraphCommand(GraphCommandType.DELETE_ISOLATED))

    def required_opsets(self) -> Dict[str, int]:
        extra_domain_versions = [
            ('ai.onnx', 13)
            ]
        return dict(extra_domain_versions)

    def transform_op(self, graph:BaseGraph) -> None:
        # this func transform representation of certain op from opset 11 to 13
        for op in graph.operations.values():
            if op.type == 'ReduceSum' or op.type == 'Squeeze' or op.type == 'Unsqueeze':
                axes = convert_any_to_torch_tensor(op.attributes.pop('axes'), dtype=torch.int64)
                var = graph.create_variable(name=None, value=axes, is_parameter=True)
                graph.create_link_with_op(variable=var, upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.append(TensorMeta.parsing_from_torch_tensor(var.value, var.name))

            elif op.type == 'Split':
                split = convert_any_to_torch_tensor(op.attributes.pop('split'), dtype=torch.int64)
                var = graph.create_variable(name=None, value=split, is_parameter=True)
                graph.create_link_with_op(variable=var, upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.append(TensorMeta.parsing_from_torch_tensor(var.value, var.name))

    def collect_compel_pair(self, graph: BaseGraph) -> None:
        # make sure settings of output of Add, Concat, Sub ops are applied to inputs as well
        # this func should be called only for a supplemental method for coherent quantization
        # setting for special ops
        compel_ops, compel_pairs = [], []
        for op in graph.operations.values():
            if op.type in COMPELING_OP_TYPES and isinstance(op, QuantableOperation):
                compel_ops.append(op)
        for op in compel_ops:
            assert isinstance(op, QuantableOperation)
            for var in op.inputs:
                assert isinstance(var, QuantableVariable)
                if var.source_op_config is not None and \
                    var.source_op_config.dominated_by != op.config.input_quantization_config[0].dominated_by:
                    compel_pairs.append((var, op, op.config.input_quantization_config[0]))
        return compel_pairs

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None) -> None:
        # remove switchers.
        if not PPQ_CONFIG.EXPORT_DEVICE_SWITCHER:
            processor = GraphDeviceSwitcher(graph)
            processor.remove_switcher()

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            super().export_quantization_config(config_path, graph)

        # collect compel var-op pair in advance to avoid graph change influence
        compel_pairs = self.collect_compel_pair(graph)

        # collect quantable vars, where we need to insert quant and dequant op
        # note that we assume all quantization configs of the same variable maintained
        # by different ops are actually the same
        quantable_vars,removed_activations = [], []
        for var in graph.variables.values():
            if isinstance(var, QuantableVariable):
                configs = [var.source_op_config] + var.dest_op_configs
                for cfg in configs:
                    if cfg is not None and not QuantizationStates.can_export(cfg.state):
                        raise AttributeError(f'quantization state of variable {var.name} is unexpected, \
                        please check if you have finished the whole quantization process')
                    elif cfg is not None and cfg.state not in {QuantizationStates.FP32, QuantizationStates.SOI}:
                        quantable_vars.append((cfg, var))
                        break

        for cfg, var in quantable_vars:
            assert isinstance(var, QuantableVariable)
            assert isinstance(cfg, TensorQuantizationConfig)
            # assume parameter var is used by only one op
            if var.is_parameter:
                if var.dest_ops[0].is_computing_op and var.dest_idx[0] > 1:
                    self.insert_dequant_param(graph, var, True)
                else:
                    self.insert_dequant_param(graph, var, False)

            elif len(var.dest_ops) == 1 and var.dest_ops[0].type in self.removed_activation_types and \
                cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                removed_activations.extend(var.dest_ops)
            else:
                self.insert_quant_dequant_on_var(graph, var)

        self.remove_activation(graph, removed_activations)

        # insert another pair of quant and dequant ops for compel pairs
        for (var, op, cfg) in compel_pairs:
            assert isinstance(var, Variable)
            # skip newly added ops
            while op not in var.dest_ops:
                assert var.dest_ops[0].type in {'QuantizeLinear', 'DequantizeLinear'}
                var = var.dest_ops[0].outputs[0]
            self.insert_quant_dequant_on_var(graph, var, cfg, True, op)

        # collect meta info for parameters and newly-added variables
        self.correct_param_meta(graph)
        self.transform_op(graph)

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
            initializer=_initilizers,
        )

        extra_opsets = self.required_opsets()

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
