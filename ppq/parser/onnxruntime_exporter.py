from typing import Any, Dict, List, Tuple
import onnx
import torch
from onnx import helper
from ppq.core import (COMPELING_OP_TYPES, EXPORT_DEVICE_SWITCHER, PPQ_NAME,
                      OperationMeta, QuantizationProperty, QuantizationStates,
                      TensorMeta, TensorQuantizationConfig,
                      convert_any_to_torch_tensor)
from ppq.IR import (BaseGraph, Operation, QuantableOperation,
                    QuantableVariable, Variable)
from ppq.IR.base.command import GraphCommand, GraphCommandType
from ppq.IR.morph import GraphDeviceSwitcher, GraphFormatter
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
        self.removed_activation_types = removed_activation_types

    def inplace_quantization(self, var: Variable, is_bias: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        config = var.dest_op_configs[0]
        tensor = var.value
        scale, offset = config.scale, config.offset
        axis = 1
        if config.policy.has_property(QuantizationProperty.PER_TENSOR):
            tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
            tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        else:
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
        return convert_any_to_torch_tensor(config.scale, dtype=torch.float32), \
            convert_any_to_torch_tensor(config.offset, dtype=var.value.dtype), axis

    def insert_quant_dequant_var(self, 
                                graph: BaseGraph,
                                var: Variable,
                                config: TensorQuantizationConfig=None,
                                single_branch: bool=False,
                                dest_op: Operation=None
    ) -> None:
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

        offset_dtype = torch.uint8 if config.policy.has_property( \
            QuantizationProperty.ASYMMETRICAL) else torch.int8

        scale = convert_any_to_torch_tensor(config.scale, dtype=torch.float32)
        offset = convert_any_to_torch_tensor(config.offset, dtype=offset_dtype)

        x_scale = Variable(name=var.name + '_scale', value=scale, is_parameter=True)
        x_zero_point = Variable(name=var.name+'_zero_point', value=offset, is_parameter=True)

        quant_op = Operation(name=var.name + '_QuantizeLinear', op_type='QuantizeLinear', attributes={})
        dequant_op = Operation(name=var.name + '_DequantizeLinear', op_type='DequantizeLinear', attributes={})


        intermediate_var_2 = Variable(name=var.name + '_DequantizeLinear', source_op=dequant_op)
        graph.append_variable(intermediate_var_2)
        if single_branch:
            assert dest_op is not None, "a destination op should be given for single branch insertion"
            intermediate_var_2.dest_ops.append(dest_op)
            dest_op.inputs[dest_op.inputs.index(var)] = intermediate_var_2
        else:
            intermediate_var_2.dest_ops.extend(var.dest_ops)
            for op in var.dest_ops:
                op.inputs[op.inputs.index(var)] = intermediate_var_2

        graph.append_operation(dequant_op)
        dequant_op.outputs.append(intermediate_var_2)

        intermediate_var_1 = Variable(name=var.name + '_QuantizeLinear', source_op=quant_op, dest_ops=[dequant_op])
        graph.append_variable(intermediate_var_1)
        dequant_op.inputs.append(intermediate_var_1)
        graph.append_operation(quant_op)
        quant_op.outputs.append(intermediate_var_1)

        quant_op.inputs.append(var)
        if single_branch:
            var.dest_ops[var.dest_ops.index(dest_op)] = quant_op
        else:
            var.dest_ops.clear()
            var.dest_ops.append(quant_op)
        
        quant_op.inputs.extend([x_scale, x_zero_point])
        dequant_op.inputs.extend([x_scale, x_zero_point])
        
        x_scale.dest_ops.extend([quant_op, dequant_op])
        x_zero_point.dest_ops.extend([quant_op, dequant_op])

        graph.append_variable(x_scale)
        graph.append_variable(x_zero_point)


    def insert_dequant_param(self, graph: BaseGraph, var: Variable, is_bias: bool) -> None:
        # apply inplace quantization for parameters and only insert dequant op
        # on pre-quant var
        scale, offset, axis = self.inplace_quantization(var, is_bias)
        dequant_op = Operation(name=var.name + '_DequantizeLinear', op_type='DequantizeLinear', attributes={'axis':axis})
        graph.insert_operation_on_var(dequant_op, var.name)
        scale = Variable(name=var.name + '_scale', value=scale, is_parameter=True, dest_ops=[dequant_op])
        offset = Variable(name=var.name + '_zero_point', value=offset, is_parameter=True, dest_ops=[dequant_op])
        graph.append_variable(scale)
        graph.append_variable(offset)
        dequant_op.inputs.extend([scale, offset])

    def correct_param_meta(self, graph: BaseGraph) -> None:
        # correct parameter meta data
        for var in graph.variables.values():
            if var.is_parameter:
                for op in var.dest_ops:
                    if op.meta_data is None:
                        op.meta_data = OperationMeta([TensorMeta(None, None, v.name) for v in 
                            op.inputs], [TensorMeta(None, None, v.name) for v in 
                            op.outputs], op.name, op.type, -1)

                    if torch.is_tensor(var.value):
                        op.meta_data.input_metas[op.inputs.index(var)] = \
                            TensorMeta.parsing_from_torch_tensor(var.value, var.name)
                    else:
                        op.meta_data.input_metas[op.inputs.index(var)] = \
                            TensorMeta.parsing_from_numpy_ndarray(var.value, var.name)

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
        for op in activation_ops:
            if not isinstance(op, QuantableOperation): continue
            input_variable = op.inputs[0]
            assert isinstance(input_variable, QuantableVariable)
            cfg = input_variable.source_op_config
            if cfg is None: continue # upstream operation is not a QuantableOperation
            
            if not cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                continue
            if len(op.inputs) <= 1:
                graph.remove_operation(op)
            else:
                for var in op.inputs[1:]:
                    var.dest_ops.clear()
                    graph.delete_variable(var.name)
                while len(op.inputs) > 1:
                    op.inputs.pop()
                graph.remove_operation(op)
        formater = GraphFormatter(graph)
        formater.process(GraphCommand(GraphCommandType.DELETE_ISOLATED))

    def required_opsets(self) -> Dict[str, int]:
        extra_domain_versions = [
            ("ai.onnx", 13),
            ("com.microsoft", 1),
            ("com.microsoft.nchwc", 1),
            ("ai.onnx.training", 1),
            ("ai.onnx.preview.training", 1),
            ("com.microsoft.experimental", 1),
            ("ai.onnx.ml", 2)
            ]
        return dict(extra_domain_versions)

    def transform_op(self, graph:BaseGraph) -> None:
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

    def collect_compel_pair(self, graph: BaseGraph) -> None:
        # make sure settings of output of Add, Concat, Sub ops are applied to inputs as well
        # this func should be called only for a supplemental method for coherent quantization
        # setting for special ops
        compel_ops, compel_pairs = [], []
        for op in graph.operations.values():
            if op.type in COMPELING_OP_TYPES and isinstance(op, QuantableOperation):
                compel_ops.append(op)
        for op in compel_ops:
            for var in op.inputs:
                if var.source_op_config is not None and \
                    var.source_op_config.dominated_by != op.config.input_quantization_config[0].dominated_by:
                    compel_pairs.append((var, op, op.config.input_quantization_config[0]))
        return compel_pairs

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None) -> None:
        # remove switchers.
        if not EXPORT_DEVICE_SWITCHER:
            processer = GraphDeviceSwitcher(graph)
            processer.remove_switcher()

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
                        raise AttributeError(f"quantization state of variable {var.name} is unexpected, \
                        please check if you have finished the whole quantization process")
                    elif cfg is not None and cfg.state not in {QuantizationStates.FP32, QuantizationStates.SOI}:
                        quantable_vars.append(var)
                        break

        for var in quantable_vars:
            # assume parameter var is used by only one op
            if var.is_parameter:
                if var.dest_ops[0].is_computing_op and var.dest_idx[0] > 1:
                    self.insert_dequant_param(graph, var, True)
                else:
                    self.insert_dequant_param(graph, var, False)
            elif not(var.source_op is not None and var.source_op.is_computing_op and\
                len(var.dest_ops) == 1 and var.dest_ops[0].type in self.removed_activation_types):
                self.insert_quant_dequant_var(graph, var)
            else:
                removed_activations.extend(var.dest_ops)

        self.remove_activation(graph, removed_activations)

        # insert another pair of quant and dequant ops for compel pairs
        for (var, op, cfg) in compel_pairs:
            # skip newly added ops
            while op not in var.dest_ops:
                assert var.dest_ops[0].type in {'QuantizeLinear', 'DequantizeLinear'}
                var = var.dest_ops[0].outputs[0]
            self.insert_quant_dequant_var(graph, var, cfg, True, op)

        # collect meta info for parameters and newly-added variables
        self.correct_param_meta(graph)
        self.transform_op(graph)

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

        extra_opsets = self.required_opsets()

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
