
from typing import List, Tuple

import torch
from ppq.core import (DataType, QuantizationStates, TargetPlatform, TensorMeta,
                      TensorQuantizationConfig, convert_any_to_torch_tensor,
                      ppq_warning)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.IR.search import SearchableGraph
from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt
from ppq.utils.round import ppq_tensor_round

from .onnxruntime_exporter import ONNXRUNTIMExporter


class ORTOOSExporter(ONNXRUNTIMExporter):

    @property
    def required_opsets(self):
        return {
            'ai.onnx': 13,
            'com.microsoft': 1,
        }

    @property
    def ONNX_QUANTABLE_TABLE(self):
        """Quantable operations for com.microsoft scope, see
        https://github.com.

        /microsoft/onnxruntime/blob/master/docs/ContribOperators.md for detail.

        com.microsoft.QAttention
        com.microsoft.QGemm
        com.microsoft.QLinearAdd
        com.microsoft.QLinearAveragePool
        com.microsoft.QLinearConcat
        com.microsoft.QLinearConv
        com.microsoft.QLinearGlobalAveragePool
        com.microsoft.QLinearLeakyRelu
        com.microsoft.QLinearMul
        com.microsoft.QLinearReduceMean
        com.microsoft.QLinearSigmoid

        Returns:
            _type_: _description_
        """
        return {
            'Add': 'QLinearAdd',
            'Mul': 'QLinearMul',
            'AveragePool': 'QLinearAveragePool',
            'Conv': 'QLinearConv',
            'GlobalAveragePool': 'QLinearGlobalAveragePool',
            'MatMul': 'QLinearMatMul', # Qlinear MatMul is a standard onnx operation.
            'Gemm': 'QGemm',
            'Concat': 'Concat', # no need to convert concat.
            'LeakyRelu': 'QLinearLeakyRelu',
             # "ReduceMean": "QLinearReduceMean", # onnx not implemented.
            'Sigmoid': 'QLinearSigmoid'}

    def conversion_preprocess(self, op: Operation) -> Tuple[List[Variable], List[TensorMeta]]:
        """Detach all input variable from given op, prepare for inserting input
        variable for it.

        Args:
            op (Operation): _description_

        Returns:
            List[Variable]: all detached variable
        """
        inputs = [var for var in op.inputs]
        input_metas = op.meta_data.input_metas.copy()
        for var in op.inputs:
            var.dest_ops.remove(op)
        op.inputs.clear()
        op.meta_data.input_metas.clear()
        return inputs, input_metas


    def convert_operation(self, graph: BaseGraph, op: QuantableOperation,
                          process_activation: bool, process_parameter: bool,
                          quant_param_to_int: bool):
        """Convert an operation to onnx operator oriented format. There are 2
        ways to represent quantized ONNX models:

        Operator Oriented. All the quantized operators have their own ONNX definitions,
            like QLinearConv, MatMulInteger and etc.

        Tensor Oriented, aka Quantize and DeQuantize (QDQ).
            This format uses DQ(Q(tensor)) to simulate the quantize and dequantize process,
            and QuantizeLinear and DeQuantizeLinear operators also carry the quantization parameters.

        Quantization-Aware training (QAT) models converted from Tensorflow or exported from PyTorch.

        Quantized models converted from tflite and other framework.

        Args:
            graph (BaseGraph): _description_
            op (QuantableOperation): _description_
            process_activation (bool): _description_
            process_parameter (bool): _description_
            quant_param_to_int (bool): _description_

        Returns:
            _type_: _description_
        """
        if op.type in self.ONNX_QUANTABLE_TABLE:
            if op.type == 'Concat': return
            # Those operation can convert to onnx operation-oriented quantized op.

            inputs, input_metas = self.conversion_preprocess(op)
            bias, bias_meta, bias_config = None, None, None

            if op.type == 'Gemm':
                if op.attributes.get('alpha') != 1:
                    raise ValueError(f'Can not export gemm {op.name} with alpha != 1')
                if op.attributes.get('beta') != 1:
                    raise ValueError(f'Can not export gemm {op.name} with beta != 1')
                op.attributes.pop('alpha')
                op.attributes.pop('beta')

            if op.type in {'Conv', 'Gemm'} and len(inputs) == 3: # has bias
                bias        = inputs[-1]
                bias_meta   = input_metas[-1]
                bias_config = op.config.input_quantization_config[-1]
                inputs = inputs[: -1] # remove bias from inputs, process it later.

            # process input
            for config, var, meta in zip(op.config.input_quantization_config, inputs, input_metas):
                otype, vtype = self.infer_qtype(config)
                scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
                offset = ppq_tensor_round(config.offset.clone()).type(otype)

                if var.is_parameter:
                    if config.state not in {
                        QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE,
                        QuantizationStates.PASSIVE_BAKED, QuantizationStates.BAKED}:
                        raise PermissionError(
                            f'Can not export operation {op.name} in onnx operator oriented quantize format, '
                            f'Cause its parameter {var.name} has not been correctly quantized.')

                    if config.num_of_bits != 8:
                        raise PermissionError(
                            f'Can not export operation {op.name} in onnx operator oriented quantize format, '
                            f'Cause its parameter {var.name} is not quantized with 8 bits.')

                    config.state = QuantizationStates.ACTIVATED
                    var.value    = PPQLinearQuant_toInt(tensor=var.value, config=config).to(vtype)

                graph.create_link_with_op(variable=var, upstream_op=var.source_op, downstream_op=op)
                graph.create_link_with_op(
                    variable=graph.create_variable(value=scale, is_parameter=True),
                    upstream_op=None, downstream_op=op)
                graph.create_link_with_op(
                    variable=graph.create_variable(value=offset, is_parameter=True),
                    upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.extend([
                    TensorMeta(dtype=DataType.convert_from_torch(vtype), shape=meta.shape),
                    TensorMeta(dtype=DataType.FP32, shape=config.scale.shape),
                    TensorMeta(dtype=DataType.convert_from_torch(otype), shape=config.offset.shape)])

            # process output
            assert len(op.outputs) == 1, 'Oops seems we got something wrong here.'
            config, var = op.config.output_quantization_config[0], op.outputs[0]
            otype, vtype = self.infer_qtype(config)
            scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(otype)

            graph.create_link_with_op(
                variable=graph.create_variable(value=scale, is_parameter=True),
                upstream_op=None, downstream_op=op)
            graph.create_link_with_op(
                variable=graph.create_variable(value=offset, is_parameter=True),
                upstream_op=None, downstream_op=op)
            op.meta_data.input_metas.extend([
                TensorMeta(dtype=DataType.FP32, shape=config.scale.shape),
                TensorMeta(dtype=DataType.convert_from_torch(otype), shape=config.offset.shape)])
            op.meta_data.output_metas = [TensorMeta(DataType.convert_from_torch(vtype),
                                                    shape=op.meta_data.output_metas[0].shape)]

            # process bias
            if bias is not None:
                assert isinstance(bias_config, TensorQuantizationConfig), 'Unexpected bias quantization configuration type.'
                if bias_config.state not in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE,
                                             QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                    raise ValueError(f'Can not export operation {op.name} cause its bias is not correctly quantized.')

                bias_config.state = QuantizationStates.ACTIVATED
                if bias_config.num_of_bits <= 16:
                    ppq_warning(f'Bias vector of operation {op.name} is quantized with {bias_config.num_of_bits} bits, '
                                'however onnx need it to be 32-bit quantized.')
                bias.value = PPQLinearQuant_toInt(tensor=bias.value, config=bias_config).to(vtype)
                graph.create_link_with_op(variable=bias, upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.extend([TensorMeta(dtype=DataType.INT32, shape=bias_meta.shape)])

                # reorder input.
                if op.type == 'Gemm':
                    meta = op.meta_data.input_metas
                    op.inputs[6], op.inputs[7], op.inputs[8] = op.inputs[8], op.inputs[6], op.inputs[7]
                    meta[6], meta[7], meta[8] = meta[8], meta[6], meta[7]

            # convert op type.
            op.type = self.ONNX_QUANTABLE_TABLE[op.type]

        else:
            # If operation is passive, skip it is safe.
            pass


    def prepare_graph(self, graph: BaseGraph, process_activation: bool = True,
                      process_parameter: bool = True, remove_activation_fn: bool = True,
                      quant_parameter_to_int: bool = True) -> BaseGraph:
        super().prepare_graph(graph, process_activation, process_parameter,
                              remove_activation_fn, quant_parameter_to_int)
        FP32_ONLY_TYPES = {'Add', 'Mul', 'Relu', 'Clip', 'Gemm', 'Conv', 'AveragePool',
                           'GlobalAveragePool', 'MatMul', 'LeakyRelu', 'Sigmoid', 'ReduceMean'}
        quantized_op = set()
        for op in graph.operations.values():
            if op.type in {'QGemm', 'QLinearConv', 'QLinearMatMul', 'QuantizeLinear',
                           'QLinearAdd', 'QLinearAveragePool', 'QLinearConcat', 'QLinearGlobalAveragePool',
                           'QLinearLeakyRelu', 'QLinearMul', 'QLinearReduceMean', 'QLinearSigmoid'}:
                quantized_op.add(op)

        processor = SearchableGraph(graph)
        # 执行子图遍历，将 int8 节点染色
        quantize_extension = processor.opset_matching(
            sp_expr=lambda x: x in quantized_op,
            rp_expr=lambda x, y: y.type not in FP32_ONLY_TYPES and TargetPlatform.is_quantized_platform(y.platform),
            ep_expr=lambda x: x.type in FP32_ONLY_TYPES or (not TargetPlatform.is_quantized_platform(x.platform)) or x.is_boundary)

        # might have some error...
        for op in [op for op in quantize_extension]:
            if op.type in FP32_ONLY_TYPES:
                var       = op.inputs[0]
                source_op = var.source_op
                assert isinstance(source_op, QuantableOperation)
                qconfig   = source_op.config.output_quantization_config[source_op.outputs.index(var)]
                self.insert_dequant_on_variable(
                    graph=graph, var=op.inputs[0],
                    config=qconfig, related_op=op)
                quantize_extension.remove(op)

        for op in quantize_extension:
            for input_var in op.inputs:
                if input_var.is_parameter: continue
                if op.type not in {'QGemm', 'QLinearConv', 'QLinearMatMul'}: continue
                if input_var.source_op not in quantize_extension or input_var.source_op is None:
                    assert isinstance(op, QuantableOperation)
                    qconfig = op.config.input_quantization_config[op.inputs.index(input_var)]
                    self.insert_quant_on_variable(
                        graph=graph, var=input_var,
                        config=qconfig, related_op=op)

        for output_var in graph.outputs.values():
            if output_var.source_op in quantize_extension:
                meta = output_var.meta

                assert isinstance(output_var.source_op, QuantableOperation)
                qconfig = output_var.source_op.config.output_quantization_config[output_var.src_idx]

                op = graph.create_operation(op_type='Temp')
                graph.create_link_with_op(output_var, upstream_op=output_var.source_op, downstream_op=op)
                self.insert_dequant_on_variable(
                    config=qconfig, graph=graph, var=output_var, related_op=op, meta=meta)
                created = op.inputs[0]
                graph.remove_operation(op)

                graph.outputs[output_var.name] = created
                output_var._name, created._name = created.name, output_var.name

        return graph
