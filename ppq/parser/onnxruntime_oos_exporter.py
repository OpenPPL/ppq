from typing import Tuple

import numpy as np
import onnx
import torch
from onnx import helper
from ppq.core import (EXPORT_DEVICE_SWITCHER, ORT_MICROSOFT_CONTRIB_LINEAR_OPS,
                      ORT_OOS_FUSE_START_OPS, PPQ_NAME, OperationMeta,
                      QuantizationProperty, QuantizationStates, TensorMeta,
                      TensorQuantizationConfig, convert_any_to_torch_tensor)
from ppq.IR import BaseGraph, Operation, QuantableVariable, Variable
from ppq.IR.morph import GraphDeviceSwitcher
from ppq.IR.quantize import QuantableOperation

from .onnxruntime_exporter import ONNXRUNTIMExporter


class ORTOOSExporter(ONNXRUNTIMExporter):
    ASYMMETRICAL_ZP_NP_TYPE = torch.uint8
    SYMMETRICAL_ZP_NP_TYPE = torch.int8
    BIAS_NP_TYPE = torch.int32
    SCALE_NP_TYPE = torch.float32

    SCALE_PARAMETER_SUFFIX = "_scale"
    ZP_PARAMETER_SUFFIX = "_zero_point"
    QUANTIZE_PARAMETER_SUFFIX = "_quantized"
    WEIGHT_QUANTIZE_PARAMETER_SUFFIX = "_weight"
    BIAS_QUANTIZE_PARAMETER_SUFFIX = "_bias"
    LINKER_VAR_SUFFIX = "_linker"
    QUANTIZE_LINEAR_SUFFIX = "_QuantizeLinear"
    DEQUANTIZE_LINEAR_SUFFIX = "_DequantizeLinear"

    @property
    def qlinear_op_map(self):
        return {
            "Add": "QLinearAdd",
            "Mul": "QLinearMul",
            "AveragePool": "QLinearAveragePool",
            "Conv": "QLinearConv",
            "GlobalAveragePool": "QLinearGlobalAveragePool",
            "MatMul": "QLinearMatMul",
        }

    @property
    def qlinear_ops(self):
        return self.qlinear_op_map.values()

    def get_qlinear_op_type(self, op_type: str) -> str:
        return self.qlinear_op_map.get(op_type, op_type)

    def get_qlinear_op_dominant_dtype(self, op: Operation) -> np.dtype:
        if op.type in [
            "QLinearConv",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearMatMul",
            "QLinearAdd",
            "QLinearMul",
        ]:
            # align with zp dtype
            return op.inputs[2].meta.dtype
        raise NotImplementedError(
            f"Please implement dominant dtype extraction for {op.type}"
        )

    @classmethod
    def get_dtype_on_symmetricity(cls, is_asymmetrical: bool) -> torch.dtype:
        return (
            cls.ASYMMETRICAL_ZP_NP_TYPE
            if is_asymmetrical
            else cls.SYMMETRICAL_ZP_NP_TYPE
        )

    @classmethod
    def is_quantize_parameter_added(cls, var: Variable, graph: BaseGraph) -> bool:
        return var.name + cls.SCALE_PARAMETER_SUFFIX in graph.variables

    def is_quantized_qlinear_op(self, op: Operation) -> bool:
        return (
            op is not None
            and op.type in self.qlinear_op_map
            and isinstance(op, QuantableOperation)
        )

    def is_asymmetrical(self, config: TensorQuantizationConfig) -> bool:
        return config.policy.has_property(QuantizationProperty.ASYMMETRICAL)

    def is_per_channel(self, config: TensorQuantizationConfig) -> bool:
        return config.policy.has_property(QuantizationProperty.PER_CHANNEL)

    def build_per_channel_param_broadcast_shape(
        self, weight: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        prefix_count = 0
        suffix_count = 0
        while weight.shape[prefix_count] != param.shape[0]:
            prefix_count += 1
        suffix_count = len(weight.shape) - prefix_count - 1
        return param[(None,) * prefix_count + (...,) + (None,) * suffix_count]

    def quantize_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        is_asymmetrical: bool,
        is_per_channel: bool,
    ) -> torch.Tensor:
        weight_dtype = self.get_dtype_on_symmetricity(is_asymmetrical)
        if is_per_channel is True:
            unsqueezed_scale = self.build_per_channel_param_broadcast_shape(
                weight, scale
            )
            unsqueezed_zp = self.build_per_channel_param_broadcast_shape(
                weight, zero_point
            )
            return (
                ((weight / unsqueezed_scale).round() + unsqueezed_zp)
                .cpu()
                .to(weight_dtype)
            )
        else:
            return ((weight / scale).round() + zero_point).cpu().to(weight_dtype)

    def quantize_bias(
        self, bias: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        return (
            ((bias / scale).round() + zero_point).cpu().to(ORTOOSExporter.BIAS_NP_TYPE)
        )

    def add_scale_and_zp_parameter(
        self,
        var: Variable,
        graph: BaseGraph,
        dest_index: int,
        quantize_config: TensorQuantizationConfig,
        link_to_source=False,
    ) -> Tuple[Variable]:
        if self.is_quantize_parameter_added(var, graph):
            scale = graph.variables[var.name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX]
            offset = graph.variables[var.name + ORTOOSExporter.ZP_PARAMETER_SUFFIX]
            scale.dest_ops.append(var.dest_ops[dest_index])
            offset.dest_ops.append(var.dest_ops[dest_index])
            if link_to_source:
                scale.dest_ops.append(var.source_op)
                offset.dest_ops.append(var.source_op)
        else:
            is_asymmetrical = self.is_asymmetrical(quantize_config)
            offset_dtype = self.get_dtype_on_symmetricity(is_asymmetrical)
            scale, offset = quantize_config.scale, quantize_config.offset
            dest_ops = (
                [var.source_op, var.dest_ops[dest_index]]
                if link_to_source
                else [var.dest_ops[dest_index]]
            )
            scale = Variable(
                name=var.name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX,
                value=convert_any_to_torch_tensor(
                    scale, dtype=ORTOOSExporter.SCALE_NP_TYPE
                ),
                is_parameter=True,
                dest_ops=dest_ops,
            )
            offset = Variable(
                name=var.name + ORTOOSExporter.ZP_PARAMETER_SUFFIX,
                value=convert_any_to_torch_tensor(offset, dtype=offset_dtype),
                is_parameter=True,
                dest_ops=dest_ops,
            )
            graph.append_variable(scale)
            graph.append_variable(offset)
        return scale, offset

    def add_quantized_parameter(
        self,
        var: Variable,
        graph: BaseGraph,
        dest_index: int,
        quantize_config: TensorQuantizationConfig,
        quantize_suffix: str,
        is_bias: bool,
    ) -> Variable:
        scale, offset = quantize_config.scale, quantize_config.offset
        if is_bias:
            quant_value = self.quantize_bias(var.value, scale, offset)
        else:
            is_asymmetrical = self.is_asymmetrical(quantize_config)
            is_per_channel = self.is_per_channel(quantize_config)
            quant_value = self.quantize_weight(
                var.value, scale, offset, is_asymmetrical, is_per_channel
            )
        quant_val = Variable(
            name=var.name + quantize_suffix,
            value=quant_value,
            is_parameter=True,
            dest_ops=[var.dest_ops[dest_index]],
        )
        graph.append_variable(quant_val)
        return quant_val

    def add_quantize_linear_op_quant_parameter(
        self, graph: BaseGraph, var: Variable, index: int
    ) -> None:
        if self.is_quantize_parameter_added(var, graph):
            # quantization parameter would be shared by multiple operations
            graph.variables[
                var.name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
            ].dest_ops.append(var.dest_ops[index])
            graph.variables[
                var.name + ORTOOSExporter.ZP_PARAMETER_SUFFIX
            ].dest_ops.append(var.dest_ops[index])
        else:
            # add new quantization parameter
            quantize_config = var.source_op_config
            self.add_scale_and_zp_parameter(
                var, graph, index, quantize_config, link_to_source=True
            )

    def add_quant_parameter_for_op(
        self, graph: BaseGraph, var: Variable
    ) -> Tuple[Variable]:
        assert len(var.dest_ops) == 1
        quantize_config = var.dest_op_configs[0]
        scale_val, offset_val = self.add_scale_and_zp_parameter(
            var, graph, 0, quantize_config, link_to_source=False
        )
        quant_val = self.add_quantized_parameter(
            var,
            graph,
            0,
            quantize_config,
            ORTOOSExporter.QUANTIZE_PARAMETER_SUFFIX,
            is_bias=False,
        )
        return quant_val, scale_val, offset_val

    def add_quant_parameter_for_conv_op(
        self, graph: BaseGraph, var: Variable, is_bias: bool
    ) -> Tuple[Variable]:
        assert len(var.dest_ops) == 1
        quantize_config = var.dest_op_configs[0]
        if is_bias is True:
            quant_val = self.add_quantized_parameter(
                var,
                graph,
                0,
                quantize_config,
                ORTOOSExporter.BIAS_QUANTIZE_PARAMETER_SUFFIX
                + ORTOOSExporter.QUANTIZE_PARAMETER_SUFFIX,
                is_bias=True,
            )
        else:
            scale_val, offset_val = self.add_scale_and_zp_parameter(
                var, graph, 0, quantize_config, link_to_source=False
            )
            quant_val = self.add_quantized_parameter(
                var,
                graph,
                0,
                quantize_config,
                ORTOOSExporter.WEIGHT_QUANTIZE_PARAMETER_SUFFIX
                + ORTOOSExporter.QUANTIZE_PARAMETER_SUFFIX,
                is_bias=False,
            )
        if is_bias is False:
            return quant_val, scale_val, offset_val
        return quant_val

    def insert_quant_Linear_operation(
        self, graph: BaseGraph, var: Variable, index: int
    ) -> Operation:
        quantize_config = var.dest_op_configs[index]
        quant_op = Operation(
            name=var.name + ORTOOSExporter.QUANTIZE_LINEAR_SUFFIX,
            op_type="QuantizeLinear",
            attributes={},
        )
        graph.append_operation(quant_op)
        link_var = Variable(
            name=var.name + ORTOOSExporter.LINKER_VAR_SUFFIX,
            dest_ops=[],
            source_op=quant_op,
        )
        graph.append_variable(link_var)
        quant_op.inputs.append(var)
        quant_op.outputs.append(link_var)
        var.dest_ops[index] = quant_op
        scale_val, offset_val = self.add_scale_and_zp_parameter(
            var, graph, index, quantize_config, link_to_source=False
        )
        quant_op.inputs.extend([scale_val, offset_val])
        return quant_op

    def insert_dequant_Linear_operation(
        self, graph: BaseGraph, var: Variable, index: int, is_output_var=False
    ) -> Operation:
        quantize_config = var.source_op_config
        dequant_op = Operation(
            name=var.name + ORTOOSExporter.DEQUANTIZE_LINEAR_SUFFIX,
            op_type="DequantizeLinear",
            attributes={},
        )
        graph.append_operation(dequant_op)
        link_var = Variable(
            name=var.name + ORTOOSExporter.LINKER_VAR_SUFFIX,
            dest_ops=[],
            source_op=dequant_op,
        )
        graph.append_variable(link_var)
        dequant_op.inputs.append(var)
        dequant_op.outputs.append(link_var)
        # case when DequantizeLinear op required to be inserted before output
        if is_output_var:
            var.dest_ops.append(dequant_op)
        else:
            var.dest_ops[index] = dequant_op
        scale_val, offset_val = self.add_scale_and_zp_parameter(
            var,
            graph,
            len(var.dest_ops) - 1 if is_output_var else index,
            quantize_config,
            link_to_source=False,
        )
        dequant_op.inputs.extend([scale_val, offset_val])
        if is_output_var:
            del graph.outputs[var.name]
            graph.outputs[link_var.name] = link_var
        return dequant_op

    def transform_qlinear_conv(self, graph: BaseGraph, op: Operation) -> None:
        # Input scale
        input_val_name = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        input_scale, input_offset = (
            graph.variables[input_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[input_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        # Weight scale
        weight_val = op.inputs[1]
        (
            weight_quant,
            weight_scale,
            weight_offset,
        ) = self.add_quant_parameter_for_conv_op(graph, weight_val, False)
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_conv_inputs = [
            op.inputs[0],
            input_scale,
            input_offset,
            weight_quant,
            weight_scale,
            weight_offset,
            output_scale,
            output_offset,
        ]
        graph.delete_variable(op.inputs[1].name, True)
        # Bias
        if len(op.inputs) == 3:
            bias = op.inputs[2]
            bias_val = self.add_quant_parameter_for_conv_op(graph, bias, True)
            qlinear_conv_inputs.append(bias_val)
            graph.delete_variable(op.inputs[2].name, True)
        op.type = self.qlinear_op_map["Conv"]
        op.inputs.clear()
        op.inputs.extend(qlinear_conv_inputs)

    def transform_qlinear_linear_op(self, graph: BaseGraph, op: Operation) -> None:
        # Input scale 0
        input_val_name_0 = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        if op.inputs[0].is_parameter:
            input_0, input_scale_0, input_offset_0 = self.add_quant_parameter_for_op(
                graph, op.inputs[0]
            )
            graph.delete_variable(op.inputs[0].name, True)
        else:
            input_0 = op.inputs[0]
            input_scale_0, input_offset_0 = (
                graph.variables[
                    input_val_name_0 + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                ],
                graph.variables[input_val_name_0 + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
            )
        # Input scale 1
        input_val_name_1 = op.inputs[1].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        if op.inputs[1].is_parameter:
            input_1, input_scale_1, input_offset_1 = self.add_quant_parameter_for_op(
                graph, op.inputs[1]
            )
            graph.delete_variable(op.inputs[1].name, True)
        else:
            input_1 = op.inputs[1]
            input_scale_1, input_offset_1 = (
                graph.variables[
                    input_val_name_1 + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                ],
                graph.variables[input_val_name_1 + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
            )
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_inputs = [
            input_0,
            input_scale_0,
            input_offset_0,
            input_1,
            input_scale_1,
            input_offset_1,
            output_scale,
            output_offset,
        ]
        if op.type in ORT_MICROSOFT_CONTRIB_LINEAR_OPS:
            op.attributes["domain"] = "com.microsoft"
        op.type = self.qlinear_op_map[op.type]
        op.inputs.clear()
        op.inputs.extend(qlinear_inputs)

    def transform_qlinear_average_pool(
        self, graph: BaseGraph, op: Operation, is_global=False
    ) -> None:
        # Input scale
        input_val_name = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        input_scale, input_offset = (
            graph.variables[input_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[input_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_inputs = [
            op.inputs[0],
            input_scale,
            input_offset,
            output_scale,
            output_offset,
        ]
        op.type = self.qlinear_op_map[
            "GlobalAveragePool" if is_global is True else "AveragePool"
        ]
        op.attributes["domain"] = "com.microsoft"
        op.inputs.clear()
        op.inputs.extend(qlinear_inputs)

    def transform_qlinear_matmul(
        self, graph: BaseGraph, op: Operation, is_global=False
    ) -> None:
        # Input scale 0
        input_val_name_0 = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        if op.inputs[0].is_parameter:
            input_0, input_scale_0, input_offset_0 = self.add_quant_parameter_for_op(
                graph, op.inputs[0]
            )
            graph.delete_variable(op.inputs[0].name, True)
        else:
            input_0 = op.inputs[0]
            input_scale_0, input_offset_0 = (
                graph.variables[
                    input_val_name_0 + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                ],
                graph.variables[input_val_name_0 + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
            )
        # Input scale 1
        input_val_name_1 = op.inputs[1].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        if op.inputs[1].is_parameter:
            input_1, input_scale_1, input_offset_1 = self.add_quant_parameter_for_op(
                graph, op.inputs[1]
            )
            graph.delete_variable(op.inputs[1].name, True)
        else:
            input_1 = op.inputs[1]
            input_scale_1, input_offset_1 = (
                graph.variables[
                    input_val_name_1 + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                ],
                graph.variables[input_val_name_1 + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
            )
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_inputs = [
            input_0,
            input_scale_0,
            input_offset_0,
            input_1,
            input_scale_1,
            input_offset_1,
            output_scale,
            output_offset,
        ]
        op.type = self.qlinear_op_map["MatMul"]
        op.inputs.clear()
        op.inputs.extend(qlinear_inputs)

    def transform_qlinear_operation(
        self, operation: Operation, graph: BaseGraph
    ) -> None:
        if operation.type == "Conv":
            self.transform_qlinear_conv(graph, operation)
        if operation.type in ["Add", "Mul"]:
            self.transform_qlinear_linear_op(graph, operation)
        if operation.type == "GlobalAveragePool":
            self.transform_qlinear_average_pool(graph, operation, is_global=True)
        if operation.type == "AveragePool":
            self.transform_qlinear_average_pool(graph, operation, is_global=False)
        if operation.type == "MatMul":
            self.transform_qlinear_matmul(graph, operation, is_global=False)

    def correct_param_meta(self, graph: BaseGraph) -> None:
        # handle QLinear ops
        for op in graph.topological_sort():
            if op.type in self.qlinear_ops:
                curr_meta_len = len(op.meta_data.input_metas)
                expected_len = len(op.inputs)
                for _ in range(expected_len - curr_meta_len):
                    op.meta_data.input_metas.append(TensorMeta(None, None))
            if op.type == "QLinearAdd":
                # the second operand's index move from 1 to 3
                op.meta_data.input_metas[3] = op.meta_data.input_metas[1]

        # correct parameter meta data
        for var in graph.variables.values():
            if var.is_parameter:
                for op in var.dest_ops:
                    if op.meta_data is None:
                        op.meta_data = OperationMeta(
                            [TensorMeta(None, None, v.name) for v in op.inputs],
                            [TensorMeta(None, None, v.name) for v in op.outputs],
                            op.name,
                            op.type,
                            -1,
                        )

                    if torch.is_tensor(var.value):
                        new_input_meta = TensorMeta.parsing_from_torch_tensor(
                            var.value, var.name
                        )
                    else:
                        new_input_meta = TensorMeta.parsing_from_numpy_ndarray(
                            var.value, var.name
                        )

                    op.meta_data.input_metas[op.inputs.index(var)] = new_input_meta

        # add variable meta info in topo order
        for op in graph.topological_sort():
            if op.type == "QuantizeLinear" and op.inputs[0].source_op is not None:
                input_var = op.inputs[0]
                op.meta_data.input_metas[0] = input_var.meta
                op.meta_data.output_metas[0].shape = input_var.meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[2].dtype
            # must be input
            elif op.type == "QuantizeLinear" and op.inputs[0].value is None:
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

                op.meta_data.input_metas[0] = meta
                op.meta_data.output_metas[0].shape = meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[2].dtype
            elif op.type == "DequantizeLinear":
                input_var = op.inputs[0]
                op.meta_data.input_metas[0] = input_var.meta
                op.meta_data.output_metas[0].shape = input_var.meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[1].dtype
            elif op.type in self.qlinear_ops:
                for output_meta in op.meta_data.output_metas:
                    output_meta.dtype = self.get_qlinear_op_dominant_dtype(op)

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None) -> None:
        # remove switchers.
        if not EXPORT_DEVICE_SWITCHER:
            processer = GraphDeviceSwitcher(graph)
            processer.remove_switcher()

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            super().export_quantization_config(config_path, graph)

        # collect quantable vars, where we need to insert quant and dequant op
        # note that we assume all quantization configs of the same variable maintained
        # by different ops are actually the same
        quantable_vars, removed_activations = [], []
        for var in graph.variables.values():
            if isinstance(var, QuantableVariable):
                configs = [var.source_op_config] + var.dest_op_configs
                for cfg in configs:
                    if cfg is not None and not QuantizationStates.can_export(cfg.state):
                        raise AttributeError(
                            f"quantization state of variable {var.name} is unexpected, \
                        please check if you have finished the whole quantization process"
                        )
                    elif cfg is not None and cfg.state not in {
                        QuantizationStates.FP32,
                        QuantizationStates.SOI,
                    }:
                        quantable_vars.append(var)
                        break

        # Pass 1: remove activations
        for var in quantable_vars:
            if (
                var.is_parameter is False
                and var.source_op is not None
                and var.source_op.type in ORT_OOS_FUSE_START_OPS
                and len(var.dest_ops) == 1
                and var.dest_ops[0].type in self.removed_activation_types
            ):
                removed_activations.extend(var.dest_ops)
        self.remove_activation(graph, removed_activations)

        # Pass 2: insert QuantizeLinear & DequantizeLinear ops
        for var in quantable_vars:
            is_output_var = var.name in graph.outputs
            # removed activations
            if not var.dest_ops and is_output_var is False:
                continue
            if var.is_parameter is False:
                # add DequantizeLinear before output if necessary
                if (
                    var.source_op is not None
                    and var.source_op.type in self.qlinear_op_map
                    and is_output_var
                ):
                    self.insert_dequant_Linear_operation(graph, var, 0, True)
                else:
                    quant_op = None
                    dequant_op = None
                    pop_list = []
                    relink_node_pair = []
                    for index, dest_op in enumerate(var.dest_ops):
                        if self.is_quantized_qlinear_op(
                            var.source_op
                        ) and self.is_quantized_qlinear_op(dest_op):
                            self.add_quantize_linear_op_quant_parameter(
                                graph, var, index
                            )
                        elif var.source_op is not None and self.is_quantized_qlinear_op(
                            var.source_op
                        ):
                            old_index = var.dest_idx[index]
                            if dequant_op is None:
                                dequant_op = self.insert_dequant_Linear_operation(
                                    graph, var, index, False
                                )
                            dequant_op.outputs[0].dest_ops.append(dest_op)
                            relink_node_pair.append((dest_op, old_index, dequant_op))
                            if (
                                dequant_op in var.dest_ops
                                and var.dest_ops.index(dequant_op) != index
                            ):
                                pop_list.append(dest_op)
                        elif self.is_quantized_qlinear_op(dest_op):
                            old_index = var.dest_idx[index]
                            if quant_op is None:
                                quant_op = self.insert_quant_Linear_operation(
                                    graph, var, index
                                )
                            quant_op.outputs[0].dest_ops.append(dest_op)
                            relink_node_pair.append((dest_op, old_index, quant_op))
                            if (
                                quant_op in var.dest_ops
                                and var.dest_ops.index(quant_op) != index
                            ):
                                pop_list.append(dest_op)
                    # successors which now follows QuantizeLinear/DequantizeLinear ops
                    for dest_op in pop_list:
                        pop_index = var.dest_ops.index(dest_op)
                        var.dest_ops.pop(pop_index)
                    for node, index, quantize_node in relink_node_pair:
                        node.inputs[index] = quantize_node.outputs[0]

        # Pass 3 Transform QLinear ops
        for operation in graph.topological_sort():
            if self.is_quantized_qlinear_op(operation):
                self.transform_qlinear_operation(operation, graph)

        # Pass 4 collect meta info for parameters and newly-added variables
        self.correct_param_meta(graph)

        # Pass 5 transform other ops
        self.transform_op(graph)

        name = graph._name
        if not name:
            name = "PPL Quantization Tool - Onnx Export"

        # Ready to export onnx graph defination.
        _inputs, _outputs, _initilizers, _nodes, _value_infos = [], [], [], [], []
        for operation in graph.topological_sort():
            _nodes.append(self.export_operation(operation))

        for variable in graph.variables.values():
            tensor_proto = self.export_var(variable)
            if variable.name in graph.inputs:
                _inputs.append(tensor_proto)
            if variable.name in graph.outputs:
                _outputs.append(tensor_proto)
            if variable.is_parameter:
                _initilizers.append(tensor_proto)
            else:
                _value_infos.append(tensor_proto)

        graph_def = helper.make_graph(
            name=name,
            nodes=_nodes,
            inputs=_inputs,
            outputs=_outputs,
            initializer=_initilizers,
            value_info=_value_infos,
        )

        extra_opsets = self.required_opsets()

        if "opsets" in graph._detail:
            opsets = []
            for opset in graph._detail["opsets"]:
                # last condition shall be removed if we wanna run checker
                if opset["domain"] in extra_opsets or opset["domain"] == "":
                    continue
                op = onnx.OperatorSetIdProto()
                op.domain = opset["domain"]
                op.version = opset["version"]
                opsets.append(op)

        for key, value in extra_opsets.items():
            op = onnx.OperatorSetIdProto()
            op.domain = key
            op.version = value
            opsets.append(op)

        onnx_model = helper.make_model(
            graph_def, producer_name=PPQ_NAME, opset_imports=opsets
        )
        onnx_model.ir_version = 7
        # onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, file_path)
