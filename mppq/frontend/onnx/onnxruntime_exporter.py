import os
from typing import Optional, Sequence

import torch

from mppq.data import convert_any_to_tensor
from mppq.frontend.onnx.onnx_exporter import OnnxExporter
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.opdef import Operation, Variable
from mppq.ir.base.quantize import QuantableOperation
from mppq.logger import warning
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    QuantVisibility,
    TensorQuantizationConfig,
)
from mppq.utils.qfunction import ppq_quant_toint, ppq_quant_tofloat
from mppq.utils.round import ppq_tensor_round


class QDQHelper:
    """Helper class for processing onnx qdq format"""

    @staticmethod
    def TQC_Exportable_Check(
        TQC: TensorQuantizationConfig, bounded_var: Variable
    ) -> bool:
        if not TQC.can_export():
            return False

        if TQC.visibility == QuantVisibility.INTERNAL:
            return False
        if TQC.num_of_bits == 8 and TQC.policy.has_property(
            QuantizationProperty.LINEAR
        ):
            if TQC.policy.has_property(QuantizationProperty.ASYMMETRIC):
                range_check = TQC.quant_max <= 255 and TQC.quant_min >= 0
            else:
                range_check = TQC.quant_max <= 127 and TQC.quant_min >= -128
        else:
            range_check = True

        if not range_check:
            warning(
                f"Is it not safe to export TQC({bounded_var.name}) to Onnx, "
                f"INT8 value range must be [-128, 127] or [0, 255], "
                f"however [{TQC.quant_min, TQC.quant_max}] was given."
            )
            return False
        return True


class ONNXRUNTIMExporter(OnnxExporter):
    r"""
    PPQ 可以将 TQC 中的量化信息导出为 Onnx QDQ 节点。

    对于 INT8 量化，PPQ 可以导出符合 Onnx 量化要求的量化信息，这要求

        * TQC.quant_min   = -128
        * TQC.quant_max   = 127
        * TQC.num_of_bits = 8

    不符合规范的量化信息将不会被导出，并且 PPQ 将给出警告。

    Onnx QDQ 格式具有较高的灵活性，因此 PPQ 与推理框架可能在一些细节格式问题上出现分歧。
    例如对于节点 A 而言，假设它有两个下游节点 B, C。此时 QDQ 节点可以插入在 A 的输出端，也可以分别插入在 B, C 的输入端。
    因此导出模型的格式可能存在不一致，这也有可能导致推理框架解析模型时发生错误。出现上述情况时，用户需要手动对导出逻辑进行干预。

    对于激活值非对称量化的策略而言，激活函数 relu, clip 的功能可以被 QDQ 节点代替，因此通常情况下它们是可省的。
    PPQ 会自动探测上述情况，在确保省略激活函数不会对网络结果产生影响时移除非对称量化中的激活函数。

    PPQ 会移除冗余的 QDQ 节点，即前后相邻的两组 Q-DQ 节点，如果它们的量化参数一致，则只会导出上游节点的量化信息。

    对于浮点量化而言，PPQ 会导出自定义节点 QuantizeFloating, DequantizeFloating， 这些节点不被推理框架识别，仅作可视化用途。

    任何导出器的导出逻辑都是原地进行的，它们将对传入的计算图对象进行原地修改，因此在导出之前你需要手动克隆计算图。
    """

    def __init__(
        self, removed_activation_types: Sequence[str] = ("Relu", "Clip")
    ) -> None:
        super().__init__()
        self.removed_activation_types = tuple(removed_activation_types)

    def infer_qtype(self, config: TensorQuantizationConfig):
        offset_dtype, value_dtype = torch.int8, torch.int8
        if config.policy.has_property(QuantizationProperty.ASYMMETRIC):
            offset_dtype = torch.uint8
            value_dtype = torch.uint8
        if config.num_of_bits > 8:
            offset_dtype = torch.int32
            value_dtype = torch.int32
        if config.exponent_bits == 4:
            offset_dtype = torch.float8_e4m3fn
            value_dtype = torch.float8_e4m3fn
        if config.exponent_bits == 5:
            offset_dtype = torch.float8_e5m2
            value_dtype = torch.float8_e5m2
        return offset_dtype, value_dtype

    def insert_quantize_node(
        self,
        graph: BaseGraph,
        var: Variable,
        config: TensorQuantizationConfig,
        op: Operation,
    ) -> Operation:
        """Insert a Quantize Node on given variable, according to given
        TensorQuantizationConfig. There is two basic type of Quantize
        Node: QuantizeLinear and QuantizeFloating.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            # Following code will export Linear Quantization Config
            # That is for FP32 -> INT
            offset_dtype, value_type = self.infer_qtype(config)
            scale = convert_any_to_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

            created = graph.create_operation(op_type="QuantizeLinear", attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes["axis"] = config.channel_axis
            else:
                created.attributes["axis"] = None

            if var in op.inputs:
                graph.insert_op_before(created, op, input_idx=op.inputs.index(var))
            elif var in op.outputs:
                graph.insert_op_after(created, op, output_idx=op.outputs.index(var))
            else:
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            graph.create_variable(
                name=None, value=scale, is_parameter=True, dest_ops=[created]
            )
            graph.create_variable(
                name=None, value=offset, is_parameter=True, dest_ops=[created]
            )

            created.outputs[0].dtype = value_type
            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape
            return created

        elif config.policy.has_property(QuantizationProperty.FLOATING):
            # Following code will export Linear Quantization Config
            # That is for FP32 -> FP8
            offset_dtype, value_type = self.infer_qtype(config)
            scale = convert_any_to_tensor(config.scale.clone(), dtype=torch.float32)
            offset = convert_any_to_tensor(config.offset.clone(), dtype=offset_dtype)

            created = graph.create_operation(
                op_type="QuantizeLinear",
                # attributes={
                #     "min": config.quant_min,
                #     "max": config.quant_max,
                #     "exponent": config.exponent_bits,
                #     "mantissa": config.mantissa_bits,
                # },
            )

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes["axis"] = config.channel_axis
            else:
                created.attributes["axis"] = None

            if var in op.inputs:
                graph.insert_op_before(created, op, input_idx=op.inputs.index(var))
            elif var in op.outputs:
                graph.insert_op_after(created, op, output_idx=op.outputs.index(var))
            else:
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            graph.create_variable(
                name=None, value=scale, is_parameter=True, dest_ops=[created]
            )
            # graph.create_variable(
            #     name=None, value=offset, is_parameter=True, dest_ops=[created]
            # ) # zero_point is not used for FP8

            created.outputs[0].dtype = value_type
            created.outputs[0].shape = var.shape
            created.inputs[0].shape = var.shape
            return created

        else:
            raise TypeError(
                "PPQ Can not export quantization information with variable "
                f"{var.name}, Unexpected Quantization property."
            )

    def insert_dequantize_node(
        self,
        graph: BaseGraph,
        var: Variable,
        config: TensorQuantizationConfig,
        op: Operation,
    ) -> Operation:
        """Insert a DeQuantize Node on given variable, according to given
        TensorQuantizationConfig. There is two basic type of DeQuantize
        Node: DeQuantizeLinear and DeQuantizeFloating.
        """
        if config.policy.has_property(QuantizationProperty.LINEAR):
            offset_dtype, value_type = self.infer_qtype(config)
            scale = convert_any_to_tensor(config.scale.clone(), dtype=torch.float32)
            offset = ppq_tensor_round(config.offset.clone()).type(offset_dtype)

            created = graph.create_operation(op_type="DequantizeLinear", attributes={})
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes["axis"] = config.channel_axis
            else:
                created.attributes["axis"] = None

            if var in op.inputs:
                graph.insert_op_before(created, op, input_idx=op.inputs.index(var))
            elif var in op.outputs:
                graph.insert_op_after(created, op, output_idx=op.outputs.index(var))
            else:
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            graph.create_variable(
                name=None, value=scale, is_parameter=True, dest_ops=[created]
            )
            graph.create_variable(
                name=None, value=offset, is_parameter=True, dest_ops=[created]
            )

            created.inputs[0].dtype = value_type
            created.inputs[0].shape = var.shape
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = torch.float32

            return created

        elif config.policy.has_property(QuantizationProperty.FLOATING):
            offset_dtype, value_type = self.infer_qtype(config)
            scale = convert_any_to_tensor(config.scale.clone(), dtype=torch.float32)
            offset = convert_any_to_tensor(config.offset.clone(), dtype=offset_dtype)

            created = graph.create_operation(
                op_type="DequantizeLinear",
                # attributes={
                #     "min": config.quant_min,
                #     "max": config.quant_max,
                #     "exponent": config.exponent_bits,
                #     "mantissa": config.mantissa_bits,
                # },
            )

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                created.attributes["axis"] = config.channel_axis
            else:
                created.attributes["axis"] = None

            if var in op.inputs:
                graph.insert_op_before(created, op, input_idx=op.inputs.index(var))
            elif var in op.outputs:
                graph.insert_op_after(created, op, output_idx=op.outputs.index(var))
            else:
                raise ValueError(
                    f"Unexpected Error in Exporting Op {op.name}({op.type})."
                )

            graph.create_variable(
                name=None, value=scale, is_parameter=True, dest_ops=[created]
            )
            # graph.create_variable(
            #     name=None, value=offset, is_parameter=True, dest_ops=[created]
            # )

            created.inputs[0].shape = var.shape
            created.inputs[0].dtype = value_type
            created.outputs[0].shape = var.shape
            created.outputs[0].dtype = torch.float32

            return created
        else:
            raise TypeError(
                "PPQ Can not export quantization information with variable "
                f"{var.name}, Unexpected Quantization property."
            )

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
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation):
                continue
            if op.type in {"Relu", "Clip"}:
                config = op.config.output_quantization_config[0]
                # Only ASYMMETRICAL quantized activations can be safely removed.
                if config.policy.has_property(QuantizationProperty.SYMMETRIC):
                    continue

                if not isinstance(config.scale, torch.Tensor):
                    continue
                if not isinstance(config.offset, torch.Tensor):
                    continue

                range_min = (
                    (config.scale * (config.quant_min - config.offset)).min().item()
                )
                range_max = (
                    (config.scale * (config.quant_max - config.offset)).max().item()
                )

                if op.type == "Relu":
                    if range_min >= 0:
                        removed_activations.append(op)

                if op.type == "Clip":
                    if op.num_of_input == 3:
                        clip_min = op.inputs[1].value
                        clip_max = op.inputs[2].value
                        if clip_min is not None:
                            clip_min = clip_min.item()
                        else:
                            clip_min = float("-inf")
                        if clip_max is not None:
                            clip_max = clip_max.item()
                        else:
                            clip_max = float("+inf")

                        if range_min >= clip_min and range_max <= clip_max:
                            removed_activations.append(op)

        # Activation op can only be relu and clip,
        # so it is safe to access op.inputs[0], op.outputs[0] as their input and output.
        for op in removed_activations:
            if not isinstance(op, QuantableOperation):
                continue
            if len(graph.get_upstream_operations(op)) == 0:
                continue
            config = op.config.output_quantization_config[0]

            upstream_op = graph.get_upstream_operations(op)[0]
            if not isinstance(upstream_op, QuantableOperation):
                continue
            if len(graph.get_downstream_operations(upstream_op)) != 1:
                continue
            input_var, input_cfg = op.inputs[0], op.config.input_quantization_config[0]
            if not input_cfg.policy.has_property(QuantizationProperty.ASYMMETRIC):
                continue

            # PATCH 20220304 Removing graph output op might cause error.
            if op.outputs[0].name in graph.outputs:
                graph.outputs.pop(op.outputs[0].name)
                graph.outputs[input_var.name] = input_var

            input_var, output_var = op.inputs[0], op.outputs[0]
            # Patch 20230103:
            # If var.source_op is DequantizeLinear, then we do not need to quantize it
            # twice.
            if input_var.source_op is not None and input_var.source_op.type in {
                "DequantizeLinear",
                "DequantizeFloating",
            }:
                assert (
                    input_var.source_op.num_of_input == 3
                ), "Quantize Node Format Error, need as least 3 inputs."
                assert isinstance(input_var.source_op, Operation)
                scale, offset = (
                    input_var.source_op.inputs[1].value,
                    input_var.source_op.inputs[2].value,
                )

                scale_diff = torch.max(torch.abs(scale - config.scale)).item()
                zeropoint_diff = torch.max(torch.abs(offset - config.offset)).item()
                if scale_diff < 1e-4 and zeropoint_diff < 1e-1:
                    continue

            if len(output_var.dest_ops) == 1 and output_var.dest_ops[0].type in {
                "QuantizeLinear",
                "QuantizeFloating",
            }:
                assert (
                    output_var.dest_ops[0].num_of_input == 3
                ), "Quantize Node Format Error, need as least 3 inputs."
                assert isinstance(output_var.dest_ops[0], Operation)
                scale, offset = (
                    output_var.dest_ops[0].inputs[1].value,
                    output_var.dest_ops[0].inputs[2].value,
                )

                scale_diff = torch.max(torch.abs(scale - config.scale)).item()
                zeropoint_diff = torch.max(torch.abs(offset - config.offset)).item()
                if scale_diff < 1e-4 and zeropoint_diff < 1e-1:
                    continue

            # insert quant & dequant op on var
            created = self.insert_quantize_node(
                graph=graph, var=input_var, config=config, op=op
            )
            self.insert_dequantize_node(
                graph=graph, var=created.outputs[0], config=config, op=created
            )

        for op in removed_activations:
            graph.remove_operation(op, keep_coherence=True)

        return graph

    def convert_operation(
        self, graph: BaseGraph, op: QuantableOperation, quantized_param: bool
    ):
        """Convert an operation to onnx quant & dequant format by inserting
        necessary quant & dequant op around it. There are 2 ways to represent
        quantized ONNX models:

        Operator Oriented. All the quantized operators have their own ONNX definitions,
            like QLinearConv, MatMulInteger and etc.

        Tensor Oriented, aka Quantize and DeQuantize (QDQ).
            This format uses DQ(Q(tensor)) to simulate the quantize and dequantize
            process, and QuantizeLinear and DeQuantizeLinear operators also carry
            the quantization parameters.

        Quantization-Aware training (QAT) models converted from Tensorflow or exported
        from PyTorch.

        Quantized models converted from tflite and other framework.

        Args:
            graph (BaseGraph): PPQ IR
            op (Operation): Converting op
            process_activation (bool): Converting op's activation
            process_parameter (bool): Converting op's parameter
            quantized_param (bool): Export parameter in quantized format.
        """
        # collect quantable vars, where we need to insert quant and dequant op
        for config, var in [_ for _ in op.config_with_variable]:
            inserting, inserting_var = op, var
            if not QDQHelper.TQC_Exportable_Check(TQC=config, bounded_var=var):
                continue

            if var.is_parameter:
                assert len(var.dest_ops) == 1, (
                    f"Can not export variable {var.name}, cause it has more than 1 "
                    "destination operations. PPQ require all parameters to have only "
                    "1 destination operation."
                )

                # override quantization state, so that we can export parameter.
                if config.state == QuantizationStates.BAKED:
                    config.state = QuantizationStates.ACTIVATED
                if config.state == QuantizationStates.PASSIVE_BAKED:
                    config.state = QuantizationStates.PASSIVE

                # if not quant parameter to int, all parameter should export as fp32.
                # needs insert both quant and dequant op for them
                if not quantized_param:
                    created = self.insert_quantize_node(
                        graph=graph, var=inserting_var, config=config, op=inserting
                    )
                    inserting_var = created.outputs[0]
                    inserting = created

                self.insert_dequantize_node(
                    graph=graph, var=inserting_var, config=config, op=inserting
                )

                if quantized_param and config.policy.has_property(
                    QuantizationProperty.LINEAR
                ):
                    var.value = ppq_quant_toint(tensor=var.value, config=config)

                if quantized_param and config.policy.has_property(
                    QuantizationProperty.FLOATING
                ):
                    var.value = ppq_quant_tofloat(tensor=var.value, config=config)

            elif not var.is_parameter:

                # Patch 20230103:
                # If var.source_op is DequantizeLinear, then we do not need to quantize
                # it twice.
                if var.source_op is not None and var.source_op.type in {
                    "DequantizeLinear",
                    "DequantizeFloating",
                }:
                    assert (
                        var.source_op.num_of_input == 3
                    ), "Quantize Node Format Error, need as least 3 inputs."
                    assert isinstance(var.source_op, Operation)
                    scale, offset = (
                        var.source_op.inputs[1].value,
                        var.source_op.inputs[2].value,
                    )

                    scale_diff = torch.max(torch.abs(scale - config.scale)).item()
                    zeropoint_diff = torch.max(torch.abs(offset - config.offset)).item()
                    if scale_diff < 1e-4 and zeropoint_diff < 1e-1:
                        continue

                if len(var.dest_ops) == 1 and var.dest_ops[0].type in {
                    "QuantizeLinear",
                    "QuantizeFloating",
                }:
                    assert (
                        var.dest_ops[0].num_of_input == 3
                    ), "Quantize Node Format Error, need as least 3 inputs."
                    assert isinstance(var.dest_ops[0], Operation)
                    scale, offset = (
                        var.dest_ops[0].inputs[1].value,
                        var.dest_ops[0].inputs[2].value,
                    )

                    scale_diff = torch.max(torch.abs(scale - config.scale)).item()
                    zeropoint_diff = torch.max(torch.abs(offset - config.offset)).item()
                    if scale_diff < 1e-4 and zeropoint_diff < 1e-1:
                        continue

                created = self.insert_quantize_node(
                    graph=graph, var=inserting_var, config=config, op=inserting
                )
                inserting_var = created.outputs[0]
                inserting = created

                self.insert_dequantize_node(
                    graph=graph, var=inserting_var, config=config, op=inserting
                )

    def prepare_graph(
        self,
        graph: BaseGraph,
        remove_activation_fn: bool = True,
        quant_parameter_to_int: bool = True,
        **kwargs,
    ) -> BaseGraph:
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
        # mark quantable variables
        for op in graph.topological_sort():
            if not isinstance(op, QuantableOperation):
                continue
            if op.type in {
                "QuantizeLinear",
                "DequantizeLinear",
                "QuantizeFloating",
                "DequantizeFloating",
            }:
                continue
            self.convert_operation(
                graph=graph, op=op, quantized_param=quant_parameter_to_int
            )

        # remove activations
        if remove_activation_fn:
            # remove useless activation.
            self.remove_activation_ops(graph)

        return graph

    def export(
        self,
        file_path: str | os.PathLike,
        graph: BaseGraph,
        config_path: Optional[str | os.PathLike] = None,
        save_as_external_data: bool = False,
        quantized_param: bool = True,
        remove_activation: bool = True,
        **kwargs,
    ) -> None:
        """
        Export PPQ Graph to Onnx QDQ format.
            This function requires a set of parameters to configure onnx format.

        Args:
            file_path (str): Onnx file name.

            graph (BaseGraph): Exporting ppq graph.

            config_path (str, optional): config file is a json file that contains quant-related
                information, this file is require by TensorRT for initialize its quantization
                pipeline. If config_path = None, no json file will be created.

            export_QDQ_op (bool, optional): whether to export QDQ node in onnx model.

            quantized_param (bool, optional): export quantized parameter, if quantized_param = False,
                PPQ will export parameter in FP32 format.

            remove_activation (bool, optional): this option will remove activation op(Relu, Clip),
                requires ASYMMTRICAL quantizaiton.

            save_as_external_data (bool, optional): for model larger than 2GB,
                this option will split model into external param files.
        """
        graph = self.prepare_graph(
            graph, quantized_param=quantized_param, remove_activation=remove_activation
        )
        super().export(file_path, graph, config_path, save_as_external_data, **kwargs)
        # Check Graph
        unsupportable_quant_op = set()
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                for cfg, var in op.config_with_variable:
                    if not QDQHelper.TQC_Exportable_Check(TQC=cfg, bounded_var=var):
                        continue
                    if cfg.num_of_bits != 8 or cfg.policy.has_property(
                        QuantizationProperty.FLOATING
                    ):
                        unsupportable_quant_op.add(op)

        if len(unsupportable_quant_op) != 0:
            warning(
                "Exported Onnx Model is not executable, following Op has onnxruntime-unsupported quant policy:"
            )
            for op in unsupportable_quant_op:
                warning(f"{op.name} (bitwidth != 8)")
            warning(
                "For Generating onnxruntime-executable Model, use TargetPlatform = Onnxruntime or OnnxruntimeQuantizer instead."
            )
