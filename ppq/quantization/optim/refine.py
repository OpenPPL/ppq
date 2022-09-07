from typing import Iterable, List, Set

import torch
from ppq.core import (ALIGNMENT_MANUL_OVERRIDE, TYPES_FOR_ALIGNMENT,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform, TensorQuantizationConfig,
                      empty_ppq_cache, ppq_warning)
from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.IR.quantize import QuantableVariable
from ppq.IR.search import SearchableGraph
from ppq.quantization.observer.range import minmax_to_scale_offset

from .base import QuantizationOptimizationPass


class QuantizeSimplifyPass(QuantizationOptimizationPass):
    """
    ## PPQ Quantize Simplify Pass(通用量化精简过程)

    PPQ use Tensor Quantization Configuration(A data structure defined in ppq.core) to
    control quantization. Each quantable op will have a list of TQC as its quantization config,
    which contains necessary quantization parameter(scale, offset), in order to quantize its input(s) and output(s).

    While TQC is a powerful tool for describing quantization, it introduces some undiserible features:
    
    For a subgraph like:
    
            Relu1 - Relu2
    
    PPQ will create at least 4 TQC here, namely the input TQC of Relu1 and Relu2, and the output TQC of Relu1 and Relu2.
    Problem here is the output TQC of Relu1 and the input TQC of Relu2 is actually duplicated, the output variable
    should not be quantized twice.

    This Simplify Pass will detect all the duplicated TQCs in your network, disable them and create a link with their
    dominating TQCs. Disabled TQC will have and inactive state(QuantizationState.OVERRLAPED), so PPQ executor will 
    simply ignore them when executing.

    A duplicated TQC is an input TQC(A) whose binding variable has been quantized by another output TQC(B),
    and the input TQC(A) should have the same bit-width as the output TQC(B)
    
    ### Parameters:

    * No Parameter

    ### Usage
    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.fusion = True
        setting.fusion_setting.remove_useless_quantization = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)
    """
    def __init__(self) -> None:
        super().__init__(name='PPQ Quantize Simplify Pass')

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        for _, variable in graph.variables.items():
            assert isinstance(variable, Variable)
            source_op = variable.source_op

            if source_op is None: continue # input variables in network, they do not have a source
            if not isinstance(source_op, QuantableOperation): continue
            source_config = source_op.config.output_quantization_config[source_op.outputs.index(variable)]

            if source_config.state in {
                QuantizationStates.FP32,
                QuantizationStates.SOI,
                QuantizationStates.DEACTIVATED}:
                continue # if source config does not have a valid state, skip it.

            for downstream_op, dest_idx in zip(variable.dest_ops, variable.dest_idx):
                if downstream_op is None: continue # output variables in network, they do not have a destination
                if not isinstance(downstream_op, QuantableOperation): continue

                input_config = downstream_op.config.input_quantization_config[dest_idx]
                if source_op.platform == downstream_op.platform:
                    if input_config.state == QuantizationStates.INITIAL:
                        input_config.dominated_by = source_config


class QuantizeRefinePass(QuantizationOptimizationPass):
    """修复算子上的定点错误，主要针对 Onnx 的一些特殊算子，其部分输入需要定点，而部分输入不需要定点.

    例如对于 Reshape 算子而言，其存在 data, shape 两个输入，其中 shape 不需要定点
    因此 QuantizeRefinePass 会纠正 Reshape 算子的 Quantization config，避免错误地对 shape 输入进行量化。

        目前我们针对 'Reshape', 'Slice', 'Gather', 'Clip', 'Pad', 'Resize', 'Split' 算子进行了详细讨论
        修正了已知的所有量化行为错误

    对于所有平台的 Quantizer 而言，都应当调用 QuantizeRefinePass 修复上述量化行为错误

    customize quantization for special operators, more specifically, for certain op, some of inputs
    need quantization while some don't, this pass refines quantization behaviors of
    'Reshape', 'Slice', 'Gather', 'Clip', 'Pad', 'Resize', 'Split' ops

    this pass should be applied regardless of backend platforms
    """
    def __init__(self) -> None:
        super().__init__(name='PPQ Quantization Config Refine Pass')

    @ empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue

            if operation.type in {'Reshape', 'Slice', 'Gather', 'Clip', 'Pad', 'Resize', 'Split'}:

                if operation.type == 'Reshape':
                    # Inputs:
                    #   data (differentiable) : T
                    #       An input tensor.
                    #   shape (non-differentiable) : tensor(int64)
                    #       Specified shape for output.
                    # see also https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
                    assert len(operation.config.input_quantization_config) == 2, f'Reshape Operation {operation.name} should have exact 2 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    operation.config.input_quantization_config[-1].state = QuantizationStates.SOI
                    continue

                if operation.type == 'Slice':
                    # Inputs (3 - 5)
                    #   data (differentiable) : T
                    #       Tensor of data to extract slices from.
                    #   starts (non-differentiable) : Tind
                    #       1-D tensor of starting indices of corresponding axis in `axes`
                    #   ends (non-differentiable) : Tind
                    #       1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
                    #   axes (optional, non-differentiable) : Tind
                    #       1-D tensor of axes that `starts` and `ends` apply to. Negative value means
                    #       counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
                    #   steps (optional, non-differentiable) : Tind
                    #       1-D tensor of slice step of corresponding axis in `axes`.
                    #       Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.
                    # see also https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Slice-11
                    assert len(operation.config.input_quantization_config) in {3, 4, 5}, f'Reshape {operation.name} Operation should have 3 - 5 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    for config in  operation.config.input_quantization_config[1: ]:
                        config.state = QuantizationStates.SOI
                    continue

                if operation.type == 'Gather':
                    # Inputs
                    #   data (differentiable) : T
                    #       Tensor of rank r >= 1.
                    #   indices (non-differentiable) : Tind
                    #       Tensor of int32/int64 indices, of any rank q.
                    #       All index values are expected to be within bounds [-s, s-1] along axis of size s.
                    #       It is an error if any of the index values are out of bounds.
                    # see also https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Gather-11
                    assert len(operation.config.input_quantization_config) == 2, f'Gather Operation {operation.name} should have 2 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    operation.config.input_quantization_config[-1].state = QuantizationStates.SOI
                    continue

                if operation.type == 'Clip':
                    # Inputs (1 - 3)
                    #   input : T
                    #       Input tensor whose elements to be clipped
                    #   min (optional) : T
                    #       Minimum value, under which element is replaced by min.
                    #       It must be a scalar(tensor of empty shape).
                    #   max (optional) : T
                    #       Maximum value, above which element is replaced by max.
                    #       It must be a scalar(tensor of empty shape).
                    # see also https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Clip-11
                    assert len(operation.config.input_quantization_config) in {1, 2, 3}, f'Clip Operation {operation.name} should have 1 - 3 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    for config in  operation.config.input_quantization_config[1: ]:
                        config.state = QuantizationStates.FP32
                    continue

                if operation.type == 'Pad':
                    # Inputs (2 - 3)
                    #   data : T
                    # Input tensor.
                    #   pads : tensor(int64)
                    #       Tensor of integers indicating the number of padding elements to add or remove
                    #       (if negative) at the beginning and end of each axis.
                    #       For 2D input tensor, it is the number of pixels. `pads` should be a 1D tensor of shape [2 * input_rank].
                    #       `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
                    #        where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end,
                    #       the number of pad values added at the end of axis `i`.
                    #   constant_value (optional) : T
                    #       (Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0).
                    # https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Pad-11
                    assert len(operation.config.input_quantization_config) in {2, 3}, f'Pad Operation {operation.name} should have 2 - 3 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    operation.config.input_quantization_config[1].state = QuantizationStates.SOI
                    if len(operation.config.input_quantization_config) == 3:
                        operation.config.input_quantization_config[-1].state = QuantizationStates.PASSIVE_INIT
                    continue

                if operation.type == 'Resize':
                    # Inputs (3 - 4)
                    #   X : T1
                    #       N-D tensor
                    #   roi : T2
                    #       1-D tensor given as [start1, ..., startN, end1, ..., endN],
                    #       where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image.
                    #       It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
                    #   scales : tensor(float)
                    #       The scale array along each dimension.
                    #       It takes value greater than 0. If it's less than 1, it's sampling down,
                    #       otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'.
                    #       Only one of 'scales' and 'sizes' can be specified.
                    #       If 'size' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.
                    #   sizes (optional) : tensor(int64)
                    #       The size of the output tensor.
                    #       The number of elements of 'sizes' should be the same as the rank of input 'X'.
                    #       Only one of 'scales' and 'sizes' can be specified.
                    # https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Resize-11
                    assert len(operation.config.input_quantization_config) in {3, 4}, f'Resize Operation {operation.name} should have 3 - 4 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    for config in  operation.config.input_quantization_config[1: ]:
                        config.state = QuantizationStates.SOI
                    continue

                if operation.type == 'Split':
                    # Inputs (1 - 2)
                    #   input (differentiable) : T
                    #       The tensor to split
                    #   split (optional, non-differentiable) : tensor(int64) (opset 13)
                    #       Optional length of each output.
                    #       Values should be >= 0.Sum of the values must be equal to the dim value at 'axis' specified.
                    # see also: https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Split-11
                    # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split
                    assert len(operation.config.input_quantization_config) in {1, 2}, f'Split Operation {operation.name} should have 1 - 2 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph definition different from onnx opset 11?'
                    for config in  operation.config.input_quantization_config[1: ]:
                        config.state = QuantizationStates.SOI
                    continue


class NxpInputRoundingRefinePass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='PPQ Input Quantization Refine Pass')

    def optimize(self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        for variable in graph.variables.values():
            if isinstance(variable, QuantableVariable):
                if variable.source_op is None or not isinstance(variable.source_op, QuantableOperation):
                    for config in variable.dest_op_configs:
                        if config is None: continue
                        config.rounding = RoundingPolicy.ROUND_HALF_DOWN


class NxpQuantizeFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='PPQ Quantization Fusion Pass')

    @ empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        processor = SearchableGraph(graph)

        relu_fusion_matching = processor.activation_matching(
            start_op_types=['Conv', 'Add'], end_types=['Relu'])
        for conv_name, activation_names in relu_fusion_matching.items():
            conv = graph.operations[conv_name]
            if not isinstance(conv, QuantableOperation): continue
            if len(activation_names) == 1:
                activation = graph.operations[activation_names[0]]
                if not isinstance(activation, QuantableOperation): continue
                activation_cfg = activation.config.output_quantization_config[0]
                conv_cfg = conv.config.output_quantization_config[0]
                conv_cfg.dominated_by = activation_cfg
                conv_cfg.state = QuantizationStates.OVERLAPPED

        concat_fusion_matching = processor.concat_matching(
            relay_pattern=lambda x, y: False, end_pattern=lambda _: True)
        for concat_name, upstream_layer_collection in concat_fusion_matching.items():
            concat = graph.operations[concat_name]
            if not isinstance(concat, QuantableOperation): continue
            for upstream_layer_name in upstream_layer_collection:
                upstream_layer = graph.operations[upstream_layer_name]
                if not isinstance(upstream_layer, QuantableOperation): continue
                upstream_cfg = upstream_layer.config.output_quantization_config[0]
                concat_cfg = concat.config.output_quantization_config[0]
                upstream_cfg.dominated_by = concat_cfg
                upstream_cfg.state = QuantizationStates.OVERLAPPED


class QuantizeFusionPass(QuantizationOptimizationPass):
    """
    ## PPQ Quantize Fusion Pass(通用量化图融合过程)
    
    Operation fusion (or kernel/layer fusion) is key optimization in many state-of-the-art execution frameworks.

    Graph fusion can combine operations into a single op to obtain higher accuracy and performance,
        Pattern like: Conv + Relu can be reduced to ConvRelu. This fusion will reduce memory accesses,
        and the quantization point after conv can also be removed.

    Technically we can fuse those layers before quantization, while fused layers are not supported by onnx standard.
        So to say ConvRelu is not a valid onnx operation, no execution framework can parse it.

    Therefore, PPQ will simulate the graph fusion by adjusting quantization config: if PPQ finds their is a
        pattern like Conv + Relu, the output quantization of Conv will be disabled, pretending that the Conv + Relu 
        fusion has happened.

    This Pass is designed for 2 types graph fusion:
        1. activation fusion
        2. passive operation fusion

    For activation fusion, PPQ will identify the pattern: Computing op + Activation Op from your network. The output
        quantization of computing op will be disabled with their state being set to QuantizationState.OVERLAPPED.

    Activation fusion here supports only simple activation patterns,
        for complex activation functions like mish, swish, 
        will be represented as mish = tanh + mul + softplus, swish = sigmoid + mul in onnx,
        cause onnx does not have a op defination for them.
        Identifying those complex patterns requires pattern matching, which is implemented in ppq.IR.search.py

    Complex quantization fusions must be invoked manually, PPQ implemented softplus & swish fusion functions in
        ppq.quantization.optim.refine.MishFusionPass
        ppq.quantization.optim.refine.SwishFusionPass

    For passive operation fusion, PPQ will keep the input and the output variable share a same scale for passive operations.
        An operation is identified as passive op only if its attribute "is_active_quant_op" = False, this
        attribute is initialized by quantizer.

    If there is a passive operation having multiple input and output, the fusion procedure will make its
    FIRST input variable and ALL output variables share the same scale(the same scale as its first input).
    The quantization states of all output variables will be set to QuantizationState.OVERLAPPED.

    ### Parameters:

    * activation_type(Set[str]):
            
            A collection contains all activation types.

            The pattern will be recognized as [Computing Op -> Activation Op],

            By graph fusion, the output quantization of the Computing Op and 
                the input quantization of the activation op will be disabled.

    * fuse_activation(bool)

            Whether to fuse activation op with computing op.

    # fuse_passive_op(bool)

            Whether to fuse passive op so that the input variable and output variable will share a same scale.

    ### Usage
    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.fusion = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)
    """
    def __init__(self,
                 activation_type: Set[str],
                 fuse_activation: bool = True,
                 fuse_passive_op: bool = True) -> None:
        self.fuse_activation  = fuse_activation
        self.fuse_passive_op  = fuse_passive_op
        self.activation_types = activation_type
        super().__init__(name='PPQ Quantization Fusion Pass')

    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])

    @ empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        processor = SearchableGraph(graph)

        # fuse computing operations and its following activation.
        if self.fuse_activation:
            patterns = processor.pattern_matching(
                patterns=[lambda x: x.is_computing_op, lambda x: x.type in self.activation_types],
                edges=[[0, 1]], exclusive=True)

            for pattern in patterns:
                computing_op, act_op = pattern

                if (not isinstance(computing_op, QuantableOperation) or 
                    computing_op.platform in {
                        TargetPlatform.TRT_INT8, TargetPlatform.OPENVINO_INT8, 
                        TargetPlatform.NCNN_INT8}): 
                    continue

                if (computing_op.platform != act_op.platform and 
                    computing_op.config.output_quantization_config[0].state != QuantizationStates.FP32):
                    ppq_warning(f'Unexpected dispatching was found: '
                                f'Op {computing_op.name} and {act_op.name} should be send to a same platform.')
                    continue

                if not isinstance(act_op, QuantableOperation):
                    ppq_warning(f'Unexpected dispatching was found: '
                                f'Op {computing_op.name} and {act_op.name} should both be quantized operation.')
                    continue

                assert isinstance(act_op, QuantableOperation)
                if (len(graph.get_downstream_operations(computing_op)) == 1 and 
                    len(graph.get_upstream_operations(act_op)) == 1):
                    computing_op.config.output_quantization_config[0].dominated_by = (
                        act_op.config.output_quantization_config[0])
                    act_op.config.input_quantization_config[0].dominated_by = (
                        act_op.config.output_quantization_config[0])

            # fuse relu and clip if possible
            for op in graph.operations.values():
                if op.type in {'Relu', 'Clip'}:
                    upstream_op = op.inputs[0].source_op
                    if not isinstance(op, QuantableOperation): continue
                    if upstream_op is None: continue
                    if upstream_op.platform != op.platform: continue
                    if not isinstance(upstream_op, QuantableOperation): continue
                    if len(graph.get_downstream_operations(upstream_op)) != 1: continue
                    upstream_op.config.output_quantization_config[0].dominated_by = (
                        op.config.output_quantization_config[0])
                    op.config.input_quantization_config[0].dominated_by = (
                        op.config.output_quantization_config[0])

        if self.fuse_passive_op:
            # all passive operations should never changes quantization configuration of its input
            # so to say their input and output share a same scale.
            for op in graph.operations.values():
                upstream_layers = graph.get_upstream_operations(op)
                if len(upstream_layers) == 0: continue # beginning op, can not merge.
                if (isinstance(op, QuantableOperation) and
                    not op.config.is_active_quant_op and
                    self.is_same_platform(upstream_layers + [op])):
                    # There are many types of passive operations.
                    # 'Resize', 'MaxPool', 'GlobalMaxPool',
                    # 'Slice', 'Pad', 'Split'

                    # Their first input variable should be data.
                    input_cfg = op.config.input_quantization_config[0]
                    for output_cfg in op.config.output_quantization_config:
                        output_cfg.dominated_by = input_cfg


class QuantAlignmentPass(QuantizationOptimizationPass):
    """
    ## PPQ Quant Alignment Pass(通用量化对齐过程)
    
    When deploy on real hardware and inference framework, 
        we will find that there are various restrictions or rules that we have to follow.

    * AVERAGE_POOL_2D: Input and outputs must all have same scale/zero_point
    
    * CONCATENATION: Input and outputs must all have same scale/zero_point
    
    * SLICE: Input and outputs must all have same scale/zero_point
    
    More detailed restrictions please refer to: https://www.tensorflow.org/lite/performance/quantization_spec

    Those restrictions, can be concluded as some quantization should share 
        the same quantization parameter with others. PPQ Quant Alignment Pass is designed
        for dealing with problems like this.

    PPQ uses Tensor Quantization Config (A data structure defined in ppq.core) to control the
        quantization logic, so to say if we want to align quantization parameters, we align
        their TQC in fact.

    The way to align TQC is simple, code like:
        tqc1.set_master(master=tqc2)
    Will make tqc1 and tqc2 share the same quantization parameters as tqc1 has, and change the
    state of tqc2 to be QuantizationState.SLAVE

    If we access the scale of tqc2, PPQ will return its master TQC's scale instead, so does offset.

    That is tqc1 and tqc2 are bonuded with statement "tqc1.set_master(master=tqc2)".

    ### Parameters:

    * elementwise_merge_method(Set[str]):
            
            Alignment method for elementwise ops.
            
            All elementwise ops are listed in ppq.core.common.py

    * concat_merge_method(bool)

            Alignment method for concat-like ops.
            
            All concat-like ops are listed in ppq.core.common.py

    # averagepool_method(bool)

            Alignment method for pooling-like ops.
            
            All pooling-like ops are listed in ppq.core.common.py

    # force_overlap(bool)

            TQC alignment might cause serious cascade effect.
            
            For subgraph like this:
            
            Conv1 ---     
                    + --- Add1
            Conv2 ---
                    + --- Conv3

            If we demand Add1 to have same input scale, this alignment will affect Conv3 also,
                cause Conv2's output is feed to both Add1 and Conv3.
            
            If force_overlap = False, PPQ alignment procedure will remain the output scale of
                Conv2 as unchanged, while only align the input scale of Add1.
            
            If force_overlap = True, the input of Add1, Conv3 and the output of Conv2 will all
                be aligned to a same scale.

    ### Usage
    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.fusion = True
        setting.fusion_setting.force_alignment_overlap = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)
    """
    def __init__(self,
                 elementwise_merge_method: str = 'Align to Large',
                 concat_merge_method: str = 'Align to Output',
                 averagepool_method: str = 'None',
                 force_overlap: bool = False) -> None:
        self.averagepool_method       = averagepool_method
        self.elementwise_merge_method = elementwise_merge_method
        self.concat_merge_method      = concat_merge_method
        self.force_overlap            = force_overlap
        assert self.elementwise_merge_method in {'Align to Large', 'Align to Output', 'None'}, (
            'elementwise_merge_method can only be (None), (Align to Large) or (Align to Output)')
        assert self.concat_merge_method in {'Align to Large', 'Align to Output', 'None'}, (
            'concat_merge_method can only be (None), (Align to Large) or (Align to Output)')
        assert self.averagepool_method in {'Align to Output', 'None'}, (
            'concat_merge_method can only be (None) or (Align to Output)')
        super().__init__(name='PPQ Quantization Alignment Pass')

    def align_to_large(self, op: QuantableOperation) -> TensorQuantizationConfig:
        """Align quant scale and offset to larger input config. The first input
        config will be set as master config, all slave config will share the
        same scale and offset with master.

        Any change to slave config will be rejected since then.
        """
        global_min, global_max, master_config = 0, 0, op.config.input_quantization_config[0]
        for config in op.config.input_quantization_config:
            assert config.policy.has_property(QuantizationProperty.PER_TENSOR), (
                'Quant Alignment can only happen with per tensor quantization.')
            local_min = config.scale * (config.quant_min - config.offset)
            local_max = config.scale * (config.quant_max - config.offset)

            assert isinstance(local_min, torch.Tensor)
            assert isinstance(local_max, torch.Tensor)
            global_max = max(global_max, local_max.item())
            global_min = min(global_min, local_min.item())

        # recompute scale and offset
        scale, offset = minmax_to_scale_offset(
            global_min, global_max, op.config.input_quantization_config[0])

        device = master_config.scale.device
        master_config._father_config = master_config
        master_config.state  = QuantizationStates.ACTIVATED
        master_config.scale  = torch.tensor(scale, dtype=torch.float32, device=device)
        master_config.offset = torch.tensor(offset, dtype=torch.float32, device=device)

        for slave_config in op.config.input_quantization_config[1: ]:
            slave_config.set_master(master=master_config)

        return master_config

    def align_to_output(self, op: QuantableOperation) -> TensorQuantizationConfig:
        """Align quant scale and offset to output config. All input configs
        would share a same scale and offset with output config. (as a slave to
        output config)

        Any change to slave config will be rejected since then.
        """
        master_config = op.config.output_quantization_config[0]
        for slave_config in op.config.input_quantization_config:
            slave_config.set_master(master=master_config)
        return master_config

    def optimize(
        self, graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor, **kwargs) -> None:

        for operation in graph.operations.values():
            if not isinstance(operation, QuantableOperation): continue

            master_config = None
            if operation.type in TYPES_FOR_ALIGNMENT['Elementwise']:
                if self.elementwise_merge_method == 'None': continue
                if self.elementwise_merge_method == 'Align to Large':
                    master_config = self.align_to_large(operation)
                else: master_config = self.align_to_output(operation)

            elif operation.type in TYPES_FOR_ALIGNMENT['Concat']:
                if self.concat_merge_method == 'None': continue
                if self.concat_merge_method == 'Align to Large':
                    master_config = self.align_to_large(operation)
                else: master_config = self.align_to_output(operation)

            elif operation.type in TYPES_FOR_ALIGNMENT['Pooling']:
                if self.averagepool_method == 'None': continue
                if self.averagepool_method == 'Align to Output':
                    master_config = self.align_to_output(operation)
                if self.averagepool_method == 'Align to Large':
                    raise ValueError('Alignment Method Error, Pooling Op can not align to lager input.')

            elif ALIGNMENT_MANUL_OVERRIDE in operation.extension_attrib:
                method = operation.extension_attrib[ALIGNMENT_MANUL_OVERRIDE]
                if self.concat_merge_method == 'Align to Large':
                    master_config = self.align_to_large(operation)
                elif self.concat_merge_method == 'Align to Large': 
                    master_config = self.align_to_output(operation)
                else:
                    ppq_warning(f'Unrecognized Alignment Method {method} for operation {operation.name}')

            if master_config is not None:
                # override up stream layer's config if possible
                for up_op in graph.get_upstream_operations(operation):
                    if not isinstance(up_op, QuantableOperation): continue

                    if self.force_overlap:
                        for cfg, var in up_op.config_with_variable:
                            if operation in var.dest_ops:
                                cfg.set_master(master=master_config, recursive=True)
                    else:
                        if len(graph.get_downstream_operations(up_op)) != 1: continue
                        for cfg, var in up_op.config_with_variable:
                            if operation in var.dest_ops:
                                cfg.set_master(master=master_config, recursive=False)


class SwishFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Swish Fusion')

    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(
            patterns = [lambda x: x.is_computing_op, 'Sigmoid', 'Mul'],
            edges = [[0, 1], [1, 2], [0, 2]],
            exclusive = True)

        for pattern in patterns:
            if any([not isinstance(op, QuantableOperation) for op in pattern]):
                ppq_warning(f'There is a pattern of swish activation in your network start from {pattern[0]}, '
                            'however part of your swish activation is not quantable, '
                            'so that graph fusion can not merge their quantization configuration.')
                continue
            if any([op.platform != pattern[0].platform for op in pattern]):
                ppq_warning(f'There is a pattern of swish activation in your network start from {pattern[0]}, '
                            'however part of your swish activation is not quantable, '
                            'so that graph fusion can not merge their quantization configuration.')
                continue
            computing, sigmoid, mul = pattern

            assert isinstance(computing, QuantableOperation)
            assert isinstance(sigmoid, QuantableOperation)
            assert isinstance(mul, QuantableOperation)

            master_config = mul.config.output_quantization_config[0]
            computing.config.output_quantization_config[0].dominated_by = master_config
            sigmoid.config.input_quantization_config[0].dominated_by    = master_config
            sigmoid.config.output_quantization_config[0].dominated_by   = master_config
            mul.config.input_quantization_config[0].dominated_by        = master_config
            mul.config.input_quantization_config[1].dominated_by        = master_config


class MishFusionPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__('Mish Fusion')

    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        search_engine = SearchableGraph(graph)
        patterns = search_engine.pattern_matching(
            patterns = [lambda x: x.is_computing_op, 'Softplus', 'Tanh', 'Mul'],
            edges = [[0, 1], [1, 2], [2, 3], [0, 3]],
            exclusive = True)

        for pattern in patterns:
            if any([not isinstance(op, QuantableOperation) for op in pattern]):
                ppq_warning(f'There is a pattern of mish activation in your network start from {pattern[0]}, '
                            'however part of your mish activation is not quantable, '
                            'so that graph fusion can not merge their quantization configuration.')
                continue
            if any([op.platform != pattern[0].platform for op in pattern]):
                ppq_warning(f'There is a pattern of mish activation in your network start from {pattern[0]}, '
                            'however part of your mish activation is not quantable, '
                            'so that graph fusion can not merge their quantization configuration.')
                continue
            computing, softplus, tanh, mul = pattern

            assert isinstance(computing, QuantableOperation)
            assert isinstance(softplus, QuantableOperation)
            assert isinstance(tanh, QuantableOperation)
            assert isinstance(mul, QuantableOperation)

            master_config = mul.config.output_quantization_config[0]
            computing.config.output_quantization_config[0].dominated_by = master_config
            tanh.config.input_quantization_config[0].dominated_by       = master_config
            tanh.config.output_quantization_config[0].dominated_by      = master_config
            softplus.config.input_quantization_config[0].dominated_by   = master_config
            softplus.config.output_quantization_config[0].dominated_by  = master_config
            mul.config.input_quantization_config[0].dominated_by        = master_config
            mul.config.input_quantization_config[1].dominated_by        = master_config
