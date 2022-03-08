from collections import defaultdict
from typing import Iterable, List

import torch
from ppq.core import (COMPELING_OP_TYPES, LINEAR_ACTIVATIONS,
                      ORT_OOS_FUSE_START_OPS, PPLCUDA_ACTIVATIONS,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform, TensorQuantizationConfig,
                      empty_ppq_cache)
from ppq.executor import BaseGraphExecutor
from ppq.IR import GraphCommandProcesser, QuantableOperation, Variable
from ppq.IR.base.graph import Operation
from ppq.IR.quantize import QuantableVariable
from ppq.IR.search import SearchableGraph, TraversalCommand
from ppq.quantization.observer.range import minmax_to_scale_offset

from .base import QuantizationOptimizationPass


class QuantizeReducePass(QuantizationOptimizationPass):
    """
        QuantizeReducePass 用来简化量化定点信息:通常每一个 Quantable 算子都有前后两个定点信息，
        而运算时通常可以屏蔽一半定点信息以加速。QuantizeReducePass 被设计用来找出可以屏蔽的定点信息。

        对于两个相邻算子(op_1 -> op_2)而言，将会出现以下几种情况
            1. op_1 与 op_2 均不量化，此时无需对数据流进行额外处理
            2. op_1 量化，op_2 不量化，op_1 需要对结果进行量化
            3. op_1 不量化，op_2 量化，此时需要按 op_2 的量化参数对数据流进行量化
            4. op_1 与 op_2 均量化，此时分情况讨论:
                4.1. op_1 量化位宽高于 op_2，此时按 op_2 的量化参数对数据流进行量化
                4.2. op_1 量化位宽低于 op_2，此时按 op_1 的量化参数对数据流进行量化
                4.3. op_1 量化位等于 op_2，此时按 op_1 的量化参数对数据流进行量化
        
                                      ------> op_2
        对于更为复杂的网络结构 op_1 ----+ 
                                      ------> op_3

            op_1 如果有定点信息，则必须对数据流进行量化
            op_2, op_3 则需要分别确认是否需要再次对输入数据执行再次量化

        总结:
            当 下游节点 的量化位宽大于等于 上游节点 时，按 上游节点 的量化信息执行量化，此时量化操作发生在上游
            当 下游节点 的量化位宽小于 上游节点 时，按 下游节点 的量化信息执行量化，此时量化操作发生在下游（上游量化未必可以省略）
        
        QuantizeReducePass is used to reduce quantization fixation: we could block half of fixation points to accelerate
        the inference

        for 2 neighbouring ops(op_1 -> op_2), there are several situations:
            1. neither of op_1 and op_2 needs quantization
            2. op_1 needs quantization while op_2 doesn't
            3. op_2 needs quantization while op_1 does
            4. both need quantization:
                4.1. bit width of op_1 is larger than op_2, then we should use quantization parameters of op_2
                4.2. bit width of op_2 is larger than op_1, then we should use quantization parameters of op_1
                4.3. equal, we should use quantization parameters of op_1

        Conclusion:
            when the bit width of downstream op is larger or equal to that of upstream op, we should use quantization
            information of upstream op, otherwise we should use quantization information of downstream op(and the upstream
            quantization may not be omitted)

    """
    def __init__(self) -> None:
        super().__init__(name='PPQ Quantize Point Reduce Pass')

    def optimize(
        self, 
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:

        graph = processer.graph
        for _, varaible in graph.variables.items():
            assert isinstance(varaible, Variable)
            source_op = varaible.source_op

            if source_op is None: continue # input variables in network, they do not have a source
            if not isinstance(source_op, QuantableOperation): continue
            source_config = source_op.config.output_quantization_config[source_op.outputs.index(varaible)]

            for downstream_op, dest_idx in zip(varaible.dest_ops, varaible.dest_idx):
                if downstream_op is None: continue # output variables in network, they do not have a destination
                if not isinstance(downstream_op, QuantableOperation): continue
                
                input_config = downstream_op.config.input_quantization_config[dest_idx]
                if source_op.platform == downstream_op.platform:
                    if input_config.state == QuantizationStates.INITIAL:
                        input_config.dominated_by = source_config


class QuantizeRefinePass(QuantizationOptimizationPass):
    """
        修复算子上的定点错误，主要针对 Onnx 的一些特殊算子，其部分输入需要定点，而部分输入不需要定点

        例如对于 Reshape 算子而言，其存在 data, shape 两个输入，其中 shape 不需要定点
        因此 QuantizeRefinePass 会纠正 Reshape 算子的 Quantization config，避免错误地对 shape 输入进行量化。

            目前我们针对 'Reshape', 'Slice', 'Gather', 'Clip', 'Pad', 'Resize', 'Split' 算子进行了详细讨论
            修正了已知的所有量化行为错误
        
        对于所有平台的 Quantizer 而言，都应当调用 QuantizeRefinePass 修复上述量化行为错误

        customize quantization for special operators, more specificly, for certain op, some of inputs
        need quantization while some don't, this pass refines quantization behaviors of 
        'Reshape', 'Slice', 'Gather', 'Clip', 'Pad', 'Resize', 'Split' ops

        this pass should be applied regardless of backend platforms

    """
    def __init__(self) -> None:
        super().__init__(name='PPQ Quantization Config Refine Pass')

    @ empty_ppq_cache
    def optimize(
        self,
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:

        graph = processer.graph
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue
            
            if operation.type in {'Reshape', 'Slice', 'Gather', 'Clip', 'Pad', 'Resize', 'Split'}:

                if operation.type == 'Reshape':
                    # Inputs:
                    #   data (differentiable) : T
                    #       An input tensor.
                    #   shape (non-differentiable) : tensor(int64)
                    #       Specified shape for output.
                    # see aslo https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
                    assert len(operation.config.input_quantization_config) == 2, f'Reshape Operation {operation.name} should have exact 2 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
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
                    # see aslo https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Slice-11
                    assert len(operation.config.input_quantization_config) in {3, 4, 5}, f'Reshape {operation.name} Operation should have 3 - 5 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
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
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
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
                    # see aslo https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Clip-11
                    assert len(operation.config.input_quantization_config) in {1, 2, 3}, f'Clip Operation {operation.name} should have 1 - 3 inputs, '\
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
                    for config in  operation.config.input_quantization_config[1: ]:
                        config.state = QuantizationStates.PASSIVE_INIT
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
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
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
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
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
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
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
                        f'while {len(operation.config.input_quantization_config)} was given, is graph defination different from onnx opset 11?'
                    for config in  operation.config.input_quantization_config[1: ]:
                        config.state = QuantizationStates.SOI
                    continue


class NxpInputRoundingRefinePass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='PPQ Input Quantization Refine Pass')

    def optimize(self, processer: GraphCommandProcesser, 
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        graph = processer.graph
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
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        graph = processer.graph
        processer = SearchableGraph(processer)

        relu_fusion_matching = processer.activation_matching(
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

        concat_fusion_matching = processer.concat_matching(
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
    def __init__(self, platform: TargetPlatform,
                 fuse_concat: bool = False, 
                 fuse_activation: bool = True,
                 fuse_passive_op: bool = True,
                 ) -> None:
        self.platform        = platform
        self.fuse_concat     = fuse_concat
        self.fuse_activation = fuse_activation
        self.fuse_passive_op = fuse_passive_op
        super().__init__(name='PPQ Quantization Fusion Pass')

    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])

    @ empty_ppq_cache
    def optimize(
        self,
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        graph = processer.graph
        processer = SearchableGraph(processer)

        # fuse computing opeartions and its following activation.
        if self.fuse_activation:

            # find all activation operations
            act_ops = []
            for op in graph.operations.values():
                if not isinstance(op, QuantableOperation): continue
                if op.type in LINEAR_ACTIVATIONS: act_ops.append(op)
                elif self.platform == TargetPlatform.PPL_CUDA_INT8:
                    if op.type in PPLCUDA_ACTIVATIONS: act_ops.append(op)
                else: continue

            # fusion
            for op in act_ops:
                assert isinstance(op, QuantableOperation)
                upstream_ops = graph.get_upstream_operations(op)
                assert len(upstream_ops) == 1, 'Oops, we got some problem here.'
                
                upstream_op = upstream_ops[0]
                if self.platform == TargetPlatform.ORT_OOS_INT8:
                    if not upstream_op.type in ORT_OOS_FUSE_START_OPS: continue

                if (isinstance(upstream_op, QuantableOperation) and 
                    len(graph.get_downstream_operations(upstream_op)) == 1 and 
                    upstream_op.platform == op.platform):
                    upstream_op.config.output_quantization_config[0].dominated_by = (
                        op.config.output_quantization_config[0])

        if self.fuse_passive_op:
            # all passive operations should never changes quantization configuration of its input
            # so to say their input and output share a same scale.
            for op in graph.operations.values():
                upstream_layers = graph.get_upstream_operations(op)
                if len(upstream_layers) == 0: continue # begining op, can not merge.
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


class InplaceQuantizationSettingPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='Inplace Qunantization Setting Pass')
    
    def optimize(self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, **kwargs) -> None:
        for op in processer.graph.operations.values():
            if isinstance(op, QuantableOperation):
                # set all tensor to be inplace quantized for memory saving.
                for cfg, var in op.config_with_variable:
                    if not var.is_parameter:
                        cfg.inplace = True
           

class PPLCudaAddConvReluMerge(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='PPL CUDA Conv(Relu) - Add - Relu Merge')
    
    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])
    
    def optimize(self, 
                 processer: GraphCommandProcesser, 
                 dataloader: Iterable, 
                 executor: BaseGraphExecutor, 
                 **kwargs) -> None:

        def ep_expr(operation: Operation):
            if not isinstance(operation, QuantableOperation): return False
            if operation.type == 'Conv': return True
            if operation.type in PPLCUDA_ACTIVATIONS:
                upstream_ops = graph.get_upstream_operations(operation=operation)
                if len(upstream_ops) == 0 and upstream_ops[0].type == 'Conv': return True
                if upstream_ops[0] in merged: return True
            return False

        def retrospect(opeartion: QuantableOperation) -> QuantableOperation:
            if not isinstance(opeartion, QuantableOperation): return None
            if len(graph.get_upstream_operations(operation)) != 1: return None
            
            parent = graph.get_upstream_operations(operation)[0]
            if parent.type != 'Conv': return None
            if not isinstance(parent, QuantableOperation): return None
            return parent

        def merge_fn(operation: QuantableOperation):
            assert isinstance(operation, QuantableOperation) and operation.type == 'Add'
            # check if upstream ops can be merged
            up_ops = graph.get_upstream_operations(operation)
            if not self.is_same_platform(up_ops + [operation]): return
            
            # Conv - Add - Relu Merge
            config = operation.config.output_quantization_config[0]

            # Step - 1: merge add output to next activation.
            down_ops = graph.get_downstream_operations(operation)
            if (len(down_ops) == 1 and 
                down_ops[0].type in PPLCUDA_ACTIVATIONS and 
                isinstance(down_ops[0], QuantableOperation) and
                down_ops[0].platform == operation.platform):
                config.dominated_by = down_ops[0].config.output_quantization_config[0]

            # Step - 2: disable input conv's quantization(only one).
            up_ops = graph.get_upstream_operations(operation)
            assert len(up_ops) == 2, f'Opeartion {operation.name} should has exact 2 input operations.'

            target_operation = None
            for op in up_ops:
                if op.type == 'Conv': 
                    target_operation = op
                elif op.type in PPLCUDA_ACTIVATIONS:
                    target_operation = retrospect(operation)
                if target_operation is not None:
                    break

            if target_operation is not None:
                target_operation.config.output_quantization_config[0].dominated_by = config

        graph, merged, unchanged = processer.graph, set(), False

        # merge conv - add iteratively, until there is no one left.
        while not unchanged:
            unchanged = True

            search_engine = SearchableGraph(processer)
            matchings = search_engine(TraversalCommand(
                sp_expr=lambda x: (x.type == 'Add' and 
                                   isinstance(x, QuantableOperation) and 
                                   x not in merged),
                rp_expr=lambda x, y: False,
                ep_expr=ep_expr,
                direction='up'))

            # count how many matched inputs does an add operation has.
            counter = defaultdict(lambda : 0)

            # path[0] is add operation.
            for path in matchings: counter[path[0]] += 1

            for operation, count in counter.items():
                if count == 2: 
                    merge_fn(operation)
                    merged.add(operation)
                    unchanged = False


class QuantAlignmentPass(QuantizationOptimizationPass):
    """
        对多输入算子执行强制定点覆盖
    """
    def __init__(self, 
                 elementwise_merge_method: str = 'Align to Large', 
                 concat_merge_method: str = 'Align to Output',
                 force_overlap: bool = False) -> None:
        self.elementwise_merge_method = elementwise_merge_method
        self.concat_merge_method      = concat_merge_method
        self.force_overlap            = force_overlap
        assert self.elementwise_merge_method in {'Align to Large', 'Align to Output'}, (
            'elementwise_merge_method can only be Align to Large or Align to Output')
        assert self.concat_merge_method in {'Align to Large', 'Align to Output'}, (
            'concat_merge_method can only be Align to Large or Align to Output')
        super().__init__(name='PPQ Quantization Alignment Pass')

    def align_to_large(self, op: QuantableOperation) -> TensorQuantizationConfig:
        """
        Align quant scale and offset to larger input config.
            The first input config will be set as master config,
            all slave config will share the same scale and offset with master.
        
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
        master_config.scale  = torch.tensor([scale], dtype=torch.float32, device=device)
        master_config.offset = torch.tensor([offset], dtype=torch.float32, device=device)

        for slave_config in op.config.input_quantization_config[1: ]:
            slave_config.set_master(master=master_config)
        
        return master_config

    def align_to_output(self, op: QuantableOperation) -> TensorQuantizationConfig:
        """
        Align quant scale and offset to output config.
            All input configs wiil share a same scale and offset with output config.
            (as a slave to output config)

        Any change to slave config will be rejected since then.
        """
        master_config = op.config.output_quantization_config[0]
        for slave_config in op.config.input_quantization_config:
            slave_config.set_master(master=master_config)
        return master_config

    def optimize(
        self, processer: GraphCommandProcesser,
        dataloader: Iterable, 
        executor: BaseGraphExecutor, **kwargs) -> None: 
        graph = processer.graph
        for operation in processer.graph.operations.values():
            if isinstance(operation, QuantableOperation) and operation.type in COMPELING_OP_TYPES:
                assert operation.config.output_quantization_config[0].state != QuantizationStates.INITIAL, (
                    f'Can not modify quantization state of opeartion {operation.name}, '
                    'cause it has not been correctly quantized.')
                
                method = self.elementwise_merge_method if operation.type != 'Concat' else self.concat_merge_method

                if method == 'Align to Large':
                    master_config = self.align_to_large(operation)
                else: master_config = self.align_to_output(operation)

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
