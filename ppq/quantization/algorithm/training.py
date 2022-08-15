import random
from random import randint
from typing import Iterable, List, Tuple

import torch
from ppq.core import (NUM_OF_CHECKPOINT_FETCHS, PPQ_CONFIG,
                      ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TensorQuantizationConfig)
from ppq.executor import TorchQuantizeDelegator
from ppq.IR import BaseGraph, Operation, Variable
from ppq.IR.search import SearchableGraph
from ppq.utils.fetch import batch_random_fetch
from ppq.utils.round import ppq_tensor_round
from torch.autograd import Function


class CuLSQ_T(Function):
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor,
                offsets: torch.Tensor, quant_min: int, quant_max: int,
                rounding: RoundingPolicy) -> torch.Tensor:
        if not PPQ_CONFIG.USING_CUDA_KERNEL:
            raise PermissionError('Can not invoke CuLSQ, Cuda kernel is not compiled.')
        from ppq.core import CUDA

        # quantization function, pure cuda implmentation
        quantized = CUDA.LinearQuantize_T(
            tensor=tensor,
            scales=scales,
            offsets=offsets,
            minimum=quant_min,
            maximum=quant_max,
            rounding=rounding.value
        )
        # https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
        ctx.save_for_backward(tensor, scales, offsets)
        ctx._quant_params = [quant_min, quant_max, rounding.value]
        return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not PPQ_CONFIG.USING_CUDA_KERNEL:
            raise PermissionError('Can not invoke CuLSQ, Cuda kernel is not compiled.')
        from ppq.core import CUDA

        dy = dy.contiguous()
        tensor, scales, offsets = ctx.saved_tensors
        quant_min, quant_max, rounding = ctx._quant_params
        dx, ds = CUDA.LinearQuantize_T_B(
            tensor, scales, offsets,
            dy, quant_min, quant_max, rounding)
        return dx, ds, None, None, None, None, None


class CuLSQ_C(Function):
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor,
                offsets: torch.Tensor, channel_axis: int,
                quant_min: int, quant_max: int,
                rounding: RoundingPolicy) -> torch.Tensor:
        if not PPQ_CONFIG.USING_CUDA_KERNEL:
            raise PermissionError('Can not invoke CuLSQ, PPQ_CONFIG.USING_CUDA_KERNEL = False.')
        from ppq.core import CUDA
        quantized = CUDA.LinearQuantize_C(
            tensor=tensor,
            scales=scales,
            offsets=offsets,
            channel_axis=channel_axis,
            minimum=quant_min,
            maximum=quant_max,
            rounding=rounding.value
        )
        # https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
        ctx.save_for_backward(tensor, scales, offsets)
        ctx._quant_params = [quant_min, quant_max, channel_axis, rounding.value]
        return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not PPQ_CONFIG.USING_CUDA_KERNEL:
            raise PermissionError('Can not invoke CuLSQ, PPQ_CONFIG.USING_CUDA_KERNEL = False.')
        from ppq.core import CUDA
        
        dy = dy.contiguous()
        tensor, scales, offsets = ctx.saved_tensors
        quant_min, quant_max, channel_axis, rounding = ctx._quant_params
        dx, ds = CUDA.LinearQuantize_C_B(
            tensor, scales, offsets,
            dy, quant_min, quant_max, channel_axis, rounding)
        return dx, ds, None, None, None, None, None, None


class FinetuneCheckPoint:
    """Finetune Check Point stores training loss for variables. It bounds to a
    specific variable, collects and stores its fp32 value as a reference.

    ATTENTION: collecting fp32 value might cause GPU memory overflow, so we use a seed to
        sample only a part of fp32 value instead(randomly pick about 2000 values from given tensor).

    Finetune Check Point maintains a seed for data collecting, a best loss, and a reference values.
    """
    def __init__(self, variable: str, random_fetch: bool = True, seed: int=None, fetchs: int=NUM_OF_CHECKPOINT_FETCHS) -> None:
        if seed is None: seed = randint(0, 0xffffffff)
        self.monitor_var = variable
        self.best_loss   = float(1e9)
        self.seed        = seed
        self.references  = []
        self.outputs     = []
        self.fetchs      = fetchs
        self.random_fetch = random_fetch

    def push(self, tensor: torch.Tensor, is_reference: bool) -> None:
        if self.random_fetch:
            tensor = batch_random_fetch(tensor, seed=self.seed, fetchs_per_batch=self.fetchs)
        if is_reference: self.references.append(tensor)
        else: self.outputs.append(tensor)

    def pop(self) -> Tuple[torch.Tensor]:
        assert len(self.outputs) == len(self.references), ('Inconsistent samples detected.'
            f'Reference output gets {len(self.references)} samples, however output has {len(self.outputs)}.')

        return self.outputs, self.references

    def clear(self):
        self.outputs.clear()


class RandomMemDataset:
    """A very little helper class for randomly pick data samples from your
    dataset."""
    def __init__(self, data: Iterable) -> None:
        self._data = data
        self._num_of_batchs = len(data)

    def pop(self):
        idx = random.randint(0, self._num_of_batchs - 1)
        return self._data[idx]


class PriorityQueue:
    """一个很低端的优先队列实现.

    因为python自带的那个实现少一个我需要的接口

    所以我就自己写了这个，它很慢，但是够用。
    """
    def __init__(self, ) -> None:
        self._data = []
        self._ops  = set()
        self._idx  = 0
        self._lazy_tag = True # 延迟操作标志

    def pop(self) -> Tuple[int, Operation]:
        if not self._lazy_tag:
            self._data = sorted(self._data, key=lambda x: x[0])
            self._lazy_tag = True
        if self._idx >= len(self._data): raise IndexError('Index out of range!')
        ele = self._data[self._idx]
        self._idx += 1
        return ele

    def push(self, depth: int, op: Operation):
        if op in self._ops: return
        self._data.append((depth, op))
        self._ops.add(op)
        self._lazy_tag = False

    def empty(self) -> bool:
        return self._idx >= len(self._data)


class TrainableBlock:
    """TrainableBlock refers to a limited subgraph extracted from integrated
    computational graph. TrainableBlock have exact one input node and one
    output node, while its depth(the distance from input node to output node)
    is limited.

    Formal definition of TrainableBlock can be found with following code of BlockBuilder

    Minimal TrainableBlock is {p, p, {p}}, this block have only one node as both input and output.
    """
    def __init__(self, sp: Operation, ep: Operation, rps: List[Operation]) -> None:
        self.sp = sp # 起始节点
        self.ep = ep # 终止节点
        self.rps = rps # 中继节点

    def __str__(self) -> str:
        return f'[Graph Block from {self.sp.name} to {self.ep.name}]'


class BlockBuilder:
    """
    Network Block Builder will cut your graph into small blocks(subgraphs)
    Each block will have exact 1 input operation and 1 output operation.
    
    Besides, there is a parameter 'limit' to control the size of each block.
    """
    def __init__(self, graph: BaseGraph, topo_order: List[Operation]) -> None:
        self.graph = graph
        self.op_orders = topo_order
        self.depth = {}
        self.search_engine = SearchableGraph(self.graph)
        self.initialize_depth()

    def create_block(self, sp: Operation, ep: Operation) -> TrainableBlock:
        if sp == ep: return TrainableBlock(sp=sp, ep=ep, rps=[sp])
        rps = self.search_engine.opset_matching(
            sp_expr = lambda x: x == sp,
            rp_expr = lambda x, y: True,
            ep_expr = lambda x: x == ep,
            direction='down')
        rps = [(self.op_orders.index(op), op) for op in rps]
        rps = sorted(rps)
        return TrainableBlock(sp=sp, ep=ep, rps=[op for _, op in rps])

    def build(self, op: Operation, limit: int) -> TrainableBlock:
        """子图分割算法, 这个算法将从指定节点出发, 构造一个满足定义的子图结构 Solving best block from given
        operation.

        Block definition:(子图定义)
            A Block is a triple contains S, E, M,
                where S is the input node of block
                where E is the output node of block
                where M contains all nodes inside block

        Property:(子图性质)
            1. Minimal TrainableBlock start from p is {p, p, {p}},
               this block have only one node as both input and output.
            2. When S != E,
               E must on every path from S to graph output.
               S must on every path from graph input to E.
            3. M contains and only contains nodes on all paths from S to E.

        Lemma:(算法引理)
            1. 如果 s 的后继节点只有一个 e, 且 e 的输入只有一个，那么 {s, e, {s, e}} 构成满足定义的子图，从 s 寻找子图的任务可以递归由 e 完成
            2. 如果 s 的后继存在多个节点，则只存在两种情况:
                2.1 从 s 出发，最大子图即为 {s, s, {s}}。
                2.2 从 s 出发，可以构成子图 {s, e, R} (e!=s), 那么R中必须含有一个节点接收多个输入。（可用反证法证明，略）

        Algorithm:(算法)
            Build(s, d):
                如果区块长度大于所需，则返回现有内容
                从 s 出发，如果 s 的后继节点只有一个e，则判断e的输入节点个数：
                    1. 如果 e 是单输入节点，执行Build(e, d-1)，并将其结果与 {s, e, {s, e}} 合并
                    2. 如果 e 是多输入节点，算法立即停机，返回 {s, s, {s}}

                如果 s 的后继节点存在多个，找出距离 s 拓扑序最近的多输入的节点 k1，判断 s 到输出的路径是否能够被 k1 阻断
                    如果 k 成功阻断所有输出，执行Build(k1, d-1)，并将其结果与 {s, k1, F(s, k1)} 合并
                    如果 k 不能阻断输出，寻找距离 s 次近的多输入节点 k2，重复判断
                直到 kn 到达 s 的距离超出限制

                函数 F(s, k1) 取出 从 s 到 k1 路上的所有节点

        可利用引理证明算法正确性，从略
        时间复杂度: O(kd) k 为节点最大度数 d 为深度限制。建立所有Block所需时间 O(nkd)
        """
        def _find_multi_input_ep(op: Operation):
            # 如果当前节点后继节点存在多个，层序遍历寻找阻断节点
            least_first_queue = PriorityQueue()

            for down_op in self.graph.get_downstream_operations(op):
                least_first_queue.push(self.depth[down_op], down_op)

            while not least_first_queue.empty():
                iter_operation = least_first_queue.pop()[-1]
                if (least_first_queue.empty() and
                    len(self.graph.get_upstream_operations(iter_operation)) > 1):
                    return iter_operation
                for down_op in self.graph.get_downstream_operations(iter_operation):
                    least_first_queue.push(self.depth[down_op], down_op)

            # if least_first_queue is empty, it means we can not find an blocking ep from given sp.
            return None

        def _find_coherent_ep(op: Operation):
            # 如果当前节点后继节点只有一个，向下寻找直系节点
            # 但如果直系后继节点有多于一个输入，算法立即停机
            ops = self.graph.get_downstream_operations(op)
            if len(ops) == 1:
                following_op = ops[0]
                # PATCH 20220811，get_upstream_operations 不足以判断算子是否只有一个输入
                # 因为算子可以直接与图的 input 相连...
                non_parameter_input = following_op.num_of_input - following_op.num_of_parameter
                upstream_ops = len(self.graph.get_upstream_operations(following_op))
                if non_parameter_input == 1 and upstream_ops == 1:
                    return ops[0]
            return None

        sp, ep, future_ep = op, op, op
        while future_ep is not None:
            if len(self.graph.get_downstream_operations(ep)) <= 1:
                future_ep = _find_coherent_ep(ep)
            else: future_ep = _find_multi_input_ep(ep)
            if future_ep is None or self.depth[future_ep] - self.depth[sp] > limit:
                return self.create_block(sp, ep)
            ep = future_ep
        return self.create_block(sp=sp, ep=ep)

    def initialize_depth(self) -> None:
        """为图中所有节点确定深度，基于拓扑排序与动态规划，O(kn)时间复杂度 k为图节点最大度数，n为图节点个数."""
        for operation in self.op_orders:
            # graph input operation, set depth as 0
            if len(self.graph.get_upstream_operations(operation)) == 0:
                self.depth[operation] = 0
                continue

            # otherwise we will go dp
            depths_cache = []
            for up_op in self.graph.get_upstream_operations(operation):
                assert up_op in self.depth, ('Oops, that should not happen to your network.')
                depths_cache.append(self.depth[up_op])
            self.depth[operation] = max(depths_cache) + 1


class LSQDelegator(TorchQuantizeDelegator):
    def __init__(
        self, config: TensorQuantizationConfig, var: Variable, 
        is_parameter_trainable: bool = True, 
        is_scale_trainable:     bool = True, 
        is_offset_trainable:    bool = True
    ) -> None:
        self.config        = config
        self.is_parameter  = var.is_parameter
        self.var           = var
        self.policy        = config.policy
        self.passive       = config.state == QuantizationStates.PASSIVE

        self.param_backup = None
        if self.is_parameter and is_parameter_trainable:
            self.param_backup = self.var.value.clone()

        # There is 4 checks for training scale:
        #   1. scale is valid
        #   2. state is active
        #   3. do not have POWER_OF_2 policy
        #   4. is_scale_trainable = True
        self.scale_backup       = None
        self.is_scale_trainable = False
        if is_scale_trainable:
            policy_check = not config.policy.has_property(QuantizationProperty.POWER_OF_2)
            state_check  = ((config.state == QuantizationStates.ACTIVATED) and (config.dominated_by == config))
            value_check  = isinstance(config.scale, torch.Tensor)
            if policy_check and state_check and value_check:
                self.is_scale_trainable = True
                self.scale_backup = self.config.scale.detach().clone()

        # There is 4 checks for training offset:
        #   1. offset is valid
        #   2. state is active
        #   3. do not have SYMMETRICAL policy
        #   4. is_scale_trainable = True
        self.offset_backup       = None
        self.is_offset_trainable = False
        if is_offset_trainable:
            policy_check = not config.policy.has_property(QuantizationProperty.SYMMETRICAL)
            state_check  = ((config.state == QuantizationStates.ACTIVATED) and (config.dominated_by == config))
            value_check  = isinstance(config.offset, torch.Tensor)
            if policy_check and state_check and value_check:
                self.is_offset_trainable = True
                self.offset_backup = self.config.offset.detach().clone()

    def trainable_tensors(self) -> List[torch.Tensor]:
        params = []
        if self.is_offset_trainable: params.append(self.config.offset)
        if self.is_scale_trainable:  params.append(self.config.scale)
        if self.is_parameter:        params.append(self.var.value)
        return params

    def withdraw(self) -> None:
        with torch.no_grad():
            if self.scale_backup is not None:
                self.config.scale.copy_(self.scale_backup)
            if self.offset_backup is not None:
                self.config.offset.copy_(self.offset_backup)
            if self.param_backup is not None:
                self.var.value.copy_(self.param_backup)

    def finalize(self) -> None:
        self.scale_backup  = None
        self.offset_backup = None
        self.param_backup  = None
        pass # do nothing here.

    def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
        if not PPQ_CONFIG.USING_CUDA_KERNEL:
            scale, offset = config.scale, config.offset

            if self.is_scale_trainable:
                scale = scale.abs()
                grad_scale = 1 / (tensor.numel() * config.quant_max) ** 0.5
                scale = scale * grad_scale + (scale - scale * grad_scale).detach()

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                assert isinstance(config, ChannelwiseTensorQuantizationConfig)
                shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
                scale = scale.view(shape)
                offset = offset.view(shape)

            quantized = ppq_tensor_round((tensor / scale), config.rounding) + offset.detach()
            quantized = torch.clamp(quantized, config.quant_min, config.quant_max)
            quantized = (quantized - offset.detach()) * scale
            quantized = quantized
            return quantized

        else:
            if not config.policy.has_property(QuantizationProperty.LINEAR):
                raise ValueError('Critical Quantization Error! Non-linear config detected.')
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                assert isinstance(config, ChannelwiseTensorQuantizationConfig), (
                    'Critical Quantization Error! Except a ChannelwiseTensorQuantizationConfig.')
                return CuLSQ_C.apply(
                    tensor, config.scale, config.offset, config.channel_axis,
                    config.quant_min, config.quant_max, config.rounding)
            elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
                return CuLSQ_T.apply(
                    tensor, config.scale, config.offset,
                    config.quant_min, config.quant_max, config.rounding)
            else:
                raise PermissionError('Oops, Unexpected Error here.')

