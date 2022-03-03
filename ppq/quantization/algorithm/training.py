# TODO move training logic to here.

import random
from math import sqrt
from random import randint
from typing import Iterable, List, Tuple

import numpy as np
import torch
from ppq.core import (NUM_OF_CHECKPOINT_FETCHS, USING_CUDA_KERNEL,
                      ChannelwiseTensorQuantizationConfig, NetworkFramework,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TensorQuantizationConfig)
from ppq.executor import TorchQuantizeDelegate
from ppq.IR import BaseGraph, Operation, Variable
from ppq.IR.search import SearchableGraph
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction
from ppq.utils.fetch import batch_random_fetch
from ppq.utils.round import ppq_tensor_round
from torch.autograd import Function

if USING_CUDA_KERNEL:
    from ppq.core import CUDA


if not USING_CUDA_KERNEL:
    class Clip_T(Function):
        """
        Tensorwise Clip function requires an input tensor t, 
            a reference tensor r, and a limit tensor.

        This function will clip t within range [r - limit, r + limit]
        Limit tensor is applied per tensor, so to say each value of tensor t
            shares a same limit value. 
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, reference: torch.Tensor, 
                    limit_t: torch.Tensor) -> torch.Tensor:
            # we do not provide a torch native implementation yet.
            return tensor
    
        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None, None
    
    class Clip_C(Function):
        """
        Channelwise Clip function requires an input tensor t, 
            a reference tensor r, and a limit tensor.

        This function will clip t within range [r - limit, r + limit]
        Limit tensor is applied per channel, so to say each channel of tensor t
            shares a same limit value. 
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, reference: torch.Tensor, 
                    limit_t: torch.Tensor, channel_axis: int) -> torch.Tensor:
            # we do not provide a torch native implementation yet.
            return tensor
    
        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None, None, None

    class RoundingLoss_T(Function):
        """
        Compute Tensorwise Rounding loss(L1)
            This function implements Tensorwise Rounding loss with torch.
            This function implements backwards with torch.
        
        Say rounding loss = RL, we will have
        
        qt = CUDA.LinearQuantize_T(t, s, o, Q_MIN, Q_MAX)
        RL = torch.sum(torch.abs(qt - t)) / sqrt(qt.numel())
        
        term sqrt(qt.numel() is used as a normalization.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy) -> torch.Tensor:
    
            qt = ppq_tensor_round((tensor / scales), rounding) + offsets
            qt = torch.clamp(qt, quant_min, quant_max)
            qt = (qt - offsets) * scales
            return torch.sum(torch.abs(qt - tensor)) / sqrt(qt.numel())

    class RoundingLoss_C(Function):
        """
        Compute Channelwise Rounding loss(L1)
            This function implements Channelwise Rounding loss with torch.
            This function implements backwards with torch.
        
        Say rounding loss = RL, we will have
        
        qt = CUDA.LinearQuantize_C(t, s, o, c, Q_MIN, Q_MAX)
        RL = torch.sum(torch.abs(qt - t)) / sqrt(qt.numel())
        
        term sqrt(qt.numel() is used as a normalization.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, channel_axis: int,
                    quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy) -> torch.Tensor:
            shape = [1 if axis != channel_axis else -1 for axis in range(tensor.ndim)]
            scale, offset = scales.view(shape), offsets.view(shape)
            
            qt = ppq_tensor_round((tensor / scale), rounding) + offset
            qt = torch.clamp(qt, quant_min, quant_max)
            qt = (qt - offset) * scale
            return torch.sum(torch.abs(qt - tensor)) / sqrt(qt.numel())

else: # if USING_CUDA_KERNEL:
    class Clip_T(Function):
        """
        Tensorwise Clip function requires an input tensor t, 
            a reference tensor r, and a limit tensor.

        This function will clip t within range [r - limit, r + limit]
        Limit tensor is applied per tensor, so to say each value of tensor t
            shares a same limit value. 
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, reference: torch.Tensor, 
                    limit_t: torch.Tensor) -> torch.Tensor:
            return CUDA.TensorClip_T(tensor=tensor, reference=reference, limit=limit_t)

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None, None

    class Clip_C(Function):
        """
        Channelwise Clip function requires an input tensor t, 
            a reference tensor r, and a limit tensor.

        This function will clip t within range [r - limit, r + limit]
        Limit tensor is applied per channel, so to say each channel of tensor t
            shares a same limit value. 
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, reference: torch.Tensor, 
                    limit_t: torch.Tensor, channel_axis: int) -> torch.Tensor:
            return CUDA.TensorClip_C(tensor=tensor, reference=reference, 
                                     limit=limit_t, channel_axis=channel_axis)

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None, None, None

    class RoundingLoss_T(Function):
        """
        Compute Tensorwise Rounding loss(L1)
            This function implements Tensorwise Rounding loss with cuda.
            This function implements backwards with cuda.
        
        Say rounding loss = RL, we will have
        
        qt = CUDA.LinearQuantize_T(t, s, o, Q_MIN, Q_MAX)
        RL = torch.sum(torch.abs(qt - t)) / sqrt(qt.numel())
        
        term sqrt(qt.numel() is used as a normalization.
        Notice this function ignore loss(or grad) from clipped value.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy) -> torch.Tensor:
            
            ctx.save_for_backward(tensor, scales, offsets)
            ctx._quant_params = [quant_min, quant_max, rounding.value]

            return CUDA.RoundingLoss_LT(
                tensor=tensor, scales=scales, 
                offsets=offsets, minimum=quant_min, maximum=quant_max, 
                rounding=rounding.value)

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            tensor, scales, offsets = ctx.saved_tensors
            quant_min, quant_max, rounding = ctx._quant_params
            return CUDA.RoundingLoss_LT_B(
                tensor=tensor, dy=dy, scales=scales, 
                offsets=offsets, minimum=quant_min, maximum=quant_max, 
                rounding=rounding), None, None, None, None, None

    class RoundingLoss_C(Function):
        """
        Compute Channelwise Rounding loss(L1)
            This function implements Channelwise Rounding loss with cuda.
            This function implements backwards with cuda.
        
        Say rounding loss = RL, we will have
        
        qt = CUDA.LinearQuantize_C(t, s, o, c, Q_MIN, Q_MAX)
        RL = torch.sum(torch.abs(qt - t)) / sqrt(qt.numel())
        
        term sqrt(qt.numel() is used as a normalization.
        Notice this function ignore loss(or grad) from clipped value.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, channel_axis: int,
                    quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy) -> torch.Tensor:
            
            ctx.save_for_backward(tensor, scales, offsets)
            ctx._quant_params = [quant_min, quant_max, channel_axis, rounding.value]

            return CUDA.RoundingLoss_LC(
                tensor=tensor, scales=scales, 
                offsets=offsets, channel_axis=channel_axis,
                minimum=quant_min, maximum=quant_max, 
                rounding=rounding.value)

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            tensor, scales, offsets = ctx.saved_tensors
            quant_min, quant_max, channel_axis, rounding = ctx._quant_params
            return CUDA.RoundingLoss_LC_B(
                tensor=tensor, dy=dy, scales=scales, 
                offsets=offsets, channel_axis=channel_axis,
                minimum=quant_min, maximum=quant_max, 
                rounding=rounding), None, None, None, None, None, None


def PPQTensorClip(
    tensor: torch.Tensor, reference: torch.Tensor, 
    limit: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        assert isinstance(config, ChannelwiseTensorQuantizationConfig)
        return Clip_C.apply(tensor, reference, limit, config.channel_axis)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return Clip_T.apply(tensor, reference, limit)
    else: raise Exception('Oops, seems we got some problems here.')


def PPQRoundingLoss(tensor: torch.Tensor,
                    config: TensorQuantizationConfig) -> torch.Tensor:
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        assert isinstance(config, ChannelwiseTensorQuantizationConfig)
        return RoundingLoss_C.apply(
            tensor, config.scale, config.offset, config.channel_axis, 
            config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return RoundingLoss_T.apply(
            tensor, config.scale, config.offset, config.quant_min, 
            config.quant_max, config.rounding)
    else: raise Exception('Oops, seems we got some problems here.')


class FinetuneCheckPoint:
    """
    Finetune Check Point stores training loss for variables.
        It bounds to a specific variable, collects and stores its fp32 value as a reference.
    
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


class RQTDelegator(TorchQuantizeDelegate):
    """
        Restricted Quantization(RQT) Functions are a set of custimized
        quantization functions which provide value restriction on your parameter.
        
        Value restriction is an essential functionaility for training your network from
            its fp32 version. All parameters go throuth this function is restricted with 
            [reference - limit, reference + limit], where reference is the fp32 value of
            your parameter.
        
        RQT Functions guarantee that parameter never goes to far away from origin.
        
        With those features, we could directly finetune a quantized network
            from 32-bit to lower bit width, avoiding the risk of over-fitting.
        
        USE THIS FUNCTION TO REPLACE PPQ EXECUTOR'S QUANTIZATION LOGIC WITH
            executor.register_quant_delegator(config, RQTDelegator())
        
        ATTENTION: RQT FUNCTIONS ARE AVAILABLE WITH USING_CUDA_KERNEL = True
    """
    def __init__(
        self, binding: Variable, config: TensorQuantizationConfig, limit: float) -> None:
        if config.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
            raise PermissionError(f'Can not create TrainableDelegate with variable {binding.name},'
                                  ' cause its value has been baked.')
        
        limit_t = torch.clone(config.scale)
        # create limit tensor
        if config.state == QuantizationStates.ACTIVATED:
            limit_t = limit_t * limit
        if config.state == QuantizationStates.PASSIVE:
            limit_t = limit_t.fill_(limit * (0.01) * torch.max(binding.value).item())

        self.reference = binding.value.clone()
        self.limit_t   = limit_t
        self.limit     = limit
        self.config    = config
        self.binding   = binding
        self.binding.value.requires_grad  = True

    def withdraw(self):
        self.binding.value = self.reference

    def finalize(self):
        self.binding.value.requires_grad = False
        self.binding.value._grad         = None
        self.binding.value               = self.__call__(self.binding.value, self.config)

    def __call__(self, tensor: torch.Tensor, 
                 config: TensorQuantizationConfig) -> torch.Tensor:
        if self.limit == 0: return PPQLinearQuantFunction(tensor, config)
        return PPQTensorClip(
            tensor=PPQLinearQuantFunction(tensor, config), 
            reference=self.reference, limit=self.limit_t, config=config
        )


class BanditDelegator(TorchQuantizeDelegate):
    """
        带有多臂赌博机的量化代理，从 ppq 0.6.2 版本后，我们引入
            多臂赌博机算法训练 scale 与 offset。在未来我们可能还会引入其他
            类似的算法，例如UCB，马尔可夫蒙特卡洛估计等。
        
        引入这些算法的原因是我们注意到 scale 与 offset 的导数非常不靠谱
        为此我们引入简单的强化学习，直接估计P(r | scale=s, context)
        即再给定上下文 context 的情况下，选取当前 scale 为 s，获利的概率
        
        Quantization with multi-arm bandit.
        
        Multi-arm bandits are introduced since PPQ 0.6.2 for training
            quantization scale and offset.
    """
    class EMA():
        def __init__(self, decay):
            self.decay = decay
            self.value = 0

        def put(self, value: float):
            value = (1.0 - self.decay) * value + self.decay * self.value
    
    def __init__(self,  arms: List[float], config: TensorQuantizationConfig, smooth: float = 1) -> None:
        if len(arms) < 2: raise ValueError('Can not initialize bandit with less than 2 arms.')
        self.arms = arms
        self.num_of_arms = len(arms)
        self.smooth = smooth
        self.active = True
        self.wins   = [0 for _ in range(self.num_of_arms)]
        self.trials = [0 for _ in range(self.num_of_arms)]
        self.last_roll = -1
        assert isinstance(config.scale, torch.Tensor)
        self.reference = config.scale.clone()
        self.config = config
        self.rewards = {r: [] for r in range(self.num_of_arms)}

    def roll(self) -> int:
        # print(self.smooth, self.wins,  self.trials)
        '''
        rolls = [
            random.betavariate(self.smooth + self.wins[i] * 0.3, self.smooth + (self.trials[i] - self.wins[i]) * 0.3)
            for i in range(self.num_of_arms)]
        '''
        # r = np.argmax(rolls)
        if random.random() > 0.5:
            r = random.randint(0, len(self.arms) - 1)
        else: 
            r = np.argmax([sum(v) / (len(v) + 1e-7) for r, v in self.rewards.items()])
        self.trials[r] += 1
        self.last_roll = r
        return r

    def mark(self, rewards: float):
        self.rewards[self.last_roll].append(rewards)
        if rewards > 0: self.wins[self.last_roll] += 1

    def finalize(self) -> bool:
        print('-------------------------------------------')
        for r, v in self.rewards.items():
            print(f"Arm {r} | Rewards {sum(v) / (len(v) + 1e-7):.3f} | wins: {self.wins[r]} | trials: {self.trials[r]}")

        selected = np.argmax(self.wins)
        eta_reward = sum(self.rewards[selected]) / (len(self.rewards[selected]) + 1e-7)
        if eta_reward > 0.01: 
            self.config.scale = self.reference * self.arms[np.argmax(self.wins)]
            return True
        else: 
            self.config.scale = self.reference
            return False

    def __call__(self, tensor: torch.Tensor, 
                 config: TensorQuantizationConfig) -> torch.Tensor:
        if self.active: config.scale = self.reference * self.arms[self.roll()]
        else: config.scale = self.reference
        return PPQLinearQuantFunction(tensor, config)


class RandomMemDataset:
    """
        A very little helper class for randomly pick data samples from your dataset.
    """
    def __init__(self, data: Iterable) -> None:
        self._data = data
        self._num_of_batchs = len(data)

    def pop(self):
        idx = random.randint(0, self._num_of_batchs - 1)
        return self._data[idx]


class TrainableSubgraph(BaseGraph):
    def __init__(self,
                 inputs: List[Variable],
                 outputs: List[Variable],
                 operations:List[Operation]) -> None:
        super().__init__(name='PPQ Trainable SubGraph', built_from=NetworkFramework.NATIVE)
        

class TimeDecay:
    """
        A helper class computing time decay.
    """
    def __init__(self, t_max: int, decay: float=0.2, beta_start: float=20, beta_end:float=2):
        self.t_max = t_max
        self.start_decay = decay * t_max
        self.start_b = beta_start
        self.end_b = beta_end

    def __call__(self, t):
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class AdaroundRegTerm(torch.nn.Module):
    """
        Adaround Reg Term is a part of Adaround optimization algorithm.
            This term represents the difference between a fp32 value and its quantized counter-part.
        We use a same implementation as proposed in Adaround paper.
    Args:
        torch ([type]): [description]
    """
    def __init__(self, max_iter: int = 20000, 
                 zeta: float = 1.1, gamma:float = -0.1, 
                 alpha: float = 0.01, beta: float = 20, 
                 warm_ratio: float = 0.2):
        self.max_iter = max_iter
        self.zeta = zeta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.warm_ratio = warm_ratio
        self.temp_anneal = TimeDecay(self.max_iter, self.warm_ratio)
        super().__init__()

    def rectified_sigmoid(self, round_mask):
        return ((self.zeta - self.gamma) * torch.sigmoid(round_mask) + self.gamma).clamp(0, 1)

    def forward(self, round_mask, iter, reduction='sum'):
        if iter < self.max_iter * self.warm_ratio:
            round_loss = 0
        else:
            self.beta = self.temp_anneal(iter)
            if reduction == 'sum':
                round_loss = self.alpha * (1 - torch.pow((self.rectified_sigmoid(round_mask) - 0.5).abs() * 2, self.beta)).sum()
            else:
                round_loss = self.alpha * (1 - torch.pow((self.rectified_sigmoid(round_mask) - 0.5).abs() * 2, self.beta)).mean()
        return round_loss


def Lp_norm(pred: torch.tensor, tgt: torch.tensor, p: float = 2.0):
    return (pred - tgt).abs().pow(p).sum(1).mean()


class PriorityQueue:
    """
        一个很低端的优先队列实现
       
        因为python自带的那个实现少一个我需要的接口
        
        所以我就自己写了这个，它很慢，但是够用。
    """
    def __init__(self, ) -> None:
        self._data = []
        self._ops  = set()
        self._idx  = 0
    
    def pop(self) -> Tuple[int, Operation]:
        if self._idx >= len(self._data): raise IndexError('Index out of range!')
        ele = self._data[self._idx]
        self._idx += 1
        return ele

    def push(self, depth: int, op: Operation):
        if op in self._ops: return
        self._data.append((depth, op))
        self._data = sorted(self._data, key=lambda x: x[0])
        self._ops.add(op)
    
    def empty(self) -> bool:
        return self._idx >= len(self._data)


class TrainableBlock:
    """
    TrainableBlock refers to a limited subgraph extracted from integrated computational graph.
        TrainableBlock have exact one input node and one output node, while
        its depth(the distance from input node to output node) is limited.
    
    Formal defination of TrainableBlock can be found with following code of BlockBuilder
    
    Minmal TrainableBlock is {p, p, {p}}, this block have only one node as both input and output.
    """
    def __init__(self, sp: Operation, ep: Operation, rps: List[Operation]) -> None:
        self.sp = sp # 起始节点
        self.ep = ep # 终止节点
        self.rps = rps # 中继节点
    
    def __str__(self) -> str:
        return f'[Graph Block from {self.sp.name} to {self.ep.name}]'


class BlockBuilder:
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
        """
        Solving best block from given operation.
        
        Block defination:  
            A Block is a triple contains S, E, M, 
                where S is the input node of block
                where E is the output node of block
                where M contains all nodes inside block
        
        Property:
            1. Minmal TrainableBlock start from p is {p, p, {p}}, 
               this block have only one node as both input and output.
            2. When S != E, 
               E must on every path from S to graph output.
               S must on every path from graph input to E.
            3. M contains and only contains nodes on all paths from S to E.
        
        Lemma:
            1. 如果 s 的后继节点只有一个 e，且 e 的输入只有一个，那么 {s, e, {s, e}} 构成区块，从 s 寻找区块的任务可以递归由 e 完成
            2. 如果 s 的后继存在多个节点，则:
                2.1 从 s 出发，不存在能够符合定义的区块。
                2.2 从 s 出发，构成的区块内必须含有一个节点接收多个输入。（可用反证法证明，略）

        Algorithm:
            0. 如果区块长度大于所需，则返回现有内容，否则转 1
            1. 从 s 出发，如果 s 的后继节点只有一个，则递归寻找从后继节点开始的区块；
               如果 s 的后继节点存在多个，找出距离 s 拓扑序最近的多输入的节点 k1，判断 s 到输出的路径是否能够被 k1 阻断
                    如果 k 成功阻断所有输出，递归寻找从 k 开始的区块
                    如果 k 不能阻断输出，寻找距离 s 次近的多输入节点 k2，重复判断
               直到 kn 到达 s 的距离超出限制

        可利用引理证明算法正确性，从略
        时间复杂度: O(kd) k 为节点最大度数，d 为深度限制。
        建立所有Block所需时间 O(nkd)
        """
        def _find_multi_input_ep(op: Operation):
            # 层序遍历寻找阻断节点
            least_first_queue = PriorityQueue()
            
            for down_op in self.graph.get_downstream_operations(op):
                least_first_queue.push(self.depth[down_op], down_op)
            
            while not least_first_queue.empty():
                iter_operation = least_first_queue.pop()[-1]
                if least_first_queue.empty(): return iter_operation
                for down_op in self.graph.get_downstream_operations(iter_operation):
                    least_first_queue.push(self.depth[down_op], down_op)

            # if least_first_queue is empty, it means we can not find an blocking ep from given sp.
            return None

        def _find_coherent_ep(op: Operation):
            ops = self.graph.get_downstream_operations(op)
            if len(ops) == 1 and len(self.graph.get_upstream_operations(ops[0])) == 1: 
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
        """
            为图中所有节点确定深度，基于拓扑排序与动态规划，O(kn)时间复杂度
            k为图节点最大度数，n为图节点个数
        """
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
