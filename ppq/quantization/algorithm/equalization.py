from enum import Enum
from typing import List

import torch
from ppq.IR import Operation


class EqualizationMethod(Enum):
    """

        EqualizationMethod - 用于列举所有可行的 scale 计算方案
                            different arithmetic methods for equalization
    """
    # key value = np.max(np.abs(x))
    ABSOLUTE_MAX = 1,
    # key value = np.mean(np.abs(x))
    ABSOLUTE_MEAN = 2,
    # key value = np.max(np.square(x))
    SQUARE_MAX = 3,
    # key value = np.mean(np.square(x))
    SQUARE_MEAN = 4,


class EqualizationHelper():

    @ staticmethod
    def key_value_from_upstream(
        op: Operation, including_bias: bool = False, including_act: bool = False, 
        bias_multiplier: float = 0.5, act_multiplier: float = 0.5) -> torch.Tensor:
        if op.type not in {'Gemm', 'MatMul', 'Conv', 'ConvTranspose'}:
            raise TypeError(f'Unsupported Op type {op.name}({op.type}) for Equalization Optimization.')
        if not op.inputs[1].is_parameter:
            raise ValueError(f'Parameter of Op {op.name} is non-static.')
        buffer = []

        # ----------------------------------
        # step - 1, extract weight from op:
        # ----------------------------------
        w = op.inputs[1].value
        if op.type == 'ConvTranspose':
            num_of_groups = op.attributes.get('group', 1)
            if w.ndim == 3:
                w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
                w = torch.permute(w, (2, 0, 1, 3))
                w = torch.reshape(w, (w.shape[0] * w.shape[1], -1))
            elif w.ndim == 4:
                w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
                w = torch.permute(w, (2, 0, 1, 3, 4))
                w = torch.reshape(w, (w.shape[0] * w.shape[1], -1))
            elif w.ndim == 5:
                w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
                w = torch.permute(w, (2, 0, 1, 3, 4, 5))
                w = torch.reshape(w, (w.shape[0] * w.shape[1], -1))
            else:
                raise ValueError(f'Unexpected dimension of weight of {op.name}.')
            buffer.append(w)

        if op.type in {'MatMul', 'Gemm'}:
            assert w.ndim == 2, f'Unexpected Error, Parameter of MatMul {op.name} should be 2-d.'
            if op.attributes.get('transB', 0) == 0:
                w = torch.transpose(w, 1, 0)
            buffer.append(w)

        if op.type == 'Conv':
            w = torch.reshape(w, (w.shape[0], -1))
            buffer.append(w)

        # ----------------------------------
        # step - 2, extract bias from op:
        # ----------------------------------
        if including_bias and op.num_of_input == 3:
            b = op.inputs[-1].value * bias_multiplier
            if op.type in {'Conv', 'Gemm'} and op.inputs[-1].is_parameter:
                b = torch.reshape(b, (w.shape[0], 1))
                buffer.append(b)

            if op.type == 'ConvTranspose':
                b = torch.reshape(b, (w.shape[0], 1))
                buffer.append(b)

        # ----------------------------------
        # step - 3, extract activation from op:
        # ----------------------------------
        if including_act and op.inputs[0].value is not None:
            a = op.outputs[0].value * act_multiplier
            buffer.append(a)

        # concat and return
        return torch.cat(buffer, dim=-1)

    @ staticmethod
    def key_value_from_downstream(op: Operation) -> torch.Tensor:
        # ----------------------------------
        # step - 1, extract weight from op:
        # ----------------------------------
        w = op.inputs[1].value
        if op.type == 'ConvTranspose':
            w = torch.reshape(w, (w.shape[0], -1))

        if op.type in {'MatMul', 'Gemm'}:
            assert w.ndim == 2, f'Unexpected Error, Parameter of MatMul {op.name} should be 2-d.'
            if op.attributes.get('transB', 0) != 0:
                w = torch.transpose(w, 1, 0)

        if op.type == 'Conv':
            # for group convolution, we have to select its weight by group
            num_of_groups = op.attributes.get('group', 1)
            if w.ndim == 3:
                w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
                w = torch.permute(w, (2, 0, 1, 3))
                w = torch.reshape(w, (w.shape[0] * w.shape[1], -1))
            elif w.ndim == 4:
                w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
                w = torch.permute(w, (2, 0, 1, 3, 4))
                w = torch.reshape(w, (w.shape[0] * w.shape[1], -1))
            elif w.ndim == 5:
                w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
                w = torch.permute(w, (2, 0, 1, 3, 4, 5))
                w = torch.reshape(w, (w.shape[0] * w.shape[1], -1))
            else:
                raise ValueError(f'Unexpected dimension of weight of {op.name}.')
        return w

    @ staticmethod
    def scale_to_upstream(op: Operation, scale_factor: torch.Tensor):
        if op.type not in {'Gemm', 'MatMul', 'Conv', 'ConvTranspose'}:
            raise TypeError(f'Unsupported Op type {op.name}({op.type}) for Equalization Optimization.')
        if not op.inputs[1].is_parameter:
            raise ValueError(f'Parameter of Op {op.name} is non-static.')

        w = op.inputs[1].value
        has_bias = op.num_of_input == 3
        if has_bias and not op.inputs[-1].is_parameter: 
            raise ValueError(f'Bias of Op {op.name} is non-static.')
        if has_bias: bias = op.inputs[-1].value

        if op.type == 'ConvTranspose':
            num_of_groups = op.attributes.get('group', 1)
            w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1:])
            w *= torch.reshape(scale_factor, [num_of_groups, 1, -1] + [1] * (w.ndim - 3))
            w = torch.reshape(w, (w.shape[0] * w.shape[1], ) + w.shape[2:])
            if has_bias: bias *= scale_factor

        elif op.type == 'Conv':
            w *= torch.reshape(scale_factor, [-1] + ([1] * (w.ndim - 1)))
            if has_bias: bias *= scale_factor

        elif op.type in {'Gemm', 'MatMul'}:
            if op.attributes.get('transB', 0) == 0: w = torch.transpose(w, 1, 0)
            w *= torch.reshape(scale_factor, (-1, 1))
            if op.attributes.get('transB', 0) == 0: w = torch.transpose(w, 1, 0)
            if has_bias: bias *= scale_factor
        
        # write back
        with torch.no_grad():
            op.inputs[1].value.copy_(w)
            if has_bias: op.inputs[-1].value.copy_(bias)

    @ staticmethod
    def scale_to_downstream(op: Operation, scale_factor: torch.Tensor):
        if op.type not in {'Gemm', 'MatMul', 'Conv', 'ConvTranspose'}:
            raise TypeError(f'Unsupported Op type {op.name}({op.type}) for Equalization Optimization.')
        if not op.inputs[1].is_parameter:
            raise ValueError(f'Parameter of Op {op.name} is non-static.')
        w = op.inputs[1].value

        if op.type == 'ConvTranspose':
            w /= torch.reshape(scale_factor, [-1] + ([1] * (w.ndim - 1)))

        if op.type == 'Conv':
            num_of_groups = op.attributes.get('group', 1)
            w = torch.reshape(w, (num_of_groups, w.shape[0] // num_of_groups) + w.shape[1: ])
            w /= torch.reshape(scale_factor, [num_of_groups, 1, -1] + [1] * (w.ndim - 3))
            w = torch.reshape(w, (w.shape[1] * num_of_groups, ) + w.shape[2: ])

        if op.type in {'Gemm', 'MatMul'}:
            if op.attributes.get('transB', 0) != 0: w = torch.transpose(w, 1, 0)
            w /= torch.reshape(scale_factor, (-1, 1))
            if op.attributes.get('transB', 0) != 0: w = torch.transpose(w, 1, 0)

        # write back
        with torch.no_grad():
            op.inputs[1].value.copy_(w)


class EqualizationPair:
    def __init__(
        self,
        upstream_layers: List[Operation],
        downstream_layers: List[Operation]
    ):
        """
            EqualizationPair - 一个数据结构，封装了 equalization 的核心数据抽象和执行逻辑
                               a class encapsulating execution logic of equalization

                在 self.upstream_layers 包含了 equalization 操作中的所有上游层(op)
                self.upstream_layers contain all upstream ops

                在 self.downstream_layers 包含了 equalization 操作中的所有下游层(op)
                self.downstream_layers contain all downstream ops

            一个 EqualizationPair 结构记录了参与 equalization 变换的所有相关层与其局部图结构信息
            从而支持在局部子图上的 equalization 操作
            An EqualizationPair records all relevant ops participating in the equalization
            transformation, thus supporting equalization on local subgraphs
        Args:
            upstream_layers (list):
                equalization 操作中的所有上游层(op)

            downstream_layers (list):
                equalization 操作中的所有下游层(op)
        """
        self.upstream_layers = upstream_layers
        self.downstream_layers = downstream_layers

    def equalize(
        self,
        value_threshold: float,
        including_act: bool = False,
        act_multiplier: float = 0.5,
        including_bias: bool = False,
        bias_multiplier: float = 0.5,
        method: EqualizationMethod = EqualizationMethod.ABSOLUTE_MAX
    ):
        # extract key value from pair
        upstream_key_values, downstream_key_values = [], []
        for op in self.upstream_layers:
            key_value = EqualizationHelper.key_value_from_upstream(
                op=op, including_bias=including_bias, including_act=including_act, 
                bias_multiplier=bias_multiplier, act_multiplier=act_multiplier)
            upstream_key_values.append(key_value)

        for op in self.downstream_layers:
            key_value = EqualizationHelper.key_value_from_downstream(op=op)
            downstream_key_values.append(key_value)

        upstream_key_values   = self.reduce_by_axis(upstream_key_values, method=method)
        downstream_key_values = self.reduce_by_axis(downstream_key_values, method=method)

        # calculate scale
        scale = self.calculate_scale(
            upstream_key_values=upstream_key_values,
            downstream_key_values=downstream_key_values,
            value_threshold=value_threshold)

        # write back all params
        for op in self.upstream_layers:
            EqualizationHelper.scale_to_upstream(op, scale)

        for op in self.downstream_layers:
            EqualizationHelper.scale_to_downstream(op, scale)

    def channel_split(
        self, 
        value_threshold: float,
        including_bias: bool):
        pass

    def calculate_scale(
        self, upstream_key_values: torch.Tensor,
        downstream_key_values: torch.Tensor,
        value_threshold: float, scale_clip_value: float = 10):
        scale = 1 / torch.sqrt(upstream_key_values / downstream_key_values)
        scale = torch.clamp(scale, 1 / scale_clip_value, scale_clip_value)
        scale[(upstream_key_values + downstream_key_values) < value_threshold] = 1
        return scale

    def reduce_by_axis(
        self,
        params: List[torch.Tensor],
        method: EqualizationMethod,
        axis: int=1,
    ) -> torch.Tensor:
        params = torch.cat(params, axis=axis)
        if method is EqualizationMethod.ABSOLUTE_MAX:
            return torch.max(torch.abs(params), axis=axis)[0]

        elif method is EqualizationMethod.ABSOLUTE_MEAN:
            return torch.mean(torch.abs(params), axis=axis)

        elif method is EqualizationMethod.SQUARE_MAX:
            return torch.max(torch.square(params), axis=axis)[0]

        elif method is EqualizationMethod.SQUARE_MEAN:
            return torch.mean(torch.square(params), axis=axis)

        else:
            raise NotImplementedError('Equalization method %s is not support.' % str(method))
