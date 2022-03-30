from enum import Enum
from typing import List

import torch
from ppq.IR import Operation
from tqdm import tqdm as Progressbar


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


class EqualizationPair:
    def __init__(
        self,
        all_upstream_layers: List[Operation],
        all_downstream_layers: List[Operation],
        method:EqualizationMethod=EqualizationMethod.ABSOLUTE_MAX
    ):
        """

            EqualizationPair - 一个数据结构，封装了 equalization 的核心数据抽象和执行逻辑
                               a class encapsulating execution logic of equalization

                在 self.all_upstream_layers 包含了 equalization 操作中的所有上游层(op)
                self.all_upstream_layers contain all upstream ops 

                在 self.all_downstream_layers 包含了 equalization 操作中的所有下游层(op)
                self.all_downstream_layers contain all downstream ops

            一个 EqualizationPair 结构记录了参与 equalization 变换的所有相关层与其局部图结构信息
            从而支持在局部子图上的 equalization 操作
            An EqualizationPair records all relevant ops participating in the equalization
            transformation, thus supporting equalization on local subgraphs
        Args:
            all_upstream_layers (list): 
                equalization 操作中的所有上游层(op)

            all_downstream_layers (list): 
                equalization 操作中的所有下游层(op)

            method (EqualizationMethod, optional): 
                equalization 操作中，变换系数 s 的计算方式. Defaults to EqualizationMethod.ABSOLUTE_MAX.

        """
        self.upstream_layers = all_upstream_layers
        self.downstream_layers = all_downstream_layers
        self.method = method


    def layerwise_equalize(
        self,
        weight_threshold: float,
        including_bias: bool
    ):
        # extract all params from upstream_layers
        upstream_params, downstream_params = [], []
        for upstream_layer in self.upstream_layers:
            assert upstream_layer.type in ('Conv', 'ConvTranspose', 'Gemm'), (
            'Only Conv or Linear layer is support in layerwise equalization now, '
            'but %s got' % upstream_layer.type)
            
            if upstream_layer.type == 'ConvTranspose':
                # weight shape is: [input channel, output channel / group, kernel, kernel]
                weight, bias = self.get_convtranspose2d_params(upstream_layer, including_bias)
                num_of_groups = upstream_layer.attributes.get('group', 1)
                weight = torch.reshape(weight, (num_of_groups, weight.shape[0] // num_of_groups) + weight.shape[1:])
                weight = weight.permute(0, 2, 1, 3, 4)
                weight = weight.reshape(weight.shape[0] * weight.shape[1], -1)

                upstream_params.append(weight)
                if including_bias and bias is not None:
                    upstream_params.append(torch.reshape(bias, (weight.shape[1] * num_of_groups, 1)))

            elif upstream_layer.type == 'Conv':
                # weight shape is: [output channel, input channel, kernel, kernel]
                weight, bias = self.get_conv2d_params(upstream_layer, including_bias)
                weight = torch.reshape(weight, (weight.shape[0], -1))

                upstream_params.append(weight)
                if including_bias and bias is not None: 
                    upstream_params.append(torch.reshape(bias, (weight.shape[0], 1)))

            elif upstream_layer.type == 'Gemm':
                # weight shape is: [output channel, input channel]
                weight, bias = self.get_linear_params(upstream_layer, including_bias)

                upstream_params.append(weight)
                if including_bias and bias is not None: 
                    upstream_params.append(torch.reshape(bias, (weight.shape[0], 1)))

        # extract all params from downstream_layers
        for downstream_layer in self.downstream_layers:
            assert downstream_layer.type in ('Conv', 'ConvTranspose', 'Gemm'), (
            'Only Conv or Linear layer is support in layerwise equalization now, '
            'but %s got' % downstream_layer.type)

            if downstream_layer.type == 'Conv':
                # weight shape is: [output channel, input channel // num_of_groups, kernel, kernel]
                weight, bias = self.get_conv2d_params(downstream_layer, False)

                # for group convolution, we have to select its weight by group
                num_of_groups = downstream_layer.attributes.get('group', 1)

                weight = torch.reshape(weight, (num_of_groups, weight.shape[0] // num_of_groups) + weight.shape[1: ])
                weight = weight.permute(2, 0, 1, 3, 4)
                weight = torch.reshape(weight, (weight.shape[0] * weight.shape[1], -1))

                downstream_params.append(weight)
            
            elif downstream_layer.type == 'ConvTranspose':
                # weight shape is: [input channel, output channel // num_of_groups, kernel, kernel]
                weight, bias = self.get_convtranspose2d_params(downstream_layer, False)

                # for group convolution, we have to select its weight by group
                num_of_groups = downstream_layer.attributes.get('group', 1)

                weight = torch.reshape(weight, (weight.shape[0], -1))

                downstream_params.append(weight)
            
            elif downstream_layer.type == 'Gemm':
                # weight shape is: [output channel, input channel]
                weight, bias = self.get_linear_params(downstream_layer, False)
                downstream_params.append(weight.permute(1, 0))

        # format all params
        upstream_key_values   = self.reduce_by_axis(upstream_params, method=self.method, aggerate_axis=1)
        downstream_key_values = self.reduce_by_axis(downstream_params, method=self.method, aggerate_axis=1)

        # calculate scale
        scale = self.calculate_scale(
            upstream_key_values=upstream_key_values,
            downstream_key_values=downstream_key_values,
            minval_threshold=weight_threshold
        )

        # write back all params
        for upstream_layer in self.upstream_layers:
            if upstream_layer.type == 'ConvTranspose':
                weight, bias = self.get_convtranspose2d_params(upstream_layer, True)
                num_of_groups = upstream_layer.attributes.get('group', 1)
                weight = torch.reshape(weight, (num_of_groups, weight.shape[0] // num_of_groups) + weight.shape[1:])
                weight *= torch.reshape(scale, (num_of_groups, 1, -1, 1, 1))
                weight = torch.reshape(weight, (weight.shape[0] * weight.shape[1], ) + weight.shape[2:])
                if bias is not None:
                    bias *= scale
                self.set_convtranspose2d_params(upstream_layer, bias, weight)
            
            elif upstream_layer.type == 'Conv':
                weight, bias = self.get_conv2d_params(upstream_layer, True)
                weight *= torch.reshape(scale, (-1, 1, 1, 1))
                if bias is not None:
                    bias *= scale
                self.set_conv2d_params(upstream_layer, bias, weight)

            elif upstream_layer.type == 'Gemm':
                weight, bias = self.get_linear_params(upstream_layer, True)
                weight *= torch.reshape(scale, (-1, 1))
                if bias is not None:
                    bias *= scale
                self.set_linear_params(upstream_layer, bias, weight)

        for downstream_layer in self.downstream_layers:
            
            if downstream_layer.type == 'ConvTranspose':
                weight, bias = self.get_convtranspose2d_params(downstream_layer, False)
                # for group convolution, we have to select its weight by group

                weight /= torch.reshape(scale, (-1, 1, 1, 1))
                self.set_convtranspose2d_params(downstream_layer, bias, weight)
            
            elif downstream_layer.type == 'Conv':
                weight, bias = self.get_conv2d_params(downstream_layer, False)
                # for group convolution, we have to select its weight by group
                num_of_groups = downstream_layer.attributes.get('group', 1)

                weight = torch.reshape(weight, (num_of_groups, weight.shape[0] // num_of_groups) + weight.shape[1: ])
                weight /= torch.reshape(scale, (num_of_groups, 1, -1, 1, 1))
                weight = torch.reshape(weight, (weight.shape[1] * num_of_groups, ) + weight.shape[2: ])
                self.set_conv2d_params(downstream_layer, bias, weight)

            elif downstream_layer.type == 'Gemm':
                weight, bias = self.get_linear_params(downstream_layer, False)
                weight /= torch.reshape(scale, (1, -1))

                self.set_linear_params(downstream_layer, bias, weight)


    def display(self) -> str:
        for layer in self.upstream_layers + self.downstream_layers:
            if layer.type == 'Conv':
                weight, bias = self.get_conv2d_params(layer, including_bias=True)
            elif layer.type == 'Gemm':
                weight, bias = self.get_linear_params(layer, including_bias=True)
            else:
                raise Exception('Expect conv layer or linear layer only, while %s was given.' % layer.type)

            print('Stat of Layer %s: \t{%.4f}(Weight Max),\t{%.4f}(Weight Std)\t{%.4f}(Bias Max),\t{%.4f}(Bias Std)' % (
                layer.name,
                torch.max(torch.abs(weight)),
                torch.std(weight),
                torch.max(torch.abs(bias)) if bias is not None else 0,
                torch.std(bias) if bias is not None else 0
            ))
        print('--- Layer-wise Equalization display end. ---')


    def __str__(self) -> str:
        return (
            'Class EqualizationPair: '
            '[all_upstream_layers: %s, all_downstream_layers: %s]' % 
            (self.upstream_layers, self.downstream_layers))


    def get_conv2d_params(self, conv: Operation, including_bias: bool):
        
        assert conv.type == 'Conv', (
            'Except input object with type Conv, but %s got' % conv.type)

        weight, bias = conv.parameters[0].value, None
        if including_bias and len(conv.parameters) > 1:
            bias = conv.parameters[1].value

        return weight, bias


    def get_convtranspose2d_params(self, conv: Operation, including_bias: bool):
        
        assert conv.type == 'ConvTranspose', (
            'Except input object with type Conv, but %s got' % conv.type)

        weight, bias = conv.parameters[0].value, None
        if including_bias and len(conv.parameters) > 1:
            bias = conv.parameters[1].value

        return weight, bias


    def get_linear_params(self, linear: Operation, including_bias: bool):

        assert linear.type == 'Gemm', (
            'Except input object with type Gemm, but %s got' % linear.type)

        weight, bias = linear.parameters[0].value, None
        if including_bias and len(linear.parameters) > 1:
            bias = linear.parameters[1].value

        if not linear.attributes.get('transB', 0):
            weight = torch.transpose(weight, 1, 0)
        if bias is not None: return weight, bias
        else: return [weight, None]


    def set_conv2d_params(self, conv: Operation, bias: torch.Tensor, weight: torch.Tensor):

        assert conv.type == 'Conv', (
            'Except input object with type Conv, but %s got' % conv.type)
        
        conv.parameters[0].value = weight
        if bias is not None and len(conv.parameters) > 1:
            conv.parameters[1].value = bias

    def set_convtranspose2d_params(self, conv: Operation, bias: torch.Tensor, weight: torch.Tensor):
        
        assert conv.type == 'ConvTranspose', (
            'Except input object with type Conv, but %s got' % conv.type)

        conv.parameters[0].value = weight
        if bias is not None and len(conv.parameters) > 1:
            conv.parameters[1].value = bias


    def set_linear_params(self, linear: Operation, bias: torch.Tensor, weight: torch.Tensor):

        assert linear.type == 'Gemm', (
            'Except input object with type Gemm, but %s got' % linear.type)
        
        if not linear.attributes.get('transB', 0):
            weight = torch.transpose(weight, 1, 0)
        linear.parameters[0].value = weight
        if bias is not None and len(linear.parameters) > 1:
            linear.parameters[1].value = bias


    def calculate_scale(
        self,
        upstream_key_values: torch.Tensor,
        downstream_key_values: torch.Tensor,
        minval_threshold: float,
        scale_clip_value: float = 10,
    ):
        scale = 1 / torch.sqrt(upstream_key_values / downstream_key_values)
        scale = torch.clamp(scale, 1 / scale_clip_value, scale_clip_value)
        scale[(upstream_key_values + downstream_key_values) < minval_threshold] = 1

        return scale

    def reduce_by_axis(
        self,
        params: List[torch.Tensor],
        method: EqualizationMethod,
        aggerate_axis: int=1,
    ) -> torch.Tensor:
        params = torch.cat(params, axis=aggerate_axis)
        if method is EqualizationMethod.ABSOLUTE_MAX:
            return torch.max(torch.abs(params), axis=aggerate_axis)[0]

        elif method is EqualizationMethod.ABSOLUTE_MEAN:
            return torch.mean(torch.abs(params), axis=aggerate_axis)

        elif method is EqualizationMethod.SQUARE_MAX:
            return torch.max(torch.square(params), axis=aggerate_axis)[0]

        elif method is EqualizationMethod.SQUARE_MEAN:
            return torch.mean(torch.abs(params), axis=aggerate_axis)
            
        else:
            raise NotImplementedError('Equalization method %s is not support.' % str(method))


def layerwise_equalization(
    equalization_pairs: List[EqualizationPair],
    weight_threshold: float = 0.5,
    incluing_bias: bool = True,
    iteration: int = 10,
    verbose: bool = False
):
    """

        layerwise_equalization - 层间权重均一化，使用该函数从大尺度上拉平各个层之间的权重与bias，从而使得量化结果更加精确
                                this func equalizes weights and biases between differenr layers and reduces
                                quantization error
        一次 equalization 操作是指利用性质: C * ( AX + b ) = C/s * ( AsX + b )，所作的恒等变换，其中s为对角矩阵
        one equalization step refers to use above formula to do equivalent transformation

        通过上述变换以及精心选取的 s，可以使得权重矩阵 A, b, C 的数值大小尽可能接近，从而使得量化更加精准
        we could make numerical ranges of weight matrice of different layers become as similar as possible by 
        choosing approriate s, reducing quantization error, while preserving correct results

        注意目前只支持关于 CONV, GEMM 的权重拉平策略
        for now only CONV and GEMM support equalization
        相关论文:

        "Markus Nagel et al., Data-Free Quantization through Weight Equalization and Bias Correction" arXiv:1906.04721, 2019.

    Args:
        equalization_pairs (list): 所有需要被拉平的层的组合结构 all equalization pairs. 
        weight_threshold (float, optional): 参与权重均一化的最小权重 minimum weight for weight equalization defaults to 0.5.
        incluing_bias (bool, optional): 是否执行带bias的权重均一化 whether to include bias defaults to True.
        iteration (int, optional): 均一化执行次数 num of equalization iterations defaults to 10.
        verbose (bool, optional): 是否输出均一化的相关结果，这将打印均一化前后的权重变化情况 whether to print details defaults to True.
    """

    if verbose:
        for equalization_pair in equalization_pairs:
            equalization_pair.display()

    print(f'{len(equalization_pairs)} equalization pair(s) was found, ready to run optimization.')
    for iter_times in Progressbar(range(iteration), desc='Layerwise Equalization', total=iteration):
        for equalization_pair in equalization_pairs:
            assert isinstance(equalization_pair, EqualizationPair), (
                "Input equalization pairs should be encapsuled with class EqualizationPair")

            equalization_pair.layerwise_equalize(
                weight_threshold=weight_threshold,
                including_bias=incluing_bias
            )

    if verbose:
        for equalization_pair in equalization_pairs:
            equalization_pair.display()