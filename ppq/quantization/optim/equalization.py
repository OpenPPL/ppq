from collections import defaultdict
from typing import Callable, Dict, Iterable, List

import torch
from ppq.core import empty_ppq_cache
from ppq.executor import BaseGraphExecutor
from ppq.executor.torch import TorchExecutor
from ppq.IR import (BaseGraph, Operation, QuantableOperation,
                    SearchableGraph, TraversalCommand)
from ppq.IR.base.graph import BaseGraph
from ppq.quantization.algorithm.equalization import EqualizationPair
from tqdm import tqdm

from .base import QuantizationOptimizationPass

OPTIMIZATION_LAYERTYPE_CONFIG = {
    1: {'Relu', 'MaxPool', 'GlobalMaxPool', 'PRelu', 'AveragePool', 'GlobalAveragePool'},                     # level - 1 optimize
    2: {'Relu', 'MaxPool', 'GlobalMaxPool', 'Add', 'Sub', 'PRelu', 'AveragePool', 'GlobalAveragePool'},       # level - 2 optimize
}
EQUALIZATION_OPERATION_TYPE = {'Conv', 'Gemm', 'ConvTranspose'}


class LayerwiseEqualizationPass(QuantizationOptimizationPass):
    """
    ## Layer-wise Equalization Pass(层间权重均衡过程)

    Weight distributions can differ strongly between output channels, 
    using only one quantization scale, per-tensor quantization has its trouble for representing the value among channels.

    For example, in the case where one channel has weights in the range [−128, 128] and another channel has weights in the range (−0.5, 0.5), 
    the weights in the latter channel will all be quantized to 0 when quantizing to 8-bits.

    Hopefully, the performance can be improved by adjusting the weights for each output channel such that their ranges are more similar.
    
    Formula:
    
            Take 2 convolution layers as an example
            
            Where Y = W_2 * (W_1 * X + b_1) + b_2
            
            Adjusting W_1, W_2 by a scale factor s:
            
            Y = W_2 / s * (W_1 * s * X + b_1 * s) + b_2

            Where s has the same dimension as the output channel of W_1

    This method is called as Layer-wise Equalization, which is proposed by Markus Nagel.
    
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf

    self, iterations: int, weight_threshold: float = 0.5,
    including_bias: bool = False, including_activation: bool = False,
    bias_multiplier: float = 0.5, activation_mutiplier: float = 0.5,
    interested_layers: List[str] = None, optimize_level: int = 2,
    verbose:bool = False
    
    ### Parameters:

    * iterations(int):
            
            Integer value of Algorithm iterations.
            
            More iterations will give more plainness in your weight distribution,
            iteration like 100 can flatten all the parameter in your network to a same level.
            
            You are not recommended to iterate until value converges, 
            in some cases stop iteration earlier will give you a better performance.
            
    * weight_threshold(float)

            A threshold that stops processing value that is too small.

            By default, the scale factor of equalization method is computed as sqrt(max(abs(W_1)) / max(abs(W_2))),
            the maximum value of W_2 can be very small(like 1e-14), while the maximum value W_1 can be 0.5.

            In this case, the computed scale factor is 1e7, the optimization will loss its numerical stability and even give an unreasonable result.

            To prevent the scale factor becoming too large, ppq clips all the value smaller than this threshold before iterations.
            
            This parameter will significantly affects the optimization result.
            
            Recommended values are 0, 0.5, 2.

    # including_bias(bool)

            Whether to include bias in computing scale factor.
            
            If including_bias is True, the scale factor will be computed as sqrt(max(abs(W_1 : b_1)) / max(abs(W_2 : b_2)))
            
            Where W_1 : b_1 mean an augmented matrix with W_1 and b_1

    # including_bias(float)
    
            Only take effects when including_bias = True

            the scale factor will be computed as sqrt(max(abs(W_1 : b_1 * bias_multiplier)) / max(abs(W_2 : b_2 * bias_multiplier)))

            This is an correction term for bias.

    # including_activation(bool)

            Same as the parameter including_bias, whether to include activation in computing scale factor.

    # activation_multiplier(float)
    
            Same as the including_bias, this is an correction term for activation.
            
    # optimize_level(int)
    
            level - 1: equalization will only cross ('Relu', 'MaxPool', 'GlobalMaxPool', 'PRelu', 'AveragePool', 'GlobalAveragePool')
            
            level - 2: equalization will cross ('Relu', 'MaxPool', 'GlobalMaxPool', 'Add', 'Sub', 'PRelu', 'AveragePool', 'GlobalAveragePool')
            
            Here is an example for illustrating the difference, if we got a graph like: 

                Conv1 - Relu - Conv2

            Both level - 1 and level - 2 optimization can find there is a equalization pair: (Conv1 - Conv2).

            however for a complex graph like: 
    
                Conv1 - Add - Conv2
            
            level - 1 optimization will simply skip Conv1 and Conv2.

            level - 2 optimization will trace another input from Add, and then PPQ will take all the input operations of Add
            as the upstream layers in equalization.

            PPQ use graph search engine for pasring graph structure, check ppq.IR.search.py for more information.
    
    # interested_layers(List[str])
    
            Only layer that listed in interested_layers will be processed by this pass.
            
            If interested_layers is None or empty list, all the layers will be processed.

    ### Warning:
    You can not compare a equalized graph with an unequalized graph layer by layer,
    since equalization pass guarantees only the output of your network will be kept as same,
    the intermediate result can be changed rapidly.

    Since then, PPQ invokes this pass before network quantization.
            
    ### Usage
    Layer-wise equalization are designed for per-layer quantization.
    
    |              | Symmetrical | Asymmetrical | Per-chanel    | Per-tensor |
    | ---          | ---         | ---          | ---           | ---        |
    | equalizaiton |             |              | Not recommend |            |
    
    Layer-wise Equalization Pass should be invoked in pre-quant optimization pipeline.
    (pre-quant optimization pipeline take effects before quantization)

    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.equalization = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)

    You can manually create this optimization by:

        from ppq import LayerwiseEqualizationPass

        optim = LayerwiseEqualizationPass()

    """
    def __init__(
        self, iterations: int, weight_threshold: float = 0.5,
        including_bias: bool = False, including_act: bool = False,
        bias_multiplier: float = 0.5, act_multiplier: float = 0.5,
        interested_layers: List[str] = None, optimize_level: int = 2,
        verbose:bool = False) -> None:
        """PPQ Customized Layerwise Equalization Pass.

        This is a highly customized implementation of layerwise equalization.
        With PPQ graph searching engine, this implementation can equalize multiple layer at once,
            Even some layers are behind Add, Sub, Pooling.

        Not only weight, bias and activation are also taken into consideration with this implementation.
        if including_bias and including_activation set as True, all weight, bias, activation will be pull
            equal with this function.

        Args:
            iterations (int): Equalization iterations.
            weight_threshold (float, optional):
                Value threshold, all weight that belows that value will keep unchanged through this function.
                ATTENTION: this threshold will greatly affects your equalization performance.
                Defaults to 0.5. recommend to try 0.5, 2, 0

            including_bias (bool, optional):
                whether to include bias into consideration.
                ATTENTION: Some hardware use int16 accumulator or even worse
                    set this to be True if your hardware does not allow a 32-bit bias.
                Defaults to False.

            including_act (bool, optional):
                whether to include activation into consideration.
                Defaults to False.

            bias_multiplier (float, optional):
                a multiplier to bias, if not necessary do not change this.
                Defaults to 0.5.

            act_multiplier (float, optional):
                a multiplier to activation, if not necessary do not change this.
                Defaults to 0.5.

            interested_layers (List[str]):
                a layer collection contains all layers which need to be equalized.
                if None is given, suppose all layer need to be equalized.

            optimize_level (int, optional): [description]. Defaults to 2.
            verbose (bool, optional): [description]. Defaults to False.
        """
        self.optimize_level         = optimize_level
        self.iterations             = iterations
        self.value_threshold        = weight_threshold

        self.including_bias         = including_bias
        self.bias_multiplier        = bias_multiplier

        self.including_act          = including_act
        self.act_multiplier         = act_multiplier

        self.interested_layers      = interested_layers
        self.verbose                = verbose
        super().__init__(name='PPQ Layerwise Equalization Pass')

    def find_equalization_pair(
        self, graph: BaseGraph, interested_operations: List[Operation]
    ) -> List[EqualizationPair]:

        # create a PPQ graph search engine.
        search_engine = SearchableGraph(graph)

        visited_ops, pairs = set(), []
        for operation in interested_operations:
            if operation in visited_ops: continue

            # skip operation that can not be equalized
            if operation.type not in EQUALIZATION_OPERATION_TYPE: continue

            # forward matching equalization pair.
            forward_matchings = search_engine(TraversalCommand(
                sp_expr=lambda x: x == operation,
                rp_expr=lambda x, y: y.type in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                ep_expr=lambda x: x.type not in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                direction='down'))

            downstream_ops = set()
            for matching in forward_matchings:
                downstream_ops.add(matching[-1])

            # backward matching equalization pair
            forward_matchings = search_engine(TraversalCommand(
                sp_expr=lambda x: x in downstream_ops,
                rp_expr=lambda x, y: y.type in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                ep_expr=lambda x: x.type not in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                direction='up'))

            upstream_ops = set()
            for matching in forward_matchings:
                upstream_ops.add(matching[-1])

            # update pairs to visited.
            visited_ops.update(upstream_ops)

            # check if all operation inside this pair can be properly processed.
            valid_flag = True
            for operation in upstream_ops:
                if operation.type not in EQUALIZATION_OPERATION_TYPE:
                    valid_flag = False

            for operation in downstream_ops:
                if operation.type not in EQUALIZATION_OPERATION_TYPE:
                    valid_flag = False

            if not valid_flag: continue

            # construct a new equalization pair.
            if len(upstream_ops) > 0 and len(downstream_ops) > 0:
                pairs.append(EqualizationPair(
                    upstream_layers=list(upstream_ops),
                    downstream_layers=list(downstream_ops)))
        return pairs

    def collect_activations(self, graph: BaseGraph,
        executor: TorchExecutor, dataloader: Iterable,
        collate_fn: Callable, operations: List[Operation],
        steps: int = 16) -> Dict[str, torch.Tensor]:

        def aggregate(op: Operation, tensor: torch.Tensor):
            if op.type in {'Conv', 'ConvTranspose'}: # Conv result: [n,c,h,w]
                num_of_channel = tensor.shape[1]
                tensor = tensor.transpose(0, 1)
                tensor = tensor.reshape(shape=[num_of_channel, -1])
                tensor = torch.max(tensor.abs(), dim=-1, keepdim=False)[0]
            elif op.type in {'MatMul', 'Gemm'}: # Gemm result: [n, c]
                tensor = tensor.transpose(0, 1)
                tensor = torch.max(tensor.abs(), dim=-1, keepdim=False)[0]
            return tensor

        output_names = []
        for operation in operations:
            assert operation.num_of_output == 1, (
                f'Num of output of layer {operation.name} is supposed to be 1')
            output_names.append(operation.outputs[0].name)

        output_collector = defaultdict(list)
        for idx, batch in tqdm(enumerate(dataloader),
                               desc='Equalization Data Collecting.',
                               total=min(len(dataloader), steps)):
            data = batch
            if collate_fn is not None:
                data = collate_fn(batch)
            outputs = executor.forward(data, output_names=output_names)
            for name, output in zip(output_names, outputs):
                op = graph.variables[name].source_op
                output_collector[name].append(aggregate(op, output).unsqueeze(-1))
            if idx > steps: break

        result = {}
        for name, output in zip(output_names, outputs):
            result[name] = torch.cat(output_collector[name], dim=-1)
        return result

    @ empty_ppq_cache
    def optimize(
        self, graph: BaseGraph, dataloader: Iterable,
        executor: BaseGraphExecutor, collate_fn: Callable, **kwargs
    ) -> None:
        interested_operations = []

        if self.interested_layers is None:

            for operation in graph.operations.values():
                if operation.type in EQUALIZATION_OPERATION_TYPE:
                    interested_operations.append(operation)
        else:

            for name in self.interested_layers:
                if name in graph.operations:
                    interested_operations.append(graph.operations[name])

        pairs = self.find_equalization_pair(
            graph=graph, interested_operations=interested_operations)

        if self.including_act:
            activations = self.collect_activations(
                graph=graph, executor=executor, dataloader=dataloader, collate_fn=collate_fn,
                operations=interested_operations)

            for name, act in activations.items():
                graph.variables[name].value = act # 将激活值写回网络

        print(f'{len(pairs)} equalization pair(s) was found, ready to run optimization.')
        for iter_times in tqdm(range(self.iterations), desc='Layerwise Equalization', total=self.iterations):
            for equalization_pair in pairs:
                equalization_pair.equalize(
                    value_threshold=self.value_threshold,
                    including_bias=self.including_bias,
                    including_act=self.including_act,
                    bias_multiplier=self.bias_multiplier,
                    act_multiplier=self.act_multiplier)

        # equalization progress directly changes fp32 value of weight,
        # store it for following procedure.
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()
