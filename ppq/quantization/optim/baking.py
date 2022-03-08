from typing import Iterable

import torch
from ppq.core import QuantizationStates, empty_ppq_cache
from ppq.executor import BaseGraphExecutor
from ppq.IR import GraphCommandProcesser, Operation, QuantableOperation
from ppq.quantization.qfunction import BaseQuantFunction

from .base import QuantizationOptimizationPass


class ParameterBakingPass(QuantizationOptimizationPass):
    """
    ParameterBakingPass is a useful tool for quantization simulation acceleration.
        By default quantizer will bake network parameters once all quantization procedures are finished.
    For a typical Convolution layer or Gemm layer, which has a non-empty bias tensor,
    ParameterBakingPass will speed up the layer execution by 30%-50%.

    ParameterBakingPass will rewrite layer parameters with their quantized version,
        the quantization procedure will strictly follow layer quantization configuration.
    Once the quantization process finished, this pass will change all parameter quantization configuration states
        to QuantizationStates.BAKED.

    State QuantizationStates.BAKED indicates corresponding tensor has been pre-quantized and its value
        can be used without further quantization, executor will directly use a baked value during execution.

    ATTENTION: value is baked inplace, so to say it will rewrite all network parameters.
    ATTENTION: For platforms using int32 accumulator, a float32 bias tensor might lose precision
        during the simulation. If you want PPQ simulator to have a consistent result with hardware, it is
        highly-recommended to calling ParameterBakingPass before deployment, baking procedure will limit bias
        precision to 23 bits (float32 only has 23 fraction bits).
    Args:
        quantize_function (BaseQuantFunction): a BaseQuantFunction instance to quantize all parameters.
    """
    def __init__(self, quantize_function: BaseQuantFunction) -> None:
        super().__init__(name='PPQ Parameter Baking Pass')
        self._quantize_function = quantize_function

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
            operation.baking_parameters(self._quantize_function)


class ConstantBakingPass(QuantizationOptimizationPass):
    def __init__(self, quantize_function: BaseQuantFunction) -> None:
        super().__init__(name='PPQ Contant Baking Pass')
        self._quantize_function = quantize_function

    @ empty_ppq_cache
    def optimize(self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, **kwargs) -> None:
        raise NotImplementedError('This pass has been removed from current PPQ version.')
    
        graph = processer.graph
        for _, operation in graph.operations.items():
            assert isinstance(operation, Operation)

            if operation.type != 'Constant': continue
            assert torch.is_tensor(operation.attributes['value']), \
                'Constant Baking Pass needs all constants to be torch.tensor.'

            # check if all down-stream opeartions are Quantable Operations
            # up-stream constant can be pre-quantized only if all down-stream opeartions are Quantable.
            # And all down-stream opeartions should share a same quant_config.
            down_stream_ops = operation.outputs[0].dest_ops
            down_stream_idx = operation.outputs[0].dest_idx
            quant_configs, quant_flag = [], True
            for op, output_idx in zip(down_stream_ops, down_stream_idx):
                if not isinstance(op, QuantableOperation): quant_flag = False
                else:
                    quant_config = op.config.input_quantization_config[output_idx]
                    if not QuantizationStates.is_activated(quant_config.state): quant_flag = False
                    quant_configs.append(quant_config)
            if len(down_stream_ops) == 0: raise ValueError(
                'Oops, isolated constant operation can not go through PPQ Contant Baking Pass')

            # ready for constant baking.
            if all([config == quant_configs[0] for config in quant_configs]) and quant_flag:
                for config in quant_configs: config.state = QuantizationStates.BAKED
                operation.attributes['value'] = self._quantize_function(
                    operation.attributes['value'], quantization_config=quant_configs[0])
