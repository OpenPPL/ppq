from typing import Iterable

import torch
from ppq.core import (ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates)
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import GraphCommandProcesser, QuantableOperation
from ppq.quantization.observer import OperationObserver

from .base import QuantizationOptimizationPass


class PassiveParameterQuantizePass(QuantizationOptimizationPass):
    """
    PPQ PassiveParameterQuantizePass completes the quantization of passive parameters
    It is the final parameter procedure during the quantization process for the most part.

    passive parameters are those parameters which required to be quantized during the computing,
    while holds no standalone quantization configuration, such as bias, padding value, etc.

    PassiveParameterQuantizePass will try to quantize all passive parameters during its "optimize" function
    if there is any parameter that can not be quantized with current quantization configuration(
        if their positive counterparts have not been properly quantized), an error will be thrown.
    """
    def __init__(self, bias_scale_multiplier=2):
        self.scale_multiplier = bias_scale_multiplier
        super().__init__(name='PPQ Passive Parameter Quantization')

    def optimize(self, processer: GraphCommandProcesser, 
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:

        def check_state(state: QuantizationStates): 
            return state in {
                QuantizationStates.ACTIVATED,
                QuantizationStates.BAKED,
                QuantizationStates.OVERLAPPED
            }

        graph = processer.graph
        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
                # inputs are [input value, weight, bias(optional)]
                if op.num_of_input == 3:
                    weight_config = op.config.input_quantization_config[1]
                    input_config = op.config.input_quantization_config[0]
                    if not check_state(weight_config.state):
                        raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                            'cause weight has not been correctly quantized.')
                    if not check_state(input_config.state):
                        raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                            'cause input has not been correctly quantized.')
                    weight_config = weight_config.dominated_by
                    input_config  = input_config.dominated_by

                    bias_config = op.config.input_quantization_config[-1]
                    bias_config.scale  = weight_config.scale * input_config.scale * self.scale_multiplier
                    bias_config.state  = QuantizationStates.PASSIVE
                    bias_config.offset = torch.zeros_like(bias_config.scale)
                    assert not bias_config.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                        'Passive parameter does not support ASYMMETRICAL quantization')

            if op.type in {'Clip'}:
                # inputs are [input value, min[optional], max[optional]]
                input_config = op.config.input_quantization_config[0]
                if not check_state(input_config.state):
                    raise PermissionError(f'Can not quantize clip value of layer {op.name}, '
                        'cause input has not been correctly quantized.')
                input_config = input_config.dominated_by
                for config in op.config.input_quantization_config[1: ]:
                    config.scale  = input_config.scale
                    config.offset = input_config.offset
                    config.state  = QuantizationStates.PASSIVE

            if op.type in {'Pad'}:
                # inputs are [input value, pad[shape-related], pad value[optional]]
                if op.num_of_input != 3: continue
                input_config = op.config.input_quantization_config[0]
                if not check_state(input_config.state):
                    raise PermissionError(f'Can not quantize pad value of layer {op.name}, '
                        'cause input has not been correctly quantized.')
                input_config = input_config.dominated_by
                pad_config = op.config.input_quantization_config[-1]
                pad_config.scale  = input_config.scale
                pad_config.offset = input_config.offset
                pad_config.state  = QuantizationStates.PASSIVE


class ParameterQuantizePass(QuantizationOptimizationPass):
    """
    PPQ PassiveParameterQuantizePass completes the quantization of positive parameters.
    By default, all parameters with initial state will be quantized during this optimization,
        all non-parameter tensors will be excluded from this pass by temporary dequantization.

    Then, operation observers will be established automatically to record necessary statistics,
        observers are aslo responsible for rendering quantization configuration (computing scale and offset). 

    This pass needs no data, however it uses fake data to finish a dummy forward process.
    see aslo: TorchExecutor.dummy_forward function
    """
    def __init__(self, method: str = None):
        self._method = method
        super().__init__(name='PPQ Parameter Quantization Pass')

    def optimize(
        self, 
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        # build observer and hook for each quantable operation
        hooks, observers, state_records = {}, {}, {}
        for op_name, operation in processer.graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue
            
            for config, var in operation.config_with_variable:
                # deactivate non-parameter variable quantization just for now
                if not var.is_parameter:
                    state_records[config] = config.state
                    config.state = QuantizationStates.DEQUANTIZED
                elif self._method is not None:
                    # override quantizer's setting if necessary
                    config.observer_algorithm = self._method

            observer = OperationObserver(
                opeartion=executor._graph.operations[op_name], 
                monitor_outputs=False, monitor_inputs=False)
            observers[op_name] = observer
            hooks[op_name]     = observer.hook

        # dummy forward, quant all parameter. 
        assert isinstance(executor, TorchExecutor), \
            'ParameterQuantizePass Only support TorchExecutor now.'
        executor.dummy_forward(hooks=hooks)

        # render quantization config, restore non-parameter quantization state
        for op_name, operation in processer.graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue

            for cfg, var in operation.config_with_variable:
                if not var.is_parameter:
                    cfg.state = state_records[cfg]

            observer = observers[op_name]
            assert isinstance(observer, OperationObserver)
            observer.render_quantization_config()
            observer.report()