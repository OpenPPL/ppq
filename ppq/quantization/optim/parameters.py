from typing import Iterable

import torch
from ppq.core import (ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates)
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import GraphCommandProcessor, QuantableOperation
from ppq.quantization.observer import OperationObserver

from .base import QuantizationOptimizationPass


class PassiveParameterQuantizePass(QuantizationOptimizationPass):
    """PPQ PassiveParameterQuantizePass completes the quantization of passive
    parameters It is the final parameter procedure during the quantization
    process for the most part.

    passive parameters are those parameters which required to be quantized during the computing,
    while holds no standalone quantization configuration, such as bias, padding value, etc.

    PassiveParameterQuantizePass will try to quantize all passive parameters during its "optimize" function
    if there is any parameter that can not be quantized with current quantization configuration(
        if their positive counterparts have not been properly quantized), an error will be thrown.
    """
    def __init__(self, bias_scale_multiplier=1):
        self.scale_multiplier = bias_scale_multiplier
        super().__init__(name='PPQ Passive Parameter Quantization')

    def optimize(self, processor: GraphCommandProcessor,
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:

        def check_state(state: QuantizationStates):
            return state in {
                QuantizationStates.SLAVE,
                QuantizationStates.ACTIVATED,
                QuantizationStates.BAKED,
                QuantizationStates.OVERLAPPED
            }

        graph = processor.graph
        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
                # inputs are [input value, weight, bias(optional)]
                if op.num_of_input == 3:
                    i_cfg, w_cfg, b_cfg = op.config.input_quantization_config
                    if b_cfg.state != QuantizationStates.PASSIVE_INIT: continue
                    if not check_state(w_cfg.state):
                        raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                            'cause weight has not been correctly quantized.')
                    if not check_state(i_cfg.state):
                        raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                            'cause input has not been correctly quantized.')
                    w_cfg = w_cfg.dominated_by
                    i_cfg  = i_cfg.dominated_by

                    b_cfg.scale  = w_cfg.scale * i_cfg.scale * self.scale_multiplier
                    b_cfg.state  = QuantizationStates.PASSIVE
                    b_cfg.offset = torch.zeros_like(b_cfg.scale)
                    assert not b_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                        'Passive parameter does not support ASYMMETRICAL quantization')

            if op.type in {'Clip'}:
                # inputs are [input value, min[optional], max[optional]]
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(f'Can not quantize clip value of layer {op.name}, '
                        'cause input has not been correctly quantized.')
                i_cfg = i_cfg.dominated_by
                for config in op.config.input_quantization_config[1: ]:
                    if config.state != QuantizationStates.PASSIVE_INIT: continue

                    config.scale  = i_cfg.scale
                    config.offset = i_cfg.offset
                    config.state  = QuantizationStates.PASSIVE

            if op.type in {'Pad'}:
                # inputs are [input value, pad[shape-related], pad value[optional]]
                if op.num_of_input != 3: continue
                i_cfg = op.config.input_quantization_config[0]
                if i_cfg.state != QuantizationStates.PASSIVE_INIT: continue

                if not check_state(i_cfg.state):
                    raise PermissionError(f'Can not quantize pad value of layer {op.name}, '
                        'cause input has not been correctly quantized.')
                i_cfg = i_cfg.dominated_by
                pad_config = op.config.input_quantization_config[-1]
                pad_config.scale  = i_cfg.scale
                pad_config.offset = i_cfg.offset
                pad_config.state  = QuantizationStates.PASSIVE


class ParameterQuantizePass(QuantizationOptimizationPass):
    """PPQ PassiveParameterQuantizePass completes the quantization of positive
    parameters. By default, all parameters with initial state will be quantized
    during this optimization, all non-parameter tensors will be excluded from
    this pass by temporary dequantization.

    Then, operation observers will be established automatically to record necessary statistics,
        observers are also responsible for rendering quantization configuration (computing scale and offset).

    This pass needs no data, however it uses fake data to finish a dummy forward process.
    see also: TorchExecutor.dummy_forward function
    """
    def __init__(self, method: str = None):
        self._method = method
        super().__init__(name='PPQ Parameter Quantization Pass')

    def optimize(
        self,
        processor: GraphCommandProcessor,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        # build observer and hook for each quantable operation
        hooks, observers, state_records = {}, {}, {}
        for op_name, operation in processor.graph.operations.items():
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
                operation=executor._graph.operations[op_name],
                monitor_outputs=False, monitor_inputs=False)
            observers[op_name] = observer
            hooks[op_name]     = observer.hook

        # dummy forward, quant all parameter.
        assert isinstance(executor, TorchExecutor), \
            'ParameterQuantizePass Only support TorchExecutor now.'
        executor.dummy_forward(hooks=hooks)

        # render quantization config, restore non-parameter quantization state
        for op_name, operation in processor.graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue

            for cfg, var in operation.config_with_variable:
                if not var.is_parameter:
                    cfg.state = state_records[cfg]

            observer = observers[op_name]
            assert isinstance(observer, OperationObserver)
            observer.render_quantization_config()
            observer.report()
