from typing import Iterable

import torch
from ppq.core import QuantizationProperty, QuantizationStates, ppq_warning
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph, QuantableOperation
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
    def __init__(self, bias_scale_multiplier: float = 1, override: bool = False):
        self.scale_multiplier = bias_scale_multiplier
        self._override = override # whether to override existed passive parameter config
        super().__init__(name='PPQ Passive Parameter Quantization')

    def optimize(self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:

        def check_state(state: QuantizationStates):
            return state in {
                QuantizationStates.SLAVE,
                QuantizationStates.ACTIVATED,
                QuantizationStates.BAKED,
                QuantizationStates.OVERLAPPED
            }

        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
                # inputs are [input value, weight, bias(optional)]
                if op.num_of_input == 3:
                    i_cfg, w_cfg, b_cfg = op.config.input_quantization_config

                    # PATCH 2022.07.29 有的时候 bias 是个多维的东西，此时要求前面的维度都是1
                    bias = op.inputs[-1].value
                    if bias is None: raise ValueError(f'Bias Varaible {op.inputs[-1].name} must be constant. '
                                                      'Please check it again.')
                    
                    assert bias.numel() == bias.shape[-1], (
                        f'For op {op.name}, expect Bias shape to be {[bias.numel()]}, '
                        f'however {bias.shape} was given')
                    op.inputs[-1].value = bias.squeeze()
                    # PATCH 2022.08.02 只有一个数的 bias 经过 squeeze 会变成零维的, 再给它多加一维补回来
                    if op.inputs[-1].value.ndim == 0 and op.inputs[-1].value.numel() == 1:
                        op.inputs[-1].value = op.inputs[-1].value.unsqueeze(0)

                    # 在两种情况下可以执行后续逻辑，1 状态为 PASSIVE_INIT，2 要求 override
                    if self._override and (b_cfg.state == QuantizationStates.PASSIVE):
                        if not check_state(w_cfg.state):
                            raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                                'cause weight has not been correctly quantized.')
                        if not check_state(i_cfg.state):
                            raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                                'cause input has not been correctly quantized.')

                        b_cfg.scale  = w_cfg.scale * i_cfg.scale * self.scale_multiplier
                        b_cfg.state  = QuantizationStates.PASSIVE
                        b_cfg.offset = torch.zeros_like(b_cfg.scale)
                        assert not b_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                            'Passive parameter does not support ASYMMETRICAL quantization')

                    if b_cfg.state == QuantizationStates.PASSIVE_INIT:
                        if not check_state(w_cfg.state):
                            raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                                'cause weight has not been correctly quantized.')
                        if not check_state(i_cfg.state):
                            raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                                'cause input has not been correctly quantized.')
                        
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

                for config in op.config.input_quantization_config[1: ]:

                    # 在两种情况下可以执行后续逻辑，1 状态为 PASSIVE_INIT，2 要求 override
                    if config.state == QuantizationStates.PASSIVE_INIT: 
                        config.scale  = i_cfg.scale
                        config.offset = i_cfg.offset
                        config.state  = QuantizationStates.PASSIVE

                    if self._override and (config.state == QuantizationStates.PASSIVE):
                        config.scale  = i_cfg.scale
                        config.offset = i_cfg.offset
                        config.state  = QuantizationStates.PASSIVE

            if op.type in {'Pad'}:
                # inputs are [input value, pad[shape-related], pad value[optional]]
                if op.num_of_input != 3: continue
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(f'Can not quantize pad value of layer {op.name}, '
                        'cause input has not been correctly quantized.')

                if len(op.config.input_quantization_config) > 1:
                    pad_config = op.config.input_quantization_config[-1]
                    # 在两种情况下可以执行后续逻辑，1 状态为 PASSIVE_INIT，2 要求 override
                    if pad_config.state == QuantizationStates.PASSIVE_INIT: 
                        pad_config = op.config.input_quantization_config[-1]
                        pad_config.scale  = i_cfg.scale
                        pad_config.offset = i_cfg.offset
                        pad_config.state  = QuantizationStates.PASSIVE

                    if self._override and (pad_config.state == QuantizationStates.PASSIVE):
                        pad_config = op.config.input_quantization_config[-1]
                        pad_config.scale  = i_cfg.scale
                        pad_config.offset = i_cfg.offset
                        pad_config.state  = QuantizationStates.PASSIVE

        # final check
        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation): continue
            for cfg, var in op.config_with_variable:
                if cfg.state == QuantizationStates.PASSIVE_INIT:
                    ppq_warning(f'Unexpected quantization state of variable {var.name} at op {op.name}, '
                                'The configuration state has been initialized as PASSIVE_INIT, '
                                'however PassiveParameterQuantizePass do not kown how to deal with it.')


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
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        # build observer and hook for each quantable operation
        hooks, observers, state_records = {}, {}, {}
        for op_name, operation in graph.operations.items():
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
        for op_name, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue

            for cfg, var in operation.config_with_variable:
                if not var.is_parameter:
                    cfg.state = state_records[cfg]

            observer = observers[op_name]
            assert isinstance(observer, OperationObserver)
            observer.render_quantization_config()
            observer.report()
