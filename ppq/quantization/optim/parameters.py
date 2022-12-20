from typing import Iterable

import torch
from ppq.core import (QuantizationProperty, QuantizationStates,
                      QuantizationVisibility, ppq_warning)
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph, QuantableOperation
from ppq.quantization.observer import OperationObserver

from .base import QuantizationOptimizationPass


class PassiveParameterQuantizePass(QuantizationOptimizationPass):
    """
    ## PPQ Passive Parameter Quantization Pass(通用被动量化过程)
    
    Passive Parameters are those parameters that must share a same scale and offset with
    other tensors. This pass process 4 types of passive parameter by default, namely:
    
        Bias value in Gemm, it must has a scale = input scale * weight scale.

        Bias value in Conv, it must has a scale = input scale * weight scale.

        Clip min & Clip max in Clip, must has a scale = input scale

        Pading Value, must has a scale = input scale

    ### Parameters:

    * process_clip(Set[str]):
            
            Whether to process clip min, max
            
            If not processed, clip min, max will has their state = QuantizationState.FP32

    * process_bias(bool)

            Whether to process bias
            
            If not processed, bias will has their state = QuantizationState.ACTIVED

    * process_pad(bool)

            Whether to process clip min, max
            
            If not processed, pad value will has their state = QuantizationState.SOI

    * clip_visiblity(bool)
    
            Whether to export quant info of clip min, max

    * pad_visiblity(bool)
    
            Whether to export quant info of pad value
    
    ### Usage
    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.parameter_setting.quantize_passive_parameter = True = True
        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)
    """
    def __init__(self, process_clip: bool = True, process_bias: bool = True, process_pad: bool = True, 
                 clip_visiblity: QuantizationVisibility = QuantizationVisibility.INTERNAL, 
                 pad_visiblity: QuantizationVisibility = QuantizationVisibility.INTERNAL):
        self.process_clip   = process_clip
        self.process_bias   = process_bias
        self.process_pad    = process_pad
        self.clip_visiblity = clip_visiblity
        self.pad_visiblity  = pad_visiblity
        super().__init__(name='PPQ Passive Parameter Quantization')

    def optimize(self, graph: BaseGraph, **kwargs) -> None:

        def check_state(state: QuantizationStates):
            return state in {
                QuantizationStates.PASSIVE,
                QuantizationStates.ACTIVATED,
                QuantizationStates.BAKED,
                QuantizationStates.OVERLAPPED
            }

        for op in graph.operations.values():
            if not isinstance(op, QuantableOperation): continue

            if op.type in {'Conv', 'ConvTranspose', 'Gemm'} and self.process_bias:
                # inputs are [input value, weight, bias(optional)]
                if op.num_of_input == 3:
                    i_cfg, w_cfg, b_cfg = op.config.input_quantization_config
                    if b_cfg.state not in {QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT}: continue

                    # PATCH 2022.07.29 有的时候 bias 是个多维的东西，此时要求前面的维度都是1
                    bias = op.inputs[-1].value
                    if bias is None: raise ValueError(f'Bias Varaible {op.inputs[-1].name} must be a constant. '
                                                      'Please check it again.')

                    assert bias.numel() == bias.shape[-1], (
                        f'For op {op.name}, expect Bias shape to be {[bias.numel()]}, '
                        f'however {bias.shape} was given')
                    op.inputs[-1].value = bias.squeeze()
                    # PATCH 2022.08.02 只有一个数的 bias 经过 squeeze 会变成零维的, 再给它多加一维补回来
                    if op.inputs[-1].value.ndim == 0 and op.inputs[-1].value.numel() == 1:
                        op.inputs[-1].value = op.inputs[-1].value.unsqueeze(0)

                    if not check_state(i_cfg.state):
                        raise PermissionError(f'Can not quantize bias of layer {op.name}, '
                            'cause input has not been correctly quantized.')

                    b_cfg.scale  = w_cfg.scale * i_cfg.scale
                    b_cfg.state  = QuantizationStates.PASSIVE
                    b_cfg.offset = torch.zeros_like(b_cfg.scale)
                    assert not b_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                        'Passive parameter does not support ASYMMETRICAL quantization')

            if op.type in {'Clip'} and self.process_clip:
                # inputs are [input value, min[optional], max[optional]]
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(f'Can not quantize clip value of layer {op.name}, '
                        'cause input has not been correctly quantized.')

                for config in op.config.input_quantization_config[1: ]:
                    config.master_by = i_cfg
                    config.visibility = self.clip_visiblity

            if op.type in {'Pad'} and self.process_pad:
                # inputs are [input value, pad[shape-related], pad value[optional]]
                if op.num_of_input != 3: continue
                i_cfg = op.config.input_quantization_config[0]

                if not check_state(i_cfg.state):
                    raise PermissionError(f'Can not quantize pad value of layer {op.name}, '
                        'cause input has not been correctly quantized.')

                if len(op.config.input_quantization_config) > 1:
                    pad_config = op.config.input_quantization_config[-1]
                    pad_config.master_by = i_cfg
                    pad_config.visibility = self.pad_visiblity

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
        executor: TorchExecutor,
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
                    config.state = QuantizationStates.FP32
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
