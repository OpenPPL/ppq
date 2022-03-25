from math import ceil
from typing import Callable, Dict, Iterable, List

from ppq.core import (ChannelwiseTensorQuantizationConfig, QuantizationStates,
                      QuantizationPolicy, QuantizationProperty,
                      TensorQuantizationConfig, empty_ppq_cache)
from ppq.executor import BaseGraphExecutor, RuntimeHook
from ppq.IR import GraphCommandProcesser, QuantableOperation
from ppq.quantization.observer import (CalibrationHook, OperationObserver,
                                       TensorObserverFactroy, TorchMinMaxObserver,
                                       TorchHistObserver, TorchMSEObserver)
import torch
from tqdm import tqdm

from .base import QuantizationOptimizationPass


class RuntimeCalibrationPass(QuantizationOptimizationPass):
    """
        PPQ Runtime Calibration Pass
        For int8 quantization, you need to calibrate or estimate the value range, 
            i.e, (min, max) of all floating-point tensors in the model. 
        
        Unlike constant tensors such as weights and biases, 
            variable tensors such as model input, activations (outputs of intermediate layers) 
            and model output cannot be calibrated unless we run a few inference cycles. 
        
        As a result, the converter requires a representative dataset to calibrate them. 
        This dataset is supposed to be a small subset (about 100~500 samples) of the training or validation data.
        
        ATTENTION: DO NOT GIVE A LARGER DATASET THAN EXPECTED, PPQ WILL RAISE AN ERROR ABOUT IT.
    """
    def __init__(self, method: str = None, override: bool = False) -> None:
        """
        Args:
            method (str, optional): calibration method, if is not None, will override quantizer's setting. 
                Defaults to None.
            
            override (bool, optional): whether to override existing quantization configurations.
        """

        super().__init__(name='PPQ Runtime Calibration Pass')
        self._method = method
        self._observers = {}
        self._collate_fn = None
        self._calib_steps = None
        self._override = override

    def calibrate(self, desc: str, dataloader: Iterable, executor: BaseGraphExecutor, 
        hooks:Dict[str, RuntimeHook], output_names: List[str] = None):

        calib_step = 0
        with tqdm(total=self._calib_steps, desc=desc) as progressing_bar:
            for calib_epoch in range(ceil(self._calib_steps / len(dataloader))):
                for data in dataloader:
                    if self._collate_fn is not None:
                        data = self._collate_fn(data)
                    executor.forward(inputs=data, hooks=hooks,
                        output_names=output_names)
                    progressing_bar.update()
                    calib_step += 1
                    if calib_step >= self._calib_steps: break

    @ empty_ppq_cache
    def optimize(
        self,
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        calib_steps: int,
        collate_fn: Callable,
        **kwargs
    ) -> None:
        self._collate_fn = collate_fn
        self._calib_steps = calib_steps
        assert calib_steps >= 8, 'Insufficient Calibration Detected, to better quantize your network, '\
            'more calibration steps is demonded, we strongly recommend you to prepare more calibration data '\
            'and more calibration steps is perferred here. (at least 8)'

        assert calib_steps <= 512, 'Calibration steps is too large, ppq is capable for quantizing your network within 32-128 '\
            'calibration steps. More calibraiton steps will greatly delay ppq\'s calibration procedure. '\
            'Reset your calib_steps parameter please.'

        # -------------------------------------------------
        # Override existing quantization configurations
        # -------------------------------------------------
        if self._override:
            for operation in processer.graph.operations.values():
                if not isinstance(operation, QuantableOperation): continue
               
                for config, var in operation.config_with_variable:
                    if (not var.is_parameter and 
                        config.state == QuantizationStates.ACTIVATED and 
                        config.dominated_by == config):
                        config.state = QuantizationStates.INITIAL

        # build observer and hook for each quantable operation
        hooks = {}
        for op_name, operation in processer.graph.operations.items():
            
            if not isinstance(operation, QuantableOperation): continue

            # override algorithm setting if necessary
            for config, var in operation.config_with_variable:
                if not var.is_parameter and self._method is not None:
                    config.observer_algorithm = self._method

            observer = OperationObserver(
                opeartion=executor._graph.operations[op_name], 
                monitor_parameter=False)
            self._observers[op_name] = observer
            hooks[op_name]           = observer.hook

        # ready for calibration
        # hook forward function, let observers take effects.
        self.calibrate(desc='Calibration Progress(Phase 1)', dataloader=dataloader, 
            executor=executor, hooks=hooks, output_names=None)

        # render calibration result.
        for _, observer in self._observers.items():
            assert isinstance(observer, OperationObserver)
            observer.render_quantization_config()
            observer.report()

        # -------------------------------------------------
        # There are some two-phase observer in ppq,
        # which means they have to be calibrated for a second time.
        #   see aslo: TorchHistObserver 
        # -------------------------------------------------

        # remove one-phase observer from hook dict.
        pop_list = []
        for op_name, observer in self._observers.items():
            assert isinstance(observer, OperationObserver)
            if all([type(var_observer) not in {TorchHistObserver, TorchMSEObserver} 
                for var_observer in observer._hook._observer_table.values()]):
                    pop_list.append(op_name)
        
        for op_name in pop_list:
            self._observers.pop(op_name)
            hooks.pop(op_name)

        if len(hooks) > 0:
            # ready for calibration(Phase 2)
            # hook forward function, let observers take effects.
            self.calibrate(desc='Calibration Progress(Phase 2)', dataloader=dataloader, 
                executor=executor, hooks=hooks, output_names=None)

            # render calibration result for a second time.
            for _, observer in self._observers.items():
                assert isinstance(observer, OperationObserver)
                observer.render_quantization_config()
                observer.report()


class RuntimePerlayerCalibrationPass(RuntimeCalibrationPass):
    """
        PPQ Runtime Calibration Pass(Per layer calibration)
        For int8 quantization, you need to calibrate or estimate the value range, 
            i.e, (min, max) of all floating-point tensors in the model. 
        
        Unlike constant tensors such as weights and biases, 
            variable tensors such as model input, activations (outputs of intermediate layers) 
            and model output cannot be calibrated unless we run a few inference cycles. 
        
        As a result, the converter requires a representative dataset to calibrate them. 
        This dataset is supposed to be a small subset (around ~100-500 samples) of the training or validation data.
        
        ATTENTION: DO NOT GIVE A LARGER DATASET THAN EXPECTED, PPQ WILL RAISE AN ERROR ABOUT IT.
    """
    def __init__(self, method: str) -> None:
        super().__init__()
        self._method = method
        self.name = 'PPQ Runtime Calibration Pass(Per Layer)'

    def optimize(self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, calib_steps: int, collate_fn: Callable, **kwargs) -> None:
        self._collate_fn = collate_fn
        self._calib_steps = calib_steps
        assert calib_steps >= 8, 'Insufficient Calibration Detected, to better quantize your network, '\
            'more calibration steps is demonded, we strongly recommend you to prepare more calibration data '\
            'and more calibration steps is perferred here. (at least 8)'

        assert calib_steps <= 512, 'Calibration steps is too large, ppq is capable for quantizing your network within 32-128 '\
            'calibration steps. More calibraiton steps will greatly delay ppq\'s calibration procedure. '\
            'Reset your calib_steps parameter please.'

        for operation in tqdm(processer.graph.topological_sort(), 
                              desc='Runtime Calibration(Per Layer)'):
            
            if not isinstance(operation, QuantableOperation): continue
            
            # override algorithm setting if necessary
            for config, var in operation.config_with_variable:
                if not var.is_parameter and self._method is not None:
                    config.observer_algorithm = self._method

            observer = OperationObserver(
                opeartion=operation, 
                monitor_parameter=False)
            
            self.calibrate(desc=f'Runtime Calibration for {operation.name}', 
                dataloader=dataloader, executor=executor, 
                hooks={operation.name: observer.hook}, 
                output_names=[var.name for var in operation.outputs])
            
            if any([type(var_observer) in {TorchHistObserver} 
                for var_observer in observer._hook._observer_table.values()]):
                self.calibrate(desc=f'Runtime Calibration for {operation.name} (Phrase 2)', 
                    dataloader=dataloader, executor=executor, 
                    hooks={operation.name: observer.hook}, 
                    output_names=[var.name for var in operation.outputs])
            
            observer.render_quantization_config()
            observer.report()


class PPLDSPTIReCalibrationPass(RuntimeCalibrationPass):
    """
       PPQ ReCalibration Pass For Computing Ops
       This pass should only be turned on when the platform is one of PPL DSP TI series,
            which needs a per-channel recalibration process for output variable of computing
            op types.

       This pass does not interfere with the normal quantization process, and will be turned
            off for most situations.
    """
    def __init__(self, method: str = None, override: bool = False) -> None:
        super().__init__(method, override)
        self.name = 'PPQ ReCalibration For Computing Op Pass'

    def optimize(self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, calib_steps: int, collate_fn: Callable, **kwargs) -> None:
        self._collate_fn = collate_fn
        self._calib_steps = calib_steps
        assert calib_steps >= 8, 'Insufficient Calibration Detected, to better quantize your network, '\
            'more calibration steps is demonded, we strongly recommend you to prepare more calibration data '\
            'and more calibration steps is perferred here. (at least 8)'

        assert calib_steps <= 512, 'Calibration steps is too large, ppq is capable for quantizing your network within 32-128 '\
            'calibration steps. More calibraiton steps will greatly delay ppq\'s calibration procedure. '\
            'Reset your calib_steps parameter please.'

        hooks = {}
        for operation in tqdm(processer.graph.topological_sort(), 
                              desc='Collecting Observer For Computing Ops'):
            
            if not isinstance(operation, QuantableOperation) or not operation.is_computing_op: continue
            output_cfg = operation.config.output_quantization_config[0]
            master_cfg, master_operation, master_var = output_cfg, operation, operation.outputs[0]
            
            observe_table = {}
            # to check if all input data is greater than 0
            # we only need a basic observer here
            if operation.inputs[0].name in processer.graph.inputs:
                input_cfg = operation.config.input_quantization_config[0]
                sym_input_cfg = TensorQuantizationConfig(
                    policy=QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR
                    ),
                    rounding=input_cfg.rounding,
                    num_of_bits=input_cfg.num_of_bits,
                    quant_min=input_cfg.quant_min,
                    quant_max=input_cfg.quant_max,
                    scale=None,
                    offset=None,
                    observer_algorithm='Minmax',
                    detail={'consumer': input_cfg}
                )
                observe_table.update({input_cfg : TensorObserverFactroy.build_observer(operation.inputs[0], sym_input_cfg)})

            # only consider overlapped by relu/clip activation here 
            downstream_ops = processer.graph.get_downstream_operations(operation)
            if len(downstream_ops) == 1 and downstream_ops[0].type in {'Relu', 'Clip'}\
                and isinstance(downstream_ops[0], QuantableOperation):
                if output_cfg.dominated_by == downstream_ops[0].config.output_quantization_config[0]:
                    # we have to register hooks before observe a different op
                    if len(observe_table) > 0:
                        hooks[operation.name] = CalibrationHook(operation, observe_table)
                        observe_table = {}
                    master_cfg = downstream_ops[0].config.output_quantization_config[0]
                    master_operation = downstream_ops[0]
                    master_var = downstream_ops[0].outputs[0]
            
            master_cfg_per_channel = ChannelwiseTensorQuantizationConfig(
                policy=QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                ),
                rounding=master_cfg.rounding,
                num_of_bits=master_cfg.num_of_bits,
                quant_min=master_cfg.quant_min,
                quant_max=master_cfg.quant_max,
                scale=None,
                offset=None,
                observer_algorithm='Minmax',
                state=QuantizationStates.INITIAL,
                channel_axis=1,
                detail={'consumer' : output_cfg}
            )

            observe_table.update({master_cfg : TensorObserverFactroy.build_observer(master_var, master_cfg_per_channel)})
            assert master_operation.name not in hooks, 'register an operation in calibration hooks twice'
            hooks[master_operation.name] = CalibrationHook(master_operation, observe_table)

        self.calibrate(desc='ReCalibration For Computing Ops', dataloader=dataloader,\
            executor=executor, hooks=hooks)
        
        for hook in hooks.values():
            assert isinstance(hook, CalibrationHook)
            # hook.render_quantization_config()
            for observer in hook._observer_table.values():
                cfg = observer._quant_cfg.detail['consumer']
                assert isinstance(cfg, TensorQuantizationConfig)
                assert isinstance(observer, TorchMinMaxObserver)
                # asym cfg for graph input
                if observer._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    min_vals = torch.min(torch.cat(observer._min_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
                    max_vals = torch.max(torch.cat(observer._max_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
                    cfg.detail.update({'range_min': min_vals, 'range_max': max_vals})

                elif observer._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                    min_val = torch.min(torch.cat(observer._min_val_collector, dim=0)).cpu().item(),
                    max_val = torch.max(torch.cat(observer._max_val_collector, dim=0)).cpu().item(),
                    cfg.detail.update({'range_min': min_val, 'range_max': max_val})