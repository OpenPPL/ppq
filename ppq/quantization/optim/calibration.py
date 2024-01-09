from math import ceil
from typing import Callable, Dict, Iterable, List

import torch
from ppq.core import (QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, TensorQuantizationConfig,
                      empty_ppq_cache, OBSERVER_ISOTONE_OBSERVER_AXIS)
from ppq.executor import BaseGraphExecutor, RuntimeHook
from ppq.IR import BaseGraph, QuantableOperation, QuantableVariable
from ppq.quantization.observer import (CalibrationHook, OperationObserver,
                                       TensorObserverFactroy, OBSERVER_TABLE)
from tqdm import tqdm

from .base import QuantizationOptimizationPass


class RuntimeCalibrationPass(QuantizationOptimizationPass):
    """
    ## Runtime Calibration Pass(量化参数校准过程)

    For integer quantization, you need to calibrate or estimate the scale of all floating-point tensors in the model.

    Formula:

            Quant(Y, scale_Y) = Clip(Round(Y / scale_Y))

            Dequant(Y, scale_Y) = Y * scale_Y

    Only activations that have quantization state = INITIAL are going to be calibrated via this optimization pass.
    While if the parameter "override" is set to True, activations with quantization state = ACTIVATED will also be re-calibrated.

    Runtime Calibration Pass will write estimated scales and offsets to tensor quantization configs, and set their state to ACTIVATED.

    Unlike constant tensors such as weights and biases, variable tensors such as model input,
    activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles.

    As a result, PPQ Runtime Calibration Pass requires a representative dataset to calibrate them.

    This dataset is supposed to be a small subset (around ~100-500 samples) of the training or validation data.

    ### Parameters:

    * method(str):

            String that representing the algorithm used to estimate scales and offsets for activations.

            Can be mse, kl, percentile, minmax, this parameter is case insensitive.

            You can register your own calibration method through functions in ppq.api

    * override(bool)

            if this parameter is set to True, activations with quantization state = ACTIVATED will also be re-calibrated,
            runtime calibration pass will overwrite their scales and offsets.

            This parameter is introduced since ppq 0.6.4

    ### observer support matrix:

    | observer     | Symmetrical | Asymmetrical | Per-chanel | Per-tensor | Cuda Acceleration   |
    | ---          | ---         | ---          | ---        | ---        |                     |
    | minmax       | [x]         | [x]          | [x]        | [x]        | [ ]                 |
    | mse          | [x]         | [x]          | [ ]        | [x]        | [x]                 |
    | precentile   | [x]         | [x]          | [x]        | [x]        | [x]                 |
    | kl           | [x]         | [ ]          | [ ]        | [x]        | [x]                 |
    | isotone      | [x]         | [x]          | [ ]        | [x]        | [ ]                 |

    ### Usage:

    Runtime Calibration Pass should be invoked before Passive Parameter Quantize Pass

    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.quantize_activation = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn)

    You can manually create this optimization by:

        from ppq import RuntimeCalibrationPass

        optim = RuntimeCalibrationPass()

    ### Register Calibration Method:

    Using api function register_calibration_observer to resister new observer algorithm to PPQ system.
    Once Algorithm is registered, Runtime Calibration Pass will automatically calling them by name.

    This feature requires PPQ > 0.6.5

    """
    def __init__(self, method: str = None, override: bool = False, calib_steps: int = 32) -> None:
        super().__init__(name='PPQ Runtime Calibration Pass')
        self._method = method
        self._observers   = {}
        self._collate_fn  = None
        self._calib_steps = calib_steps
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
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        calib_steps: int = 32,
        collate_fn: Callable = None,
        **kwargs
    ) -> None:
        if collate_fn is not None: self._collate_fn = collate_fn
        if calib_steps is not None: self._calib_steps = calib_steps

        assert calib_steps >= 8, ('Insufficient Calibration Detected, to get a better quantization performance, '
            'more calibration steps is required, we strongly recommend you to prepare more calibration data '
            'and more calibration steps is preferred here. (at least 8)')

        assert calib_steps <= 512, ('Calibration steps is too large, ppq can quantize your network within 8-512 '
            'calibration steps. More calibration steps will greatly delay ppq\'s calibration procedure. '
            'Reset your calib_steps parameter please.')

        # -------------------------------------------------
        # Override existing quantization configurations
        # -------------------------------------------------
        if self._override:
            for operation in graph.operations.values():
                if not isinstance(operation, QuantableOperation): continue

                for config, var in operation.config_with_variable:
                    if (not var.is_parameter and
                        config.state == QuantizationStates.ACTIVATED and
                        config.dominated_by == config):
                        config.state = QuantizationStates.INITIAL

        # build observer and hook for each quantable operation
        hooks = {}
        for op_name, operation in graph.operations.items():

            if not isinstance(operation, QuantableOperation): continue

            # override algorithm setting if necessary
            for config, var in operation.config_with_variable:
                if not var.is_parameter and self._method is not None:
                    config.observer_algorithm = self._method

            observer = OperationObserver(
                operation=executor._graph.operations[op_name],
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
        #   see also: TorchHistObserver
        # -------------------------------------------------

        # remove one-phase observer from hook dict.
        pop_list = []
        for op_name, observer in self._observers.items():
            assert isinstance(observer, OperationObserver)
            if all([type(var_observer) not in {OBSERVER_TABLE['kl'], OBSERVER_TABLE['mse']}
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


class PPLDSPTIReCalibrationPass(RuntimeCalibrationPass):
    """PPQ ReCalibration Pass For Computing Ops This pass should only be turned
    on when the platform is one of PPL DSP TI series, which needs a per-channel
    recalibration process for output variable of computing op types.

    This pass does not interfere with the normal quantization process, and will
    be turned      off for most situations.
    """
    def __init__(self, method: str = None, override: bool = False) -> None:
        super().__init__(method, override)
        self.name = 'PPQ ReCalibration For Computing Op Pass'

    def optimize(self, graph: BaseGraph, dataloader: Iterable,
        executor: BaseGraphExecutor, calib_steps: int, collate_fn: Callable, **kwargs) -> None:
        self._collate_fn = collate_fn
        self._calib_steps = calib_steps
        assert calib_steps >= 8, 'Insufficient Calibration Detected, to better quantize your network, '\
            'more calibration steps is demonded, we strongly recommend you to prepare more calibration data '\
            'and more calibration steps is preferred here. (at least 8)'

        assert calib_steps <= 512, 'Calibration steps is too large, ppq is capable for quantizing your network within 32-128 '\
            'calibration steps. More calibraiton steps will greatly delay ppq\'s calibration procedure. '\
            'Reset your calib_steps parameter please.'

        hooks = {}
        for operation in tqdm(graph.topological_sort(),
                              desc='Collecting Observer For Computing Ops'):

            if not isinstance(operation, QuantableOperation) or not operation.is_computing_op: continue
            output_cfg = operation.config.output_quantization_config[0]
            master_cfg, master_operation, master_var = output_cfg, operation, operation.outputs[0]

            observe_table = {}
            # to check if all input data is greater than 0
            # we only need a basic observer here
            if operation.inputs[0].name in graph.inputs:
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
            downstream_ops = graph.get_downstream_operations(operation)
            if len(downstream_ops) == 1 and downstream_ops[0].type in {'Relu', 'Clip'}\
                and isinstance(downstream_ops[0], QuantableOperation):
                if len(observe_table) > 0:
                    hooks[operation.name] = CalibrationHook(operation, observe_table)
                    observe_table = {}
                master_cfg = downstream_ops[0].config.output_quantization_config[0]
                master_operation = downstream_ops[0]
                master_var = downstream_ops[0].outputs[0]

            master_cfg_per_channel = TensorQuantizationConfig(
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
                assert isinstance(observer, OBSERVER_TABLE['minmax'])

                if observer._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    min_vals = torch.min(torch.cat(observer._min_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
                    max_vals = torch.max(torch.cat(observer._max_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
                    cfg.detail.update({'range_min': min_vals, 'range_max': max_vals})

                elif observer._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                    min_val = torch.min(torch.cat(observer._min_val_collector, dim=0)).cpu().item(),
                    max_val = torch.max(torch.cat(observer._max_val_collector, dim=0)).cpu().item(),
                    cfg.detail.update({'range_min': min_val, 'range_max': max_val})


class IsotoneCalibrationPass(RuntimeCalibrationPass):
    """
    ## Isotone Calibration Pass(保序量化校准过程)

    在神经网络中，一些算子的输出并不需要保证总体的精确性，而只关注于最大最小值所在的位置，
    例如图像分类网络中，网络的输出通常是一个1000维的向量，用于表达图像属于特定类别的概率。
    为了保证分类的正确性，我们并不需要这个1000维的向量在量化后是整体准确的，只需要其中的最大值出现在正确的位置上。
    因此我们希望最大值与次大值之间相差至少半个 scale，并且次大值能够不被截断。

    因此传统的 min-max, percentile, kl 方法在这一情景中并不能得到最高的分类精度，
    保序量化是为了解决这一问题而设计的，在这一校准过程中，程序将网络输出变量的校准方式改写为 Isotone(保序校准)。
    默认设置下，该过程只对 softmax 算子的输出进行保序校准。对于其他情况，用户需要手动指定需要进行保序校准的变量名。

    保序量化需要设定一个分类轴，同样地以分类网络为例，其输出形为 [Batch, 1000]。
    分类操作将在数据的最后一维展开，因此需要设置保序轴为 -1。

    Algorithm:

        For softmax or sigmoid activations, usually we just need
        argmax(softmax(x)) == argmax(softmax(quant(x)))

        Inspired by this Property, Isotone Observer is designed to provide an order-preserving calibration method,
            which cares only about argmax(x) [or argmin(x)]

        To keep argmax(x) == argmax(quant(x)), we only need to
            distinguish the largest element and the second largert element with quantization

            let L1 represents the largest element of x,
            while L2 represents the second largest.

            For Symmetrical Quantization, We want:

                1. round(L1 / scale) - round(L2 / scale) > 0

                2. round(L2 / scale) < quant_max

            Hence that, we will have:

                1. scale < 2 * (L1 - L2)

                2. scale > L2 / (self._quant_cfg.quant_max - .5)

            For Asymmetircal Quantization, We want:

                1. round(L1 / scale) + offset - round(L2 / scale) - offset > 0

                2. round(L2 / scale) + offset < quant_max

            Hence that, we will have:

                1. scale < 2 * (L1 - L2)

                2. scale > L2 / (self._quant_cfg.quant_max - offset - .5)

        The best setting of scale, offset can be solved by PPQ Isotone observer.

        Time Complexity: O(nlogn)
    """
    def __init__(self, variables: List[str] = None, axis: int = -1, verbose: bool = True, calib_steps: int = 32) -> None:
        super().__init__(calib_steps=calib_steps)
        self.name = "Isotone Calibration Pass"
        self.variables = variables
        self.axis      = axis
        self.verbose   = verbose

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        if self.variables is None:
            for op in graph.operations.values():
                if op.type == 'Softmax' and isinstance(op, QuantableOperation):

                    # had not been dominated.
                    if op.output_quant_config[0].dominated_by == op.output_quant_config[0]:
                        op.output_quant_config[0].state = QuantizationStates.INITIAL
                        op.output_quant_config[0].observer_algorithm = 'Isotone'
                        op.output_quant_config[0].detail[OBSERVER_ISOTONE_OBSERVER_AXIS] = op.attributes.get('axis', -1)

                        if self.verbose:
                            print(f'Calibration Method of Op {op.name} '
                                  f'has been changed to Isotone[axis={op.attributes.get("axis", -1)}].')

        else: # self.variables is not None
            if not isinstance(self.variables, list):
                raise TypeError('Isotone Calibration Pass needs a list of variable name as its input.')
            for var in self.variables:
                if not isinstance(var, str):
                    raise TypeError('Isotone Calibration Pass needs a list of variable name as its input.')
                if var not in graph.variables:
                    raise ValueError(f'Variable {var} not in current graph.')

                var = graph.variables[var]
                if isinstance(var, QuantableVariable):
                    var.source_op_config.state = QuantizationStates.INITIAL
                    var.source_op_config.observer_algorithm = 'Isotone'
                    var.source_op_config.detail[OBSERVER_ISOTONE_OBSERVER_AXIS] = self.axis
                    if self.verbose: print(
                        f'Calibration Method of Variable {var.name} has been changed to Isotone[axis={self.axis}].')

        super().optimize(graph=graph, **kwargs)
