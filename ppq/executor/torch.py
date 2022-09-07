from typing import Any, Callable, Dict, List, Union

import numpy
from ppq.core import (IS_DISPATCHED_GRAPH, DataType, OperationMeta,
                      QuantizationStates, TargetPlatform, TensorMeta,
                      TensorQuantizationConfig, empty_ppq_cache, ppq_warning)
from ppq.IR import BaseGraph, Operation, QuantableOperation, RunnableGraph
from ppq.IR.base.command import GraphDeployCommand
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction

import torch

from .base import (GLOBAL_DISPATCHING_TABLE, BaseGraphExecutor,
                   QuantOPRuntimeHook, RuntimeHook)
from .op import TorchBackendContext


def build_meta(value: Any) -> TensorMeta:
    if isinstance(value, torch.Tensor):
        return TensorMeta.parsing_from_torch_tensor(value)
    if isinstance(value, numpy.ndarray):
        return TensorMeta.parsing_from_numpy_ndarray(value)
    raise TypeError(f'Can not tracing meta for given value(type: {type(value)}), check your graph again.')


class TorchMetaDataTracingHook(RuntimeHook):
    def __init__(self, operation: Operation) -> None:
        self.input_metas, self.output_metas = [], []
        super().__init__(operation, operation_meta=None)
    def pre_forward_hook(self, inputs: list, **kwargs) -> list:
        # some operations got none as its input
        # therefore we have to create meta for those none input value manually.
        for tensor, var in zip(inputs, self._hook_to.inputs):
            if tensor is None:
                ppq_warning(
                    f'Unexpected input value of operation {self._hook_to.name}, '
                    f'recieving "None" at its input {self._hook_to.inputs.index(var)}')
                self.input_metas.append(TensorMeta(dtype=DataType.NONETYPE, shape=None))
            else:
                self.input_metas.append(build_meta(tensor))
        return inputs
    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        self.output_metas = [build_meta(tensor) for tensor in outputs]
        return outputs


class TorchQuantizeDelegator(Callable):
    """Since PPQ 0.6.2, Interface TorchQuantizeDelegate is introduced to
    customize quantization logic: To be specific, you are suppose to inherit
    this class, and define your own computation logic within function __call__.

        Pass your Delegate to TorchExecutor by TorchExecutor.register_quantize_delegate(c, d)
            Where c is the target quantization config, d is your delegator class.
            Once you invoke this function, PPQ execution system will hand the quantization
            computation of config c over to your delegate. PPQ execution system will no
            longer quantize variable related with config c anymore.

    Notice that a delegate replaces quantization computation only, it still under the control of PPQ quantization
    System, so to say if your config has an invalid state like DEQUANTIZED, PPQ execution system will never been
    required to quantize related tensor and so your delegate class will take no effects on config c.

        Remove delegate function by TorchExecutor.remove_quantize_delegate(c)

    If you have some customized parameter of your delegator logic, set them as class attributes.
    Like: self.param1 = ..., self.param2 = ...

    Do not edit config structure directly.

    Args:
        Callable (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, tensor: torch.Tensor,
                 config: TensorQuantizationConfig) -> torch.Tensor:
        raise NotImplementedError('Implement this function first.')


class TorchExecutor(BaseGraphExecutor, torch.nn.Module):
    def __init__(
        self, graph: BaseGraph, fp16_mode: bool = True,
        device: str = 'cuda') -> None:
        """
            TorchExecutor - executor object which use torch as its backend.
                torch backend is used to graph simulating & training(QAT)

                all operation forward functions are written with pytorch,
                so that they will have gradient recorded by torch engine.

                which means you can directly access to tensor.grad after using output.backward()
        Args:
            graph (BaseGraph):
                executing graph object,
                TorchExecutor will automatically send all graph parameters towards executing device.

            rounding_policy (RoundingPolicy)
                rounding_policy takes responsibility for quantizing input/output/parameters during graph executing.
                Notice that all quantizer will possess a platform specified rounding_policy (within BaseQuantizer._quant_function),
                you are supposed to pass BaseQuantizer._quant_function._rounding_policy
                    to initialize this executor whenever you have a quantizer.

                Different rounding_policy will brings totally different quant behaviour and rounding.
                It will greatly change the output of your network in some cases.

            fp16_mode (bool, optional): [whether the simulator is running in fp16 mode(unimplemented).]. Defaults to True.

            device (str, optional): [
                executing device, as same as torch.device,
                you can not select gpu to executing yet,
                graph will always be send to the very first visible cuda device.
            ]. Defaults to 'cuda'.
        """
        self._default_quant_fn = PPQLinearQuantFunction
        self._deployed = False
        self._device = device
        self._executing_context = TorchBackendContext(executing_device=self._device)
        super().__init__(graph)
        
        if not graph.extension_attrib.get(IS_DISPATCHED_GRAPH, False):
            ppq_warning('Can not create executor with your graph, graph is not correctly dispatched, '
                        'use dispatch_graph(graph=ir, platform=platfrom, setting=setting) first.')
        
        self._runnable_graph = RunnableGraph(self._graph)
        self._delegates = {}

        # fp16 is not available for now.
        self.fp16_mode = fp16_mode
        self.deploy()

    def register_quantize_delegate(
        self, config: TensorQuantizationConfig,
        delegator: TorchQuantizeDelegator):
        """Since PPQ 0.6.2, Interface TorchQuantizeDelegate is introduced to
        customize quantization logic: To be specific, you are suppose to
        inherit this class, and define your own computation logic within
        function __call__.

            Pass your Delegate to TorchExecutor by TorchExecutor.register_quantize_delegate(c, d)
                Where c is the target quantization config, d is your delegator class.
                Once you invoke this function, PPQ execution system will hand the quantization
                computation of config c over to your delegate. PPQ execution system will no
                longer quantize variable related with config c anymore.

        Notice that a delegate replaces quantization computation only, it still under the control of PPQ quantization
        System, so to say if your config has an invalid state like DEQUANTIZED, PPQ execution system will never been
        required to quantize related tensor and so your delegate class will take no effects on config c.

        Remove delegate function by TorchExecutor.remove_quantize_delegate(c)
        """
        if not isinstance(delegator, TorchQuantizeDelegator):
            raise TypeError(
                f'You can only register a TorchQuantizeDelegate as quantization delegator function,'
                f' however a/an {type(delegator)} was given')
        if not isinstance(config, TensorQuantizationConfig):
            raise TypeError(
                f'Except a TensorQuantizationConfig instance, however {type(config)} was passed.')
        self._delegates[config] = delegator

    def remove_quantize_delegate(
        self, config: TensorQuantizationConfig):
        """Since PPQ 0.6.2, Interface TorchQuantizeDelegate is introduced to
        customize quantization logic: To be specific, you are suppose to
        inherit this class, and define your own computation logic within
        function __call__.

            Pass your Delegate to TorchExecutor by TorchExecutor.register_quantize_delegate(c, d)
                Where c is the target quantization config, d is your delegator class.
                Once you invoke this function, PPQ execution system will hand the quantization
                computation of config c over to your delegate. PPQ execution system will no
                longer quantize variable related with config c anymore.

        Notice that a delegate replaces quantization computation only, it still under the control of PPQ quantization
        System, so to say if your config has an invalid state like DEQUANTIZED, PPQ execution system will never been
        required to quantize related tensor and so your delegate class will take no effects on config c.

        Remove delegate function by TorchExecutor.remove_quantize_delegate(c)
        """
        if not isinstance(config, TensorQuantizationConfig):
            raise TypeError(
                f'Except a TensorQuantizationConfig instance, however {type(config)} was passed.')
        if config in self._delegates:
            self._delegates.pop(config)

    def deploy(self):
        """Deploy graph parameters towards target device.

        Raises:
            ValueError: [when target device is unacceptable]
        """
        self._deployed = True
        self._runnable_graph(GraphDeployCommand(device=self._device))

    def to(self, device: str):
        # just keep TorchExecutor behaving like torch.nn.Module
        self._device = torch.device(device)
        self.deploy()
        return self

    @ torch.no_grad()
    def forward(
        self,
        inputs: Union[dict, list, torch.Tensor],
        output_names:List[str] = None,
        hooks: Dict[str, RuntimeHook] = None
    ) -> List[torch.Tensor]:
        """Forward function of this executor.

        Notice this forward function will never store and compute gradients.

        Args:
            inputs (Union[dict, list, torch.Tensor]): [input tensor or somewhat]

            output_names (List[str], optional):
                onnx output node names, which used to confirm a output order.

                Defaults to None.

            hooks (Dict[str, RuntimeHook], optional):
                A hook table for customizing operation behaviour and collate data during executing.
                All hooks should inherit from class RuntimeHook, with all necessary methods implemented.
                    See also: ppq.executor.base.RuntimeHook

                Executor calls hook.pre_forward_hook(operation, input_data) before dispatching operation,
                by using this feature, you can dynamically dispatch operation during executing,
                or processing input data as you want.(remember to return processed input data)

                Executor calls hook.post_forward_hook(operation, output_data) after the execution,
                you are supposed to  gather all necessary data from execution via this feature.

                For Quantable Operation, a much more powerful class:
                    ppq.executor.base.QuantOpRuntimeHook is provided.
                see also: ppq.executor.base.QuantOpRuntimeHook

                Defaults to None.

        Returns:
            List[torch.Tensor]: [executing result, list of tensor objects.]
        """
        return self.__forward(
            inputs=inputs,
            output_names=output_names,
            executing_order=self._executing_order,
            hooks=hooks
        )

    def forward_with_gradient(
        self,
        inputs: Union[dict, list, torch.Tensor],
        output_names:List[str] = None,
        hooks: Dict[str, RuntimeHook] = None,
    ) -> List[torch.Tensor]:
        """forward function of this executor.

            Notice this one will store and compute gradient.

        Args:
            inputs (Union[dict, list, torch.Tensor]): [input tensor or somewhat]
            output_names (List[str], optional):
                onnx output node names, which used to confirm a output order.

                Defaults to None.

            hooks (Dict[str, RuntimeHook], optional):
                A hook table for customizing operation behaviour and collate data during executing.
                All hooks should inherit from class RuntimeHook, with all necessary methods implemented.
                    See also: ppq.executor.base.RuntimeHook

                Executor calls hook.pre_forward_hook(operation, input_data) before dispatching operation,
                by using this feature, you can dynamically dispatch operation during executing,
                or processing input data as you want.(remember to return processed input data)

                Executor calls hook.post_forward_hook(operation, output_data) after the execution,
                you are supposed to  gather all necessary data from execution via this feature.

                For Quantable Operation, a much more powerful class:
                    ppq.executor.base.QuantOpRuntimeHook is provided.
                see also: ppq.executor.base.QuantOpRuntimeHook

                Defaults to None.

        Returns:
            List[torch.Tensor]: [executing result, list of tensor objects.]
        """
        return self.__forward(
            inputs=inputs,
            output_names=output_names,
            executing_order=self._executing_order,
            hooks=hooks
        )

    def __forward(
        self,
        inputs: Union[dict, list, torch.Tensor],
        executing_order: List[Operation],
        output_names:List[str] = None,
        hooks: Dict[str, RuntimeHook] = None,
    ) -> List[torch.Tensor]:
        # processing with different input format
        if isinstance(inputs, dict):
            # directly feed value into variables
            for name, value in inputs.items():
                if name in self._graph.variables:
                    var = self._graph.variables[name]
                    var.value = value
                else:
                    print(f'Can not find variable {name} in your graph, please check.')
        else:
            inputs = self.prepare_input(inputs=inputs)
            for key, value in inputs.items():
                assert isinstance(value, torch.Tensor), \
                    f'TorchExecutor can only accept tensor as its input, while {type(value)} was given'
                # input is acceptable, feed input value
                self._graph_input_dictionary[key].value = value

        # processing with output
        last_idx = 0 # record last variable
        if output_names is None:
            output_names = [name for name in self._graph.outputs]
        for name in output_names:
            if name not in self._graph.variables:
                raise KeyError(f'You are requiring output value of variable {name}(is not a variable name), '
                    'however it is not a valid variable of current graph.')
            source_op = self._graph.variables[name].source_op
            if source_op is not None:
                last_idx = max(last_idx, executing_order.index(source_op) + 1)

        visited_op, result_collector = [], [None for _ in output_names]
        # output name can be the same as input name, collect them directly.
        for name in output_names:
            if name in inputs:
                result_collector[output_names.index(name)] = inputs[name]

        for operation in executing_order[: last_idx]:
            try:
                assert isinstance(operation, Operation), 'Oops, seems you got something weird in your graph'
                assert isinstance(operation.platform, TargetPlatform), (
                    f'Operation {operation.name} has an invalid platform setting, '
                    f'only PPQ.core.TargetPlatform is expected here, while {type(operation.platform)} was given')
                platform_dispatching_table = GLOBAL_DISPATCHING_TABLE[operation.platform]
                if operation.type not in platform_dispatching_table:
                    raise NotImplementedError(
                        f'Graph op: {operation.name}({operation.type}) '
                        f'has no backend implementation on target platform {operation.platform}.'
                        'Register this op to ppq.executor.base.py and ppq.executor.op first')
                operation_forward_func = platform_dispatching_table[operation.type]
                operation_runtime_hook = hooks[operation.name] if (hooks is not None) and (operation.name in hooks) else None
                inputs = [var.value for var in operation.inputs]

                # if operation is an QuantableOperation, we have to quant its inputs and outputs at first.
                if isinstance(operation, QuantableOperation):
                    input_configs = [_ for _ in operation.config.input_quantization_config]
                    inputs = [self.quantize_function(input, config) for input, config in zip(inputs, input_configs)]

                # PATCH 20220208
                for idx, var in enumerate(operation.inputs):
                    if var.name in output_names:
                        result_collector[output_names.index(var.name)] = inputs[idx]

                # invoking pre-forward hook
                if operation_runtime_hook is not None:
                    if isinstance(operation_runtime_hook, QuantOPRuntimeHook):
                        inputs = operation_runtime_hook.pre_forward_hook(
                            inputs=[var.value for var in operation.inputs],
                            quant_inputs=inputs, quant_configs=input_configs)
                    elif isinstance(operation_runtime_hook, RuntimeHook):
                        inputs = operation_runtime_hook.pre_forward_hook(inputs=inputs)
                    else: raise TypeError(f'invalid hook instance was given with operation: {operation}')

                # forward and collecting result
                outputs = operation_forward_func(operation, inputs, self._executing_context)
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                fp_outputs = outputs

                # quantize all result if is necessary
                if isinstance(operation, QuantableOperation):
                    output_configs = [_ for _ in operation.config.output_quantization_config]
                    outputs = [self.quantize_function(output, config) for output, config in zip(outputs, output_configs)]

                # invoking post-forward hook
                if operation_runtime_hook is not None:
                    if isinstance(operation_runtime_hook, QuantOPRuntimeHook):
                        outputs = operation_runtime_hook.post_forward_hook(
                            outputs=fp_outputs, quant_outputs=outputs,
                            quant_configs=output_configs)
                    elif isinstance(operation_runtime_hook, RuntimeHook):
                        outputs = operation_runtime_hook.post_forward_hook(outputs=outputs)
                    else: raise TypeError(f'invalid hook instance was given with operation: {operation}')

                # feed value to graph variables.
                for output_idx, output_var in enumerate(operation.outputs):
                    output_var       = operation.outputs[output_idx]
                    output_var.value = outputs[output_idx]

                    if output_var.name in output_names:
                        result_collector[output_names.index(output_var.name)] = outputs[output_idx]
            except Exception as _:
                raise RuntimeError(f'Error happens when dealing with operation {str(operation)}')

            # remove useless value(runtime clear).
            visited_op.append(operation)
            for var in operation.inputs:
                if var.is_parameter: continue
                if all(op in visited_op for op in var.dest_ops):
                    var.value = None

        # clear all variable(static clear).
        for var in self._graph.variables.values():
            if not var.is_parameter:
                var.value = None
        # end for
        return result_collector

    @ torch.no_grad()
    @ empty_ppq_cache
    def tracing_operation_meta(
        self,
        inputs: Union[dict, list, torch.Tensor],
        output_names: List[str] = None,
    ) -> None:
        """Tracing meta data for each operation, if there are some already
        created meta data with your operation, They will be override without
        warrning.

        Args:
            inputs (Union[dict, list, torch.Tensor]): [description]
            output_names (List[str], optional): [description]. Defaults to None.
        """
        hooks = {}
        for op_name, operation in self._graph.operations.items():
            hooks[op_name] = TorchMetaDataTracingHook(operation=operation)

        self.__forward(
            inputs=inputs,
            output_names=output_names,
            executing_order=self._executing_order,
            hooks=hooks)

        for op_name, operation in self._graph.operations.items():
            operation.meta_data = OperationMeta(
                input_metas     = hooks[op_name].input_metas,
                output_metas    = hooks[op_name].output_metas,
                operation_name  = operation.name,
                operation_type  = operation.type,
                executing_order = self._executing_order.index(operation)
            )

    def load_graph(self, graph: BaseGraph) -> dict:
        super().load_graph(graph)
        self._deployed = False
        self._runnable_graph = RunnableGraph(self._graph)
        self._runnable_graph(GraphDeployCommand(device=self._device))

    def quantize_function(self, input: torch.Tensor, config: TensorQuantizationConfig = None) -> torch.Tensor:
        if config is None: return self._default_quant_fn(input, config)
        elif not QuantizationStates.is_activated(config.state): return input
        elif config in self._delegates: return self._delegates[config](input, config)
        else: return self._default_quant_fn(input, config)

    def dummy_forward(self, hooks: Dict[str, RuntimeHook] = None) -> None:
        """This function allows you to execute entire graph without feeding any
        data. This feature is required for operation parameter quantization.
        See also: ppq.quantization.optim.ParameterQuantizePass.

        This function fakes some input tensors via operation metadata.
            ATTENTION: operation must have metadata before invoking this function.

        Args:
            hooks (Dict[str, RuntimeHook], optional):
                A hook table for customizing operation behaviour and collate data during executing.
                All hooks should inherit from class RuntimeHook, with all necessary methods implemented.
                    See also: ppq.executor.base.RuntimeHook

                Executor calls hook.pre_forward_hook(operation, input_data) before dispatching operation,
                by using this feature, you can dynamically dispatch operation during executing,
                or processing input data as you want.(remember to return processed input data)

                Executor calls hook.post_forward_hook(operation, output_data) after the execution,
                you are supposed to  gather all necessary data from execution via this feature.

                For Quantable Operation, a much more powerful class:
                    ppq.executor.base.QuantOpRuntimeHook is provided.
                see also: ppq.executor.base.QuantOpRuntimeHook

                Defaults to None.
        """
        # build dummy input based on meta data
        feed_dict = {}
        for var_name, input_var in self._graph.inputs.items():
            if len(input_var.dest_ops) == 0: continue
            dest_op  = input_var.dest_ops[0]
            dest_idx = dest_op.inputs.index(input_var)

            assert isinstance(dest_op, Operation) and dest_op.meta_data is not None, \
                'Operation meta has not been traced. Please invoke TorchExecutor.tracing_meta_data() first'
            tensor_meta = dest_op.meta_data.input_metas[dest_idx]
            feed_dict[var_name] = tensor_meta.create_tensor(device=self._device)
        self.forward(inputs=feed_dict, hooks=hooks)

    def partial_graph_forward(
        self, operations: List[Operation],
        feed_dict: Dict[str, torch.Tensor],
        output_names:List[str]) -> List[torch.Tensor]:
        """This forward function allows you to execute a series operations in
        your graph. (only operations list in your params will be executed with
        this function) Which serves as a great feature for quantization aware
        training.

        Args:
            operations (List[Operation]):
                operations that you want to execute,
                notice that executing will strictly follow your operation order.

            feed_dict (Dict[str, torch.Tensor]):
                an dictionary contains {variable name: value}, as an input to this execution.

            output_names (List[str]):
                output variable names.

        Returns:
            List[torch.Tensor]: [description]
        """

        return self.__forward(
            inputs=feed_dict,
            output_names=output_names,
            executing_order=operations
        )
