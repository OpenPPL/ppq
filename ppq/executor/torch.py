from typing import Callable, Dict, List, Union

import torch

from ppq.core import (QuantizationStates, TargetPlatform, DataType,
                      TensorQuantizationConfig, empty_ppq_cache, ppq_warning)
from ppq.IR import BaseGraph, Operation, QuantableOperation, RunnableGraph
from ppq.IR.base.command import GraphDeployCommand
from ppq.quantization.qfunction import PPQuantFunction

from .base import (OPERATION_FORWARD_TABLE, BaseGraphExecutor,
                   QuantOPRuntimeHook, RuntimeHook)
from .op import TorchBackendContext


class TorchMetaDataTracingHook(RuntimeHook):
    def __init__(self, operation: Operation) -> None:
        super().__init__(operation)

    def pre_forward_hook(self, inputs: List[torch.Tensor], **kwargs) -> list:
        # some operations got none as its input
        # therefore we have to create meta for those none input value manually.
        for tensor, var in zip(inputs, self._hook_to.inputs):
            if tensor is None:
                ppq_warning(
                    f'Unexpected input value of operation {self._hook_to.name}, '
                    f'recieving "None" at its input {self._hook_to.inputs.index(var)}')
            else:
                var.shape = tensor.shape
                var.dtype = tensor.dtype

        return inputs

    def post_forward_hook(self, outputs: List[torch.Tensor], **kwargs) -> list:
        for tensor, var in zip(outputs, self._hook_to.outputs):
            if tensor is not None:
                var.shape = tensor.shape
                var.dtype = tensor.dtype

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


class TorchExecutor(BaseGraphExecutor):
    def __init__(
        self, graph: BaseGraph, fp16_mode: bool = True,
        device: str = 'cuda') -> None:
        """
        ## PPQ Graph Executor(PPQ 执行引擎)

        为了量化并优化神经网络模型，PPQ 实现了基于 Pytorch 的执行引擎，该执行引擎能够执行 Onnx 与 Caffe 的模型文件，目前支持 90 余种常见 Onnx 算子，涵盖 1d, 2d, 3d 视觉、语音、文本模型。
        PPQ 的执行引擎位于 ppq.executor 目录下，由两个主要部分组成： ppq.executor.torch.py 文件中包含了执行引擎自身； ppq.executor.op 文件夹中则包含了不同后端的算子库。

        在开始阅理解执行引擎之前，我们先介绍算子库的相关内容

        ### PPQ Backend Functions(PPQ 算子库)

        核心算子库位于 ppq.executor.op.torch.default 文件中，该文件中包含了所有算子默认的执行逻辑。

        我们知道，对于一个量化算子而言，由于硬件的不同其执行逻辑也可能发生变化。例如 LeakyRelu 算子的负数部分在 GPU 上会采用 x * alpha 的方式完成计算，
        而在 FPGA 则会采用 x = x >> 3 完成计算。正因为这种差异的存在， PPQ 允许相同的算子在不同平台(TargetPlatform)上拥有不同的执行逻辑。
        这也意味着针对每一个平台，我们都将实现一个平台独特的算子库文件，这些算子库都继承于 ppq.executor.op.torch.default。

            def Mul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
                ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
                values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
                multiplicand, multiplier = values
                return multiplicand * multiplier

        上文中的内容即 ppq.executor.op.torch.default 中 Mul 算子的执行逻辑，在 PPQ 中，所有算子在执行时都将接受一系列 torch.Tensor 作为输入，而后我们调用 pytorch 完成算子的计算逻辑。
        你可以打开 PPQ 的算子库文件查看其他算子的执行逻辑，并且 PPQ 也提供了 register_operation_handler 函数，借助该函数你可以注册自定义算子的执行逻辑；或是覆盖现有算子的执行逻辑。

            def register_operation_handler(handler: Callable, operation_type: str, platform: TargetPlatform):
                if platform not in GLOBAL_DISPATCHING_TABLE:
                    raise ValueError('Unknown Platform detected, Please check your platform setting.')
                GLOBAL_DISPATCHING_TABLE[platform][operation_type] = handler

        该函数位于 ppq.api, 你可以使用语句 from ppq.api import register_operation_handler 来引入它。

        ### PPQ Executor(PPQ 执行引擎)
        接下来我们向你介绍 PPQ 执行引擎 TorchExecutor，你可以使用语句 from ppq import TorchExecutor 导入执行引擎。初始化执行引擎则需要传入一个 PPQ 计算图实例对象，
        在这里我们假设已经获取到了一个量化后的计算图对象 ppq_quant_ir，并使用下面的语句初始化计算引擎


            executor = TorchExecutor(graph=ppq_quant_ir)
            executor.forward(inputs=..., output_names=..., hooks=...)

        我们使用 executor.forward 接口获取图的执行结果，它将可以传入三个参数:

        * inputs: inputs (Union[dict, list, torch.Tensor]): [input tensors or somewhat]
        * output_names (List[str], optional): output variable names. default is None.
        * hooks (Dict[str, RuntimeHook], optional): A hook table for customizing operation behaviour and collate data during executing.

        当执行引擎获取到推理请求时，它将按拓扑顺序依次执行图中的算子：下图展示了一个简单的示例

        ![图1. 执行图示例](https://user-images.githubusercontent.com/43309460/190056256-8b664993-3af4-4151-8451-f60d42f3145c.png)

        在这里，我们的图中包含三个算子 Conv, Relu, Softmax，他们将按照拓扑次序被依次执行。PPQ 的执行引擎会在执行完 Conv 算子后，将 Conv 算子的结果暂存于 Var 1 中，供 Relu 算子取用。
        而在执行完 Relu 算子后，PPQ 执行引擎则会及时地释放 Var 1 中暂存的数据，因为他们不会被其他算子取用，而且也不是网络的输出 Variable。在每一次推理过后，PPQ 还会清空网络中所有的暂存变量以释放显存。
        下面的代码段展示了一个非量化算子的执行逻辑：

            for operation in executing_order:
                outputs = operation_forward_func(operation, inputs, self._executing_context)
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                fp_outputs = outputs

                for output_idx, output_var in enumerate(operation.outputs):
                    output_var       = operation.outputs[output_idx]
                    output_var.value = outputs[output_idx]

            for var in self._graph.variables.values():
                if not var.is_parameter:
                    var.value = None

        PPQ 的执行引擎是专为量化计算图的执行而设计的————接下来让我们深入到量化算子的执行过程中去。
        对于一个量化算子而言，其每一个输入和输出变量都会有一个 Tensor Quantization Config (TQC) 控制结构体对量化过程进行描述。

        对于一个量化 Conv 算子而言，PPQ 将为它创建 2-3 个 Input TQC，以及一个 Output TQC。分别对其输入变量以及输出变量的量化行为进行描述。
        下面的代码展示了如何为量化算子创建特定的 TQC 描述量化逻辑。

            if operation.type == 'Conv':
                config = self.create_default_quant_config(
                    op                 = operation,
                    num_of_bits        = 8,
                    quant_max          = 127,
                    quant_min          = -128,
                    observer_algorithm = 'percentile',
                    policy             = QuantizationPolicy(
                        QuantizationProperty.PER_TENSOR +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.SYMMETRICAL),
                    rounding           = RoundingPolicy.ROUND_HALF_EVEN)

                for tensor_quant_config in config.input_quantization_config:
                    tensor_quant_config.state = QuantizationStates.FP32

                operation.config = config

        在 图2 中，我们展示了 PPQ 执行引擎对于量化算子的执行逻辑：

        ![图2. 算子执行流程](https://user-images.githubusercontent.com/43309460/190065996-02bf7fb4-7421-4417-883d-77967dd1863a.png)

        在 PPQ 中，算子的执行被分为四个过程：
        * 首先 PPQ 将根据算子上的 TQC 信息量化算子的输入。量化过程并非是原地的，量化后的数据将会是一个新的 torch.Tensor。
        * 随后 PPQ 在算子库中寻找算子的执行逻辑，我们已经提到对于每一个平台，他们都可以拥有自己的一套独立的算子库。
        PPQ 将按照算子的平台找到特定的计算逻辑，并调用他们完成计算得到结果。
        * PPQ 将根据算子上的 TQC 信息量化算子的输出。同样地，输出的量化也不是原地的。
        * 最后我们将量化好的结果写入到计算图的 Variable 上，从而供后续的算子取用。

        对于一个非量化算子而言，上述步骤中的 1，3 是可以省略的。

        下 图3 展示了一个量化卷积算子 与 TQC 之间的关系：

        ![图3. 量化 Conv 的执行过程](https://user-images.githubusercontent.com/43309460/190067258-4a4daec7-f898-4734-85a3-19f449c6a963.png)

        ### Quantize Delegate (量化代理函数)

        PPQ 允许你为网络中特定的 TQC 注册量化代理函数。这样你就可以注册自定义的量化处理逻辑，而非使用 PPQLinearQuantFunction 完成量化。

            def register_quantize_delegate(
                self, config: TensorQuantizationConfig,
                delegator: TorchQuantizeDelegator):

        使用 executor.register_quantize_delegate(config, function) 完成函数注册，被注册的函数必须满足 TorchQuantizeDelegator 所定义的接口。
        下面我们给出一个简单的量化代理函数例子：

            class MyQuantDelegator(TorchQuantizeDelegator):
                def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
                    if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                        raise ValueError('Sorry, this delegator handles only Symmetrical Quantizations.')
                    print('You are invoking cusitmized quant function now.')
                    return torch.round(tensor / config.scale) * config.scale

        在执行器遇到 TQC 时，将会调用 executor.quantize_function 执行量化，其逻辑为：

            def quantize_function(self, tensor: torch.Tensor, config: TensorQuantizationConfig = None) -> torch.Tensor:
                if config is None or not QuantizationStates.is_activated(config.state): return tensor
                elif config in self._delegates: return self._delegates[config](tensor, config)
                else:
                    if config.policy.has_property(QuantizationProperty.DYNAMIC):
                        return self._dynamic_quant_fn(tensor, config)
                    else:
                        return self._default_quant_fn(tensor, config)

        ### Usage (用法示例)

        PPQ 的执行器初始化需要一个计算图实例作为参数：

            executor = TorchExecutor(graph=ppq_quant_ir)
            executor.forward(inputs=..., output_names=..., hooks=...)

        这一计算图可以是量化过后的，也可以是没有量化的。但 PPQ 希望传入的计算图经过正确调度，传入没有调度的计算图将会触发警报：

            if not graph.extension_attrib.get(IS_DISPATCHED_GRAPH, False):
                ppq_warning('Can not create executor with your graph, graph is not correctly dispatched, '
                            'use dispatch_graph(graph=ir, platform=platfrom, setting=setting) first.')

        executor.forward 需要三个参数，下面举例对其进行说明：

            # 传入三个变量 a, b, c 作为输入
            executor.forward(inputs=[a, b, c], output_names=..., hooks=...)

            # 分别对图中 input, var 1 两个变量传入 a, b 作为输入
            executor.forward(inputs={'input': a, 'var 1': b}, output_names=..., hooks=...)

            # 传入一个完整的 tensor 作为输入
            executor.forward(inputs=torch.zeros(shape=[1,3,224,224]), output_names=..., hooks=...)

            # 要求网络输出 output, Var 1 的值
            executor.forward(inputs=..., output_names=['output 1', 'Var 1'], hooks=...)

        executor.forward 函数默认不需要梯度，如果希望执行带有梯度的网络，需要使用 executor.forward_with_gradient 函数。 forward 函数的返回值永远是一个 torch.Tensor 数组，其中元素的顺序由 output_names 参数决定。


        ### Hook (执行钩子函数)
        在调用 executor.forward 函数时可以传入 hooks 参数。钩子函数是注册在 op 上的，你可以传入一个字典用来说明需要调用的钩子函数：
        字典 {'Conv 1': myhook} 说明了希望在算子 Conv 1 的执行器件调用钩子函数 myhook。

        钩子函数必须继承于 RuntimeHook 类，必须实现成员函数 pre_forward_hook, post_forward_hook。在这两个函数中，你可以制定特定的逻辑修改算子输入输出的值。

            class RuntimeHook(metaclass=ABCMeta):
                def __init__(self, operation: Operation) -> None:
                    self._hook_to = operation

                def pre_forward_hook(self, inputs: list, **kwargs) -> list:
                    return inputs

                def post_forward_hook(self, outputs: list, **kwargs) -> list:
                    return outputs

            TorchExecutor - executor object which use torch as its backend.
                torch backend is used to graph simulating & training(QAT)

                all operation forward functions are written with pytorch,
                so that they will have gradient recorded by torch engine.

                which means you can directly access to tensor.grad after using output.backward()
        Args:
            graph (BaseGraph):
                executing graph object,
                TorchExecutor will automatically send all graph parameters towards executing device.

            fp16_mode (bool, optional): [whether the simulator is running in fp16 mode(unimplemented).]. Defaults to True.

            device (str, optional): [
                executing device, as same as torch.device,
                you can not select gpu to executing yet,
                graph will always be send to the very first visible cuda device.
            ]. Defaults to 'cuda'.
        """
        self._default_quant_fn = PPQuantFunction
        self._deployed = False
        self._device = device
        self._executing_context = TorchBackendContext(executing_device=self._device)
        super().__init__(graph)

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
        self._device = device
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
                platform_dispatching_table = OPERATION_FORWARD_TABLE[operation.platform]
                if operation.type not in platform_dispatching_table:
                    raise NotImplementedError(
                        f'Graph op: {operation.name}({operation.type}) '
                        f'has no backend implementation on target platform {operation.platform}. '
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
                raise RuntimeError(f'Op Execution Error: {str(operation)}') from _

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

    def load_graph(self, graph: BaseGraph) -> dict:
        super().load_graph(graph)
        self._deployed = False
        self._runnable_graph = RunnableGraph(self._graph)
        self._runnable_graph(GraphDeployCommand(device=self._device))

    def quantize_function(self, tensor: torch.Tensor, config: TensorQuantizationConfig = None) -> torch.Tensor:
        if config is None or not QuantizationStates.is_activated(config.state): return tensor
        elif config in self._delegates: return self._delegates[config](tensor, config)
        else: return self._default_quant_fn(tensor, config)

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

            assert input_var.shape is not None, (
                f'Can not generate dummy input for input variable {input_var.name}, input shape is not specified.')

            feed_dict[var_name] = torch.Tensor(size=input_var.shape, device='cpu').fill_(
                0).type(dtype=DataType.to_torch(input_var.dtype)).to(self._device)
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
