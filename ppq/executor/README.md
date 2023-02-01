## PPQ Graph Executor(PPQ 执行引擎)

为了量化并优化神经网络模型，PPQ 实现了基于 Pytorch 的执行引擎，该执行引擎能够执行 Onnx 与 Caffe 的模型文件，目前支持 90 余种常见 Onnx 算子，涵盖 1d, 2d, 3d 视觉、语音、文本模型。
PPQ 的执行引擎位于 ppq.executor 目录下，由两个主要部分组成： ppq.executor.torch.py 文件中包含了执行引擎自身； ppq.executor.op 文件夹中则包含了不同后端的算子库。

在开始阅理解执行引擎之前，我们先介绍算子库的相关内容

### PPQ Backend Functions(PPQ 算子库)

核心算子库位于 ppq.executor.op.torch.default 文件中，该文件中包含了所有算子默认的执行逻辑。

我们知道，对于一个量化算子而言，由于硬件的不同其执行逻辑也可能发生变化。例如 LeakyRelu 算子的负数部分在 GPU 上会采用 x * alpha 的方式完成计算，而在 FPGA 则会采用 x = x >> 3 完成计算。正因为这种差异的存在， PPQ 允许相同的算子在不同平台(TargetPlatform)上拥有不同的执行逻辑。这也意味着针对每一个平台，我们都将实现一个平台独特的算子库文件，这些算子库都继承于 ppq.executor.op.torch.default。

    def Mul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
        """Performs element-wise binary multiplication (with Numpy-style
        broadcasting support).

        This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

        (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
        """
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
        values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
        multiplicand, multiplier = values
        return multiplicand * multiplier

上文中的内容即 ppq.executor.op.torch.default 中 Mul 算子的执行逻辑，在 PPQ 中，所有算子在执行时都将接受一系列 torch.Tensor 作为输入，而后我们调用 pytorch 完成算子的计算逻辑。
你可以打开 PPQ 的算子库文件查看其他算子的执行逻辑，并且 PPQ 也提供了 register_operation_handler 函数，借助该函数你可以注册自定义算子的执行逻辑；或是覆盖现有算子的执行逻辑。

    def register_operation_handler(handler: Callable, operation_type: str, platform: TargetPlatform):
        """Regitser a custimized function as operation handler.

        Function should accept at least 3 input parameters, return one or more tensor as result:
        func(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:

        If there is already another operation handler for given operation_type,
            new handler will replace the old one without warrning.
        """
        if platform not in GLOBAL_DISPATCHING_TABLE:
            raise ValueError('Unknown Platform detected, Please check your platform setting.')
        GLOBAL_DISPATCHING_TABLE[platform][operation_type] = handler
        
该函数位于 ppq.api, 你可以使用语句 from ppq.api import register_operation_handler 来引入它。
  
### PPQ Executor(PPQ 执行引擎)
接下来我们向你介绍 PPQ 执行引擎 TorchExecutor，你可以使用语句 from ppq import TorchExecutor 导入执行引擎。初始化执行引擎则需要传入一个 PPQ 计算图实例对象，在这里我们假设已经获取到了一个量化后的计算图对象 ppq_quant_ir，并使用下面的语句初始化计算引擎

    
    executor = TorchExecutor(graph=ppq_quant_ir)
    executor.forward(inputs=..., output_names=..., hooks=...)

我们使用 executor.forward 接口获取图的执行结果，它将可以传入三个参数:

* inputs: inputs (Union[dict, list, torch.Tensor]): [input tensors or somewhat]
* output_names (List[str], optional): output variable names. default is None.
* hooks (Dict[str, RuntimeHook], optional): A hook table for customizing operation behaviour and collate data during executing.

当执行引擎获取到推理请求时，它将按拓扑顺序依次执行图中的算子：下图展示了一个简单的示例

![图1. 执行图示例](https://user-images.githubusercontent.com/43309460/190056256-8b664993-3af4-4151-8451-f60d42f3145c.png)

在这里，我们的图中包含三个算子 Conv, Relu, Softmax，他们将按照拓扑次序被依次执行。PPQ 的执行引擎会在执行完 Conv 算子后，将 Conv 算子的结果暂存于 Var 1 中，供 Relu 算子取用。而在执行完 Relu 算子后，PPQ 执行引擎则会及时地释放 Var 1 中暂存的数据，因为他们不会被其他算子取用，而且也不是网络的输出 Variable。在每一次推理过后，PPQ 还会清空网络中所有的暂存变量以释放显存。
下面的代码段展示了一个非量化算子的执行逻辑：

    for operation in executing_order:
        outputs = operation_forward_func(operation, inputs, self._executing_context)
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        fp_outputs = outputs

        for output_idx, output_var in enumerate(operation.outputs):
            output_var       = operation.outputs[output_idx]
            output_var.value = outputs[output_idx]
 
    # clear all variable(static clear).
    for var in self._graph.variables.values():
        if not var.is_parameter:
            var.value = None

PPQ 的执行引擎是专为量化计算图的执行而设计的————接下来让我们深入到量化算子的执行过程中去。
对于一个量化算子而言，其每一个输入和输出变量都会有一个 Tensor Quantization Config (TQC) 控制结构体对量化过程进行描述。

对于一个量化 Conv 算子而言，PPQ 将为它创建 2-3 个 Input TQC，以及一个 Output TQC。分别对其输入变量以及输出变量的量化行为进行描述。下面的代码展示了如何为量化算子创建特定的 TQC 描述量化逻辑。

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

        # ------------------------------------------------------------
        # 关闭所有输入量化，状态设置为fp32
        # ------------------------------------------------------------
        for tensor_quant_config in config.input_quantization_config:
            tensor_quant_config.state = QuantizationStates.FP32

        operation.config = config

在 图2 中，我们展示了 PPQ 执行引擎对于量化算子的执行逻辑：

![图2. 算子执行流程](https://user-images.githubusercontent.com/43309460/190065996-02bf7fb4-7421-4417-883d-77967dd1863a.png)

在 PPQ 中，算子的执行被分为四个过程：
* 首先 PPQ 将根据算子上的 TQC 信息量化算子的输入。量化过程并非是原地的，量化后的数据将会是一个新的 torch.Tensor。
* 随后 PPQ 在算子库中寻找算子的执行逻辑，我们已经提到对于每一个平台，他们都可以拥有自己的一套独立的算子库。PPQ 将按照算子的平台找到特定的计算逻辑，并调用他们完成计算得到结果。
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
        """Use This class to realize your quantization logic.

        Inherit class TorchQuantizeDelegate, implement interface __call__, then
        register your delegator with executor.register_quantize_delegate
        """
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
        """RuntimeHook is an abstract class designed for executor customizing."""

        def __init__(self, operation: Operation) -> None:
            self._hook_to = operation

        def pre_forward_hook(self, inputs: list, **kwargs) -> list:
            return inputs

        def post_forward_hook(self, outputs: list, **kwargs) -> list:
            return outputs

