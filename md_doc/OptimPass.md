## PPQ Optim Pass(PPQ 优化过程)

在 PPQ 的量化逻辑中，我们使用一个个的优化过程来修改量化控制信息、模型结构与权重，优化过程是模型量化任务的实际承担者，如 RuntimeCalibrationPass 负责对模型进行 Calibration 从而得到 Scale 与 Offset 信息，QuantFusionPass 负责修改量化信息以模拟目标平台的图融合策略等等。优化过程是一个可扩展的类，在 PPQ 中用户可以使用系统预定义的数十种优化过程完成模型量化任务，也可以自定义新的优化过程以解决特定的问题。

PPQ 的 api 函数 quantize_onnx_model, quantize_caffe_model, quantize_torch_model 等，将根据输入的量化配置 (QuantizationSetting) 决定启动那些优化过程，同时传递相应的参数，用户可以检阅 ppq\api\setting.py 文件获取更多的信息。当启用上述函数进行量化时，用户无法调用自定义的优化过程。如需使用自定义优化过程，用户需要参考 ppq\samples\FP8\fp8_sample.py 样例，手动创建量化管线。

### 1. QuantizationOptimizationPass(优化过程):

该类型定义于 ppq\quantization\optim\base.py

该类型是抽象基类，描述了一个施加在计算图上的具体优化行为。事实上 PPQ 的大部分量化功能均是由一个一个的优化过程所承担的，如 RuntimeCalibrationPass 负责对模型进行 Calibration 从而得到 Scale 与 Offset 信息，QuantFusionPass 负责修改量化信息以模拟目标平台的图融合策略等等。用户可以继承该类并实现自己的优化过程逻辑。

#### 成员函数：

  * **optimize(self, graph: BaseGraph, \*\*kwargs) -> None:**

      optimize 函数是一个抽象函数，任何继承于 QuantizationOptimizationPass 的子类均需要实现该函数从而完成对计算图的修改，PPQ 将在量化过程中依次调用优化过程的 optimize 函数完成模型的量化功能。
      该函数是不定参数函数，用户可以扩展定义该函数的入参，但入参中必须包含 graph 对象。使用 PPQ 系统 api 完成量化过程调用时，将为优化过程传入 graph, dataloader, executor 三个参数(见上文)。
      用户可以通过手动创建优化管线的方式传入更多参数。


### 2. QuantizationOptimizationPipeline(优化过程管线):

该类型定义于 ppq\quantization\optim\base.py

该类型是 QuantizationOptimizationPass 的容器类，即一个包含若干 QuantizationOptimizationPass 的列表，它的行为与普通的 list 无异。

#### 成员函数：

  * **__init__(self, passes: List[QuantizationOptimizationPass]):**

      创建一个量化管线，这需要用户输入一个仅包含 QuantizationOptimizationPass 的列表，请注意该列表是有序的，其中的优化过程将按顺序依次完成调用。

  * **optimize(self, graph: BaseGraph, \*\*kwargs) -> None:**

      将该量化管线中的过程依次施加于计算图之上，该函数为不定参函数，用户可以传入任意多的参数。所有传入的参数将原封不动地传递给所包含的量化优化过程。

  * **append_optimization_to_pipeline(self, optim_pass: QuantizationOptimizationPass, at_front:bool = False):**

      为当前管线添加一个新的优化过程

使用语句 from ppq.quantization.optim import * 来引入所有 PPQ 提供的量化过程，在此我们列举常用优化过程如下：

| Optim Pass |  |
| ---- | ---- |
| BiasCorrectionPass | 该过程会分析网络量化后的激活值误差情况，并尝试修改每一层的 Bias 参数降低量化误差。 |
| HorizontalLayerSplitPass | 该过程会尝试将图中的卷积层分解成两个，从而降低量化误差 |
| LayerwiseEqualizationPass | 该过程执行跨层权重、激活值均衡，将使用恒等变换的方式降低网络的量化误差 |
| LearnedStepSizePass | 该过程会会利用传入的 Calibration 数据集对网络进行微调，从而降低量化误差 |
| RuntimeCalibrationPass | 该过程收集激活值的统计信息，从而为网络中的激活值创建 Scale 与 Offset，并将他们的状态设置为 ACTIVE |
| QuantizeFusionPass | 该过程会根据传入的图融合模式修改量化信息，被该过程关闭的量化信息状态将被设置为 FP32，并指向一个父量化节点。 |
| QuantAlignmentPass | 该过程会处理 Concat, Add, AveragePooling 等算子的输入输出对齐，它们的状态将被设置为 PASSIVE，并指向一个父量化节点。 |
| ParameterBakingPass | 该过程会使得参数静态量化，它们的状态将被设置为 BAKED，它们的值将被量化后的值所替代 |
| QuantizeSimplifyPass | 该过程会根据图的连接关系关闭冗余的量化信息，它们的状态将被设置为 FP32，并指向一个父量化节点。 |

### 3. 定义新的优化过程

用户可以声明子类，继承于 ppq.QuantizationOptimizationPass 基类，并实现相关接口函数，参考代码如下：

        # this file shows how to create new optim pass

        from ppq import BaseGraph
        from ppq import QuantizationOptimizationPass # base class

        class MyOptimPass(QuantizationOptimizationPass): # inherit this
            # 1. realize __init__ function, name your optim pass
            def __init__(self) -> None:
                super().__init__('My Optim Pass')
            
            # 2. realize optimize function, do your work
            def optimize(self, graph: BaseGraph, **kwargs) -> None:
                for op in graph.operations.values():
                    print(f'Hi, nice to meet you {op.name}')

用户需要使用自定义的 optimize 函数对图中信息进行修改，量化过程可以修改图中的任何信息，但当用户试图在量化过程中修改图的结构时，请注意图上 Variable 的 shape, dtype 属性可能需要更新。针对这种情况，推荐用户

### 4. 优化过程参数的传递

用户可以使用两种方式向优化过程传递参数：
    1. 在 __init__ 函数中定义参数，并在创建优化过程时传参
    2. 在 optimize 函数中定义参数，并在调用优化过程时传参

当用户使用 quantize_onnx_model, quantize_caffe_model, quantize_torch_model 函数对模型进行量化时，PPQ 会在 ppq\quantization\quantizer\base.py 文件中完成所有量化优化过程的创建，并赋予它们初始化的参数。而后 Quantizer 将创建量化管线并完成优化过程的调用，此时传入的参数如下：

        # Quantizer 创建量化管线
        quant_pipeline = self.build_quant_pipeline(setting)
        # Quantizer 调用量化管线，传入参数
        quant_pipeline.optimize(
            graph=self._graph,
            dataloader=calib_dataloader,
            executor=executor,
            verbose=self._verbose,
            **kwargs)

上述属性: graph, dataloader, executor 均可被优化过程的 optimize 函数访问，用户可以使用传入的数据展开后续的工作。定义实例如下：

        from typing import Iterable
        from ppq import BaseGraph, TorchExecutor
        from ppq import QuantizationOptimizationPass # base class

        class MyOptimPass(QuantizationOptimizationPass): # inherit this
            # 1. realize __init__ function, name your optim pass
            def __init__(self) -> None:
                super().__init__('My Optim Pass')
            
            # 2. realize optimize function, do your work
            def optimize(self, graph: BaseGraph, dataloader: Iterable, executor: TorchExecutor, **kwargs) -> None:
                for data in dataloader:
                    print('Hi, there is a data batch')

当用户自行创建优化管线时，所有量化过程的参数由用户手动传递，此时用户可以自行设计 optimize 与 __init__ 函数的接口：

        import torch
        from torchvision import models

        import ppq.lib as PFL
        from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
        from ppq.api import ENABLE_CUDA_KERNEL, load_torch_model
        from ppq.core.quant import (QuantizationPolicy, QuantizationProperty,
                                    RoundingPolicy)
        from ppq.quantization.optim import (LearnedStepSizePass, ParameterBakingPass,
                                            ParameterQuantizePass,
                                            RuntimeCalibrationPass)

        pipeline = PFL.Pipeline([
            ParameterQuantizePass(),
            RuntimeCalibrationPass(),
            LearnedStepSizePass(
                steps=1000, is_scale_trainable=False, 
                lr=1e-4, block_size=4, collecting_device='cuda'),
            ParameterBakingPass()
        ])

        with ENABLE_CUDA_KERNEL():
            # 调用管线完成量化，下列参数将传递给每一个优化过程
            pipeline.optimize(
                graph=graph, dataloader=dataset, verbose=True, 
                calib_steps=32, collate_fn=collate_fn, executor=executor)
