## PPQ.lib(PPQ 接口函数库)
你正浏览 PPQ 的接口函数库定义，于此文件中定义的内容使你可以在任何 Python 程序中嵌入 PPQ 的执行逻辑。

### 1. PPQ.lib.common.Quantizer

截止 PPQ 0.6.6，软件内共提供 20 种不同的量化器，分别对应不同的部署平台。在 PPQ 中 [量化器](https://github.com/openppl-public/ppq/tree/master/ppq/quantization/quantizer) 负责为算子初始化量化信息，对于不同的硬件平台，其量化策略是不同的，因此量化器也有不一样的实现逻辑。
量化器必须实现 init_quant_config 函数，用于根据算子类别为算子初始化量化配置信息。量化器还必须告知所有可量化的算子类型，用于协助调度器确定调度方案。

```python
{
    TargetPlatform.PPL_DSP_INT8: PPL_DSP_Quantizer,
    TargetPlatform.PPL_DSP_TI_INT8: PPL_DSP_TI_Quantizer,
    TargetPlatform.SNPE_INT8:    PPL_DSP_Quantizer,
    TargetPlatform.QNN_DSP_INT8: PPL_DSP_Quantizer,
    TargetPlatform.TRT_INT8:     TensorRTQuantizer,
    TargetPlatform.MNN_INT8:     MNNQuantizer,
    TargetPlatform.ASC_INT8:     AscendQuantizer,
    TargetPlatform.NCNN_INT8:    NCNNQuantizer,
    TargetPlatform.NXP_INT8:     NXP_Quantizer,
    TargetPlatform.RKNN_INT8:    RKNN_PerTensorQuantizer,
    TargetPlatform.METAX_INT8_C: MetaxChannelwiseQuantizer,
    TargetPlatform.METAX_INT8_T: MetaxTensorwiseQuantizer,
    TargetPlatform.PPL_CUDA_INT8: PPLCUDAQuantizer,
    TargetPlatform.EXTENSION:     ExtQuantizer,
    TargetPlatform.FPGA_INT8   :  FPGAQuantizer,
    TargetPlatform.OPENVINO_INT8: OpenvinoQuantizer,
    TargetPlatform.TENGINE_INT8:  TengineQuantizer,
    TargetPlatform.GRAPHCORE_FP8: GraphCoreQuantizer,
    TargetPlatform.TRT_FP8:       TensorRTQuantizer_FP8,
    TargetPlatform.ONNXRUNTIME:   OnnxruntimeQuantizer
}
```

#### 注册新的量化器

用户可以使用接口函数 ppq.lib.register_network_quantizer 注册自定义的量化器，须知被注册的量化器必须继承 ppq.quantization.quantizer.BaseQuantizer 基类，并实现相应接口：

```python
    # 示例代码
    from ppq.IR import Operation
    from ppq.core import OperationQuantizationConfig
    from ppq.quantization.quantizer import BaseQuantizer
    from ppq.core import TargetPlatform

    class MyQuantizer(BaseQuantizer):
        def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
            # Implement this function first!
            return super().init_quantize_config(operation)
        
        def quant_operation_types(self) -> set:
            return {'Conv', 'Gemm'}

    # register this quantizer to PPL_CUDA_INT8
    register_network_quantizer(quantizer=MyQuantizer, platform=TargetPlatform.PPL_CUDA_INT8)
```

#### 获取一个量化器示例

用户可以通过接口函数 ppq.lib.quant.Quantizer 获取一个与平台对应的量化器，亦可 from ppq.quantization.quantizer import ... 导入所需的量化器。

### 2. PPQ.lib.common.Exporter

在 PPQ 中，我们以 [量化控制信息](https://github.com/openppl-public/ppq/tree/master/ppq/core) 描述网络的量化情况，网络中与量化相关的参数均保存于算子的量化控制信息中。
因此网络的导出也即将算子上绑定的量化控制信息导出到文件，你应当根据推理框架的需要导出相应的模型格式。截止 PPQ 0.6.6，目前共支持 19 种 [导出格式](https://github.com/openppl-public/ppq/tree/master/ppq/parser) ：

```python
{
    TargetPlatform.PPL_DSP_INT8:  PPLDSPCaffeExporter,
    TargetPlatform.PPL_DSP_TI_INT8: PPLDSPTICaffeExporter,
    TargetPlatform.QNN_DSP_INT8:  QNNDSPExporter,
    TargetPlatform.PPL_CUDA_INT8: PPLBackendExporter,
    TargetPlatform.SNPE_INT8:     SNPECaffeExporter,
    TargetPlatform.NXP_INT8:      NxpExporter,
    TargetPlatform.ONNX:          OnnxExporter,
    TargetPlatform.ONNXRUNTIME:   ONNXRUNTIMExporter,
    TargetPlatform.OPENVINO_INT8: OpenvinoExporter,
    TargetPlatform.CAFFE:         CaffeExporter,
    TargetPlatform.NATIVE:        NativeExporter,
    TargetPlatform.EXTENSION:     ExtensionExporter,
    TargetPlatform.RKNN_INT8:     OnnxExporter,
    TargetPlatform.METAX_INT8_C:  ONNXRUNTIMExporter,
    TargetPlatform.METAX_INT8_T:  ONNXRUNTIMExporter,
    TargetPlatform.TRT_INT8:      TensorRTExporter_JSON,
    TargetPlatform.ASC_INT8:      AscendExporter,
    TargetPlatform.TRT_FP8:       ONNXRUNTIMExporter,
    TargetPlatform.NCNN_INT8:     NCNNExporter,
    TargetPlatform.TENGINE_INT8:  TengineExporter,
    TargetPlatform.MNN_INT8:      MNNExporter,
}
```

用户可以使用接口函数 ppq.lib.register_network_quantizer 注册自定义的导出逻辑，须知被注册的量化器必须继承 ppq.parser.GraphExporter 基类，并实现相应接口：

```python
    # 示例代码
    from ppq.IR import BaseGraph
    from ppq.core import TargetPlatform

    class MyExporter(GraphExporter):
        def export(self, file_path: str, graph: BaseGraph, config_path: str = None, **kwargs):
            return super().export(file_path, graph, config_path, **kwargs)

    # register this Exporter to PPL_CUDA_INT8
    register_network_exporter(exporter=MyExporter, platform=TargetPlatform.PPL_CUDA_INT8)
```

#### 获取一个量化器示例

用户可以通过接口函数 ppq.lib.quant.Exporter 获取一个与平台对应的导出器，亦可 from ppq.parser import ... 导入所需的导出器。

### 3. PPQ.lib.quant 函数库

1. Quantizer(platform: TargetPlatform, graph: BaseGraph) -> BaseQuantizer: 

    Get a pre-defined Quantizer corresponding to your platform.
    Quantizer in PPQ initializes Tensor Quantization Config for each Operation,
    it describes how operations are going to be quantized.

    根据目标平台获取一个系统预定义的量化器。在 PPQ 中，量化器是一个用于为算子初始化量化信息 Tensor Quantization Config 的对象。
    量化器决定了你的算子是如何被量化的，你也可以设计新的量化器来适配不同的后端推理框架。
    在 PPQ 中我们为不同的推理后端设计好了一些预定义的量化器，你可以通过 ppq.lib.Quantizer 来访问它们。

2. Pipeline(optims: List[QuantizationOptimizationPass]) -> QuantizationOptimizationPipeline:

    Build a Pipeline with given Optimization Passes Collection

    使用给定的量化过程集合创建量化管线。

3. Observer(quant_config: TensorQuantizationConfig, variable: Variable = None) -> BaseTensorObserver:

    Get a Calibration Observer based on quant_config.observer_algorithm attribute.

    根据 TQC 中 observer_algorithm 属性获取对应的 Observer.

4. LinearQuantizationConfig(
    symmetrical: bool = True,
    dynamic: bool = False,
    power_of_2: bool = False,
    channel_axis: int = None,
    quant_min: int = -128,
    quant_max: int = 127,
    num_of_bits = 8,
    calibration: str = 'minmax',
    rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN) -> TensorQuantizationConfig:

    Create a Linear Quantization Config.
    
    创建线性量化配置信息。

5. FloatingQuantizationConfig(
    symmetrical: bool = True,
    power_of_2: bool = True,
    channel_axis: int = None,
    quant_min: float = -448.0,
    quant_max: float = 448.0,
    exponent: int = 4,
    mantissa: int = 3,
    calibration: str = 'constant',
    rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN) -> TensorQuantizationConfig:

    Create a Floating Quantization Config.

    创建浮点量化配置信息。

6. Dispatcher(graph: BaseGraph, method: str='conservative') -> GraphDispatcher:

    Get a Graph Dispatcher.
    
    获取一个指定的调度器。

7. OperationForwardFunction(optype: str, platform: TargetPlatform = TargetPlatform.FP32) -> Callable:
    
    Get an Operation forward function. Same Op are allows to have different forward function on different platform,
    to get a default forward function, use platform=TargetPlatform.FP32.

    获取一个算子前向传播执行函数。在 PPQ 中，相同的算子可以在不同的平台上注册成不同的执行逻辑，
    使用 platform = TargetPlatform.FP32 来获取默认执行逻辑。

8. Exporter(platform: TargetPlatform) -> GraphExporter:

    Get an network Exporter.

    获取一个网络导出器。

9. Parser(framework: NetworkFramework) -> GraphExporter:

    Get an network Parser.

    获取一个网络解析器。

### 3. PPQ.lib.extension 函数库

1. register_network_quantizer(quantizer: type, platform: TargetPlatform):
    
    Register a quantizer to ppq quantizer collection.
    
    This function will override the default quantizer collection:
        register_network_quantizer(MyQuantizer, TargetPlatform.TRT_INT8) will replace the default TRT_INT8 quantizer.

    Quantizer should be a subclass of BaseQuantizer, do not provide an instance here as ppq will initilize it later.
    Your quantizer must require no initializing params.

    Args:

    *. quantizer (type): quantizer to be inserted.
    
    *. platform (TargetPlatform): corresponding platfrom of your quantizer.

2. register_network_parser(parser: type, framework: NetworkFramework):
    
    Register a parser to ppq parser collection. 

    This function will override the default parser collection:
        register_network_parser(MyParser, NetworkFramework.ONNX) will replace the default ONNX parser.

    Parser should be a subclass of GraphBuilder, do not provide an instance here as ppq will initilize it later.
    Your quantizer must require no initializing params.

    Args:

    *. parser (type): parser to be inserted.

    *. framework (NetworkFramework): corresponding NetworkFramework of your parser.

3. register_network_exporter(exporter: type, platform: TargetPlatform):
    
    Register an exporter to ppq exporter collection.

    This function will override the default exporter collection:
        register_network_quantizer(MyExporter, TargetPlatform.TRT_INT8) will replace the default TRT_INT8 exporter.

    Exporter should be a subclass of GraphExporter, do not provide an instance here as ppq will initilize it later.
    Your Exporter must require no initializing params.

    Args:
        
    *. exporter (type): exporter to be inserted.
    
    *. platform (TargetPlatform): corresponding platfrom of your exporter.

4. register_calibration_observer(algorithm: str, observer: type):
    
    Register an calibration observer to OBSERVER_TABLE.

    This function will override the existing OBSERVER_TABLE without warning.
    
    registed observer must be a sub class of OperationObserver.

    Args:

    *. exporter (type): exporter to be inserted.

    *. platform (TargetPlatform): corresponding platfrom of your exporter.

5. register_operation_handler(handler: Callable, operation_type: str, platform: TargetPlatform):
    
    Regitser a custimized function as operation handler.
    
    Function should accept at least 3 input parameters, return one or more tensor as result:
    func(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    
    If there is already another operation handler for given operation_type,
        new handler will replace the old one without warrning.

    Args:
        
    *. handler (Callable): Callable function, which interface follows restriction above.
    
    *. operation_type (str): Operation type.
    
    *. platform (TargetPlatform): Register platform.