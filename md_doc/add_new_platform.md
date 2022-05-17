# Customize Your Quantization Platform

PPQ has supported many backends, you can check [quant.py](../ppq/core/quant.py) for details about all supported
platforms, this tutorial illustrates you how to add your own quantization backend support in PPQ. For simplicity,
the *TargetPlatform.ACADEMIC_INT8* platform will be given as an example here.


## Create Your Platform

First thing first, you need to name and create your platform in [TargetPlatform](../ppq/core/quant.py), there are
all PPQ suppported platforms. Don't forget to add your created platform in the *is_quantized_platform* collection
```python
@ classmethod
def is_quantized_platform(cls, platform) -> bool:
    return platform in {
        cls.PPL_DSP_INT8, cls.PPL_DSP_TI_INT8, cls.QNN_DSP_INT8, cls.TRT_INT8, cls.NCNN_INT8, cls.NXP_INT8,
        cls.SNPE_INT8, cls.PPL_CUDA_INT8, cls.PPL_CUDA_INT4, cls.EXTENSION, cls.PPL_CUDA_MIX, cls.ORT_OOS_INT8,
        cls.ACADEMIC_INT4, cls.ACADEMIC_INT8, cls.ACADEMIC_MIX, cls.METAX_INT8_C, cls.METAX_INT8_T
    }
```

## Inherit BaseQuantizer 

As you can see in [quantizer](../ppq/quantization/quantizer), there are many quantizers, each corresponding with
a backend platform, and they all inherit from the basic class *BaseQuantizer*, the basic quantizer class regulates
the basic quantization workflow and the process of applying quantization passes, which are guided by quantization
setting designated by user in advance. So your quantizer should inherit the basic class as well. Take *ACADEMICQuantizer*
as an example
```python

class ACADEMICQuantizer(BaseQuantizer):
    """ACADEMICQuantizer applies a loose quantization scheme where only input
    variables of computing ops are quantized(symmetrical per-tensor for weight
    and asymmetrical per-tensor for activation).

    This setting doesn't align with any kind of backend for now and it's
    designed only for purpose of paper reproducing and algorithm verification.
    """
    def __init__(self, graph: BaseGraph, verbose: bool = True) -> None:
        self._num_of_bits = 8
        self._quant_min = 0
        self._quant_max = 255
        super().__init__(graph, verbose)
```
the *TargetPlatform.ACADEMIC_INT8* platform takes 8-bit asymmetric quantization scheme for activation,  thus 
*_num_of_bits = 8* and *_quant_min = 0*, *_quant_max = 255*, note that if your platform takes symmetric scheme,
then you should modify like 
```python
self._num_of_bits = 8
self._quant_min   = -128 # or -127, it depends the clip boundary of your backend
self._quant_max   = 127 
```

## Confirm Quantization Scheme

You need to specify your target platform and default platform of your quantizer, these platforms will be dispatched
to different operations by PPQ graph dispatcher when PPQ loads and schedules your model. The target platform should
be your created platform, and the default platform, in almost all circumstances you may leave it as *TargetPlatform.FP32*
```python
@ property
def target_platform(self) -> TargetPlatform:

    return TargetPlatform.ACADEMIC_INT8

@ property
def default_platform(self) -> TargetPlatform:

    return TargetPlatform.FP32

```

You also need to specify quantable operation types of your backend, for example, in most academic settings, only computing
operations(Conv, Gemm, ConvTranspose) need quantization, then you only need to specify those quantable operation types in
*quant_operation_types*
```python
@ property
def quant_operation_types(self) -> set:
    return {
            'Conv', 'Gemm', 'ConvTranspose'
        }
```

To implement the whole quantizer, you should confirm the quantization scheme(per tensor/ per channel, symmetric / asymmetric)
of your backend platform, and in many platforms weight parameters and activations may take different quantization schemes.
Please see [QuantizationProperty](../ppq/core/quant.py) for all supported quantization schemes. Your quantizer class should 
implement the abstract funtion *quantize_policy* to identify the quantization scheme of activation 
```python
@ property
def quantize_policy(self) -> QuantizationPolicy:
    return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )
```
as you can see from above, the *ACADEMICQuantizer* takes a asymmetric linear per-tensor scheme for activation quantization, the
most-common quantization setting in academic papers. 

Similarly, you need to confirm the rounding policy of your backend platform, for example, the *TargetPlatfom.PPL_CUDA_INT8* 
platform takes a round-to-nearest-even policy, while the *TargetPlatform.NCNN_INT8* platform takes a round-half-away-from-zero
policy, in order to better align with your backend, you should make sure coherent rounding behavior between your quantizer
and your real backend.
```python
@ property
def rounding_policy(self) -> RoundingPolicy:

    return RoundingPolicy.ROUND_HALF_EVEN

```
## Correct Quantization Details

In most circumstances, weight parameters may take different quantization scheme from activation, thus we need to correct quantization scheme for weight parameter and special operations in  *init_quantize_config*, note that this func generates quantization configs for
every quantable operation in your graph, you need firstly generate quantization config for common activation
```python
def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

    # create a basic quantization configuration.
    config = self.create_default_quant_config(
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
    )
```
the *create_default_quant_config* creates a default asymmetric per-tensor config for every variable(includes weight parameters), so the next step you need to correct it to symmetric config for academic platform because in academic setting, weight parameters takes a per-tensor symmetric quantization scheme rather than the default per-tensor asymmetric scheme
```python
    # actually usually only support quantization of inputs of computing
    # ops in academic settings
    if operation.type in {'Conv', 'Gemm', 'ConvTranspose'}:

        W_config = config.input_quantization_config[1]
        output_config = config.output_quantization_config[0]

        W_config.quant_max = int(pow(2, self._num_of_bits - 1) - 1)
        W_config.quant_min = - int(pow(2, self._num_of_bits - 1))
        W_config.policy = QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )
        output_config.state = QuantizationStates.FP32


        if operation.num_of_input == 3:
            bias_config = config.input_quantization_config[-1]
            # bias should be quantized with 32 bits
            # in python3, int indicates long long in C++
            # so that it has enough precision to represent a number like 2^32
            # however, it may cause a scale underflow
            # here we give bias a 30 bits precision, which is pettery enough in all cases
            bias_config.num_of_bits = 30
            bias_config.quant_max = int(pow(2, 30 - 1) - 1)
            bias_config.quant_min = - int(pow(2, 30 - 1))
            bias_config.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.PER_TENSOR)
            bias_config.state = QuantizationStates.PASSIVE_INIT

        for tensor_config in config.input_quantization_config[1: ]:
            tensor_config.observer_algorithm = 'minmax'
```
as you can see, all you need to do is to find the special operations(which have variables taking a different quantization scheme),
and locate the corresponding config(*W_config* in this example), and modify the *quant_max*, *quant_min* and *policy* attribute 
to your backend needs(symmetric per-tensor for weight parameters in this example). Also note that in most academic settings, only
input variables need quantization, so we pick the output config *output_config* out, set its state to *QuantizationStates.FP32*,
which means PPQ will skip the quantization of output activation of every quantable operations and execute in fp32 mode.

## Register Your Quantizer And Platform

Now that you have created your platform and corresponding quantizer, then you need to register your platform so that PPQ knows
how to execute operations dispatched to your platform. Most platforms in PPQ uses the same operation executing table implemented
by PyTorch, since it's very difficult to write a platform-specific executing table for tens of different supported platforms in
PPQ, and the real backend executing behavior is almost impossible to replicate in a simulator like PPQ. The default executing
table should do all the good and all you need is to append your platform in the [GLOBAL_DISPATCHING_TABLE](../ppq/executor/base.py) 
```python

GLOBAL_DISPATCHING_TABLE[TargetPlatform.ACADEMIC_INT8] = ACADEMIC_BACKEND_TABLE # could also be DEFAULT_BACKEND_TABLE
```
then when PPQ executor encounters operations dispatched to ypur platform, it will search in the registered table, find the operation
implementation and execute.

To use PPQ to run your platform like any other in-position platforms, the last thing you need to do is to register your platform
and corresponding quantizer in the API table [QUANTIZER_COLLECTION](../ppq/api/interface.py] so that PPQ will treat your platform
like any one else
```python
QUANTIZER_COLLECTION[TargetPlatform.ACADEMIC_INT8] = ACADEMICQuantizer
```

then you can call your quantizer to run the quantization simulation as specified in [how_to_use](./how_to_use.md)
```python

from ppq.api.interface import QUANTIZER_COLLECTION
from ppq.executor import TorchExecutor
from ppq.api import load_onnx_graph, load_caffe_graph
from ppq.api.interface import dispatch_graph

target_platform = TargetPlatform.ACADEMIC_INT8 # your created platform
EXECUTING_DEVICE = 'cuda' # run on gpu

ppq_graph_ir = load_onnx_graph(model_path) # for onnx
ppq_graph_ir = dispatch_graph(ppq_graph_ir, target_platform, setting) # schedule your graph

executor = TorchExecutor(ppq_graph_ir, device=EXECUTING_DEVICE) # initialize executor

quantizer = QUANTIZER_COLLECTION[target_platform](graph=ppq_graph_ir) # your quantizer
quantizer.quantize(
        inputs=dummy_input,                         # some random input tensor, should be list or dict for multiple inputs
        calib_dataloader=dataloader,                # calibration dataloader
        executor=executor,                          # executor in charge of everywhere graph execution is needed
        setting=setting,                            # quantization setting
        calib_steps=calib_steps,                    # number of batched data needed in calibration, 8~512
        collate_fn=lambda x: x.to(EXECUTING_DEVICE) # final processing of batched data tensor
)
```