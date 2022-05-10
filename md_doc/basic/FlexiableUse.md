# Advanced Usage
This tutorial illustrates how you could jump out of the high-level API box and actually control your own
quantization workflow by customizing your own scripts. Note that [ProgramEntrance](./ProgramEntrance.md)
hides many details behind several high-level API calls in order to make the whole process easier for new
users, however, in some degree this tutorial shows how you can build your own workflow, thus providing
more flexiablity.

## Prepare Your DataLoader
You still need to prepare your model and calibration data folder, however, you can place them anywhere
you want and all you need is to provide the location of your model and calibration folder

```python
model_path = '/path/to/your/model.onnx'
data_path  = '/path/to/your/dataFolder'
```
or if you are quantizing a caffe model
```python
prototxt_path = '/path/to/your/model.prototxt'
weight_path   = '/path/to/your/model.caffemodel'
data_path     = '/path/to/your/dataFolder'
```
and you can customize your own dataloader, your dataloader must be iterable, like a List. For example,
to create a random 32-length calibration dataloader with batchsize being 16

```python
dataloader = [torch.randn(16, 3, 224, 224) for _ in range(32)]
```
and if you are using cuda and have enough memory, you can obtain acceleration by putting all input data
on gpu in advance
```python
dataloader = [torch.randn(16, 3, 224, 224).to('cuda') for _ in range(32)]
```
and if your model has multiple inputs
```python
dataloader = [{'input_1': torch.randn(16, 3, 224, 224), 'input_2': torch.randn(16, 3, 224, 224)} for \
                _ in range(32)]
```

## Load Your Model
PPQ needs to load your model into PPQ IR graph before anything could go further, and only onnx/caffe
models are supported
```python
from ppq.api import load_onnx_graph, load_caffe_graph

ppq_graph_ir = load_onnx_graph(model_path) # for onnx
ppq_graph_ir = load_caffe_graph(prototxt_path, weight_path) # for caffe
```

## Confirm Target Platform
You have to choose your target platform before quantization, i.e., the backend you want to deploy your
quantized model on. For example, if you want to deploy your model on TensorRT, you just need to specify
```python
from ppq.core import TargetPlatform

target_platform = TargetPlatform.TRT_INT8
```
please check [ppq.core](../../ppq/core/quant.py) for all supported backends, PPQ will issue a quantizer
and an exporter for a specific target platform, and different target platforms might lead to completely
different quantization schemes and exported file formats.


## Prepare Your Setting
As illustrated in [ProgramEntrance](./ProgramEntrance.md), quantization setting acts as a guider which
conducts the quantization process. PPQ has provided default settings for some backend platforms, see
[ppq.api.setting](../../ppq/api/setting.py) for more details
```python
from ppq import QuantizationSettingFactory

setting = QuantizationSettingFactory.pplcuda_setting() # for OpenPPL CUDA
setting = QuantizationSettingFactory.dsp_setting()     # for DSP/SNPE
```
if you want to customize your own setting, you can start from
```python
setting = QuantizationSettingFactory.default_setting()
```
say if you want to apply ssd equalization algorithm instead of default equalization method, all you need is
to turn on the corresponding pass in your setting
```python
setting = QuantizationSettingFactory.default_setting()
setting.ssd_equalization = True
```
or if you want to apply training-based optimization and control more details of some specific pass
```python
setting.lsq_optimization              = True    # turn on pass
setting.lsq_optimization_setting.lr   = 1e-4    # adjust learning rate
setting.lsq_optimization_setting.mode = 'local' # mode of lsq optimization
```
see [ppq.api.setting](../../ppq/api/setting.py) for more information about all supported passes and their
applications

## Schedule Your Graph
Before IR graph can be processed by *Quantizer*, PPQ needs to dispatch operations in the IR graph to different
platforms, for example, shape-related operations will be dispatched to *TargetPlatform.SHAPE_OR_INDEX*, and
non-quantable operations will be dispatched to *TargetPlatform.FP32*
```python
from ppq.api.interface import dispatch_graph

ppq_graph_ir = dispatch_graph(ppq_graph_ir)
```
then we can begin our quantization process using all prepared information

## Initialize An Executor
All operations are exectuted by *Executor* instances in PPQ, and as you can see from
[default.py](../../ppq/executor/torch/default.py), the inner operation executing logic
is implemented using pytorch
```python
from ppq.executor import TorchExecutor

executor = TorchExecutor(ppq_graph_ir, device='cuda') # for cuda
executor = TorchExecutor(ppq_graph_ir, device='cpu')  # for cpu
```

## Quantization
PPQ will designate a quantizer for your target platform which would follow the following
conventions to actually run the quantization
1. Prepare dispatched graph IR for calibration
2. Refine quantization behaviors for certain operations
3. Run calibration process for parameters and activations
4. Render quantization parameters
5. Run activated optimization algorithms specified by quantization setting

```python

from ppq.api.interface import QUANTIZER_COLLECTION

quantizer = QUANTIZER_COLLECTION[target_platform](graph=ppq_graph_ir)
quantizer.quantize(
        inputs=dummy_input,               # some random input tensor, should be list or dict for multiple inputs
        calib_dataloader=dataloader,      # calibration dataloader
        executor=executor,                # executor in charge of everywhere graph execution is needed
        setting=setting,                  # quantization setting
        calib_steps=calib_steps,          # number of batch data needed in calibration, 8~512
        collate_fn=lambda x: x.to('cuda') # final processing of batched data tensor
)

```

## Inference Simulation
After quantization, the quantized ppq IR graph is in quantization mode, so if you directly run the quantized
ppq IR graph using a *TorchExecutor*
```python
for data in dataloader:
    if collate_fn is not None: # process batched data tensor
        data = collate_fn(data)
    outputs = executor.forward(data)
```
you will obtain the final outputs with every quantable operation executed in quantization mode, however, if
you want to disable quantization and obtain fp32 outputs, you just have to disable quantization for every
quantable operation
```python
for op in ppq_ir_graph.operations.values():
    if isinstance(op, QuantableOperation):
        op.dequantize()     # disable quantization

outputs = executor.forward(data) # execution in fp32 mode
```

## Analysis

See [ProgramEntrance](./ProgramEntrance.md) for more information.


## Export
Usually the chozen target platform determines the exact exporting format of the quantized IR graph, but sometimes
you might want to export in a different format
```python
from ppq.api.interface import export_ppq_graph

export_platform = target_platform  # won't necessary be equal
export_ppq_graph(graph=ppq_ir_graph, platform=export_platform, graph_save_to='quantized', config_save_to='quantized.json')
```
