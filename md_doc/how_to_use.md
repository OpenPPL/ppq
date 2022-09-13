# Usage
This tutorial illustrates how you could jump out of the high-level API box and actually control your own
quantization workflow by customizing your own scripts. It's assumed that you have PPQ installed successfully
in your working environment, for the installation part, please refer to [Installation](../README.md).
In some degree this tutorial shows how you can build your own workflow, thus providing more flexiablity.

## CUDA Kernel for Acceleration
PPQ has implemented some cuda kernels for the process of quantization-dequantization execution of tensors,
these kernels can accelerate the process of graph execution in quantization mode, and any algorithm based
on heavy graph execution may gain speedup when you turn on cuda kernel option ahead of all the following
steps
```python
from ppq.core.config import PPQ_CONFIG

PPQ_CONFIG.USING_CUDA_KERNEL = True
```
note that you don't have to turn on the above option if your environment fails to compile the shared libraries,
it's just for accelration and PPQ will do fine without it turning on.

## Prepare Your DataLoader
First thing first, you need to prepare your model and calibration data folder, note that only onnx and caffe
models are supported in PPQ for now, and you may need to preprocess your calibration data in advance and store
them in your calibration data folder in npy or binary files. If your model is in onnx format

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
you can customize your own dataloader, your dataloader could be anything iterable, like a List. It's really
easy for you to construct a dataloader if your calibration data is stored as npy files

```python
import os
import numpy as np
import torch

dataloader = [torch.from_numpy(np.load(os.path.join(data_path, file_name))) for file_name in os.listdir(data_path)]
```
or if your calibration data is stored as bin files, you can load them as numpy array at first

```python
INPUT_SHAPE = [1, 3, 224, 224]
npy_array = [np.fromfile(os.path.join(data_path, file_name), dtype=np.float32).reshape(*INPUT_SHAPE) for file_name in os.listdir(data_path)]
dataloader = [torch.from_numpy(np.load(npy_tensor)) for npy_tensor in npy_array]
```
you can even create some random dataloader just for testing purpose, to create a random 32-length calibration
dataloader with batchsize being 16

```python
dataloader = [torch.randn(16, 3, 224, 224) for _ in range(32)]
```
and if you are using cuda and have enough memory, you can obtain acceleration by putting all input data
on gpu in advance
```python
dataloader = [torch.randn(16, 3, 224, 224).to('cuda') for _ in range(32)]
```
and if your model has multiple inputs, then you could use a dict to specify every input of your graph
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
please check [ppq.core](../ppq/core/quant.py) for all supported backends, PPQ will issue a quantizer
and an exporter for a specific target platform, and different target platforms might lead to completely
different quantization schemes and exported file formats.


## Prepare Your Setting
Quantization setting acts as a guider which conducts the quantization process. PPQ has provided default 
settings for some backend platforms, see [ppq.api.setting](../ppq/api/setting.py) for more details
```python
from ppq import QuantizationSettingFactory

setting = QuantizationSettingFactory.pplcuda_setting() # for OpenPPL CUDA
setting = QuantizationSettingFactory.dsp_setting()     # for DSP/SNPE
```

if you want to customize your own setting, you can start from a default setting

```python
setting = QuantizationSettingFactory.default_setting()
```

say if you want to apply ssd equalization algorithm instead of default equalization method, all you need is
to turn on the corresponding pass in your setting

```python
setting = QuantizationSettingFactory.default_setting()
setting.ssd_equalization = True
```
or if you want to apply training-based lsq optimization and control more details of the specific pass
```python
setting.lsq_optimization                = True    # turn on pass
setting.lsq_optimization_setting.lr     = 1e-4    # adjust learning rate
setting.lsq_optimization_setting.epochs = 30      # adjust number of training epochs for every block
```
see [ppq.api.setting](../ppq/api/setting.py) for more information about all supported passes and their
applications

## Schedule Your Graph
Before IR graph can be processed by a *Quantizer*, PPQ needs to dispatch operations in the IR graph to different
platforms, for example, shape-related operations will be dispatched to *TargetPlatform.SHAPE_OR_INDEX*, and
non-quantable operations will be dispatched to *TargetPlatform.FP32*, which means they will always run in
fp32 mode and no quantization is applied ever

```python
from ppq.api.interface import dispatch_graph

ppq_graph_ir = dispatch_graph(ppq_graph_ir, target_platform, setting)
```
then we can begin our quantization process using all prepared information

## Initialize An Executor
All operations are exectuted by *TorchExecutor* instances in PPQ, and as you can see from
[default.py](../ppq/executor/torch/default.py), the inner operation executing logic
is implemented using pytorch
```python
from ppq.executor import TorchExecutor

executor = TorchExecutor(ppq_graph_ir, device='cuda') # for cuda execution
executor = TorchExecutor(ppq_graph_ir, device='cpu')  # for cpu execution
```

## Quantization
PPQ will designate a quantizer for your target platform which would follow the following
conventions to actually run the quantization
1. Prepare dispatched graph IR for calibration
2. Refine quantization behaviors for certain operations
3. Run calibration process for parameters and activations
4. Render quantization parameters
5. Run activated optimization algorithms specified in quantization setting

```python

from ppq.api.interface import QUANTIZER_COLLECTION

quantizer = QUANTIZER_COLLECTION[target_platform](graph=ppq_graph_ir)
quantizer.quantize(
        inputs=dummy_input,                         # some random input tensor, should be list or dict for multiple inputs
        calib_dataloader=dataloader,                # calibration dataloader
        executor=executor,                          # executor in charge of everywhere graph execution is needed
        setting=setting,                            # quantization setting
        calib_steps=calib_steps,                    # number of batched data needed in calibration, 8~512
        collate_fn=lambda x: x.to(EXECUTING_DEVICE) # final processing of batched data tensor
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

PPQ has provided powerful analysis tools to analyse precision degradation in different layers of the quantized graph,
*graphwise_error_analyse* takes quantization error accumulation during execution into consideration while
*layerwise_error_analyse* considers quantization error layer-separetely, if you want to know the global performance
of quantized graph by analysing the signal noise ratio of fp32 outputs and quantized outputs of different layers.

```python

from ppq.quantization.analyse import layerwise_error_analyse, graphwise_error_analyse

graphwise_error_analyse(
    graph=quantized, # ppq ir graph
    running_device=EXECUTING_DEVICE, # cpu or cuda
    method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
    steps=32, # how many batches of data will be used for error analysis
    dataloader=dataloader,
    collate_fn=lambda x: x.to(EXECUTING_DEVICE)
)
```

or you want to analyse quantization error brought in by every layer,

```python

layerwise_error_analyse(
    graph=quantized,
    running_device=EXECUTING_DEVICE,
    method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
    steps=32,
    dataloader=dataloader,
    collate_fn=lambda x: x.to(EXECUTING_DEVICE)
)
```


## Export

To deploy your model on the target backend, appropriate format of quantized model and corresponding quantization
parameters should be exported from the quantized PPQ IR graph. PPQ will designate different exporters for different
target platforms. For example, if OpenPPL CUDA(*PPL_CUDA_INT8*) is the desired backend, *PPLBackendExporter* will
export an onnx model and a json file specifying quantization parameters, for more target platforms and exporters,
please check [interface.py](../ppq/api/interface.py)

Usually the chozen target platform determines the exact exporting format of the quantized IR graph, but sometimes
you might want to export in a different format, say if you want to deploy your model on *PPL_CUDA_INT8*

```python
from ppq.api.interface import export_ppq_graph

export_platform = TargetPlatform.PPL_CUDA_INT8  # could be other platforms in TargetPlatform class
export_ppq_graph(graph=ppq_ir_graph, platform=export_platform, graph_save_to='quantized', config_save_to='quantized.json')
```

or if you want to deploy your model on *NCNN_INT8*, where a quantization table file is needed

```python
export_ppq_graph(graph=ppq_ir_graph, platform=TargetPlatform.NCNN_INT8, graph_save_to='quantized', config_save_to='quantized.table')
```
