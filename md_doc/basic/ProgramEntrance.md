# Basic Usage
This tutorial illustrates how you could quantize a model using high level API provided in PPQ, it's assumed that you have PPQ installed successfully in your working environment, for the installation part, please refer to [Installation](../../README.md). Code provided in this tutorial can be found in [ProgramEntrance.py](../../ProgramEntrance.py), basically it's where you get started and you could adjust it to your own needs.

## Preparation Work
Before everything gets started, you need to prepare your model(usually in onnx format or caffe format, if you are using models in other format, you need to convert them to onnx/caffe first) and the corresponding calibration dataset. Note that the
calibration dataset is commonly composed of a small subset(usually 200~1000 preprocessed input files in npy format or binary format) of training set. If you are using onnx model, generally the layout of your working directory could be

```
|--working
      |--data
           |--1.npy
           |--2.npy
           |--3.npy
           |-- ...
      |--model.onnx
```
or if your model is in caffe format,

```
|--working
      |--data
           |--1.npy
           |--2.npy
           |--3.npy
           |-- ...
      |--model.caffemodel
      |--model.prototxt
```

note that the prefix of the name of your model should be exactly *"model.\*"* for PPQ to identify your model correctly.

## Configuration

You need to specify the following configurations so that PPQ could parse your model and load calibration data correctly

* WORKING_DIRECTORY: the location of your working directory
* TARGET_PLATFORM: the target platform on which you want to deploy your model
* MODEL_TYPE: the model format, onnx/caffe are supported
* INPUT_LAYOUT: memory layout of input files in your calibration dataset, hwc/chw
* NETWORK_INPUTSHAPE: input shape of input files in your calibration dataset
* CALIBRATION_BATCHSIZE: batchsize of calibration dataloader, 16/32/64 are good choices if input data can be batched
* EXECUTING_DEVICE: executing device on which the graph is executed, cpu/cuda
* REQUIRE_ANALYSE: whether need to analyse quantization performance in a layerwise way
* DUMP_RESULT: whether need to dump output results of internal operations

for example, if you want to deploy an onnx model on OpenPPL CUDA INT8 platform, all you need is to specify

```python
from ppq import *
from ppq.api import *

WORKING_DIRECTORY     = 'working'
TARGET_PLATFORM       = TargetPlatform.PPL_CUDA_INT8
MODEL_TYPE            = NetworkFramework.ONNX  #  MODEL_TYPE = NetworkFramwork.CAFFE for caffe model
INPUT_LAYOUT          = 'chw'                  #  input data will be transposed to 'chw' if set 'hwc'
NETWORK_INPUTSHAPE    = [1, 3, 224, 224]       #  must be specified when input files are in binary format
CALIBRATION_BATCHSIZE = 16                     #  must be equal to 1 when input files are in dynamic shapes
EXECUTING_DEVICE      = 'cuda'                 #  set to cpu if you don't have gpu or cuda installed
REQUIRE_ANALYSE       = False
DUMP_RESULT           = False


dataloader = load_calibration_dataset(         # only support single-input situation, you need to create
    directory    = WORKING_DIRECTORY,          # your own dataloader when graph has multiple inputs
    input_shape  = NETWORK_INPUTSHAPE,
    batchsize    = CALIBRATION_BATCHSIZE,
    input_format = INPUT_LAYOUT)

```

## Quantization Setting

Quantization setting is an essential part which guides the behavior of quantizer, it controls steps taken in quantization process,
providing you the largest flexiability to adjust and customize the quantization process, for users who are not familiar with PPQ,
we provide a simple API to create a fundamental setting which allows you to focus on a small subset of configurations.

```python
SETTING = UnbelievableUserFriendlyQuantizationSetting(
    platform = TARGET_PLATFORM,   # target platform to deploy your model
    finetune_steps = 2500,        # number of total iterations in AdavancedOptimization algorithm
    finetune_lr = 1e-3,           # learning rate in AdavancedOptimization algorithm
    calibration = 'percentile',   # calibration algorithm controling computation of quantization parameters
    equalization = True,          # whether to perform data free quantization equalization
    non_quantable_op = None       # concrete operations which need fp32 precision and shouldn't be quantized
)
SETTING = SETTING.convert_to_daddy_setting()
```

Note that if you just want a plain quantization without involving any optimization methods

```python
SETTING = UnbelievableUserFriendlyQuantizationSetting(
    platform = TARGET_PLATFORM,
    finetune_steps = 0,
    calibration = 'percentile',
    equalization = False
)
SETTING = SETTING.convert_to_daddy_setting()
```
then the quantizer will simply quantize the graph and observe quantization parameters using algorithm
specified by *calibration*.

## Quantization

For every target platform, PPQ will designate a quantizer to conduct execution logic in quantization process,
for example, *PPLCUDAQuantizer* is responsible for *PPL_CUDA_INT8* platform, and *PPL_DSP_Quantizer* is responsible for
*PPL_DSP_INT8* platform, check [interface.py](../../ppq/api/interface.py) for more details. The quantizer will

1. prepare parsed graph IR for calibration
2. refine quantization behaviors for certain operations
3. run the calibration process for parameters and activations
4. compute and render quantization parameters
5. issue optimization algorithms as specified in quantization setting

You don't have to look into those details, a simple function call will do all the work

```python
quantized = quantize(
    working_directory=WORKING_DIRECTORY, setting=SETTING,
    model_type=MODEL_TYPE, executing_device=EXECUTING_DEVICE,
    input_shape=NETWORK_INPUTSHAPE, target_platform=TARGET_PLATFORM,
    dataloader=dataloader, calib_steps=32
)
```
Note that *calib_steps* specifies how many batches of data in dataloader will be referred in calibration,
it should be in 8~512 in consideration of efficiency.

## Inference
The quantized PPQ IR graph could be executed by *TorchExecutor*, which takes charge of graph execution in
PPQ, if you want to run inferences after quantization

```python

Input = torch.randn(1, 3, 224, 224).to(EXECUTING_DEVICE)
executor = TorchExecutor(graph=quantized, device=EXECUTING_DEVICE)

Outputs = executor.forward(Input) # quantized mode with quantization error

for op in quantized.operations.values(): # dequantize the graph, so that we can run
    if isinstance(op, QuantableOperation): # in fp32 mode with no precision loss
        op.dequantize()

Outputs = executor.forward(Input) # for fp32 mode
```

## Analysis

PPQ has provided powerful analysis tools to analyse precision degration in different layers of the quantized graph,
*graphwise_error_analysis* takes quantization error accumulation during execution into consideration while
*layerwise_error_analysis* considers quantization error layer-separetely, if you want to know the global performance
of quantized graph by analysing the signal noise ratio of fp32 outputs and quantized outputs of different layers.

```python

graphwise_error_analyse(
    graph=quantized,
    running_device=EXECUTING_DEVICE,
    method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
    steps=32,
    dataloader=dataloader,
    collate_fn=lambda x: x.to(EXECUTING_DEVICE)
)
```

or you want to analyse layer by layer, considering quantization error brought in every layer separately

```python

layerwise_error_analysis(
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
please check [interface.py](../../ppq/api/interface.py)

```python
export(working_directory=WORKING_DIRECTORY, quantized=quantized, platform=TARGET_PLATFORM)
```
then the exported files will appear in your working directory as *"quantized.\*"*

```
|--working
      |--data
      |--model.onnx
      |--quantized.onnx
      |--quantized.json
```
or if the model is supposed to be exported in caffe format, then *input_shapes* should be specified
```python
export(working_directory=WORKING_DIRECTORY, quantized=quantized, platform=TARGET_PLATFORM, input_shapes=[NETWORK_INPUTSHAPE])
```
