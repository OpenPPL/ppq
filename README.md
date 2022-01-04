# PPL Quantization Tool(PPL 量化工具)

PPL Quantization Tool (PPQ) is a powerful offline neural network quantization tool with custimized IR, executor, dispacher and optimization passes.

# Features

* Quantable graph, an quantization-oriented network representation.
* Quantize with Cuda, quantization simulating are 3x ~ 50x faster than PyTorch.
* Hardware-friendly, simulating calculations are mostly identical with hardware.
* Multi-platform support.

# Installation

To release the power of this advanced quantization tool, at least one CUDA computing device is required.
Install CUDA from [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive), PPL Quantization Tool will use CUDA compiler to compile cuda kernels at runtime.

ATTENTION: For users of pytorch, pytorch might bring you a minimized CUDA libraries, which will not satisfy the requirement of this tool, you have to install CUDA from NVIDIA manually.

ATTENTION: Make sure your python version is >= 3.6.0. PPL Quantization Tool is written with dialects that only by python >= 3.6.0.

* Install from source:

1. Run following code with your terminal(For windows user, use command line instead).

```bash
git clone https://github.com/openppl-public/ppq.git
cd ppq
python setup.py install
```

2. Wait for python finish its installation and pray for bug free.

* Install from Pip:

1. pre-built wheels are maintained in [PPQ](https://pypi.org/project/ppq/0.5.2/#files), you could simply install ppq with the following command

```bash
python3 -m pip install ppq
```

# Tutorials and Examples

1. User guide, system design doc can be found at /doc/pages/instructions of this repository, PPL Quantization Tool documents are written with pure html5.
2. Examples can be found at /ppq/samples.
3. Let's quantize your network with following code:

```python
from ppq.api import export_ppq_graph, quantize_torch_model
from ppq import TargetPlatform

# quantize your model within one single line:
quantized = quantize_torch_model(
    model=model, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=(1, 3, 224, 224),
    setting=quant_setting, collate_fn=collate_fn,
    platform=TargetPlatform.PPL_CUDA_INT8,
    device=DEVICE, verbose=0)

# export quantized graph with another line:
export_ppq_graph(
    graph=quantized, platform=TargetPlatform.PPL_CUDA_INT8,
    graph_save_to='Output/quantized(onnx).onnx',
    config_save_to='Output/quantized(onnx).json')
```

# Contact Us

| WeChat Official Account | QQ Group |
| :----:| :----: |
| OpenPPL | 627853444 |
| ![OpenPPL](doc/assets/img/qrcode_for_gh_303b3780c847_258.jpg)| ![QQGroup](doc/assets/img/qqgroup_s.jpg) |

Email: openppl.ai@hotmail.com

# Other Resources

* [Sensetime Parrots](https://www.sensetime.com/cn)
* [Sensetime Parrots Primitive Libraries](https://github.com/openppl-public/ppl.nn)
* [Sensetime mmlab](https://github.com/open-mmlab)

# Contributions

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

# Benchmark

PPQ is tested with models from mmlab-classification, mmlab-detection, mmlab-segamentation, mmlab-editing, here we listed part of out testing result.

* No quantization optimization procedure is applied with following models.

| Model | Type | Calibration | Dispatcher | Metric | PPQ(sim) | PPLCUDA | FP32 |
|  ----  | ----  |   ----  | ----  |  ----  | ----  |  ----  |  ----  |
| Resnet-18  | Classification | 512 imgs | conservative | Acc-Top-1 | 69.50% | 69.42% | 69.88% |
| ResNeXt-101 | Classification | 512 imgs | conservative | Acc-Top-1 | 78.46% | 78.37% | 78.66% |
| SE-ResNet-50 | Classification | 512 imgs | conservative | Acc-Top-1 | 77.24% | 77.26% | 77.76% |
| ShuffleNetV2 | Classification | 512 imgs | conservative | Acc-Top-1 | 69.13% | 68.85% | 69.55% |
| MobileNetV2  | Classification | 512 imgs | conservative | Acc-Top-1 | 70.99% | 71.1% | 71.88% |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| retinanet | Detection | 32 imgs | pplnn | bbox_mAP | 36.1% | 36.1% | 36.4% |
| faster_rcnn | Detection | 32 imgs | pplnn | bbox_mAP | 36.6% | 36.7% | 37.0% |
| fsaf | Detection | 32 imgs | pplnn | bbox_mAP | 36.5% | 36.6% | 37.4% |
| mask_rcnn | Detection | 32 imgs | pplnn | bbox_mAP | 37.7% | 37.6% | 37.9% |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| deeplabv3 | Segamentation | 32 imgs | conservative | aAcc / mIoU | 96.13% / 78.81% | 96.14% / 78.89%  | 96.17% / 79.12% |
| deeplabv3plus | Segamentation | 32 imgs | conservative | aAcc / mIoU | 96.27% / 79.39% | 96.26% / 79.29%  | 96.29% / 79.60% |
| fcn | Segamentation | 32 imgs | conservative | aAcc / mIoU | 95.75% / 74.56% | 95.62% / 73.96%  | 95.68% / 72.35% |
| pspnet | Segamentation | 32 imgs | conservative | aAcc / mIoU | 95.79% / 77.40% | 95.79% / 77.41%  | 95.83% / 77.74% |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| srcnn | Editing | 32 imgs | conservative | PSNR / SSIM | 27.88% / 79.70% | 27.88% / 79.07%  | 28.41% / 81.06% |
| esrgan | Editing | 32 imgs | conservative | PSNR / SSIM | 27.84% / 75.20% | 27.49% / 72.90%  | 27.51% / 72.84% |

* PPQ(sim) stands for PPQ quantization simulator's result.
* Dispatcher stands for dispatching policy of PPQ.
* Classification models are evaluated with ImageNet, Detection and Segamentation models are evaluated with COCO dataset, Editing models are evaluated with DIV2K dataset.
* All calibration datasets are randomly picked from training data.

# License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
