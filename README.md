# PPL Quantization Tool 0.6.5(PPL 量化工具)
PPL QuantTool (PPQ) is a highly efficient neural network quantization tool with custimized IR, cuda based executor, automatic dispacher and powerful optimization passes. Together with OpenPPL ecosystem, we offer you this industrial-grade network deploy tool that empowers AI developers to unleash the full potential of AI hardware. With quantization and other optimizations, nerual network model can run 5~10x faster than ever.

PPL QuantTool 是一个高效的工业级神经网络量化工具。
PPQ 被设计为一个灵活而全面的神经网络离线量化工具，我们允许你控制对量化进行细致入微的控制，同时严格控制硬件模拟误差。即便在网络极度复杂的情况下，我们依然能够得到正确的网络量化结果。PPQ 有着自定义的量化算子库、网络执行器、神经网络调度器、量化计算图等独特设计，这些为量化而设计的组件相互协作，共同构成了这一先进神经网络量化框架。借助 PPQ, OpenPPL, TensorRT, Tengine，ncnn等框架，你可以将神经网络模型提速 10 ~ 100 倍，并部署到多种多样的目标硬件平台，我们期待你将人工智慧带到千家万户之间。

目前 PPQ 使用 onnx(opset 11 ~ 13) 模型文件作为输入，覆盖常用的 90 余种 onnx 算子类型。如果你是 Pytorch, tensorflow 的用户，你可以使用这些框架提供的方法将模型转换到 onnx。PPQ 支持向 TensorRT, OpenPPL, Openvino, ncnn, Onnxruntime, Tengine, Snpe 等多个推理引擎生成目标文件并完成部署。借助算子自定义与平台自定义功能，你还可以将 PPQ 的量化能力扩展到其他可能的硬件上。

## Learning Path 学习路线

### PPQ Basic 基础内容
| | **Description 介绍** | **Link 链接** |
| :-: | :- | :-: |
| 01 | 欢迎，在第一部分的内容中，我们首先向你展示如何使用 ppq 量化来自 pytorch, onnx, caffe 的模型 | [onnx](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/quantize.py), [caffe](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_caffe_model.py), [pytorch](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_torch_model.py) |
| 02 | 接下来让我们看看如何执行量化后的模型 | [executor](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/execute.py) |
| 03 | 渐入佳境，让我们试着使用 PPQ 的误差分析功能 | [analyser](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/analyse.py) |
| 04 | 我的网络误差很高？让我们调整校准算法来尝试降低误差 | [calibration](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/calibration.py) |
| 05 | 进一步降低量化误差，为什么不让我们对网络展开进一步的训练？ | [finetune](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/finetune.py) |
| 06 | 让我们看看 PPQ 的图调度功能能帮我们做什么 | [dispatch](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/dispatch.py) |
| 07 | 最佳实践！向你展示模型在 PPQ 中的量化流程 | [Best Practice](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/bestPractice.py) |
|  |  | |
| 08 | 创建我们自己的量化规则！了解目标平台与量化器 | [platform](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/targetPlatform.py) |
| 09 | 自定义量化优化过程 | [Optim](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/optimization.py) |
| 10 | 自定义图融合过程与量化管线 | [Fusion](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/fusion.py) |

### PPQ Optim 优化过程文档
| | **Description 介绍** | **Link 链接** |
| :-: | :-: | :-: |
| 01 | QuantSimplifyPass(通用量化精简过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/QuantSimplify.md) |
| 02 | QuantFusionPass(通用量化图融合过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/QuantFusion.md) |
| 03 | QuantAlignmentPass(通用量化对齐过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/QuantAlignment.md) |
| 04 | RuntimeCalibrationPass(参数校准过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/RuntimeCalibrationPass.md) |
| 05 | BiasCorrectionPass(Bias修正过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/BiasCorrectionPass.md) |
| 06 | QuantSimplifyPass(通用量化精简过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/QuantSimplify.md) |
| 07 | LayerwiseEqualizationPass(层间权重均衡过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/LayerwiseEqualization.md) |
| 08 | LayerSpilitPass(算子分裂过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/LayerSpilit.md) |
| 09 | LearnedStepSizePass(网络微调过程) | [doc](https://github.com/openppl-public/ppq/blob/master/md_doc/Passes/LearnedStepSizePass.md) |
| 10 | Other(其他) | [refer to](https://github.com/openppl-public/ppq/tree/master/ppq/quantization/optim) |

### Quantized Computing 量化计算
|  | **Desc 介绍** | **Link 链接** |
| :-: | :-: | :-: |
| 01 | 计算机体系结构基础知识 |  [link](https://www.bilibili.com/video/BV1gS4y1Y7KR) |
| 02 | 网络性能分析 |  [link](https://www.bilibili.com/video/BV1oT4y1h73e) |
| 03 | 量化计算原理 | [part1](https://www.bilibili.com/video/BV1fB4y1m7fJ), [part2](https://www.bilibili.com/video/BV1qA4y1Q7Uh) |
| 04 | 图优化与量化模拟 |  [link](https://www.bilibili.com/video/BV1Kr4y1n7cy) |
| 05 | 图调度与模式匹配 |  [link](https://www.bilibili.com/video/BV1xY411A7ea) |
| 06 | 神经网络部署 |  [link](https://www.bilibili.com/video/BV1t34y1E7Fz) |
| 07 | 量化参数选择 |  [link](https://www.bilibili.com/video/BV1QF41157aM) |
| 08 | 量化误差传播分析 |  [link](https://www.bilibili.com/video/BV1CU4y1q7tr) |

### PPQ Deploy 量化部署教程
| **使用例子(Examples)** | **网络部署平台(Platform)** | **输入模型格式(Format)** | **链接(Link)** | **相关视频(Video)** |
| :- | :-: | :-: | :-: | :-: |
| `TensorRT` |  |  |  |  |
| 使用 Torch2trt 加速你的网络 | pytorch | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Torch2trt.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 量化训练 | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 后训练量化(PPQ) | TensorRT | onnx |<p align="left">[1. Quant with TensorRT OnnxParser](https://github.com/openppl-public/ppq/blob/master/md_doc/deploy_trt_by_OnnxParser.md)<p align="left">[2. Quant with TensorRT API](https://github.com/openppl-public/ppq/blob/master/md_doc/deploy_trt_by_api.md)|  |
| TensorRT fp32 部署 | TensorRT | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Fp32.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 性能比较 | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Benchmark.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT Profiler | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Profiling.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| `onnxruntime` |  |  |  |  |
| 使用 onnxruntime 加速你的网络 | onnxruntime | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Onnxruntime/Example_Fp32.py) | [link](https://www.bilibili.com/video/BV1t34y1E7Fz "Network Deploy") |
| onnx 后训练量化(PPQ) | onnxruntime | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Onnxruntime/Example_PTQ.py) | [link](https://www.bilibili.com/video/BV1t34y1E7Fz "Network Deploy") |
| onnxruntime 性能比较 | onnxruntime | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Onnxruntime/Example_Benchmark.py) | [link](https://www.bilibili.com/video/BV1t34y1E7Fz "Network Deploy") |
| `openvino` |  |  |  |  |
| 使用 openvino 加速你的网络 | openvino | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Openvino/Example_Fp32.py) ||
| openvino 量化训练 | openvino | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Openvino/Example_QAT.py) ||
| openvino 后训练量化(PPQ) | openvino | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Openvino/Example_PTQ.py) ||
| openvino 性能比较 | openvino | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Openvino/Example_Benchmark.py) ||
| `snpe` |  |  |  |  |
| snpe 后训练量化(PPQ) | snpe | caffe | [link](https://github.com/openppl-public/ppq/blob/master/md_doc/inference_with_snpe_dsp.md) ||
| `ncnn` |  |  |  |  |
| ncnn 后训练量化(PPQ) | ncnn | onnx | [link](https://github.com/openppl-public/ppq/blob/master/md_doc/inference_with_ncnn.md) ||
| `OpenPPL` |  |  |  |  |
| ppl cuda 后训练量化(PPQ) | ppl cuda | onnx | [link](https://github.com/openppl-public/ppq/blob/master/md_doc/inference_with_ppl_cuda.md) ||

### Appendix 额外内容
| **使用例子(Examples)** | **网络部署平台(Platform)** | **输入模型格式(Format)** | **链接(Link)** | **相关视频(Video)** |
| :- | :-: | :-: | :-: | :-: |
| 注册量化代理函数 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/custimize_quant_func.py) ||
| 自定义量化算子 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/custimized_quant.py) ||
| 绕过与量化无关的算子 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/bypass_nms.py) ||
| onnx 格式转换 | - | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/onnx_converter.py) ||
| `Yolo` |  |  |  |  |
| 使用 TensorRT 推理 Yolo 模型 | TensorRT | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Yolo/00_FloatModel.py) | [link](https://www.bilibili.com/video/BV1ua411D7vn) |
| 使用 PPQ 量化 Yolo | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Yolo/02_Quantization.py) | [link](https://www.bilibili.com/video/BV1ua411D7vn) |
| 分析 Yolo 量化性能 | TensorRT | onnx | [benckmark](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Yolo/04_Benchmark.py), [profiler](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Yolo/03_Profile.py) | [link](https://www.bilibili.com/video/BV1jN4y1M7jt) |
| 尝试修改 Yolo 量化策略以提高性能 | TensorRT | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Yolo/05_QuantizationAgain.py) | [link](https://www.bilibili.com/video/BV1ra411S7io) |

### Dive into PPQ 深入理解量化框架
|  | **Desc 介绍** | **Link 链接** |
| :-: | :-: | :-: |
| 01 | PPQ 量化执行流程 |  [link](https://www.bilibili.com/video/BV1kt4y1b75m) |
| 02 | PPQ 网络解析 |  [link](https://www.bilibili.com/video/BV16B4y1h7u4) |
| 03 | PPQ 量化图调度 | [link](https://www.bilibili.com/video/BV1ig411f7f5) |
| 04 | PPQ 目标平台与 TQC |  [link](https://www.bilibili.com/video/BV1Lf4y1o7Zd) |
| 05 | PPQ 量化器 |  [link](https://www.bilibili.com/video/BV1494y1971i) |
| 06 | PPQ 量化优化过程 |  [link](https://www.bilibili.com/video/BV1zT411g7Ly) |
| 07 | PPQ 量化函数 |  [link](https://www.bilibili.com/video/BV1CU4y1q7tr) |

## Installation

To release the power of this advanced quantization tool, at least one CUDA computing device is required.
Install CUDA from [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive), PPL Quantization Tool will use CUDA compiler to compile cuda kernels at runtime.

ATTENTION: For users of PyTorch, PyTorch might bring you a minimized CUDA libraries, which will not satisfy the requirement of this tool, you have to install CUDA from NVIDIA manually.

ATTENTION: Make sure your Python version is >= 3.6.0. PPL Quantization Tool is written with dialects that only supported by Python >= 3.6.0.

* Install dependencies:
    * For Linux User, use following command to install ninja:
    ```bash
    sudo apt install ninja-build
    ```

    * For Windows User:
        * Download ninja.exe from [https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases), add it to Windows PATH Environment
        * Download Visual Studio 2019 from [https://visualstudio.microsoft.com](https://visualstudio.microsoft.com/zh-hans/), if you already got a c++ compiler, you can skip this step.
        * Add your c++ compiler to Windows PATH Environment, if you are using Visual Studio, it should be something like "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x86"
        * Update pytorch to 1.10+.

#### There are three ways to install ppq

* Install PPQ from source:

Run following code with your terminal(For windows user, use command line instead).

```bash
git clone https://github.com/openppl-public/ppq.git
cd ppq
pip install -r requirements.txt
python setup.py install
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

* Install PPQ from our docker image:

```bash
docker pull stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5

docker run -it --rm --ipc=host --gpus all --mount type=bind,source=your custom path,target=/workspace stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5 /bin/bash

git clone https://github.com/openppl-public/ppq.git
cd ppq
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

* Install PPQ from Pip:

**Note that this installation method currently does not support tensorrt write parameter quantization, we haven't updated yet**.
Wait for Python finish its installation and pray for bug free.
pre-built wheels are maintained in [PPQ](https://pypi.org/project/ppq/), you could simply install ppq with the following command(**You should notice that install from pypi might get an outdated version ...**.)
```bash
python3 -m pip install ppq
```

## Contact Us

| WeChat Official Account | QQ Group |
| :----:| :----: |
| OpenPPL | 627853444 |
| ![OpenPPL](doc/assets/img/qrcode_for_gh_303b3780c847_258.jpg)| ![QQGroup](doc/assets/img/qqgroup_s.jpg) |

Email: openppl.ai@hotmail.com

## Other Resources

* [Sensetime Parrots](https://www.sensetime.com/cn)
* [Sensetime Parrots Primitive Libraries](https://github.com/openppl-public/ppl.nn)
* [Sensetime mmlab](https://github.com/open-mmlab)

## Contributions

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

## Benchmark

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

## License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
