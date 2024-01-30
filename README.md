![Banner](https://oss.sensetime.com/20210820/9212d4b51db2e186dc39095b9e01cd3a/ccaf7b3f572fbe398f0d42e24435fc59.jpg)

## PPL Quantization Tool 0.6.6 (PPL 量化工具)

PPQ 是一个可扩展的、高性能的、面向工业应用的神经网络量化工具。

神经网络量化，作为一种常用的神经网络加速方案自 2016 年以来被广泛地应用。相比于神经网络剪枝与架构搜索，网络量化的泛用性更强，具有较高的工业实用价值。特别是对于端侧芯片而言，在片上面积与功耗都受到限制的场景下，我们总是希望将所有浮点运算转换为定点运算。量化技术的价值在于浮点运算与访存是十分昂贵的，它依赖于复杂的浮点运算器以及较高的访存带宽。如果我们能够在可接受的范围内使用较低位宽的定点运算近似浮点结果，这将使得我们在芯片电路设计、系统功耗、系统延迟与吞吐量等多方面获得显著的优势。

我们正处在时代的浪潮之中，基于神经网络的人工智能正快速发展，图像识别、图像超分辨率、内容生成、模型重建等技术正改变我们的生活。与之俱来的，是不断变化的模型结构，成为摆在模型量化与部署前的第一道难关。为了处理复杂结构，我们设计了完整的计算图逻辑结构与图调度逻辑，这些努力使得 PPQ 能够解析并修改复杂的模型结构，自动判定网络中的量化区与非量化区，并允许用户对调度逻辑进行手动控制。

网络的量化与性能优化是严峻的工程问题，我们希望用户能够参与到网络的量化与部署过程中来，参与到神经网络的性能优化中来。为此我们在 Github 中提供相应的与部署相关学习资料，并在软件设计上刻意强调接口的灵活性。在我们不断的尝试与探索中，我们抽象出量化器这一逻辑类型，负责初始化不同硬件平台上的量化策略，并允许用户自定义网络中每一个算子、每一个张量的量化位宽、量化粒度与校准算法等。我们将量化逻辑重组为27个独立的量化优化过程 (Quantization Optimization Pass)，PPQ 的用户可以根据需求任意组合优化过程，完成高度灵活的量化任务。作为 PPQ 的使用者，您能够根据需求新增、修改所有优化过程，探索量化技术的新边界。

这是一个为处理复杂量化任务而生的框架 —— PPQ 的执行引擎是专为量化设计的，截止 PPQ 0.6.6 版本，软件一共内置 99 种常见的 Onnx 算子执行逻辑，并原生支持执行过程中的量化模拟操作。PPQ 可以脱离 Onnxruntime 完成 Onnx 模型的推理与量化。作为架构设计一部分，我们允许用户使用 Python + Pytorch 或 C++ / Cuda 为 PPQ 注册新的算子实现，新的逻辑亦可替换现有的算子实现逻辑。PPQ 允许相同的算子在不同平台上有不同的执行逻辑，从而支撑不同硬件平台的运行模拟。借助定制化的执行引擎与 PPQ Cuda Kernel 的高性能实现，使得 PPQ 具有极其显著的性能优势，往往能以惊人的效率完成量化任务。

PPQ 的开发与推理框架关系密切，这使得我们能够了解硬件推理的诸多细节，从而严格控制硬件模拟误差。在国内外众多开源工作者共同努力之下，目前 PPQ 支持与 TensorRT, OpenPPL, Openvino, ncnn, mnn, Onnxruntime, Tengine, Snpe, GraphCore, Metax 等多个推理框架协同工作，并预制了对应量化器与导出逻辑。PPQ 是一个高度可扩展的模型量化框架，借助 ppq.lib 中的函数功能，您能够将 PPQ 的量化能力扩展到其他可能的硬件与推理库上。我们期待与您一起把人工智能带到千家万户之间。

#### 在 0.6.6 的版本更新中，我们为你带来了这些功能：
   1. [FP8 量化规范](https://zhuanlan.zhihu.com/p/574825662)，PPQ 现在支持 E4M3, E5M2 等多种规范的 FP8 [量化模拟与训练](https://github.com/openppl-public/ppq/blob/master/ppq/samples/fp8_sample.py)
   2. [PFL 基础类库](https://github.com/openppl-public/ppq/blob/master/ppq/samples/yolo6_sample.py)，PPQ 现在提供一套更为基础的 api 函数帮助你完成更为灵活的量化
   3. 更为强大的 [图模式匹配](https://github.com/openppl-public/ppq/blob/master/ppq/IR/search.py) 与 [图融合功能](https://github.com/openppl-public/ppq/blob/master/ppq/IR/morph.py)
   4. 基于 Onnx 的模型 [QAT](https://github.com/openppl-public/ppq/blob/master/ppq/samples/QAT/imagenet.py) 功能
   5. 全新的 [TensorRT](https://github.com/openppl-public/ppq/blob/master/md_doc/deploy_trt_by_OnnxParser.md) 量化与导出逻辑
   6. 全球最大的量化模型库 [OnnxQuant](https://github.com/openppl-public/ppq/tree/master/ppq/samples/QuantZoo)
   7. 其他未知的软件特性

### Installation (安装方法)

1. Install CUDA from [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

2. Install Complier

```bash
apt-get install ninja-build # for debian/ubuntu user
yum install ninja-build # for redhat/centos user
```

For Windows User:

  (1) Download ninja.exe from [https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases), add it to Windows PATH.

  (2) Install Visual Studio 2019 from [https://visualstudio.microsoft.com](https://visualstudio.microsoft.com/zh-hans/).

  (3) Add your C++ compiler to Windows PATH Environment, if you are using Visual Studio, it should be like "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x86"

  (4) Update PyTorch version to 1.10+.

3. Install PPQ

```bash
git clone https://github.com/openppl-public/ppq.git
cd ppq
pip install -r requirements.txt
python setup.py install
```

* Install PPQ from our docker image (optional):

```bash
docker pull stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5

docker run -it --rm --ipc=host --gpus all --mount type=bind,source=your custom path,target=/workspace stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5 /bin/bash

git clone https://github.com/openppl-public/ppq.git
cd ppq
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

* Install PPQ using pip (optional):

```bash
python3 -m pip install ppq
```

### Learning Path (学习路线)

#### PPQ 基础用法及示例脚本
| | **Description 介绍** | **Link 链接** |
| :-: | :- | :-: |
| 01 | 模型量化 | [onnx](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/quantize.py), [caffe](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_caffe_model.py), [pytorch](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_torch_model.py) |
| 02 | 执行器 | [executor](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/execute.py) |
| 03 | 误差分析 | [analyser](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/analyse.py) |
| 04 | 校准器 | [calibration](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/calibration.py) |
| 05 | 网络微调 | [finetune](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/finetune.py) |
| 06 | 网络调度 | [dispatch](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/dispatch.py) |
| 07 | 最佳实践 | [Best Practice](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/bestPractice.py) |
|  |  | |
| 08 | 目标平台 | [platform](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/targetPlatform.py) |
| 09 | 优化过程 | [Optim](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/optimization.py) |
| 10 | 图融合 | [Fusion](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/fusion.py) |

#### PPQ 优化过程文档
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

#### 视频资料
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

#### 量化部署教程
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

#### Dive into PPQ 深入理解量化框架
|  | **Desc 介绍** | **Link 链接** |
| :-: | :-: | :-: |
| 01 | PPQ 量化执行流程 |  [link](https://www.bilibili.com/video/BV1kt4y1b75m) |
| 02 | PPQ 网络解析 |  [link](https://www.bilibili.com/video/BV16B4y1h7u4) |
| 03 | PPQ 量化图调度 | [link](https://www.bilibili.com/video/BV1ig411f7f5) |
| 04 | PPQ 目标平台与 TQC |  [link](https://www.bilibili.com/video/BV1Lf4y1o7Zd) |
| 05 | PPQ 量化器 |  [link](https://www.bilibili.com/video/BV1494y1971i) |
| 06 | PPQ 量化优化过程 |  [link](https://www.bilibili.com/video/BV1zT411g7Ly) |
| 07 | PPQ 量化函数 |  [link](https://www.bilibili.com/video/BV1CU4y1q7tr) |

### Contact Us

| WeChat Official Account | QQ Group |
| :----:| :----: |
| OpenPPL | 627853444 |
| ![OpenPPL](assets/OpenPPL.jpg)| ![QQGroup](assets/QQGroup.jpg) |

Email: openppl.ai@hotmail.com

### Other Resources

* [Sensetime Parrots](https://www.sensetime.com/cn)
* [Sensetime Parrots Primitive Libraries](https://github.com/openppl-public/ppl.nn)
* [Sensetime mmlab](https://github.com/open-mmlab)

### Contributions

  We appreciate all contributions. If you are planning to contribute back bug fixes, please do so without any further discussion.

  If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

### Benchmark

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
| deeplabv3 | Segmentation | 32 imgs | conservative | aAcc / mIoU | 96.13% / 78.81% | 96.14% / 78.89%  | 96.17% / 79.12% |
| deeplabv3plus | Segmentation | 32 imgs | conservative | aAcc / mIoU | 96.27% / 79.39% | 96.26% / 79.29%  | 96.29% / 79.60% |
| fcn | Segmentation | 32 imgs | conservative | aAcc / mIoU | 95.75% / 74.56% | 95.62% / 73.96%  | 95.68% / 72.35% |
| pspnet | Segmentation | 32 imgs | conservative | aAcc / mIoU | 95.79% / 77.40% | 95.79% / 77.41%  | 95.83% / 77.74% |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| srcnn | Editing | 32 imgs | conservative | PSNR / SSIM | 27.88% / 79.70% | 27.88% / 79.07%  | 28.41% / 81.06% |
| esrgan | Editing | 32 imgs | conservative | PSNR / SSIM | 27.84% / 75.20% | 27.49% / 72.90%  | 27.51% / 72.84% |

* PPQ(sim) stands for PPQ quantization simulator's result.
* Dispatcher stands for dispatching policy of PPQ.
* Classification models are evaluated with ImageNet, Detection and Segmentation models are evaluated with the COCO dataset, Editing models are evaluated with DIV2K dataset.
* All calibration datasets are randomly picked from training data.

### License

![Logo](assets/logo.png)

This project is distributed under the [Apache License, Version 2.0](LICENSE).
