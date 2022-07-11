# PPL Quantization Tool 0.6.5(PPL 量化工具)
PPL QuantTool (PPQ) is a highly efficient neural network quantization tool with custimized IR, cuda based executor, automatic dispacher and powerful optimization passes. Together with OpenPPL ecosystem, we offer you this industrial-grade network deploy tool that empowers AI developers to unleash the full potential of AI hardware. With quantization and other optimizations, nerual network model can run 5~10x faster than ever.

PPL QuantTool 是一个高效的工业级神经网络量化工具。
PPQ 被设计为一个灵活而全面的神经网络离线量化工具，我们允许你控制对量化进行细致入微的控制，同时严格控制硬件模拟误差。即便在网络极度复杂的情况下，我们依然能够得到正确的网络量化结果。PPQ 有着自定义的量化算子库、网络执行器、神经网络调度器、量化计算图等独特设计，这些为量化而设计的组件相互协作，共同构成了这一先进神经网络量化框架。借助 PPQ, OpenPPL, TensorRT, Tengine，ncnn等框架，你可以将神经网络模型提速 10 ~ 100 倍，并部署到多种多样的目标硬件平台，我们期待你将人工智慧带到千家万户之间。

目前 PPQ 使用 onnx(opset 11 ~ 13) 模型文件作为输入，覆盖常用的 90 余种 onnx 算子类型。如果你是 Pytorch, tensorflow 的用户，你可以使用这些框架提供的方法将模型转换到 onnx。PPQ 支持向 TensorRT, OpenPPL, Openvino, ncnn, Onnxruntime, Tengine, Snpe 等多个推理引擎生成目标文件并完成部署。借助算子自定义与平台自定义功能，你还可以将 PPQ 的量化能力扩展到其他可能的硬件上。

# Code Example
| **使用例子(Examples)** | **网络部署平台(Platform)** | **输入模型格式(Format)** | **链接(Link)** | **相关视频(Video)** |
| :- | :-: | :-: | :-: | :-: |
| `新手上路` |  |  |  | [link](https://www.bilibili.com/video/BV1oT4y1h73e "Analysing Your Model") |
| 量化你的第一个 pytorch 网络 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_torch_model.py) | |
| 量化你的第一个 onnx 网络 | - | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_onnx_model.py) | |
| 量化你的第一个 caffe 网络 | - | caffe | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_caffe_model.py) | |
| 走进 PPQ | - | onnx | [link](https://github.com/openppl-public/ppq/blob/master/md_doc/how_to_use.md) | [link](https://www.bilibili.com/video/BV1934y147p2 "PPQ User Guide") |
| 量化函数 | - | - | [link](https://github.com/openppl-public/ppq/blob/master/md_doc/ppq_qlinear_function.md) |  |
| 量化参数选择 | - | - |  | [link](https://www.bilibili.com/video/BV1QF41157aM) |
| 量化误差分析 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/analyse.py) | [link](https://www.bilibili.com/video/BV1xY411A7ea "Graph Dispatching & Pattern Matching.") |
| 算子调度 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/dispatch.py) | [link](https://www.bilibili.com/video/BV1xY411A7ea "Graph Dispatching & Pattern Matching.") |
| 执行量化网络 | PPQ Executor | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/execute.py) ||
| 启动 cuda kernel 加速执行 | PPQ Executor | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/enable_cuda_kernel.py) ||
| `TensorRT` |  |  |  |  |
| 使用 Torch2trt 加速你的网络 | pytorch | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Torch2trt.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 量化训练 | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 后训练量化(PPQ) | TensorRT | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_PTQ.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT fp32 部署 | TensorRT | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Fp32.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 性能比较 | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Benchmark.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
| TensorRT 性能分析工具 | TensorRT | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Profiling.py) | [link](https://www.bilibili.com/video/BV1AU4y127Uo) |
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
| `自定义量化` |  |  |  |  |
| 添加自定义量化平台 1 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/md_doc/add_new_platform.md) ||
| 添加自定义量化平台 2 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/create_your_platform.py) ||
| 注册量化代理函数 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/custimize_quant_func.py) ||
| 自定义量化算子 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/custimized_quant.py) ||
| 绕过与量化无关的算子 | - | pytorch | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/bypass_nms.py) ||
| `其他` |  |  |  |  |
| onnx 格式转换 | - | onnx | [link](https://github.com/openppl-public/ppq/blob/master/ppq/samples/onnx_converter.py) ||

# Video Tutorial(Bilibili 视频教程)
Watch video tutorial series on www.bilibili.com, following are links of PPQ tutorial links(Only Chinese version).

* 安装教程: [https://www.bilibili.com/video/BV1WS4y1N7Kn](https://www.bilibili.com/video/BV1WS4y1N7Kn "PPQ Installation Tutorial")
* 使用教程: [https://www.bilibili.com/video/BV1934y147p2](https://www.bilibili.com/video/BV1934y147p2 "PPQ User Guide")
* 基础知识：[https://www.bilibili.com/video/BV1gS4y1Y7KR](https://www.bilibili.com/video/BV1gS4y1Y7KR "Basic Theory")
* 网络性能分析：[https://www.bilibili.com/video/BV1oT4y1h73e](https://www.bilibili.com/video/BV1oT4y1h73e "Analysing Your Model")
* 量化计算原理(Part 1)：[https://www.bilibili.com/video/BV1fB4y1m7fJ](https://www.bilibili.com/video/BV1fB4y1m7fJ "Quantized Computing")
* 量化计算原理(Part 2)：[https://www.bilibili.com/video/BV1qA4y1Q7Uh](https://www.bilibili.com/video/BV1qA4y1Q7Uh "Quantized Computing")
* 图优化与量化模拟：[https://www.bilibili.com/video/BV1Kr4y1n7cy](https://www.bilibili.com/video/BV1Kr4y1n7cy "Graph Optimization & quantization simulating.")
* 图调度与模式匹配：[https://www.bilibili.com/video/BV1xY411A7ea](https://www.bilibili.com/video/BV1xY411A7ea "Graph Dispatching & Pattern Matching.")
* 神经网络部署: [https://www.bilibili.com/video/BV1t34y1E7Fz](https://www.bilibili.com/video/BV1t34y1E7Fz "Network Deploy")
* TensorRT 部署: [https://www.bilibili.com/video/BV1AU4y127Uo](https://www.bilibili.com/video/BV1AU4y127Uo "TensorRT Deploy")
* 量化参数选择: [https://www.bilibili.com/video/BV1QF41157aM](https://www.bilibili.com/video/BV1QF41157aM "Quantization Param Searching")
* 其他教程: 等待缓慢更新...

# Installation

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

* Install PPQ from source:

1. Run following code with your terminal(For windows user, use command line instead).

```bash
git clone https://github.com/openppl-public/ppq.git
cd ppq
pip install -r requirements.txt
python setup.py install
```

2. Wait for Python finish its installation and pray for bug free.

* Install PPQ from Pip:

1. pre-built wheels are maintained in [PPQ](https://pypi.org/project/ppq/), you could simply install ppq with the following command(You should notice that install from pypi might get an outdated version ...)

```bash
python3 -m pip install ppq
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
