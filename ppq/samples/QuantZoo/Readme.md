# ONNX Quantization Model Zoo (OnnxQuant)

:smile: OnnxQuant 是目前最大的模型量化数据集，它包含 ONNX 模型，数据，以及相关的测试脚本。该数据集的提出用于推动模型量化在视觉模型中的应用与量化算法的研究，具备以下特点：

1. 可移植与可复现，所有模型均由 ONNX 格式提供。
2. 包含图像分类、图像分割、图像超分辨率、图像-文字识别、目标检测、姿态检测等多个任务的模型。
3. 提供切分好的 calibration 数据和 test 数据，提供模型精度测试脚本。
4. 提供灵活的量化器用于确定模型在不同量化规则下的理论性能，并提供 FP8 量化器。

:eyes: OnnxQuant 目前处于公开测试阶段，近几个月内仍然将发生修改与变动。

## 1. 如何使用

### 1.1 下载数据集：

1. 图像分类: 
    * https://pan.baidu.com/s/1VdMp9fPxPwh2sVPd1ikFKQ?pwd=lebu
2. 图像超分:
    * https://pan.baidu.com/s/1u7ZAVNtlaMHBzDzzq-1eCw?pwd=gzsb
    * https://drive.google.com/file/d/1ILJu4Y5RifqOuYCKjnVRe5MCkaXkJpoo/view?usp=sharing
3. 图像分割: 
    * https://pan.baidu.com/s/1LAS8LYyklz7kgkVUuxDlLg?pwd=db6s
    * https://drive.google.com/file/d/1U87xZwF39M6jr-k4QGrJ6e5sruZw_xAv/view?usp=sharing
4. 目标检测: 
    * https://pan.baidu.com/s/1uBvK-Wm1AKVrNgvA9E4lhA?pwd=9n06(还在努力补档)
    * https://drive.google.com/file/d/1fiu3VYvIb1L7fpI0T1EXggvxIE4mtxk8/view?usp=sharing
5. 姿态检测: 
    * https://pan.baidu.com/s/1F4Ui1j1AqsjfV5OOS-Fd4A?pwd=scff
    * https://drive.google.com/file/d/1HoJtpwHXfivO8imgIVJW1wnWBuF0jdZR/view?usp=sharing
6. 图像-文字识别: 
    * https://pan.baidu.com/s/1GyYvLbhibLL6kPIA1J0X7Q?pwd=vpxi
    * https://drive.google.com/file/d/1_WjP2a8g6fQubFNT63bx1oo98Cm347Py/view?usp=sharing
7. NLP:

### 1.2 建立工作目录：

在工作目录下建立文件夹 QuantZoo，解压上述文件到 QuantZoo 中。你将获得这样的文件夹结构

```
~/QuantZoo/Data/Cityscapes
~/QuantZoo/Data/Coco
~/QuantZoo/Data/DIV2K
~/QuantZoo/Data/IC15
~/QuantZoo/Data/Imagenet
~/QuantZoo/Model/yolo
~/QuantZoo/Model/Imagenet
~/QuantZoo/Model/mmedit
~/QuantZoo/Model/mmseg
~/QuantZoo/Model/ocr
~/QuantZoo/Quantizers.py
~/QuantZoo/Util.py
```

### 1.3 创建入口文件

将 https://github.com/openppl-public/ppq/tree/master/ppq/samples/QuantZoo 目录下的文件复制到工作目录下

```
~/QuantZoo_Imagenet.py
~/QuantZoo_OCR.py
~/QuantZoo_Yolo.py
~/QuantZoo_SuperRes.py
~/QuantZoo_Segmentation.py
~/QuantZoo
```

运行工作目录中的 python 文件即可完成 OnnxQuant 测试。

## 2. 环境依赖

1. 下载量化工具 ppq 0.6.6 以上版本，用户可以通过 pypi 进行安装

``` bash
pip install ppq
```

亦可以使用 github 上的最新版本进行安装

``` bash
git clone https://github.com/openppl-public/ppq.git
cd ppq
pip install -r requirements.txt
python setup.py install
```

2. 下载并安装所需的其他依赖库

  * numpy
  * onnx >= 1.9.0
  * protobuf
  * torch >= 1.6.0
  * tqdm
  * mmcv-full
  * cityscapesscripts
  * pycocotools

3. 安装编译环境(对于 FP8 量化而言，该环境是必须的)

``` bash
apt-get install ninja-build
```

## 3. 数据集与模型简介

在 OnnxQuant 中的所有模型已经预先完成了 Batchnorm 层的合并，并且模型已经完成训练过程。

### 3.1 图像分类

数据集：Imagenet

数据切分方式：
  * Calibration 数据为 Imagenet Validation Set 中随机抽取的 5000 张图片。
  * Test 数据为 Imagenet Validation Set 中随机抽取的 5000 张图片。
  * 提供数据切分脚本。

模型 efficientnet, mnasnet, mobilenetv2, mobilenetv3, resnet18, resnet50, vit 来自 torchvision.

模型 once_for_all: https://github.com/mit-han-lab/once-for-all

模型测试标准：分类准确率。

### 3.2 实例分割

数据集：cityscapes

数据切分方式：
  * Calibration 数据为 Cityscapes val 中随机抽取的 300 张图片。
  * Test 数据为 Cityscapes val 中随机抽取的 200 张图片。
  * 提供数据切分脚本。

模型：全部来自 mmseg

模型测试标准：Miou。

### 3.3 目标检测

数据集: Coco 2017

数据切分方式：

  * Calibration 数据为  Coco 2017 Validation Set 中随机抽取的 1500 张图片。
  * Test 数据为 Coco 2017 Validation Set 中随机抽取的 300 张图片。
  * 提供数据切分脚本。

模型：全部来自 mmyolo

模型测试标准：目标检测精准度(mAP)。

### 3.4 OCR

数据集：IC15

数据切分方式：

  * Calibration 数据为 IC15 train 数据集。
  * Test 数据为 IC15 test 数据集。

模型：全部来自 paddle ocr

模型测试标准：文字识别准确率。

### 3.5 图像超分辨率

数据集：DIV2K

数据切分方式：

  * Calibration 数据为 DIV2K_train 数据集。
  * Test 数据为 DIV2K_valid 数据集。降采样方式为 x4, bicubic。

模型：全部来自 mmedit

模型测试标准：峰值信噪比。

### 3.6 姿态检测

数据集：Coco 2017

数据切分方式：

  * Calibration 数据为 Coco validation 数据集中随机抽取的 1500 张人像。
  * Test 数据为 Coco validation 数据集中随机抽取的 ~500 张人像。

模型：全部来自 mmpose

模型测试标准：仅提供量化误差测试，不提供模型准确度测试。

## 4. OnnxQuant 模型量化规则

### 4.1 综述：
在 OnnxQuant 中，我们希望评估模型量化的理论性能，同时隔离推量框架的具体执行细节。

我们要求量化所有的 **卷积与全连接层**，并且只对上述层的 **输入变量和权重** 添加量化-反量化操作，其 **偏置项(Bias)与输出变量** 依然保留为浮点精度。
这一规则能够总体上模拟推理框架的图融合情况，简化量化过程并提升计算效率，并可获得较为准确的模型量化性能。

对于 Transformer Based 模型，OnnxQuant 将在 Layernormalization 层的输入变量上添加量化-反量化操作，其权重不参与量化。

### 4.2 量化细则
OnnxQuant 关注以下三类量化细则下的模型性能：
| INT8 PERCHANNEL | INT8 PERTENSOR POWER-OF-2 | GRAPHCORE FP8 |
|:---|:---|:---|
| 权重使用 PERCHANNEL 量化 | 权重使用 PERTENSOR 量化，Scale 附加 POWER-OF-2 限制 | 权重使用 PERTENSOR FP8 量化 |
| 激活值使用 PERTENSOR 量化 | 激活值使用 PERTENSOR 量化，Scale 附加 POWER-OF-2 限制 |  激活值使用 PERTENSOR FP8 量化 | 
| 量化范围为[-128, 127] | 量化范围为[-128, 127] |  量化范围为[-448.0, 448.0] | 

## 5. OnnxQuant Baseline

在前文中，我们已经介绍了不同模型的测试标准，OnnxQuant 将以此为标准测试量化模型在测试数据集上的分类 / 检测 / 超分辨率 / 文字识别 / 实例分割的性能。

除此以外，OnnxQuant 额外引入三项通用指标对模型量化结果进行评估：

  * Average Quantization Error(AQE): 模型平均量化误差

  * Max Quantization Error(MQE): 模型最大量化误差

  * Output Quantization Error(OQE): 模型输出最大量化误差

OnnxQuant 使用相对误差评估模型量化误差，对于量化网络中的任意一个卷积层、全连接层、Layernormalization 层而言，OnnxQuant 取该层的输出结果 A 与浮点网络对应层的输出结果 B 进行对比。
相对误差等于 || A - B || 除以 || B ||，其中 || B || 表示向量 B 的 F-范数。

  * 模型平均量化误差(AQE)：模型中所有层的量化误差均值

  * 模型最大量化误差(MQE)：模型中所有层的量化误差最大值

  * 模型输出最大量化误差(OQE)：模型中所有输出层的量化误差最大值

模型测试结果请参考下图：

![Baseline](https://user-images.githubusercontent.com/43309460/227209943-88b13ecd-ffd3-4725-a493-e9b91a7dfe4b.png)

## 6 Contribution

您可以在 github 上提交 issue 来反馈运行过程中遇到的问题。

您可以通过微信群，qq群，github等渠道联系我们提交新的模型与测试数据加入 OnnxQuant 测试集合

## Licence

This project is distributed under the Apache License, Version 2.0.
