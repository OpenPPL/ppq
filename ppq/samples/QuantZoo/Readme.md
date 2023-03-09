# ONNX Quantization Model Zoo (OnnxQuant)

:smile: OnnxQuant 是目前最大的模型量化数据集，它包含 ONNX 模型，数据，以及相关的测试脚本。该数据集的提出用于推动模型量化在视觉模型中的应用与量化算法的研究，具备以下特点：

1. 可移植与可复现，所有模型均由 ONNX 格式提供。
2. 包含图像分类、图像分割、图像超分辨率、图像-文字识别、目标检测、姿态检测等多个任务的模型。
3. 提供切分好的 calibration 数据和 test 数据，提供模型精度测试脚本。
4. 提供灵活的量化器用于确定模型在不同量化规则下的理论性能，并提供 FP8 量化器。

:eyes: OnnxQuant 目前处于公开测试阶段，近几个月内仍然将发生修改与变动。

## 1. 如何使用

### 1.1 下载数据集：

1. 图像分类: https://pan.baidu.com/s/1CIrQBvMkPaI-19M8IpVP8w?pwd=z5z8
2. 图像超分: https://pan.baidu.com/s/1u7ZAVNtlaMHBzDzzq-1eCw?pwd=gzsb
3. 图像分割: https://pan.baidu.com/s/1LAS8LYyklz7kgkVUuxDlLg?pwd=db6s
4. 目标检测: https://pan.baidu.com/s/1uBvK-Wm1AKVrNgvA9E4lhA?pwd=9n06
5. 姿态检测: 
6. 图像-文字识别: https://pan.baidu.com/s/1GyYvLbhibLL6kPIA1J0X7Q?pwd=vpxi
7. NLP:

### 1.2 建立工作目录：

在工作目录下建立文件夹 QuantZoo，解压上述文件到 QuantZoo 中。你将获得这样的文件夹结构

```
~/QuantZoo/Data/Cityscapes
~/QuantZoo/Data/Coco
~/QuantZoo/Data/DIV2K
~/QuantZoo/Data/IC15
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

模型：全部来自 mmedit

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
相对误差等于 || A - B || / || B ||，其中 || B || 表示向量 B 的 F-范数。

  * 模型平均量化误差(AQE)：模型中所有层的量化误差均值

  * 模型最大量化误差(MQE)：模型中所有层的量化误差最大值

  * 模型输出最大量化误差(OQE)：模型中所有输出层的量化误差最大值

图例：❗: 量化精度很差的模型；💔: 很差的单一指标

### INT8 PERCHANNEL

| Classification     | Float Accuracy | Quant Accuracy | AQE    | MQE     | OQE     |
| ------------------ | ----------- | -------------- | ------ | ------- | ------- |
| ❗efficientnet_v1_b0 | 76.90%      | 66.19%         | 20.81%💔 | 60.34%💔 | 70.51%💔  |
| efficientnet_v1_b1 | 76.66%      | 75.64%         | 4.16%  | 20.23%💔  | 15.50%💔  |
| efficientnet_v2    | 80.29%      | 80.03%         | 6.52%  | 44.53%💔  | 41.45%💔  |
| ❗mnasnet 0.5        | 67.75%      | 64.42%         | 5.40%  | 15.51%💔  | 24.88%💔  |
| mnasnet 1.0        | 73.48%      | 72.54%         | 2.29%  | 5.51%   | 5.71%   |
| mobilenet_v2       | 71.37%      | 70.99%         | 4.75%  | 21.13%💔  | 6.02%   |
| ❗mobilenet_v3_small | 67.89%      | 2.80%          | 55.57%💔 | 123.36%💔 | 131.26%💔 |
| mobilenet_v3_large | 73.37%      | 72.48%         | 2.17%  | 7.10%   | 5.57%   |
| resnet18           | 69.65%      | 69.51%         | 0.55%  | 1.48%   | 1.17%   |
| resnet50           | 75.56%      | 75.48%         | 1.24%  | 3.60%   | 1.95%   |
| once_for_all_71    | 72.30%      | 71.75%         | 4.11%  | 33.14%💔  | 7.88%   |
| once_for_all_73    | 74.54%      | 74.38%         | 3.49%  | 32.25%💔  | 5.51%   |
| ❗vit_b_16           | 80.00%      | 77.90%         | \*     | \*      | \*      |

| Segmentation                                 | Float mIou | Quant mIou | AQE   | MQE    | OQE   |
| -------------------------------------------- | ------- | ---------- | ----- | ------ | ----- |
| stdc1_512x1024_80k_cityscapes                | 71.44%  | 71.21%     | 1.31% | 2.88%  | 0.66% |
| pspnet_r50-d8_512x1024_40k_cityscapes        | 76.48%  | 76.34%     | 1.77% | 4.77%  | 1.40% |
| pointrend_r50_512x1024_80k_cityscapes        | 75.66%  | 75.99%     | \*    | \*     | \*    |
| fpn_r50_512x1024_80k_cityscapes              | 73.86%  | 75.20%     | 1.73% | 5.62%  | 0.61% |
| icnet_r18-d8_832x832_80k_cityscapes          | 67.07%  | 66.72%     | 0.61% | 1.27%  | 0.31% |
| fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes | 68.34%  | 68.17%     | 1.26% | 2.92%  | 0.54% |
| ❗fast_scnn_lr0.12_8x4_160k_cityscapes         | 70.22%  | 69.38%     | 3.71% | 10.19%💔 | 0.75% |
| fcn_r50-d8_512x1024_40k_cityscapes           | 70.96%  | 73.67%     | 2.12% | 7.81%💔  | 2.14% |
| bisenetv2_fcn_4x4_1024x1024_160k_cityscapes  | 71.67%  | 71.91%     | 1.28% | 5.80%  | 0.92% |
| deeplabv3_r50-d8_512x1024_40k_cityscapes     | 77.90%  | 77.48%     | 1.97% | 7.95%💔  | 1.17% |

| SuperRes | Float PSNR | Quant PSNR | AQE   | MQE    | OQE   |
| -------- | ------- | ---------- | ----- | ------ | ----- |
| edsr     | 28.98   | 28.76      | 0.15% | 0.36%  | 0.06% |
| rdn      | 29.20   | 28.65      | 7.99%💔 | 19.94%💔 | 0.30% |
| srcnn    | 27.76   | 27.18      | 4.89% | 29.80%💔 | 0.13% |
| srgan    | 26.56   | 26.53      | 2.56% | 6.16%  | 0.10% |

| OCR                             | Float Accuracy | Quant Accuracy | AQE     | MQE     | OQE    |
| ------------------------------- | ----------- | -------------- | ------- | ------- | ------ |
| ❗en_PP-OCRv3_rec_infer           | 67.36%      | 40.64%         | 24.41%💔  | 64.69%💔  | 22.11%💔 |
| ❗en_number_mobile_v2.0_rec_infer | 46.22%      | 21.47%         | 81.85%💔  | 398.62%💔 | 6.48%💔  |
| ❗ch_PP-OCRv2_rec_infer           | 54.89%      | 4.91%          | 114.55%💔 | 985.66%💔 | 99.83%💔 |
| ❗ch_PP-OCRv3_rec_infer           | 62.69%      | 0.63%          | 116.20%💔 | 516.44%💔 | 72.43%💔 |
| ❗ch_ppocr_mobile_v2.0_rec_infer  | 40.15%      | 10.39%         | 97.96%💔  | 523.46%💔 | 41.99%💔 |
| ch_ppocr_server_v2.0_rec_infer  | 54.98%      | 53.92%         | 3.30%   | 14.72%💔  | 8.95%  |

| Detection     | Float mAP | Quant mAP | AQE   | MQE    | OQE   |
| ------------- | ------ | --------- | ----- | ------ | ----- |
| yolov6p5_n    | 49.80% | 47.10%    | 2.40% | 13.36%💔 | 1.46% |
| yolov6p5_t    | 52.40% | 50.60%    | 6.00% | 17.56%💔 | 3.40% |
| ❗yolov5s6_n    | 39.80% | 35.60%    | 1.20% | 2.60%  | 0.29% |
| ❗yolov5s6_s    | 47.90% | 42.40%    | 1.29% | 3.05%  | 0.20% |
| ❗yolov7p5_tiny | 46.60% | 41.50%    | 1.63% | 3.60%  | 3.03% |
| ❗yolov7p5_l    | 59.50% | 51.20%    | 1.29% | 2.54%  | 2.68% |
| ❗yolox_s       | 49.30% | 43.00%    | 4.64% | 12.61%💔 | 3.96% |
| ❗yolox_tiny    | 45.00% | 39.10%    | 1.35% | 3.12%  | 0.90% |
| ppyoloe_m     | 55.80% | 54.60%    | 2.69% | 8.10%  | 0.87% |
| ppyoloe_s     | 50.30% | 49.00%    | 1.55% | 3.97%  | 0.77% |

### INT8 PERTENSOR POWER-OF-2

| Classification     | Float Accuracy | Quant Accuracy | AQE      | MQE        | OQE     |
| ------------------ | ----------- | -------------- | -------- | ---------- | ------- |
| ❗efficientnet_v1_b0 | 76.90%      | 0.06%          | 20609%💔   | 746505%💔    | 12887%💔  |
| ❗efficientnet_v1_b1 | 76.66%      | 0.46%          | 279.77%💔  | 5344.33%💔   | 138.59%💔 |
| efficientnet_v2    | 80.29%      | 78.83%         | 10.39%   | 56.11%💔     | 24.38%💔  |
| ❗mnasnet 0.5        | 67.75%      | 0.08%          | 524.41%💔  | 5873.77%💔   | 255.34%💔 |
| ❗mnasnet 1.0        | 73.48%      | 67.37%         | 12.74%💔   | 27.71%💔     | 30.01%💔  |
| ❗mobilenet_v2       | 71.37%      | 62.30%         | 16.56%💔   | 44.13%💔     | 37.90%💔  |
| ❗mobilenet_v3_small | 67.89%      | 0.14%          | 3537.28% | 183137.50%💔 | 173.55%💔 |
| ❗mobilenet_v3_large | 73.37%      | 68.53%         | 7.96%    | 21.39%💔     | 24.24%💔  |
| resnet18           | 69.65%      | 68.45%         | 1.98%    | 5.39%      | 4.20%   |
| resnet50           | 75.56%      | 75.18%         | 3.19%    | 11.04%     | 5.06%   |
| ❗once_for_all_71    | 72.30%      | 0.24%          | 324.85%💔 | 1351.20%💔   | 248.77%💔 |
| ❗once_for_all_73    | 74.54%      | 0.48%          | 352.27%💔  | 1570.90%💔   | 218.98%💔 |

| Segmentation                                 | Float mIou | Quant mIou | AQE    | MQE    | OQE   |
| -------------------------------------------- | ------- | ---------- | ------ | ------ | ----- |
| stdc1_512x1024_80k_cityscapes                | 71.44%  | 71.36%     | 2.45%  | 5.56%  | 1.36% |
| pspnet_r50-d8_512x1024_40k_cityscapes        | 76.48%  | 76.02%     | 3.75%  | 6.76%  | 3.04% |
| pointrend_r50_512x1024_80k_cityscapes        | 75.66%  | 75.79%     | \*     | \*     | \*    |
| fpn_r50_512x1024_80k_cityscapes              | 73.86%  | 74.06%     | 5.29%  | 15.85%💔 | 2.53% |
| icnet_r18-d8_832x832_80k_cityscapes          | 67.07%  | 67.02%     | 0.97%  | 2.21%  | 0.51% |
| fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes | 68.34%  | 67.30%     | 3.23%  | 7.67%  | 4.70% |
| ❗fast_scnn_lr0.12_8x4_160k_cityscapes         | 70.22%  | 68.33%     | 10.04%💔 | 24.91%💔 | 2.03% |
| fcn_r50-d8_512x1024_40k_cityscapes           | 70.96%  | 73.10%     | 4.55%  | 15.92%💔 | 4.18% |
| bisenetv2_fcn_4x4_1024x1024_160k_cityscapes  | 71.67%  | 71.39%     | 4.21%  | 44.38%💔 | 2.76% |
| deeplabv3_r50-d8_512x1024_40k_cityscapes     | 77.90%  | 77.92%     | 4.14%  | 9.54%  | 2.74% |

| SuperRes | Float PSNR | Quant PSNR | AQE   | MQE    | OQE   |
| -------- | ------- | ---------- | ----- | ------ | ----- |
| edsr     | 28.98   | 28.83      | 4.40% | 12.12%💔 | 0.06% |
| rdn      | 29.20   | 28.83      | 3.91% | 26.16%💔 | 0.07% |
| ❗srcnn    | 27.76   | 22.21      | 6.38% | 16.31%💔 | 1.67% |
| srgan    | 26.56   | 26.25      | 9.19% | 23.23%💔 | 0.33% |

| OCR                             | Float Accuracy | Quant Accuracy | AQE        | MQE         | OQE     |
| ------------------------------- | ----------- | -------------- | ---------- | ----------- | ------- |
| ❗en_PP-OCRv3_rec_infer           | 67.36%      | 0.00%          | 123.00%💔    | 813.22%💔     | 92.18%💔  |
| ❗en_number_mobile_v2.0_rec_infer | 46.22%      | 3.42%          | 141.80%💔    | 566.63%💔     | 9.33%   |
| ❗ch_PP-OCRv2_rec_infer           | 54.89%      | 0.00%          | 211218.00%💔 | 5565940.00%💔 | 430.37%💔 |
| ❗ch_PP-OCRv3_rec_infer           | 62.69%      | 0.00%          | 430.96%💔    | 5892.44%💔    | 218.08%💔 |
| ❗ch_ppocr_mobile_v2.0_rec_infer  | 40.15%      | 0.00%          | 364.47%💔   | 5660.18%💔    | 54.53%💔  |
| ch_ppocr_server_v2.0_rec_infer  | 54.98%      | 54.41%         | 3.84%      | 13.75%💔      | 5.47%   |

| Detection     | Float mAP | Quant mAP | AQE    | MQE     | OQE    |
| ------------- | ------ | --------- | ------ | ------- | ------ |
| ❗yolov6p5_n    | 49.80% | 42.70%    | 19.20%💔 | 111.97%💔 | 9.81%  |
| ❗yolov6p5_t    | 52.40% | 20.40%    | 54.68%💔 | 153.04%💔 | 34.37%💔 |
| ❗yolov5s6_n    | 39.80% | 32.10%    | 4.56%  | 11.63%💔  | 1.16%  |
| ❗yolov5s6_s    | 47.90% | 38.10%    | 4.73%  | 9.80%   | 0.80%  |
| ❗yolov7p5_tiny | 46.60% | 37.50%    | 7.02%  | 14.71%  | 10.45%💔 |
| ❗yolov7p5_l    | 59.50% | 39.70%    | 7.72%  | 20.07%💔  | 12.56%💔 |
| ❗yolox_s       | 49.30% | 37.40%    | 15.44%💔 | 35.91%💔  | 12.00%💔 |
| ❗yolox_tiny    | 45.00% | 34.70%    | 4.27%  | 11.52%  | 3.66%  |
| ❗ppyoloe_m     | 55.80% | 51.20%    | 14.31%💔 | 35.48%💔  | 4.74%  |
| ❗ppyoloe_s     | 50.30% | 45.50%    | 14.69%💔 | 36.28%💔 | 6.52%  |

## 6 Contribution

您可以在 github 上提交 issue 来反馈运行过程中遇到的问题。

您可以通过微信群，qq群，github等渠道联系我们提交新的模型与测试数据加入 OnnxQuant 测试集合

## Licence

This project is distributed under the Apache License, Version 2.0.
