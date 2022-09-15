# Detection Benchmark
本仓库是一个验证ppq在**多硬件平台**上量化**目标检测模型**的性能Benchmark，测试集为MS COCO, 数据校准集来自COCO训练集采样。

对检测和分割模型：
> Retinanet-end2end
> Retinanet-wo
> MaskRCNN
> FCN

在四个平台上:
> TensorRT(gpu), OpenVino(x86 cpu), Snpe(dsp&npu), Ncnn(arm cpu)

测试四个精度：
> FP32 Onnxruntime(全精度)，PPQ INT8(模拟量化精度), QDQ onnxruntime INT8(部署参考精度), TargetPlatform INT8(目标硬件推理精度) 

## 使用方法
首先保证你已经下载MS COCO数据集，下载链接[ILSVRC2019](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)  
本仓库所有需要修改的内容都只在*cfg.py*这个配置文件中.
```python3
# must be modified in cfg.py
BASE_PATH = "/home/geng/tinyml/ppq/benchmark/classification"  #项目目录
VALIDATION_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Valid'   # ImageNet验证集目录
TRAIN_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Train' #ImageNet训练集目录   
```
其他参数根据需求修改后，获取测试结果
```python3
python benchmark.py
```

## 测试结果
|model|TargetPlatform|FP32 ORT|PPQ INT8|QDQ ORT INT8|RealPlatform INT8|
|----|----|----|----|----|----|
|Retinanet-end2end|OpenVino|34.1|19.6|19.5|19.2|
|Retinanet-end2end|TRT|34.1|33.8|33.8|-|
|Retinanet-end2end|Snpe|34.1|29.5|29.6|-|
|Retinanet-end2end|Ncnn|34.1|33.7|33.7|-|
|Retinanet-wo|OpenVino|31.5|21.9|21.8|-|
|Retinanet-wo|TRT|31.5|31.2|31.2|-|
|Retinanet-wo|Snpe|31.5|22.3|22.3|-|
|Retinanet-wo|Ncnn|31.5|31.1|31.1|-|



- [x] coco数据集加载
- [x] 数据预处理
- [x] 模型推理
- [x] 模型后处理
- [x] map评估

目前存在的问题：
- [x] 