# Classification Benchmark
本仓库是一个验证ppq在**多硬件平台**上量化**分类模型**的性能Benchmark，测试集为ImageNet(ILSVRC2019) val, 数据校准集来自ImageNet训练集采样。

对5种分类模型：
> Resnet-18, ResNeXt101_64x4d, RegNet_X_1_6GF, ShuffleNetV2_x1_0, MobileNetV2

在四个平台上:
> TensorRT(gpu), OpenVino(x86 cpu), Snpe(dsp&npu), Ncnn(arm cpu)

测试四个精度：
> FP32 Onnxruntime(全精度)，PPQ INT8(模拟量化精度), QDQ onnxruntime INT8(部署参考精度), TargetPlatform INT8(目标硬件推理精度) 

## 使用方法
首先保证你已经下载ImageNet数据集，下载链接[ILSVRC2019](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)  
本仓库所有需要修改的内容都只在*cfg.py*这个配置文件中.

```python
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
|resnet18|OpenVino|69.764|69.466|69.480|**69.438**|
|resnet18|TRT|69.764|69.578|69.550|69.484|
|resnet18|Snpe|69.764|69.278|69.266|-|
|resnet18|Ncnn|69.764|69.132|69.064|-|
|mobilenetV2|OpenVino|72.017|71.317|71.383|**71.584**|
|mobilenetV2|TRT|72.017|71.413|71.441|71.367|
|mobilenetV2|Snpe|72.017|70.102|70.072|-|
|mobilenetV2|Ncnn|72.017|71.671|71.657|-|
|ResNeXt101_64x4d|OpenVino|82.774|81.978|81.834|**81.926**|
|ResNeXt101_64x4d|TRT|82.985|81.920|81.854|81.882|
|ResNeXt101_64x4d|Snpe|82.985|81.492|81.468|-|
|ResNeXt101_64x4d|Ncnn|82.985|82.911|82.885|-|
|RegNet_X_1_6GF|OpenVino|79.341|78.443|78.593|**78.407**|
|RegNet_X_1_6GF|TRT|79.341|78.531|78.737|78.539|
|RegNet_X_1_6GF|Snpe|79.341|76.616|76.735|-|
|RegNet_X_1_6GF|Ncnn|79.341|79.243|79.199|-|
|ShuffleNetV2_x1_0|OpenVino|69.370|68.806|68.802|68.437|
|ShuffleNetV2_x1_0|TRT|69.370|68.718|68.752|68.706|
|ShuffleNetV2_x1_0|Snpe|69.370|68.564|68.520|-|
|ShuffleNetV2_x1_0|Ncnn|69.370|68.748|68.768|-|

*以上分类模型来源于torchvision.models,转为onnx模型推理测试。测试数据集为ImageNet(ILSVRC2019) val，所有图片都被预处理为 224\*224​ 的尺寸，评价指标为top-1 ACC。