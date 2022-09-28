# Detection Benchmark
本仓库是一个验证ppq在**多硬件平台**上量化**目标检测和语义分割模型**的性能Benchmark，测试集为MS COCO 2017, 数据校准集来自COCO训练集采样。

对于目标检测模型，通常有两种部署方式：

- 模型计算图（onnx模型）`包含后处理`操作，直接输出少量的原图尺度的检测框。
- 模型计算图（onnx模型）`不包含后处理`操作，只输出检测头预测结果，包含数百万未解码预测框。

本项目以经典的RetinaNet为例，详细研究了两类模型在多平台量化部署的效果以及存在问题。

>**RetinaNet**：带后处理。输入为固定尺寸的图片，输出为两个tensor：
>
>检测框（100,5） 代表100个检测框，每个检测框包含左上和右下坐标以及预测置信度 $[x_1,y_1,x_2,y_2,score]$
>
>类别（100,1）代表100个检测框的类别索引，对于coco数据集来说，每个类别是一个在 $[1,80]$范围内的整数。

>**RetinaNet-wo**：不带后处理。输入为固定尺寸的图片，输出为5个检测头共10个tensor：
>
>每个多尺度检测头输出两个tensor，分别为检测框 $[36,h,w]$和置信度 $[720,h,w]$ 。其中h,w取决于输入尺寸和检测头scale。
>
>对于输出的tensor，需要利用预设anchor（网络超参），模型scale因子，NMS操作等进行解码和去冗余。

我们也测试了实例分割模型在多平台上量化部署的效果。这里使用的能够同时完成目标检测和实例分割的模型MaskRCNN，他在目标检测输出的基础上，对每个bbox协同输出了对应的实例掩码，能够完成实例分割任务。

> **MaskRCNN（dynamic）:** 输出分割掩码，能够支持 $(1,3,height,width)$ 任意尺寸的输入图像。

每个模型测试四个精度：

> FP32 Onnxruntime(全精度)，PPQ INT8(模拟量化精度), QDQ onnxruntime INT8(部署参考精度), TargetPlatform INT8(目标硬件推理精度) 

## 使用方法
首先保证你已经下载MS COCO数据集，下载链接[MS COCO2017](https://cocodataset.org/#download)  

你可以使用fiftyone库方便地下载MSCOCO数据集。

```python
import fiftyone
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",   # 只下载验证集
    label_types=["detections", "segmentations"],
    dataset_dir = "/home/user/"  #数据保存路径
)
```

本项目使用的所有目标检测与分割模型，主要来源于[mmdeploy](https://github.com/open-mmlab/mmdeploy)以及[onnx model zoo](https://github.com/onnx/models)，其中mmdeploy只能导出带后处理的端到端目标检测模型，而onnx model zoo有不带后处理的目标检测模型。所有使用的模型也可以[网盘获取](https://pan.baidu.com/s/1D1xZuQN6bR221u_NkmHPpA )(提取码：s655)。你需要将下载好的模型放入配置文件的`$FP32_BASE_PATH `路径中。

本仓库所有需要修改的内容都只在*cfg.py*这个配置文件中.

```python
# must be modified in cfg.py
BASE_PATH = "/home/geng/tinyml/ppq/benchmark/detection"
FP32_BASE_PATH = BASE_PATH + "/FP32_model"
ANN_PATH = "/home/geng/fiftyone/coco-2017/validation/labels.json"  # 用来读取 validation dataset
DATA_ROOT = '/home/geng/fiftyone/coco-2017/validation/data/'

# can be modified if necessary
# 你需要给出onnx模型的输入shape，方便数据做预处理
# 如果你需要添加自己的自定义模型，请在utils.decoder中添加对应的后处理代码
MODELS = {
'Retinanet':
    {
    'INPUT_SHAPE':(1,3,800,1216)
    }
}

# 量化配置信息
# 当用户进行量化误差分析，发现模型精度下降严重时，可以采用优化算法对模型进行重训。
DO_QUANTIZATION = ["PLATFORM","ORT"]  #是否进行量化，以及要导出的模型。如果为空则不进行量化。
OPTIMIZER =  False #开启优化算法
ERROR_ANALYSE = False  #开启误差分析

# 精度评估
# 以下的四个精度将会出现在最终的测试报告中，你可以根据需求选择是否测试。如果为空则不进行任何推理操作。
# PF32 全精度,PPQ模拟量化精度, ORT测试精度, PLATFORM平台部署精度
EVAL_LIST = ["FP32","PPQ","ORT","PLATFORM"] #测试全部精度

# 量化方案
#用户可以针对不同的部署平台，设置量化平台QuantPlatform，量化方案QuanSetting，导出平台格式ExportPlatform，量化后模型输出路径OutputPath，以及量化调度器Dispatcher。
PLATFORM_CONFIGS = {
    "OpenVino":{
        "QuantPlatform": TargetPlatform.OPENVINO_INT8,
        "QuanSetting": QuantizationSettingFactory.default_setting(),
        "ExportPlatform": TargetPlatform.OPENVINO_INT8,
        "OutputPath":f"{BASE_PATH}/OpenVino_output",
        "Dispatcher":"conservative"
    }
}
```
其他参数根据需求修改后，获取测试结果
```python3
python benchmark.py
```

## 实用工具

本项目提供了一些独立的工具，可以单独使用。

`COCODataset` 实现了数据annotation加载、数据预处理、模型元信息保存以及模型精度评估。使用方法如下：

- 构建数据集

  只需要annotations和data路径，以及要预处理的图像尺寸，就能轻松构建目标检测与分割数据集。在构建数据集的过程中，同时完成数据的预处理操作，进行图像Resize、Normalize、Pad等操作。

  ```python
  from dataset import build_dataset
  dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size,keep_ratio=True) #构建数据集
  ```

- 保存图片元信息

  dataset每个item都包含着该图片的各项元信息，包括图像文件名、原图分辨率，处理后图像分辨率、图像放缩比、正则化参数等。这些元信息对于目标检测和实例分割任务的output decode十分重要。

- 结果格式化

  对于coco数据集来说，要想进行精度的评估，必须要将模型的输出结果转为标准的cocoresult.json格式，本项目实现了这个方法，将任何模型在任何平台上的后处理输出，都统一转为模型无关的results.json格式，并可以持久化保存到本地。

  ```python
  dataset.results2json(results=results,outfile_prefix="retinanet-wo-int8") #格式化results并保存到本地
  ```

- 模型精度评估

  dataset包含数据集的所有annotations信息，可以对results进行精度的评估。本文实现了两种评估精度的接口，一种是将模型输出并后处理的结构直接传入dataset进行评估；另一种是使用持久化到磁盘上的results.json文件进行评估。同时该接口可以同时实现目标检测`bbox map`和实例分割`segm map`精度的评估。

  ```python
  dataset.evaluate(results=results,metric="bbox") # 直接使用输出结果评估目标检测精度
  dataset.evaluate(results_json_path="retinanet-wo-int8.bbox.json",metric="bbox") # 使用results.json文件评估目标检测精度
  
  dataset.evaluate(results=results,metric="segm") # 直接使用输出结果评估实例分割精度
  dataset.evaluate(results_json_path="retinanet-wo-int8.bbox.json",metric="segm") # 使用results.json文件评估实例分割精度
  ```

`inference` 针对四种平台实现了模型无关的推理代码，即对于任意的模型，只要部署平台相同，都可使用相同的推理代码接口。

```python
from inference import ppq_inference,openvino_inference,trt_inference,onnxruntime_inference
outputs = trt_inference(dataloader,trt_model_path)
```

`utils.decoder`实现了三种模型的解码与后处理方式，能够将任意平台推理出的原生输出，处理为标准的coco result.

```python
from detection.utils import post_process
results = post_process(model_type,outputs,class_num)
```

使用以上工具，就可以自定义地实现模型在多平台上的推理与评估，可以见`test_demo.py`了解整个流程。

