from torchvision.models import resnet18,mobilenet_v2,resnext101_64x4d,vit_b_16,shufflenet_v2_x1_0
from torchvision.models import ResNet18_Weights,MobileNet_V2_Weights,ResNeXt101_64X4D_Weights,ViT_B_16_Weights,ShuffleNet_V2_X1_0_Weights
from ppq import *
import os


# 分类模型相关配置
MODELS = {
    # 'ResNet18':  (resnet18, ResNet18_Weights.DEFAULT),
    # 'MobileNetV2': (mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
    # "ResNeXt101_64x4d": (resnext101_64x4d, ResNeXt101_64X4D_Weights.DEFAULT),
    # "Vit_B_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),   
    "ShuffleNetV2_x1_0": (shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.DEFAULT)
    }

# 是否获取测试的onnx模型，只用获取一次即可
GET_FP32_MODEL = False

# 想要测试的精度,要出现在最终的report中。其中ORT测试很占时间
GET_FP32_ACC = False
GET_PPQ_ACC = False
GET_ORT_ACC = False   #目前ORT和PLATFORM两者只能开一个，需要后续志哥修复copy的bug
GET_PLATFORM_ACC = True 

# 一些重要的目录
BASE_PATH = "/home/geng/tinyml/ppq/benchmark/classification"
FP32_BASE_PATH = BASE_PATH + "/FP32_model"
VALIDATION_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Valid'   # 用来读取 validation dataset
TRAIN_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型


# 通用的配置信息
BATCHSIZE = 32     # 因为onnx模型输入固定，因此设置全局的batchsize
CALIBRATION_NUM = 5120
INPUT_SHAPE = (BATCHSIZE,3,224,224)
DEVICE = "cuda"

# 不同平台的量化策略
PLATFORM_CONFIGS = {
    "OpenVino":{
        "TargetPlatform": TargetPlatform.OPENVINO_INT8,
        "QuanSetting": QuantizationSettingFactory.default_setting(),
        "OutputPath":f"{BASE_PATH}/Openvino_output" 
    },
    "TRT":{
        "TargetPlatform": TargetPlatform.TRT_INT8,
        "QuanSetting": QuantizationSettingFactory.trt_setting(),
        "OutputPath":f"{BASE_PATH}/TRT_output" 
    },
    "Snpe":{
        "TargetPlatform": TargetPlatform.SNPE_INT8,
        "QuanSetting": QuantizationSettingFactory.dsp_setting(),
        "OutputPath":f"{BASE_PATH}/Snpe_output"  
    },
    "Ncnn":{
        "TargetPlatform": TargetPlatform.NCNN_INT8,
        "QuanSetting": QuantizationSettingFactory.ncnn_setting(),
        "OutputPath":f"{BASE_PATH}/Ncnn_output"  
    }
}  









