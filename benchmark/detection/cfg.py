from ppq import *
import os


# 检测模型相关配置
MODELS = {
'Retinanet':
    {
    'INPUT_SHAPE':(1,3,800,1216)
    }

}

# 通用的配置信息
CALIBRATION_NUM = 512
CALIBRATION_BATCH_SIZE = 1  # 候选batchsizeduplicateeva
DEVICE = "cuda"

# 以下的四个精度将会出现在最终的测试报告中，你可以根据需求选择是否测试
# PF32 全精度,PPQ模拟量化精度, ORT测试精度, PLATFORM平台部署精度
#  ECVAL_LIST = ["FP32","PPQ","ORT","PLATFORM"] #测试全部精度
EVAL_LIST = []

CLASS_NUM = 80

# 一些重要的目录
BASE_PATH = "/home/geng/tinyml/ppq/benchmark/detection"
FP32_BASE_PATH = BASE_PATH + "/FP32_model"
ANN_PATH = "/home/geng/fiftyone/coco-2017/validation/labels.json"  # 用来读取 validation dataset
DATA_ROOT = '/home/geng/fiftyone/coco-2017/validation/data/'

# 不同平台的量化策略
PLATFORM_CONFIGS = {
    "OpenVino":{
        "QuantPlatform": TargetPlatform.OPENVINO_INT8,
        "QuanSetting": QuantizationSettingFactory.default_setting(),
        "ExportPlatform": TargetPlatform.OPENVINO_INT8,
        "OutputPath":f"{BASE_PATH}/OpenVino_output" 
    },
    "TRT":{
        "QuantPlatform": TargetPlatform.TRT_INT8,
        "QuanSetting": QuantizationSettingFactory.trt_setting(),
       "ExportPlatform": TargetPlatform.TRT_INT8,
        "OutputPath":f"{BASE_PATH}/TRT_output" 
    },
    "Snpe":{
        "QuantPlatform": TargetPlatform.SNPE_INT8,
        "QuanSetting": QuantizationSettingFactory.dsp_setting(),
        "ExportPlatform": TargetPlatform.SNPE_INT8,
        "OutputPath":f"{BASE_PATH}/Snpe_output"  
    },
    "Ncnn":{
        "QuantPlatform": TargetPlatform.NCNN_INT8,
        "QuanSetting": QuantizationSettingFactory.ncnn_setting(),
        "ExportPlatform": TargetPlatform.NCNN_INT8,
        "OutputPath":f"{BASE_PATH}/Ncnn_output"  
    }
}  









