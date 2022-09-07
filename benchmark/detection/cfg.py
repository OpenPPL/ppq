from ppq import *
import os


# 检测模型相关配置
MODELS = ['Retinanet_FPN']

# 通用的配置信息
BATCHSIZE = 1     # 因为onnx模型输入固定，因此设置全局的batchsize
CALIBRATION_NUM = 5120
INPUT_SHAPE = (BATCHSIZE,3,768,1344)
DEVICE = "cuda"

# 是否获取测试的fp32 onnx模型,为了节省时间只需获取一次即可
# 但每次更改batchsize后必须将其设为True
GET_FP32_MODEL = False  

# 以下的四个精度将会出现在最终的测试报告中，你可以根据需求选择是否测试
GET_FP32_ACC = False  # PF32 全精度
GET_PPQ_ACC = False  #  PPQ模拟量化精度
GET_ORT_ACC = False   #qdq ort测试精度。很耗时间，目前存在bug不能和平台部署精度同时测试。 
GET_PLATFORM_ACC = False  #平台部署精度。该脚本只支持opevino和tensorRT

# 一些重要的目录
BASE_PATH = "/home/geng/tinyml/ppq/benchmark/detection"
FP32_BASE_PATH = BASE_PATH + "/FP32_model"
VALIDATION_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Valid'   # 用来读取 validation dataset
TRAIN_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型


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









