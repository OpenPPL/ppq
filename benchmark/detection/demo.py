from ppq import *
from ppq.api import *
import torch
import cfg

model_path = "/home/geng/tinyml/ppq/benchmark/detection/FP32_model/Retinanet-FP32.onnx"
# model_path = "/home/geng/tinyml/ppq/benchmark/detection/FP32_model/end2end-12.onnx"

from dataset import build_dataset
import torch
import cfg

model_name = "Retinanet"
input_size = cfg.MODELS[model_name]["INPUT_SHAPE"]
# 获取校准数据集
_,calib_dataloader =  build_dataset(ann_file=cfg.ANN_PATH,data_root=cfg.DATA_ROOT,
        input_size=input_size,batch_size=cfg.CALIBRATION_BATCH)

# calib_dataloader = [torch.rand(size=input_size) for _ in range(512)]


config = cfg.PLATFORM_CONFIGS["TRT"]
config["QuanSetting"].dispatcher = "conservative"

ppq_quant_ir = quantize_onnx_model(
    onnx_import_file=model_path, calib_dataloader=calib_dataloader, calib_steps=cfg.CALIBRATION_NUM // cfg.CALIBRATION_BATCH, 
    setting=config["QuanSetting"],input_shape=input_size, collate_fn=lambda x: x["img"][0].to(cfg.DEVICE), 
    platform=config["QuantPlatform"], do_quantize=True)