# 这个文件会对多模型按照多平台策略进行量化操作，并将量化的模型导出到预设的目录
import torchvision
from benchmark.classification.utils import *
from ppq import *
from ppq.api import *
from Utilities.Imagenet import load_imagenet_from_directory
import os
import cfg

acc_all = {}
for model_name,_ in cfg.MODELS:
    acc_model = {}
    for platform,config in cfg.PLATFORM_CONFIGS:

        if not os.path.exists(config["OutputPath"]):
            os.makedirs(config["OutputPath"])

        with ENABLE_CUDA_KERNEL():

            print(f'---------------------- PPQ Quantization Test Running with {model_name} ----------------------')
            model_path = f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx'
            
            # 获取校准数据
            dataloader = load_imagenet_from_directory(
                directory=cfg.TRAIN_DIR, batchsize=cfg.BATCHSIZE,
                shuffle=False, subset=5120, require_label=False,
                num_of_workers=8)

            # 量化onnx模型
            ppq_quant_ir = quantize_onnx_model(
                onnx_import_file=model_path, calib_dataloader=dataloader, calib_steps=5120 // cfg.BATCHSIZE, 
                setting=config["QuanSetting"],input_shape=cfg.INPUT_SHAPE, collate_fn=lambda x: x.to(cfg.DEVICE), 
                platform=config["TargetPlatform"], do_quantize=True)
                

            # 导出ORT模型
            export_ppq_graph(
                graph=ppq_quant_ir,
                copy_graph=True, 
                platform=TargetPlatform.ONNXRUNTIME,
                graph_save_to=f'{os.path.join(config["OutputPath"], model_name)}-ORT-INT8.onnx')
            
            # 导出平台模型
            export_ppq_graph(
                graph=ppq_quant_ir, 
                copy_graph=True,
                platform=config["TargetPlatform"],
                graph_save_to=f'{os.path.join(config["OutputPath"], model_name)}-{platform}-INT8.onnx')

            # 评估FP32准确度
            fp32_acc,_ = get_fp32_accuracy(model_name)

            # 评估PPQ模拟量化准确度
            ppq_acc,_ = get_ppq_accuracy()

            # 评估ORT模型准确度

            # 评估目标平台部署准确度
    

