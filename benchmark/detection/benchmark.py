# 这个文件会对多模型按照多平台策略进行量化操作，并将量化的模型导出到预设的目录
from utils import *
from ppq import *
from ppq.api import *
from Utilities.Imagenet import load_imagenet_from_directory
import os
import cfg
import pandas as pd

report = []

# 获取校准数据
dataloader = load_imagenet_from_directory(
    directory=cfg.TRAIN_DIR, batchsize=cfg.BATCHSIZE,
    shuffle=False, subset=cfg.CALIBRATION_NUM, require_label=False,
    num_of_workers=16)

with ENABLE_CUDA_KERNEL():
    for model_name in cfg.MODELS.keys():

        fp32_map = None
        # 评估FP32准确度
        if cfg.GET_FP32_ACC:
            fp32_map,_ = get_fp32_report(model_name)
            pass

        for platform,config in cfg.PLATFORM_CONFIGS.items():

            if not os.path.exists(config["OutputPath"]):
                os.makedirs(config["OutputPath"])

            ppq_map,ort_map,platform_map = None,None,None

            print(f'---------------------- PPQ Quantization Test Running with {model_name} on {platform}----------------------')
            model_path = f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx'
            
            # 量化onnx模型
            ppq_quant_ir = quantize_onnx_model(
                onnx_import_file=model_path, calib_dataloader=dataloader, calib_steps=cfg.CALIBRATION_NUM // cfg.BATCHSIZE, 
                setting=config["QuanSetting"],input_shape=cfg.INPUT_SHAPE, collate_fn=lambda x: x.to(cfg.DEVICE), 
                platform=config["QuantPlatform"], do_quantize=True)
                
            # 评估PPQ模拟量化准确度
            if cfg.GET_PPQ_ACC:
                ppq_map,_ = get_ppq_report(model_name,platform,ppq_quant_ir)

            # 评估ORT模型准确度
            if cfg.GET_ORT_ACC:
                # 导出ORT模型
                export_ppq_graph(
                    graph = ppq_quant_ir,
                    platform=TargetPlatform.ONNXRUNTIME,
                    graph_save_to=f'{os.path.join(config["OutputPath"], model_name)}-ORT-INT8.onnx')

                ort_map,_ = get_ort_report(model_name,platform,config["OutputPath"])


            # 评估目标平台部署准确度
            if cfg.GET_PLATFORM_ACC:
                # 导出平台模型,暂时无法和ORT一起导出
                export_ppq_graph(
                    graph = ppq_quant_ir,
                    platform=config["ExportPlatform"],
                    graph_save_to=f'{os.path.join(config["OutputPath"], model_name)}-{platform}-INT8.onnx')

                if platform in {"OpenVino","TRT"}:
                    platform_map,_  = get_platform_report(model_name,platform,config["OutputPath"])
                else:
                    platform_map = None


            report.append([model_name,platform,fp32_map,ppq_map,ort_map,platform_map])

report = pd.DataFrame(report,columns=["model","paltform","fp32_map","ppq_map","ort_map","platform_map"])
print("-------------------测试报告如下---------------------")
print(report)
report.to_csv(f"{cfg.BASE_PATH}/report.csv",index=False)