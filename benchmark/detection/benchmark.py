# 这个文件会对多模型按照多平台策略进行量化操作，并将量化的模型导出到预设的目录
from utils import *
from ppq import *
from ppq.api import *
from dataset import build_dataset
from inference import ppq_inference
from utils import post_process
from utils import (onnxruntime_inference2json,openvino_inference2json,
                    trt_inference2json,ppq_inference2json)


import os
import cfg
import pandas as pd

from mmcv.parallel import collate
from torch.utils.data import DataLoader


report = []
with ENABLE_CUDA_KERNEL():
    for model_name in cfg.MODELS.keys():
        input_size = cfg.MODELS[model_name]["INPUT_SHAPE"]
        dataset =  build_dataset(ann_file=cfg.ANN_PATH,data_root=cfg.DATA_ROOT,
                input_size=input_size)
        
        # 获取校准数据集
        # calib_dataloader = DataLoader(dataset,batch_size=cfg.CALIBRATION_BATCH_SIZE,collate_fn=collate)
        calib_dataloader = [dataset[i]["img"][0].unsqueeze(0) for i in range(cfg.CALIBRATION_NUM)]

        for platform,config in cfg.PLATFORM_CONFIGS.items():
            config["QuanSetting"].dispatcher =  config["Dispatcher"]#修改调度策略

            if cfg.OPTIMIZER:
                print("Open LSQ Optimization...")
                config["QuanSetting"].lsq_optimization = True                                      # 启动网络再训练过程，降低量化误差
                config["QuanSetting"].lsq_optimization_setting.steps = 512                         # 再训练步数，影响训练时间，500 步大概几分钟
                config["QuanSetting"].lsq_optimization_setting.collecting_device = cfg.DEVICE    # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'
            
            if not os.path.exists(config["OutputPath"]):
                os.makedirs(config["OutputPath"])
            print(f'---------------------- PPQ Quantization Test Running with {model_name} on {platform}----------------------')
            fp32_model_path = f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx'
            path_prefix = os.path.join(config["OutputPath"], model_name)

            # 量化onnx模型
            if len(cfg.DO_QUANTIZATION) > 0:
                ppq_quant_ir = quantize_onnx_model(
                    onnx_import_file=fp32_model_path, calib_dataloader=calib_dataloader, calib_steps=cfg.CALIBRATION_NUM // cfg.CALIBRATION_BATCH_SIZE, 
                    setting=config["QuanSetting"],input_shape=input_size, collate_fn=lambda x: x.to(cfg.DEVICE), 
                    platform=config["QuantPlatform"],device=cfg.DEVICE, do_quantize=True)

                if cfg.ERROR_ANALYSE: 
                    layerwise_error_analyse(graph=ppq_quant_ir, running_device=cfg.DEVICE,
                                    interested_outputs=None,dataloader=calib_dataloader, collate_fn=lambda x: x.to(cfg.DEVICE))
                
                # 进行模型推理，并将结果统一保存为coco result json格式
                if "PPQ" in cfg.EVAL_LIST:
                    print("inference ppq model")
                    ppq_inference2json(dataset=dataset,model_name=model_name,ppq_ir=ppq_quant_ir,
                        batch_size=input_size[0],class_num=cfg.CLASS_NUM,
                        outfile_prefix=f'{path_prefix}-PPQ-INT8',
                        device=cfg.DEVICE)

                # 导出ort模型
                if "ORT" in cfg.DO_QUANTIZATION:
                    export_ppq_graph(
                        graph = ppq_quant_ir,
                        # copy_graph=True,
                        platform=TargetPlatform.ONNXRUNTIME,
                        graph_save_to=f'{path_prefix}-ORT-INT8.onnx')
                        
                if "PLATFORM" in cfg.DO_QUANTIZATION:
                     # 导出平台模型 
                    export_ppq_graph(
                        graph = ppq_quant_ir,
                        platform=config["ExportPlatform"],
                        graph_save_to=f'{path_prefix}-{platform}-INT8.onnx')

            
            if "FP32" in cfg.EVAL_LIST:
                print("inference fp32 model")
                onnxruntime_inference2json(dataset=dataset,model_name=model_name,model_path=fp32_model_path,
                    batch_size=input_size[0],class_num=cfg.CLASS_NUM,
                    outfile_prefix=f'{fp32_model_path[:-5]}',
                    device=cfg.DEVICE)
            if "ORT" in cfg.EVAL_LIST:
                print("inference ort model")
                onnxruntime_inference2json(dataset=dataset,model_name=model_name,model_path=f'{path_prefix}-ORT-INT8.onnx',
                    batch_size=input_size[0],class_num=cfg.CLASS_NUM,
                    outfile_prefix=f'{path_prefix}-ORT-INT8',
                    device=cfg.DEVICE)
            if "PLATFORM" in cfg.EVAL_LIST:
                if platform  == "OpenVino":
                    print("inference opevino model")
                    openvino_inference2json(dataset=dataset,model_name=model_name,model_path=f'{path_prefix}-{platform}-INT8.onnx',
                        batch_size=input_size[0],class_num=cfg.CLASS_NUM,
                        outfile_prefix=f'{path_prefix}-{platform}-INT8',
                        device=cfg.DEVICE)
                elif platform  == "TRT":
                    print("inference trt model")
                    trt_inference2json(dataset=dataset,model_name=model_name,model_path=f'{path_prefix}-{platform}-INT8.engine',
                        batch_size=input_size[0],class_num=cfg.CLASS_NUM,
                        outfile_prefix=f'{path_prefix}-{platform}-INT8',
                        device=cfg.DEVICE)
                else:
                    raise ValueError(f"not implement inference plarform of {platform}")
            

            result_json_paths = [
                # f'{fp32_model_path[:-5]}.bbox.json',  #fp32 result json
                f'{path_prefix}-PPQ-INT8.bbox.json',
                # f'{path_prefix}-ORT-INT8.bbox.json',
                # f'{path_prefix}-{platform}-INT8.bbox.json'
            ]
            maps = []
            for result_json in result_json_paths:
                if os.path.exists(result_json):
                    print(f"eval result json path :{result_json}")
                    maps.append(dict(dataset.evaluate(results_json_path=result_json))["bbox_mAP"])
                else:
                    maps.append(None)
            fp32_map,ppq_map,ort_map,platform_map = maps
            report.append([model_name,platform,fp32_map,ppq_map,ort_map,platform_map])

report = pd.DataFrame(report,columns=["model","paltform","fp32_map","ppq_map","ort_map","platform_map"])
print("-------------------测试报告如下---------------------")
print(report)
report.to_csv(f"{cfg.BASE_PATH}/report.csv",index=False)