# 这个文件会对多模型按照多平台策略进行量化操作，并将量化的模型导出到预设的目录
from utils import *
from ppq import *
from ppq.api import *
import cfg
from dataset import build_dataset


import os
import pandas as pd
import random

report = []
random.seed(0)
with ENABLE_CUDA_KERNEL():
    for model_name in cfg.MODELS.keys():
        input_size = cfg.MODELS[model_name]["INPUT_SHAPE"]
        dataset =  build_dataset(ann_file=cfg.ANN_PATH,data_root=cfg.DATA_ROOT,
                input_size=input_size)
        
        calib_dataloader = [dataset[i]["img"][0].unsqueeze(0) for i in random.sample(range(len(dataset)),cfg.CALIBRATION_NUM)]

        for platform,config in cfg.PLATFORM_CONFIGS.items():

            setting = config["QuanSetting"]
            setting.dispatcher =  config["Dispatcher"] #修改调度策略

            if cfg.OPTIMIZER:
                print("Open LSQ Optimization...")
                setting.lsq_optimization = True                                      # 启动网络再训练过程，降低量化误差
                setting.lsq_optimization_setting.steps = cfg.CALIBRATION_NUM          # 再训练步数，影响训练时间，500 步大概几分钟
                setting.lsq_optimization_setting.collecting_device = "cpu"   # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'
            
            if not os.path.exists(config["OutputPath"]):
                os.makedirs(config["OutputPath"])
                
            print(f'---------------------- PPQ Quantization Test Running with {model_name} on {platform}----------------------')
            fp32_model_path = f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx'
            path_prefix = os.path.join(config["OutputPath"], model_name)

            graph = load_onnx_graph(onnx_import_file = fp32_model_path)

            # 量化onnx模型
            if len(cfg.DO_QUANTIZATION) > 0:
                ppq_quant_ir = quantize_native_model(
                    setting=setting,                     # setting 对象用来控制标准量化逻辑
                    model=graph,
                    calib_dataloader=calib_dataloader,
                    calib_steps=cfg.CALIBRATION_NUM // cfg.CALIBRATION_BATCH_SIZE,
                    input_shape=input_size, # 如果你的网络只有一个输入，使用这个参数传参
                    collate_fn=lambda x: x.to(cfg.DEVICE),  # collate_fn 跟 torch dataloader 的 collate fn 是一样的，用于数据预处理，
                    platform=config["QuantPlatform"],
                    device=cfg.DEVICE,
                    do_quantize=True)

                if cfg.ERROR_ANALYSE:
                    print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
                    reports = graphwise_error_analyse(
                        graph=ppq_quant_ir, running_device=cfg.DEVICE, steps=32,
                        dataloader=calib_dataloader, collate_fn=lambda x: x.to(cfg.DEVICE))
                    for op, snr in reports.items():
                        if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

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
                        graph_save_to=f'{path_prefix}-ORT-INT8')
                        
                if "PLATFORM" in cfg.DO_QUANTIZATION:
                     # 导出平台模型
                    if platform == "TRT":
                        config_path = f'{path_prefix}-{platform}-INT8.json'
                    else:
                        config_path = None
                    export_ppq_graph(
                        graph = ppq_quant_ir,
                        platform=config["ExportPlatform"],
                        graph_save_to=f'{path_prefix}-{platform}-INT8',
                        config_save_to=config_path)

            
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
                f'{fp32_model_path[:-5]}.bbox.json',  #fp32 result json
                f'{path_prefix}-PPQ-INT8.bbox.json',
                f'{path_prefix}-ORT-INT8.bbox.json',
                f'{path_prefix}-{platform}-INT8.bbox.json'
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