import torchvision
from ppq import *
from ppq.api import *
from Utilities.Imagenet import (evaluate_mmlab_module_with_imagenet,
                                evaluate_onnx_module_with_imagenet,
                                evaluate_ppq_module_with_imagenet,
                                evaluate_torch_module_with_imagenet,
                                load_imagenet_from_directory)
import os
import cfg

CFG_PLATFORM = TargetPlatform.SNPE_INT8  # 用来指定目标平台
QUANT_SETTING = QuantizationSettingFactory.dsp_setting() # 用来指定量化配置
platform = "Snpe"  #记得修改上面两个

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 64                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Valid'   # 用来读取 validation dataset
CFG_TRAIN_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_DUMP_PATH = '/home/geng/tinyml/ppq/benchmark/classification/'+platform+'_output'    # 所有模型保存的路径名

if not os.path.exists(CFG_DUMP_PATH):
    os.makedirs(CFG_DUMP_PATH)

with ENABLE_CUDA_KERNEL():
    model_name = "ResNet18"
    print(f'---------------------- PPQ Quantization Test Running with {model_name} ----------------------')
    model_path = f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx'
    # 获取校准数据
    dataloader = load_imagenet_from_directory(
        directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
        shuffle=False, subset=5120, require_label=False,
        num_of_workers=8)

    # 量化onnx模型
    ppq_quant_ir = quantize_onnx_model(
        onnx_import_file=model_path, calib_dataloader=dataloader, calib_steps=5120 // CFG_BATCHSIZE, 
        setting=QUANT_SETTING,input_shape=CFG_INPUT_SHAPE, collate_fn=lambda x: x.to(CFG_DEVICE), 
        platform=CFG_PLATFORM, do_quantize=True)
        
    # 评估PPQ量化后的模型
    # ppq_int8_report = evaluate_ppq_module_with_imagenet(
    #     model=ppq_quant_ir, imagenet_validation_dir=CFG_VALIDATION_DIR,
    #     batchsize=CFG_BATCHSIZE, device=CFG_DEVICE, verbose=True)

    # 导出ORT模型
    # export_ppq_graph(
    #     graph=ppq_quant_ir, 
    #     platform=TargetPlatform.ONNXRUNTIME,
    #     graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}-ORT-INT8.onnx')
    
    # 导出平台模型
    export_ppq_graph(
        graph=ppq_quant_ir, 
        platform=CFG_PLATFORM,
        graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}-{platform}-INT8.onnx')
    print("successfully export ort and trt model!")

    # 评估onnx运行模型
    # evaluate_onnx_module_with_imagenet(
    #     onnxruntime_model_path=f'{os.path.join(CFG_DUMP_PATH, model_name)}-ORT-INT8.onnx', 
    #     imagenet_validation_dir=CFG_VALIDATION_DIR, batchsize=CFG_BATCHSIZE, 
    #     device=CFG_DEVICE)
    

