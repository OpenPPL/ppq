from ppq.api import dump_torch_to_onnx
import torch
import os
from .imagenet_util import (evaluate_onnx_module_with_imagenet,
                                evaluate_ppq_module_with_imagenet,
                                evaluate_openvino_module_with_imagenet,
                                evaluate_trt_module_with_imagenet)
import cfg
from onnxsim import simplify
import onnx
from ppq import *


def get_onnx_models():
    for name,(model_builder,weights) in cfg.MODELS.items():
        print(f"正在获取预训练模型：{name}")
        model = model_builder(weights=weights).to("cpu")  #放在cpu上，防止爆显存
        dump_torch_to_onnx(model=model, onnx_export_file=f'{os.path.join(cfg.FP32_BASE_PATH, name)}-FP32.onnx',
            input_shape=cfg.INPUT_SHAPE,input_dtype=torch.float,device="cpu")

        onnx_model = onnx.load(f'{os.path.join(cfg.FP32_BASE_PATH, name)}-FP32.onnx') 
        model_simp, check = simplify(onnx_model)   #对onnx模型进行简化，消除冗余算子        
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f'{os.path.join(cfg.FP32_BASE_PATH, name)}-FP32.onnx')



def get_fp32_accuracy(model_name):

    print(f"正在测试模型 {model_name} 的FP32准确度，测试集为ImageNet")
    report = evaluate_onnx_module_with_imagenet(
        onnxruntime_model_path=f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx', 
        imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE, 
        device=cfg.DEVICE)
    
    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

@ empty_ppq_cache
def get_ppq_accuracy(model_name,platform,ppq_ir):
    print(f"正在测试模型 {model_name} 在{platform}上的PPQ准确度，测试集为ImageNet")
    
    report = evaluate_ppq_module_with_imagenet(
        model=ppq_ir, imagenet_validation_dir=cfg.VALIDATION_DIR,
        batchsize=cfg.BATCHSIZE, device=cfg.DEVICE, verbose=True)

    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

def get_ort_accuracy(model_name,platform,output_path):
    print(f"正在测试模型 {model_name} 在{platform}上的ORT准确度，测试集为ImageNet")
    report = evaluate_onnx_module_with_imagenet(
        onnxruntime_model_path=f'{os.path.join(output_path, model_name)}-ORT-INT8.onnx', 
        imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE, 
        device=cfg.DEVICE)

    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

def get_platform_accuracy(model_name,platform,output_path):
    print(f"正在测试模型 {model_name} 在{platform}上的实际部署准确率，测试集为ImageNet")
    if platform == "OpenVino":
        report = evaluate_openvino_module_with_imagenet(
            model_path=f'{os.path.join(output_path, model_name)}-OpenVino-INT8.onnx', 
            imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE,
            device=cfg.DEVICE)

    elif platform == "TRT":
        report = evaluate_trt_module_with_imagenet(
            model_path=f'{os.path.join(output_path, model_name)}-TRT-INT8.engine', 
            imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE,
            device=cfg.DEVICE)
    else:
        print("只能进行TensorRt和OpenVino的部署推理精度测试!")
        return None
        
    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

if __name__ == "__main__":
    get_onnx_models()
    # get_fp32_accuracy("ResNet18")