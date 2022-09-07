import os
import cfg
from ppq import *

def get_fp32_report(model_name):

    print(f"正在测试模型 {model_name} 的FP32准确度，测试集为ImageNet")
    report = evaluate_onnx_module_with_imagenet(
        onnxruntime_model_path=f'{os.path.join(cfg.FP32_BASE_PATH, model_name)}-FP32.onnx', 
        imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE, 
        device=cfg.DEVICE)
    
    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

@ empty_ppq_cache
def get_ppq_report(model_name,platform,ppq_ir):
    print(f"正在测试模型 {model_name} 在{platform}上的PPQ准确度，测试集为ImageNet")
    
    report = evaluate_ppq_module_with_imagenet(
        model=ppq_ir, imagenet_validation_dir=cfg.VALIDATION_DIR,
        batchsize=cfg.BATCHSIZE, device=cfg.DEVICE, verbose=True)

    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

def get_ort_report(model_name,platform,output_path):
    print(f"正在测试模型 {model_name} 在{platform}上的ORT准确度，测试集为ImageNet")
    report = evaluate_onnx_module_with_imagenet(
        onnxruntime_model_path=f'{os.path.join(output_path, model_name)}-ORT-INT8.onnx', 
        imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE, 
        device=cfg.DEVICE)

    top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
    top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])

    return (top1,top5)

def get_platform_report(model_name,platform,output_path):
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