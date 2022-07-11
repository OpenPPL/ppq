from ppq.api import dump_torch_to_onnx
import torch
import os
from Utilities.Imagenet import evaluate_onnx_module_with_imagenet
import cfg


def get_onnx_models():
    for name,model_builder in cfg.MODELS.items():
        model = model_builder(pretrained=True).to(cfg.DEVICE)
        dump_torch_to_onnx(model=model, onnx_export_file=f'{os.path.join(cfg.FP32_BASE_PATH, name)}-FP32.onnx',
            input_shape=cfg.INPUT_SHAPE,input_dtype=torch.float,device=cfg.DEVICE)

def get_fp32_accuracy():
    acc = {}
    for name in cfg.MODELS.keys():
        report = evaluate_onnx_module_with_imagenet(
            onnxruntime_model_path=f'{os.path.join(cfg.FP32_BASE_PATH, name)}-FP32.onnx', 
            imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE, 
            device=cfg.DEVICE)
        
        top1=sum(report['top1_accuracy'])/len(report['top1_accuracy'])
        top5=sum(report['top5_accuracy'])/len(report['top5_accuracy'])
        acc[name] = (top1,top5)
    return acc

if __name__ == "__main__":
    get_onnx_models()
    print(get_fp32_accuracy())