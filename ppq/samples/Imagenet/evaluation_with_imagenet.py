import torchvision
from ppq import *
from ppq.api import *
from Utilities.Imagenet import (evaluate_mmlab_module_with_imagenet,
                                evaluate_onnx_module_with_imagenet,
                                evaluate_ppq_module_with_imagenet,
                                evaluate_torch_module_with_imagenet,
                                load_imagenet_from_directory)

"""
    使用这个脚本来测试量化 torchvision 中的典型分类模型
        使用 imagenet 中的数据测试量化精度与 calibration
        默认的 imagenet 数据集位置: Assets/Imagenet_Train, Assets/Imagenet_Valid
        你可以通过软连接创建它们:
            ln -s /home/data/Imagenet/val Assets/Imagenet_Valid
            ln -s /home/data/Imagenet/train Assets/Imagenet_Train
"""

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 64                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = 'Assets/Imagenet_Valid'   # 用来读取 validation dataset
CFG_TRAIN_DIR = 'Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_PLATFORM = TargetPlatform.PPL_CUDA_INT8    # 用来指定目标平台
CFG_DUMP_PATH = 'Output/'                      # 所有模型保存的路径名
QUANT_SETTING = QuantizationSettingFactory.default_setting() # 用来指定量化配置
QUANT_SETTING.lsq_optimization = True

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':
        for model_builder, model_name in (
            # (torchvision.models.alexnet, 'alexnet'),
            (torchvision.models.inception_v3, 'inception_v3'),

            (torchvision.models.mnasnet0_5, 'mnasnet0_5'),
            (torchvision.models.mnasnet1_0, 'mnasnet1_0'),

            (torchvision.models.squeezenet1_0, 'squeezenet1_0'),
            # (torchvision.models.squeezenet1_1, 'squeezenet1_1'),
            (torchvision.models.shufflenet_v2_x1_0, 'shufflenet_v2_x1_0'),
            (torchvision.models.resnet18, 'resnet18'),
            (torchvision.models.resnet34, 'resnet34'),
            (torchvision.models.resnet50, 'resnet50'),
            # (torchvision.models.resnext50_32x4d, 'resnext50_32x4d'),

            # (torchvision.models.wide_resnet50_2, 'wide_resnet50_2'),
            (torchvision.models.googlenet, 'googlenet'),

            (torchvision.models.vgg11, 'vgg11'),
            # (torchvision.models.vgg16, 'vgg16'),

            (torchvision.models.mobilenet.mobilenet_v2, 'mobilenet_v2'),
        ):
            print(f'---------------------- PPQ Quantization Test Running with {model_name} ----------------------')
            model = model_builder(pretrained=True).to(CFG_DEVICE)

            dataloader = load_imagenet_from_directory(
                directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
                shuffle=False, subset=5120, require_label=False,
                num_of_workers=8)

            ppq_quant_ir = quantize_torch_model(
                model=model, calib_dataloader=dataloader, input_shape=CFG_INPUT_SHAPE,
                calib_steps=5120 // CFG_BATCHSIZE, collate_fn=lambda x: x.to(CFG_DEVICE), verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)
                
            ppq_int8_report = evaluate_ppq_module_with_imagenet(
                model=ppq_quant_ir, imagenet_validation_dir=CFG_VALIDATION_DIR,
                batchsize=CFG_BATCHSIZE, device=CFG_DEVICE, verbose=True)

            export_ppq_graph(
                graph=ppq_quant_ir, 
                platform=TargetPlatform.ONNXRUNTIME,
                graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx')
            
            evaluate_onnx_module_with_imagenet(
                onnxruntime_model_path=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx', 
                imagenet_validation_dir=CFG_VALIDATION_DIR, batchsize=CFG_BATCHSIZE, 
                device=CFG_DEVICE)
    else:
        raise Exception('You may not import this file.')
