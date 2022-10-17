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
CFG_BATCHSIZE = 128                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = 'Assets/Imagenet_Valid'   # 用来读取 validation dataset
CFG_TRAIN_DIR = 'Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_PLATFORM = TargetPlatform.PPL_CUDA_INT8    # 用来指定目标平台
CFG_DUMP_PATH = 'Output/'                      # 所有模型保存的路径名
QUANT_SETTING = QuantizationSettingFactory.default_setting() # 用来指定量化配置
QUANT_SETTING.lsq_optimization = True
QUANT_SETTING.quantize_activation_setting.calib_algorithm = 'kl'

if 1:
    if __name__ == '__main__':
        for model_builder, model_name in (
            (torchvision.models.mobilenet.mobilenet_v3_large, 'mobilenet_v3_large'),
            # (torchvision.models.resnet18, 'resnet18'),
        ):
            print(f'---------------------- PPQ Quantization Test Running with {model_name} ----------------------')
            model = model_builder(pretrained=True).to(CFG_DEVICE)

            dataloader = load_imagenet_from_directory(
                directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
                shuffle=False, subset=1280, require_label=False,
                num_of_workers=8)

            dataloader_test = load_imagenet_from_directory(
                directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
                shuffle=False, subset=1280, require_label=True,
                num_of_workers=8)

            ppq_quant_ir = quantize_torch_model(
                model=model, calib_dataloader=dataloader, input_shape=CFG_INPUT_SHAPE,
                calib_steps=1280 // CFG_BATCHSIZE, collate_fn=lambda x: x.to(CFG_DEVICE), verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)

            for op in ppq_quant_ir.operations.values():
                if isinstance(op, QuantableOperation):
                    op.dequantize()

            ppq_int8_report = evaluate_ppq_module_with_imagenet(
                model=ppq_quant_ir, imagenet_validation_dir=CFG_VALIDATION_DIR,
                batchsize=CFG_BATCHSIZE, device=CFG_DEVICE, verbose=True,
                imagenet_validation_loader=dataloader_test)

            # reports = graphwise_error_analyse(
            #     graph=ppq_quant_ir, running_device='cpu', steps=32,
            #     dataloader=dataloader)
            # for op, snr in reports.items():
            #     if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

            # layerwise_error_analyse(graph=ppq_quant_ir, running_device=CFG_DEVICE,
            #                         interested_outputs=None,
            #                         dataloader=dataloader, collate_fn=lambda x: x.to(CFG_DEVICE))

            export_ppq_graph(
                graph=ppq_quant_ir, platform=TargetPlatform.ONNX,
                graph_save_to = 'model_int8.onnx', config_save_to='model_int8.json')

            # export_ppq_graph(
            #     graph=ppq_quant_ir, 
            #     platform=TargetPlatform.ONNXRUNTIME,
            #     graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx')
            
            # evaluate_onnx_module_with_imagenet(
            #     onnxruntime_model_path=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx', 
            #     imagenet_validation_dir=CFG_VALIDATION_DIR, batchsize=CFG_BATCHSIZE, 
            #     device=CFG_DEVICE)
    else:
        raise Exception('You may not import this file.')
