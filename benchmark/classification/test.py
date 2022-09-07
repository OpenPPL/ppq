# 这是一个测试脚本，测试一些jupyter无法使用的代码
import trt_infer
import tensorrt as trt
from tqdm import tqdm
from ppq.api import *
from trt_infer import TrtInferenceModel

def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    你需要注意我这里留了数据 IO，数据总是从 host 送往 device 的
    如果你只关心 GPU 上的运行时间，你应该修改这个方法使得数据不发生迁移
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        for sample in tqdm(samples, desc='TensorRT is running...'):
            inputs[0].host = convert_any_to_numpy(sample)
            [output] = trt_infer.do_inference(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream, batch_size=64)
            results.append(convert_any_to_torch_tensor(output))
    return results

import cfg
# INPUT_SHAPE = (64, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
# SAMPLES = torch.rand(size=INPUT_SHAPE)
name = 'ResNet18'
target = "TRT"
# model = TrtInferenceModel(model_path,batch_size=64)
# # res = infer_trt(model_path,SAMPLES)
# res = model(SAMPLES)
# print(res.shape)

from Utilities.Imagenet import evaluate_trt_module_with_imagenet

# evaluate_trt_module_with_imagenet(model_path=model_path,imagenet_validation_dir=cfg.VALIDATION_DIR,
# batchsize=64,device="cuda")

import cfg
from Utilities.Imagenet import (evaluate_onnx_module_with_imagenet,
                                evaluate_ppq_module_with_imagenet,
                                evaluate_openvino_module_with_imagenet,
                                evaluate_trt_module_with_imagenet)
# report = evaluate_openvino_module_with_imagenet(
#     model_path='/home/geng/tinyml/ppq/benchmark/classification/OpenVino_output/ResNet18-ORT-INT8.onnx', 
#     imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE,
#     device=cfg.DEVICE)
report = evaluate_onnx_module_with_imagenet(
    onnxruntime_model_path="/home/geng/tinyml/ppq/benchmark/classification/OpenVino_output/ResNet18-OpenVino-INT8.onnx", 
    imagenet_validation_dir=cfg.VALIDATION_DIR, batchsize=cfg.BATCHSIZE, 
    device=cfg.DEVICE)
