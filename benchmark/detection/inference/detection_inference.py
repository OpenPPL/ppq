from tqdm import tqdm
import onnxruntime
from ppq import *
import torch
import numpy as np
import openvino.inference_engine as ie
from .trt_infer import TrtInferenceModel


# ppq 推理过程
@ empty_ppq_cache
def ppq_inference(dataloader,ppq_ir,device="cuda"):
    executor = TorchExecutor(graph=ppq_ir, device=device)
    model_forward_function = lambda input_tensor: [x.to("cpu") for x in executor(*[input_tensor])]
    return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)

#  onnx推理过程
def onnxruntime_inference(dataloader,onnxruntime_model_path,device="cuda"):
    providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(path_or_bytes=onnxruntime_model_path, providers=providers)
    input_placeholder_name = sess.get_inputs()[0].name
    outputnames = [x.name for x in sess.get_outputs()]
    outputs = []
    with torch.no_grad():
            # 将[numpy.darray]提前转为numpy.darray,提升推理速度
        model_forward_function = lambda input_tensor: sess.run(
            input_feed={input_placeholder_name: input_tensor.cpu().numpy()}, output_names=outputnames)
        return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)

# openvino 推理过程
def openvino_inference(dataloader,openvino_model_path,device):
    core = ie.IECore()
    network = core.read_network(openvino_model_path)
    model_openvino = core.load_network(network, "CPU")
    input_name = list(model_openvino.input_info.keys())[0]
    model_forward_function = lambda input_tensor: list(
        model_openvino.infer({input_name:convert_any_to_numpy(input_tensor)}).values())
    return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)

# tensorRT 推理过程
def trt_inference(dataloader,trt_model_path,device):
    model_forward_function = TrtInferenceModel(trt_model_path)
    return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)



# coco数据集推理
def _inference_any_module_with_coco(model_forward_function,dataloader,device):
    outputs = []
    for x in tqdm(dataloader):
        input_tensor = x["img"][0].to(device)
        img_metas = x["img_metas"][0].data[0][0]
        output = model_forward_function(input_tensor)
        img_metas["output"] = output
        outputs.append(img_metas)
    return outputs

