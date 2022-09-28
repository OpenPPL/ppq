from tqdm import tqdm
import onnxruntime
from ppq import *
import torch
import openvino.inference_engine as ie
from .trt_infer import TrtInferenceModel
import re


# ppq 推理过程
@ empty_ppq_cache
def ppq_inference(dataloader,ppq_ir,device="cuda"):
    executor = TorchExecutor(graph=ppq_ir, device=device)
    model_forward_function = lambda input_tensor: [x.to("cpu") for x in executor(*[input_tensor])]
    return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)

#  onnx推理过程
def onnxruntime_inference(dataloader,onnxruntime_model_path,device="cuda"):
    providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    onnxruntime.set_default_logger_severity(3)
    sess = onnxruntime.InferenceSession(path_or_bytes=onnxruntime_model_path, providers=providers)
    input_placeholder_name = sess.get_inputs()[0].name
    outputnames = [x.name for x in sess.get_outputs()]
    # 指定output的顺序，按照升序排列
    if len(outputnames) > 3:
        outputnames.sort(key=lambda x:int(re.findall("\d+",x)[0]))
    with torch.no_grad():
            # 将[numpy.darray]提前转为numpy.darray,提升推理速度
        model_forward_function = lambda input_tensor: sess.run(
            input_feed={input_placeholder_name: input_tensor.cpu().numpy()}, output_names=outputnames)
        return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)

# openvino 推理过程
def openvino_inference(dataloader,openvino_model_path,device="cpu"):
    core = ie.IECore()
    network = core.read_network(openvino_model_path)
    model_openvino = core.load_network(network, "CPU")
    input_name = list(model_openvino.input_info.keys())[0]
    model_forward_function = lambda input_tensor: list(
        model_openvino.infer({input_name:convert_any_to_numpy(input_tensor)}).values())
    return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device)

# tensorRT 推理过程
def trt_inference(dataloader,trt_model_path,device="cuda"):
    def trt_outputs_map(outputs):
        if len(outputs) < 10:
            return outputs
            
        shape_map = {
            # 记录trt每个输入应该的形状以及正确的索引
            3456000:((1, 720, 60, 80),0),
            864000:((1, 720, 30, 40),1),
            216000:((1, 720, 15, 20),2),
            57600:((1, 720, 8, 10),3),
            14400:((1, 720, 4, 5),4),
            172800:((1,36,60,80),5),
            43200:((1, 36, 30, 40),6),
            10800:((1, 36, 15, 20),7),
            2880:((1, 36, 8, 10),8),
            720:((1, 36, 4, 5),9)          
        }
        
        standard_outputs = [None for _ in range(len(outputs))]
        for output in outputs:
            shape,idx = shape_map[output.shape[0]]
            output = torch.tensor(output).reshape(shape)  #这里必须要用torch的reshape，用numpy的会出错
            standard_outputs[idx] = output
        return  standard_outputs
        
    

    model_forward_function = TrtInferenceModel(trt_model_path)
    return _inference_any_module_with_coco(model_forward_function=model_forward_function,dataloader=dataloader,device=device,output_map = trt_outputs_map)



# coco数据集推理
def _inference_any_module_with_coco(model_forward_function,dataloader,device,output_map=None):
    outputs = []
    print("infering on coco dataset....")
    from time import time
    start = time()
    for x in tqdm(dataloader):
        input_tensor = x["img"][0].to(device)
        img_metas = x["img_metas"][0].data[0][0]
        output = model_forward_function(input_tensor)
        if output_map:
            output = output_map(output)
        img_metas["output"] = output
        outputs.append(img_metas)
    end = time()
    print("Inference has finished! Speed is {:.2f} FPS".format(len(dataloader)/(end-start)))
    return outputs

