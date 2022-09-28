from inference import  ppq_inference,openvino_inference,trt_inference,onnxruntime_inference
from .decoder import post_process
from mmcv.parallel import collate
from torch.utils.data import DataLoader


def ppq_inference2json(dataset,model_name,ppq_ir,batch_size,class_num,outfile_prefix,device):
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate)
    outputs = ppq_inference(dataloader=dataloader,ppq_ir = ppq_ir,device = device)
    results = post_process(model_name,outputs,class_num)
    dataset.results2json(results=results,outfile_prefix=outfile_prefix)  # 结果统一保存为coco json


def onnxruntime_inference2json(dataset,model_name,model_path,batch_size,class_num,outfile_prefix,device):
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate)
    outputs = onnxruntime_inference(dataloader,model_path,device=device)
    results = post_process(model_name,outputs,class_num)
    dataset.results2json(results=results,outfile_prefix=outfile_prefix)  # 结果统一保存为coco json

def trt_inference2json(dataset,model_name,model_path,batch_size,class_num,outfile_prefix,device):
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate)
    outputs = trt_inference(dataloader,model_path,device)
    results = post_process(model_name,outputs,class_num)
    dataset.results2json(results=results,outfile_prefix=outfile_prefix)  # 结果统一保存为coco json

def openvino_inference2json(dataset,model_name,model_path,batch_size,class_num,outfile_prefix,device):
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate)
    outputs = openvino_inference(dataloader,model_path,device)
    results = post_process(model_name,outputs,class_num)
    dataset.results2json(results=results,outfile_prefix=outfile_prefix)  # 结果统一保存为coco json