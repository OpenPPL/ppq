from dataset import CocoDataset
from torch.utils.data import DataLoader
import os

ann_file = "/home/geng/fiftyone/coco-2017/validation/labels.json"
data_root = '/home/geng/fiftyone/coco-2017/validation/data/'  # 数据的根路径。
batch_size = 1
input_size = (480,640)

from dataset import build_dataset
from mmcv.parallel import collate
dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size)
dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate)

from inference import onnxruntime_inference
# onnxruntime_model_path = "/home/geng/tinyml/ppq/benchmark/detection/FP32_model/Retinanet-wo-FP32.onnx"
onnxruntime_model_path = "/home/geng/tinyml/ppq/benchmark/detection/OpenVino_output/Retinanet-wo-ORT-INT8.onnx"
dataloader = [next(iter(dataloader))]
outputs = onnxruntime_inference(dataloader,onnxruntime_model_path)

from utils import post_process
post_process("Retinanet-wo",outputs,80)