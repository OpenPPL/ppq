from dataset import CocoDataset
from torch.utils.data import DataLoader
import os

ann_file = "/home/geng/fiftyone/coco-2017/validation/labels.json"
# dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = '/home/geng/fiftyone/coco-2017/validation/data/'  # 数据的根路径。
batch_size = 1
input_size = (1216,800)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

from dataset import build_dataset
from mmcv.parallel import collate
dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size)
calib_dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate)

from utils import (onnxruntime_inference2json,openvino_inference2json,
                    trt_inference2json,ppq_inference2json)

# 测试opevino推理
model_name = "Retinanet"
model_path = "/home/geng/tinyml/ppq/benchmark/classification/FP32_model/MobileNetV2-FP32.onnx"
openvino_inference2json(dataset=dataset,model_name=model_name,model_path=model_path,
    batch_size=input_size[0],class_num=80,
    outfile_prefix=model_path[:-5],
    device="cuda")

# trt_inference2json(dataset=dataset,model_name=model_name,model_path=model_path,
#     batch_size=input_size[0],class_num=80,
#     outfile_prefix=model_path[:-5],
#     device="cuda")