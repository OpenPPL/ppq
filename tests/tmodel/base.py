"""This file defines all ppq test models."""

from enum import Enum
from typing import Callable, List

import torch
from ppq.core import TargetPlatform


class ModelType(Enum):
    CLASSIFY     = 1 # 图像分类
    DETECTION    = 2 # 图像检测
    SEGMENTATION = 3 # 图像分割
    SUPERRES     = 4 # 超分辨率
    POINTCLOUD   = 5 # 三维点云
    OCR          = 6 # OCR
    TEXT_CLASSIFY = 7 # 文本分类
    TEXT_LABELING = 8 # 文本序列标注
    GAN          = 9 # 生成对抗网络
    SEQ2SEQ      = 10 # 文本生成
    NERF         = 11 # 神经辐射场
    REC          = 12 # 推荐系统
    BLOCK        = 13 # 小型子网


class PPQTestCase():
    def __init__(self, model_builder: Callable,
                 input_generator: Callable, model_type: ModelType,
                 model_name: str, running_device = 'cuda',
                 deploy_platforms: List[TargetPlatform] = None) -> None:
        self.deploy_platforms = deploy_platforms
        self.model_builder    = model_builder
        self.input_generator  = input_generator
        self.model_type       = model_type
        self.model_name       = model_name
        self.running_device   = running_device


def rand_tensor_generator(shape: List[int]):
    return torch.rand(size=shape)
