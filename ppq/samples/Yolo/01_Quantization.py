# 这里我们展示两种不同的方法去生成 TensorRT Engine

# Plan A: TensorRT 自己校准
import glob
import json
import os

import numpy as np
import onnx
import pycuda.autoinit  # noqa F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision.transforms as transforms
from PIL import Image


ONNX_PATH        = 'models/yolov5s.v5.onnx'       # 你的模型位置
ENGINE_PATH      = 'Output/yolov5s.v5(trt).engine' # 生成的 Engine 位置
CALIBRATION_PATH = 'imgs'                         # 校准数据集


class EntropyCalibrator(trt.IInt8MinMaxCalibrator):

    def __init__(self, files_path=r'imgs'):
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = 'Net.cache'

        self.batch_size = 1
        self.Channel = 3
        self.Height = 640
        self.Width = 640
        self.transform = transforms.Compose([
            transforms.Resize([self.Height, self.Width]),  # [h,w]
            transforms.ToTensor(),
        ])

        self.imgs = glob.glob(os.path.join(files_path, '*'))
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs)//self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel,self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size:\
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = Image.open(f).convert('RGB')
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size/self.batch_size), 'not valid img!'+f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.Channel*self.Height*self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# create builder and network
device = torch.device('cuda')
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, logger)

onnx_model = onnx.load(ONNX_PATH)
parser.parse(onnx_model.SerializeToString())

config = builder.create_builder_config()
# config.max_workspace_size = 4 << 30 no need for trt 8.4GA
config.set_flag(trt.BuilderFlag.INT8)
calibrator = EntropyCalibrator(files_path=CALIBRATION_PATH)
config.int8_calibrator = calibrator

plan = builder.build_serialized_network(network, config)

with open(ENGINE_PATH, mode='wb') as f:
    f.write(plan)
