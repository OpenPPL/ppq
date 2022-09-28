import os
import random
import sys
import pdb
import argparse

import json
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import common
from calibrator import MNISTEntropyCalibrator

TRT_LOGGER = trt.Logger()

def json_load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def build_int8_engine_trt(onnx_info, calib):
    """build the float32 engine."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.max_workspace_size = common.GiB(1)

    ### 如果是动态的onnx模型，则需要加入以下内容
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 150, 220), (1, 3, 480, 640), (1, 3, 650,650))
    config.add_optimization_profile(profile)

    if not os.path.exists(onnx_info["onnx_file"]):
        quit('ONNX file {} not found'.format(onnx_info["onnx_file"]))

    with open(onnx_info["onnx_file"], 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calib

    engine = builder.build_engine(network, config)

    with open(onnx_info["int8_trt_engine_file"], "wb") as f:
        f.write(engine.serialize())
        print("\033[1;32mgenerate %s\033[0m" % onnx_info["int8_trt_engine_file"])


def main():
    onnx_config = "config.json"
    if not os.path.exists(onnx_config):
        quit('ONNX config {} not found'.format(onnx_config))
    onnx_info = json_load(onnx_config)
    
    calib = MNISTEntropyCalibrator(onnx_info=onnx_info)
    build_int8_engine_trt(onnx_info, calib)


if __name__ == '__main__':
    main()
