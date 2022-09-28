import os
import pdb

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image


def load_data(data_dir):
    file_names = os.listdir(data_dir)
    file_paths = []
    for i, fname in enumerate(file_names):
        file_paths.append(os.path.join(data_dir, fname))
    return file_paths


class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, onnx_info):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_table = onnx_info["calibration_table"]
        self.file_paths = load_data(onnx_info["calibrate_dir"])
        self.batch_size = onnx_info["onnx_model_batchSize"]
        self.channel = onnx_info["channel"]
        self.height = onnx_info["height"]
        self.width = onnx_info["width"]
        self.current_index = 0
        self.batch = []
        self.data = np.zeros((self.batch_size, self.channel, self.height, self.width), dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The   list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        print("current_index:", self.current_index)

        if self.current_index + self.batch_size >= len(self.file_paths):
            return None

        if self.batch_size == 1:
            input_tensor = np.fromfile(self.file_paths[self.current_index], dtype=np.float32)
            cuda.memcpy_htod(self.device_input, input_tensor)
            self.current_index += self.batch_size

        else:
            batch = []
            start = self.current_index
            for num in range(self.current_index, self.current_index + self.batch_size):
                input_tensor = np.fromfile(self.file_paths[num], dtype=np.float32)
                if num == start:
                    self.batch = input_tensor
                else:
                    self.batch = np.vstack([self.batch, input_tensor])
            assert (self.batch.shape[0] == self.batch_size)
            batch = self.batch.ravel()
            cuda.memcpy_htod(self.device_input, batch)
            self.current_index += self.batch_size

        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_table):
            print("\033[1;32mFound %s,trt using it\033[0m" % self.cache_table)
            with open(self.cache_table, "rb") as f:
                return f.read()
        else:
            print("\033[1;32mNot found %s,trt will generate it\033[0m" % self.cache_table)

    def write_calibration_cache(self, cache):
        with open(self.cache_table, "wb") as f:
            f.write(cache)
        print("\033[1;32mGenerate %s successfully\033[0m" % self.cache_table)
