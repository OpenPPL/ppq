import os
import numpy as np
import pdb


def generate_bin(dst_dir):
    for num in range(500):
        data = np.random.random((1,3,224,224))
        name = os.path.join(dst_dir, str(num) + ".bin")
        np.array(data).astype(np.float32).tofile(name)

def npy2bin(src_dir,dst_dir):
    file_list = os.listdir(src_dir)
    for file in file_list:
        src_path = os.path.join(src_dir,file)
        dst_path = os.path.join(dst_dir,os.path.splitext(file)[0]+".bin")
        data = np.load(src_path)
        np.array(data).astype(np.float32).tofile(dst_path)

if __name__ == '__main__':
    dst_dir = "/home/geng/tinyml/ppq/benchmark/dynamic_shape_quant/onnx2tensorrt_python/calibrate-data"
    src_dir = "/home/geng/tinyml/ppq/benchmark/dynamic_shape_quant/calib_data/data"
    # generate_bin(dst_dir)
    npy2bin(src_dir,dst_dir)
