#npy to bin
#coding:utf-8
import os
import numpy as np
import argparse
from tqdm import tqdm
import pdb


npy_dir = "data_npy_100"
bin_dir = "calibrate-data"


if __name__ == '__main__':
    all_npy_files = sorted([fname for fname in tqdm(os.listdir(npy_dir))if fname.endswith(".npy")])

    for i in range(len(all_npy_files)):
        npy_file_path = os.path.join(npy_dir, all_npy_files[i])
        npy_data = np.load(npy_file_path)
        bin_file_path = os.path.join(bin_dir, str(i)+".bin")
        np.array(npy_data).astype(np.float32).tofile(bin_file_path)


