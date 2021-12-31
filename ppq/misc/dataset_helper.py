import os
import numpy as np


def random_calib_data(size, path, num=100):
    for i in range(num):
        data = np.random.randn(*size).astype('f')
        filename = os.path.join(path, str(i))
        np.save(filename, data)
        print(f'save random cali data file {filename}.npy')
