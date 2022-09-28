import random
import numpy as np
import os

def generate_calib_data(dataset,seed_num,calib_num):
    random.seed(seed_num)
    calib_dataloader = [dataset[i]["img"][0].unsqueeze(0) for i in random.sample(range(len(dataset)),calib_num)]
    if not os.path.exists("./calib_data/data/"):
        os.makedirs("./calib_data/data/")
    for i,img in enumerate(calib_dataloader):
        np.save(f"./calib_data/data/{i+1}.npy",img.numpy())