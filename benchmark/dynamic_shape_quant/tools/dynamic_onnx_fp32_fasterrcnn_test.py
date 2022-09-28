# # 动态ONNX推理量化测试
# ## 1. FP32精度测试 
import sys
sys.path.append("../detection")

import numpy as np
from PIL import Image
import cv2
import torch
import onnxruntime
from tqdm import tqdm
from dataset import build_dataset


def preprocess(image):
    channel_num = image.layers  #通道数

    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])),  Image.Resampling.BILINEAR)
    
    if channel_num == 1:  #灰度图像转三通道
        # print("trans")
        image = cv2.cvtColor(np.array(image) , cv2.COLOR_GRAY2RGB)
   
    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

#  加载数据集路径

with open("./images_list.txt","r") as f:
   img_path_list =  f.readlines()
img_path_list  = [x.rstrip() for x in img_path_list]


# 模型推理
onnxruntime_model_path = "/home/geng/tinyml/ppq/benchmark/detection/FP32_model/FasterRCNN-12-FP32.onnx"
providers = ['CUDAExecutionProvider']  
sess = onnxruntime.InferenceSession(path_or_bytes=onnxruntime_model_path, providers=providers)
input_placeholder_name = sess.get_inputs()[0].name

outputs = []
for img_path in tqdm(img_path_list):
    img = Image.open(img_path)
    try:
        input_tensor = preprocess(img)
    except IndexError:
        print(f"error happend in precess {img_path}")
        break
    output = sess.run(input_feed={input_placeholder_name: input_tensor}, output_names=None)
    outputs.append({"file_name":img_path,"img_size":img.size,"output":output})



# output格式化
class_num = 80
results = []
for output in tqdm(outputs):
    bboxs,labels,scores = output["output"]
    result = [[] for _ in range(class_num)]
    ratio = 800.0 / min(output["img_size"][0], output["img_size"][1])
    bboxs /= ratio
    
    bboxs = torch.cat((torch.from_numpy(bboxs),torch.from_numpy(scores.reshape((-1,1)))),dim=1)
    for i,label in enumerate(labels):
        result[label-1].append(bboxs[i])
    results.append(result)


# 结果评估

ann_file = "/home/geng/fiftyone/coco-2017/validation/labels.json"
data_root = '/home/geng/fiftyone/coco-2017/validation/data/'  # 数据的根路径。
batch_size = 1
input_size = (480,640)  #或者 (1,3,480,640)

# 数据加载
dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size)
dataset.results2json(results=results,outfile_prefix="/home/geng/tinyml/ppq/benchmark/detection/FP32_model/FasterRCNN-12-FP32") #也可以将推理结果持久化


dataset.evaluate(results=results) 
# dataset.evaluate(results_json_path="/home/geng/tinyml/ppq/benchmark/detection/FP32_model/FasterRCNN-12-FP32.bbox.json") 




