from tqdm import tqdm
from ...detection.dataset import build_dataset
from mmcv import collate
from torch.utils.data import DataLoader

# 获取coco数据集的图片列表

ann_file = "/home/geng/fiftyone/coco-2017/validation/labels.json"
data_root = '/home/geng/fiftyone/coco-2017/validation/data/'  # 数据的根路径。
batch_size = 1
input_size = (480,640)  #或者 (1,3,480,640)

# 数据加载
dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size)
dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate)

file_name_list = []
for x in tqdm(dataloader):
    file_name = x["img_metas"][0].data[0][0]["filename"]
    file_name_list.append(file_name)
with open("./images_list.txt","w") as f:
    f.writelines(file_name_list)