import os
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection import ssd300_vgg16,SSD300_VGG16_Weights

data_path= "/home/geng/fiftyone/coco-2017/validation/data"
labels_path="/home/geng/fiftyone/coco-2017/validation/labels.json"
transform = transforms.Compose([
    transforms.Resize(size=(300,300)),
    transforms.ToTensor(),
])
# def collate_fn_coco(batch):
#     return tuple(zip(*batch))
# coco 数据集存在空标签
coco_dataset = torchvision.datasets.CocoDetection(data_path, labels_path, transform=transform)
coco_dataloader = torch.utils.data.DataLoader(coco_dataset,batch_size=1,num_workers=2)



# Step 1: Initialize model with the best available weights
weights = SSD300_VGG16_Weights.COCO_V1
model = ssd300_vgg16(weights=weights,score_thresh=0.95)
model.eval()

# Step 2: Initialize the inference transforms

# Step 3: Apply inference preprocessing transforms
# Step 4: Use the model and visualize the prediction

from tqdm import tqdm
res = []
for images,labels in tqdm(coco_dataloader):
    if not labels: continue
    output = model(images)[0]
    if len(labels) == 0:
        print("error!")
        print(labels)
    for box in output["boxes"]:
        res.append({
            'score': output["scores"],
            'category_id': output["labels"],
            'bbox': box,
            'image_id': labels[0]["image_id"]
        })

import json
b = json.dump({'annotations':res})
with open("prediction.json","w") as f:
    f.write(b)


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tempfile import NamedTemporaryFile


# json file in coco format, original annotation data
anno_file =labels_path
coco_gt = COCO(anno_file)

# Use GT box as prediction box for calculation, the purpose is to get detection_res
with open(anno_file,'r') as f:
    json_file = json.load(f)
annotations = json_file['annotations']
detection_res = []
for anno in annotations:
    detection_res.append({
        'score': 1.,
        'category_id': anno['category_id'],
        'bbox': anno['bbox'],
        'image_id': anno['image_id']
    })
with NamedTemporaryFile(suffix='.json') as tf:
    # Due to subsequent needs, first convert detection_res to binary and then write it to the json file
    content = json.dumps(detection_res).encode(encoding='utf-8')
    tf.write(content)
    res_path = tf.name

    # loadRes will generate a new COCO type instance based on coco_gt and return
    coco_dt = coco_gt.loadRes(res_path)

    cocoEval = COCOeval(coco_gt, coco_dt,'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

print(cocoEval.stats)