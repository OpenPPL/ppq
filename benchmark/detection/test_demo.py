from torch.utils.data import DataLoader
from dataset import build_dataset
from mmcv.parallel import collate
"""
这个文件演示了如何对检测模型在某个平台上推理，并评测精度。
"""


# 数据配置文件
ann_file = "/home/geng/fiftyone/coco-2017/validation/labels.json"
data_root = '/home/geng/fiftyone/coco-2017/validation/data/'  # 数据的根路径。
batch_size = 1
input_size = (480,640)  #或者 (1,3,480,640)


# 数据加载
dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size)
dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate)


# 检测模型推理模块，该模块的输出严格按照模型结构输出
from inference import onnxruntime_inference,trt_inference
trt_model_path = "/home/geng/tinyml/ppq/benchmark/detection/TRT_output/Retinanet-wo-TRT-INT8.engine"
dataloader = [next(iter(dataloader))]  #方便测试可以只推理了一张图
outputs = trt_inference(dataloader,trt_model_path)



# 后处理与格式化，该模块实现了模型后处理方法，并转为coco友好的result格式
"""
results = [N,result]  N为样本数
result: [C,[M,5]] 其中C为类别,M为bbox数目，5为左上角、右下角坐标与置信分数。
并且根据尺度进行复原
"""
# 目前实现了两类模型处理方式: Retinanet （端到端）和 Retinanet-wo （无后处理）
from utils import post_process
results = post_process("Retinanet-wo",outputs,80)
# dataset.results2json(results=results,outfile_prefix="./retinanet-trt-ort") #也可以将推理结果持久化

# MAP结果评估
dataset.evaluate(results=results) # 直接使用输出结果进行测试
# dataset.evaluate(results_json_path="Retinanet-PPQ-INT8.bbox.json") #方法2，使用持久化的推理结果进行评估    

