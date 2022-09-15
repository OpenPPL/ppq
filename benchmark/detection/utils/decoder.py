from .box import retinanet_postprocess
import torch
from tqdm import tqdm
def post_process(model_type,outputs,class_num):
    """
    将任意模型的输出转换为常规输出
    [N,result]  N为样本数
    result: [C,[M,5]] 其中C为类别,M为bbox数目，5为左上角、右下角坐标与置信分数。
    并且根据尺度进行复原
    """
    if model_type == "Retinanet":
        print("Post-process with outputs in Retinanet type")
        results = []
        for output in tqdm(outputs):
            img_scale = output["scale_factor"] 
            labels,bboxs = output["output"]
            result = [[] for _ in range(class_num)]
            if len(labels.shape) > 2:
                # labels 和 bbox的输出顺序发生错换，需要调整
                bboxs,labels = labels,bboxs
                
            bboxs,labels = bboxs[0],labels[0]
            for i,label in enumerate(labels):
                bboxs[i][:4] = bboxs[i][:4] / img_scale  # 进行bbox的尺寸复原
                result[label].append(bboxs[i])
            results.append(result)
        return results
    elif model_type == "Retinanet-wo":
        print("Post-process with outputs in Retinanet-wo type")
        results = []
        for output in tqdm(outputs):
            img_scale = output["scale_factor"] 
            input_size = output["img_shape"][:2] 
            cls_heads,reg_heads = output["output"][:5],output["output"][5:]
            result = [[] for _ in range(class_num)]
            
            print(reg_heads[0].shape[1],cls_heads[0].shape[1])
            if reg_heads[0].shape[1] > cls_heads[0].shape[1]:
                # 执行ppq后，可能出现outputs通道顺序改变的情况
                # 正常reg_head.shape = (bs,9*4,h,w) cls_head.shape = (bs,9*80,h,w)
                # TODO:这里可能不严谨，只使用第二维大小来确定回归和分类输出，更好的应该只用output的名字
                reg_heads,cls_heads = cls_heads,reg_heads

            scores, bboxs, labels = retinanet_postprocess(input_size,cls_heads,reg_heads)
            bboxs,labels,scores = bboxs[0],labels[0].int(),scores.t()
            bboxs = torch.cat((bboxs,scores),dim=1)
            for i,label in enumerate(labels):
                bboxs[i][:4] = bboxs[i][:4] / img_scale  # 进行bbox的尺寸复原
                result[label].append(bboxs[i])
            results.append(result)
        return results