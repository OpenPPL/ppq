from .box import retinanet_postprocess
import torch
from tqdm import tqdm
import pycocotools.mask as mk
import cv2
import numpy as np

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

    elif model_type == "Retinanet-wo":
        print("Post-process with outputs in Retinanet-wo type")
        results = []
        for output in tqdm(outputs):
            img_scale = output["scale_factor"] 
            input_size = output["img_shape"][:2] 
            cls_heads,reg_heads = output["output"][:5],output["output"][5:]
            result = [[] for _ in range(class_num)]
            
            # print(reg_heads[0].shape[1],cls_heads[0].shape[1])
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
        
    elif model_type == "MaskRCNN":
        print("Post-process with outputs in MaskRCNN type")
        results = []
        # 实例分割格式化
        for output in tqdm(outputs):
            img_scale = output["scale_factor"] 
            bboxs,labels,scores,masks = output["output"]
            result_det = [[] for _ in range(class_num)]
            result_seg = [[] for _ in range(class_num)]

            bboxs /= img_scale 
            
            """
            注意opencv读取的image.size=(w,h) 而转为numpy时将变为(h,w),h代表的是y纬度,w代表的是x纬度
            """
            img_array_size = output["img_size"][::-1]

            # seg result
            for mask, box, label in zip(masks, bboxs, labels):
                mask = mask[0, :, :, None]
                # print(box)
                int_box = [int(i) for i in box]
                mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
                mask = mask > 0.5
                
                im_mask = np.zeros((img_array_size[0], img_array_size[1]), dtype=np.uint8)
                x_0 = min(max(int_box[0], 0),img_array_size[1]) #约束不让超出图像尺寸宽度
                x_1 = min(int_box[2] + 1, img_array_size[1])
                y_0 = min(max(int_box[1], 0),img_array_size[0])  #约束不让超出图像高度
                y_1 = min(int_box[3] + 1, img_array_size[0])
                
                mask_y_0,mask_x_0 = 0,0
                mask_y_1 = mask_y_0 + y_1 - y_0
                mask_x_1 = mask_x_0 + x_1 - x_0
                im_mask[y_0:y_1, x_0:x_1] = mask[mask_y_0 : mask_y_1, mask_x_0 : mask_x_1]  #原图尺寸的掩膜
                
                rle = mk.encode(np.asfortranarray(im_mask[:,:,None]))
                result_seg[label-1].append(rle[0])
            
            bboxs = torch.cat((torch.from_numpy(bboxs),torch.from_numpy(scores.reshape((-1,1)))),dim=1)
            # bbox result
            for i,label in enumerate(labels):
                result_det[label-1].append(bboxs[i])
                    
            results.append((result_det,result_seg))
    return results