from .box import retinanet_postprocess
def post_process(model_type,outputs,class_num):
    """
    将任意模型的输出转换为常规输出
    [N,result]  N为样本数
    result: [C,[M,5]] 其中C为类别,M为bbox数目，5为左上角、右下角坐标与置信分数。
    并且根据尺度进行复原
    """
    if model_type == "Retinanet":
        print("deal with outputs in Retinanet type")
        results = []
        for output in outputs:
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
        print("deal with outputs in Retinanet-wo type")
        results = []
        for output in outputs:
            img_scale = output["scale_factor"] 
            input_size = output["img_shape"][:2] 
            cls_heads,reg_heads = output["output"][:5],output["output"][5:]
            result = [[] for _ in range(class_num)]
            
            scores, bboxs, labels = retinanet_postprocess(input_size,cls_heads,reg_heads)
            bboxs,labels = bboxs[0],labels[0].int()
            for i,label in enumerate(labels):
                bboxs[i] = bboxs[i] / img_scale  # 进行bbox的尺寸复原
                result[label].append(bboxs[i])
            results.append(result)
        return results