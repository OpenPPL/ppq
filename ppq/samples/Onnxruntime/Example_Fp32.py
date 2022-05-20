import onnxruntime
import numpy as np 

# -------------------------------------------------------------------
# Onnxruntime 需要你提供一个 feed dict 和 output 的名字才能跑推理
# feed dict 就是 input name: data 的形式表示的输入数据
# output name 和 input name 你如果不知道的话，用可视化工具打开 onnx 文件就可以看到了。
# -------------------------------------------------------------------

MODEL        = 'model.onnx'
FEED_DICT    = {'input name': np.zeros(shape=[1, 3, 224, 224])}
OUTPUT_NAMES = ['output name']

session = onnxruntime.InferenceSession(MODEL, providers=['CUDAExecutionProvider'])
result = session.run(OUTPUT_NAMES, FEED_DICT)
