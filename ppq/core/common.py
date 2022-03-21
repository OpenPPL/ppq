# PPQ 全局配置，你可以自由修改下列属性以达成特定目的。
# PPQ System configuration
# You can modify following codes for your own purpose.


# Observer 中，最小 scale 限制，所有小于该值的 scale 将被该值覆盖
OBSERVER_MIN_SCALE = 1e-7
# Observer 中 kl 散度的计算设备
OBSERVER_KL_COMPUTING_DEVICE = 'cpu'
# Observer 中 hist 箱子的个数
OBSERVER_KL_HIST_BINS = 8192
# Observer 中 percentile 的参数
OBSERVER_PERCENTILE = 0.9999
# Observer 中 mse 校准方法 hist 箱子的个数
OBSERVER_MSE_HIST_BINS = 8192


# PPLCUDA 中所有需要与 Conv 融合的激活函数
PPLCUDA_ACTIVATIONS = {'Clip', 'LeakyRelu', 'Relu', 'Sigmoid'}

ORT_OOS_FUSE_START_OPS = {'Conv', 'GlobalAveragePool', 'AveragePool', 'Add', 'Mul', 'Matmul'}
ORT_MICROSOFT_CONTRIB_LINEAR_OPS = {'Add', 'Mul'}

# PASSIVE OPERATIONS 是那些不参与计算的 Op, 这些 op 的输入与输出将直接共享 scale
PASSIVE_OPERATIONS = {
    'Resize', 'MaxPool', 'GlobalMaxPool', 'Reshape',
    'Slice', 'Pad', 'Split', 'Transpose'}
# LINEAR ACTIVATIONS 是所有线性激活层，PPQ 将执行计算层与线性激活层的联合定点，不论后端是否真的做了图融合。
# 事实上就算后端不融合这些层，执行联合定点也是有益无害的。
LINEAR_ACTIVATIONS = {'Relu', 'Clip'}

# COPUTING OP 是所有计算层，该属性被用于联合定点和子图切分
COMPUTING_OP = {'Conv', 'Gemm', 'ConvTranspose'}
# SOI OP 是所有产生 SOI 输出的节点类型，该属性被用于子图切分
SOI_OP = {'TopK', 'Shape', 'NonMaxSuppression'}
# 强制联合定点的算子种类
COMPELING_OP_TYPES = {'Add',' Sub', 'Concat'}


# 要做 Bias Correction 的算子种类
BIAS_CORRECTION_INTERST_TYPE = {'Conv', 'Gemm', 'ConvTranspose'}

# Training Based Pass 抽样大小
NUM_OF_CHECKPOINT_FETCHS = 4096

# 误差容忍度
CHECKPOINT_TOLERANCE = 1

# SUB GRAPH 最大深度
# PPQ 使用子图切割算法寻找子图，这个参数控制了子图的大小。
OPTIM_ADVOPT_GRAPH_MAXDEPTH = 4
# ROUNDING LOSS 系数
OPTIM_ADVOPT_RLOSS_MULTIPLIER = 1
# 是否使用 LR SCEHDULER
OPTIM_ADVOPT_USING_SCEHDULER = True

# ONNX 导出图的时候，producer的名字
ONNX_EXPORT_NAME = 'PPL Quantization Tool - Onnx Export'
# ONNX 导出图的时候，opset的版本，这玩意改了可能就要起飞了
ONNX_EXPORT_OPSET = 11
# ONNX 导出图的时候，onnx version，这玩意改了可能就要起飞了
ONNX_VERSION = 6