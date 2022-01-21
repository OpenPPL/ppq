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

# PASSIVE OPERATIONS 是那些不参与计算的 Op, 这些 op 的输入与输出将直接共享 scale
PASSIVE_OPERATIONS = {
    'Resize', 'MaxPool', 'GlobalMaxPool', 'Reshape',
    'Slice', 'Pad', 'Split', 'Transpose', 'Clip'}

# LINEAR ACTIVATIONS 是所有线性激活层，PPQ 将执行计算层与线性激活层的联合定点，不论后端是否真的做了图融合。
# 事实上就算后端不融合这些层，执行联合定点也是有益无害的。
LINEAR_ACTIVATIONS = {
    'Relu', 'Clip',
}

# COPUTING OP 是所有计算层，该属性被用于联合定点和子图切分
COMPUTING_OP = {'Conv', 'Gemm', 'ConvTranspose'}

# SOI OP 是所有产生 SOI 输出的节点类型，该属性被用于子图切分
SOI_OP = {'TopK', 'Shape', 'NonMaxSuppression'}

# 强制联合定点的算子种类
COMPELING_OP_TYPES = {'Add',' Sub', 'Concat'}

# 要做 Bias Correction 的算子种类
BIAS_CORRECTION_INTERST_TYPE = {'Conv', 'Gemm', 'ConvTranspose'}

# Training Based Pass 抽样大小
NUM_OF_CHECKPOINT_FETCHS = 2048

# 误差容忍度
CHECKPOINT_TOLERANCE = 1

# SUB GRAPH 中最多计算算子的个数
OPTIM_ADVOPT_GRAPH_MAXSIZE  = 4
OPTIM_ADVOPT_THRESHOLD_STEP = 0.1
OPTIM_ADVOPT_STEP_PER_EPOCH = 32
OPTIM_ADVOPT_PASSIVE_BOOST  = 32
OPTIM_ADVOPT_PATIENT        = 5
OPTIM_ADVOPT_INITIAL_THRES  = 0.99