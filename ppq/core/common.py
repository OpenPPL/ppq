# PPQ 全局配置，你可以自由修改下列属性以达成特定目的。
# PPQ System configuration
# You can modify following codes for your own purpose.

# PPLCUDA 中所有需要与 Conv 融合的激活函数
PPLCUDA_ACTIVATIONS = {'Clip', 'LeakyRelu', 'Relu', 'Sigmoid'}

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
