# 是否启动 cuda kernel 加速计算
USING_CUDA_KERNEL = False
if USING_CUDA_KERNEL:
    import os
    # 这玩意我也不知道是干嘛的，但是 cuda 报错了它就总是让我设置这个为 1
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 开启 PPQ 调试模式，将打印所有量化点插入信息
PPQ_DEBUG = False

# PPQ 的名字
PPQ_NAME = 'PPL Quantization Tool'

# PPQ 的版本号
PPQ_VERSION = '0.6.4'

# 导出图时是否导出权重（仅影响 Native 格式导出）
DUMP_VALUE = True

# 导出图时，是否导出 Device Switcher
EXPORT_DEVICE_SWITCHER = False

# 导出图时，是否导出调度信息
EXPORT_PPQ_INTERNAL_INFO = False
