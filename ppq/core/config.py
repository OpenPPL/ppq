class PPQ_GLOBAL_CONFIGURATION:
    def __init__(self) -> None:
        # 是否启动 cuda kernel 加速计算
        self.USING_CUDA_KERNEL        = False
        
        # PPQ 的名字
        self.NAME                     = 'PPL Quantization Tool'
        
        # PPQ 的版本号
        self.VERSION                  = '0.6.6'
        
        # 导出图时是否导出权重（仅影响 Native 格式导出）
        self.DUMP_VALUE_WHEN_EXPORT   = True

        # 导出图时，是否导出调度信息
        self.EXPORT_PPQ_INTERNAL_INFO = False
        
        # 开启 PPQ 调试模式，将打印所有量化点插入信息
        self.PPQ_DEBUG                = False

PPQ_CONFIG = PPQ_GLOBAL_CONFIGURATION()  
