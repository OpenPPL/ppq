from ppq import *

class PPQTestScheme():
    def __init__(
        self, name: str, quant_platform: TargetPlatform, 
        export_platform: TargetPlatform, setting: QuantizationSetting):
        self.name = name
        self.quant_platform  = quant_platform
        self.export_platform = export_platform
        self.setting = setting

TEST_SCHEMES = [
    PPQTestScheme(
        name = 'Tengine',
        quant_platform=TargetPlatform.TENGINE_INT8, 
        export_platform=TargetPlatform.TENGINE_INT8, 
        setting=QuantizationSettingFactory.pplcuda_setting()),

    PPQTestScheme(
        name = 'TRT FP8',
        quant_platform=TargetPlatform.TRT_FP8, 
        export_platform=TargetPlatform.TRT_FP8, 
        setting=QuantizationSettingFactory.default_setting()),

    PPQTestScheme(
        name = 'TRT INT8',
        quant_platform=TargetPlatform.TRT_INT8, 
        export_platform=TargetPlatform.TRT_INT8, 
        setting=QuantizationSettingFactory.default_setting()),
    
    PPQTestScheme(
        name = 'Sensetime Caffe[DSP INT8]',
        quant_platform=TargetPlatform.PPL_DSP_INT8, 
        export_platform=TargetPlatform.PPL_DSP_INT8, 
        setting=QuantizationSettingFactory.dsp_setting()),
    
    PPQTestScheme(
        name = 'Sensetime Caffe[DSP INT8]',
        quant_platform=TargetPlatform.SNPE_INT8, 
        export_platform=TargetPlatform.SNPE_INT8, 
        setting=QuantizationSettingFactory.dsp_setting()),

    PPQTestScheme(
        name = 'Sensetime PPL[GPU INT8]',
        quant_platform=TargetPlatform.PPL_CUDA_INT8, 
        export_platform=TargetPlatform.PPL_CUDA_INT8, 
        setting=QuantizationSettingFactory.pplcuda_setting()),

    PPQTestScheme(
        name = 'Sensetime PPL[GPU INT8 - ONNX RUNTIME EXPORT]',
        quant_platform=TargetPlatform.PPL_CUDA_INT8, 
        export_platform=TargetPlatform.ONNXRUNTIME, 
        setting=QuantizationSettingFactory.pplcuda_setting()),
    
    PPQTestScheme(
        name = 'ONNX RUNTIME[MetaX INT8]',
        quant_platform=TargetPlatform.METAX_INT8_T, 
        export_platform=TargetPlatform.ONNXRUNTIME, 
        setting=QuantizationSettingFactory.pplcuda_setting()),

    PPQTestScheme(
        name = 'ONNX RUNTIME[MetaX INT8 Channelwise]',
        quant_platform=TargetPlatform.METAX_INT8_C, 
        export_platform=TargetPlatform.ONNXRUNTIME, 
        setting=QuantizationSettingFactory.pplcuda_setting()),

    PPQTestScheme(
        name = 'ONNX RUNTIME OP ORITNETD[INT8]',
        quant_platform=TargetPlatform.RKNN_INT8, 
        export_platform=TargetPlatform.RKNN_INT8, 
        setting=QuantizationSettingFactory.pplcuda_setting()),

    PPQTestScheme(
        name = 'NXP [NXP INT8]',
        quant_platform=TargetPlatform.NXP_INT8, 
        export_platform=TargetPlatform.NXP_INT8, 
        setting=QuantizationSettingFactory.nxp_setting()),

    PPQTestScheme(
        name = 'Native',
        quant_platform=TargetPlatform.PPL_CUDA_INT8, 
        export_platform=TargetPlatform.NATIVE, 
        setting=QuantizationSettingFactory.pplcuda_setting()),
    
]