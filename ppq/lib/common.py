from ppq.core import NetworkFramework, TargetPlatform
from ppq.parser import (AscendExporter, CaffeExporter, CaffeParser,
                        ExtensionExporter, NativeExporter, NativeImporter,
                        NCNNExporter, NxpExporter, OnnxExporter, OnnxParser,
                        ONNXRUNTIMExporter, PPLBackendExporter,
                        PPLDSPCaffeExporter, PPLDSPTICaffeExporter,
                        QNNDSPExporter, SNPECaffeExporter, TengineExporter,
                        TensorRTExporter_JSON, TensorRTExporter_QDQ)
from ppq.quantization.quantizer import (AscendQuantizer, ExtQuantizer,
                                        FPGAQuantizer, GraphCoreQuantizer,
                                        MetaxChannelwiseQuantizer,
                                        MetaxTensorwiseQuantizer,
                                        NCNNQuantizer, NXP_Quantizer,
                                        OnnxruntimeQuantizer,
                                        OpenvinoQuantizer, PPL_DSP_Quantizer,
                                        PPL_DSP_TI_Quantizer, PPLCUDAQuantizer,
                                        RKNN_PerTensorQuantizer,
                                        TengineQuantizer, TensorRTQuantizer,
                                        TensorRTQuantizer_FP8)

__QUANTIZER_COLLECTION__ = {
    TargetPlatform.PPL_DSP_INT8: PPL_DSP_Quantizer,
    TargetPlatform.PPL_DSP_TI_INT8: PPL_DSP_TI_Quantizer,
    TargetPlatform.SNPE_INT8:    PPL_DSP_Quantizer,
    TargetPlatform.QNN_DSP_INT8: PPL_DSP_Quantizer,
    TargetPlatform.TRT_INT8:     TensorRTQuantizer,
    TargetPlatform.ASC_INT8:     AscendQuantizer,
    TargetPlatform.NCNN_INT8:    NCNNQuantizer,
    TargetPlatform.NXP_INT8:     NXP_Quantizer,
    TargetPlatform.RKNN_INT8:    RKNN_PerTensorQuantizer,
    TargetPlatform.METAX_INT8_C: MetaxChannelwiseQuantizer,
    TargetPlatform.METAX_INT8_T: MetaxTensorwiseQuantizer,
    TargetPlatform.PPL_CUDA_INT8: PPLCUDAQuantizer,
    TargetPlatform.EXTENSION:     ExtQuantizer,
    TargetPlatform.FPGA_INT8   :  FPGAQuantizer,
    TargetPlatform.OPENVINO_INT8: OpenvinoQuantizer,
    TargetPlatform.TENGINE_INT8:  TengineQuantizer,
    TargetPlatform.GRAPHCORE_FP8: GraphCoreQuantizer,
    TargetPlatform.TRT_FP8:       TensorRTQuantizer_FP8,
    TargetPlatform.ONNXRUNTIME:   OnnxruntimeQuantizer,
}


__PARSERS__ = {
    NetworkFramework.ONNX: OnnxParser,
    NetworkFramework.CAFFE: CaffeParser,
    NetworkFramework.NATIVE: NativeImporter
}


__EXPORTERS__ = {
    TargetPlatform.PPL_DSP_INT8:  PPLDSPCaffeExporter,
    TargetPlatform.PPL_DSP_TI_INT8: PPLDSPTICaffeExporter,
    TargetPlatform.QNN_DSP_INT8:  QNNDSPExporter,
    TargetPlatform.PPL_CUDA_INT8: PPLBackendExporter,
    TargetPlatform.SNPE_INT8:     SNPECaffeExporter,
    TargetPlatform.NXP_INT8:      NxpExporter,
    TargetPlatform.ONNX:          OnnxExporter,
    TargetPlatform.ONNXRUNTIME:   ONNXRUNTIMExporter,
    TargetPlatform.OPENVINO_INT8: ONNXRUNTIMExporter,
    TargetPlatform.CAFFE:         CaffeExporter,
    TargetPlatform.NATIVE:        NativeExporter,
    TargetPlatform.EXTENSION:     ExtensionExporter,
    TargetPlatform.RKNN_INT8:     OnnxExporter,
    TargetPlatform.METAX_INT8_C:  ONNXRUNTIMExporter,
    TargetPlatform.METAX_INT8_T:  ONNXRUNTIMExporter,
    TargetPlatform.TRT_INT8:      TensorRTExporter_JSON,
    TargetPlatform.ASC_INT8:      AscendExporter,
    TargetPlatform.TRT_FP8:       ONNXRUNTIMExporter,
    TargetPlatform.NCNN_INT8:     NCNNExporter,
    TargetPlatform.TENGINE_INT8:  TengineExporter
}


__all__ = ['__QUANTIZER_COLLECTION__', '__PARSERS__', '__EXPORTERS__']
