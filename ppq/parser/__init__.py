from ppq.core import NetworkFramework, TargetPlatform, ppq_warning
from ppq.IR import BaseGraph, GraphBuilder, GraphExporter

from .caffe_exporter import (CaffeExporter, PPLDSPCaffeExporter,
                             SNPECaffeExporter, PPLDSPTICaffeExporter)
from .caffe_parser import CaffeParser
from .extension import ExtensionExporter
from .native import NativeExporter, NativeImporter
from .nxp_exporter import NxpExporter
from .onnx_exporter import OnnxExporter
from .onnx_parser import OnnxParser
from .onnxruntime_exporter import ONNXRUNTIMExporter
from .onnxruntime_oos_exporter import ORTOOSExporter
from .ppl import PPLBackendExporter

'''
PARSERS = {
    NetworkFramework.ONNX: OnnxParser,
    NetworkFramework.CAFFE: CaffeParser,
    NetworkFramework.NATIVE: NativeImporter
}

EXPORTERS = {
    TargetPlatform.PPL_DSP_INT8:  PPLDSPCaffeExporter,
    TargetPlatform.PPL_CUDA_INT8: PPLBackendExporter,
    TargetPlatform.SNPE_INT8:     SNPECaffeExporter,
    TargetPlatform.NXP_INT8:      NxpExporter,
    TargetPlatform.ONNX:          OnnxExporter,
    TargetPlatform.ONNXRUNTIME:   ONNXRUNTIMExporter,
    TargetPlatform.CAFFE:         CaffeExporter,
    TargetPlatform.NATIVE:        NativeExporter,
    TargetPlatform.EXTENSION:     ExtensionExporter,
    # TargetPlatform.ORT_OOS_INT8:  ONNXRUNTIMExporter,
    TargetPlatform.ORT_OOS_INT8:  ORTOOSExporter,
}
try:
    from .tensorRT import TensorRTExporter
    EXPORTERS[TargetPlatform.TRT_INT8] = TensorRTExporter
except ImportError as e:
    ppq_warning('Since PPQ can not found a tensorRT installation, tensorRT export has been deactived.')


def load_graph(file_path: str, from_framework: NetworkFramework=NetworkFramework.ONNX, **kwargs) -> BaseGraph:
    if from_framework not in PARSERS:
        raise KeyError(f'Requiring framework {from_framework} does not support parsing now.')
    parser = PARSERS[from_framework]()
    assert isinstance(parser, GraphBuilder), 'Unexpected Parser found.'
    if from_framework == NetworkFramework.CAFFE:
        assert 'caffemodel_path' in kwargs, ('parameter "caffemodel_path" is required here for loading caffe model from file, '
                                             'however it is missing from your invoking.')
        graph = parser.build(prototxt_path=file_path, caffemodel_path=kwargs['caffemodel_path'])
    else:
        graph = parser.build(file_path)
    return graph

def dump_graph_to_file(file_path: str, config_path: str, target_platform: TargetPlatform, graph: BaseGraph, **kwargs) -> None:
    if target_platform not in EXPORTERS:
        raise KeyError(f'Requiring framework {target_platform} does not support export now.')
    exporter = EXPORTERS[target_platform]()
    assert isinstance(exporter, GraphExporter), 'Unexpected Exporter found.'
    exporter.export(file_path=file_path, config_path=config_path, graph=graph, **kwargs)
'''