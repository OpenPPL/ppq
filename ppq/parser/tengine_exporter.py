import os
from typing import List

from ppq.core import (DataType, NetworkFramework, QuantizationProperty,
                      QuantizationStates)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation
from .util import convert_value



class TengineExportUtils():
    pass




class TengineExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        pass

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        pass