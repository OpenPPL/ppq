from ppq.core import QuantizationStates
from ppq.IR import BaseGraph, GraphExporter
from ppq.IR.quantize import QuantableOperation


class ExtensionExporter(GraphExporter):
    """ExtensionExporter is an empty exporter for you to implement customized
    logic. rewrite function export in order to dump ppq graph to disk.

    use export_ppq_graph(..., platform=TargetPlatform.EXTENSION) to invoke this class.

    Args:
        GraphExporter ([type]): [description]
    """

    def __init__(self) -> None:
        super().__init__()

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None):
        """Sample Export Function -- export all quantization params into txt"""
        
        if config_path is None:
            raise ValueError('Can not export ppq quantization params, cause configuration path is empty.')
        with open(file=config_path, mode='w') as file:
        
            for op in graph.operations.values():
                if not isinstance(op, QuantableOperation): continue

                for cfg, var in op.config_with_variable:
                    if QuantizationStates.can_export(cfg.state):
                        
                        pass
