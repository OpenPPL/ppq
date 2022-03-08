from ppq.IR import BaseGraph, GraphExporter


class ExtensionExporter(GraphExporter):
    """
    ExtensionExporter is an empty exporter for you to implement custimized logic.
        rewrite function export in order to dump ppq graph to disk.
    
    use export_ppq_graph(..., platform=TargetPlatform.EXTENSION) to invoke this class.

    Args:
        GraphExporter ([type]): [description]
    """
    
    def __init__(self) -> None:
        super().__init__()

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None):
        """
        Write cusimized logic for dumping ppq graph.

        Args:
            file_path (str): [description]
            graph (BaseGraph): [description]
            config_path (str, optional): [description]. Defaults to None.
        """
        print('You are using Extension Exporter now, however there has no logic yet, so i just print this.')
        print('你调用了 Extension Exporter，但你很可能还没有在这里写任何逻辑，所以我就打印了这行信息。')
