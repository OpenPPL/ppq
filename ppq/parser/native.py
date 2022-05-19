from pickle import dump, load

from ppq.core import PPQ_CONFIG
from ppq.IR import BaseGraph, GraphExporter
from ppq.IR.base.graph import GraphBuilder


class NativeExporter(GraphExporter):
    def __init__(self) -> None:
        super().__init__()
    def export(self, file_path: str, graph: BaseGraph,
               config_path: str = None, dump_value: bool = True):
        def dump_elements_to_file(file, elements: list):
            for element in elements: dump(element, file)

        with open(file_path, 'wb') as file:
            dump_elements_to_file(file, elements=[
                'PPQ GRAPH DEFINITION', # PPQ Signature.
                PPQ_CONFIG.VERSION,            # PPQ Signature.
                graph,
            ])

class NativeImporter(GraphBuilder):
    def __init__(self) -> None:
        super().__init__()

    def build(self, file_path: str, **kwargs) -> BaseGraph:
        def load_elements_from_file(file, num_of_elements: int) -> list:
            try: return [load(file) for _ in range(num_of_elements)]
            except EOFError as e:
                raise Exception('File format parsing error. Unexpected EOF found.')

        with open(file_path, 'rb') as file:
            signature, version, graph = load_elements_from_file(file, 3)
            if signature != 'PPQ GRAPH DEFINITION':
                raise Exception('File format parsing error. Graph Signature has been damaged.')
            if str(version) > PPQ_CONFIG.VERSION:
                print(f'\033[31mWarning: Dump file is created by PPQ({str(version)}), '
                f'however you are using PPQ({PPQ_CONFIG.VERSION}).\033[0m')

        assert isinstance(graph, BaseGraph), (
            'File format parsing error. Graph Definition has been damaged.')
        try:
            for op in graph.operations.values():
                input_copy, _ = op.inputs.copy(), op.inputs.clear()
                for name in input_copy: op.inputs.append(graph.variables[name])
                output_copy, _ = op.outputs.copy(), op.outputs.clear()
                for name in output_copy: op.outputs.append(graph.variables[name])

            for var in graph.variables.values():
                dest_copy, _ = var.dest_ops.copy(), var.dest_ops.clear()
                for name in dest_copy: var.dest_ops.append(graph.operations[name])
                if var.source_op is not None:
                    var.source_op = graph.operations[var.source_op]

            graph._graph_inputs = {name: graph.variables[name] for name in graph._graph_inputs}
            graph._graph_outputs = {name: graph.variables[name] for name in graph._graph_outputs}
        except Exception as e:
            raise Exception('File format parsing error. Graph Definition has been damaged.')
        return graph
