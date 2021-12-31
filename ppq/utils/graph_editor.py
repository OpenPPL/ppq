from typing import List
from ppq.IR import BaseGraph
from ppq.IR.morph import GraphFormatter


def truncate_graph(graph: BaseGraph, outputs: List[str]):
    graph.outputs.clear()
    for name in outputs:
        if name not in graph.variables:
            raise KeyError(f'Output variable name {name} is not included in current graph.')
        graph.outputs[name] = graph.variables[name]

    delete_set = set()
    for output_var in graph.outputs.values():
        for dest_op in output_var.dest_ops:
            delete_set.add(dest_op)

    for op in delete_set:
        graph.delete_operation(op_name=op.name, cascade=True)
    graph_formatter = GraphFormatter(graph)
    graph_formatter.delete_isolated()
    return graph