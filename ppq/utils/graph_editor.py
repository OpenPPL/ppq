from typing import List
from ppq.IR import BaseGraph
from ppq.IR.morph import GraphFormatter


def truncate_graph(graph: BaseGraph, outputs: List[str]):
    """
    truncate your graph, so that all operations behind outputs(function parameter) will be eliminated.
        A list of output variable is given as parameter of this function.
        PPQ will goes forward from all those variables, mark all downstream opeartions for removing.
    
    A truncated graph object will return as result.
    
    ATTENTION: do not attempt to delete input variable.
    ATTETNION: you should invoke this function before quantization.
    
    Args:
        graph (BaseGraph): graph to be truncated
        outputs (List[str]): truncating from where

    Raises:
        KeyError: truncating variable is not in graph

    Returns:
        [type]: truncated graph
    """
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
        graph.delete_operation(op_name=op.name, cascade=True, force_delete=True)
    graph_formatter = GraphFormatter(graph)
    graph_formatter.delete_isolated()
    return graph