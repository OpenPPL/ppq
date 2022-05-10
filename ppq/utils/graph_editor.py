from typing import List
from ppq.IR import BaseGraph
from ppq.IR import GraphFormatter


def truncate_graph(graph: BaseGraph, outputs: List[str]):
    """truncate your graph, so that all operations behind outputs(function
    parameter) will be eliminated. A list of output variable is given as
    parameter of this function. PPQ will goes forward from all those variables,
    mark all downstream operations for removing.

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
    for output in outputs:
        if output not in graph.variables:
            raise KeyError(f'Can not find variable {output} in current graph.')
    processor = GraphFormatter(graph)

    for output in outputs:
        output_var = graph.variables[output]
        processor.truncate_on_var(output_var, mark_as_output=True)
    processor.delete_isolated()
    return graph
