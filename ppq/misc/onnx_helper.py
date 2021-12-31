import logging

logger = logging.getLogger(__name__)


def onnx_similarity(model_1, model_2):
    """
    Check the similarity of two onnx model
    Only check the model's attributes and some simple topology, not the strict checking
    Did not check the value of initializers
    The name will not be changed in onnx model, so this func also check the input name
    :param model_1: onnx model
    :param model_2: onnx model
    :return: bool
    """
    graph_1 = model_1.graph
    graph_2 = model_2.graph

    # Check input
    input_same = (len(graph_1.input) == len(graph_2.input))
    if input_same:
        input_1 = {item_1.name: item_1.type for item_1 in graph_1.input}
        input_2 = {item_2.name: item_2.type for item_2 in graph_2.input}
        for name_1, type_1 in input_1.items():
            if name_1 not in input_2:
                input_same = False
            else:
                input_same = input_same and (input_2[name_1] == type_1)
            if not input_same:
                break

    check_flag = 'Pass!' if input_same else 'Fail!'
    logger.info(f'Check input ... {check_flag}')

    # Check node
    node_same = (len(graph_1.node) == len(graph_2.node))
    if node_same:
        for node_1, node_2 in zip(graph_1.node, graph_2.node):
            node_same = node_same and (node_1.op_type == node_2.op_type)
            node_same = node_same and (len(node_1.input) == len(node_2.input))
            node_same = node_same and (len(node_1.output) == len(node_2.output))
            node_same = node_same and (node_1.attribute == node_2.attribute)
            if not node_same:
                break

    check_flag = 'Pass!' if node_same else 'Fail!'
    logger.info(f'Check node ... {check_flag}')

    # Check output
    output_same = (len(graph_1.output) == len(graph_2.output))
    if output_same:
        for item_1, item_2 in zip(graph_1.output, graph_2.output):
            output_same = output_same and (item_1.type == item_2.type)
            if not output_same:
                break

    check_flag = 'Pass!' if output_same else 'Fail!'
    logger.info(f'Check output ... {check_flag}')

    # Check length of initializer
    init_same = (len(graph_1.initializer) == len(graph_2.initializer))

    check_flag = 'Pass!' if init_same else 'Fail!'
    logger.info(f'Check length of initializer ... {check_flag}')

    return input_same and output_same and node_same and init_same
