import logging
from ppq.IR import Variable, Operation

logger = logging.getLogger(__name__)


def make_op(op_type, inputs=None, parameters=None, buffers=None, attributes=None, output_num=1, name=''):
    """
    :param op_type: Operation type
    :param inputs: the type is dict, given inputs information like {name(str): value(torch.tensor)}
    :param parameters: the type is dict, given parameters information like {name(str): value(torch.tensor)}
    :param buffers: the type is dict, given buffers information like {name(str): value(torch.tensor)}
    :param attributes: dict
    :param output_num: the number of the op's outputs
    :param name: Operation name
    :return: A new Operation instance
    """
    op = Operation(op_type=op_type, name=name, attributes=attributes)
    if inputs is not None:
        for name, value in inputs.items():
            var = Variable(dims=list(value.shape), value=value, data_type=value.dtype, name=name)
            op.add_input(var)

    if parameters is not None:
        for name, value in parameters.items():
            var = Variable(dims=list(value.shape), value=value, data_type=value.dtype, name=name)
            op.add_parameter(var)

    if buffers is not None:
        for name, value in buffers.items():
            var = Variable(dims=list(value.shape), value=value, data_type=value.dtype, name=name)
            op.add_buffer(var)

    for i in range(output_num):
        var = Variable(name='output_' + str(i))
        op.add_output(var)

    return op


def compare_variable(var_1, var_2, checked):
    if (var_1, var_2) in checked:
        return True

    # can not compare var.source or var.destinations directly because they are pointers
    flag = (var_1.dims == var_2.dims) and (var_1.data_type == var_2.data_type) and \
           (var_1.src_idx == var_2.src_idx) and (len(var_1.destinations) == len(var_2.destinations)) and \
           (bool(var_1.source) == bool(var_2.source))

    value_flag = True if var_1.value is None and var_2.value is None else (var_1.value == var_2.value).all()
    flag = flag and value_flag

    if not flag:
        return False

    if var_1.source is not None and var_2.source is not None:
        if not compare_operation(var_1.source, var_2.source, checked):
            return False
    elif var_1.source is not None or var_2.source is not None:
        return False

    checked.add((var_1, var_2))
    return True


def compare_operation(op_1, op_2, checked):
    flag = (op_1.type == op_2.type) and (len(op_1.inputs) == len(op_2.inputs)) and \
           (len(op_1.outputs) == len(op_2.outputs)) and (len(op_1.parameters) == len(op_2.parameters)) and \
           (len(op_1.buffers) == len(op_2.buffers)) and (op_1.attributes == op_2.attributes)

    # didn't compare outputs here
    if flag:
        var_flag = True
        input_var_1 = sorted([(var.dest_idx[var.destinations.index(op_1)], var) for var in
                              op_1.inputs + op_1.parameters + op_1.buffers])
        input_var_2 = sorted([(var.dest_idx[var.destinations.index(op_2)], var) for var in
                              op_2.inputs + op_2.parameters + op_2.buffers])
        # item generated is (0, <ppq.IR.graph.Variable>)
        for var_1, var_2 in zip(input_var_1, input_var_2):
            var_flag = var_flag and compare_variable(var_1[1], var_2[1], checked)
        flag = flag and var_flag

    return flag


def compare_var_set(var_list_1, var_list_2, checked):
    # Multi-var may not in the same order in two graph
    same_flag = (len(var_list_1) == len(var_list_2))

    if same_flag:
        flag = True
        matched = []
        for var_1 in var_list_1:
            for var_2 in var_list_2:
                if var_2 not in matched:
                    flag = compare_variable(var_1, var_2, checked)
                    if flag:
                        matched.append(var_2)
                        break
        same_flag = same_flag and flag
    return same_flag


def graph_similarity(graph_1, graph_2):
    """
    Check whether given ppq computational graphs have the same structures
    :param graph_1: PPQ ComputationalGraph instance
    :param graph_2: PPQ ComputationalGraph instance
    :return: True or False
    """
    # Basic check
    len_flag = (len(graph_1.inputs) == len(graph_2.inputs)) and (len(graph_1.outputs) == len(graph_2.outputs)) and \
               (len(graph_1.operations) == len(graph_2.operations)) and \
               (len(graph_1.variables) == len(graph_2.variables)) and \
               (len(graph_1.parameters) == len(graph_2.parameters)) and (len(graph_1.buffers) == len(graph_2.buffers))
    check_flag = 'Pass!' if len_flag else 'Fail!'
    logger.info(f'Check input ... {check_flag}')
    if not len_flag:
        return False

    if not graph_1.sorted:
        graph_1.topological_sort()
    if not graph_2.sorted:
        graph_2.topological_sort()

    # Check outputs:
    # Topology is checked recursively in compare_var_set func
    checked_var = set()
    output_same = compare_var_set(list(graph_1.outputs.values()), list(graph_2.outputs.values()), checked_var)
    check_flag = 'Pass!' if output_same else 'Fail!'
    logger.info(f'Check output ... {check_flag}')
    return output_same
