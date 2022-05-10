from ppq import *
from ppq.IR.morph import GraphFormatter
from ppq.api.interface import export_ppq_graph

# TEST CASE 1
graph = BaseGraph(name='Graph', built_from=NetworkFramework.ONNX)
graph.append_operation(operation=Operation(name='op1', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op2', op_type='Conv', attributes={}))
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op1'],
                          downstream_op=graph.operations['op2'])

processor = SearchableGraph(graph)
paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: False,
    ep_expr=lambda x: x.type == 'Conv',
    direction='down')

assert len(paths) == 1
path = paths[0]
assert path[0].name == 'op1'
assert path[1].name == 'op2'
assert path[-1].name == 'op2'
export_ppq_graph(graph=graph, platform=TargetPlatform.ONNX, graph_save_to='graph')

# TEST CASE 2
graph = BaseGraph(name='Graph', built_from=NetworkFramework.ONNX)
graph.append_operation(operation=Operation(name='op1', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op2', op_type='Add', attributes={}))
graph.append_operation(operation=Operation(name='op3', op_type='Relu', attributes={}))
graph.append_operation(operation=Operation(name='op4', op_type='Sigmoid', attributes={}))
graph.append_operation(operation=Operation(name='op5', op_type='Add', attributes={}))
graph.append_operation(operation=Operation(name='op6', op_type='Relu', attributes={}))
graph.append_operation(operation=Operation(name='op7', op_type='Sigmoid', attributes={}))
graph.append_operation(operation=Operation(name='op8', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op9', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op10', op_type='Conv', attributes={}))
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op1'],
                          downstream_op=graph.operations['op2'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op8'],
                          downstream_op=graph.operations['op2'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op2'],
                          downstream_op=graph.operations['op3'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op3'],
                          downstream_op=graph.operations['op4'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op9'],
                          downstream_op=graph.operations['op5'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op10'],
                          downstream_op=graph.operations['op5'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op5'],
                          downstream_op=graph.operations['op6'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op6'],
                          downstream_op=graph.operations['op7'])
processor = SearchableGraph(graph)
paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Sigmoid',
    direction='down')

assert len(paths) == 4
path = paths[0]
assert path[0].type == 'Conv'
assert path[1].type == 'Add'
assert path[-1].type == 'Sigmoid'

paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Relu',
    direction='down')

assert len(paths) == 4
path = paths[0]
assert path[0].type == 'Conv'
assert path[1].type == 'Add'
assert path[-1].type == 'Relu'

paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: False,
    ep_expr=lambda x: x.type == 'Relu',
    direction='down')
assert len(paths) == 0

graph.remove_operation(graph.operations['op1'])
paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Sigmoid',
    direction='down')
assert len(paths) == 3
path = paths[0]
assert path[0].type == 'Conv'
assert path[1].type == 'Add'
assert path[-1].type == 'Sigmoid'

opset = processor.opset_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Sigmoid',
    direction='down')
assert len(opset) == 9

graph.remove_operation(graph.operations['op2'])
graph.remove_operation(graph.operations['op3'])
graph.remove_operation(graph.operations['op4'])
opset = processor.opset_matching(
    sp_expr=lambda x: x.type == 'Conv',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Sigmoid',
    direction='down')
assert len(opset) == 5

processor = GraphFormatter(graph)
processor.delete_isolated()
assert len(graph.operations) == 0
export_ppq_graph(graph=graph, platform=TargetPlatform.ONNX, graph_save_to='graph')

# TEST CASE 3
graph = BaseGraph(name='Graph', built_from=NetworkFramework.ONNX)
graph.append_operation(operation=Operation(name='op1', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op2', op_type='Add', attributes={}))
graph.append_operation(operation=Operation(name='op3', op_type='Relu', attributes={}))
graph.append_operation(operation=Operation(name='op4', op_type='Sigmoid', attributes={}))
graph.append_operation(operation=Operation(name='op5', op_type='Add', attributes={}))
graph.append_operation(operation=Operation(name='op6', op_type='Relu', attributes={}))
graph.append_operation(operation=Operation(name='op7', op_type='Sigmoid', attributes={}))
graph.append_operation(operation=Operation(name='op8', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op9', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op10', op_type='Conv', attributes={}))
graph.append_operation(operation=Operation(name='op11', op_type='Conv', attributes={}))
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op1'],
                          downstream_op=graph.operations['op2'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op8'],
                          downstream_op=graph.operations['op2'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op2'],
                          downstream_op=graph.operations['op3'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op3'],
                          downstream_op=graph.operations['op4'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op9'],
                          downstream_op=graph.operations['op5'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op10'],
                          downstream_op=graph.operations['op5'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op5'],
                          downstream_op=graph.operations['op6'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op6'],
                          downstream_op=graph.operations['op7'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op6'],
                          downstream_op=graph.operations['op7'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op11'],
                          downstream_op=graph.operations['op1'])
graph.create_link_with_op(variable=graph.create_variable(),
                          upstream_op=graph.operations['op11'],
                          downstream_op=graph.operations['op10'])
processor = GraphFormatter(graph)
graph.mark_variable_as_graph_output(var=graph.operations['op6'].outputs[0])
processor.truncate_on_var(graph.operations['op1'].outputs[0], mark_as_output=True)
processor.delete_isolated()
assert len(graph.outputs) == 2
assert len(graph.operations) == 6

for var in graph.variables.copy():
    graph.insert_op_on_var(graph.create_operation(op_type='Test_1', attributes={}), var=var)
for var in graph.variables.copy():
    graph.insert_op_on_var(graph.create_operation(op_type='Test_2', attributes={}), var=var)

export_ppq_graph(graph=graph, platform=TargetPlatform.ONNX, graph_save_to='graph')
processor = SearchableGraph(graph)
paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Test_1',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Test_2',
    direction='down')
assert len(paths) == 8

paths = processor.path_matching(
    sp_expr=lambda x: x.type == 'Test_1',
    rp_expr=lambda x, y: True,
    ep_expr=lambda x: x.type == 'Test_2',
    direction='up')
assert len(paths) == 8
