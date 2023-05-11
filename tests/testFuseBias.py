from ppq import BaseGraph, TorchExecutor
import torch
graph = BaseGraph(name='TestBed')

# Create a Gemm
gemm = graph.create_operation(
    op_type='Gemm', 
    inputs=[graph.create_variable()], 
    outputs=[graph.create_variable()]
)

graph.create_variable(
    value=torch.rand(size=[32, 32]), 
    is_parameter=True,
    dest_ops=[gemm])

add = graph.create_operation(
    op_type='Add', 
    inputs=[gemm.outputs[0]], 
    outputs=[graph.create_variable()])

graph.create_variable(
    value=torch.rand(size=[1, 32]), 
    is_parameter=True, 
    dest_ops=[add])

graph.mark_variable_as_graph_input(gemm.inputs[0])
graph.mark_variable_as_graph_output(add.outputs[0])

sample = torch.rand(size=[8, 32]).cuda()
executor = TorchExecutor(graph=graph)
ref_out = executor.forward(inputs=sample)[0]

from ppq.IR import GraphMerger
GraphMerger(graph).fuse_bias_add()
executor = TorchExecutor(graph=graph)
out = executor.forward(inputs=sample)[0]

for op in graph.operations.values():
    assert op.type != 'Add', 'Bias Add Fusion Failed.'
assert torch.max(ref_out - out).item() < 1e-7, 'Bias Add Fusion Failed.'

# Create a Gemm
graph = BaseGraph(name='TestBed')
gemm = graph.create_operation(
    op_type='Gemm', 
    inputs=[graph.create_variable()], 
    outputs=[graph.create_variable()]
)

graph.create_variable(
    value=torch.rand(size=[32, 32]), 
    is_parameter=True,
    dest_ops=[gemm])

add = graph.create_operation(
    op_type='Add', 
    inputs=[gemm.outputs[0]], 
    outputs=[graph.create_variable()])

graph.create_variable(
    value=torch.rand(size=[8, 32]), 
    is_parameter=True, 
    dest_ops=[add])

graph.mark_variable_as_graph_input(gemm.inputs[0])
graph.mark_variable_as_graph_output(add.outputs[0])

sample = torch.rand(size=[8, 32]).cuda()
executor = TorchExecutor(graph=graph)
ref_out = executor.forward(inputs=sample)[0]

from ppq.IR import GraphMerger
GraphMerger(graph).fuse_bias_add()
executor = TorchExecutor(graph=graph)
out = executor.forward(inputs=sample)[0]

for op in graph.operations.values():
    assert op.type != 'Add', 'Bias Add Fusion Failed.'
assert torch.max(ref_out - out).item() < 1e-7, 'Bias Add Fusion Failed.'

# Create a Conv
graph = BaseGraph(name='TestBed')
conv = graph.create_operation(
    op_type='Conv', attributes={
        'dilations': [1, 1], 
        'group': 1, 
        'kernel_shape': [3, 3], 
        'pads': [1, 1, 1, 1], 
        'strides': [1, 1]
    },
    inputs=[graph.create_variable()], 
    outputs=[graph.create_variable()]
)

graph.create_variable(
    value=torch.rand(size=[16, 3, 3, 3]), 
    is_parameter=True,
    dest_ops=[conv])

add = graph.create_operation(
    op_type='Add', 
    inputs=[conv.outputs[0]], 
    outputs=[graph.create_variable()])

graph.create_variable(
    value=torch.rand(size=[1, 16, 1, 1]), 
    is_parameter=True, 
    dest_ops=[add])

graph.mark_variable_as_graph_input(conv.inputs[0])
graph.mark_variable_as_graph_output(add.outputs[0])

sample = torch.rand(size=[1, 3, 32, 32]).cuda()
executor = TorchExecutor(graph=graph)
ref_out = executor.forward(inputs=sample)[0]

from ppq.IR import GraphMerger
GraphMerger(graph).fuse_bias_add()
executor = TorchExecutor(graph=graph)
out = executor.forward(inputs=sample)[0]

for op in graph.operations.values():
    assert op.type != 'Add', 'Bias Add Fusion Failed.'
assert torch.max(ref_out - out).item() < 1e-7, 'Bias Add Fusion Failed.'