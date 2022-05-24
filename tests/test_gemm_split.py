from ppq import *
from ppq.IR.morph import GraphDecomposer
from ppq.api import *
import torch

graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Gemm', name='gemm', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(), graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10]))],
                       outputs=[graph.create_variable()])
graph.create_operation(op_type='Softmax', name='softmax', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                       outputs=[graph.create_variable()])
processor = GraphDecomposer(graph)
processor.decompose_gemm()

assert len(graph.operations) == 2
assert len(graph.operations['gemm'].inputs) == 2
assert graph.operations['gemm'].type == 'Matmul'

graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Gemm', name='gemm', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(), 
                               graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10])),
                               graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10]))],
                       outputs=[graph.create_variable()])
graph.create_operation(op_type='Softmax', name='softmax', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                       outputs=[graph.create_variable()])
processor = GraphDecomposer(graph)
processor.decompose_gemm()

assert len(graph.operations) == 3
assert len(graph.operations['gemm'].inputs) == 2
assert graph.operations['gemm'].type == 'Matmul'


graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Gemm', name='gemm', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(),
                               graph.create_variable(is_parameter=True, value=torch.ones(size=[10, 10])),
                               graph.create_variable(is_parameter=True, value=torch.ones(size=[10, 10]))],
                       attributes={'alpha': 2, 'beta': 3},
                       outputs=[graph.create_variable()])
graph.create_operation(op_type='Softmax', name='softmax', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                       outputs=[graph.create_variable()])
processor = GraphDecomposer(graph)
processor.decompose_gemm()

assert len(graph.operations) == 3
assert len(graph.operations['gemm'].inputs) == 2
assert graph.operations['gemm'].type == 'Matmul'
assert graph.operations['gemm'].inputs[1].value.mean().item() == 2


graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Gemm', name='gemm', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(),
                               graph.create_variable(is_parameter=True, value=torch.ones(size=[10, 10])),
                               graph.create_variable(is_parameter=True, value=torch.ones(size=[10, 10]))],
                       attributes={'transA': 0, 'transB': 1},
                       outputs=[graph.create_variable()])
graph.create_operation(op_type='Softmax', name='softmax', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                       outputs=[graph.create_variable()])
processor = GraphDecomposer(graph)
processor.decompose_gemm()

assert len(graph.operations) == 3
assert len(graph.operations['gemm'].inputs) == 2
assert graph.operations['gemm'].type == 'Matmul'

try:
    graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
    matmul = \
    graph.create_operation(op_type='Gemm', name='gemm', 
                        platform=TargetPlatform.UNSPECIFIED, 
                        inputs=[graph.create_variable(),
                                graph.create_variable(is_parameter=True, value=torch.ones(size=[10, 10])),
                                graph.create_variable(is_parameter=True, value=torch.ones(size=[10, 10]))],
                        attributes={'transA': 1, 'transB': 0},
                        outputs=[graph.create_variable()])
    graph.create_operation(op_type='Softmax', name='softmax', platform=TargetPlatform.UNSPECIFIED, 
                        inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                        outputs=[graph.create_variable()])
    processor = GraphDecomposer(graph)
    processor.decompose_gemm()
except ValueError as e:
    pass