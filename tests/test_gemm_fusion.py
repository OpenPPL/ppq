from ppq import *
from ppq.IR.morph import GraphMerger
from ppq.api import *
import torch

graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Matmul', name='matmul', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(), graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10]))],
                       outputs=[graph.create_variable()])
graph.create_operation(op_type='Add', name='add', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                       outputs=[graph.create_variable()])
processor = GraphMerger(graph)
processor.fuse_gemm()

assert len(graph.operations) == 1
assert len(graph.operations['matmul'].inputs) == 3
assert graph.operations['matmul'].type == 'Gemm'

graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Matmul', name='matmul', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(), graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10]))],
                       outputs=[graph.create_variable()])
test = \
graph.create_operation(op_type='Test', name='test', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[], outputs=[graph.create_variable()])
graph.create_operation(op_type='Add', name='add', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], test.outputs[0]], 
                       outputs=[graph.create_variable()])
processor = GraphMerger(graph)
processor.fuse_gemm()

assert len(graph.operations) == 3
assert len(graph.operations['matmul'].inputs) == 2
assert graph.operations['matmul'].type == 'Gemm'


graph = BaseGraph(name='test', built_from=NetworkFramework.ONNX)
matmul = \
graph.create_operation(op_type='Matmul', name='matmul', 
                       platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[graph.create_variable(), graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, 10]))],
                       outputs=[graph.create_variable()])
graph.create_operation(op_type='Add', name='add', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[1, ]))], 
                       outputs=[graph.create_variable()])
processor = GraphMerger(graph)
processor.fuse_gemm()

assert len(graph.operations) == 2
assert len(graph.operations['matmul'].inputs) == 2
assert graph.operations['matmul'].type == 'Gemm'