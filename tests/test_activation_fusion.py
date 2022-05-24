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
graph.create_operation(op_type='Relu', name='relu', platform=TargetPlatform.UNSPECIFIED, 
                       inputs=[matmul.outputs[0], graph.create_variable(is_parameter=True, value=torch.zeros(size=[10, ]))], 
                       outputs=[graph.create_variable()])
processor = QuantableGraph(graph)
processor.quantize_operation('matmul', target_platform=TargetPlatform.PPL_CUDA_INT8)
processor.quantize_operation('relu', target_platform=TargetPlatform.PPL_CUDA_INT8)
