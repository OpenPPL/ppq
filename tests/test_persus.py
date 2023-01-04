from ppq import BaseGraph
from ppq.core.quant import NetworkFramework
from ppq.scheduler.core.opsocket import OType
from ppq.scheduler.perseus import Perseus

graph = BaseGraph(name='TestCase1', built_from=NetworkFramework.ONNX)
gemm = graph.create_operation(op_type='Gemm', name='FirstGemm')
conv = graph.create_operation(op_type='Conv', name='Conv')
# g1.create_operation(op_type='Shape')
sigmoid = graph.create_operation(op_type='Sigmoid', name='Sigmoid')
shape1  = graph.create_operation(op_type='Shape', name='Shape1')
relu    = graph.create_operation(op_type='Relu', name='Relu')
gemm2   = graph.create_operation(op_type='Gemm', name='ShapeGemm')
shape2  = graph.create_operation(op_type='Shape', name='Shape2')
reshape = graph.create_operation(op_type='Reshape', name='FinalReshape')
gemm3   = graph.create_operation(op_type='Gemm', name='FinalGemm')

graph.create_link_with_op(graph.create_variable(), A=None, B=gemm)
graph.create_link_with_op(graph.create_variable(), A=gemm, B=sigmoid)
graph.create_link_with_op(graph.create_variable(), A=sigmoid, B=shape1)
graph.create_link_with_op(graph.create_variable(), A=shape1, B=relu)
graph.create_link_with_op(graph.create_variable(), A=relu, B=gemm2)
graph.create_link_with_op(graph.create_variable(), A=gemm2, B=None)
graph.create_link_with_op(graph.create_variable(), A=conv, B=reshape)
graph.create_link_with_op(graph.create_variable(), A=relu, B=shape2)
graph.create_link_with_op(graph.create_variable(), A=shape2, B=reshape)
graph.create_link_with_op(graph.create_variable(), A=reshape, B=gemm3)
graph.create_link_with_op(graph.create_variable(), A=None, B=conv)
dispatcher = Perseus(graph)

for op in graph.operations.values():
    print(op.name, [op.name for op in dispatcher.parse_transitive_fanout(parsing_from=[op])])
dispatch_table = dispatcher.dispatch()
print(dispatch_table)

assert dispatch_table['FirstGemm'] == OType.QUANTABLE
assert dispatch_table['Shape1'] == OType.NONQUANTABLE
assert dispatch_table['Shape2'] == OType.NONQUANTABLE
assert dispatch_table['ShapeGemm'] == OType.CONTROVERSIAL
assert dispatch_table['FinalReshape'] == OType.QUANTABLE
assert dispatch_table['FinalGemm'] == OType.QUANTABLE

graph = BaseGraph(name='TestCase2', built_from=NetworkFramework.ONNX)
gemm = graph.create_operation(op_type='Gemm', name='FirstGemm')
topk = graph.create_operation(op_type='TopK', name='TopK')
gemm2 = graph.create_operation(op_type='Gemm', name='Gemm2')
gemm3 = graph.create_operation(op_type='Gemm', name='Gemm3')
mul = graph.create_operation(op_type='Mul', name='Mul')

graph.create_link_with_op(graph.create_variable(), A=None, B=gemm)
graph.create_link_with_op(graph.create_variable(), A=gemm, B=topk)
graph.create_link_with_op(graph.create_variable(), A=topk, B=gemm2)
graph.create_link_with_op(graph.create_variable(), A=topk, B=gemm3)
graph.create_link_with_op(graph.create_variable(), A=gemm2, B=mul)
graph.create_link_with_op(graph.create_variable(), A=gemm3, B=mul)
graph.create_link_with_op(graph.create_variable(), A=mul, B=None)

dispatcher = Perseus(graph)
dispatch_table = dispatcher.dispatch()
print(dispatch_table)

for op in graph.operations.values():
    print(op.name, [op.name for op in dispatcher.parse_transitive_fanout(parsing_from=[op])])

assert dispatch_table['FirstGemm'] == OType.CONTROVERSIAL
assert dispatch_table['Gemm2'] == OType.CONTROVERSIAL
assert dispatch_table['Gemm3'] == OType.CONTROVERSIAL
assert dispatch_table['Mul'] == OType.CONTROVERSIAL


graph = BaseGraph(name='TestCase3', built_from=NetworkFramework.ONNX)
gemm = graph.create_operation(op_type='Gemm', name='FirstGemm')
reshape = graph.create_operation(op_type='Reshape', name='Reshape')
add = graph.create_operation(op_type='Add', name='Add')
shape = graph.create_operation(op_type='Shape', name='Shape')
const = graph.create_operation(op_type='Constant', name='Constant')
conv = graph.create_operation(op_type='Conv', name='Conv')
mul = graph.create_operation(op_type='Mul', name='Mul')
cos = graph.create_operation(op_type='ConstantOfShape', name='ConstantOfShape')
shape2 = graph.create_operation(op_type='Shape', name='Shape2')

graph.create_link_with_op(graph.create_variable(), A=None, B=gemm)
graph.create_link_with_op(graph.create_variable(), A=gemm, B=reshape)
graph.create_link_with_op(graph.create_variable(), A=None, B=shape)
graph.create_link_with_op(graph.create_variable(), A=None, B=shape2)
graph.create_link_with_op(graph.create_variable(), A=shape, B=add)
graph.create_link_with_op(graph.create_variable(), A=const, B=add)
graph.create_link_with_op(graph.create_variable(), A=add, B=reshape)

graph.create_link_with_op(graph.create_variable(), A=reshape, B=conv)
graph.create_link_with_op(graph.create_variable(), A=conv, B=mul)
graph.create_link_with_op(graph.create_variable(), A=shape2, B=cos)
graph.create_link_with_op(graph.create_variable(), A=cos, B=mul)
graph.create_link_with_op(graph.create_variable(), A=mul, B=None)

dispatcher = Perseus(graph)
print([op.name for op in dispatcher.parse_transitive_fanout(parsing_from=[conv])])
dispatch_table = dispatcher.dispatch()
print(dispatch_table)

assert dispatch_table['FirstGemm'] == OType.QUANTABLE
assert dispatch_table['Add']       == OType.NONQUANTABLE
assert dispatch_table['Constant']  == OType.NONQUANTABLE
assert dispatch_table['Mul']       == OType.QUANTABLE
assert dispatch_table['Conv']      == OType.QUANTABLE
assert dispatch_table['Reshape']   == OType.QUANTABLE

