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

graph.create_link_with_op(graph.create_variable(), upstream_op=None, downstream_op=gemm)
graph.create_link_with_op(graph.create_variable(), upstream_op=gemm, downstream_op=sigmoid)
graph.create_link_with_op(graph.create_variable(), upstream_op=sigmoid, downstream_op=shape1)
graph.create_link_with_op(graph.create_variable(), upstream_op=shape1, downstream_op=relu)
graph.create_link_with_op(graph.create_variable(), upstream_op=relu, downstream_op=gemm2)
graph.create_link_with_op(graph.create_variable(), upstream_op=gemm2, downstream_op=None)
graph.create_link_with_op(graph.create_variable(), upstream_op=conv, downstream_op=reshape)
graph.create_link_with_op(graph.create_variable(), upstream_op=relu, downstream_op=shape2)
graph.create_link_with_op(graph.create_variable(), upstream_op=shape2, downstream_op=reshape)
graph.create_link_with_op(graph.create_variable(), upstream_op=reshape, downstream_op=gemm3)
graph.create_link_with_op(graph.create_variable(), upstream_op=None, downstream_op=conv)
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

graph.create_link_with_op(graph.create_variable(), upstream_op=None, downstream_op=gemm)
graph.create_link_with_op(graph.create_variable(), upstream_op=gemm, downstream_op=topk)
graph.create_link_with_op(graph.create_variable(), upstream_op=topk, downstream_op=gemm2)
graph.create_link_with_op(graph.create_variable(), upstream_op=topk, downstream_op=gemm3)
graph.create_link_with_op(graph.create_variable(), upstream_op=gemm2, downstream_op=mul)
graph.create_link_with_op(graph.create_variable(), upstream_op=gemm3, downstream_op=mul)
graph.create_link_with_op(graph.create_variable(), upstream_op=mul, downstream_op=None)

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

graph.create_link_with_op(graph.create_variable(), upstream_op=None, downstream_op=gemm)
graph.create_link_with_op(graph.create_variable(), upstream_op=gemm, downstream_op=reshape)
graph.create_link_with_op(graph.create_variable(), upstream_op=None, downstream_op=shape)
graph.create_link_with_op(graph.create_variable(), upstream_op=None, downstream_op=shape2)
graph.create_link_with_op(graph.create_variable(), upstream_op=shape, downstream_op=add)
graph.create_link_with_op(graph.create_variable(), upstream_op=const, downstream_op=add)
graph.create_link_with_op(graph.create_variable(), upstream_op=add, downstream_op=reshape)

graph.create_link_with_op(graph.create_variable(), upstream_op=reshape, downstream_op=conv)
graph.create_link_with_op(graph.create_variable(), upstream_op=conv, downstream_op=mul)
graph.create_link_with_op(graph.create_variable(), upstream_op=shape2, downstream_op=cos)
graph.create_link_with_op(graph.create_variable(), upstream_op=cos, downstream_op=mul)
graph.create_link_with_op(graph.create_variable(), upstream_op=mul, downstream_op=None)

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

