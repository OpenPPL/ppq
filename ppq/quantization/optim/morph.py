from typing import Iterable, List, Tuple

import torch
from ppq.executor import BaseGraphExecutor
from ppq.IR import (BaseGraph, BaseGraph, GraphFormatter,
                    Operation, QuantableOperation, Variable)
from ppq.IR.morph import GraphDecomposer
from ppq.log import NaiveLogger

from .base import QuantizationOptimizationPass

logger = NaiveLogger.get_logger('PPQ')


class NXPResizeModeChangePass(QuantizationOptimizationPass):
    """This optimization pass overwrite resize mode to 'nearest' for all resize
    operations."""
    def __init__(self) -> None:
        super().__init__(name='NXP Resize Operation Transformation')

    def optimize(self, graph: BaseGraph, dataloader: Iterable,
        executor: BaseGraphExecutor, **kwargs) -> None:
        for op in graph.operations.values():
            if op.type == 'Resize':
                op.attributes['mode'] = 'nearest'
                op.attributes['coordinate_transformation_mode'] = 'half_pixel'


class NCNNFormatGemmPass(QuantizationOptimizationPass):
    def __init__(self, name: str = 'ncnn Format Gemm Pass') -> None:
        super().__init__(name)

    def optimize(self, graph: BaseGraph, dataloader: Iterable,
        executor: BaseGraphExecutor, **kwargs) -> None:

        for op in graph.operations.values():
            if op.type == 'Gemm':
                if op.attributes.get('transB', 0) == 0:
                    op.attributes['transB'] = 1
                    weight = op.parameters[0].value
                    assert isinstance(weight, torch.Tensor)
                    op.parameters[0].value = weight.transpose(1, 0).contiguous()
                if  op.attributes.get('alpha', 1.0) != 1.0:
                    op.parameters[0].value = op.parameters[0].value * op.attributes.get('alpha', 1.0)
                    op.attributes['alpha'] = 1.0
                if  op.attributes.get('beta', 1.0) != 1.0:
                    if op.num_of_input > 2:
                        op.parameters[1].value = op.parameters[1].value * op.attributes.get('beta', 1.0)
                    op.attributes['beta'] = 1.0


class HorizontalLayerSplitPass(QuantizationOptimizationPass):
    """
    Horizontal Layer Split Pass(算子分裂过程)

    Split convolution layers or GEMM layers for better performance.
    
    Formula:
    
            Y = W * X + b
            
            where W can be divided into W_1 + W_2
        
            Y = (W_1 * X + b) + (W_2 * X)
    
    By splitting W like this, we are able to represent W more accurately. 
    In the case where one channel has weights in the range [-32, 32] and another channel has weights in the range [-0.5, 0.5].
    the large channel will be divided so the range will come to [-16, 16], which leads us to use scale = 0.125 for representing
    the weight tensor rather than 0.25.
    
    The Estimation of Quantization Error is shown as a quadratic function of scale:
    
            E(Quantization Error) = scale ^ 2 / 12
    
    This Formula is proved by Bernard Widrow, according to the formula, a scale = 0.125 will decrease the quantization error by 75%.
    
    All the value larger than value_threshold will be divided into 2 part via this function, thus the layer itself will be
    splitted, an new Add operation are going to be created.
    
    ### Parameters:
        self.interested_layers = interested_layers
        self.value_threshold   = value_threshold
        self.method            = str(method).lower()
        self.verbose           = verbose

    # interested_layers(List[str])
    
            Only layer that listed in interested_layers will be processed by this pass.
            
            If interested_layers is None or empty list, NO layer will be processed.
    
    # value_threshold(float)

            This pass split value only when value is larger than value_threshold
            
            If there is no value large enough to be processed, corresponding layer will be skipped.
    
    # method(str)
    
            Splitting method, 'balance' or 'random'

            With balance method, W_1 and W_2 will be evenly divided.

            With random method, W_1 and W_2 will be randomly divided.

    ### Warning:
    
    Creating new operation in your network probably slows down the execution.
    
    Thus horizontal splitting is somehow a trade-off between speed and accuracy.
    
    ### Usage

    You can create this optimization manually:

        from ppq import HorizontalLayerSplitPass

        optim = HorizontalLayerSplitPass()
    """
    def __init__(self, interested_layers: List[str] = None, 
                 value_threshold: float = 1, method: str = 'balance',
                 verbose: bool = True) -> None:
        super().__init__('Layer Split Pass(Lateral)')
        self.interested_layers = interested_layers
        self.value_threshold   = value_threshold
        self.method            = str(method).lower()
        self.verbose           = verbose

        if self.interested_layers is None or len(self.interested_layers) == 0:
            raise ValueError('Layer Split Pass(Lateral) Requires a list of spliting layers, '
                             'while parameter interested_layers is empty.')
        
        if self.method not in {'balance', 'random'}:
            raise ValueError(f'Split method must be balance or random. While {self.method} is given.')


    def h_split(self, op: Operation) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # split weight
        value  = op.inputs[1].value
        mask   = (value.abs() > self.value_threshold)
        processed_values = mask.sum().item()

        s_value = value
        if self.method == 'balance':
            s_value = (value / 2) * mask
        elif self.method == 'random':
            s_value = (value * torch.rand_like(value)) * mask
        else: raise Exception('Oops, seems we got some troubles here.')
        r_value = value - s_value

        # print
        if self.verbose:
            print('')
            print(f'# Layer {op.name} has been splited, '
                  f'{processed_values}/{value.numel()} value(s) was processed.')
        return r_value, s_value, processed_values
        
    def optimize(self, graph: BaseGraph, 
                 dataloader: Iterable, executor: BaseGraphExecutor, 
                 **kwargs) -> None:
        with torch.no_grad():
            for name in self.interested_layers:
                # op check
                if name not in graph.operations:
                    raise KeyError(f'Operation {name} is not in current graph.')
                op1 = graph.operations[name]
                if op1.type not in {'Gemm', 'MatMul', 'Conv', 'ConvTranspose'}:
                    raise TypeError(f'Operation {op1.name} can not be splited, op type is invalid({op1.type})')
                if not op1.inputs[1].is_parameter:
                    raise ValueError(f'Operation {op1.name} can not be splited, input 1 is not parameter.')
                if isinstance(op1, QuantableOperation):
                    raise TypeError(f'Can not split a quantized operation, '
                                    'Layer Split Pass should only be invoked as a pre-quant optimziation.')

                r_value, s_value, processed_values = self.h_split(op1)

                if processed_values > 0:
                    # clone current operation
                    op2 = graph.create_operation(
                        op_type=op1.type, attributes=op1.attributes.copy(), 
                        platform=op1.platform)
                    input_var, output_var = op1.inputs[0], op1.outputs[0]
                    graph.create_link_with_op(
                        variable=op1.inputs[0], upstream_op=input_var.source_op, 
                        downstream_op=op2)
                    
                    # create weight for cloned operation.
                    graph.create_link_with_op(
                        variable=graph.create_variable(value=op1.inputs[1].value.clone(), is_parameter=True), 
                        upstream_op=None, downstream_op=op2)

                    # set splited value
                    op1.inputs[1].value.copy_(r_value)
                    op2.inputs[1].value.copy_(s_value)

                    op1.outputs.clear()
                    adder = graph.create_operation(op_type='Add', platform=op1.platform, outputs=[output_var])
                    output_var.source_op = adder
                    
                    graph.create_link_with_op(
                        variable=graph.create_variable(), 
                        upstream_op=op1, downstream_op=adder)
                    graph.create_link_with_op(
                        variable=graph.create_variable(), 
                        upstream_op=op2, downstream_op=adder)


class MetaxGemmSplitPass(QuantizationOptimizationPass):
    """Metax 不支持 Gemm 的量化，这个 pass 将 Gemm 拆分成.

    --- MatMul -----|
                    + --- Add ---
        bias   -----|
    """
    def __init__(self, name: str = 'Metax Gemm Split Pass') -> None:
        super().__init__(name)

    # Implementation of Gemm Split will move to IR.morph soon.
    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        morpher = GraphDecomposer(graph)
        morpher.decompose_gemm()


class GRUSplitPass(QuantizationOptimizationPass):
    """执行 GRU 算子分解，这个 Pass 将 GRU 算子分解为单步执行的形式.

    请注意，对于 ONNX GRU 算子而言, 它有两个输出, 一个是完整的hidden vector, 另一个是单步的 last state 这个
    pass 是针对单步执行而设计的，它将直接删除 hidden vector 之后的所有输出
    """
    def __init__(self, name: str = 'Metax Gemm Split Pass') -> None:
        super().__init__(name)

    def delete_hidden_vec(self, graph: BaseGraph, hidden_vec: Variable):
        processor = GraphFormatter(graph)
        processor.truncate_on_var(var=hidden_vec, mark_as_output=False)

    # Implementation of Gemm Split will move to IR.morph soon.
    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 **kwargs) -> None:

        interested_ops = []
        for operation in graph.operations.values():
            if operation.type == 'GRU':
                interested_ops.append(operation)

        for op in interested_ops:
            assert isinstance(op, Operation)
            # fetch all related variables
            rnn_x, rnn_w, rnn_r, rnn_b, _, rnn_h = op.inputs
            hidden_size = op.attributes['hidden_size']

            # Take a further look at
            # https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU
            Wz = rnn_w.value[0, hidden_size * 0: hidden_size * 1]
            Wr = rnn_w.value[0, hidden_size * 1: hidden_size * 2]
            Wh = rnn_w.value[0, hidden_size * 2: hidden_size * 3]

            Rz = rnn_r.value[0, hidden_size * 0: hidden_size * 1]
            Rr = rnn_r.value[0, hidden_size * 1: hidden_size * 2]
            Rh = rnn_r.value[0, hidden_size * 2: hidden_size * 3]

            Wbz = rnn_b.value[0, hidden_size * 0: hidden_size * 1]
            Wbr = rnn_b.value[0, hidden_size * 1: hidden_size * 2]
            Wbh = rnn_b.value[0, hidden_size * 2: hidden_size * 3]

            Rbz = rnn_b.value[0, hidden_size * 3: hidden_size * 4]
            Rbr = rnn_b.value[0, hidden_size * 4: hidden_size * 5]
            Rbh = rnn_b.value[0, hidden_size * 5: hidden_size * 6]

            # create operations
            op1 = graph.create_operation(op_type='Gemm', attributes={'transB': 1})
            op2 = graph.create_operation(op_type='Gemm', attributes={'transB': 1})
            op3 = graph.create_operation(op_type='Add')
            op4 = graph.create_operation(op_type='Sigmoid')
            op5 = graph.create_operation(op_type='Slice')
            op6 = graph.create_operation(op_type='Slice')
            op7 = graph.create_operation(op_type='Gemm', attributes={'transB': 1})
            op8 = graph.create_operation(op_type='Gemm', attributes={'transB': 1})
            op9 = graph.create_operation(op_type='Mul')
            op10 = graph.create_operation(op_type='Mul')
            op11 = graph.create_operation(op_type='Sub')
            op12 = graph.create_operation(op_type='Add')
            op13 = graph.create_operation(op_type='Mul')
            op14 = graph.create_operation(op_type='Tanh')
            op15 = graph.create_operation(op_type='Add')

            # create parameter variables
            # 为了加速运算，我们将Wz, Wr合并成Wzr, Rzh等同理
            # 参考 https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU
            Wzr_var  = graph.create_variable(value=torch.cat([Wz, Wr]), is_parameter=True)
            Rzr_var  = graph.create_variable(value=torch.cat([Rz, Rr]), is_parameter=True)
            Wbzr_var = graph.create_variable(value=torch.cat([Wbz, Wbr]), is_parameter=True)
            Rbzr_var = graph.create_variable(value=torch.cat([Rbz, Rbr]), is_parameter=True)

            Wh_var  = graph.create_variable(value=Wh, is_parameter=True)
            Rh_var  = graph.create_variable(value=Rh, is_parameter=True)
            Wbh_var = graph.create_variable(value=Wbh, is_parameter=True)
            Rbh_var = graph.create_variable(value=Rbh, is_parameter=True)

            constant_of_sub = graph.create_variable(value=torch.tensor(1.0).to(Wz.device), is_parameter=True)

            # link variables
            graph.create_link_with_op(variable=constant_of_sub, upstream_op=None, downstream_op=op11)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op1, downstream_op=op3)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op2, downstream_op=op3)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op3, downstream_op=op4)

            var = graph.create_variable()
            graph.create_link_with_op(variable=var, upstream_op=op4, downstream_op=op5)
            graph.create_link_with_op(variable=var, upstream_op=op4, downstream_op=op6)

            var = graph.create_variable()
            graph.create_link_with_op(variable=var, upstream_op=op5, downstream_op=op11)
            graph.create_link_with_op(variable=var, upstream_op=op5, downstream_op=op10)

            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op6, downstream_op=op9)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op7, downstream_op=op9)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op8, downstream_op=op12)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op9, downstream_op=op12)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op10, downstream_op=op15)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op11, downstream_op=op13)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op12, downstream_op=op14)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op13, downstream_op=op15)
            graph.create_link_with_op(variable=graph.create_variable(), upstream_op=op14, downstream_op=op13)

            # mark h as graph input, link h to op2, op10 and op7
            rnn_h.source_op.outputs.remove(rnn_h)
            rnn_h.source_op = None
            rnn_h.dest_ops.remove(op)
            graph.mark_variable_as_graph_input(rnn_h)
            graph.create_link_with_op(variable=rnn_h, upstream_op=None, downstream_op=op2)
            graph.create_link_with_op(variable=rnn_h, upstream_op=None, downstream_op=op7)
            graph.create_link_with_op(variable=rnn_h, upstream_op=None, downstream_op=op10)

            # link x to op1 and op8
            rnn_x.dest_ops.remove(op)
            graph.create_link_with_op(variable=rnn_x, upstream_op=rnn_x.source_op, downstream_op=op1)
            graph.create_link_with_op(variable=rnn_x, upstream_op=rnn_x.source_op, downstream_op=op8)

            # create parameters
            graph.create_link_with_op(variable=Wzr_var, upstream_op=None, downstream_op=op1)
            graph.create_link_with_op(variable=Rzr_var, upstream_op=None, downstream_op=op2)
            graph.create_link_with_op(variable=Wh_var, upstream_op=None, downstream_op=op8)
            graph.create_link_with_op(variable=Rh_var, upstream_op=None, downstream_op=op7)
            graph.create_link_with_op(variable=Wbzr_var, upstream_op=None, downstream_op=op1)
            graph.create_link_with_op(variable=Rbzr_var, upstream_op=None, downstream_op=op2)
            graph.create_link_with_op(variable=Wbh_var, upstream_op=None, downstream_op=op8)
            graph.create_link_with_op(variable=Rbh_var, upstream_op=None, downstream_op=op7)

            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([0]), is_parameter=True), upstream_op=None, downstream_op=op5)
            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([hidden_size]), is_parameter=True), upstream_op=None, downstream_op=op5)
            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([1]), is_parameter=True), upstream_op=None, downstream_op=op5)
            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([1]), is_parameter=True), upstream_op=None, downstream_op=op5)

            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([hidden_size]), is_parameter=True), upstream_op=None, downstream_op=op6)
            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([2 * hidden_size]), is_parameter=True), upstream_op=None, downstream_op=op6)
            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([1]), is_parameter=True), upstream_op=None, downstream_op=op6)
            graph.create_link_with_op(variable=graph.create_variable(
                value=torch.tensor([1]), is_parameter=True), upstream_op=None, downstream_op=op6)

            hidden_vec, last_state = op.outputs
            last_state.source_op = op15
            op15.outputs.append(last_state)

            op.inputs.clear()
            op.outputs.clear()
            graph.remove_operation(op)
            self.delete_hidden_vec(graph, hidden_vec)
