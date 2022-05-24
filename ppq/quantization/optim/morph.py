from math import ceil
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import torch
from ppq.IR.morph import GraphDecomposer
from ppq.core import (NetworkFramework, OperationQuantizationConfig,
                      QuantizationStates, TensorMeta, TensorQuantizationConfig,
                      empty_ppq_cache, ppq_warning)
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import (BaseGraph, GraphCommandProcessor, GraphFormatter,
                    GraphReplacer, Operation, Path, QuantableGraph,
                    QuantableOperation, SearchableGraph, Variable)
from ppq.IR.base.command import QuantizeOperationCommand
from ppq.log import NaiveLogger
from ppq.quantization.observer import TensorObserverFactroy
from tqdm import tqdm

from .base import QuantizationOptimizationPass
from .calibration import RuntimeCalibrationPass
from .parameters import ParameterQuantizePass, PassiveParameterQuantizePass
from .refine import QuantAlignmentPass, QuantizeReducePass
from .training import compute_loss

logger = NaiveLogger.get_logger('PPQ')


class NXPResizeModeChangePass(QuantizationOptimizationPass):
    """This optimization pass overwrite resize mode to 'nearest' for all resize
    operations."""
    def __init__(self) -> None:
        super().__init__(name='NXP Resize Operation Transformation')

    def optimize(self, processor: GraphCommandProcessor, dataloader: Iterable,
        executor: BaseGraphExecutor, **kwargs) -> None:
        for op in processor.graph.operations.values():
            if op.type == 'Resize':
                op.attributes['mode'] = 'nearest'
                op.attributes['coordinate_transformation_mode'] = 'half_pixel'


class NCNNFormatGemmPass(QuantizationOptimizationPass):
    def __init__(self, name: str = 'ncnn Format Gemm Pass') -> None:
        super().__init__(name)

    def optimize(self, processor: GraphCommandProcessor, dataloader: Iterable,
        executor: BaseGraphExecutor, **kwargs) -> None:
        for op in processor.graph.operations.values():
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


class MatrixFactorizationPass(QuantizationOptimizationPass):
    """Use Matrix Farctorization to minimize quantization error. This pass will
    split a computing layer with 2 sub layers.

        before split:  WX + b = Y
        after split:   B(AX) + b = Y

        Where W = BA

    However i do not know how to minimize quant loss until now.

    Args:
        QuantizationOptimizationPass ([type]): [description]

    Returns:
        [type]: [description]
    """
    def __init__(self, interested_layers: List[str], method: str = 'training',
                 name: str = 'SVD Split Pass') -> None:
        self.interested_layers = interested_layers
        self.method = method
        super().__init__(name=name)

    @ empty_ppq_cache
    def train_for_factorization(
        self, w: torch.Tensor, penalty = 0.1,
        executing_device: str = 'cuda', max_iter: int = 100000):
        assert w.ndim == 2

        a, b = torch.rand(size=[w.shape[0], w.shape[1]]), torch.rand(size=[w.shape[1], w.shape[1]])

        a = a.to(executing_device)
        b = b.to(executing_device)
        w = w.to(executing_device)

        a.requires_grad = True
        b.requires_grad = True
        w.requires_grad = True

        optimizer = torch.optim.Adam(params=[a, b], lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        last_loss = 1e9
        for _ in tqdm(range(max_iter), 'Training for factorization ...'):
            penalty_loss = (torch.mean(torch.square(a)) + torch.mean(torch.square(b))) * penalty
            loss = loss_fn(w, torch.matmul(a, b)) + penalty_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()
            if abs(loss - last_loss) < 1e-7: break
            else: last_loss = loss

        a.requires_grad = False
        b.requires_grad = False
        a._grad = None
        b._grad = None

        return a, b

    def svd_for_factorization(self, w: torch.Tensor):
        assert w.ndim == 2
        u, s, v = torch.svd(w)
        a = torch.matmul(u, torch.diag(torch.sqrt(s)))
        b = torch.matmul(torch.diag(torch.sqrt(s)), v.transpose(0, 1))
        print(a.max(), b.max(), w.max())
        return a, b

    def optimize(self, processor: GraphCommandProcessor,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        graph = processor.graph
        spliting_layers = []
        for name in self.interested_layers:
            if name not in graph.operations:
                raise ValueError(f'Can not Split layer {name}, can not find it in current graph.')

        for operation in graph.operations.values():
            if operation.name in self.interested_layers:
                assert operation.type in {'Conv', 'Gemm'}, (
                    f'Can not split layer, cause layer type is not support')
                spliting_layers.append(operation)

        for operation in spliting_layers:
            assert isinstance(operation, Operation)
            if operation.type == 'Gemm':
                w = operation.parameters[0].value
                w = w.transpose(0, 1)
                if self.method == 'svd':
                    a, b = self.svd_for_factorization(w)
                elif self.method == 'training':
                    a, b = self.train_for_factorization(w)
                else: raise ValueError(f'Invalid method {self.method}, only support training and svd now.')
                a = a.transpose(0, 1)
                b = b.transpose(0, 1)
            elif operation.type == 'Conv':
                if operation.attributes['kernel_shape'] != [1, 1]:
                    raise PermissionError(f'Can not split layer {operation.name}, cause it kernel shape is not [1, 1]')
                w = operation.parameters[0].value
                assert isinstance(w, torch.Tensor)
                w = w.squeeze(-1).squeeze(-1).transpose(0, 1)
                print(w.shape)
                if self.method == 'svd':
                    a, b = self.svd_for_factorization(w)
                elif self.method == 'training':
                    a, b = self.train_for_factorization(w)
                else: raise ValueError(f'Invalid method {self.method}, only support training and svd now.')
                a = a.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
                b = b.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            else: raise TypeError(f'Unsupported operation type {operation.type}.')
            operation.parameters[0].value = a

            # create new operation & dirty work
            attributes = {}
            if operation.type == 'Conv':
                attributes['kernel_shape'] = [1, 1]
                attributes['pads']         = [0, 0, 0, 0]
                attributes['strides']      = [1, 1]
                attributes['dilations']    = [1, 1]
                attributes['group']        = 1

            if operation.type == 'Gemm':
                attributes['alpha']        = 1
                attributes['beta']         = 1
                attributes['transB']       = 1

            splited = Operation(
                name=operation.name + '_splited',
                op_type=operation.type,
                attributes=attributes,
                platform=operation.platform
            )

            graph.insert_op_on_var(
                inserting_op=splited, var=operation.outputs[0].name)
            if operation.outputs[0].name in graph.outputs:
                graph.outputs.pop(operation.outputs[0].name)
                graph.outputs[splited.outputs[0].name] = splited.outputs[0]

            # add weight link
            spilted_w = Variable(name=splited.name + '_weight',
                                 value=b, is_parameter=True)
            graph.append_variable(spilted_w)

            splited.inputs.append(spilted_w)
            spilted_w.dest_ops.append(splited)

            # if has bias, relink bias
            if len(operation.parameters) > 1:
                bias_var = operation.parameters[-1]
                bias_var.dest_ops.remove(operation)
                bias_var.dest_ops.append(splited)
                splited.inputs.append(bias_var)
                operation.inputs.remove(bias_var)


class WeightSplitPass(QuantizationOptimizationPass):
    """WeightSplitPass is similar to Outlier ChannelSplitPass below, both
    designed for per-tensor quantization, the difference is that
    ChannelSplitPass needs to find a counterpart computing operation right next
    to the desired computing layers so that no new operators will be added to
    graph. However, WeightSplitPass can operate under any kind of situation,
    and the cost will be computing overhead for bringing in new quantization
    operators.

       Conv    ==>   Conv_child_1  Conv_child_2
                            \       /
                             \     /
                               Add
                                |

        Use this pass when some very bad layer appears in precision analysis, i.e., the layer encountering a huge snr
    error or reflecting poor cosine similarity.
    """
    def __init__(self,
                interested_layers: List[str]=[],
                loss_reduce_thre: float=0.80
                ) -> None:
        """Weight Split Pass.

        Args:
            interested_layers (List[str], optional): layers which need split. Defaults to [].
            loss_reduce_thre (float, optional): the split will take effect only if mse loss after split is less than
                                                this threshold wrt. the original loss. Defaults to 0.80.
        """
        super().__init__('Weight Split Pass')
        self.interested_layers = interested_layers
        self.loss_reduce_thre  = loss_reduce_thre

    def init_tri_op_graph(self, computing_op: QuantableOperation) -> BaseGraph:
        # build tri-op subgraph for testing different split points
        graph = BaseGraph(name='TempGraph', built_from=NetworkFramework.NATIVE)
        input_var = Variable(name='Input', is_parameter=False)

        first_child_param_weight  = Variable(name=f'{computing_op.name}_first_child_weight', \
                            value=computing_op.parameters[0].value, is_parameter=True)
        first_child_output        = Variable(name=f'{computing_op.name}_first_child_output')

        second_child_param_weight = Variable(name=f'{computing_op.name}_second_child_weight', \
                            value=computing_op.parameters[0].value, is_parameter=True)
        second_child_output       = Variable(name=f'{computing_op.name}_second_child_output')

        final_output = Variable(name='Output')

        graph.append_variable(input_var)
        graph.append_variable(first_child_param_weight)
        graph.append_variable(first_child_output)
        graph.append_variable(second_child_param_weight)
        graph.append_variable(second_child_output)
        graph.append_variable(final_output)

        first_child = Operation(name=f'{computing_op.name}_first_child', op_type=computing_op.type, \
            attributes=computing_op.attributes, inputs=[input_var, first_child_param_weight], outputs=[first_child_output])

        second_child = Operation(name=f'{computing_op.name}_second_child', op_type=computing_op.type, \
            attributes=computing_op.attributes, inputs=[input_var, second_child_param_weight], outputs=[second_child_output])

        add_receiver = Operation(name=f'{computing_op.name}_Add_Receiver', op_type='Add', attributes={}, \
            inputs=[first_child_output, second_child_output], outputs=[final_output])

        graph.append_operation(first_child)
        graph.append_operation(second_child)
        graph.append_operation(add_receiver)

        input_var.dest_ops.extend([first_child, second_child])
        first_child_param_weight.dest_ops.append(first_child)
        first_child_output.source_op = first_child
        first_child_output.dest_ops.append(add_receiver)
        second_child_param_weight.dest_ops.append(second_child)
        second_child_output.source_op = second_child
        second_child_output.dest_ops.append(add_receiver)
        final_output.source_op = add_receiver

        if len(computing_op.parameters) == 2:
            first_bias = Variable(name=f'{computing_op.name}_first_child_bias', is_parameter=True, \
                value=computing_op.parameters[1].value, dest_ops=[first_child])
            second_bias = Variable(name=f'{computing_op.name}_second_child_bias', is_parameter=True, \
                value=computing_op.parameters[1].value, dest_ops=[second_child])
            first_child.inputs.append(first_bias)
            second_child.inputs.append(second_bias)
            graph.append_variable(first_bias)
            graph.append_variable(second_bias)

        graph.inputs['Input']  = input_var
        graph.outputs['Output'] = final_output

        return graph

    def quantize(self, computing_op: QuantableOperation, processor: GraphCommandProcessor) -> None:
        graph = processor._graph
        configs = {}
        sample_cfg = computing_op.config.input_quantization_config[0]
        # init configs
        for op in graph.operations.values():
            input_quantization_cfg = [TensorQuantizationConfig(
                policy=sample_cfg.policy,
                rounding=sample_cfg.rounding,
                num_of_bits=sample_cfg.num_of_bits,
                quant_min=sample_cfg.quant_min,
                quant_max=sample_cfg.quant_max,
                scale=None,
                offset=None,
                observer_algorithm=sample_cfg.observer_algorithm)]
            output_quantization_cfg = [input_quantization_cfg[0].copy()]
            # computing layers
            if len(op.parameters) > 0:
                for cfg in computing_op.config.input_quantization_config[1:]:
                    assert isinstance(cfg, TensorQuantizationConfig)
                    input_quantization_cfg.append(cfg.copy())
            # add
            else:
                input_quantization_cfg.append(input_quantization_cfg[0].copy())
            configs[op.name] = OperationQuantizationConfig(input_quantization_cfg, output_quantization_cfg)

        for op_name, cfg in configs.items():
            processor(QuantizeOperationCommand(op_name, computing_op.platform, cfg))

    def build_quantization_pipeline(self, computing_op: QuantableOperation) -> List[QuantizationOptimizationPass]:
        # routine passes
        quantization_pipeline = [
            QuantizeReducePass(),
            ParameterQuantizePass(method=computing_op.config.input_quantization_config[1].observer_algorithm),
            RuntimeCalibrationPass(method=computing_op.config.input_quantization_config[0].observer_algorithm),
            QuantAlignmentPass(),
            PassiveParameterQuantizePass()
        ]
        return quantization_pipeline

    def prepare_for_quantization(
        self,
        computing_op: QuantableOperation,
        graph: BaseGraph,
        first_child_weight: torch.Tensor,
        second_child_weight: torch.Tensor,
        first_child_bias: Union[torch.Tensor, None],
        second_child_bias: Union[torch.Tensor, None]
    ) -> None:
        # change weight values of computing layers according to split point
        # initialize all state for calibration
        graph.variables[f'{computing_op.name}_first_child_weight'].value = first_child_weight
        graph.variables[f'{computing_op.name}_second_child_weight'].value = second_child_weight
        if first_child_bias is not None and second_child_bias is not None:
            graph.variables[f'{computing_op.name}_first_child_bias'].value = first_child_bias
            graph.variables[f'{computing_op.name}_second_child_bias'].value = second_child_bias

        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()
                for cfg, var in op.config_with_variable:
                    if cfg.state == QuantizationStates.ACTIVATED or cfg.state == QuantizationStates.SLAVE:
                        cfg.state = QuantizationStates.INITIAL
                        if cfg._father_config != cfg:
                            cfg._father_config = cfg
                    elif cfg.state == QuantizationStates.PASSIVE:
                        cfg.state = QuantizationStates.PASSIVE_INIT

    def prepare_data(self,
            computing_op: QuantableOperation,
            executor: TorchExecutor,
            dataloader: Iterable,
            calib_steps: int,
            collate_fn: Callable
    ) -> List[torch.Tensor]:
        # prepare data for subgraph input
        input_data = []
        calib_step = 0
        for calib_epoch in range(ceil(calib_steps / len(dataloader))):
            for data in dataloader:
                if collate_fn is not None:
                    data = collate_fn(data)
                input_data.extend(executor.forward(data, output_names=[computing_op.inputs[0].name]))
                calib_step += 1
                if calib_step >= calib_steps: break
        return input_data

    def split_weight(self,
                    op: QuantableOperation,
                    sub_graph: BaseGraph,
                    weight: torch.Tensor,
                    split_point: float,
                    bias: Union[torch.Tensor, None]
    ) -> Tuple[torch.Tensor]:
        abs_max = weight.abs().max()
        if op.type == 'Conv' or op.type == 'ConvTranspose':
            channel_filter = (weight.reshape(weight.shape[0], -1).max(dim=1)[0] > abs_max * split_point).float()
            first_child_weight = weight * channel_filter.reshape(-1, 1, 1, 1)
            second_child_weight = weight * (1 - channel_filter.reshape(-1, 1, 1, 1))

        elif op.type == 'Gemm':
            channel_filter = (weight.reshape(weight.shape[0], -1).max(dim=1)[0] > abs_max * split_point).float()
            first_child_weight = weight * channel_filter.reshape(-1, 1)
            second_child_weight = weight * (1 - channel_filter.reshape(-1, 1))

        if len(op.parameters) == 2:
            first_child_bias = bias * channel_filter
            second_child_bias = bias * (1 - channel_filter)
        else:
            first_child_bias, second_child_bias = None, None
        self.prepare_for_quantization(op, sub_graph, first_child_weight, \
                    second_child_weight, first_child_bias, second_child_bias)

    def compute_original_loss(
        self,
        computing_op: QuantableOperation,
        dataloader: List[torch.Tensor],
        calib_steps: int,
        device: Union[str, torch.device]
    ) -> float:
        # build single-op graph for computing original fp-quant MSE loss
        single_op_graph = BaseGraph(name='TempGraph', built_from=NetworkFramework.NATIVE)
        input_var, output_var = Variable(name='Input'), Variable(name='Output')
        weight_param = Variable(name='weight_param', value=computing_op.parameters[0].value , is_parameter=True)

        single_op_graph.append_variable(input_var)
        single_op_graph.append_variable(output_var)
        single_op_graph.append_variable(weight_param)

        conv_op = Operation(name='ConvOp', op_type='Conv', attributes=computing_op.attributes,\
                    inputs=[input_var, weight_param], outputs=[output_var])
        single_op_graph.append_operation(conv_op)

        if len(computing_op.parameters) == 2:
            bias = Variable(name='bias_param', value=computing_op.parameters[1].value, is_parameter=True, dest_ops=[conv_op])
            single_op_graph.append_variable(bias)
            conv_op.inputs.append(bias)

        input_var.dest_ops.append(conv_op)
        weight_param.dest_ops.append(conv_op)

        output_var.source_op = conv_op
        single_op_graph.inputs['Input']   = input_var
        single_op_graph.outputs['Output'] = output_var

        sub_executor  = TorchExecutor(single_op_graph, False, device)
        self.trace_meta(sub_executor, dataloader)
        sub_processor = QuantableGraph(GraphReplacer(single_op_graph))
        self.quantize(computing_op, sub_processor)
        sub_executor.load_graph(single_op_graph)
        sub_quantization_pipeline = self.build_quantization_pipeline(computing_op)

        for pipeline in sub_quantization_pipeline:
            pipeline.optimize(sub_processor, dataloader, sub_executor, calib_steps=calib_steps, collate_fn=None)

        return compute_loss(['Output'], single_op_graph, dataloader, None, sub_executor)['Output']

    def merge_graph(self, graph: BaseGraph, sub_graph: BaseGraph, replace_op: QuantableOperation) -> None:
        # a fast and simple way to directly merge sub_graph into original graph
        # the replace_op will be removed from graph and subsititued by ops in the
        # sub_graph

        # append variable
        for var in sub_graph.variables.values():
            if var.name not in sub_graph.inputs and var.name not in sub_graph.outputs:
                assert var.name not in graph.variables
                graph.variables[var.name] = var
        input_var, output_var = replace_op.inputs[0], replace_op.outputs[0]

        first_child  = sub_graph.operations[f'{replace_op.name}_first_child']
        second_child = sub_graph.operations[f'{replace_op.name}_second_child']
        add_receiver = sub_graph.operations[f'{replace_op.name}_Add_Receiver']

        # cut off connection in the original graph
        input_var.dest_ops.pop(input_var.dest_ops.index(replace_op))
        output_var.source_op = None
        replace_op.inputs.pop(0)
        replace_op.outputs.pop(0)
        graph.delete_operation(replace_op.name)
        # connect input_var with subgraph
        input_var.dest_ops.extend([first_child, second_child])
        first_child.inputs[0].dest_ops.clear()
        first_child.inputs.pop(0)
        second_child.inputs.pop(0)
        first_child.inputs.insert(0, input_var)
        second_child.inputs.insert(0, input_var)

        # connect output_var with subgraph
        add_receiver.outputs.pop(0)
        add_receiver.outputs.append(output_var)
        output_var.source_op = add_receiver

        # append operations in sub_graph
        graph.append_operation(first_child)
        graph.append_operation(second_child)
        graph.append_operation(add_receiver)

        # clear sub_graph
        sub_graph.operations.clear()
        sub_graph.variables.clear()

    def trace_meta(self, executor: TorchExecutor, dataloader: Iterable, collate_fn: Callable=None) -> None:
        for data in dataloader:
            if collate_fn is not None:
                data = collate_fn(data)
            executor.tracing_operation_meta(data)
            break

    def optimize(self,
                processor: GraphCommandProcessor,
                dataloader: Iterable,
                executor: BaseGraphExecutor,
                calib_steps: int,
                collate_fn: Callable,
                **kwargs
    ) -> None:
        for op_name in self.interested_layers:
            assert op_name in processor.graph.operations, f'{op_name} is not in current graph'
            op = processor.graph.operations[op_name]
            assert op.is_computing_op and isinstance(op, QuantableOperation), \
                f'only quantable computing op supports weight split'

            op = processor.graph.operations[op_name]
            sub_graph = self.init_tri_op_graph(op)
            sub_executor = TorchExecutor(graph=sub_graph, fp16_mode=False, device=executor._device)
            sub_dataloader = self.prepare_data(op, executor, dataloader, calib_steps, collate_fn)
            self.trace_meta(sub_executor, sub_dataloader)
            sub_processor = QuantableGraph(GraphReplacer(sub_graph))
            self.quantize(op, sub_processor)
            sub_executor.load_graph(graph=sub_graph)
            sub_quantization_pipeline = self.build_quantization_pipeline(op)

            weight, bias = op.parameters[0].value, None
            if len(op.parameters) == 2:
                bias = op.parameters[1].value

            losses = []
            for split_point in np.arange(0.005, 1.00, 0.005):
                self.split_weight(op, sub_graph, weight, split_point, bias)
                for pipeline in sub_quantization_pipeline:
                    pipeline.optimize(sub_processor, sub_dataloader, sub_executor, calib_steps=calib_steps, collate_fn=None)
                loss = compute_loss(['Output'], sub_graph, sub_dataloader, None, sub_executor)['Output']
                losses.append((split_point, loss))
                logger.debug(f'Split Point {split_point :.2f} || MSE Loss {loss :.6f}')

            losses = sorted(losses, key=lambda x: x[1])
            original_loss = self.compute_original_loss(op, sub_dataloader, calib_steps, executor._device)
            logger.info(f'Original MSE Loss {original_loss :.6f}')

            if losses[0][1] < self.loss_reduce_thre * original_loss:
                self.split_weight(op, sub_graph, weight, losses[0][0], bias)
                self.merge_graph(processor.graph, sub_graph, op)
                executor.load_graph(processor.graph)
                self.trace_meta(executor, dataloader, collate_fn)

                logger.info(f'{op.name} splitted successfully, original loss: {original_loss :.6f}'
                    f' => splitted loss : {losses[0][1] :.6f}')
            else:
                logger.warning(f'{op.name} fail to split due to loss not improved enough')


class ChannelSplitPass(QuantizationOptimizationPass):
    """ChannelSplitPass is designed for per-tenser quantization only, this
    implementation is based on the original paper:

      "zhao, Ritchie et al., Improving Neural Network Quantization without Retraining using Outlier Channel Splitting"

      Basically this pass shrinks ranges of outlier channels by first half-down the channel value
          then duplicate the whole channel, making it more friendly for per-tensor quantization
          while preserving the fp output same

      In this implementation, to avoid bringing in supplemental ops, for each user-given op, we find its counterpart op,
      split input/output channel of the user-given op and duplicate output/input channel of its counterpart

              split Conv1                                          split Conv2

      Conv1  --  Relu  --  Conv2                               Conv1  --  Relu  --  Conv2
    (C2,C1,H1,W1)      (C3,C2,H2,W2)                        (C2,C1,H1,W1)      (C3,C2,H2,W2)
          || split          || duplicate                          || duplicate       || split
          \/ duplicate      \/                                    \/                 \/ duplicate
    (C2+C,C1,H1,W1)    (C3,C2+C,H2,W2)                      (C2+C,C1,H1,W1)    (C3,C2+C,H2,W2)
    """
    def __init__(self,
                interested_layers: List[str],
                search_directions: List[str] = None,
                expand_ratio: float=0.1,
                split_ratio: float=0.5,
                grid_aware: bool=True
    ) -> None:
        """ChannelSplitPass, try this when other algorithms fail to improve
        your per-tensor quantization accuracy, interested_layers and
        corresponding search_directions should decided by user, user should
        make sure every split operation in interested_layers has a counterpart
        along the corresponding search direction.

        Args:
            interested_layers (List[str]): a list of strings representing ops you want to apply channel split.
            search_directions (List[str]): a list of strings representing search directions(up or down) of ops given in
                                           interested_layers by user.
            expand_ratio (float, optional): ratio of duplicate channels. Defaults to 0.1.
            split_ratio (float, optional): ratio of values in outlier channel which will half-down. Defaults to 0.5.
            grid_aware (bool, optional): whether to apply quantization aware split. Defaults to True.
        """
        self.interested_layers = interested_layers
        self.search_directions = search_directions

        if not self.search_directions or len(search_directions) != len(interested_layers):
            ppq_warning('You do not provide a valid search direction. '
                        'All layer will split with its upstream layers by default.')
            self.search_directions = ['up' for _ in self.interested_layers]

        self.expand_ratio = expand_ratio
        self.grid_aware = grid_aware
        self.split_ratio = split_ratio
        self.current_search_direction = None
        super().__init__(name='Channel Split Pass')

    def calculate_scale(self, split_op: QuantableOperation) -> torch.Tensor:
        config = split_op.config.input_quantization_config[1]
        observer = TensorObserverFactroy.build_observer(split_op.parameters[0], config)
        observer.observe(split_op.parameters[0].value)
        observer.render_quantization_config()
        return config.scale

    def flip(self, op: Operation) -> bool:
        return (self.current_search_direction == 'down') != (op.type == 'ConvTranspose' or (op.type == 'Gemm'\
            and op.attributes.get('transB', 0) == 0))

    def OCS_forward(self, split_op: Operation) -> List[int]:
        weight = split_op.parameters[0].value
        axes = list(range(weight.ndim))

        # update bias when the out dimension needs half-down and duplicate
        update_bias = (self.current_search_direction == 'down' and len(split_op.parameters) > 1)
        if update_bias:
            bias = split_op.parameters[1].value

        # permute weight so that we can always operate on the second axis
        if self.flip(split_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        num_channels = weight.shape[1]
        ocs_channels = ceil(num_channels * self.expand_ratio)
        in_channels_to_copy = []
        orig_idx_dict = {}
        if self.grid_aware:
            assert isinstance(split_op, QuantableOperation), (
                f'Operation {split_op.name} is not quantable, can not be splited via this function.')
            w_scale = self.calculate_scale(split_op)

        for c in range(ocs_channels):
            flatten_weight = weight.permute(1, 0, *axes[2:]).contiguous()
            flatten_weight = flatten_weight.reshape(flatten_weight.shape[0], -1)
            max_per_channel = torch.max(flatten_weight.abs(), 1)[0]
            idxs = torch.argsort(max_per_channel, descending=True)
            split_idx = idxs[0].item()
            ch_slice = weight[:, split_idx:(split_idx + 1)].clone()
            ch_slice_half = ch_slice / 2
            ch_slice_zero = torch.zeros_like(ch_slice)

            # for a top-down search, we need to directly half-down the weight and bias
            if self.current_search_direction == 'up':
                split_value = ch_slice.max() * self.split_ratio
            else:
                split_value = 0

            if (not self.grid_aware) or self.current_search_direction == 'down':
                ch_slice_1 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half, ch_slice)
                ch_slice_2 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half, ch_slice_zero)
            else:
                # assert per-tensor
                ch_slice_half /= w_scale
                ch_slice_1 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half-0.25, ch_slice / w_scale) * w_scale
                ch_slice_2 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half+0.25, ch_slice_zero) * w_scale
            weight[:, split_idx:(split_idx+1)] = ch_slice_1
            weight = torch.cat([weight, ch_slice_2], dim=1)

            if update_bias:
                bias_slice_half = bias[split_idx:(split_idx+1)] / 2
                bias[split_idx] = bias_slice_half
                bias = torch.cat([bias, bias_slice_half], dim=0)

            if split_idx < num_channels:
                in_channels_to_copy.append(split_idx)
                orig_idx_dict[num_channels+c] = split_idx
            else:
                in_channels_to_copy.append(orig_idx_dict[split_idx])
                orig_idx_dict[num_channels+c] = orig_idx_dict[split_idx]

        # permute back
        if self.flip(split_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        # update param
        split_op.parameters[0].value = weight
        if update_bias:
            split_op.parameters[1].value = bias
        return in_channels_to_copy

    def update_counterpart(self, counterpart_op: Operation, in_channels_to_copy: List[int]) -> None:
        weight = counterpart_op.parameters[0].value
        axes = list(range(weight.ndim))

        # permute weight so that we can always operate on the first axis
        if self.flip(counterpart_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        weight_split = torch.index_select(weight, dim=0, index=torch.tensor(in_channels_to_copy, dtype=torch.int64, device=weight.device))
        weight = torch.cat([weight, weight_split], dim=0)

        # update bias when the output dimension needs duplicate
        update_bias = (self.current_search_direction == 'up' and len(counterpart_op.parameters) > 1)
        if update_bias:
            bias = counterpart_op.parameters[1].value
            bias_split = torch.index_select(bias, dim=0, index=torch.tensor(in_channels_to_copy, dtype=torch.int64, device=bias.device))
            bias = torch.cat([bias, bias_split], dim=0)

        # flip back
        if self.flip(counterpart_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        # update param
        counterpart_op.parameters[0].value = weight
        if update_bias:
            counterpart_op.parameters[1].value = bias

    def check(self, graph: BaseGraph, path:Path) -> bool:
        if self.current_search_direction == 'up':
            for op in path.tolist()[1:]:
                if len(graph.get_downstream_operations(op)) != 1:
                    return False
            upstream_op, downstream_op = path[-1], path[0]
        else:
            for op in path.tolist()[:-1]:
                if len(graph.get_downstream_operations(op)) != 1:
                    return False
            upstream_op, downstream_op = path[0], path[-1]
        # not support group conv yet
        if upstream_op.attributes.get('group', 1) != 1 or downstream_op.attributes.get('group', 1) != 1:
            return False
        # should have as least one weight parameter
        if upstream_op.type == 'Gemm' and len(upstream_op.parameters) < 1 or\
            downstream_op.type == 'Gemm' and len(downstream_op.parameters) < 1:
            return False

        # check if weight shapes of upstream and downstream computing ops match
        up_axis, down_axis = 0, 1
        if upstream_op.type == 'ConvTranspose' or (upstream_op.type == 'Gemm' and upstream_op.attributes.get('transB', 0) == 0):
            up_axis = 1
        if downstream_op.type == 'ConvTranspose' or (downstream_op.type == 'Gemm' and downstream_op.attributes.get('transB', 0) == 0):
            down_axis = 0

        if upstream_op.parameters[0].meta.shape[up_axis] != downstream_op.parameters[0].meta.shape[down_axis]:
            return False

        return True

    def modify_meta(self, path:Path, num_channels:int) -> None:
        # all the activations along the path and changed params
        # needs modifying their meta info, i.e., add up the
        # duplicated channels to the second dimension
        for op in path.tolist():
            for var in op.inputs:
                if var.is_parameter:
                    op.meta_data.input_metas[op.inputs.index(var)] = TensorMeta.parsing_from_torch_tensor(var.value)
        path_ = path.tolist()[1:] if self.current_search_direction == 'up' else path.tolist()[:-1]
        for op in path_:
            output_meta = op.meta_data.output_metas[0]
            shape = output_meta.shape
            shape[1] += num_channels

    def store_parameter(self, path: Path) -> None:
        for op in path:
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()

    def initiate_path_state(self, path: Path) -> None:
        for op in path:
            if isinstance(op, QuantableOperation):
                for quant_config in op.config.input_quantization_config + op.config.output_quantization_config:
                    if quant_config.state == QuantizationStates.ACTIVATED:
                        quant_config.state = QuantizationStates.INITIAL
                    elif quant_config.state == QuantizationStates.PASSIVE:
                        quant_config.state = QuantizationStates.PASSIVE_INIT


    def optimize(self, processor: GraphCommandProcessor,
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 **kwargs) -> None:

        graph = processor.graph
        search_engine = SearchableGraph(processor)

        for name, search_direction in zip(self.interested_layers, self.search_directions):
            if name not in graph.operations:
                ppq_warning(f'Can not find operation {name} in your graph, skip its split.')
                continue
            op = graph.operations[name]
            if not op.is_computing_op or not isinstance(op, QuantableOperation):
                ppq_warning(f'Operation {name} can not be splited via channel spilt function, '
                            'cause it is not quantable or it has no parameter.')
                continue

            self.current_search_direction = search_direction
            matching = search_engine.path_matching(
                sp_expr=lambda x: x.name == name,
                rp_expr=lambda x, y: True, # be careful when choosing interested_layers, we assume a reasonable path
                ep_expr=lambda x: x.is_computing_op,
                direction=search_direction
            )

            if len(matching) != 1:
                ppq_warning(f'Can not find a counterpart of operation {name}, '
                            'graph is too complex.')
                continue

            path = matching[0]
            if not self.check(graph, path):
                logger.warning(f'Not support such path due to op constraints for now')
                continue

            if search_direction == 'up':
                logger.info(f"Now processing path {'--'.join(reversed([op.name for op in path]))}")
            else:
                logger.info(f"Now processing path {'--'.join([op.name for op in path])}")

            split_op, counterpart_op = path[0], path[-1]

            copy_channels = self.OCS_forward(split_op)
            self.update_counterpart(counterpart_op, copy_channels)
            self.modify_meta(path, len(copy_channels))
            self.store_parameter(path)
            self.initiate_path_state(path)


class MetaxGemmSplitPass(QuantizationOptimizationPass):
    """Metax 不支持 Gemm 的量化，这个 pass 将 Gemm 拆分成.

    --- MatMul -----|
                    + --- Add ---
        bias   -----|
    """
    def __init__(self, name: str = 'Metax Gemm Split Pass') -> None:
        super().__init__(name)

    # Implementation of Gemm Split will move to IR.morph soon.
    def optimize(self, processor: GraphCommandProcessor,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        morpher = GraphDecomposer(processor)
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
    def optimize(self, processor: GraphCommandProcessor,
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 **kwargs) -> None:

        graph = processor.graph

        interested_ops = []
        for operation in processor.graph.operations.values():
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
