import logging
from math import ceil
from typing import Iterable, List

import torch
from ppq.core import (QuantizationStates, TensorMeta, empty_ppq_cache,
                      ppq_warning)
from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, GraphCommandProcesser, Operation, Variable
from ppq.IR.morph import GraphReplacer
from ppq.IR.quantize import QuantableOperation
from ppq.IR.search import Path, SearchableGraph
from ppq.quantization.observer import TensorObserverFactroy
from tqdm import tqdm

from .base import QuantizationOptimizationPass

logger = logging.getLogger('PPQ')


class NXPResizeModeChangePass(QuantizationOptimizationPass):
    """
    This optimization pass overwrite resize mode to 'nearest' for all resize operations.
    """
    def __init__(self) -> None:
        super().__init__(name='NXP Resize Operation Transformation')
    
    def optimize(self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, **kwargs) -> None:
        for op in processer.graph.operations.values():
            if op.type == 'Resize':
                op.attributes['mode'] = 'nearest'
                op.attributes['coordinate_transformation_mode'] = 'half_pixel'


class MatrixFactorizationPass(QuantizationOptimizationPass):
    """
    Use Matrix Farctorization to minimize quantization error.
        This pass will split a computing layer with 2 sub layers.
        
        before split:  WX + b = Y
        after split:   B(AX) + b = Y

        Where W = BA

    However i do not konw how to minimize quant loss until now.

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

    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        graph = processer.graph
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
            else: raise TypeError(f'Unsupported opeartion type {operation.type}.')
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


class ChannelSplitPass(QuantizationOptimizationPass):
    """
    ChannelSplitPass is designed for per-tenser quantization only, this implementation
    is based on the original paper:

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
        """ChannelSplitPass, try this when other algorithms fail to improve your per-tensor quantization
        accuracy, interested_layers and corresponding search_directions should decided by user, user should
        make sure every split operation in interested_layers has a counterpart along the corresponding 
        search direction

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


    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor, 
                 **kwargs) -> None:

        graph = processer.graph
        search_engine = SearchableGraph(processer)

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
    """
    Metax 不支持 Gemm 的量化，这个 pass 将 Gemm 拆分成 
    
        --- Matmul -----|
                        + --- Add ---
            bias   -----|
    
    """
    def __init__(self, name: str = 'Metax Gemm Split Pass') -> None:
        super().__init__(name)
    
    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        morpher = GraphReplacer(processer)
        interested_ops = []
        for operation in processer.graph.operations.values():
            if operation.type == 'Gemm':
                interested_ops.append(operation)

        for op in interested_ops:
            assert isinstance(op, Operation)
            if op.num_of_input == 2: # no bias gemm
                inserting_matmul = Operation(name=f'{op.name}', op_type='Gemm')
                morpher.replace_op(op_name=op.name, replace_to=inserting_matmul)
            elif op.num_of_input == 3:
                inserting_add      = Operation(name=f'{op.name}', op_type='Add', attributes={})
                inserting_matmul   = Operation(name=f'{op.name}_matmul', op_type='MatMul', attributes={})
                morpher.replace_op(op_name=op.name, replace_to=inserting_add)

                # process with matmul
                weight_var = op.inputs[1]
                inserting_add.inputs.remove(weight_var)
                upstream_op = morpher.graph.get_upstream_operations(inserting_add)
                assert len(upstream_op) == 1, 'Gemm is expected to have at most 1 income op.'
                upstream_op = upstream_op[0]
                
                morpher.graph.insert_op_between_ops(inserting_op=inserting_matmul, up_op=upstream_op, down_op=inserting_add)
                op.inputs.remove(weight_var)
                inserting_matmul.inputs.append(weight_var)
                weight_var.dest_ops.clear()
                weight_var.dest_ops.append(inserting_matmul)
            else: raise ValueError(f'Operation {op.name} should contains 2-3 input, however {op.num_of_input} was given.')
            
            if op.attributes.get('transA') == 1:
                raise ValueError(f'Can not process with operation {op.name}, transA=1 is not allowed.')
            if op.attributes.get('transB') == 1:
                inserting_matmul.inputs[-1].value = torch.permute(inserting_matmul.inputs[-1].value, (1, 0))
            
