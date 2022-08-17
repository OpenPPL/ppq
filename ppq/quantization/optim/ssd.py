from math import ceil
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from ppq.core import QuantizationProperty, QuantizationStates
from ppq.executor import BaseGraphExecutor
from ppq.executor.base import GLOBAL_DISPATCHING_TABLE
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.IR.search import Path, SearchableGraph, TraversalCommand
from ppq.log import NaiveLogger
from ppq.quantization.measure import torch_mean_square_error
from ppq.quantization.observer import CalibrationHook, OperationObserver
from ppq.quantization.observer.range import TorchHistObserver
from ppq.quantization.qfunction import BaseQuantFunction
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction
from tqdm import tqdm

from .base import QuantizationOptimizationPass

logger = NaiveLogger.get_logger('PPQ')


OPTIMIZATION_LAYERTYPE_CONFIG = {
    1: {'Relu', 'MaxPool', 'GlobalMaxPool', 'PRelu', 'AveragePool', 'GlobalAveragePool'}, # support more relay operation types
}
EQUALIZATION_OPERATION_TYPE = {'Conv', 'Gemm', 'ConvTranspose'} # support all computing op types


class SSDEqualizationPass(QuantizationOptimizationPass):
    """PPQ Custimized Layerwise Equalization Pass.

    This is another layerwise equalization pass which takes quantization error into consideration, for more
    details of equalization, please refer to LayerwiseEqualizationPass

    Equalization algo will only take effect when the estimated loss is less than loss_threshold * original_loss

    Compared with LayerwiseEqualizationPass, it guarantees a better result than the original model while consuming
    more time for loss estimation
    """
    def __init__(
        self,
        optimize_level: int=1,
        channel_ratio: float = 0.5,
        loss_threshold: float = 0.8,
        layer_norm: bool = False,
        quant_func: BaseQuantFunction = PPQLinearQuantFunction,
        iteration: int = 3
    ):
        """SSD Equalization Pass With Loss Checking.

        Args:
            optimize_level (int, optional): level of optimization. Only support level 1 for now.

            channel_ratio (float, optional): all values below this ratio of maximum value of corresponding weight will
            be clipped to this ratio when calculating scale. Defaults to 0.5.

            loss_threshold (float, optional): the estimated loss should be below this threshold * original_loss for algo
            to take effect. Defaults to 0.8.

            layer_norm (bool, optional): whether to apply weight normalization. Defaults to False.

            quant_func (BaseQuantFunction, optional): quantization function. Defaults to PPQLinearQuantFunction.

            iteration (int, optional): num of iterations to run, usually 3 would be enough. Defaults to 3.
        """
        self.channel_ratio   = channel_ratio
        self.loss_threshold  = loss_threshold
        self.layer_norm      = layer_norm
        self.quant_func      = quant_func
        self.start_op_types  = EQUALIZATION_OPERATION_TYPE
        self.relay_op_types  = OPTIMIZATION_LAYERTYPE_CONFIG[optimize_level]
        self.end_op_types    = EQUALIZATION_OPERATION_TYPE
        self.iteration       = iteration
        super().__init__(name='SSD Equalization Pass')

    def collect_all_pairs(self, graph: BaseGraph) -> List[List[Operation]]:
        def limitation(path: Path) -> bool:
            for op in path.tolist()[:-1]:
                if len(graph.get_downstream_operations(op)) != 1:
                    return False
            return True

        search_engine = SearchableGraph(graph)
        forward_matchings = search_engine.process(TraversalCommand(
            sp_expr=lambda x: x.type in self.start_op_types,
            rp_expr=lambda x, y: y.type in self.relay_op_types,
            ep_expr=lambda x: x.type in self.end_op_types,
            direction='down'))
        return [path.tolist() for path in forward_matchings if limitation(path)]

    def collect_activation_range(
        self,
        pair:  List[Operation],
        executor: BaseGraphExecutor,
        data_loader: Iterable,
        collate_fn: Callable,
        calib_steps: int
    ) -> Dict[Operation, torch.Tensor]:
        """Collect activation ranges for Conv ops in the pair.

        Args:
            pair (List[Operation]): equalization pair
            executor (BaseGraphExecutor): graph executor
            data_loader (Iterable): data loader
            collate_fn (Callable): batch func
            calib_steps (int): calibration steps

        Returns:
            Dict[Operation, torch.Tensor]: activation range of Conv ops in the pair
        """
        op_act_ranges = {op: 0.0 for op in pair if op.type in self.start_op_types}
        output_names = [op.outputs[0].name for op in op_act_ranges]
        calib_step = 0
        for calib_epoch in range(ceil(calib_steps / len(data_loader))):
            for _, data in enumerate(data_loader):
                if collate_fn is not None:
                    data = collate_fn(data)
                outputs = executor.forward(inputs=data, output_names=output_names)
                for op in op_act_ranges:
                    op_conv_data = outputs[output_names.index(op.outputs[0].name)]
                    op_conv_data_relu = F.relu(op_conv_data) # take abs value
                    op_conv_data_relu = op_conv_data_relu.permute(
                        1, 0, *(range(op_conv_data_relu.ndim)[2:])).contiguous()
                    op_conv_data_relu = op_conv_data_relu.reshape((op_conv_data_relu.shape[0], -1))
                    op_conv_data_relu = op_conv_data_relu.max(1)[0]
                    op_act_ranges[op] += op_conv_data_relu
                calib_step += 1
                if calib_step >= calib_steps: break
        for op in op_act_ranges:
            op_act_ranges[op] /= calib_steps

        for op,op_range in op_act_ranges.items():
            assert isinstance(op_range, torch.Tensor)
            op_range_max = op_range.max()
            adjust_range = torch.where(op_range < op_range_max * self.channel_ratio, op_range_max * self.channel_ratio, op_range)
            op_act_ranges[op] = adjust_range
        return op_act_ranges

    def layer_weight_norm(self, pairs: List[List[Operation]]):
        for pair in pairs:
            op_first_weight = pair[0].parameters[0].value
            op_second_weight = pair[-1].parameters[0].value
            op_first_max = op_first_weight.abs().max()
            op_second_max = op_second_weight.abs().max()
            scale = (op_second_max / op_first_max).sqrt()
            pair[0].parameters[0].value = pair[0].parameters[0].value * scale
            if len(pair[0].parameters) > 1:
                pair[0].parameters[1].value = pair[0].parameters[1].value * scale
            pair[-1].parameters[0].value = pair[-1].parameters[0].value / scale

    def prepare_weight_for_equalization(self, pair: List[Operation]) -> Tuple[torch.Tensor]:
        first_computing_op_weight = pair[0].parameters[0].value
        last_computing_op_weight  = pair[-1].parameters[0].value

        assert isinstance(first_computing_op_weight, torch.Tensor)
        assert isinstance(last_computing_op_weight, torch.Tensor)

        if pair[0].type == 'Conv':
            # [C_out, C_in, K, K]
            C_out = first_computing_op_weight.shape[0]
            first_weight_range = first_computing_op_weight.reshape(C_out, -1).abs().max(dim=1)[0]

        elif pair[0].type == 'Gemm':
            if pair[0].attributes.get('transB', 0):
                # [C_out, C_in]
                first_weight_range = first_computing_op_weight.abs().max(dim=1)[0]
            else:
                # [C_in, C_out]
                first_weight_range = first_computing_op_weight.abs().max(dim=0)[0]

        elif pair[0].type == 'ConvTranspose':
            # [C_in, C_out // g, K, K]
            num_group = pair[0].attributes.get('group', 1)
            C_in, C_out_g, K1, K2 = first_computing_op_weight.shape
            first_computing_op_weight = first_computing_op_weight.reshape(num_group, C_in // num_group, C_out_g, K1, K2)
            first_computing_op_weight = first_computing_op_weight.permute(0, 2, 1, 3, 4).contiguous()
            first_computing_op_weight = first_computing_op_weight.reshape(num_group * C_out_g, -1)
            first_weight_range = first_computing_op_weight.abs().max(dim=1)[0]

        if pair[-1].type == 'Conv':
            # [C_out, C_in // g, K, K]
            num_group = pair[-1].attributes.get('group', 1)
            C_out, C_in_g, K1, K2 = last_computing_op_weight.shape
            last_computing_op_weight = last_computing_op_weight.reshape(num_group, C_out // num_group, C_in_g, K1, K2)
            last_computing_op_weight = last_computing_op_weight.permute(0, 2, 1, 3, 4).contiguous()
            last_computing_op_weight = last_computing_op_weight.reshape(num_group * C_in_g, -1)
            last_weight_range = last_computing_op_weight.abs().max(dim=1)[0]

        elif pair[-1].type == 'Gemm':
            C_out = first_weight_range.shape[0]
            if pair[-1].attributes.get('transB', 0):
                if C_out != last_computing_op_weight.shape[1]:
                    last_computing_op_weight = last_computing_op_weight.reshape(last_computing_op_weight.shape[0], C_out, -1)
                    last_computing_op_weight = last_computing_op_weight.permute(1, 0, 2).contiguous().reshape(C_out, -1)
                    last_weight_range = last_computing_op_weight.abs().max(dim=1)[0]
                else:
                    last_weight_range = last_computing_op_weight.abs().max(dim=0)[0]
            else:
                if C_out != last_computing_op_weight.shape[0]:
                    last_computing_op_weight = last_computing_op_weight.reshape(C_out, -1)
                    last_weight_range = last_computing_op_weight.abs().max(dim=1)[0]
                else:
                    last_weight_range = last_computing_op_weight.abs().max(dim=1)[0]

        elif pair[-1].type == 'ConvTranspose':
            C_in = last_computing_op_weight.shape[0]
            last_weight_range = last_computing_op_weight.reshape(C_in, -1).abs().max(dim=1)[0]

        return first_weight_range, last_weight_range

    def write_back(self, pair: List[Operation], scale: torch.Tensor) -> None:
        first_computing_op_weight = pair[0].parameters[0].value
        last_computing_op_weight  = pair[-1].parameters[0].value

        assert isinstance(first_computing_op_weight, torch.Tensor)
        assert isinstance(last_computing_op_weight, torch.Tensor)

        if pair[0].type == 'Conv':
            pair[0].parameters[0].value = first_computing_op_weight * scale.reshape(-1, 1, 1, 1)

        elif pair[0].type == 'Gemm':
            if pair[0].attributes.get('transB', 0):
                pair[0].parameters[0].value = first_computing_op_weight * scale.reshape(-1, 1)
            else:
                pair[0].parameters[0].value = first_computing_op_weight * scale.reshape(1, -1)

        elif pair[0].type == 'ConvTranspose':
            num_group = pair[0].attributes.get('group', 1)
            C_in, C_out_g, K1, K2 = first_computing_op_weight.shape
            first_computing_op_weight = first_computing_op_weight.reshape(num_group, C_in // num_group, C_out_g, K1, K2)
            first_computing_op_weight = first_computing_op_weight * scale.reshape(num_group, 1, -1, 1, 1)
            pair[0].parameters[0].value = first_computing_op_weight.reshape(C_in, C_out_g, K1, K2)

        if len(pair[0].parameters) > 1:
            pair[0].parameters[1].value = pair[0].parameters[1].value * scale

        if pair[-1].type == 'Conv':
            num_group = pair[-1].attributes.get('group', 1)
            C_out, C_in_g, K1, K2 = last_computing_op_weight.shape
            last_computing_op_weight = last_computing_op_weight.reshape(num_group, C_out // num_group, C_in_g, K1, K2)
            last_computing_op_weight = last_computing_op_weight / scale.reshape(num_group, 1, -1, 1, 1)
            pair[-1].parameters[0].value = last_computing_op_weight.reshape(C_out, C_in_g, K1, K2)

        elif pair[-1].type == 'Gemm':
            if pair[-1].attributes.get('transB', 0):
                if scale.numel() != last_computing_op_weight.shape[1]:
                    last_computing_op_weight = last_computing_op_weight.reshape(last_computing_op_weight.shape[0], scale.numel(), -1)
                    last_computing_op_weight = last_computing_op_weight / scale.reshape(1, -1, 1)
                    pair[-1].parameters[0].value = last_computing_op_weight.reshape(last_computing_op_weight.shape[0], -1)
                else:
                    pair[-1].parameters[0].value = last_computing_op_weight / scale.reshape(1, -1)
            else:
                if scale.numel() != last_computing_op_weight.shape[0]:
                    last_computing_op_weight = last_computing_op_weight.reshape(scale.numel(), -1, last_computing_op_weight.shape[-1])
                    last_computing_op_weight = last_computing_op_weight / scale.reshape(-1, 1, 1)
                    pair[-1].parameters[0].value = last_computing_op_weight.reshape(-1, last_computing_op_weight.shape[-1])
                else:
                    pair[-1].parameters[0].value = last_computing_op_weight / scale.reshape(-1, 1)

        elif pair[-1].type == 'ConvTranspose':
            pair[-1].parameters[0].value = last_computing_op_weight / scale.reshape(-1, 1, 1, 1)

    def one_step_equalization(
        self,
        pair: List[Operation],
        op_act_channel_range: Dict[Operation, torch.Tensor]={},
        algo_type: int=2,
        ssd_min_scale: float=8,
        ssd_max_scale: float=2,
        dfq_min_scale: float=0.1,
        dfq_max_scale: float=10,
        eps: float=1e-8
    ):
        """Equalization step with scale being calculated in the way specified
        by algo_type.

        Args:
            pair (List[Operation]): a list of operations representing a equalzation pair
            op_act_channel_range (Dict[Operation, torch.Tensor]): channel-wise activation range of all Conv ops in the graph
            algo_type (int, optional): minor algo type. 0 represents dfq algo, 1~3 represents ssd algo. Defaults to 2.
            ssd_min_scale (float, optional): minimum clip value of scale for ssd algo. Defaults to 8.
            ssd_max_scale (float, optional): maximum clip value of scale for ssd algo. Defaults to 2.
            dfq_min_scale (float, optional): minimum clip value of scale for dfq algo. Defaults to 0.1.
            dfq_max_scale (float, optional): maximum clip value of scale for dfq algo. Defaults to 10.
            eps (float, optional): small constant for numerical stability. Defaults to 1e-8.
        """
        first_weight_range, last_weight_range = self.prepare_weight_for_equalization(pair)

        if algo_type == 0:
            # dfq
            scale = torch.sqrt(last_weight_range / (first_weight_range + eps))
            scale = torch.clamp(scale, dfq_min_scale, dfq_max_scale)

        else:
            first_weight_range = torch.where(first_weight_range < first_weight_range.max() * self.channel_ratio,\
                                        first_weight_range.max() * self.channel_ratio, first_weight_range)
            last_weight_range = torch.where(last_weight_range < last_weight_range.max() * self.channel_ratio,\
                                        last_weight_range.max() * self.channel_ratio, last_weight_range)

            kernel_scale = first_weight_range.max() / (first_weight_range + eps)
            next_kernel_scale = last_weight_range.max() / (last_weight_range + eps)
            first_weight_act_range = op_act_channel_range[pair[0]]
            first_weight_act_range = torch.where(first_weight_act_range < 0.01, torch.tensor(0.01,\
                device=first_weight_act_range.device, dtype=torch.float32), first_weight_act_range)
            act_scale = first_weight_act_range.max() / (first_weight_act_range + eps)

            if algo_type == 1:
                scale = torch.min(kernel_scale, act_scale)
            elif algo_type == 2:
                kernel_scale = kernel_scale / next_kernel_scale
                act_scale = act_scale / next_kernel_scale
                scale = torch.min(kernel_scale, act_scale)
                scale = torch.min(scale, torch.tensor(ssd_min_scale, dtype=torch.float32, device=scale.device))
                scale /= scale.min()
                scale = torch.clamp(scale, 1.0, ssd_max_scale)
            else:
                kernel_scale = (kernel_scale / next_kernel_scale).sqrt()
                scale = (act_scale * kernel_scale).sqrt()
                scale = torch.clamp(scale, 1.0, ssd_max_scale)

        self.write_back(pair, scale)

    def build_observer_pair(self, pair: List[Operation]) -> Dict[Operation, OperationObserver]:
        observers = {}
        for operation in pair:
            if not isinstance(operation, QuantableOperation): continue
            observer = OperationObserver(operation=operation)
            observers[operation] = observer
        return observers

    def calibrate(
        self,
        pair: List[Operation],
        data_loader: Iterable,
        executor: BaseGraphExecutor,
        hooks: Dict[Operation, CalibrationHook],
        collate_fn: Callable,
        calib_steps: int
    ):
        calib_step = 0
        for calib_epoch in range(ceil(calib_steps / len(data_loader))):
            for _,data in enumerate(data_loader):
                if collate_fn is not None:
                    data = collate_fn(data)
                # get the input of first op
                inputs = executor.forward(data, output_names=[pair[0].inputs[0].name])
                self.run_pair(pair, inputs, hooks)
                calib_step += 1
                if calib_step >= calib_steps:
                    break

    def calibration_passive_param(self, pair: List[Operation], scale_multiplier: float=1.0):
        for op in pair:
            if not isinstance(op, QuantableOperation): continue
            if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
                if op.num_of_input == 3:
                    weight_config = op.config.input_quantization_config[1]
                    input_config = op.config.input_quantization_config[0]
                    weight_config = weight_config.dominated_by
                    input_config  = input_config.dominated_by

                    bias_config = op.config.input_quantization_config[-1]
                    if bias_config.state != QuantizationStates.PASSIVE_INIT:
                        continue

                    bias_config.scale  = weight_config.scale * input_config.scale * scale_multiplier
                    bias_config.state  = QuantizationStates.PASSIVE
                    bias_config.offset = torch.zeros_like(bias_config.scale, dtype=torch.float)
                    assert not bias_config.policy.has_property(QuantizationProperty.ASYMMETRICAL), (
                        'Negative parameter does not support ASYMMETRICAL quantization')

    def initiate_pair_state(self, pair: List[Operation]):
        for op in pair:
            if isinstance(op, QuantableOperation):
                for quant_config in op.config.input_quantization_config + op.config.output_quantization_config:
                    if quant_config.state == QuantizationStates.ACTIVATED:
                        quant_config.state = QuantizationStates.INITIAL
                    elif quant_config.state == QuantizationStates.PASSIVE:
                        quant_config.state = QuantizationStates.PASSIVE_INIT

    def dequantize_pair(self, pair: List[Operation]):
        for op in pair:
            if isinstance(op, QuantableOperation):
                op.dequantize(expire_device=None)

    def restore_quantize_state(self, pair: List[Operation]):
        for op in pair:
            if isinstance(op, QuantableOperation):
                op.restore_quantize_state(expire_device=None)

    def run_pair(self,
                 pair: List[Operation],
                 inputs: List[torch.Tensor],
                 hooks: Dict[Operation, CalibrationHook]={}) -> List[torch.Tensor]:
        for op in pair:
            inputs = inputs + [param.value for param in op.parameters]
            if isinstance(op, QuantableOperation):
                input_configs = [_ for _ in op.config.input_quantization_config]
                assert(len(inputs) == len(input_configs))
                inputs_quant = [self.quant_func(input, config) for input, config in zip(inputs, input_configs)]
                hook = hooks.get(op, None)
                if hook is not None:
                    hook.pre_forward_hook(inputs, inputs_quant, input_configs)
            else:
                inputs_quant = inputs

            f = GLOBAL_DISPATCHING_TABLE[op.platform][op.type]
            outputs = f(op, inputs_quant)
            outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]

            if isinstance(op, QuantableOperation):
                output_configs = [_ for _ in op.config.output_quantization_config]
                outputs_quant = [self.quant_func(output, config) for output, config in zip(outputs, output_configs)]
                hook = hooks.get(op, None)
                if hook is not None:
                    hook.post_forward_hook(outputs, outputs_quant, output_configs)
                inputs = outputs_quant
            else:
                inputs = outputs
        return inputs

    # mse calculation for a list of tensors
    def calculate_mse(self, fp_res: List[torch.Tensor], quant_res: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        for i in range(len(fp_res)):
            losses.append(torch_mean_square_error(fp_res[i], quant_res[i]))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def test_ssd_loss(self,
                    pair: List[Operation],
                    executor: BaseGraphExecutor,
                    data_loader: Iterable,
                    collate_fn: Callable,
                    calib_steps: int
    ) -> float:
        observers = self.build_observer_pair(pair)
        hooks = {op:observers[op].hook for op in observers}
        self.calibrate(pair, data_loader, executor, hooks, collate_fn, calib_steps)
        for _, observer in observers.items():
            observer.render_quantization_config()
        pop_list = []
        for op, observer in observers.items():
            if all([type(var_observer) not in {TorchHistObserver}
                for var_observer in observer._hook._observer_table.values()]):
                    pop_list.append(op)
        for op in pop_list:
            observers.pop(op)
            hooks.pop(op)
        if len(hooks) > 0:
            self.calibrate(pair, data_loader, executor, hooks, collate_fn, calib_steps)
            for _, observer in observers.items():
                observer.render_quantization_config()
        self.calibration_passive_param(pair)
        # calculate loss
        loss = []
        for calib_epoch in range(ceil(calib_steps / len(data_loader))):
            for _,data in enumerate(data_loader):
                if collate_fn is not None:
                    data = collate_fn(data)
                # get the input of first op
                inputs = executor.forward(data, output_names=[pair[0].inputs[0].name])
                # dequant to get fp output
                self.dequantize_pair(pair)
                fp_output = self.run_pair(pair, inputs)
                # restore quant state to get quant output
                self.restore_quantize_state(pair)
                quant_output = self.run_pair(pair, inputs)
                # mse calculation
                loss.append(self.calculate_mse(fp_output, quant_output))
        return torch.stack(loss).mean().item()

    # maintain original parameter for restoration in case of a larger loss after equalization
    def collect_original_parameter(self, pair: List[Operation]) -> Dict[Variable, torch.Tensor]:
        original_weights = {}
        for op in pair:
            for var in op.inputs + op.outputs:
                if var.is_parameter:
                    original_weights[var] = var.value.clone()
        return original_weights

    # discard original fp value and store the current value as new fp value
    def store_parameter(self, pair: List[Operation]):
        for op in pair:
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()

    # recover from maintained original parameters
    def recover_original_parameter(self,
                                pair: List[Operation],
                                original_weights: Dict[Variable, torch.Tensor],
    ):
        for op in pair:
            for var in op.inputs + op.outputs:
                if var.is_parameter:
                    var.value = original_weights[var]
        self.store_parameter(pair)

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        collate_fn: Callable,
        calib_steps: int,
        **kwargs
    ) -> None:
        # restrain maximum img number used for loss checking
        batchsize = 1
        for data in dataloader:
            if collate_fn is not None:
                data = collate_fn(data)
            if isinstance(data, torch.Tensor):
                batchsize = data.shape[0]
            elif isinstance(data, (list, tuple)):
                for value in data:
                    if isinstance(value, torch.Tensor):
                        batchsize = value.shape[0]
                        break
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        batchsize = value.shape[0]
                        break
            break

        calib_steps = min(calib_steps, ceil(200 / batchsize))

        all_pairs = self.collect_all_pairs(graph)
        if self.layer_norm:
            self.layer_weight_norm(all_pairs)

        for i in range(self.iteration):
            logger.debug(f'DFQ/SSD Equalization Iteration {i + 1}/{self.iteration}')
            for _,pair in tqdm(enumerate(all_pairs), desc=f'SSD/DFQ Equalization Iteration {i+1}/{self.iteration}', total=len(all_pairs)):
                logger.debug(f"Now Processing Pair {_ + 1}/{len(all_pairs)}: {'--'.join([op.name for op in pair])}")
                self.store_parameter(pair)

                logger.debug(f'Collecting Activation Range for Pair...')
                op_act_range = self.collect_activation_range(pair, executor, dataloader, collate_fn, calib_steps)
                logger.debug(f'Collecting Done!')

                original_weights = self.collect_original_parameter(pair)
                basic_loss = self.test_ssd_loss(pair, executor, dataloader, collate_fn, calib_steps)
                best_loss = basic_loss
                best_idx = -1

                # now apply equalization and estimate loss
                for algo in range(0,4):
                    self.one_step_equalization(pair, op_act_range, algo)
                    self.store_parameter(pair)
                    self.initiate_pair_state(pair)
                    loss = self.test_ssd_loss(pair, executor, dataloader, collate_fn, calib_steps)
                    if algo == 0:
                        logger.debug(f'DFQ Step, Loss Before Equalization {basic_loss} || Loss After Equalization {loss}')
                    else:
                        logger.debug(f'SSD Algo {algo}, Loss Before Equalization {basic_loss} || Loss After Equalization {loss}')
                    if loss < basic_loss * self.loss_threshold and loss < best_loss:
                        best_idx = algo
                        best_loss = loss
                    self.recover_original_parameter(pair, original_weights)

                if best_idx >= 0:
                    if best_idx == 0:
                        logger.debug(f'DFQ Step Activated, Loss Before Equalization {basic_loss} || Loss After Equalization {best_loss}')
                    else:
                        logger.debug(f'SSD Algo {best_idx} Activated, Loss Before Equalization {basic_loss} || Loss After Equalization {best_loss}')
                    self.one_step_equalization(pair, op_act_range, best_idx)
                    self.store_parameter(pair)
                else:
                    logger.debug('SSD and DFQ Deactivated')
                self.initiate_pair_state(pair)
