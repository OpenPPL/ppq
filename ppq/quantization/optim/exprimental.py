from typing import Callable, Iterable, List

import torch
from ppq.core import *
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import BaseGraph, QuantableOperation
from ppq.quantization.algorithm.training import *
from ppq.quantization.algorithm.exprimental import BanditDelegator
from ppq.quantization.measure import torch_snr_error
from ppq.quantization.optim.base import QuantizationOptimizationPass
from ppq.utils.ema import EMARecorder
from torch.cuda import empty_cache
from tqdm import tqdm

from .training import TrainingBasedPass


class LearningToCalibPass(TrainingBasedPass):
    """This is an Experimental Pass, do not invoke.

    PPQ Learning Based Calibration Pass
    For int8 quantization, you need to calibrate or estimate the value range,
        i.e, (min, max) of all floating-point tensors in the model.

    Choose value range carefully is really importance procedure during quantization.
        Usually we use methods like MSE, Percentile, KL to solve a good value range
        from prospective view, while this pass offers you another possibility.

    This pass will make all your quantization range as trainable, and learn to quantize
        your network with sampling methods.

    ATTENTION: YOU SHALL USE THIS FUNCTION AFTER ACTIVATIONS HAVE BEEN CORRECTLY CALIBRATED
        SINCE THIS FUNCTION NEEDS A SCALE AND OFFSET AS INITIALIZED VALUE.

    ATTENTION: ONLY CONFIGURATION WITH STATE "ACTIVATED" WILL BE TUNED VIA THIS FUNCTION.
    """

    def __init__(self, method: str = 'e-greedy',
                 calib_act: bool = True, calib_weight: bool = True) -> None:
        self.method            = method
        self.calib_act         = calib_act
        self.calib_weight      = calib_weight
        self.target_step       = 7500
        self.e                 = 0.1
        self.collecting_device = 'cuda'
        self.arms              = [1, 0.9, 1.1, 0.7, 1.3]
        # for power-of-2 policy, use bandit like [0.5, 1, 2]
        super().__init__('RL Based Calibration Pass')

    @ torch.no_grad()
    def calib_block(self, quant_inputs: List[torch.Tensor], fp32_outputs: List[torch.Tensor],
        executor: TorchExecutor, block: TrainableBlock, dataloader: Iterable, collate_fn: Callable):

        # create trainable delegators for each parameter.
        delegators = []
        for operation in block.rps:
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state == QuantizationStates.ACTIVATED:
                        delegators.append(BanditDelegator(arms=self.arms, config=cfg))
        delegators = [d for d in delegators if isinstance(d, BanditDelegator)]
        dataset = RandomMemDataset(data=[[qt, fp] for qt, fp in zip(quant_inputs, fp32_outputs)])
        device  = executor._executing_context.executing_device
        loss_ema    = EMARecorder(beta=0.98)
        output_var  = block.ep.outputs[0]
        input_var   = block.sp.inputs[0]

        for delegator in delegators:
            executor.register_quantize_delegate(config=delegator.config, delegator=delegator)

        cur_iter = 0
        with tqdm(total=self.target_step) as t:
            while cur_iter < self.target_step:
                qt_input, fp_output = dataset.pop()
                qt_input, fp_output = qt_input.to(device), fp_output.to(device)

                qt_output = executor.partial_graph_forward(
                    operations=block.rps, feed_dict={input_var.name: qt_input},
                    output_names=[output_var.name])[0]

                loss = torch_snr_error(y_pred=qt_output, y_real=fp_output).item()
                for delegator in delegators: delegator.mark(1 - loss)
                loss_ema.push(loss)

                cur_iter += 1
                if cur_iter % 50 == 0:
                    t.set_description(desc=f'Block [{self._bidx + 1}/{self._num_of_blocks}]')
                    t.set_postfix(loss = loss_ema.pop())
                    t.update(50)

        for delegator in delegators:
            executor.remove_quantize_delegate(config=delegator.config)
            delegator.finalize()

        if not self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn):
            for delegator in delegators:
                delegator.withdraw()

    def collect_training_data(
        self, output_name: str,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        collate_fn: Callable) -> List[List[torch.Tensor]]:

        output_collector = []
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            [output] = executor.forward(data, output_names=[output_name])
            output_collector.append(output.to(self.collecting_device))
        return output_collector

    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: TorchExecutor,
                 collate_fn: Callable, **kwargs) -> None:

        block_builder = BlockBuilder(graph=graph, topo_order=executor._executing_order)

        # check if there is any baked value inside your graph
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                        raise PermissionError('Can not apply advanced optimization pass when weight value is baked. '
                                              f'Variable {var.name} has a baked value.')

        # build all blocks, drop overlapped layers.
        blocks, visited = [], set()
        for op in graph.operations.values():
            if op in visited: continue
            block = block_builder.build(op, limit=OPTIM_ADVOPT_GRAPH_MAXDEPTH)

            # PATCH 20220317 drop block that has no computing op.
            if all([rp.is_computing_op == False for rp in block.rps]): continue
            if block.sp.is_computing_op == False: continue

            for rp in block.rps: visited.add(rp)
            blocks.append(block)

        self.initialize_checkpoints(
            graph=graph, executor=executor,
            dataloader=dataloader, collate_fn=collate_fn)

        block_builder = BlockBuilder(graph=graph, topo_order=executor._executing_order)
        for bidx, block in enumerate(blocks):
            self._bidx, self._num_of_blocks = bidx, len(blocks)
            assert isinstance(block, TrainableBlock)

            end_op       = block.ep
            block_input  = block.sp.inputs[0]
            block_output = end_op.outputs[0]

            # dequantize prefix operations and block operations
            for op in graph.operations.values():
                if isinstance(op, QuantableOperation):
                    op.dequantize()
                    # can not use dequantize_immediately cause weight has been changed.
                    # self.dequantize_immediately(op)

            fp32_outputs = self.collect_training_data(
                output_name=block_output.name, dataloader=dataloader,
                executor=executor, collate_fn=collate_fn)

            # quantize prefix operations and block operations
            for op in graph.operations.values():
                if isinstance(op, QuantableOperation):
                    op.restore_quantize_state()

            quant_inputs = self.collect_training_data(
                output_name= block_input.name, dataloader=dataloader,
                executor=executor, collate_fn=collate_fn)

            # start training, solve the best parameters
            self.calib_block(
                quant_inputs=quant_inputs, fp32_outputs=fp32_outputs,
                executor=executor, block=block,
                dataloader=dataloader, collate_fn=collate_fn)

            # empty cache.
            fp32_outputs.clear()
            quant_inputs.clear()
            empty_cache()


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

    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        splitting_layers = []
        for name in self.interested_layers:
            if name not in graph.operations:
                raise ValueError(f'Can not Split layer {name}, can not find it in current graph.')

        for operation in graph.operations.values():
            if operation.name in self.interested_layers:
                assert operation.type in {'Conv', 'Gemm'}, (
                    f'Can not split layer, cause layer type is not support')
                splitting_layers.append(operation)

        for operation in splitting_layers:
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

            splitted = Operation(
                name=operation.name + '_splited',
                op_type=operation.type,
                attributes=attributes,
                platform=operation.platform
            )

            graph.insert_op_on_var(
                inserting_op=splitted, var=operation.outputs[0].name)
            if operation.outputs[0].name in graph.outputs:
                graph.outputs.pop(operation.outputs[0].name)
                graph.outputs[splitted.outputs[0].name] = splitted.outputs[0]

            # add weight link
            spilted_w = Variable(name=splitted.name + '_weight',
                                 value=b, is_parameter=True)
            graph.append_variable(spilted_w)

            splitted.inputs.append(spilted_w)
            spilted_w.dest_ops.append(splitted)

            # if has bias, relink bias
            if len(operation.parameters) > 1:
                bias_var = operation.parameters[-1]
                bias_var.dest_ops.remove(operation)
                bias_var.dest_ops.append(splitted)
                splitted.inputs.append(bias_var)
                operation.inputs.remove(bias_var)

