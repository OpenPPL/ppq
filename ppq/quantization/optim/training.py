from collections import defaultdict
from typing import Callable, Dict, Iterable, List

import torch
from ppq.core import *
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import (BaseGraph, BaseGraph, Operation,
                    QuantableOperation)
from ppq.IR.quantize import QuantableGraph
from ppq.quantization.algorithm.training import *
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction
from tqdm import tqdm

from .base import QuantizationOptimizationPass


class TrainingBasedPass(QuantizationOptimizationPass):
    """Training Based Pass is a basic class that provides necessary function
    for all training optimizition passes. Optimization will be more stable and
    accurate with functions provided by this pass. (Might be a little slower).

    This pass will collect result of interested outputs after optimization and
        check if the optimized result has a lower SNR. If so, the optimization will be
        accepted, layer weight will be updated, otherwise optimization will be rejected and
        takes no effects.

    Choose interested_outputs carefully, cause we compare loss only with those output variables.
        If interested_outputs is None, all graph output variables will be chosen.

    YOUR SHOULD NOTICE THAT SNR REFERS TO: POWER OF NOISE / POWER OF SIGNAL IN PPQ.

    Args:
        QuantizationOptimizationPass ([type]): [description]
    """
    def __init__(self, name: str = 'Default Quanzation Optim',
                 interested_outputs: List[str] = None, verbose: bool = True) -> None:
        self._loss_fn = torch_snr_error
        self._interested_outputs = interested_outputs
        self._checkpoints = {}
        self._verbose = verbose
        self._quant_state_recorder = {}
        super().__init__(name=name)

    @ empty_ppq_cache
    def initialize_checkpoints(
        self, graph: BaseGraph, executor: BaseGraphExecutor,
        dataloader: Iterable, collate_fn: Callable):
        """
        Establish a series of network checkpoints with your network.
            Checkpoint is a data structure that helps us compare quant results and fp32 results.
        Args:
            graph (BaseGraph): [description]
            executor (BaseGraphExecutor): [description]
            dataloader (Iterable): [description]
            collate_fn (Callable): [description]

        Raises:
            PermissionError: [description]
        """
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                        raise PermissionError('Can not initialize checkpoints when weight value is baked. '
                                              f'Variable {var.name} has a baked value.')

        if self._interested_outputs is None or len(self._interested_outputs) == 0:
            self._interested_outputs = [name for name in graph.outputs]

        for name in self._interested_outputs:
            self._checkpoints[name] = FinetuneCheckPoint(variable=name)

        # dequantize graph, collect references
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.dequantize()

        for data in tqdm(dataloader, desc='Collecting Referecens'):
            if collate_fn is not None: data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=self._interested_outputs)
            for name, output in zip(self._interested_outputs, outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.push(tensor=output, is_reference=True)

        # restore quantization state:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.restore_quantize_state()

        # update state
        verbose, self._verbose = self._verbose, False
        self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn)
        self._verbose = verbose

    def check(self, executor: BaseGraphExecutor,
        dataloader: Iterable, collate_fn: Callable):
        """Check quantization error with a given dataloader with current
        checkpoints. Return whether quantization error is lower than before.

        Args:
            executor (BaseGraphExecutor): [description]
            dataloader (Iterable): [description]
            collate_fn (Callable): [description]

        Returns:
            [type]: [description]
        """

        # step - 1, collecting data
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=self._interested_outputs)
            for name, output in zip(self._interested_outputs, outputs):
                self._checkpoints[name].push(tensor=output, is_reference=False)

        # step - 2, calculating loss
        losses = []
        for name in self._interested_outputs:
            ckpt = self._checkpoints[name]
            assert isinstance(ckpt, FinetuneCheckPoint)
            qt_out, fp_out = ckpt.pop()
            qt_out = torch.cat([tensor for tensor in qt_out])
            fp_out = torch.cat([tensor for tensor in fp_out])
            losses.append(self._loss_fn(y_pred=qt_out, y_real=fp_out).item())
            ckpt.clear()

        # step - 3, comparing loss
        loss_now, loss_old = sum(losses), sum([ckpt.best_loss for ckpt in self._checkpoints.values()])
        loss_now, loss_old = loss_now / len(losses), loss_old / len(losses)
        if self._verbose: print(f'NOISE-SIGNAL RATIO: {loss_old * 100 :.4f}% -> {loss_now * 100:.4f}%.')

        # if there is a loss drop, update all losses.
        if loss_old > (loss_now * CHECKPOINT_TOLERANCE):
            for idx, name in enumerate(self._interested_outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.best_loss = losses[idx]
            return True
        return False

    def optimize(
        self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        raise NotImplementedError('Can not invoke this function. '
                                  'Please inherit this class and give an implementation to override this function.')

    @ empty_ppq_cache
    def enable_block_gradient(self, block: TrainableBlock):
        """
        Make all tensors inside a given block to be trainable(requres_grad = True)
        Both quantization scale and weight itself are going to be trained in training procedure

        Args:
            block (TrainableBlock): _description_
        """
        for op in block.rps:
            for var in (op.inputs + op.outputs):
                if var.is_parameter and isinstance(var.value, torch.Tensor):
                    # PATCH 2022 08 01 Clip op can not be train
                    if op.type == 'Clip': continue
                    if var.value.dtype == torch.float:
                        var.value.requires_grad = True
            if isinstance(op, QuantableOperation):
                for cfg, _ in op.config_with_variable:
                    if isinstance(cfg.scale, torch.Tensor):
                        cfg.scale.requires_grad = True

    @ empty_ppq_cache
    def disable_block_gradient(self, block: TrainableBlock):
        for op in block.rps:
            for var in (op.inputs + op.outputs):
                if var.is_parameter and isinstance(var.value, torch.Tensor):
                    if var.value.is_leaf:
                        var.value.requires_grad = False
                        var.value._grad = None
            if isinstance(op, QuantableOperation):
                for cfg, _ in op.config_with_variable:
                    if isinstance(cfg.scale, torch.Tensor):
                        if cfg.scale.is_leaf:
                            cfg.scale.requires_grad = False
                            cfg.scale._grad = None

    def split_graph_into_blocks(
        self, graph: BaseGraph, executing_order: List[Operation],
        blocksize: int = None, overlap: bool = False, 
        interested_layers: List[str] = None) -> List[TrainableBlock]:
        """block construction function for training-based algorithms, if
        `block_limit` is not specified, block grandularity will be controlled by
        the default value OPTIM_ADVOPT_GRAPH_MAXSIZE specified in ppq.core.common.

        Args:
            graph (BaseGraph): ppq ir graph
            executing_order (List[Operation]): topo search order
            block_limit (int, optional): controls maximum depth of a block. Defaults to None.

        Returns:
            List[TrainableBlock]: list of all partitioned blocks
        """
        if blocksize is None: blocksize = OPTIM_ADVOPT_GRAPH_MAXDEPTH
        visited_ops, blocks = set(), []
        block_builder = BlockBuilder(graph=graph, topo_order=executing_order)

        for op in graph.operations.values():
            # start from computing op
            if op in visited_ops and overlap is False: continue
            if isinstance(op, QuantableOperation) and op.is_computing_op:
                block = block_builder.build(op, blocksize)
                # by default blocks are exclusive from each other
                for op in block.rps: visited_ops.add(op)
                blocks.append(block)

        ret = []
        if interested_layers is None or len(interested_layers) == 0:
            ret = blocks # if no interested_layers, finetune all.
        else:
            for candidate in blocks:
                assert isinstance(candidate, TrainableBlock)
                if any([op.name in interested_layers for op in candidate.rps]):
                    ret.append(candidate)
        return ret

    def collect(
        self, graph: BaseGraph, block: TrainableBlock, executor: TorchExecutor, 
        dataloader: Iterable, collate_fn: Callable, collecting_device: str, steps: int = None
        ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Collect training data for given block.
        This function will collect fp32 output and quantized input data by
            executing your graph twice.
        For collecting fp32 output, all related operations will be dequantized.
        For collecting quantized input, all related operations' quantization state will be restored.

        collecting device declares where cache to be stored:
            executor - store cache to executor device.(default)
            cpu      - store cache to system memory.
            cuda     - store cache to gpu memory.(2x speed up)
            disk     - not implemented.

        Args:
            block (TrainableBlock): _description_
            executor (TorchExecutor): _description_
            dataloader (Iterable): _description_
            collate_fn (Callable): _description_
            collecting_device (str): _description_

        Returns:
            _type_: _description_
        """
        def cache_fn(data: torch.Tensor):
            # TODO move this function to ppq.core.IO
            if not isinstance(data, torch.Tensor):
                raise TypeError('Unexpected Type of value, Except network output to be torch.Tensor, '
                                f'however {type(data)} was given.')
            if collecting_device == 'cpu': data = data.cpu()
            if collecting_device == 'cuda': data = data.cuda()
            # TODO restrict collecting device.
            return data

        with torch.no_grad():
            try:
                if len(dataloader) > 1024:
                    ppq_warning('Large finetuning dataset detected(>1024). '
                                'You are suppose to prepare a smaller dataset for finetuning. '
                                'Large dataset might cause system out of memory, '
                                'cause all data are cache in memory.')
            except Exception as e:
                pass # dataloader has no __len__

            quant_graph = QuantableGraph(graph) # helper class
            fp_outputs, qt_inputs = [], []
            
            cur_iter = 0
            # dequantize graph, collect fp32 outputs
            quant_graph.dequantize_graph()
            for data in dataloader:
                if collate_fn is not None: data = collate_fn(data)
                fp_output = executor.forward(data, [var.name for var in block.ep.outputs])
                fp_output = {var.name: cache_fn(data) for data, var in zip(fp_output, block.ep.outputs)}
                fp_outputs.append(fp_output)
                cur_iter += 1
                if steps is not None and cur_iter > steps: break

            cur_iter = 0
            # restore quantization state, collect quant inputs
            quant_graph.restore_quantize_state()
            for data in dataloader:
                if collate_fn is not None: data = collate_fn(data)
                # PATCH 20220829, 有些 computing op 权重并非定值
                non_constant_input = [var for var in block.sp.inputs if not var.is_parameter]
                qt_input = executor.forward(data, [var.name for var in non_constant_input])
                qt_input = {var.name: cache_fn(value) for var, value in zip(non_constant_input, qt_input)}
                qt_inputs.append(qt_input)
                cur_iter += 1
                if steps is not None and cur_iter > steps: break

        return qt_inputs, fp_outputs 

    def compute_block_loss(
        self, block: TrainableBlock,
        qt_inputs: List[Dict[str, torch.Tensor]], fp_outputs: List[Dict[str, torch.Tensor]],
        executor: TorchExecutor, loss_fn: Callable=torch_mean_square_error
    ) -> float:
        """
        loss computing for fp32 and quantized graph outputs, used
        in multiple training-based algorithms below

        Args:
            output_names (List[str]): output variable names
            graph (BaseGraph): ppq ir graph
            dataloader (Iterable): calibration dataloader
            collate_fn (Callable): batch collate func
            executor (TorchExecutor): ppq torch executor
            loss_fn (Callable, optional): loss computing func. Defaults to torch_mean_square_error.
        Returns:
            Dict[str, float]: loss dict for variables specified in `output_names`
        """
        with torch.no_grad():
            losses       = {var.name: 0.0 for var in block.ep.outputs}
            output_names = [var.name for var in block.ep.outputs]

            for qt_input, fp_output in zip(qt_inputs, fp_outputs):
                feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}

                qt_output = executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict,
                    output_names=output_names)

                for name, quant_output in zip(output_names, qt_output):
                    batch_loss = loss_fn(quant_output, fp_output[name].to(executor._device))
                    losses[name] += batch_loss.detach().item()

            for name in losses: losses[name] /= len(qt_inputs)
        return sum([v for v in losses.values()])


class BiasCorrectionPass(TrainingBasedPass):
    """
    ## Bias Correction Optimization Pass(Bias 校准过程)

    Bias correction is the process of shifting quantized model outputs to account for their statistical errors.

    Network quantization will bring some error(noise) to the result. To improve the accuracy of a quantized model, 
    we can correct the network by adding an extra term on bias in order to make the output has zero expectation. 
    
    Bias correction is used to eliminate bias error, generally it will take a few minutes to correct all bias terms.

    For those layers have no bias, Bias Correction Optimization will skip them directly.

        let: Y = WX + b

        Quant(Y) = Qunat(W) Quant(X) + b

        bias_error = reduce_mean(Y - Quant(Y))

        This pass will correct bias with: b = b + bias_error

    ### Parameters:

    * interested_layers(List[str]):

            A list of operation names, only the layers listed in this parameter will be processed.

            If interested_layers is None, all layers will be processed.

    * steps(int)

            Forward steps for collecting bias error, 
            a large value of this parameter means more data will be collected so 
            the bias error will be estimated better, while it takes more time.

            Usually 8 ~ 32 step is enough in most cases.

    * block_size(int)

            Bias Correction Optimization will split your graph into blocks, 
            bias error will be collected and corrected block by block.

            A large block size will greatly reduce running time of this optimization, 
            while it might give an unstable result when blocksize is too large.

            By default this value is set to 4, to have the best result of optimization, you are recommended to set blocksize = 1.

    * loss_fn(Callable)

            A function that used to measure the loss after optimization.

            Bias Correction Optimization is a training-based pass, 
            we will check the loss at the end of block optimization.

            If the optimization created worsen result, the optimization result will be drop.

    ### Usage:

    Bias Correction Optimization Pass should be invoked after Runtime Calibration Pass.

    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.bias_correct = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)

    You can manually create this optimization by:

        from ppq import BiasCorrectionPass

        optim = BiasCorrectionPass()

    ### Version:

    Require PPQ 0.5.2 +

    Interface changed since PPQ 0.6.5
    """
    def __init__(self, interested_layers: List[str] = [], 
                 collecting_device: str = 'cuda',
                 steps: int = 32, block_size: int = 1) -> None:

        super().__init__(name='PPQ Bias Correction Pass')
        self.interested_layers = interested_layers
        self.steps             = steps
        self.block_size        = block_size
        self.collecting_device = collecting_device
        self.loss_fn           = torch_mean_square_error

    @ empty_ppq_cache
    def correct_bias(
        self, qt_inputs: List[Dict[str, torch.Tensor]], fp_outputs: List[Dict[str, torch.Tensor]],
        block: TrainableBlock, executor: TorchExecutor, graph: BaseGraph) -> Tuple[float, float]:

        def collect_bias(output: torch.Tensor, op_type: str) -> torch.Tensor:
            if output.ndim < 1: raise ValueError('Forward value has an unexpected dimension.')
            if op_type in {'Conv', 'ConvTranspose'}:
                # for convolution layer, bias always been added on axis 1
                reduce_dims = [i for i in range(output.ndim) if i != 1]
                return torch.mean(output, dim=reduce_dims).unsqueeze(0)
            elif op_type in {'Gemm'}:
                # for convolution layer, bias always been added on axis -1
                reduce_dims = [i for i in range(output.ndim) if i != (output.ndim - 1)]
                return torch.mean(output, dim=(0, )).unsqueeze(0)
            else: raise TypeError(f'Unsupported Operation type: {op_type}')
        
        def dequantize_block(block: TrainableBlock):
            for op in block.rps: 
                if isinstance(op, QuantableOperation): 
                    op.dequantize()
    
        def restore_block_quantize_state(block: TrainableBlock):
            for op in block.rps: 
                if isinstance(op, QuantableOperation): 
                    op.restore_quantize_state()

        with torch.no_grad():
            # record pre training loss.
            pre_loss = self.compute_block_loss(
                block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs,
                executor=executor, loss_fn=self.loss_fn)

            bias_cloned, interested_outputs, fp_cache, qt_cache = {}, [], defaultdict(list), defaultdict(list)
            for op in block.rps:
                op_check = op.type in {'Conv', 'ConvTranspose', 'Gemm'} and len(op.inputs) == 3
                bias_check = op.inputs[-1].is_parameter
                type_check = isinstance(op.inputs[-1].value, torch.Tensor)

                if op_check and bias_check and type_check:
                    bias_cloned[op.name] = op.inputs[-1].value.clone()
                    interested_outputs.append(op.outputs[0].name)

            # Phrase 1, collect fp32 output
            dequantize_block(block)
            for qt_input, _ in tqdm(zip(qt_inputs, fp_outputs), '# Bias Correction Phrase 1'):
                feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}

                outputs = executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict, 
                    output_names=interested_outputs)

                for name, value in zip(interested_outputs, outputs):
                    source_op = graph.variables[name].source_op
                    fp_cache[name].append(collect_bias(value, source_op.type))

            # Phrase 2, collect quant output
            restore_block_quantize_state(block)
            for qt_input, _ in tqdm(zip(qt_inputs, fp_outputs), '# Bias Correction Phrase 2'):
                feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}

                outputs = executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict, 
                    output_names=interested_outputs)

                for name, value in zip(interested_outputs, outputs):
                    source_op = graph.variables[name].source_op
                    qt_cache[name].append(collect_bias(value, source_op.type))

            # correct bias error
            for name in fp_cache:
                DC_term_fp = fp_cache[name]
                DC_term_qt = qt_cache[name]

                if len(DC_term_fp) == 0 or len(DC_term_qt) == 0:
                    raise ValueError('Bias correction failed, No data was collected.')
                DC_term_fp = torch.mean(torch.cat(DC_term_fp, axis=0), dim=0)
                DC_term_qt = torch.mean(torch.cat(DC_term_qt, axis=0), dim=0)
                bias_error = DC_term_fp - DC_term_qt

                source_op = graph.variables[name].source_op
                source_op.inputs[-1].value += bias_error

            # record pre training loss.
            post_loss = self.compute_block_loss(
                block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs,
                executor=executor, loss_fn=self.loss_fn)
            
            # loss check
            if post_loss > pre_loss:
                for name, value in bias_cloned.items():
                    op = graph.operations[name]
                    op.inputs[-1].value.copy_(value)
                post_loss = pre_loss

        return pre_loss, post_loss

    @ empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        collate_fn: Callable,
        **kwargs
    ) -> None:

        blocks = self.split_graph_into_blocks(
            graph=graph, executing_order=executor._executing_order,
            blocksize=self.block_size, interested_layers=self.interested_layers)

        # ready for finetuning, print information.
        print('')
        print('Check following parameters:')
        print(f'Interested Layers:         {self.interested_layers}')
        print(f'Num of blocks:             {len(blocks)}')
        print(f'Steps:                     {self.steps}')
        print(f'collecting_device:         {self.collecting_device}')
        print('') # blank line

        # do per-block finetune
        for block_idx, block in enumerate(blocks):
            qt_inputs, fp_outputs = self.collect(
                graph=graph, block=block, executor=executor, 
                dataloader=dataloader, collate_fn=collate_fn, 
                collecting_device=self.collecting_device)

            print(f'# Block [{block_idx + 1} / {len(blocks)}]: '
                  f'[{block.sp.name} -> {block.ep.name}]')
            pre_loss, post_loss = self.correct_bias(
                qt_inputs=qt_inputs, fp_outputs=fp_outputs,
                block=block, executor=executor, graph=graph)
            print(f'# Tuning Finished  : ({pre_loss:.4f} -> {min(pre_loss, post_loss):.4f}) [Block Loss]')
            print('') # blank line


class LearnedStepSizePass(TrainingBasedPass):
    """
    ## Learned Step Size Pass(网络微调过程-LSQ)

    Learned Step Size optimization, a training-based optimization pass that tunes weights and scales for high precision quantization.

    [This method is proposed by Steven K. Esser] (https://arxiv.org/pdf/1902.08153.pdf)

    This is an alternative version of LSQ, this pass will split your graph into multiple trainable blocks, each blocks will be trained separately.
    Warning: PPQ Learned Step Size minimize only the output loss of each block, which means after training the internal results probably goes far away from original. 

    PPQ Learned Step Size optimization requires 256 ~ 2048 samples for finetuning your network, while the data label is not necessary. All training data are cache in GPU memory or CPU memory for acceleration.

    The training loss will be computed as:

        let: Y = WX + b

        Quant(Y, scale_Y) = Qunat(W, scale_W) Quant(X, scale_X) + b

        loss = loss_func(Y, Quant(Y, scale_Y)) # loss between fp output and int8 output, that is why we do not need labeled data.

    The formula of calculating the derivatives of y and scale_Y:

        if y > scale_Y * -128 and y < scale_Y * 127:
        dQuant(y, scale_Y)/dy       = dQuant(y, scale_Y)
        dQuant(y, scale_Y)/dscale_Y = Quant(y, scale_Y) - y

        if y < scale_Y * -128:
        dQuant(y, scale_Y)/dy       = 0
        dQuant(y, scale_Y)/dscale_Y = -128

        if y > scale_Y * 127:
        dQuant(y, scale_Y)/dy       = 0
        dQuant(y, scale_Y)/dscale_Y = 127

    ### Parameters:

    * interested_layers(List[str]):

            A list of operation names, only the layers listed in this parameter will be trained.

            If interested_layers is None, all layers(conv and gemm) will be trained.

    * steps(int)

            Training steps for finetuning your network, default is 500.

    * block_size(int)

            PPQ Learned Step Size optimization split your graph into blocks at first, 
            each block will be finetuned separately.

            A large block size will greatly reduce running time of this optimization,
            while it might give an unstable result when blocksize is too large.

            By default this value is set to 4.

    * is_scale_trainable(bool)

            If is_scale_trainable = False, optimization will not apply to scales, only network parameters will be tuned.

            Scale is trainable when all the following conditions are fulfilled:

                1. scale is valid
                2. corresponding tensor quantization config state is active
                3. do not have POWER_OF_2 policy
                4. is_scale_trainable = True

    * gamma(float)

            A regularization term for minimize the distance of Y and Quant(Y)

            If gamma is not 0, loss = loss_func(Y, Quant(Y, scale_Y)) + MSE(Y, Quant(Y)) * gamma

            Default is 0

    * lr(float)

            Learning rate, Default is 5e-5

    * collecting_device(str)

            String that representing the device on which cache data is or will be allocated.

            Can be cpu, cuda, disk

    * loss_fn(Callable)

            A function that used to measure the loss after optimization.

            LSQ is a training-based pass, 
            we will check the loss at the end of block optimization.

            If the result goes worsen, optimized weights and scales will be drop.

    ### Usage:

    LSQ Optimization Pass should be invoked after Runtime Calibration Pass.

    This pass is inclueded in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.lsq_optimization = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
        collate_fn=collate_fn)

    You can manually create this optimization by:

        from ppq import LearnedStepSizePass

        optim = LearnedStepSizePass()


    ### Block-partitioning Algorithm

    PPQ use Block-partitioning algorithm for spliting your graph into blocks, this method is based on graph theory.

    Parameter block_size will controls the maximum size of created blocks.

    If block_size = 1, then each block will contains exactly 1 layer within it, blockwise optimization will degenerate to layerwise optimization then.

    If block_size is set to a large value, training progress will be unstable since batchnorm layers have been merged at first.

    ### Version:

    Require PPQ 0.6.2 +

    Interface changed since PPQ 0.6.5
    """
    def __init__(
        self, name: str = 'PPQ LSQ Optimization', interested_layers: List[str] = [],
        steps: int = 500, gamma: float = 0.0, is_scale_trainable: bool = True,
        lr: float = 5e-5, block_size: int = None,
        collecting_device: str = 'cuda', loss_fn: Callable = torch_mean_square_error,
    ) -> None:
        super().__init__(name=name)
        self.interested_layers  = interested_layers
        self.collecting_device  = collecting_device
        self.is_scale_trainable = is_scale_trainable
        self.block_size         = block_size
        self.loss_fn            = loss_fn
        self.gamma              = gamma
        self.steps              = steps
        self.lr                 = lr

    def finetune(
        self, steps: int, learning_rate: float, block: TrainableBlock, executor: TorchExecutor,
        qt_inputs: List[Dict[str, torch.Tensor]], fp_outputs: List[Dict[str, torch.Tensor]], 
        optimizer: torch.optim.Optimizer=None, scheduler: object=None) -> Tuple[float, float]:

        # step - 1: enable gradient for training.
        self.enable_block_gradient(block)

        # record pre training loss.
        pre_loss = self.compute_block_loss(
            block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs,
            executor=executor, loss_fn=self.loss_fn)

        # collect trainable params
        trainable_params, delegators = [], {}
        trainable_scales = []
        for op in block.rps:
            if not isinstance(op, QuantableOperation): continue

            if op.is_computing_op: 
                for var in op.inputs[1:]: 
                    if var.is_parameter:
                        trainable_params.append(var.value)

            # register quant delegator
            for cfg, var in op.config_with_variable:
                if cfg.state in {QuantizationStates.ACTIVATED, QuantizationStates.SLAVE}:
                    delegator = LSQDelegator(config=cfg, var=var)
                    trainable_scales.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator

        # check if empty.
        tensors = [tensor for tensor in trainable_params + trainable_scales if tensor.requires_grad]
        if len(tensors) == 0:
            for cfg, delegator in delegators.items():
                executor.remove_quantize_delegate(config=cfg)
            return 0, 0

        # initilize optimizer.
        if optimizer is None:
            optimizer = torch.optim.Adam(tensors, lr=learning_rate)

        dataset_length = len(qt_inputs)
        if dataset_length == 0: raise ValueError('Dataset is empty.')

        # step 2 - training procedure
        for idx in tqdm(range(steps), desc='# Tuning Procedure '):
            qt_input, fp_output = qt_inputs[idx % dataset_length], fp_outputs[idx % dataset_length]

            # forward
            optimizer.zero_grad()
            feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}
            output_names = [name for name in fp_output]

            qt_output = executor.partial_graph_forward(
                operations=block.rps, feed_dict=feed_dict, 
                output_names=output_names)

            # compute loss
            loss = 0.0
            for idx, name in enumerate(output_names):
                loss += self.loss_fn(qt_output[idx], fp_output[name].to(executor._device))

            for op in block.rps:
                if self.gamma == 0: continue
                if isinstance(op, QuantableOperation) and op.is_computing_op:
                    weight  = op.inputs[1].value
                    wconfig = op.config.input_quantization_config[1]
                    loss += torch_mean_square_error(
                        weight, PPQLinearQuantFunction(weight, wconfig)) * self.gamma

            # backward from loss
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()

        # step - 3: record post training loss
        post_loss = self.compute_block_loss(
            block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs,
            executor=executor, loss_fn=self.loss_fn)
        # check and withdraw
        if post_loss > pre_loss:
            for cfg, delegator in delegators.items():
                delegator.withdraw()

        for cfg, delegator in delegators.items():
            delegator.finalize()
            executor.remove_quantize_delegate(config=cfg)

        # disable gradient for evaluation.
        self.disable_block_gradient(block)
        
        # clear cache
        torch.cuda.empty_cache()
        return pre_loss, post_loss

    def optimize(
        self, graph: BaseGraph,
        dataloader: Iterable, executor: BaseGraphExecutor,
        collate_fn: Callable, **kwargs) -> None:

        blocks = self.split_graph_into_blocks(
            graph=graph, executing_order=executor._executing_order,
            blocksize=self.block_size, interested_layers=self.interested_layers)

        # ready for finetuning, print information.
        print('')
        print('Check following parameters:')
        print(f'Is Scale Trainable:        {self.is_scale_trainable}')
        print(f'Interested Layers:         {self.interested_layers}')
        print(f'Collecting Device:         {self.collecting_device}')
        print(f'Num of blocks:             {len(blocks)}')
        print(f'Learning Rate:             {self.lr}')
        print(f'Steps:                     {self.steps}')
        print(f'Gamma:                     {self.gamma}')
        print('') # blank line

        # do per-block finetune
        for block_idx, block in enumerate(blocks):
            # collect data for training
            qt_inputs, fp_outputs = self.collect(
                graph=graph, block=block, executor=executor, 
                dataloader=dataloader, collate_fn=collate_fn, 
                collecting_device=self.collecting_device)

            print(f'# Block [{block_idx + 1} / {len(blocks)}]: '
                  f'[{block.sp.name} -> {block.ep.name}]')
            pre_loss, post_loss = self.finetune(
                steps=self.steps, learning_rate=self.lr, block=block, 
                qt_inputs=qt_inputs, fp_outputs=fp_outputs, executor=executor)
            print(f'# Tuning Finished  : ({pre_loss:.4f} -> {min(pre_loss, post_loss):.4f}) [Block Loss]')
            print('') # blank line
