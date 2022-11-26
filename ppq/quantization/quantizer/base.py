from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Iterable, Union

import torch
from ppq.api.setting import *
from ppq.core import (OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationStates, RoundingPolicy, TargetPlatform,
                      TensorQuantizationConfig, empty_ppq_cache)
from ppq.executor import BaseGraphExecutor
from ppq.IR import (BaseGraph, GraphReplacer, Operation, QuantableGraph,
                    QuantableOperation, QuantableVariable)
from ppq.IR.base.command import QuantizeOperationCommand
from ppq.quantization.optim import *


class BaseQuantizer(metaclass = ABCMeta):
    def __init__(
        self,
        graph: BaseGraph,
        verbose: bool = True
    ) -> None:
        """

        Args:
            graph (BaseGraph): _description_
            verbose (bool, optional): _description_. Defaults to True.

        Raises:
            TypeError: _description_
        """
        if not isinstance(graph, BaseGraph):
            raise TypeError(f'To initialize a Quantizer, a BaseGraph instance is needed.'\
                f' While {type(graph)} was givne, if your graph is maintained by GraphCommandProcessor, '\
                'use GraphCommandProcessor.graph here instead.')
        self._verbose   = verbose
        self._graph     = graph
        self._processor = QuantableGraph(GraphReplacer(self._graph))

    @ empty_ppq_cache
    def quantize(
        self,
        inputs: Union[torch.Tensor, list, dict],
        calib_dataloader: Iterable,
        executor: BaseGraphExecutor,
        setting: QuantizationSetting,
        **kwargs
    ) -> None:
        # step - 1, prequant pipeline:
        # prequant pipeline will change your network structure and float value.
        prequant_pipeline = self.build_prequant_pipeline(
            setting, executor=executor)
        prequant_pipeline.optimize(
            graph=self._graph,
            dataloader=calib_dataloader,
            executor=executor,
            verbose=self._verbose,
            **kwargs)
        
        # step - 2, quantize all operations
        executor.load_graph(self._graph)
        executor.tracing_operation_meta(inputs=inputs)
        
        for op_name, operation in self._graph.operations.items():
            if (operation.platform == TargetPlatform.UNSPECIFIED):
                if operation.type in self.quant_operation_types:
                    operation.platform = self.target_platform
                else: operation.platform = TargetPlatform.FP32

            if TargetPlatform.is_quantized_platform(operation.platform):
                self.quantize_operation(op_name)

        # quantize operation will modify network structure
        # it is necessary calling self._executor before further execution
        # step - 3, calling graph optimization pipeline
        executor.load_graph(self._graph)
        quant_pipeline = self.build_quant_pipeline(setting)

        quant_pipeline.optimize(
            graph=self._graph,
            dataloader=calib_dataloader,
            executor=executor,
            verbose=self._verbose,
            **kwargs)

        if self._verbose:
            print(self.report(), end='')
            print('Network Quantization Finished.')

    def quantize_operation(self, op_name: str, platform: TargetPlatform=None) -> QuantableOperation:
        if op_name not in self._graph.operations:
            raise KeyError(f'Can not find op {op_name} in your graph, chech operation name again.')
        converting_operation = self._graph.operations[op_name]
        if isinstance(converting_operation, QuantableOperation):
            ppq_warning(f'Operation {op_name} has been quantized, can not to quantize it twice.')
            return converting_operation

        # override platform with calling parameter.
        if platform is not None: converting_operation.platform = platform
        else: platform = converting_operation.platform

        # if platform == TargetPlatform.UNSPECIFIED we can skip its quantization when type is not supported.
        if platform == TargetPlatform.UNSPECIFIED and converting_operation.type not in self.quant_operation_types:
            return self._graph.operations[op_name]

        if TargetPlatform.is_quantized_platform(platform):
            # create quantize config and convert operation.
            self._processor(QuantizeOperationCommand(
                op_name=op_name, target_platform=platform,
                config=self.init_quantize_config(operation=converting_operation)
            ))
        return self._graph.operations[op_name]

    @ staticmethod
    def create_default_quant_config(
        op: Operation, num_of_bits: int,
        quant_min: Union[int, float], quant_max: Union[int, float], 
        observer_algorithm: str, policy: QuantizationPolicy, 
        rounding: RoundingPolicy, exponent_bits: int = 0,
    ) -> OperationQuantizationConfig:
        """
        为你的算子创建一个默认量化信息
        
        对于一个 Onnx 算子而言，它总是会有几个输入和输出 Variable
        你需要为每一个相连的 Variable 初始化量化信息 TensorQuantConfig
        
        这个函数就是用来帮你初始化这些信息的。
        
        一个麻烦的问题是：

            对于很多 onnx 算子而言，他们的部分输入都是不需要量化的:
            
            如 Clip 算子的三个输入 value, min, max, 大部分框架不要求量化 min, max
            如 Reshape 算子的两个输入 value, shape, 其中 shape 不能够被量化

            PPQ 的算子接线器中记录了这些信息

            算子接线器中记录了所有标准 onnx 的默认量化策略
            该函数将使用预定义的算子量化策略初始化量化信息
        
        你可以在 Quantizer 中对默认量化策略进行进一步修改

        Create a default quantization configuration for given op.
        For each onnx op, there will be some input and output variables.
        
        You are required to create tensor quantization config for every
        input and output variables.
        
        This function is designed for creating a default quantization config for you.

            The created OQC(Op Quantization Config) is based on OpSocket.

            In fact, there are some rules or templates when creating the OQC:
            For Clip Op which has 3 input variable, namely value, min and max
                most framework does not require a quantization config for min and max.
            For Reshape Op which has 2 input variable, namely value and shape
                the input shape can never be quantized.

        Those rules are pre-defined within OpSocket, thus ppq will create
        OQC based on underlying OpSocket of your Op.
        
        After the default OQC got created, you can overwrite its state in quantizer.
        """
        assert isinstance(op, Operation), (
            f'Can only initialize OQC for PPQ.IR.Operation, however {type(op)} was given.')
        assert isinstance(policy, QuantizationPolicy), (
            f'Can not create quantization config - Quantization Policy Type Error.')
        assert isinstance(rounding, RoundingPolicy), (
            f'Can not create quantization config - Rounding Policy Type Error.')

        socket = op.socket
        input_cfgs, output_cfgs = [], []
        for index in range(op.num_of_input):
            state = QuantizationStates.INITIAL
            # for those unexpected inputs and outputs
            # ppq just initilize them as normal variable.
            if index < len(socket.in_plat):
                target_plat = socket.in_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            input_cfgs.append(TensorQuantizationConfig(
                policy=policy, rounding=rounding,
                num_of_bits=num_of_bits, scale=None, offset=None,
                exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                observer_algorithm=observer_algorithm, state=state))

        for index in range(op.num_of_output):
            state = QuantizationStates.INITIAL
            # for those unexpected inputs and outputs
            # ppq just initilize them as normal variable.
            if index < len(socket.out_plat):
                target_plat = socket.out_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            output_cfgs.append(TensorQuantizationConfig(
                policy=policy, rounding=rounding, num_of_bits=num_of_bits, scale=None, offset=None,
                exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                observer_algorithm=observer_algorithm, state=state))

        return OperationQuantizationConfig(input_cfgs, output_cfgs)

    @ abstractmethod
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        raise NotImplementedError('Implement this first.')

    @ abstractproperty
    @ property
    def quant_operation_types(self) -> set:
        raise NotImplementedError('Quantizier does not have a quantable op set yet.')

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.INT8

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}

    def report(self) -> str:
        debug_str = ''
        # stats:
        quant_ops = [op for op in self._graph.operations.values() if isinstance(op, QuantableOperation)]
        quant_vars = [var for var in self._graph.variables.values() if isinstance(var, QuantableVariable)]
        quant_cfgs = []

        config_states_cnt = {state: 0 for state in QuantizationStates}
        for op in quant_ops:
            for cfg, _ in op.config_with_variable:
                config_states_cnt[cfg.state] += 1
                quant_cfgs.append(cfg)

        debug_str += '--------- Network Snapshot ---------\n'
        debug_str += f'Num of Op:                    [{len(self._graph.operations)}]\n'
        debug_str += f'Num of Quantized Op:          [{len(quant_ops)}]\n'
        debug_str += f'Num of Variable:              [{len(self._graph.variables)}]\n'
        debug_str += f'Num of Quantized Var:         [{len(quant_vars)}]\n'
        debug_str += '------- Quantization Snapshot ------\n'
        debug_str += f'Num of Quant Config:          [{len(quant_cfgs)}]\n'
        for state, cnt in config_states_cnt.items():
            if cnt <= 0: continue
            padding_str = ' ' * max(28 - len(state.name), 0)
            debug_str += f'{state.name}:{padding_str} [{cnt}]\n'
        return debug_str

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        assert isinstance(setting, QuantizationSetting), (
            f'PPQ needs a OptimSetting instance to initialize optimization pipeline,'
            f' however {type(setting)} was given.')

        if setting.matrix_factorization == True:
            ppq_warning('PPQ Matrix Factorization Pass has been removed from QuantizationSetting since 0.6.5, this pass must be called manually now.')
            ppq_warning('PPQ Matrix Factorization Pass 已经不能通过 QuantizationSetting 调用，现在你必须手动调用该优化过程')

        list_of_passes = []
        if setting.ssd_equalization:
            equalization_setting = setting.ssd_setting
            list_of_passes.append(SSDEqualizationPass(
                optimize_level       = equalization_setting.opt_level,
                channel_ratio        = equalization_setting.channel_ratio,
                loss_threshold       = equalization_setting.loss_threshold,
                layer_norm           = equalization_setting.layer_norm,
                iteration            = equalization_setting.iteration
            ))

        if setting.fusion:
            fusion_setting  = setting.fusion_setting
            list_of_passes.append(QuantizeFusionPass(
                fuse_activation=fusion_setting.fuse_activation,
                fuse_passive_op=fusion_setting.fuse_passive_op,
                activation_type=self.activation_fusion_types
            ))

            if fusion_setting.remove_useless_quantization:
                list_of_passes.append(QuantizeSimplifyPass())

        if setting.quantize_parameter:
            param_setting = setting.quantize_parameter_setting
            list_of_passes.append(ParameterQuantizePass(
                method=param_setting.calib_algorithm))

        if setting.quantize_activation:
            act_setting = setting.quantize_activation_setting
            list_of_passes.append(RuntimeCalibrationPass(
                method=act_setting.calib_algorithm))

        if setting.fusion:
            if fusion_setting.align_quantization:
                list_of_passes.append(QuantAlignmentPass(
                    elementwise_merge_method = fusion_setting.align_elementwise_to,
                    concat_merge_method = fusion_setting.align_concat_to,
                    averagepool_method  = fusion_setting.align_avgpooling_to,
                    force_overlap = fusion_setting.force_alignment_overlap
                ))

        if setting.quantize_parameter:
            param_setting = setting.quantize_parameter_setting
            if param_setting.quantize_passive_parameter:
                list_of_passes.append(PassiveParameterQuantizePass())

        if setting.bias_correct:
            bias_correct_setting = setting.bias_correct_setting
            list_of_passes.append(BiasCorrectionPass(
                block_size=bias_correct_setting.block_size,
                interested_layers=bias_correct_setting.interested_layers,
                steps=bias_correct_setting.steps,
                collecting_device=bias_correct_setting.collecting_device
            ))

        if setting.lsq_optimization:
            lsq_setting = setting.lsq_optimization_setting
            list_of_passes.append(LearnedStepSizePass(
                interested_layers  = lsq_setting.interested_layers,
                lr                 = lsq_setting.lr,
                collecting_device  = lsq_setting.collecting_device,
                steps              = lsq_setting.steps,
                gamma              = lsq_setting.gamma,
                is_scale_trainable = lsq_setting.is_scale_trainable,
                block_size         = lsq_setting.block_size
            ))
            # requant passive parameters
            list_of_passes.append(PassiveParameterQuantizePass())

        if setting.blockwise_reconstruction:
            blockwise_reconstruction_setting = setting.blockwise_reconstruction_setting
            list_of_passes.append(AdaroundPass(
                interested_layers  = blockwise_reconstruction_setting.interested_layers,
                lr                 = blockwise_reconstruction_setting.lr,
                collecting_device  = blockwise_reconstruction_setting.collecting_device,
                steps              = blockwise_reconstruction_setting.steps,
                gamma              = blockwise_reconstruction_setting.gamma,
                is_scale_trainable = blockwise_reconstruction_setting.is_scale_trainable,
                block_size         = blockwise_reconstruction_setting.block_size
            ))
            # requant passive parameters
            list_of_passes.append(PassiveParameterQuantizePass())

        if setting.quantize_parameter:
            if param_setting.baking_parameter:
                list_of_passes.append(ParameterBakingPass())

        if setting.extension:
            list_of_passes.append(ExtensionPass(
                setting.extension_setting.my_first_parameter))

        return QuantizationOptimizationPipeline(passes=list_of_passes)

    def build_prequant_pipeline(
        self, setting: QuantizationSetting,
        executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        assert isinstance(setting, QuantizationSetting), (
            f'PPQ needs a OptimSetting instance to initialize optimization pipeline,'
            f' however {type(setting)} was given.')

        list_of_passes = []
        if setting.weight_split:
            weight_split_setting = setting.weight_split_setting
            list_of_passes.append(HorizontalLayerSplitPass(
                interested_layers    = weight_split_setting.interested_layers,
                method               = weight_split_setting.method,
                value_threshold      = weight_split_setting.value_threshold,
            ))

        if setting.channel_split:
            channel_split_setting = setting.channel_split_setting
            list_of_passes.append(ChannelwiseSplitPass(
                optimize_level       = channel_split_setting.opt_level,
                iterations           = channel_split_setting.iterations,
                threshold            = channel_split_setting.value_threshold,
                including_bias       = channel_split_setting.including_bias,
                including_act        = channel_split_setting.including_act,
                bias_multiplier      = channel_split_setting.bias_multiplier,
                act_multiplier       = channel_split_setting.act_multiplier
            ))

        if setting.equalization:
            equalization_setting = setting.equalization_setting
            list_of_passes.append(LayerwiseEqualizationPass(
                optimize_level       = equalization_setting.opt_level,
                iterations           = equalization_setting.iterations,
                weight_threshold     = equalization_setting.value_threshold,
                including_bias       = equalization_setting.including_bias,
                including_act        = equalization_setting.including_act,
                bias_multiplier      = equalization_setting.bias_multiplier,
                act_multiplier       = equalization_setting.act_multiplier
            ))

        return QuantizationOptimizationPipeline(passes=list_of_passes)
