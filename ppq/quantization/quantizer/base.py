from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Iterable, Union

import torch
from ppq.api.setting import *
from ppq.core import (OperationMeta, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationStates, RoundingPolicy,
                      TargetPlatform, TensorQuantizationConfig,
                      empty_ppq_cache)
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
        if not isinstance(graph, BaseGraph):
            raise TypeError(f'To initialize a Quantizer, a BaseGraph instance is needed.'\
                f' While {type(graph)} was givne, if your graph is maintained by GraphCommandProcessor, '\
                'use GraphCommandProcessor.graph here instead.')
        self._verbose = verbose
        self._processor_chain = None
        self._graph = graph
        
        self._quant_min = -128
        self._quant_max = +127
        self._num_of_bits = 8

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

        # step - 2, quantize all operation(need meta data.)
        executor.load_graph(self._graph)
        executor.tracing_operation_meta(inputs=inputs)
        self.quantize_operations(quantable_operation_types=self.quant_operation_types)

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
            **kwargs
        )

        if self._verbose:
            print(self.report(), end='')
            print('Network Quantization Finished.')

    @ empty_ppq_cache
    def quantize_operations(
        self,
        quantable_operation_types: set,
        operation_platforms: dict = None,
        operation_quantization_configs: dict = None,
    ) -> None:
        quantize_chain = QuantableGraph(GraphReplacer(self._graph))
        if operation_platforms is None: operation_platforms = {}
        if operation_quantization_configs is None: operation_quantization_configs = {}

        # build operation_platforms
        # every op MUST have a target platform
        for op_name, operation in self._graph.operations.items():
            # some operation has a predefined platform, just skip.
            if operation.platform != TargetPlatform.UNSPECIFIED:
                operation_platforms[op_name] = operation.platform
            elif operation.type in quantable_operation_types:
                operation_platforms[op_name] = self.target_platform
            else: operation_platforms[op_name] = self.default_platform

            # manual override.
            if op_name in operation_platforms:
                operation.platform = operation_platforms[op_name]

        # build operation_quantization_configs
        # every quantable op MUST have a quantization config
        # if operation.type is listed in quantable_operation_types while a operation_quantization_configs is given
        # it will override the setting of quantable_operation_types
        for op_name, operation in self._graph.operations.items():
            if not TargetPlatform.is_quantized_platform(operation_platforms[op_name]): continue
            # operation information is tracing data, which created in self.__init__
            # it contains useful metadata from creating a Quantizable Operation object
            if operation.meta_data is None:
                raise ValueError(f'Operation {op_name} has no meta data yet. calling executor.tracing_meta')

            if operation.type in quantable_operation_types or TargetPlatform.is_quantized_platform(operation.platform):
                # TargetPlatform.is_quantized_platform(operation.platform) means override.
                if op_name in operation_quantization_configs: continue
                else: operation_quantization_configs[op_name] = (
                    self.init_quantize_config(operation=operation)
                )

        for op_name, operation in list(self._graph.operations.items()):
            # check whether given operation has been quantized,
            # if operation has been quantized, that always means aonther
            # quantizer is in charge of processing the given graph,
            # which is not allowed in ppq.
            if isinstance(operation, QuantableOperation):
                raise TypeError(
                    f'Operation {operation} has been quantized, it is not allowed to quantize a graph for multiple times.')

            # operation_quantization_configs, operation_platforms defines a detailed quantization scheme
            # it is a combination of user-written quantization config and quantizer's internal policy.
            # once quantization_config_lookup_table was given, it will override the quantization config
            # defined in Quantizier.default_operation_quantization_config
            target_platform  = operation_platforms[op_name]

            if TargetPlatform.is_quantized_platform(target_platform):
                if op_name not in operation_quantization_configs:
                    raise KeyError(f'Can not find quantization configuration for operation {op_name}, '
                                   'if you are dispatching operations manually, '
                                   'make sure all operations that you sent to quantized platform is '
                                   'quantable and recognizable for your quantizer.')
                quantization_config = operation_quantization_configs[op_name]
                assert isinstance(quantization_config, OperationQuantizationConfig), (
                    f'Expect an Operation Quantization Config here, however {type(quantization_config)} was given.')
                quantize_chain(
                    QuantizeOperationCommand(
                        op_name=operation.name,
                        target_platform=target_platform,
                        config=quantization_config
                    ))
            else:
                operation.platform = target_platform
        # end for

    @ staticmethod
    def create_default_quant_config(
        operation_meta: OperationMeta, num_of_bits: int,
        quant_min: int, quant_max: int, observer_algorithm: str,
        policy: QuantizationPolicy, rounding: RoundingPolicy,
    ) -> OperationQuantizationConfig:
        assert isinstance(policy, QuantizationPolicy), (
            f'Can not create quantization config - Quantization Policy Type Error.')
        assert isinstance(rounding, RoundingPolicy), (
            f'Can not create quantization config - Rounding Policy Type Error.')
        num_of_related_vars = operation_meta.num_of_input + operation_meta.num_of_output
        configs = [TensorQuantizationConfig(
            policy=policy, rounding=rounding,
            num_of_bits=num_of_bits, scale=None, offset=None,
            quant_min=quant_min, quant_max=quant_max,
            observer_algorithm=observer_algorithm,
        ) for _ in range(num_of_related_vars)]
        return OperationQuantizationConfig(
            input_quantization_configs=configs[: operation_meta.num_of_input],
            output_quantization_configs=configs[operation_meta.num_of_input: ],
        )

    @ abstractmethod
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')
        return base_quant_config

    @ abstractproperty
    @ property
    def quant_operation_types(self) -> set:
        raise NotImplementedError('Quantizier does not have a quantable op set yet.')

    @ abstractproperty
    @ property
    def target_platform(self) -> TargetPlatform:
        raise NotImplementedError('Quantizier does not have a default platform setting yet.')

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        raise NotImplementedError('Quantizier does not have a default quantization policy yet.')

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

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
        if setting.advanced_optimization == True:
            ppq_warning('PPQ Advanced optimization has been removed since 0.6.5, use setting.finetune = True instead')
            ppq_warning('PPQ Advanced optimization 在 0.6.5 版本中已经被移除且不会起到任何效果，作为替代方案我们建议你使用 setting.lsq_optimization = True')
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

        if setting.channel_split:
            channel_split_setting = setting.channel_split_setting
            list_of_passes.append(ChannelSplitPass(
            interested_layers = channel_split_setting.interested_layers,
            search_directions = channel_split_setting.search_directions,
            expand_ratio      = channel_split_setting.expand_ratio,
            split_ratio       = channel_split_setting.split_ratio,
            grid_aware        = channel_split_setting.grid_aware
            ))

        if setting.fusion:
            fusion_setting  = setting.fusion_setting
            if fusion_setting.refine_quantization:
                list_of_passes.append(QuantizeRefinePass())

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
            list_of_passes.append(PassiveParameterQuantizePass(override=True))

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
            list_of_passes.append(PassiveParameterQuantizePass(override=True))

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
