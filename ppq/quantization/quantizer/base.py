from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Iterable, Union
import torch
from ppq.IR.base.graph import Operation
from ppq.IR.quantize import QuantableVariable
from ppq.api.setting import *
from ppq.core import (OperationMeta, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationStates, RoundingPolicy,
                      TargetPlatform, TensorQuantizationConfig,
                      empty_ppq_cache, ppq_debug_function)
from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, QuantableGraph, QuantableOperation
from ppq.IR.base.command import QuantizeOperationCommand
from ppq.IR.morph import GraphReplacer
from ppq.IR.search import SearchableGraph
from ppq.quantization.optim import *


class BaseQuantizer(metaclass = ABCMeta):
    def __init__(
        self,
        graph: BaseGraph,
        verbose: bool = True
    ) -> None:
        if not isinstance(graph, BaseGraph):
            raise TypeError(f'To initialize a Quantizer, a BaseGraph instance is needed.'\
                f' While {type(graph)} was givne, if your graph is maintained by GraphCommandProcesser, '\
                'use GraphCommandProcesser.graph here instead.')
        self._verbose = verbose
        self._processer_chain = None
        self._graph = graph

    @ empty_ppq_cache
    def quantize(
        self,
        inputs: Union[torch.Tensor, list, dict],
        calib_dataloader: Iterable,
        executor: BaseGraphExecutor,
        setting: QuantizationSetting,
        **kwargs
    ) -> None:

        # step - 1, build graph processer chain
        self._processer_chain = SearchableGraph(QuantableGraph(GraphReplacer(self._graph)))

        # step - 2, prequant pipeline:
        # prequant pipeline will change your network sturcture and float value.
        prequant_pipeline = self.build_prequant_pipeline(
            setting, executor=executor)
        prequant_pipeline.optimize(                
            graph=self._processer_chain,
            dataloader=calib_dataloader,
            executor=executor,
            verbose=self._verbose,
            **kwargs)

        # step - 3, quantize all operation(need meta data.)
        executor.load_graph(self._graph)
        executor.tracing_operation_meta(inputs=inputs)
        self.quantize_operations(quantable_opeartion_types=self.quant_operation_types)

        # quantize operation will modify network sturcture
        # it is necessary calling self._executor before further execution
        # step - 4, calling graph optimization pipeline
        executor.load_graph(self._graph)
        quant_pipeline = self.build_quant_pipeline(
            setting, executor=executor)

        quant_pipeline.optimize(
            graph=self._processer_chain,
            dataloader=calib_dataloader,
            executor=executor,
            verbose=self._verbose,
            **kwargs
        )

        if self._verbose: 
            print(self.report(), end='')
            print('Network Quantization Finished.')
        # check if all quantization configs have been processed
        for name, operation in self._graph.operations.items():
            if isinstance(operation, QuantableOperation):
                for config in operation.config.input_quantization_config + \
                    operation.config.output_quantization_config:
                    if QuantizationStates.can_export(config.state) and False:
                        raise RuntimeError(
                            f'Quantize point of opeartion {name} has not been initilized, '\
                            'All quantize points must got processed during your optim passes.'
                        )

    @ empty_ppq_cache
    def quantize_operations(
        self,
        quantable_opeartion_types: set,
        operation_platforms: dict = None,
        operation_quantization_configs: dict = None,
    ) -> None:
        if operation_platforms is None: operation_platforms = {}
        if operation_quantization_configs is None: operation_quantization_configs = {}

        # build operation_platforms
        # every op MUST have a target platform
        for op_name, operation in self._graph.operations.items():
            # some operation has a predefined platform, just skip.
            if operation.platform != TargetPlatform.UNSPECIFIED:
                operation_platforms[op_name] = operation.platform
            elif operation.type in quantable_opeartion_types:
                operation_platforms[op_name] = self.target_platform
            else: operation_platforms[op_name] = self.default_platform

            # maunnl override.
            if op_name in operation_platforms: 
                operation.platform = operation_platforms[op_name]

        # build operation_quantization_configs
        # every quantable op MUST have a quantization config
        # if operation.type is listed in quantable_opeartion_types while a operation_quantization_configs is given
        # it will override the setting of quantable_opeartion_types
        for op_name, operation in self._graph.operations.items():
            if not TargetPlatform.is_quantized_platform(operation_platforms[op_name]): continue
            # operation information is tracing data, which created in self.__init__
            # it contains useful metadata from creating a Quantizable Operation object
            if operation.meta_data is None:
                raise ValueError(f'Operation {op_name} has no meta data yet. calling executor.tracing_meta')

            if operation.type in quantable_opeartion_types:
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
            # it is a combination of user-written quantization config and quantizer's interal policy.
            # once quantization_config_lookup_table was given, it wiil override the quantization config
            # defined in Quantizier.default_operation_quantization_config
            target_platform  = operation_platforms[op_name]

            if TargetPlatform.is_quantized_platform(target_platform):
                quantization_config = operation_quantization_configs[op_name]
                self._processer_chain(
                    QuantizeOperationCommand(
                        op_name=operation.name, 
                        target_platform=target_platform, 
                        config=quantization_config
                    )
                )
            else:
                operation.platform = target_platform
        # end for

    @ staticmethod
    def create_default_quant_config(
        operation_meta: OperationMeta, num_of_bits: int, 
        quant_min: int, quant_max: int, observer_algorithm: str,
        policy: QuantizationPolicy, rounding: RoundingPolicy,
    ) -> OperationQuantizationConfig:
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
        raise NotImplementedError('Quantizier does not have a default operation quantization config yet.')

    @ abstractproperty
    @ property
    def quant_operation_types(self) -> set:
        raise NotImplementedError('Quantizier does not have a quantable op set yet.') 

    @ abstractproperty
    @ property
    def target_platform(self) -> TargetPlatform:
        raise NotImplementedError('Quantizier does not have a default platfrom setting yet.')

    @ abstractproperty
    @ property
    def default_platform(self) -> TargetPlatform:
        raise NotImplementedError('Quantizier does not have a default platfrom setting yet.')

    @ abstractproperty
    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        raise NotImplementedError('Quantizier does not have a default quantization policy yet.')

    @ abstractproperty
    @ property
    def rounding_policy(self):
        raise NotImplementedError('Implement this first.')

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

    def build_quant_pipeline(
        self, setting: QuantizationSetting, 
        executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        assert isinstance(setting, QuantizationSetting), (
            f'PPQ needs a OptimSetting instance to initialize optimization pipeline,'
            f' however {type(setting)} was given.')
        
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
                platform=self.target_platform,
                fuse_activation=fusion_setting.fuse_activation,
                fuse_passive_op=fusion_setting.fuse_passive_op
            ))

            if fusion_setting.fuse_conv_add:
                list_of_passes.append(PPLCudaAddConvReluMerge())
            
            if fusion_setting.remove_useless_quantization:
                list_of_passes.append(QuantizeReducePass())

        if setting.quantize_parameter:
            param_setting = setting.quantize_parameter_setting
            list_of_passes.append(ParameterQuantizePass(
                method=param_setting.calib_algorithm))

        if setting.quantize_activation:
            act_setting = setting.quantize_activation_setting
            if act_setting.per_layer_calibration:
                list_of_passes.append(RuntimePerlayerCalibrationPass(
                    method=act_setting.calib_algorithm))
            else:
                list_of_passes.append(RuntimeCalibrationPass(
                    method=act_setting.calib_algorithm))
            
            if act_setting.inplace_act_quantization:
                list_of_passes.append(InplaceQuantizationSettingPass())
        
        if setting.fusion:
            if fusion_setting.align_quantization:
                list_of_passes.append(QuantAlignmentPass(
                    elementwise_merge_method = fusion_setting.align_elementwise_to,
                    concat_merge_method = fusion_setting.align_concat_to,
                    force_overlap = fusion_setting.force_alignment_overlap
                ))

        if setting.quantize_parameter:
            param_setting = setting.quantize_parameter_setting
            if param_setting.quantize_passive_parameter:
                list_of_passes.append(PassiveParameterQuantizePass())
        
        if setting.lsq_optimization:
            lsq_setting = setting.lsq_optimization_setting
            list_of_passes.append(LearningStepSizeOptimization(
                interested_layers      = lsq_setting.interested_layers,
                interested_layers_only = lsq_setting.interested_layers_only,
                output_names           = lsq_setting.output_names,
                loss_weights           = lsq_setting.loss_weights,
                epochs                 = lsq_setting.epochs,
                lr                     = lsq_setting.lr,
                scale_multiplier       = lsq_setting.scale_multiplier,
                mode                   = lsq_setting.mode
            )
        )
        
        if setting.blockwise_reconstruction:
            brecq_setting = setting.blockwise_reconstruction_setting
            list_of_passes.append(BlockwiseReconstructionPass(
                interested_layers = brecq_setting.interested_layers,
                tune_act_scale    = brecq_setting.tune_act_scale,
                epochs            = brecq_setting.epochs,
                lr                = brecq_setting.lr,
                lamda             = brecq_setting.lamda,
                scale_multiplier  = brecq_setting.scale_multiplier
            )
        )

        if setting.advanced_optimization:
            optim_setting = setting.advanced_optimization_setting
            list_of_passes.append(AdvancedQuantOptimization(
                collecting_device = optim_setting.collecting_device,
                limit             = optim_setting.limit,
                lr                = optim_setting.lr,
                check             = optim_setting.auto_check,
                interested_layers = optim_setting.interested_layers,
                interested_outputs= optim_setting.interested_outputs,
                steps             = optim_setting.steps
            )
        )

        if setting.bias_correct:
            list_of_passes.append(BiasCorrectionPass(
                auto_check = setting.bias_correct_setting.auto_check,
                interested_output = setting.bias_correct_setting.interested_outputs,
                verbose    = setting.bias_correct_setting.verbose,
                max_steps  = setting.bias_correct_setting.max_steps
            )
        )
        if setting.quantize_parameter:
            if param_setting.baking_parameter:
                list_of_passes.append(ParameterBakingPass(
                    quantize_function=executor.quantize_function))
            
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

        if setting.equalization:
            equalization_setting = setting.equalization_setting
            list_of_passes.append(LayerwiseEqualizationPass(
                optimize_level       = equalization_setting.opt_level,
                iterations           = equalization_setting.iterations,
                weight_threshold     = equalization_setting.value_threshold,
                including_bias       = equalization_setting.including_bias,
                including_activation = equalization_setting.including_act,
                bias_mutiplier       = equalization_setting.bias_multiplier,
                activation_mutiplier = equalization_setting.act_multiplier
            ))
        
        if setting.matrix_factorization:
            list_of_passes.append(MatrixFactorizationPass(
                interested_layers=setting.matrix_factorization_setting.interested_layers,
                method=setting.matrix_factorization_setting.method
            ))

        return QuantizationOptimizationPipeline(passes=list_of_passes)