# ---------------------------------------------
# This is a quantizer template for custimizing.
# ---------------------------------------------

from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation

from .base import BaseQuantizer


class ExtQuantizer(BaseQuantizer):
    """ExtQuantizer 是一个空的量化器模板，你可以在这里面实现你自定义的平台量化逻辑 你需要实现该类所有成员函数与方法，并且使用
    TargetPlatform.EXTENSION 调用这个量化器.

        --> quantize_torch_model(..., platform=EXTENSION)

    ExtQuantizer is a empty quantizer template for you to implement custimized quantization logic.
        Implement all member function and properties, and invoke this with TargetPlatform.EXTENSION

        --> quantize_torch_model(..., platform=EXTENSION)

    Args:
        BaseQuantizer ([type]): [description]
    """
    def __init__(
        self, graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph) # do not forget to initialize super class.
        
        # Quantization basic setting -------------------------
        # update following properties:
        self._num_of_bits = 8        # platform bit width.
        self._quant_min = -128       # int value min
        self._quant_max = +127       # int value max
        # ----------------------------------------------------

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        """使用这个函数来初始化算子的量化配置，该函数会被父类作为接口调用。

        我们提供了一个方便的函数 create_default_quant_config 来帮助你实现相关逻辑
            调用这个函数将直接产生一个默认的量化配置，你可以在此基础上进行修改。

        注意并不是所有算子都可以使用默认量化配置完成量化，有些算子明显具有更加复杂的量化逻辑
            对于这种情况，你需要在此函数中手动创建它们的量化配置信息。

        Initial your quantization configuration for each operation.
            (This function will be invoked by BaseQuantizer as an interface.)

        We provide a helper function called self.create_default_quant_config()
            use this function to create a default quantization configuration.

        However, some operations' quantization logic might be much more complex so that
            default configuration won't meet their requirements. In this situation a
            manually initialized configuration is demended.

        Args:
            operation (Operation): [description]

        Returns:
            OperationQuantizationConfig: [description]
        """

        # create a basic quantization configuration.
        config = self.create_default_quant_config(
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )

        # initialize configuration for conv manually
        if operation.type == 'Conv':
            # override initialized config

            if operation.num_of_input == 3:
                bias_config = config.input_quantization_config[-1]
                # bias should be quantized with 32 bits
                # in python3, int indicates long long in C++
                # so that it has enough precision to represent a number like 2^32
                # however, it may cause a scale underflow
                # here we give bias a 30 bits precision, which is pettery enough in all cases
                bias_config.num_of_bits = 30
                bias_config.quant_max = int(pow(2, 30 - 1) - 1)
                bias_config.quant_min = - int(pow(2, 30 - 1))
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR)
                bias_config.state = QuantizationStates.PASSIVE_INIT
            for tensor_config in config.input_quantization_config[1: ]:
                tensor_config.observer_algorithm = 'minmax'

        # mark some operation as passive op.
        # all quantizations of a passive op will be overlapped during graph merge pass.
        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            config.is_active_quant_op = False
        return config

    @ property
    def target_platform(self) -> TargetPlatform:
        """target_platform 属性是提供给子图切分使用的， 所有量化区的算子将被调度到这个设备上。

        Property target_platform is acquired by graph dispather.     It states
        where your quantized operation depoly.
        """
        return TargetPlatform.EXTENSION

    @ property
    def default_platform(self) -> TargetPlatform:
        """default_platform 属性是提供给子图切分使用的， 所有冲突区的算子将被调度到这个设备上。

        Property default_platform is acquired by graph dispather.     It states
        where non-quantable operation depoly.
        """
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        """quant_operation_types 指明了所有可以被量化的算子类型。

        聪明的你可能想问了，如果我不是按类型进行量化，而是想特定去量化某几个算子怎么办。
            我建议你通过手写一个调度表来实现这样的功能，只需要把其他算子调度到FP32上就行了。

        请注意，并不是你写在这里的类型就一定会被量化，在量化之前还有调度器进行子图切分，
            只有量化子图上的算子才可以被量化。

        Property quant_operation_types contains all quantable operations' type.

        Notice that PPQ has a dispatching logic before quantizer,
            so to say quantizer is only in charge of quantable subgraph,
            those operation within non-quantable subgraph will never be
            quantized even their type is listed here.
        """
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'Max', 'Sub', 'Div',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Slice'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        """quantize_policy 指明了默认量化策略 被函数 create_default_quant_config 所使用.

        quantize_policy is used by create_default_quant_config(),
            to generate a default quantization configuration.

        Returns:
            QuantizationPolicy: [description]
        """
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        """rounding_policy 指明了默认取整策略 被函数 create_default_quant_config 所使用.

        rounding_policy is used by create_default_quant_config(),
            to generate a default quantization configuration.

        Returns:
            QuantizationPolicy: [description]

        Returns:
            RoundingPolicy: [description]
        """
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        """activation_fusion_types 指明了所有要参与图融合的激活函数类型 被后续 activation fustion pass 所使用的
        所有被列举在此的激活函数将会尝试和之前的计算节点进行联合定点。

        Returns:
            set: _description_
        """
        return {'Relu', 'Clip'}
