"""
    PPQ Core Data Structure Abstraction
    PPQ 核心量化结构抽象

    You are not allowed to modify this
    请勿修改此文件
"""

import time  # for hash generation
from abc import abstractmethod
from enum import Enum
from typing import Any, Iterable, List

from .storage import Serializable


class NetworkFramework(Enum):
    PPL     = 1
    ONNX    = 2
    CAFFE   = 3
    NXP     = 4
    NATIVE  = 5


class TargetPlatform(Enum):
    """
    TargetPlatform is a core abstraction of PPQ framework,
        it defines "platform" as an attribute of an operation.
    Platform attribute of an operation indicates where this operation is going to be depoly.
    This feature enables PPQ to simulate inter-device computing.

    Platform attribute also tells PPQ how to quantize an operation, and how to execute it.
        ATTENTION: Different platform might bring different behaviour of a same operation.
        ATTENTION: Operation which is assigned to an non-quantizible platform will never be quantized.

    There are several supported platforms for PPQ now, 
        however you are supposed to be aware of some particular platforms here:

    SHAPE_OR_INDEX is a virtual platform, however it is an EXTREMELY IMPORTANT components in PPQ.
        Dispatch an operation to platform SHAPE_OR_INDEX means this operation is SOI-related,
        it processes a SOI tensor and gives a processed SOI, all calculation of this operation must be sent to CPU
            (or any platform capable for calculating this.) when depoly.

        An operation with SHAPE_OR_INDEX platform assigned will never be quantized regardless of its type.
        It is a crucial feature for quantizing network that contains SOI-related operation. (Shufflenet etc.)

        By default, PPQ automatically detects all SOI-related operations, and dispatch them to SHAPE_OR_INDEX platform.
        To understand how this feature works, see also: ppq.sche

    UNSPECIFIED is a virtual platform, all operations are sent to this platform once they were created.
        Quantizer then dispatches them towards desired platform through its quantization logic.
    """
    TRT_INT8  = 101
    TRT_INT4  = 102
    TRT_FP16  = 103
    
    PPL_CUDA_INT8 = 201
    PPL_CUDA_INT4 = 202
    PPL_CUDA_FP16 = 203

    DSP_INT8  = 301

    HOST_INT8 = 402

    NXP_INT8  = 501
    
    FP32 = 0
    # SHAPE-OR-INDEX related operation
    SHAPE_OR_INDEX = -1
    # initial state
    UNSPECIFIED   = -2
    # boundary op
    BOUNDARY      = -3
    # just used for calling exporter
    ONNX          = -4
    CAFFE         = -5
    NATIVE        = -6
    
    # THIS IS A DUUMY PLATFORM JUST FOR CREATING YOUR OWN EXTENSTION.
    EXTENSION     = -10086
    

    @ classmethod
    def is_quantized_platform(cls, platform) -> bool:
        return platform in {cls.DSP_INT8, cls.TRT_INT4, cls.TRT_INT8, cls.NXP_INT8, 
                            cls.PPL_CUDA_INT8, cls.PPL_CUDA_INT4, cls.EXTENSION}


class RoundingPolicy(Enum):
    """
    RoundingPolicy is a core setting for PPQ quantization calculation.
        It defines rounding behaviour inside quantization calculation.

    Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

    PPQ Supports 7 different rounding policies now.
    Take a look at https://en.wikipedia.org/wiki/Rounding

    ATTENTION: RoundingPolicy greatly affects PPQ executor behaviour in some cases, 
        to get a correct result from PPQ executor, 
        make sure your RoundingPolicy is the same as your hardware.
    """
    ROUND_HALF_EVEN            = 0
    ROUND_HALF_UP              = 1
    ROUND_HALF_DOWN            = 2
    ROUND_HALF_TOWARDS_ZERO    = 3
    ROUND_HALF_FAR_FORM_ZERO   = 4
    ROUND_TO_NEAR_INT          = 5
    ROUND_UP                   = 6


class QuantizationProperty(Enum):
    """
    QuantizationProperty is a core abstraction for PPQ quantization calculation.
    QuantizationProperty and QuantizationPolicy together build a bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.
    
    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 7 different quantization property(s) supported by PPQ now.
    
        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)
        
        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Indicates a linear quantization, follow formula: quant(x) = clip(round(x / scale))

        EXPONENTIAL: Indicates an exponential quantization, not yet used.

        SYMMETRICAL: Indicates a symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Indicates an asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Indicates a power-of-2 quantization, scale must be pow(2, k) in this mode.

    ATTENTION: Not all combinations of all 7 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of 
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    PER_TENSOR   = 0x00000001
    PER_CHANNEL  = 0x00000002
    LINEAR       = 0x00000004
    EXPONENTIAL  = 0x00000008
    SYMMETRICAL  = 0x00000010
    ASYMMETRICAL = 0x00000020
    POWER_OF_2   = 0x00000040

    def __or__(self, other: int) -> int:
        return self.value + other
    
    def __ror__(self, other: int) -> int:
        return self.value + other
    
    def __and__(self, other: int) -> int:
        return self.value & other
    
    def __rand__(self, other: int) -> int:
        return self.value & other

    def __radd__(self, other: int) -> int:
        return self.value + other

    def __add__(self, other: int) -> int:
        return self.value + other

    def __sub__(self, other: int) -> int:
        return self - (self.value & other)

    def __rsub__(self, other: int) -> int:
        return other - (self.value & other)


class QuantizationPolicy:
    """
    QuantizationPolicy is a core abstraction for PPQ quantization calculation.
    QuantizationProperty and QuantizationPolicy together build a bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 7 different quantization property(s) supported by PPQ now.
    
        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)
        
        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Indicates a linear quantization, follow formula: quant(x) = clip(round(x / scale))

        EXPONENTIAL: Indicates an exponential quantization, not yet used.

        SYMMETRICAL: Indicates a symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Indicates an asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Indicates a power-of-2 quantization, scale must be pow(2, k) in this mode.

    ATTENTION: Not all combinations of all 7 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of 
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    def __init__(self, policy: int) -> None:
        if not QuantizationPolicy.__check_valid(policy):
            raise ValueError(
                'invalid quantization pattern, valid partterns are listed in '
                'ppq.core.OperationQuantizationPolicy.__check_valid'
            )
        self._policy = policy

    def has_property(self, property: QuantizationProperty):
        return (self._policy & property.value) != 0

    @ classmethod
    def __check_valid(cls, policy):
        return policy in {
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.EXPONENTIAL | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.EXPONENTIAL | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.EXPONENTIAL | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.EXPONENTIAL | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
        }


class QuantizationStates(Enum):
    """
    QuantizationStates is a core data structure for PPQ quantization.
    QuantizationStates tells whether a quantization configuration is activated.

    ATTENTION: Changes of QuantizationState will greatly affect execution result.

    For a TensorQuantizationConfig instance, there are 11 available quantization states now.
    Only when state is ACTIVATED or NEGATIVE, corresponding tensor will be quantized during the execution.

    Here we give a brief description of each quantization state:

        INITIAL: given when TensorQuantizationConfig is created, is an initial state of all quantization configuration.

        PASSIVE_INIT: for particular parameter like bias of GEMM(Convolution) and padding value of Pad. Usually it
        does not have an independent quantization scale and offset, while gets quantized with other tensor's configuration.
            For GEMM and Convolution, there bias will be quantized with input scale * weight scale.
            For padding value and clip value, it shares the same scale with its input.
        Those parameters will have a PASSIVE_INIT state when created.

        ATTENTION: if there is any quantization configuration with INITIAL or PASSIVE_INIT state, PPQ will refuse
            to deploy your model and an error will be thrown. 
            This inspection will be ignored when PPQ.core.config.DEBUG set as True.
        
        OVERLAPPED: state OVERLAPPED means there is someone else takes control of current tensor,
        and overlapped tensor quantization configuration will be ignored by optimization algorithms and executor.
        
        Graph fusion always generate overlapped quantization, for a typical conv - relu fusion,
        the output quantization of convolution will be overlapped by the output tensor of relu. 
        State OVERLAPPED cares only about quantization behaviour that cross layers.

        DEACTIVATED: state DEACTIVATED is related with "dequantize" function, once an operation is dequantized,
        all related tensor configurations will be replaced as DEACTIVATED, so that skipping all quantization during
        execution.

        SOI: whenever a tensor quantization configuration holds SOI state,
            it will be never quantized and will not be included into any optimization algorithm.
        it means underlying tensor is SOI-related tensor, and it can not be quantized.

        ACTIVATE: means corresponding tensor is ready to be quantized with its configuration.

        PASSIVE: means corresponding tensor is ready to be quantized with its configuration.
            (however its configuration is not stand alone, it still depends on someone else.)
        
        BAKED: means corresponding tensor has been pre-quantized, its value can directly 
            go forward without quantization.
    """
    INITIAL     = 1   # 量化参数刚刚被初始化，当前 config 不生效，数据不能被使用
    BAKED       = 2   # 只针对参数量化，表示参数已经被静态量化，当前 config 不生效，数据可以直接使用
    OVERLAPPED  = 3   # 只针对activation量化，表示数据流的量化由其他 config 管理，当前 config 不生效
    DEACTIVATED = 4   # 表示当前 config 不生效
    ACTIVATED   = 5   # 表示当前 config 生效
    DEQUANTIZED = 6   # 表示当前 config 处于解量化状态，解量化是 PPQ 中的一个系统操作
    SOI         = 7   # 表示这一路输入与 Shape or index 相关，不量化
    PASSIVE     = 8   # 表示这一路输入被动量化，如 bias, clip value 等
    PASSIVE_INIT = 9  # 表示这一路输入被动量化，并且刚刚初始化不能被使用
    PASSIVE_BAKED = 10 # 被动量化且静态量化，当前config不生效，数据可以直接使用
    FP32        = 11   # 表示这一路输入直接为FP32浮点数

    @ classmethod
    def is_activated(cls, state)->bool:
        return state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}

    @ classmethod
    def can_export(cls, state)->bool:
        return state not in {QuantizationStates.INITIAL, QuantizationStates.DEACTIVATED, 
                             QuantizationStates.DEQUANTIZED, QuantizationStates.PASSIVE_INIT,
                             QuantizationStates.FP32}


class TensorQuantizationConfig(Serializable):
    """
    TensorQuantizationConfig, as known as tensor quantization configuration, is the most
        important data structure in PPQ system.
    
    PPQ generates quantization configuration for all tensors that need to be quantized, and control their 
        quantization logic via this abstraction. As a basic building block of PPQ quantization system, tensor
        quantization is designed to store and manage all quantization related information like:

        Quantization policy, rounding policy, quantization bits, scale, offset, quantization state, etc.

    ATTENTION: tensor(or variable in PPQ) might have more than one quantization configuration, since
        PPQ is designed as an operation-oriented quantization system, so to say tensor quantization configurations
        are created operation by operation. Considering a pattern conv - conv, both the upstream convolution layer
        and the downstream convolution layer will hold a tensor quantization configuration of the middle variable. 
        Duplicated quantization configuration will be disabled by optimization pass later.

    PPQ is designed as an operation-oriented quantization system, literally all tensor quantization configurations
        are managed by operations, through you can access their image by variable instance.
        (see the defination of PPQ.IR.quant.QuantableVariable for more information)

    You are supposed to change tensor quantization configuration during optimization passes, this abstraction 
        is widely tested among various platforms, it shall satisfy most of your quantization demands.
    """
    def __init__(
        self,
        policy: QuantizationPolicy,
        rounding: RoundingPolicy,
        num_of_bits: int,
        quant_min: int,
        quant_max: int,
        scale: Any,
        offset: Any,
        observer_algorithm: str,
        detail: Any = None,
        inplace: bool = False,
        state: QuantizationStates = QuantizationStates.INITIAL
    ):
        """
        Create a PPQ Tensor Quantization Configuration Instance

        Args:
            policy (QuantizationPolicy): 
                Quantization policy instance which defines the quantization behaviour from marco view.
            
            rounding (RoundingPolicy): Rounding policy used in quantization. 
            
            num_of_bits (int): Quantization bits. (2 < num_of_bits < 32)
            
            quant_min (int): An integer value represents the upper bound(inclusive) of quantized value.
            
            quant_max (int): An integer value represents the lower bound(inclusive) of quantized value.
            
            scale (Any): 
                Scale of quantized value, for per-tensor quantization policy, we use a single float as its scale,
                while for per-channel quantization policy, it will be an array that contains scales for each channel.
            
            offset (Any): Quantization offset for ASYMMETRICAL quantization policy, 
                it will be set as 0 in SYMMETRICAL quantization schema.
            
            observer_algorithm (str): A string represents an observing algorithm for this tensor.
                PPQ support 'kl', 'minmax' observer now.
            
            detail (Any, optional): Only used by PPQ internal logic, detail is used to store some internal data, 
                you are not supposed to use it.
            
            inplace (bool, optional): Indicates whether quantization is taken inplace(rewrite tensor value).

            state (QuantizationStates, optional): 
                Defaults to QuantizationStates.INITIAL, see QuantizationStates for more detail.
        """

        assert num_of_bits <= 32, 'Cannot quantize a tensor with more than 32 bits.'
        assert num_of_bits >= 2, 'Cannot quantize a tensor with less than 2 bits.'

        self._policy = policy
        self._num_of_bits = num_of_bits
        self._scale = scale
        self._offset = offset
        self.state = state
        self._rounding = rounding
        self._quant_min = quant_min
        self._quant_max = quant_max
        self.observer_algorithm = observer_algorithm
        self.inplace = inplace
        self.detail = {} if detail is None else detail
        self._father_config = self # union-find
        self._hash = self.__create_hash()
        super().__init__()

    @ abstractmethod
    def export(self) -> str:
        raise Exception('Implement this first')

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TensorQuantizationConfig): 
            raise TypeError('Can only compare TensorQuantizationConfig object '\
                'with another TensorQuantizationConfig object.')
        return self._hash == o._hash

    def __str__(self) -> str:
        return f'PPQ TensorQuantizationConfig({self.__hash__()})'

    _hash_seed = int(time.time())
    @ staticmethod
    def __create_hash():
        TensorQuantizationConfig._hash_seed = (
            0x343FD * TensorQuantizationConfig._hash_seed + 0x269EC3) % (2 << 31)
        return TensorQuantizationConfig._hash_seed

    def __hash__(self) -> int:
        return self._hash

    @ property
    def dominated_by(self):
        """
        dominated_by is a crucial feature for tensor quantization configuration in PPQ.
        This property is actually maintained by union-find set data structure.

        Every tensor quantization configuration(A) is created with dominated_by = self, and only when
            it is overlapped by other configuration(B), it shall set A.dominated_by = B. 
            Setting A.dominated_by = B also makes A, B as a quantization group.
            (quantization state of A is always set as OVERLAPPED here)

        So to say every tensor quantization configuration with dominated_by != self is overrlaped by
            other quantization configuration. When a tensor quantization configuration is overlapped,
            it means this tensor is already been quantized with another quantization configuration, 
            and there is no need to be quantized with this configuration anymore.

        PPQ use this property to find root configuration for each configuration group,

        Returns:
            [TensorQuantizationConfig]: root configuration of this quantization group.
        
        ATTENTION: This configuration is invalid when self.dominated_by != self.
        """
        if self._father_config == self: 
            return self
        else: 
            root = self._father_config.dominated_by
            self._father_config = root
            return root

    @ dominated_by.setter
    def dominated_by(self, o):
        assert isinstance(o, TensorQuantizationConfig), (
            'Can only set this attribute with another tensor config.')
        root, dominator = self.dominated_by, o.dominated_by
        if dominator != root:
            root._father_config = dominator
            self._father_config = dominator
            root.state = QuantizationStates.OVERLAPPED
            self.state = QuantizationStates.OVERLAPPED

    @ property
    def scale(self) -> Any:
        if self.dominated_by == self: return self._scale
        else: return self.dominated_by.scale

    @ property
    def offset(self) -> Any:
        if self.dominated_by == self: return self._offset
        else: return self.dominated_by.offset

    @ property
    def policy(self) -> QuantizationPolicy:
        if self.dominated_by == self: return self._policy
        else: return self.dominated_by.policy

    @ property
    def num_of_bits(self) -> int:
        if self.dominated_by == self: return self._num_of_bits
        else: return self.dominated_by.num_of_bits

    @ property
    def rounding(self) -> RoundingPolicy:
        if self.dominated_by == self: return self._rounding
        else: return self.dominated_by.rounding

    @ property
    def quant_min(self) -> int:
        if self.dominated_by == self: return self._quant_min
        else: return self.dominated_by.quant_min

    @ property
    def quant_max(self) -> int:
        if self.dominated_by == self: return self._quant_max
        else: return self.dominated_by.quant_max

    @ scale.setter
    def scale(self, value: Any):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._scale = value
    
    @ offset.setter
    def offset(self, value: Any):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._offset = value
    
    @ policy.setter
    def policy(self, policy: QuantizationPolicy):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._policy = policy

    @ num_of_bits.setter
    def num_of_bits(self, bits: int):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._num_of_bits = bits

    @ rounding.setter
    def rounding(self, policy: RoundingPolicy):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._rounding = policy

    @ quant_min.setter
    def quant_min(self, min: int):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._quant_min = min

    @ quant_max.setter
    def quant_max(self, max: int):
        if self.dominated_by != self:
            raise PermissionError(
                f'You are trying to edit property of a tensor quantization configuration({self}). '
                f'While this configuration has been dominated by {self.dominated_by}. '
                'Cause being overlapped, any change of this configuration is not allowed.')
        else:
            self._quant_max = max


class ChannelwiseTensorQuantizationConfig(TensorQuantizationConfig):
    """
    ChannelwiseTensorQuantizationConfig is a special case for tensor quantization configuration.
    
    Comparing with per-tensor quantization configuration, pre-channel quantization has a property 
        "channel_axis" to indicate a channel axis where quantization takes effects.

    Along this axis, all tensor values will be quantized with a sharing scale and offset,
        and all scales and offsets form all channels will be stored by this configuration.

    see the definition of TensorQuantizationConfig for more detail.
    """
    def __init__(self, 
        policy: QuantizationPolicy, rounding:RoundingPolicy,
        num_of_bits: int, quant_min: int, quant_max: int, 
        scale: Any, offset: Any, observer_algorithm: str, 
        state: QuantizationStates, channel_axis: int, detail: dict = {}
    ):
        if policy.has_property(QuantizationProperty.PER_TENSOR): 
            raise TypeError('Can not assign QuantizationProperty.PER_TENSOR policy '\
                'to a Channel-wise Tensor Quantization Config instance.'
            )
        super().__init__(
            policy=policy, num_of_bits=num_of_bits, 
            quant_min=quant_min, quant_max=quant_max, scale=scale, offset=offset, 
            observer_algorithm=observer_algorithm, detail=detail, state=state,
            rounding=rounding
        )
        self.channel_axis = channel_axis

    @ classmethod
    def convert_from_tensor_config(cls, 
        convert_from:TensorQuantizationConfig,
        scales: Iterable,
        offsets: Iterable,
        channel_axis: int
    ):
        this = ChannelwiseTensorQuantizationConfig(
            policy=convert_from.policy,
            num_of_bits=convert_from.num_of_bits,
            quant_min=convert_from.quant_min,
            quant_max=convert_from.quant_max,
            scale=scales, offset=offsets,
            observer_algorithm=convert_from.observer_algorithm,
            detail=convert_from.detail,
            state=convert_from.state,
            channel_axis=channel_axis,
            rounding=convert_from.rounding
        )
        return this


class OperationQuantizationConfig(Iterable):
    """
    OperationQuantizationConfig serves as a collection of tensor quantization configuration.

    See TensorQuantizationConfig for more information.
    """
    def __init__(
        self,
        input_quantization_configs: List[TensorQuantizationConfig] = None,
        output_quantization_configs: List[TensorQuantizationConfig] = None,
        is_positive_quant_op: bool = True
    ):
        """
        Create an operation quantization configuration.

        Args:
            input_quantization_configs (List[TensorQuantizationConfig], optional): 
                a list contains all configuration of all input variables.

            output_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all output variables.
            
            ATTENTION: whether a variable is gonna to be quantized or not, it must have a quantization configuration.
            
            is_positive_quant_op (bool, optional): [description]. Defaults to True.
                some operations are passively quantized, such as Maxpooling, Padding.
                For those operations, set this property as False, PPQ will use this property to optimize your graph.
        """
        self.input_quantization_config     = self.__check_famliy_config(input_quantization_configs)
        self.output_quantization_config    = self.__check_famliy_config(output_quantization_configs)
        self.is_active_quant_op = is_positive_quant_op

    def export(self) -> str:
        raise Exception('Implement this first')

    def __check_famliy_config(self, famliy_configs):
        for famliy_config in famliy_configs:
            if not isinstance(famliy_config, TensorQuantizationConfig):
                raise TypeError(
                    f'You are trying to set famliy quantization config of {str(self)}, ' \
                    f'However your input is invalid, except one TensorQuantizationConfig object, ' \
                    f'while a {type(famliy_config)} was given.'
                )
        return famliy_configs

    def __str__(self) -> str:
        return f'Inputs config: {self.input_quantization_config}, '\
            f'Outputs config {self.output_quantization_config}'

    def __iter__(self) -> TensorQuantizationConfig:
        return (self.input_quantization_config + self.output_quantization_config).__iter__()
